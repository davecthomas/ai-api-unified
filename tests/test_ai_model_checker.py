# tests/test_ai_model_checker.py
"""
AiModelChecker core: memoization keyed by credentials, parallel
de-duplication, probe failure mapping, log events, mode parsing, and
candidate pruning.
"""

from __future__ import annotations

import logging
import threading
import os
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from ai_api_unified.completions.ai_fallback_completions import AIFallbackCandidate
from ai_api_unified.util.ai_model_checker import (
    AiModelChecker,
    AiModelCheckResult,
    AiModelCheckStatus,
    prune_candidates,
    resolve_model_check_mode,
)

LOGGER_NAME: str = "ai_api_unified.util.ai_model_checker"


@pytest.fixture(autouse=True)
def _fresh_checker() -> Iterator[None]:
    dict_saved = dict(AiModelChecker._DICT_PROBE_BY_ENGINE)
    dict_saved_settings = dict(AiModelChecker._DICT_CREDENTIAL_SETTINGS_BY_ENGINE)
    AiModelChecker.reset_cache()
    yield
    AiModelChecker._DICT_PROBE_BY_ENGINE.clear()
    AiModelChecker._DICT_PROBE_BY_ENGINE.update(dict_saved)
    AiModelChecker._DICT_CREDENTIAL_SETTINGS_BY_ENGINE.clear()
    AiModelChecker._DICT_CREDENTIAL_SETTINGS_BY_ENGINE.update(dict_saved_settings)
    AiModelChecker.reset_cache()


def _c(engine: str, model: str, base_url: str | None = None) -> AIFallbackCandidate:
    return AIFallbackCandidate(engine=engine, model=model, base_url=base_url)


class TestCheck:
    def test_probe_runs_once_per_model(self) -> None:
        calls: list[str] = []

        def _probe(model: str, base_url: str | None):
            calls.append(model)
            return AiModelCheckStatus.AVAILABLE, "ok"

        AiModelChecker.register_probe("fake", _probe)
        checker = AiModelChecker()
        first = checker.check(_c("fake", "m1"))
        second = checker.check(_c("fake", "m1"))
        assert first is second and calls == ["m1"]
        assert first.status is AiModelCheckStatus.AVAILABLE and first.keeps_model
        assert AiModelChecker.cached(_c("fake", "m1")) is first

    def test_credential_change_reprobes_and_key_never_in_memo(self) -> None:
        calls: list[str] = []

        def _probe(model: str, base_url: str | None):
            calls.append(os.environ["FAKE_KEY"])
            return AiModelCheckStatus.AVAILABLE, "ok"

        AiModelChecker.register_probe("fake", _probe, ("FAKE_KEY",))
        checker = AiModelChecker()
        with patch.dict(os.environ, {"FAKE_KEY": "secret-one"}):
            checker.check(_c("fake", "m1"))
            checker.check(_c("fake", "m1"))
        with patch.dict(os.environ, {"FAKE_KEY": "secret-two"}):
            checker.check(_c("fake", "m1"))
        assert calls == ["secret-one", "secret-two"]
        assert not any(
            "secret" in str(part) for key in AiModelChecker._DICT_CACHE for part in key
        )

    def test_error_result_is_not_memoized(self) -> None:
        calls: list[str] = []

        def _flaky(model: str, base_url: str | None):
            calls.append(model)
            if len(calls) == 1:
                raise ConnectionError("blip")
            return AiModelCheckStatus.AVAILABLE, "ok"

        AiModelChecker.register_probe("flaky", _flaky)
        checker = AiModelChecker()
        assert checker.check(_c("flaky", "m")).status is AiModelCheckStatus.ERROR
        assert checker.check(_c("flaky", "m")).status is AiModelCheckStatus.AVAILABLE
        assert checker.check(_c("flaky", "m")).status is AiModelCheckStatus.AVAILABLE
        assert calls == ["m", "m"]

    def test_base_url_is_part_of_the_key(self) -> None:
        AiModelChecker.register_probe(
            "fake", lambda m, b: (AiModelCheckStatus.AVAILABLE, b or "")
        )
        checker = AiModelChecker()
        assert checker.check(_c("fake", "m", "https://a")).detail == "https://a"
        assert checker.check(_c("fake", "m", "https://b")).detail == "https://b"

    def test_check_all_dedupes_and_runs_in_parallel(self) -> None:
        lock = threading.Lock()
        int_active = 0
        int_peak = 0

        def _probe(model: str, base_url: str | None):
            nonlocal int_active, int_peak
            with lock:
                int_active += 1
                int_peak = max(int_peak, int_active)
            time.sleep(0.05)
            with lock:
                int_active -= 1
            return AiModelCheckStatus.AVAILABLE, ""

        AiModelChecker.register_probe("fake", _probe)
        list_results = AiModelChecker().check_all(
            [_c("fake", "a"), _c("fake", "b"), _c("fake", "a"), _c("fake", "c")]
        )
        assert [r.model for r in list_results] == ["a", "b", "c"]
        assert int_peak >= 2

    def test_missing_unverifiable_and_no_probe(self) -> None:
        AiModelChecker.register_probe(
            "gone", lambda m, b: (AiModelCheckStatus.MISSING, "404")
        )
        AiModelChecker.register_probe(
            "badkey", lambda m, b: (AiModelCheckStatus.UNREACHABLE, "401")
        )
        AiModelChecker.register_probe(
            "nokey", lambda m, b: (AiModelCheckStatus.UNVERIFIABLE, "no key")
        )
        checker = AiModelChecker()
        assert checker.check(_c("gone", "m")).keeps_model is False
        assert checker.check(_c("badkey", "m")).keeps_model is False
        assert checker.check(_c("nokey", "m")).keeps_model is True
        unknown = checker.check(_c("unregistered", "m"))
        assert (
            unknown.status is AiModelCheckStatus.UNVERIFIABLE
            and "no model probe" in unknown.detail
        )

    def test_probe_exception_and_timeout_map_to_error_and_keep(self) -> None:
        def _boom(model: str, base_url: str | None):
            raise ConnectionError("down")

        def _slow(model: str, base_url: str | None):
            time.sleep(0.5)
            return AiModelCheckStatus.AVAILABLE, ""

        AiModelChecker.register_probe("boom", _boom)
        AiModelChecker.register_probe("slow", _slow)
        checker = AiModelChecker(float_timeout_seconds=0.05)
        errored = checker.check(_c("boom", "m"))
        assert (
            errored.status is AiModelCheckStatus.ERROR
            and "ConnectionError" in errored.detail
        )
        timed = checker.check(_c("slow", "m"))
        assert timed.status is AiModelCheckStatus.ERROR and "timed out" in timed.detail
        assert errored.keeps_model and timed.keeps_model

    def test_log_events(self, caplog: pytest.LogCaptureFixture) -> None:
        AiModelChecker.register_probe(
            "gone", lambda m, b: (AiModelCheckStatus.MISSING, "404")
        )
        AiModelChecker.register_probe(
            "nokey", lambda m, b: (AiModelCheckStatus.UNVERIFIABLE, "no key")
        )
        AiModelChecker.register_probe(
            "badkey", lambda m, b: (AiModelCheckStatus.UNREACHABLE, "401")
        )
        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            AiModelChecker().check_all(
                [_c("gone", "x"), _c("nokey", "y"), _c("badkey", "z")]
            )
        dict_events = {
            getattr(r, "ai_model_check_event", None): r for r in caplog.records
        }
        missing = dict_events["model_missing"]
        assert missing.levelno == logging.WARNING
        assert "not available and will be ignored" in missing.getMessage()
        assert missing.engine == "gone" and missing.status == "missing"
        assert dict_events["model_unverifiable"].levelno == logging.INFO
        unreachable = dict_events["model_unreachable"]
        assert unreachable.levelno == logging.WARNING
        assert "cannot be reached and will be ignored" in unreachable.getMessage()


class TestHelpers:
    @pytest.mark.parametrize(
        ("raw", "mode"),
        [("", "prune"), (None, "prune"), (" STRICT ", "strict"), ("off", "off")],
    )
    def test_resolve_mode(self, raw: object, mode: str) -> None:
        assert resolve_model_check_mode(raw) == mode

    def test_unknown_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="AI_MODEL_CHECK"):
            resolve_model_check_mode("maybe")

    def test_prune_candidates_drops_missing_and_unreachable(self) -> None:
        now = datetime.now(timezone.utc)

        def _r(model: str, status: AiModelCheckStatus) -> AiModelCheckResult:
            return AiModelCheckResult(
                engine="e", model=model, status=status, checked_at=now
            )

        dict_results = {
            ("e", "a", None): _r("a", AiModelCheckStatus.AVAILABLE),
            ("e", "b", None): _r("b", AiModelCheckStatus.MISSING),
            ("e", "c", None): _r("c", AiModelCheckStatus.ERROR),
            ("e", "d", None): _r("d", AiModelCheckStatus.UNREACHABLE),
        }
        kept = prune_candidates(
            [
                _c("e", "a"),
                _c("e", "b"),
                _c("e", "c"),
                _c("e", "d"),
                _c("e", "unchecked"),
            ],
            dict_results,
        )
        assert [c.model for c in kept] == ["a", "c", "unchecked"]
