# tests/test_configured_model_check.py
"""
The factory's configured-model check: pruning of fallback and tier models,
dropped tiers remapped upward, the primary that cannot be pruned, strict
and off modes, a missing classifier model, and the exposed report.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from ai_api_unified import AIFactory, AiFallbackCompletions, AiRouterCompletions
from ai_api_unified.util.ai_model_checker import AiModelChecker, AiModelCheckStatus as S

LOGGER_FACTORY: str = "ai_api_unified.ai_factory"
LOGGER_CHECK: str = "ai_api_unified.util.ai_model_checker"

BASE: dict[str, str] = {
    "ANTHROPIC_API_KEY": "test-key",
    "COMPLETIONS_ENGINE": "claude",
    "COMPLETIONS_MODEL_NAME": "claude-haiku-4-5",
}
ROUTER: str = (
    '{"tiers": ['
    '{"name": "easy", "models": ["claude:claude-haiku-4-5", "claude:claude-gone-1"]},'
    '{"name": "medium", "models": ["claude:claude-gone-2"]},'
    '{"name": "hard", "models": ["claude:claude-opus-5-5"]}'
    '], "default_tier": "medium"}'
)


@contextmanager
def _settings(**values: str) -> Iterator[None]:
    dict_all = {**BASE, **values}
    with (
        patch(
            "ai_api_unified.util.env_settings.EnvSettings.get_setting",
            side_effect=lambda key, default=None: dict_all.get(key, default),
        ),
        patch.dict(os.environ, dict_all, clear=False),
    ):
        yield


@pytest.fixture(autouse=True)
def _fake_probes() -> Iterator[None]:
    """Every engine answers from a fixed table: ids containing 'gone' are missing."""
    pytest.importorskip("anthropic")
    dict_saved = dict(AiModelChecker._DICT_PROBE_BY_ENGINE)
    AiModelChecker.reset_cache()

    def _probe(model: str, base_url: str | None):
        if "gone" in model:
            return S.MISSING, "fake 404"
        if model == "unverifiable":
            return S.UNVERIFIABLE, "no key"
        return S.AVAILABLE, "ok"

    for token in ("claude", "openai", "openai-decisions", "google-gemini"):
        AiModelChecker.register_probe(token, _probe)
    yield
    AiModelChecker._DICT_PROBE_BY_ENGINE.clear()
    AiModelChecker._DICT_PROBE_BY_ENGINE.update(dict_saved)
    AiModelChecker.reset_cache()


class TestFallbackChain:
    def test_missing_fallback_is_pruned_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            _settings(
                AI_MODEL_CHECK="prune",
                COMPLETIONS_FALLBACKS="claude:claude-gone-1,claude:claude-sonnet-5",
            ),
            caplog.at_level(logging.WARNING, logger=LOGGER_CHECK),
        ):
            client = AIFactory.get_ai_completions_client()
        assert isinstance(client, AiFallbackCompletions)
        assert [c.label for c in client.candidates] == [
            "claude:claude-haiku-4-5",
            "claude:claude-sonnet-5",
        ]
        assert any(
            "claude:claude-gone-1 is not available and will be ignored"
            in r.getMessage()
            for r in caplog.records
        )

    def test_unverifiable_and_errored_models_are_kept(self) -> None:
        with _settings(
            AI_MODEL_CHECK="prune", COMPLETIONS_FALLBACKS="claude:unverifiable"
        ):
            client = AIFactory.get_ai_completions_client()
        assert isinstance(client, AiFallbackCompletions) and len(client.candidates) == 2

    def test_every_fallback_pruned_leaves_the_plain_engine(self) -> None:
        with _settings(
            AI_MODEL_CHECK="prune", COMPLETIONS_FALLBACKS="claude:claude-gone-1"
        ):
            client = AIFactory.get_ai_completions_client()
        assert not isinstance(client, AiFallbackCompletions)

    def test_primary_missing_raises_even_in_prune_mode(self) -> None:
        with _settings(
            AI_MODEL_CHECK="prune",
            COMPLETIONS_MODEL_NAME="claude-gone-0",
            COMPLETIONS_FALLBACKS="claude:claude-sonnet-5",
        ):
            with pytest.raises(ValueError, match="default route"):
                AIFactory.get_ai_completions_client()

    def test_strict_mode_raises_listing_missing_models(self) -> None:
        with _settings(
            AI_MODEL_CHECK="strict", COMPLETIONS_FALLBACKS="claude:claude-gone-1"
        ):
            with pytest.raises(ValueError, match="strict.*claude:claude-gone-1"):
                AIFactory.get_ai_completions_client()

    def test_off_mode_skips_the_check(self) -> None:
        with _settings(
            AI_MODEL_CHECK="off", COMPLETIONS_FALLBACKS="claude:claude-gone-1"
        ):
            client = AIFactory.get_ai_completions_client()
        assert isinstance(client, AiFallbackCompletions) and len(client.candidates) == 2
        assert AiModelChecker.cached(client.candidates[1]) is None

    def test_no_chain_and_no_router_runs_no_check(self) -> None:
        with _settings(AI_MODEL_CHECK="strict", COMPLETIONS_MODEL_NAME="claude-gone-0"):
            AIFactory.get_ai_completions_client()  # the engine itself would complain at call time

    def test_report_is_exposed(self) -> None:
        with _settings(
            AI_MODEL_CHECK="prune", COMPLETIONS_FALLBACKS="claude:claude-gone-1"
        ):
            AIFactory.get_ai_completions_client()
        report = AIFactory.last_model_check_report()
        assert {(r.label, r.status) for r in report} == {
            ("claude:claude-haiku-4-5", S.AVAILABLE),
            ("claude:claude-gone-1", S.MISSING),
        }

    def test_models_are_probed_once_across_factory_calls(self) -> None:
        with _settings(
            AI_MODEL_CHECK="prune", COMPLETIONS_FALLBACKS="claude:claude-sonnet-5"
        ):
            AIFactory.get_ai_completions_client()
            with patch.object(
                AiModelChecker, "_probe", side_effect=AssertionError("re-probed")
            ):
                AIFactory.get_ai_completions_client()


class TestRouter:
    def test_tier_models_are_pruned_and_empty_tiers_remap_upward(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            _settings(
                AI_MODEL_CHECK="prune",
                COMPLETIONS_ROUTER_CONFIG=ROUTER,
                COMPLETIONS_ROUTER_CLASSIFIER="none",
            ),
            caplog.at_level(logging.WARNING, logger=LOGGER_FACTORY),
        ):
            client = AIFactory.get_ai_completions_client()
        assert isinstance(client, AiRouterCompletions)
        assert [t.label for t in client.tiers] == [
            "easy(claude:claude-haiku-4-5)",
            "hard(claude:claude-opus-5-5)",
        ]
        assert client.config.pruned_tiers == {"medium": "hard"}
        # default_tier was medium, which was dropped: it moves up to hard.
        assert client.config.default_tier == "hard"
        pruned = [
            r
            for r in caplog.records
            if getattr(r, "ai_model_check_event", "") == "router_tier_pruned"
        ]
        assert pruned[0].tier == "medium" and pruned[0].escalate_to == "hard"

    def test_top_tier_dropped_maps_to_the_default_route(self) -> None:
        router = '{"tiers": [{"name": "easy", "models": ["claude:claude-haiku-4-5"]}, {"name": "hard", "models": ["claude:claude-gone-9"]}]}'
        with _settings(
            AI_MODEL_CHECK="prune",
            COMPLETIONS_ROUTER_CONFIG=router,
            COMPLETIONS_ROUTER_CLASSIFIER="none",
        ):
            client = AIFactory.get_ai_completions_client()
        assert isinstance(client, AiRouterCompletions)
        assert client.config.pruned_tiers == {"hard": None}
        assert client.for_tier("hard") is client.default_client

    def test_every_tier_dropped_disables_routing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        router = '{"tiers": [{"name": "easy", "models": ["claude:claude-gone-1"]}]}'
        with (
            _settings(AI_MODEL_CHECK="prune", COMPLETIONS_ROUTER_CONFIG=router),
            caplog.at_level(logging.WARNING, logger=LOGGER_FACTORY),
        ):
            client = AIFactory.get_ai_completions_client()
        assert not isinstance(client, AiRouterCompletions)
        assert any(
            getattr(r, "ai_model_check_event", "") == "router_disabled"
            for r in caplog.records
        )

    def test_missing_classifier_model_degrades_the_router(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            _settings(
                AI_MODEL_CHECK="prune",
                COMPLETIONS_ROUTER_CONFIG=ROUTER,
                COMPLETIONS_ROUTER_CLASSIFIER="claude:claude-gone-7",
            ),
            caplog.at_level(logging.WARNING),
        ):
            client = AIFactory.get_ai_completions_client()
        assert isinstance(client, AiRouterCompletions) and client.classifier is None
        unavailable = [
            r
            for r in caplog.records
            if getattr(r, "ai_router_event", "") == "classifier_unavailable"
        ]
        assert "claude:claude-gone-7" in unavailable[0].reason

    def test_classifier_model_is_part_of_the_report(self) -> None:
        with _settings(
            AI_MODEL_CHECK="prune",
            COMPLETIONS_ROUTER_CONFIG=ROUTER,
            COMPLETIONS_ROUTER_CLASSIFIER="claude:claude-haiku-4-5",
        ):
            AIFactory.get_ai_completions_client()
        labels = {r.label for r in AIFactory.last_model_check_report()}
        assert "claude:claude-gone-2" in labels and "claude:claude-haiku-4-5" in labels
