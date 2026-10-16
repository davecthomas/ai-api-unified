# ai_model_checker.py
"""
Verifies that every configured model exists and can be reached before it
is relied on.

A fallback chain, a router tier, or a classifier names models the process
expects to call later. A typo, a retired id, a model not offered in the
configured region, or a fallback whose key is missing would only surface
when that model is finally tried, in the middle of a failover. The checker
asks each provider for the model's metadata once (one free call, no
completion) with the credentials the engine would use, memoizes the answer,
and the factory prunes anything missing or unreachable from the effective
configuration with a loud warning.

The memo key is (engine, model, base_url, credential fingerprint), so a
model is probed again when its id, base URL, or credentials change, and
never otherwise. The fingerprint is a hash of the settings the engine's
probe reads; a key never appears in it. An error result is not memoized,
so the next factory call probes that model again.

Statuses:
    available: the provider knows the model.
    missing: the provider says there is no such model.
    unreachable: the library cannot call the model: no credentials, the
        credentials were rejected as invalid, no base URL, or the engine's
        extra is not installed.
    unverifiable: the model cannot be asked about, but nothing says it
        cannot be called (the engine default model, no probe for the
        engine, a permission denial on the catalog call only, a server
        without a model catalog); the model is kept.
    error: the probe failed for another reason (network, a 5xx); the model
        is kept, since a transient outage is not a configuration error.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ..completions.ai_fallback_completions import AIFallbackCandidate
from .env_settings import EnvSettings

_LOGGER: logging.Logger = logging.getLogger(__name__)

# Values of the `ai_model_check_event` field on this module's log records.
MODEL_CHECK_EVENT_AVAILABLE: str = "model_available"
MODEL_CHECK_EVENT_MISSING: str = "model_missing"
MODEL_CHECK_EVENT_UNREACHABLE: str = "model_unreachable"
MODEL_CHECK_EVENT_UNVERIFIABLE: str = "model_unverifiable"
MODEL_CHECK_EVENT_ERROR: str = "model_check_error"
MODEL_CHECK_EVENT_PRUNED_TIER: str = "router_tier_pruned"

# The AI_MODEL_CHECK setting.
AI_MODEL_CHECK_KEY: str = "AI_MODEL_CHECK"
AI_MODEL_CHECK_TIMEOUT_SECONDS_KEY: str = "AI_MODEL_CHECK_TIMEOUT_SECONDS"
MODEL_CHECK_MODE_PRUNE: str = "prune"
MODEL_CHECK_MODE_STRICT: str = "strict"
MODEL_CHECK_MODE_OFF: str = "off"
FROZENSET_MODEL_CHECK_MODES: frozenset[str] = frozenset(
    {MODEL_CHECK_MODE_PRUNE, MODEL_CHECK_MODE_STRICT, MODEL_CHECK_MODE_OFF}
)
DEFAULT_MODEL_CHECK_MODE: str = MODEL_CHECK_MODE_PRUNE

# Probes run in parallel so startup pays one round trip, not one per model.
DEFAULT_MODEL_CHECK_MAX_WORKERS: int = 8
DEFAULT_MODEL_CHECK_TIMEOUT_SECONDS: float = 10.0


class AiModelCheckStatus(str, Enum):
    """What the provider said about a configured model."""

    AVAILABLE = "available"
    MISSING = "missing"
    UNREACHABLE = "unreachable"
    UNVERIFIABLE = "unverifiable"
    ERROR = "error"


class AiModelCheckResult(BaseModel):
    """
    One model's check.

    Attributes:
        engine: Engine token.
        model: Model id as configured.
        base_url: API base-URL override, if any.
        status: See AiModelCheckStatus.
        detail: Why, in one line.
        checked_at: When the probe ran.
        probe_seconds: How long it took.
    """

    model_config = ConfigDict(frozen=True)

    engine: str
    model: str
    base_url: str | None = None
    status: AiModelCheckStatus
    detail: str = ""
    checked_at: datetime
    probe_seconds: float = 0.0

    @property
    def label(self) -> str:
        """engine:model, for logs."""
        # Normal return with the log label.
        return f"{self.engine}:{self.model}"

    @property
    def keeps_model(self) -> bool:
        """True unless the model is missing or cannot be reached."""
        # Normal return: MISSING and UNREACHABLE prune.
        return self.status not in FROZENSET_PRUNING_STATUSES


# Statuses that take a model out of the effective configuration.
FROZENSET_PRUNING_STATUSES: frozenset[AiModelCheckStatus] = frozenset(
    {AiModelCheckStatus.MISSING, AiModelCheckStatus.UNREACHABLE}
)


# A probe takes (model, base_url) and returns (status, detail). It must make
# at most one metadata call and must not run a completion.
TypeModelProbe = Callable[[str, str | None], tuple[AiModelCheckStatus, str]]

# (engine, model, base_url, credential fingerprint).
TypeCheckKey = tuple[str, str, str | None, str]


class AiModelChecker:
    """
    Memoized, parallel existence checks for configured models.

    One instance per process is enough; the memo is class-level so every
    factory call, chain, and tier shares it.
    """

    _DICT_CACHE: ClassVar[dict[TypeCheckKey, AiModelCheckResult]] = {}
    _LOCK: ClassVar[threading.Lock] = threading.Lock()
    _DICT_PROBE_BY_ENGINE: ClassVar[dict[str, TypeModelProbe]] = {}
    _DICT_CREDENTIAL_SETTINGS_BY_ENGINE: ClassVar[dict[str, tuple[str, ...]]] = {}

    def __init__(
        self,
        *,
        int_max_workers: int = DEFAULT_MODEL_CHECK_MAX_WORKERS,
        float_timeout_seconds: float = DEFAULT_MODEL_CHECK_TIMEOUT_SECONDS,
    ) -> None:
        self.int_max_workers: int = int_max_workers
        self.float_timeout_seconds: float = float_timeout_seconds

    # ── Probe registry ──────────────────────────────────────────────────────

    @classmethod
    def register_probe(
        cls,
        str_engine: str,
        probe: TypeModelProbe,
        tuple_credential_settings: tuple[str, ...] = (),
    ) -> None:
        """
        Registers the probe for one engine token.

        Args:
            str_engine: Engine token.
            probe: The probe.
            tuple_credential_settings: Settings the probe reads to reach the
                provider (keys, base URLs, region); a change to any of them
                makes the memoized result stale.
        """
        str_token: str = str_engine.strip().lower()
        cls._DICT_PROBE_BY_ENGINE[str_token] = probe
        cls._DICT_CREDENTIAL_SETTINGS_BY_ENGINE[str_token] = tuple(
            tuple_credential_settings
        )

    @classmethod
    def probe_for(cls, str_engine: str) -> TypeModelProbe | None:
        """The probe for one engine token, or None."""
        # Normal return with the registered probe, if any.
        return cls._DICT_PROBE_BY_ENGINE.get(str_engine.strip().lower())

    @classmethod
    def reset_cache(cls) -> None:
        """Forgets every memoized result (tests, or a deliberate recheck)."""
        with cls._LOCK:
            cls._DICT_CACHE.clear()

    @classmethod
    def cached(cls, candidate: AIFallbackCandidate) -> AiModelCheckResult | None:
        """The memoized result for one candidate, or None."""
        # Normal return with the cached result.
        return cls._DICT_CACHE.get(cls._key(candidate))

    @classmethod
    def _key(cls, candidate: AIFallbackCandidate) -> TypeCheckKey:
        # Normal return with the memo key.
        return (
            candidate.engine,
            candidate.model,
            candidate.base_url,
            cls._credential_fingerprint(candidate.engine),
        )

    @classmethod
    def _credential_fingerprint(cls, str_engine: str) -> str:
        """A hash of the settings the engine's probe reads; never the values."""
        tuple_settings: tuple[str, ...] = cls._DICT_CREDENTIAL_SETTINGS_BY_ENGINE.get(
            str_engine.strip().lower(), ()
        )
        if not tuple_settings:
            # Early return: the engine registered no credential settings.
            return ""
        env: EnvSettings = EnvSettings()
        digest: Any = hashlib.sha256()
        # Loop over the settings so any change alters the hash.
        for str_setting in tuple_settings:
            str_value: str = str(env.get_setting(str_setting, "") or "").strip()
            digest.update(f"{str_setting}={str_value}\0".encode())
        # Normal return with a short, one-way fingerprint.
        return digest.hexdigest()[:16]

    # ── Checks ──────────────────────────────────────────────────────────────

    def check(self, candidate: AIFallbackCandidate) -> AiModelCheckResult:
        """
        Checks one model, from the memo when it was checked before.

        Args:
            candidate: Engine, model, and base-URL to check.

        Returns:
            The result; never raises.
        """
        key: TypeCheckKey = self._key(candidate)
        cached: AiModelCheckResult | None = self._DICT_CACHE.get(key)
        if cached is not None:
            # Early return: checked earlier in this process.
            return cached
        result: AiModelCheckResult = self._probe(candidate)
        if result.status is AiModelCheckStatus.ERROR:
            # Early return without memoizing: the next check probes again.
            self._log(result)
            return result
        with self._LOCK:
            # Keep the first result when two threads raced on the same model.
            existing: AiModelCheckResult | None = self._DICT_CACHE.get(key)
            if existing is None:
                self._DICT_CACHE[key] = result
                self._log(result)
            else:
                result = existing
        # Normal return with the memoized result.
        return result

    def check_all(
        self, candidates: Iterable[AIFallbackCandidate]
    ) -> list[AiModelCheckResult]:
        """
        Checks many models at once, each at most once.

        Args:
            candidates: Models to check; duplicates are collapsed.

        Returns:
            One result per distinct candidate, in first-seen order.
        """
        list_unique: list[AIFallbackCandidate] = []
        set_seen: set[TypeCheckKey] = set()
        # Loop over candidates so each model is probed once.
        for candidate in candidates:
            key: TypeCheckKey = self._key(candidate)
            if key not in set_seen:
                set_seen.add(key)
                list_unique.append(candidate)
        if not list_unique:
            # Early return: nothing to check.
            return []
        if len(list_unique) == 1:
            # Early return without a pool for a single model.
            return [self.check(list_unique[0])]
        with ThreadPoolExecutor(
            max_workers=min(self.int_max_workers, len(list_unique)),
            thread_name_prefix="ai-model-check",
        ) as executor:
            # Normal return with every result, in input order.
            return list(executor.map(self.check, list_unique))

    def _probe(self, candidate: AIFallbackCandidate) -> AiModelCheckResult:
        """Runs the engine's probe with a timeout and maps failures."""
        probe: TypeModelProbe | None = self.probe_for(candidate.engine)
        float_started: float = time.perf_counter()
        status: AiModelCheckStatus
        str_detail: str
        if probe is None:
            status, str_detail = (
                AiModelCheckStatus.UNVERIFIABLE,
                f"no model probe for engine {candidate.engine!r}",
            )
        else:
            try:
                status, str_detail = self._run_with_timeout(probe, candidate)
            except TimeoutError:
                status, str_detail = (
                    AiModelCheckStatus.ERROR,
                    f"probe timed out after {self.float_timeout_seconds}s",
                )
            except Exception as exception:
                status, str_detail = (
                    AiModelCheckStatus.ERROR,
                    f"{type(exception).__name__}: {exception}",
                )
        # Normal return with the result.
        return AiModelCheckResult(
            engine=candidate.engine,
            model=candidate.model,
            base_url=candidate.base_url,
            status=status,
            detail=str_detail,
            checked_at=datetime.now(timezone.utc),
            probe_seconds=time.perf_counter() - float_started,
        )

    def _run_with_timeout(
        self, probe: TypeModelProbe, candidate: AIFallbackCandidate
    ) -> tuple[AiModelCheckStatus, str]:
        """Runs one probe on a helper thread so a hung SDK cannot stall startup."""
        # Not a context manager: __exit__ would wait for the probe thread,
        # so a hung SDK call would stall startup for as long as it hangs.
        pool: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ai-model-probe"
        )
        try:
            future = pool.submit(probe, candidate.model, candidate.base_url)
            # Normal return with the probe's answer, or TimeoutError.
            return future.result(timeout=self.float_timeout_seconds)
        finally:
            pool.shutdown(wait=False)

    @staticmethod
    def _log(result: AiModelCheckResult) -> None:
        dict_extra: dict[str, Any] = {
            "engine": result.engine,
            "model": result.model,
            "base_url": result.base_url,
            "status": result.status.value,
            "detail": result.detail,
            "probe_seconds": round(result.probe_seconds, 3),
        }
        if result.status is AiModelCheckStatus.MISSING:
            _LOGGER.warning(
                "MODEL CHECK: configured model %s is not available and will be "
                "ignored (%s).",
                result.label,
                result.detail,
                extra={"ai_model_check_event": MODEL_CHECK_EVENT_MISSING, **dict_extra},
            )
        elif result.status is AiModelCheckStatus.UNREACHABLE:
            _LOGGER.warning(
                "MODEL CHECK: configured model %s cannot be reached and will be "
                "ignored (%s).",
                result.label,
                result.detail,
                extra={
                    "ai_model_check_event": MODEL_CHECK_EVENT_UNREACHABLE,
                    **dict_extra,
                },
            )
        elif result.status is AiModelCheckStatus.ERROR:
            _LOGGER.warning(
                "MODEL CHECK: could not verify %s, keeping it (%s).",
                result.label,
                result.detail,
                extra={"ai_model_check_event": MODEL_CHECK_EVENT_ERROR, **dict_extra},
            )
        elif result.status is AiModelCheckStatus.UNVERIFIABLE:
            _LOGGER.info(
                "MODEL CHECK: %s not verifiable, keeping it (%s).",
                result.label,
                result.detail,
                extra={
                    "ai_model_check_event": MODEL_CHECK_EVENT_UNVERIFIABLE,
                    **dict_extra,
                },
            )
        else:
            _LOGGER.debug(
                "MODEL CHECK: %s available.",
                result.label,
                extra={
                    "ai_model_check_event": MODEL_CHECK_EVENT_AVAILABLE,
                    **dict_extra,
                },
            )


def resolve_model_check_mode(object_value: object) -> str:
    """
    Normalizes the AI_MODEL_CHECK setting.

    Args:
        object_value: Raw setting value.

    Returns:
        "prune" (default), "strict", or "off".

    Raises:
        ValueError: An unknown mode.
    """
    str_mode: str = str(object_value or "").strip().lower() or DEFAULT_MODEL_CHECK_MODE
    if str_mode not in FROZENSET_MODEL_CHECK_MODES:
        raise ValueError(
            f"{AI_MODEL_CHECK_KEY} must be one of "
            f"{', '.join(sorted(FROZENSET_MODEL_CHECK_MODES))}; got {object_value!r}."
        )
    # Normal return with the mode.
    return str_mode


def prune_candidates(
    list_candidates: list[AIFallbackCandidate],
    dict_results: dict[tuple[str, str, str | None], AiModelCheckResult],
) -> list[AIFallbackCandidate]:
    """
    Drops candidates that are missing or unreachable; keeps the rest in order.

    Args:
        list_candidates: Configured chain.
        dict_results: Check results by (engine, model, base_url).

    Returns:
        The chain without missing or unreachable models.
    """
    # Normal return with the surviving candidates.
    return [
        candidate
        for candidate in list_candidates
        if (
            result := dict_results.get(
                (candidate.engine, candidate.model, candidate.base_url)
            )
        )
        is None
        or result.keeps_model
    ]
