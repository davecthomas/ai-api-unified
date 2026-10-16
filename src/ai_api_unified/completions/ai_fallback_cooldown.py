# ai_fallback_cooldown.py
"""
Cooldowns: a fallback candidate that just failed sits out for a while.

Without a cooldown every call starts at the primary, so during an outage
each call pays the primary's full retry backoff and sends one more doomed
request before failing over. With one, the first trigger failure benches
the candidate for a period that depends on the reason; calls during that
period start at the next candidate that is not resting; when the period
ends, the candidate is back in its place and the next call tries it.

Rules:

- Only a trigger failure (a reason in the chain's fallback_on set) benches
  a candidate. An account-wide failure (QUOTA_EXHAUSTED or AUTH_REJECTED,
  unless the engine marks it per-model) benches every candidate on the
  same account (engine and base_url).
- A resting candidate is moved to the end of a call's order, not removed:
  when every other candidate fails, it is still tried, so a cooldown never
  fails a call that would have been served without one.
- A rotated key ends the cooldowns on that engine: each entry records a
  fingerprint of the engine's credential settings (the model check's
  one-way hash), and an entry whose fingerprint no longer matches is
  dropped.
- A candidate that serves a call is cleared at once.

The registry is per process and shared by every chain and router tier, so a
model benched by one chain is benched for all of them. Nothing is stored
outside the process.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..ai_provider_exceptions import AiFallbackReason

_LOGGER: logging.Logger = logging.getLogger(
    "ai_api_unified.completions.ai_fallback_completions"
)

# Values of the `ai_fallback_event` field on cooldown log records.
FALLBACK_EVENT_COOLDOWN: str = "candidate_cooldown"
FALLBACK_EVENT_RECOVERED: str = "candidate_recovered"
FALLBACK_EVENT_RESTING: str = "candidate_resting"

# Default rest per reason, in seconds. A reason absent here benches for
# DEFAULT_COOLDOWN_SECONDS; INVALID_OUTPUT is about one answer, not the
# model's health, so it never benches.
DICT_DEFAULT_COOLDOWN_SECONDS: dict[AiFallbackReason, float] = {
    AiFallbackReason.UNAVAILABLE: 30.0,
    AiFallbackReason.RATE_LIMITED: 30.0,
    AiFallbackReason.TIMEOUT: 30.0,
    AiFallbackReason.QUOTA_EXHAUSTED: 300.0,
    AiFallbackReason.MODEL_UNAVAILABLE: 300.0,
    AiFallbackReason.AUTH_REJECTED: 3600.0,
    AiFallbackReason.INVALID_OUTPUT: 0.0,
}
DEFAULT_COOLDOWN_SECONDS: float = 30.0

TypeCandidateKey = tuple[str, str, str | None]
TypeAccountKey = tuple[str, str | None]


class AIFallbackCooldownPolicy(BaseModel):
    """
    How long a failed candidate rests.

    Attributes:
        enabled: False turns cooldowns off; every call starts at the primary.
        seconds: Rest per reason value (for example {"rate_limited": 10}),
            merged over DICT_DEFAULT_COOLDOWN_SECONDS. 0 means no rest.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    seconds: dict[AiFallbackReason, float] = Field(default_factory=dict)

    @field_validator("seconds", mode="before")
    @classmethod
    def _parse_seconds(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            # Early return: pydantic reports the type error.
            return value
        dict_parsed: dict[AiFallbackReason, float] = {}
        # Loop over the entries to accept reason values in any case.
        for key, raw in value.items():
            reason: AiFallbackReason = (
                key
                if isinstance(key, AiFallbackReason)
                else AiFallbackReason(str(key).strip().lower())
            )
            float_seconds: float = float(raw)
            if float_seconds < 0:
                raise ValueError(f"cooldown seconds for {reason.value} must be >= 0")
            dict_parsed[reason] = float_seconds
        # Normal return with the parsed overrides.
        return dict_parsed

    def seconds_for(self, reason: AiFallbackReason) -> float:
        """The rest for one reason; 0 when disabled or when it never benches."""
        if not self.enabled:
            # Early return: cooldowns are off.
            return 0.0
        # Normal return with the override, the default, or the fallback.
        return self.seconds.get(
            reason, DICT_DEFAULT_COOLDOWN_SECONDS.get(reason, DEFAULT_COOLDOWN_SECONDS)
        )


def parse_cooldown_policy(value: Any) -> AIFallbackCooldownPolicy:
    """
    Normalizes a configured cooldown.

    Args:
        value: None (defaults), a bool (on or off), a dict
            ({"enabled": ..., "seconds": {...}}), or a policy.

    Returns:
        The policy.
    """
    if isinstance(value, AIFallbackCooldownPolicy):
        # Early return: already a policy.
        return value
    if value is None:
        # Early return: the defaults.
        return AIFallbackCooldownPolicy()
    if isinstance(value, bool):
        # Early return: on with the defaults, or off.
        return AIFallbackCooldownPolicy(enabled=value)
    # Normal return with the parsed mapping.
    return AIFallbackCooldownPolicy.model_validate(value)


@dataclass(frozen=True)
class _Rest:
    """One bench: until when, why, and the credentials it was taken with."""

    float_until: float
    reason: AiFallbackReason
    str_fingerprint: str


class AiCandidateCooldowns:
    """Process-wide benches, keyed by candidate and by account."""

    _DICT_CANDIDATES: ClassVar[dict[TypeCandidateKey, _Rest]] = {}
    _DICT_ACCOUNTS: ClassVar[dict[TypeAccountKey, _Rest]] = {}
    _LOCK: ClassVar[threading.Lock] = threading.Lock()
    # Clock; tests replace it.
    callable_now: ClassVar[Callable[[], float]] = time.monotonic

    @classmethod
    def reset(cls) -> None:
        """Ends every cooldown (tests, or a deliberate reset)."""
        with cls._LOCK:
            cls._DICT_CANDIDATES.clear()
            cls._DICT_ACCOUNTS.clear()

    @staticmethod
    def _fingerprint(str_engine: str) -> str:
        # Imported here: the checker imports the fallback module.
        from ..util.ai_model_checker import AiModelChecker

        # Normal return with the engine's credential fingerprint.
        return AiModelChecker._credential_fingerprint(str_engine)

    @classmethod
    def bench(
        cls,
        *,
        str_engine: str,
        str_model: str,
        base_url: str | None,
        reason: AiFallbackReason,
        bool_account_wide: bool,
        float_seconds: float,
        str_label: str,
    ) -> None:
        """Benches one candidate (or its whole account) for float_seconds."""
        if float_seconds <= 0:
            # Early return: this reason does not bench.
            return
        rest: _Rest = _Rest(
            float_until=cls.callable_now() + float_seconds,
            reason=reason,
            str_fingerprint=cls._fingerprint(str_engine),
        )
        with cls._LOCK:
            if bool_account_wide:
                cls._DICT_ACCOUNTS[(str_engine, base_url)] = rest
            else:
                cls._DICT_CANDIDATES[(str_engine, str_model, base_url)] = rest
        _LOGGER.warning(
            "FALLBACK candidate %s rests for %.0fs after %s%s; calls start at the "
            "next candidate until then.",
            str_label,
            float_seconds,
            reason.value,
            " (its whole account)" if bool_account_wide else "",
            extra={
                "ai_fallback_event": FALLBACK_EVENT_COOLDOWN,
                "fallback_candidate": str_label,
                "fallback_reason": reason.value,
                "cooldown_seconds": float_seconds,
                "account_wide": bool_account_wide,
            },
        )

    @classmethod
    def resting(
        cls, *, str_engine: str, str_model: str, base_url: str | None
    ) -> _Rest | None:
        """
        The bench covering one candidate, or None.

        A bench taken with other credentials is dropped here. An expired one
        stays until the candidate serves a call (clear logs the recovery) or
        fails again (bench replaces it).
        """
        if not cls._DICT_CANDIDATES and not cls._DICT_ACCOUNTS:
            # Early return: nothing is resting.
            return None
        tuple_keys: tuple[tuple[dict[Any, _Rest], Any], ...] = (
            (cls._DICT_CANDIDATES, (str_engine, str_model, base_url)),
            (cls._DICT_ACCOUNTS, (str_engine, base_url)),
        )
        found: _Rest | None = None
        # Loop over the candidate's own bench and its account's.
        for dict_rests, key in tuple_keys:
            rest: _Rest | None = dict_rests.get(key)
            if rest is None:
                continue
            if rest.str_fingerprint != cls._fingerprint(str_engine):
                with cls._LOCK:
                    if dict_rests.get(key) is rest:
                        del dict_rests[key]
                continue
            if rest.float_until <= cls.callable_now():
                # Expired: the candidate is back in its place.
                continue
            if found is None or rest.float_until > found.float_until:
                found = rest
        # Normal return with the longest live bench, if any.
        return found

    @classmethod
    def clear(
        cls, *, str_engine: str, str_model: str, base_url: str | None, str_label: str
    ) -> None:
        """Ends a candidate's bench after it served a call."""
        if not cls._DICT_CANDIDATES and not cls._DICT_ACCOUNTS:
            # Early return: nothing is resting.
            return
        with cls._LOCK:
            removed_candidate: _Rest | None = cls._DICT_CANDIDATES.pop(
                (str_engine, str_model, base_url), None
            )
            removed_account: _Rest | None = cls._DICT_ACCOUNTS.pop(
                (str_engine, base_url), None
            )
        if removed_candidate is not None or removed_account is not None:
            _LOGGER.info(
                "FALLBACK candidate %s served a call; its cooldown is over.",
                str_label,
                extra={
                    "ai_fallback_event": FALLBACK_EVENT_RECOVERED,
                    "fallback_candidate": str_label,
                },
            )
