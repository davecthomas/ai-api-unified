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

The benches live in the library's cache store (middleware/cache_store.py):
in memory and per process by default, shared by every chain and router
tier, so a model benched by one chain is benched for all of them. An
application that hands the library a shared store (AIFactory.set_cache_store)
shares benches across its processes too: one worker's rate-limit hit then
benches the model for every worker. The store is treated as unreliable: a
store failure means no bench, never a failed call.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..ai_provider_exceptions import AiFallbackReason
from ..middleware.cache_store import store_delete, store_get, store_set

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

# Store key prefixes; see middleware/cache_store.py for the namespace rule.
KEY_PREFIX_CANDIDATE: str = "cd:v1:cand"
KEY_PREFIX_ACCOUNT: str = "cd:v1:acct"
# A bench stays in the store this much longer than it lasts, so the clear
# that follows a served call still finds it and can log the recovery.
FLOAT_STORE_GRACE_SECONDS: float = 60.0


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
    """Benches in the cache store, keyed by candidate and by account."""

    _LOCK: ClassVar[threading.Lock] = threading.Lock()
    # Keys this process has written, so reset() can remove them from a store
    # that cannot enumerate.
    _SET_KEYS: ClassVar[set[str]] = set()
    # Clock; tests replace it.
    callable_now: ClassVar[Callable[[], float]] = time.monotonic

    @staticmethod
    def _candidate_key(str_engine: str, str_model: str, base_url: str | None) -> str:
        # Normal return with the candidate's store key.
        return f"{KEY_PREFIX_CANDIDATE}:{str_engine}|{str_model}|{base_url or ''}"

    @staticmethod
    def _account_key(str_engine: str, base_url: str | None) -> str:
        # Normal return with the account's store key.
        return f"{KEY_PREFIX_ACCOUNT}:{str_engine}|{base_url or ''}"

    @classmethod
    def reset(cls) -> None:
        """Ends every cooldown this process set (tests, or a deliberate reset)."""
        with cls._LOCK:
            set_keys: set[str] = set(cls._SET_KEYS)
            cls._SET_KEYS.clear()
        # Loop over the keys written here and remove each from the store.
        for str_key in set_keys:
            store_delete(str_key)

    @staticmethod
    def _fingerprint(str_engine: str) -> str:
        # Imported here: the checker imports the fallback module.
        from ..util.ai_model_checker import AiModelChecker

        # Normal return with the engine's credential fingerprint.
        return AiModelChecker._credential_fingerprint(str_engine)

    @staticmethod
    def _encode(rest: _Rest) -> bytes:
        # Normal return with the bench as JSON bytes.
        return json.dumps(
            {
                "until": rest.float_until,
                "reason": rest.reason.value,
                "fp": rest.str_fingerprint,
            }
        ).encode("utf-8")

    @staticmethod
    def _decode(value: bytes | None) -> _Rest | None:
        if value is None:
            # Early return: nothing stored.
            return None
        try:
            dict_value: Any = json.loads(value.decode("utf-8"))
            # Normal return with the bench.
            return _Rest(
                float_until=float(dict_value["until"]),
                reason=AiFallbackReason(str(dict_value["reason"])),
                str_fingerprint=str(dict_value["fp"]),
            )
        except Exception:
            # Early return: a foreign or corrupt entry is no bench.
            return None

    @classmethod
    def _write(cls, str_key: str, rest: _Rest, float_seconds: float) -> None:
        with cls._LOCK:
            cls._SET_KEYS.add(str_key)
        store_set(str_key, cls._encode(rest), float_seconds + FLOAT_STORE_GRACE_SECONDS)

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
        cls._write(
            (
                cls._account_key(str_engine, base_url)
                if bool_account_wide
                else cls._candidate_key(str_engine, str_model, base_url)
            ),
            rest,
            float_seconds,
        )
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
        stays until the candidate serves a call (clear logs the recovery),
        fails again (bench replaces it), or the store's grace period ends.
        """
        found: _Rest | None = None
        # Loop over the candidate's own bench and its account's.
        for str_key in (
            cls._candidate_key(str_engine, str_model, base_url),
            cls._account_key(str_engine, base_url),
        ):
            rest: _Rest | None = cls._decode(store_get(str_key))
            if rest is None:
                continue
            if rest.str_fingerprint != cls._fingerprint(str_engine):
                store_delete(str_key)
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
        bool_removed: bool = False
        # Loop over the candidate's own bench and its account's.
        for str_key in (
            cls._candidate_key(str_engine, str_model, base_url),
            cls._account_key(str_engine, base_url),
        ):
            if store_get(str_key) is not None:
                store_delete(str_key)
                bool_removed = True
        if bool_removed:
            _LOGGER.info(
                "FALLBACK candidate %s served a call; its cooldown is over.",
                str_label,
                extra={
                    "ai_fallback_event": FALLBACK_EVENT_RECOVERED,
                    "fallback_candidate": str_label,
                },
            )
