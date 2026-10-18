# ai_model_pin.py
"""
Per-call model pins: one call names the model that answers it.

`send_prompt_with_result(model=...)` pins one call. The pin skips routing
(no classifier, no tier pick) and keeps everything else: the call starts on
the pinned model, falls back along the chain the configuration defines when
that model fails with a trigger reason, and is metered, logged, and
redacted like any other call.

Where the call falls back:

- On a fallback chain: the pinned model first, then the chain's own
  candidates in order. A pinned model outside the chain is built on first
  use and kept for later pins.
- On a router: the chain of the tier that holds the pinned model, else the
  default chain. A pinned call never escalates to another tier; the pin
  already decided the difficulty.
- On a single configured model (no chain): the pinned model, then the
  configured model.

Each pinned call logs one MODEL PIN line and carries two observability
tags, `model_pin` (the pinned engine:model) and `route_method`
(`model_pin`), on every event and on its request cost report, so an
operator can find every call that bypassed routing.

A pin is `engine:model` (`claude:claude-sonnet-4-5`) or a bare model id
that one of the client's configured candidates uses. The split is on the
first colon, and only when the text before it is a completions engine
token, since Bedrock model ids contain colons. The models.json allow/deny
policy applies to pinned models as it does to configured ones.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

from ..ai_provider_exceptions import AiProviderConfigurationError
from ..ai_provider_registry import (
    AI_PROVIDER_CAPABILITY_COMPLETIONS,
    get_ai_provider_spec,
)
from ..middleware.observability_runtime import (
    push_observability_tags,
    reset_observability_context,
)
from .ai_fallback_completions import AIFallbackCandidate

_LOGGER: logging.Logger = logging.getLogger(__name__)

# Value of the `route_method` tag, and of a routed event's `method`, on a
# pinned call.
ROUTE_METHOD_MODEL_PIN: str = "model_pin"
# Observability tag naming the pinned engine:model.
TAG_MODEL_PIN: str = "model_pin"
TAG_ROUTE_METHOD: str = "route_method"
# Value of the `ai_route_event` field on the pin's log line.
ROUTE_EVENT_MODEL_PIN: str = "model_pin"

MODEL_PIN: ContextVar[AIFallbackCandidate | None] = ContextVar(
    "ai_api_unified_model_pin", default=None
)


def current_model_pin() -> AIFallbackCandidate | None:
    """The model the running call is pinned to, or None."""
    # Normal return with the contextvar value.
    return MODEL_PIN.get()


def _is_completions_engine(str_token: str) -> bool:
    """True when the token names a registered completions engine."""
    try:
        get_ai_provider_spec(AI_PROVIDER_CAPABILITY_COMPLETIONS, str_token)
    except AiProviderConfigurationError:
        # Early return: not an engine token.
        return False
    # Normal return: a registered engine.
    return True


def _same_model(candidate: AIFallbackCandidate, pin: AIFallbackCandidate) -> bool:
    """True when a configured candidate is the pinned engine and model."""
    # Normal return: engine and model decide; a pin carries no base_url.
    return candidate.engine == pin.engine and candidate.model == pin.model


def find_pinned_candidate(
    pin: AIFallbackCandidate, candidates: Sequence[AIFallbackCandidate]
) -> int | None:
    """
    Position of the configured candidate the pin names.

    Args:
        pin: The resolved pin.
        candidates: Candidates to search, in order.

    Returns:
        The first matching position, or None.
    """
    # Loop over the candidates for the pinned engine and model.
    for int_index, candidate in enumerate(candidates):
        if _same_model(candidate, pin):
            # Early return with the match.
            return int_index
    # Normal return: the pin is not configured here.
    return None


def resolve_model_pin(
    str_pin: str, known_candidates: Sequence[AIFallbackCandidate]
) -> AIFallbackCandidate:
    """
    Turns a caller's pin into the candidate it names.

    Args:
        str_pin: `engine:model`, or a bare model id.
        known_candidates: The client's configured candidates, searched for a
            bare model id and for the base_url of a configured pair.

    Returns:
        The configured candidate when the pin names one (keeping its
        base_url), else a new candidate for the engine and model.

    Raises:
        ValueError: The pin is blank, names an engine without a model, or
            is a bare model id no configured candidate uses.
    """
    str_text: str = (str_pin or "").strip()
    if not str_text:
        raise ValueError("A model pin cannot be blank; pass engine:model.")
    str_engine, str_sep, str_model = str_text.partition(":")
    str_engine = str_engine.strip().lower()
    if str_sep and _is_completions_engine(str_engine):
        str_model = str_model.strip()
        if not str_model:
            raise ValueError(
                f"Model pin {str_pin!r} names the {str_engine!r} engine but no "
                "model; pass engine:model."
            )
        pin: AIFallbackCandidate = AIFallbackCandidate(
            engine=str_engine, model=str_model
        )
        int_known: int | None = find_pinned_candidate(pin, known_candidates)
        # Normal return with the configured pair (and its base_url), or the
        # new pair.
        return known_candidates[int_known] if int_known is not None else pin
    # Loop over the configured candidates for one using this bare model id.
    for candidate in known_candidates:
        if candidate.model == str_text:
            # Early return with the configured candidate.
            return candidate
    raise ValueError(
        f"Model pin {str_pin!r} names no engine, and no configured candidate "
        f"uses that model ({', '.join(c.label for c in known_candidates) or 'none'}); "
        "pass engine:model, for example openai:gpt-4o-mini."
    )


@contextmanager
def model_pin_scope(
    pin: AIFallbackCandidate | None, str_operation: str, *, config_id: str | None
) -> Iterator[None]:
    """
    Pins every call inside the block to one model, tags it, and logs it.

    A None pin is a no-op, so callers can use the scope unconditionally.

    Args:
        pin: The resolved pin, or None.
        str_operation: Public method name, for the log line.
        config_id: The client's configuration id, for the log line.
    """
    if pin is None:
        yield
        # Early return: nothing pinned.
        return
    _LOGGER.info(
        "MODEL PIN: %s pinned to %s; routing skipped, fallback kept.",
        str_operation,
        pin.label,
        extra={
            "ai_route_event": ROUTE_EVENT_MODEL_PIN,
            "operation": str_operation,
            "model_pin": pin.label,
            "config_id": config_id,
        },
    )
    token_pin: Token[AIFallbackCandidate | None] = MODEL_PIN.set(pin)
    token_tags: Any = push_observability_tags(
        {TAG_MODEL_PIN: pin.label, TAG_ROUTE_METHOD: ROUTE_METHOD_MODEL_PIN}
    )
    try:
        yield
    finally:
        reset_observability_context(token_tags)
        MODEL_PIN.reset(token_pin)
