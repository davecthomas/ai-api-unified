"""
Shared observability runtime objects, request context helpers, and wrapper utilities.
"""

from __future__ import annotations

import logging
import time
from types import MappingProxyType
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, TypeVar

from collections.abc import Callable, Mapping

if TYPE_CHECKING:
    from ai_api_unified.middleware.observability import (
        AiApiObservabilityMiddleware,
    )

_LOGGER: logging.Logger = logging.getLogger(__name__)

OBSERVABILITY_DIRECTION_INPUT: str = "input"
OBSERVABILITY_DIRECTION_OUTPUT: str = "output"
OBSERVABILITY_DIRECTION_ERROR: str = "error"
ORIGINATING_CALLER_ID_SOURCE_APPLICATION_CONTEXT: str = "application_context"
ORIGINATING_CALLER_ID_SOURCE_LEGACY_SETTING: str = "legacy_setting"
ORIGINATING_CALLER_ID_SOURCE_NONE: str = "none"
TOKEN_COUNT_SOURCE_PROVIDER: str = "provider"
TOKEN_COUNT_SOURCE_ESTIMATED: str = "estimated"
TOKEN_COUNT_SOURCE_NONE: str = "none"
OBSERVABILITY_CONTEXT_HOOK_FAILURE_LOG_MESSAGE: str = (
    "observability_hook_failed stage=%s error_type=%s"
)

ObservabilityMetadataValue = str | int | float | bool | None | tuple[str, ...]
ProviderCallReturnType = TypeVar("ProviderCallReturnType")


@dataclass(frozen=True)
class ObservabilityContextModel:
    """
    Request-scoped caller correlation data supplied explicitly by application code.

    Attributes:
        caller_id: Optional stable, non-sensitive caller identifier for correlation.
        session_id: Optional session identifier supplied by application code.
        workflow_id: Optional workflow identifier supplied by application code.
    """

    caller_id: str | None = None
    session_id: str | None = None
    workflow_id: str | None = None


DEFAULT_OBSERVABILITY_CONTEXT: ObservabilityContextModel = ObservabilityContextModel()
OBSERVABILITY_CONTEXT: ContextVar[ObservabilityContextModel] = ContextVar(
    "ai_api_unified_observability_context",
    default=DEFAULT_OBSERVABILITY_CONTEXT,
)


@dataclass(frozen=True)
class AiApiCallContextModel:
    """
    Immutable provider-call metadata used by observability lifecycle middleware.

    Attributes:
        call_id: Stable identifier shared across input, output, and error events for one public call.
        event_time_utc: UTC timestamp for the specific event being emitted.
        capability: Capability name such as `completions`, `embeddings`, `images`, or `tts`.
        operation: Public operation name such as `send_prompt` or `generate_embeddings`.
        provider_vendor: Best-effort provider vendor name.
        provider_engine: Best-effort provider engine name used by the concrete client.
        model_name: Best-effort model identifier used for the provider call.
        model_version: Best-effort model version identifier for the provider call.
        direction: Lifecycle event direction label (`input`, `output`, or `error`).
        originating_caller_id: Optional explicit caller identifier supplied by application code.
        originating_caller_id_source: Source label describing how the caller id was resolved.
        dict_metadata: Additional scalar metadata fields safe to emit in logs.
    """

    call_id: str
    event_time_utc: datetime
    capability: str
    operation: str
    provider_vendor: str
    provider_engine: str
    model_name: str | None
    model_version: str | None
    direction: str
    originating_caller_id: str | None = None
    originating_caller_id_source: str = ORIGINATING_CALLER_ID_SOURCE_NONE
    dict_metadata: Mapping[str, ObservabilityMetadataValue] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """
        Freezes caller-supplied metadata so the call-context object remains effectively immutable.

        Args:
            None

        Returns:
            None after metadata has been copied and wrapped in an immutable mapping.
        """
        frozen_metadata: Mapping[str, ObservabilityMetadataValue] = MappingProxyType(
            dict(self.dict_metadata)
        )
        object.__setattr__(self, "dict_metadata", frozen_metadata)
        # Normal return after replacing metadata with an immutable mapping copy.
        return None

    def with_direction(self, direction: str) -> "AiApiCallContextModel":
        """
        Returns a copy of the call context for a different lifecycle direction.

        Args:
            direction: Lifecycle direction label for the derived event context.

        Returns:
            New AiApiCallContextModel with a fresh UTC event time and the requested direction.
        """
        # Normal return with a derived immutable call-context instance for one event direction.
        return replace(
            self,
            direction=direction,
            event_time_utc=datetime.now(timezone.utc),
        )


@dataclass(frozen=True)
class AiApiCallResultSummaryModel:
    """
    Lightweight output metadata summary emitted by observability lifecycle middleware.

    Attributes:
        provider_elapsed_ms: Wall-clock provider-call duration in milliseconds.
        input_token_count: Optional input token count when cheaply available.
        input_token_count_source: Source label for the input token count.
        output_token_count: Optional output token count when cheaply available.
        output_token_count_source: Source label for the output token count.
        provider_prompt_tokens: Optional provider-reported prompt/input token count.
        provider_completion_tokens: Optional provider-reported completion/output token count.
        provider_total_tokens: Optional provider-reported total token count.
        finish_reason: Optional provider finish reason.
        dict_metadata: Additional scalar metadata fields safe to emit in logs.
    """

    provider_elapsed_ms: float
    input_token_count: int | None = None
    input_token_count_source: str = TOKEN_COUNT_SOURCE_NONE
    output_token_count: int | None = None
    output_token_count_source: str = TOKEN_COUNT_SOURCE_NONE
    provider_prompt_tokens: int | None = None
    provider_completion_tokens: int | None = None
    provider_total_tokens: int | None = None
    finish_reason: str | None = None
    dict_metadata: Mapping[str, ObservabilityMetadataValue] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """
        Freezes caller-supplied metadata so the result-summary object remains effectively immutable.

        Args:
            None

        Returns:
            None after metadata has been copied and wrapped in an immutable mapping.
        """
        frozen_metadata: Mapping[str, ObservabilityMetadataValue] = MappingProxyType(
            dict(self.dict_metadata)
        )
        object.__setattr__(self, "dict_metadata", frozen_metadata)
        # Normal return after replacing metadata with an immutable mapping copy.
        return None


def _normalize_observability_identifier(value: str | None) -> str | None:
    """
    Trims and validates one optional observability identifier value.

    Args:
        value: Raw optional identifier value supplied by application or legacy config.

    Returns:
        Trimmed identifier string, or None when the value is missing or blank after trimming.
    """
    if value is None:
        # Early return because no identifier value was provided.
        return None
    normalized_value: str = value.strip()
    if normalized_value == "":
        # Early return because blank identifiers are not meaningful for correlation.
        return None
    # Normal return with the trimmed identifier value.
    return normalized_value


def set_observability_context(
    *,
    caller_id: str | None = None,
    session_id: str | None = None,
    workflow_id: str | None = None,
) -> Token[ObservabilityContextModel]:
    """
    Stores request-scoped observability correlation identifiers using `contextvars`.

    Args:
        caller_id: Optional stable, non-sensitive caller identifier for log correlation.
        session_id: Optional session identifier for request correlation.
        workflow_id: Optional workflow identifier for request correlation.

    Returns:
        ContextVar token that can be passed to `reset_observability_context`.
    """
    observability_context: ObservabilityContextModel = ObservabilityContextModel(
        caller_id=_normalize_observability_identifier(caller_id),
        session_id=_normalize_observability_identifier(session_id),
        workflow_id=_normalize_observability_identifier(workflow_id),
    )
    context_token: Token[ObservabilityContextModel] = OBSERVABILITY_CONTEXT.set(
        observability_context
    )
    # Normal return with the token required to restore prior request-scoped context.
    return context_token


def get_observability_context() -> ObservabilityContextModel:
    """
    Returns the current request-scoped observability correlation context.

    Args:
        None

    Returns:
        ObservabilityContextModel describing the active request-scoped caller, session,
        and workflow identifiers, or the shared empty default context when unset.
    """
    observability_context: ObservabilityContextModel = OBSERVABILITY_CONTEXT.get()
    # Normal return with the active request-scoped observability context.
    return observability_context


def reset_observability_context(
    context_token: Token[ObservabilityContextModel],
) -> None:
    """
    Restores the previous request-scoped observability context using a saved token.

    Args:
        context_token: Token returned by `set_observability_context`.

    Returns:
        None after the prior request-scoped observability context has been restored.
    """
    OBSERVABILITY_CONTEXT.reset(context_token)
    # Normal return after request-scoped observability context restoration.
    return None


def resolve_originating_caller(
    legacy_caller_id: str | None = None,
) -> tuple[str | None, str]:
    """
    Resolves the effective originating caller identifier using request-scoped context first.

    Args:
        legacy_caller_id: Optional explicit legacy caller hint supplied by configuration.

    Returns:
        Tuple containing the resolved caller identifier and its source label.
    """
    observability_context: ObservabilityContextModel = get_observability_context()
    normalized_caller_id: str | None = _normalize_observability_identifier(
        observability_context.caller_id
    )
    if normalized_caller_id is not None:
        # Normal return with the explicit request-scoped caller identifier.
        return normalized_caller_id, ORIGINATING_CALLER_ID_SOURCE_APPLICATION_CONTEXT
    normalized_legacy_caller_id: str | None = _normalize_observability_identifier(
        legacy_caller_id
    )
    if normalized_legacy_caller_id is not None:
        # Normal return with the explicit legacy caller identifier.
        return normalized_legacy_caller_id, ORIGINATING_CALLER_ID_SOURCE_LEGACY_SETTING
    # Normal return because no caller identifier was explicitly supplied.
    return None, ORIGINATING_CALLER_ID_SOURCE_NONE


def execute_observed_call(
    *,
    observability_middleware: AiApiObservabilityMiddleware,
    callable_build_call_context: Callable[[], AiApiCallContextModel],
    callable_execute: Callable[[], ProviderCallReturnType],
    callable_build_result_summary: Callable[
        [ProviderCallReturnType, float], AiApiCallResultSummaryModel
    ],
) -> ProviderCallReturnType:
    """
    Wraps one provider call with shared observability lifecycle emission.

    Args:
        observability_middleware: Effective observability middleware implementation for the call.
        callable_build_call_context: Zero-argument callable that builds the shared call context
            only when observability is enabled.
        callable_execute: Zero-argument callable that performs the provider call.
        callable_build_result_summary: Callable that summarizes provider output using the
            provider return value and elapsed milliseconds.

    Returns:
        Original provider return value from `callable_execute`.

    Raises:
        Propagates the original provider exception unchanged when `callable_execute` fails.
    """
    if not observability_middleware.bool_enabled:
        provider_result_without_observability: ProviderCallReturnType = (
            callable_execute()
        )
        # Normal return because observability is disabled and the provider result is unchanged.
        return provider_result_without_observability

    try:
        call_context: AiApiCallContextModel = callable_build_call_context()
    except Exception as exception:
        _LOGGER.warning(
            OBSERVABILITY_CONTEXT_HOOK_FAILURE_LOG_MESSAGE,
            "build_call_context",
            exception.__class__.__name__,
        )
        provider_result_without_call_context: ProviderCallReturnType = (
            callable_execute()
        )
        # Normal return because call-context construction failed and observability must fail open.
        return provider_result_without_call_context
    _safe_emit_observability_hook(
        stage="before_call",
        callable_emit=lambda: observability_middleware.before_call(
            call_context=call_context.with_direction(OBSERVABILITY_DIRECTION_INPUT)
        ),
    )

    float_started_at: float = time.perf_counter()
    try:
        provider_result: ProviderCallReturnType = callable_execute()
    except Exception as exception:
        caught_exception: Exception = exception
        provider_elapsed_ms: float = (time.perf_counter() - float_started_at) * 1000.0
        _safe_emit_observability_hook(
            stage="on_error",
            callable_emit=lambda: observability_middleware.on_error(
                call_context=call_context.with_direction(OBSERVABILITY_DIRECTION_ERROR),
                exception=caught_exception,
                float_elapsed_ms=provider_elapsed_ms,
            ),
        )
        # Early return via re-raise because provider exception behavior must remain unchanged.
        raise

    provider_elapsed_ms = (time.perf_counter() - float_started_at) * 1000.0
    try:
        call_result_summary: AiApiCallResultSummaryModel = (
            callable_build_result_summary(
                provider_result,
                provider_elapsed_ms,
            )
        )
    except Exception as exception:
        _LOGGER.warning(
            OBSERVABILITY_CONTEXT_HOOK_FAILURE_LOG_MESSAGE,
            "build_result_summary",
            exception.__class__.__name__,
        )
        # Normal return because result-summary construction failed and observability must fail open.
        return provider_result
    _safe_emit_observability_hook(
        stage="after_call",
        callable_emit=lambda: observability_middleware.after_call(
            call_context=call_context.with_direction(OBSERVABILITY_DIRECTION_OUTPUT),
            call_result_summary=call_result_summary,
        ),
    )
    # Normal return with the original provider result after observability emission.
    return provider_result


def _safe_emit_observability_hook(
    *,
    stage: str,
    callable_emit: Callable[[], None],
) -> None:
    """
    Executes one observability hook while preserving fail-open provider behavior.

    Args:
        stage: Lifecycle stage label used for local warning logs on hook failures.
        callable_emit: Zero-argument callable that emits one observability hook.

    Returns:
        None after the hook completes or a local warning is recorded.
    """
    try:
        callable_emit()
    except Exception as exception:
        _LOGGER.warning(
            OBSERVABILITY_CONTEXT_HOOK_FAILURE_LOG_MESSAGE,
            stage,
            exception.__class__.__name__,
        )
    # Normal return because observability hook failures must not escape into provider code.
    return None
