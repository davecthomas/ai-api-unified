"""
Lifecycle-style observability middleware contracts and implementations.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

from ai_api_unified.middleware.middleware import (
    METRICS_LOGGER_NAME,
    execute_middleware_with_timing,
    log_middleware_observability_events,
)
from ai_api_unified.middleware.middleware_config import (
    DIRECTION_INPUT_OUTPUT,
    DIRECTION_OUTPUT_ONLY,
    INPUT_ONLY,
    MiddlewareConfig,
    ObservabilitySettingsModel,
    TOKEN_COUNT_MODE_NONE,
)
from ai_api_unified.middleware.observability_runtime import (
    AiApiCallContextModel,
    AiApiCallResultSummaryModel,
    OBSERVABILITY_DIRECTION_ERROR,
    OBSERVABILITY_DIRECTION_INPUT,
    OBSERVABILITY_DIRECTION_OUTPUT,
    ORIGINATING_CALLER_ID_SOURCE_NONE,
)

_LOGGER: logging.Logger = logging.getLogger(__name__)
OBSERVABILITY_LOGGER_NAME: str = "ai_api_unified.middleware.observability"
OBSERVABILITY_EVENT_INPUT: str = "ai_api_call_input"
OBSERVABILITY_EVENT_OUTPUT: str = "ai_api_call_output"
OBSERVABILITY_EVENT_ERROR: str = "ai_api_call_error"
OBSERVABILITY_MIDDLEWARE_NAME: str = "observability"
OBSERVABILITY_LOG_FAILURE_WARNING_MESSAGE: str = (
    "observability_log_failed stage=%s error_type=%s"
)
OBSERVABILITY_EVENT_LOG_MESSAGE: str = "%s %s"
OBSERVABILITY_ERROR_LOG_LEVEL: int = logging.ERROR
OBSERVABILITY_MIDDLEWARE_TIMING_LOG_LEVEL: int = logging.INFO
OBSERVABILITY_METADATA_COLLISION_PREFIX: str = "meta_"
DICT_STR_LOG_LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
SET_STR_MEDIA_DETAIL_METADATA_KEYS: set[str] = {
    "media_attachment_count",
    "media_total_bytes",
    "media_mime_types",
}
SET_STR_IMAGE_BYTE_COUNT_METADATA_KEYS: set[str] = {"total_output_bytes"}
SET_STR_AUDIO_BYTE_COUNT_METADATA_KEYS: set[str] = {"output_audio_byte_count"}
SET_STR_PROVIDER_USAGE_EVENT_KEYS: set[str] = {
    "provider_prompt_tokens",
    "provider_completion_tokens",
    "provider_total_tokens",
}
SET_STR_TOKEN_COUNT_EVENT_KEYS: set[str] = {
    "input_token_count",
    "input_token_count_source",
    "output_token_count",
    "output_token_count_source",
}


class AiApiObservabilityMiddleware(ABC):
    """
    Abstract base class for lifecycle-style observability middleware components.

    This contract is intentionally separate from the text-transforming middleware
    interface. Implementations observe provider call lifecycle events without
    mutating request or response payloads.
    """

    @property
    @abstractmethod
    def bool_enabled(self) -> bool:
        """
        Indicates whether runtime observability emission is effectively enabled.

        Args:
            None

        Returns:
            True when the middleware should emit lifecycle observability events.
        """
        ...

    @abstractmethod
    def before_call(self, call_context: AiApiCallContextModel) -> None:
        """
        Observes input-side metadata immediately before a provider call executes.

        Args:
            call_context: Immutable provider-boundary call metadata for the input event.

        Returns:
            None after the middleware finishes its pre-call observation work.
        """
        ...

    @abstractmethod
    def after_call(
        self,
        call_context: AiApiCallContextModel,
        call_result_summary: AiApiCallResultSummaryModel,
    ) -> None:
        """
        Observes output-side metadata immediately after a provider call completes.

        Args:
            call_context: Immutable provider-boundary call metadata for the output event.
            call_result_summary: Scalar provider-result metadata derived from the call output.

        Returns:
            None after the middleware finishes its post-call observation work.
        """
        ...

    @abstractmethod
    def on_error(
        self,
        call_context: AiApiCallContextModel,
        exception: Exception,
        float_elapsed_ms: float,
    ) -> None:
        """
        Observes metadata when a provider call raises an exception.

        Args:
            call_context: Immutable provider-boundary call metadata for the error event.
            exception: Exception raised by the provider call path.
            float_elapsed_ms: Elapsed provider-call time in milliseconds before failure.

        Returns:
            None after the middleware finishes its error-side observation work.
        """
        ...


class NoOpObservabilityMiddleware(AiApiObservabilityMiddleware):
    """
    Disabled observability middleware implementation that intentionally does nothing.
    """

    def __init__(self) -> None:
        """
        Initializes the disabled observability middleware implementation.

        Args:
            None

        Returns:
            None after the no-op middleware is ready for lifecycle calls.
        """
        self._bool_enabled: bool = False

    @property
    def bool_enabled(self) -> bool:
        """
        Indicates that this implementation is always disabled.

        Args:
            None

        Returns:
            False because the implementation intentionally emits no observability events.
        """
        # Normal return because the no-op implementation is always disabled.
        return self._bool_enabled

    def before_call(self, call_context: AiApiCallContextModel) -> None:
        """
        Intentionally skips all pre-call observability work.

        Args:
            call_context: Immutable provider-boundary call metadata supplied by the caller.

        Returns:
            None because the disabled middleware performs no work.
        """
        # Normal return because disabled middleware intentionally does nothing.
        return None

    def after_call(
        self,
        call_context: AiApiCallContextModel,
        call_result_summary: AiApiCallResultSummaryModel,
    ) -> None:
        """
        Intentionally skips all post-call observability work.

        Args:
            call_context: Immutable provider-boundary call metadata supplied by the caller.
            call_result_summary: Scalar provider-result metadata supplied by the caller.

        Returns:
            None because the disabled middleware performs no work.
        """
        # Normal return because disabled middleware intentionally does nothing.
        return None

    def on_error(
        self,
        call_context: AiApiCallContextModel,
        exception: Exception,
        float_elapsed_ms: float,
    ) -> None:
        """
        Intentionally skips all error-side observability work.

        Args:
            call_context: Immutable provider-boundary call metadata supplied by the caller.
            exception: Exception raised by the provider call path.
            float_elapsed_ms: Elapsed provider-call time in milliseconds before failure.

        Returns:
            None because the disabled middleware performs no work.
        """
        # Normal return because disabled middleware intentionally does nothing.
        return None


class LoggerBackedObservabilityMiddleware(AiApiObservabilityMiddleware):
    """
    Logger-backed observability middleware that emits metadata-only lifecycle events.
    """

    def __init__(self, observability_settings: ObservabilitySettingsModel) -> None:
        """
        Initializes the logger-backed observability middleware implementation.

        Args:
            observability_settings: Typed observability middleware settings for runtime behavior.

        Returns:
            None after the logger-backed middleware is ready for lifecycle calls.
        """
        self.observability_settings: ObservabilitySettingsModel = observability_settings
        self._bool_enabled: bool = True
        self.event_logger: logging.Logger = logging.getLogger(OBSERVABILITY_LOGGER_NAME)
        self.metrics_logger: logging.Logger = logging.getLogger(METRICS_LOGGER_NAME)
        self.int_log_level: int = DICT_STR_LOG_LEVELS.get(
            observability_settings.log_level,
            logging.INFO,
        )

    @property
    def bool_enabled(self) -> bool:
        """
        Indicates that the logger-backed middleware is enabled.

        Args:
            None

        Returns:
            True because this implementation emits observability lifecycle events.
        """
        # Normal return because the logger-backed implementation is enabled.
        return self._bool_enabled

    def before_call(self, call_context: AiApiCallContextModel) -> None:
        """
        Emits a metadata-only input event when observability is enabled for the call.

        Args:
            call_context: Immutable provider-boundary call metadata for the input event.

        Returns:
            None after the metadata-only input event and timing metric have been handled.
        """
        if not self._should_emit_direction(
            call_context=call_context,
            direction=OBSERVABILITY_DIRECTION_INPUT,
        ):
            # Early return because this input event is not enabled by runtime settings.
            return None
        self._emit_timed_event(
            stage="before_call",
            direction=OBSERVABILITY_DIRECTION_INPUT,
            callable_emit=lambda: self._emit_input_event(call_context=call_context),
        )
        # Normal return after input-side observability handling.
        return None

    def after_call(
        self,
        call_context: AiApiCallContextModel,
        call_result_summary: AiApiCallResultSummaryModel,
    ) -> None:
        """
        Emits a metadata-only output event when observability is enabled for the call.

        Args:
            call_context: Immutable provider-boundary call metadata for the output event.
            call_result_summary: Scalar provider-result metadata derived from the call output.

        Returns:
            None after the metadata-only output event and timing metric have been handled.
        """
        if not self._should_emit_direction(
            call_context=call_context,
            direction=OBSERVABILITY_DIRECTION_OUTPUT,
        ):
            # Early return because this output event is not enabled by runtime settings.
            return None
        self._emit_timed_event(
            stage="after_call",
            direction=OBSERVABILITY_DIRECTION_OUTPUT,
            callable_emit=lambda: self._emit_output_event(
                call_context=call_context,
                call_result_summary=call_result_summary,
            ),
        )
        # Normal return after output-side observability handling.
        return None

    def on_error(
        self,
        call_context: AiApiCallContextModel,
        exception: Exception,
        float_elapsed_ms: float,
    ) -> None:
        """
        Emits a metadata-only error event when error events are enabled for the call.

        Args:
            call_context: Immutable provider-boundary call metadata for the error event.
            exception: Exception raised by the provider call path.
            float_elapsed_ms: Elapsed provider-call time in milliseconds before failure.

        Returns:
            None after the metadata-only error event and timing metric have been handled.
        """
        if not self._should_emit_error(call_context=call_context):
            # Early return because error events are disabled by runtime settings.
            return None
        self._emit_timed_event(
            stage="on_error",
            direction=OBSERVABILITY_DIRECTION_ERROR,
            callable_emit=lambda: self._emit_error_event(
                call_context=call_context,
                exception=exception,
                float_elapsed_ms=float_elapsed_ms,
            ),
        )
        # Normal return after error-side observability handling.
        return None

    def _emit_timed_event(
        self,
        *,
        stage: str,
        direction: str,
        callable_emit: Callable[[], None],
    ) -> None:
        """
        Emits one observability event while recording middleware execution timing.

        Args:
            stage: Lifecycle stage label used for fail-open warning logs.
            direction: Direction label included in shared middleware timing metrics.
            callable_emit: Zero-argument callable that emits one observability event.

        Returns:
            None after the event emission attempt and timing log handling complete.
        """
        try:
            timing_result = execute_middleware_with_timing(
                callable_execute=callable_emit
            )
            if self.metrics_logger.isEnabledFor(
                OBSERVABILITY_MIDDLEWARE_TIMING_LOG_LEVEL
            ):
                log_middleware_observability_events(
                    str_middleware_name=OBSERVABILITY_MIDDLEWARE_NAME,
                    str_direction=direction,
                    float_elapsed_ms=timing_result.float_elapsed_ms,
                    metrics_logger=self.metrics_logger,
                )
        except Exception as exception:
            _LOGGER.warning(
                OBSERVABILITY_LOG_FAILURE_WARNING_MESSAGE,
                stage,
                exception.__class__.__name__,
            )
        # Normal return because observability middleware must fail open.
        return None

    def _emit_input_event(self, call_context: AiApiCallContextModel) -> None:
        """
        Emits the input lifecycle event for one provider call.

        Args:
            call_context: Immutable provider-boundary call metadata for the input event.

        Returns:
            None after the event has been logged or intentionally skipped.
        """
        if not self.event_logger.isEnabledFor(self.int_log_level):
            # Early return because the configured event log level is disabled.
            return None
        dict_event_fields: dict[str, object] = self._build_shared_event_fields(
            call_context=call_context
        )
        self._apply_input_metadata_filters(dict_event_fields=dict_event_fields)
        self.event_logger.log(
            self.int_log_level,
            OBSERVABILITY_EVENT_LOG_MESSAGE,
            OBSERVABILITY_EVENT_INPUT,
            dict_event_fields,
        )
        # Normal return after input event logging.
        return None

    def _emit_output_event(
        self,
        *,
        call_context: AiApiCallContextModel,
        call_result_summary: AiApiCallResultSummaryModel,
    ) -> None:
        """
        Emits the output lifecycle event for one provider call.

        Args:
            call_context: Immutable provider-boundary call metadata for the output event.
            call_result_summary: Scalar provider-result metadata derived from the call output.

        Returns:
            None after the event has been logged or intentionally skipped.
        """
        if not self.event_logger.isEnabledFor(self.int_log_level):
            # Early return because the configured event log level is disabled.
            return None
        dict_event_fields: dict[str, object] = self._build_shared_event_fields(
            call_context=call_context
        )
        dict_event_fields["provider_elapsed_ms"] = round(
            call_result_summary.provider_elapsed_ms, 3
        )
        self._merge_result_summary_fields(
            dict_event_fields=dict_event_fields,
            call_result_summary=call_result_summary,
        )
        self._apply_output_metadata_filters(
            dict_event_fields=dict_event_fields,
            call_context=call_context,
        )
        self.event_logger.log(
            self.int_log_level,
            OBSERVABILITY_EVENT_LOG_MESSAGE,
            OBSERVABILITY_EVENT_OUTPUT,
            dict_event_fields,
        )
        # Normal return after output event logging.
        return None

    def _emit_error_event(
        self,
        *,
        call_context: AiApiCallContextModel,
        exception: Exception,
        float_elapsed_ms: float,
    ) -> None:
        """
        Emits the error lifecycle event for one failed provider call.

        Args:
            call_context: Immutable provider-boundary call metadata for the error event.
            exception: Exception raised by the provider call path.
            float_elapsed_ms: Elapsed provider-call time in milliseconds before failure.

        Returns:
            None after the event has been logged or intentionally skipped.
        """
        if not self.event_logger.isEnabledFor(OBSERVABILITY_ERROR_LOG_LEVEL):
            # Early return because the error event log level is disabled.
            return None
        dict_event_fields: dict[str, object] = self._build_shared_event_fields(
            call_context=call_context
        )
        dict_event_fields["provider_elapsed_ms"] = round(float_elapsed_ms, 3)
        dict_event_fields["exception_type"] = exception.__class__.__name__
        self.event_logger.log(
            OBSERVABILITY_ERROR_LOG_LEVEL,
            OBSERVABILITY_EVENT_LOG_MESSAGE,
            OBSERVABILITY_EVENT_ERROR,
            dict_event_fields,
        )
        # Normal return after error event logging.
        return None

    def _build_shared_event_fields(
        self,
        *,
        call_context: AiApiCallContextModel,
    ) -> dict[str, object]:
        """
        Builds the common scalar field payload emitted on all observability events.

        Args:
            call_context: Immutable provider-boundary call metadata for the current event.

        Returns:
            Ordered dictionary of stable event fields safe for metadata-only logs.
        """
        dict_event_fields: dict[str, object] = {
            "call_id": call_context.call_id,
            "direction": call_context.direction,
            "capability": call_context.capability,
            "operation": call_context.operation,
            "provider_vendor": call_context.provider_vendor,
            "provider_engine": call_context.provider_engine,
            "model_name": call_context.model_name,
            "model_version": call_context.model_version,
            "event_time_utc": call_context.event_time_utc.isoformat(),
        }
        if call_context.originating_caller_id is not None:
            dict_event_fields["originating_caller_id"] = (
                call_context.originating_caller_id
            )
            dict_event_fields["originating_caller_id_source"] = (
                call_context.originating_caller_id_source
            )
        elif (
            call_context.originating_caller_id_source
            != ORIGINATING_CALLER_ID_SOURCE_NONE
        ):
            dict_event_fields["originating_caller_id_source"] = (
                call_context.originating_caller_id_source
            )
        self._merge_safe_metadata_fields(
            dict_event_fields=dict_event_fields,
            dict_metadata=call_context.dict_metadata,
        )
        # Normal return with the shared event-field payload.
        return dict_event_fields

    def _merge_result_summary_fields(
        self,
        *,
        dict_event_fields: dict[str, object],
        call_result_summary: AiApiCallResultSummaryModel,
    ) -> None:
        """
        Appends result-summary scalar fields to the output-event payload.

        Args:
            dict_event_fields: Mutable event-field payload receiving output metadata.
            call_result_summary: Scalar provider-result metadata derived from the call output.

        Returns:
            None after the event payload has been updated in place.
        """
        if call_result_summary.input_token_count is not None:
            dict_event_fields["input_token_count"] = (
                call_result_summary.input_token_count
            )
        dict_event_fields["input_token_count_source"] = (
            call_result_summary.input_token_count_source
        )
        if call_result_summary.output_token_count is not None:
            dict_event_fields["output_token_count"] = (
                call_result_summary.output_token_count
            )
        dict_event_fields["output_token_count_source"] = (
            call_result_summary.output_token_count_source
        )
        if call_result_summary.provider_prompt_tokens is not None:
            dict_event_fields["provider_prompt_tokens"] = (
                call_result_summary.provider_prompt_tokens
            )
        if call_result_summary.provider_completion_tokens is not None:
            dict_event_fields["provider_completion_tokens"] = (
                call_result_summary.provider_completion_tokens
            )
        if call_result_summary.provider_total_tokens is not None:
            dict_event_fields["provider_total_tokens"] = (
                call_result_summary.provider_total_tokens
            )
        if call_result_summary.finish_reason is not None:
            dict_event_fields["finish_reason"] = call_result_summary.finish_reason
        self._merge_safe_metadata_fields(
            dict_event_fields=dict_event_fields,
            dict_metadata=call_result_summary.dict_metadata,
        )
        # Normal return after output payload augmentation.
        return None

    def _apply_input_metadata_filters(
        self,
        *,
        dict_event_fields: dict[str, object],
    ) -> None:
        """
        Applies runtime settings that suppress optional input-side observability fields.

        Args:
            dict_event_fields: Mutable input-event payload to filter in place.

        Returns:
            None after optional input-side fields have been filtered in place.
        """
        if not self.observability_settings.include_media_details:
            # Loop through media-detail keys and drop them when that setting is disabled.
            for str_metadata_key in SET_STR_MEDIA_DETAIL_METADATA_KEYS:
                dict_event_fields.pop(str_metadata_key, None)
        if self.observability_settings.token_count_mode == TOKEN_COUNT_MODE_NONE:
            # Loop through token-count keys and drop them when token counts are disabled.
            for str_metadata_key in SET_STR_TOKEN_COUNT_EVENT_KEYS:
                dict_event_fields.pop(str_metadata_key, None)
        # Normal return after input-side metadata filtering completes.
        return None

    def _apply_output_metadata_filters(
        self,
        *,
        dict_event_fields: dict[str, object],
        call_context: AiApiCallContextModel,
    ) -> None:
        """
        Applies runtime settings that suppress optional output-side observability fields.

        Args:
            dict_event_fields: Mutable output-event payload to filter in place.
            call_context: Immutable provider-boundary call metadata for the current event.

        Returns:
            None after optional output-side fields have been filtered in place.
        """
        if self.observability_settings.token_count_mode == TOKEN_COUNT_MODE_NONE:
            # Loop through token-count keys and drop them when token counts are disabled.
            for str_event_key in SET_STR_TOKEN_COUNT_EVENT_KEYS:
                dict_event_fields.pop(str_event_key, None)
            # Loop through provider-usage keys and drop them because they are token counts too.
            for str_event_key in SET_STR_PROVIDER_USAGE_EVENT_KEYS:
                dict_event_fields.pop(str_event_key, None)
        elif not self.observability_settings.include_provider_usage:
            # Loop through provider-usage keys and drop them when provider usage is disabled.
            for str_event_key in SET_STR_PROVIDER_USAGE_EVENT_KEYS:
                dict_event_fields.pop(str_event_key, None)

        if not self.observability_settings.include_image_byte_count:
            # Loop through image-byte keys and drop them when image byte counts are disabled.
            for str_event_key in SET_STR_IMAGE_BYTE_COUNT_METADATA_KEYS:
                dict_event_fields.pop(str_event_key, None)

        if not self.observability_settings.include_audio_byte_count:
            # Loop through audio-byte keys and drop them when audio byte counts are disabled.
            for str_event_key in SET_STR_AUDIO_BYTE_COUNT_METADATA_KEYS:
                dict_event_fields.pop(str_event_key, None)
        if not self.observability_settings.include_media_details:
            # Loop through media-detail keys and drop them when that setting is disabled.
            for str_metadata_key in SET_STR_MEDIA_DETAIL_METADATA_KEYS:
                dict_event_fields.pop(str_metadata_key, None)
        # Normal return after output-side metadata filtering completes.
        return None

    def _merge_safe_metadata_fields(
        self,
        *,
        dict_event_fields: dict[str, object],
        dict_metadata: Mapping[str, object],
    ) -> None:
        """
        Merges caller-supplied metadata without allowing collisions with reserved event fields.

        Args:
            dict_event_fields: Mutable event payload receiving metadata fields.
            dict_metadata: Caller-supplied metadata fields to merge into the payload.

        Returns:
            None after metadata fields have been merged into the payload in place.
        """
        set_reserved_keys: set[str] = set(dict_event_fields.keys())
        # Loop through caller-supplied metadata and prefix colliding keys to preserve stable event fields.
        for str_metadata_key, metadata_value in dict_metadata.items():
            if str_metadata_key in set_reserved_keys:
                prefixed_metadata_key: str = (
                    f"{OBSERVABILITY_METADATA_COLLISION_PREFIX}{str_metadata_key}"
                )
                dict_event_fields[prefixed_metadata_key] = metadata_value
                continue
            dict_event_fields[str_metadata_key] = metadata_value
        # Normal return after metadata merge-in-place completes.
        return None

    def _should_emit_direction(
        self,
        *,
        call_context: AiApiCallContextModel,
        direction: str,
    ) -> bool:
        """
        Determines whether an input or output event should be emitted for the call.

        Args:
            call_context: Immutable provider-boundary call metadata for the current event.
            direction: Event direction under consideration (`input` or `output`).

        Returns:
            True when the current settings allow the event for the given capability and direction.
        """
        if not self._is_capability_enabled(call_context=call_context):
            # Early return because this capability is disabled in observability settings.
            return False
        if direction == OBSERVABILITY_DIRECTION_INPUT:
            # Normal return for input direction enablement checks.
            return self.observability_settings.direction in (
                INPUT_ONLY,
                DIRECTION_INPUT_OUTPUT,
            )
        if direction == OBSERVABILITY_DIRECTION_OUTPUT:
            # Normal return for output direction enablement checks.
            return self.observability_settings.direction in (
                DIRECTION_OUTPUT_ONLY,
                DIRECTION_INPUT_OUTPUT,
            )
        # Early return because other directions are not handled by this helper.
        return False

    def _should_emit_error(self, *, call_context: AiApiCallContextModel) -> bool:
        """
        Determines whether an error event should be emitted for the call.

        Args:
            call_context: Immutable provider-boundary call metadata for the current event.

        Returns:
            True when error events are enabled and the capability is allowed.
        """
        if not self.observability_settings.emit_error_events:
            # Early return because error events are disabled in observability settings.
            return False
        bool_capability_enabled: bool = self._is_capability_enabled(
            call_context=call_context
        )
        # Normal return with capability-aware error-event enablement.
        return bool_capability_enabled

    def _is_capability_enabled(self, *, call_context: AiApiCallContextModel) -> bool:
        """
        Determines whether the call capability is enabled by the allow-list settings.

        Args:
            call_context: Immutable provider-boundary call metadata for the current event.

        Returns:
            True when the allow-list is empty or explicitly includes the current capability.
        """
        if not self.observability_settings.capabilities:
            # Early return because an empty allow-list enables all capabilities.
            return True
        bool_capability_enabled: bool = (
            call_context.capability in self.observability_settings.capabilities
        )
        # Normal return with the capability allow-list decision.
        return bool_capability_enabled


def get_observability_middleware() -> AiApiObservabilityMiddleware:
    """
    Returns the effective observability middleware implementation for current config.

    Args:
        None

    Returns:
        The disabled no-op middleware when observability is missing or disabled, otherwise
        the logger-backed observability middleware configured for runtime use.
    """
    middleware_config: MiddlewareConfig = MiddlewareConfig()
    observability_settings: ObservabilitySettingsModel | None = (
        middleware_config.get_observability_settings()
    )
    if observability_settings is None:
        # Early return because observability is not enabled in the active configuration.
        return NoOpObservabilityMiddleware()
    logger_backed_middleware: LoggerBackedObservabilityMiddleware = (
        LoggerBackedObservabilityMiddleware(observability_settings)
    )
    # Normal return with the enabled logger-backed observability middleware.
    return logger_backed_middleware
