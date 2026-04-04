from __future__ import annotations

import logging
from datetime import datetime, timezone
from types import MappingProxyType

import pytest

from ai_api_unified.ai_base import AIBase
from ai_api_unified.middleware.middleware import METRICS_LOGGER_NAME
from ai_api_unified.middleware.middleware_config import (
    MiddlewareConfig,
    ObservabilitySettingsModel,
)
from ai_api_unified.middleware.observability import (
    OBSERVABILITY_EVENT_ERROR,
    OBSERVABILITY_EVENT_INPUT,
    OBSERVABILITY_EVENT_OUTPUT,
    OBSERVABILITY_LOGGER_NAME,
    AiApiObservabilityMiddleware,
    LoggerBackedObservabilityMiddleware,
    get_observability_middleware,
)
from ai_api_unified.middleware.observability_runtime import (
    AiApiCallContextModel,
    AiApiCallResultSummaryModel,
    OBSERVABILITY_CONTEXT_HOOK_FAILURE_LOG_MESSAGE,
    OBSERVABILITY_DIRECTION_ERROR,
    OBSERVABILITY_DIRECTION_INPUT,
    OBSERVABILITY_DIRECTION_OUTPUT,
    ORIGINATING_CALLER_ID_SOURCE_APPLICATION_CONTEXT,
    ORIGINATING_CALLER_ID_SOURCE_LEGACY_SETTING,
    ObservabilityContextModel,
    execute_observed_call,
    get_observability_context,
    reset_observability_context,
    resolve_originating_caller,
    set_observability_context,
)
from ai_api_unified.voice.ai_voice_base import (
    AIVoiceBase,
    AIVoiceModelBase,
    AIVoiceSelectionBase,
)
from ai_api_unified.voice.audio_models import AudioFormat

TEST_CALL_ID: str = "call-123"
TEST_CALLER_ID: str = "tenant-123"
TEST_LEGACY_CALLER_ID: str = "legacy-tenant"
TEST_SESSION_ID: str = "session-123"
TEST_WORKFLOW_ID: str = "workflow-123"
TEST_CAPABILITY_COMPLETIONS: str = "completions"
TEST_OPERATION_SEND_PROMPT: str = "send_prompt"
TEST_OPERATION_TEXT_TO_VOICE: str = "text_to_voice"
TEST_PROVIDER_VENDOR_OPENAI: str = "openai"
TEST_PROVIDER_ENGINE_OPENAI: str = "openai"
TEST_MODEL_NAME: str = "gpt-test"
TEST_MODEL_NAME_VOICE: str = "voice-test"
TEST_PROVIDER_RESULT: str = "provider result"
TEST_PROVIDER_AUDIO_BYTES: bytes = b"voice-bytes"
TEST_EXCEPTION_MESSAGE: str = "boom"
TEST_INPUT_PROMPT_CHAR_COUNT: int = 24
TEST_OUTPUT_CHAR_COUNT: int = len(TEST_PROVIDER_RESULT)
TEST_AUDIO_BYTE_COUNT: int = len(TEST_PROVIDER_AUDIO_BYTES)
TEST_PROVIDER_ELAPSED_MS: float = 12.5
TEST_CONTEXT_WARNING_LOGGER: str = "ai_api_unified.middleware.observability_runtime"
TEST_COLLIDING_CALL_ID: str = "shadow-call-id"
TEST_COLLIDING_PROVIDER_ELAPSED_MS: float = 999.0


class RecordingObservabilityMiddleware(AiApiObservabilityMiddleware):
    """
    Test double that records lifecycle events emitted by shared observability wrappers.

    Args:
        None

    Returns:
        Stateful middleware test double used for shared runtime assertions.
    """

    def __init__(self) -> None:
        """
        Initializes empty event-collection lists for each lifecycle direction.

        Args:
            None

        Returns:
            None after the middleware test double is ready to record events.
        """
        self.list_call_contexts_before: list[AiApiCallContextModel] = []
        self.list_tuple_output_events: list[
            tuple[AiApiCallContextModel, AiApiCallResultSummaryModel]
        ] = []
        self.list_tuple_error_events: list[
            tuple[AiApiCallContextModel, Exception, float]
        ] = []

    @property
    def bool_enabled(self) -> bool:
        """
        Indicates that the recording middleware is always enabled.

        Args:
            None

        Returns:
            True because this test double should exercise wrapper emission paths.
        """
        # Normal return because the recording middleware is always enabled for tests.
        return True

    def before_call(self, call_context: AiApiCallContextModel) -> None:
        """
        Records one input-side lifecycle event.

        Args:
            call_context: Immutable input event metadata emitted by the wrapper.

        Returns:
            None after storing the input event payload.
        """
        self.list_call_contexts_before.append(call_context)
        # Normal return after recording the input-side event.
        return None

    def after_call(
        self,
        call_context: AiApiCallContextModel,
        call_result_summary: AiApiCallResultSummaryModel,
    ) -> None:
        """
        Records one output-side lifecycle event.

        Args:
            call_context: Immutable output event metadata emitted by the wrapper.
            call_result_summary: Scalar output metadata emitted by the wrapper.

        Returns:
            None after storing the output event payload.
        """
        self.list_tuple_output_events.append((call_context, call_result_summary))
        # Normal return after recording the output-side event.
        return None

    def on_error(
        self,
        call_context: AiApiCallContextModel,
        exception: Exception,
        float_elapsed_ms: float,
    ) -> None:
        """
        Records one error-side lifecycle event.

        Args:
            call_context: Immutable error event metadata emitted by the wrapper.
            exception: Provider exception emitted by the wrapper.
            float_elapsed_ms: Elapsed provider time measured by the wrapper.

        Returns:
            None after storing the error event payload.
        """
        self.list_tuple_error_events.append((call_context, exception, float_elapsed_ms))
        # Normal return after recording the error-side event.
        return None


class FailingObservabilityMiddleware(AiApiObservabilityMiddleware):
    """
    Test double that raises from every lifecycle hook to verify fail-open behavior.

    Args:
        None

    Returns:
        Always-failing middleware test double used for fail-open assertions.
    """

    @property
    def bool_enabled(self) -> bool:
        """
        Indicates that the failing middleware is enabled.

        Args:
            None

        Returns:
            True because the wrapper must attempt hook execution before failing open.
        """
        # Normal return because the failing middleware remains enabled for tests.
        return True

    def before_call(self, call_context: AiApiCallContextModel) -> None:
        """
        Raises to simulate a broken input-side observability hook.

        Args:
            call_context: Immutable input event metadata emitted by the wrapper.

        Returns:
            None because this method always raises.
        """
        raise RuntimeError(TEST_EXCEPTION_MESSAGE)

    def after_call(
        self,
        call_context: AiApiCallContextModel,
        call_result_summary: AiApiCallResultSummaryModel,
    ) -> None:
        """
        Raises to simulate a broken output-side observability hook.

        Args:
            call_context: Immutable output event metadata emitted by the wrapper.
            call_result_summary: Scalar output metadata emitted by the wrapper.

        Returns:
            None because this method always raises.
        """
        raise RuntimeError(TEST_EXCEPTION_MESSAGE)

    def on_error(
        self,
        call_context: AiApiCallContextModel,
        exception: Exception,
        float_elapsed_ms: float,
    ) -> None:
        """
        Raises to simulate a broken error-side observability hook.

        Args:
            call_context: Immutable error event metadata emitted by the wrapper.
            exception: Provider exception emitted by the wrapper.
            float_elapsed_ms: Elapsed provider time measured by the wrapper.

        Returns:
            None because this method always raises.
        """
        raise RuntimeError(TEST_EXCEPTION_MESSAGE)


class FakeObservedClient(AIBase):
    """
    Minimal concrete AI base client used to exercise shared observability helpers.

    Args:
        model: Optional model identifier exposed by the base client.

    Returns:
        Concrete AIBase test double suitable for shared wrapper tests.
    """

    @property
    def list_model_names(self) -> list[str]:
        """
        Returns the single test model exposed by this client.

        Args:
            None

        Returns:
            One-item model list used only for abstract base compliance in tests.
        """
        # Normal return with the test client model list.
        return [TEST_MODEL_NAME]


class FakeVoiceClient(AIVoiceBase):
    """
    Minimal concrete voice client used to exercise shared observability helpers.

    Args:
        None

    Returns:
        Concrete AIVoiceBase test double suitable for shared wrapper tests.
    """

    def text_to_voice(
        self,
        text_to_convert: str,
        voice: AIVoiceSelectionBase | None = None,
        audio_format: AudioFormat | None = None,
        speaking_rate: float = 1.0,
        use_ssml: bool = False,
    ) -> bytes:
        """
        Returns a fixed byte payload for abstract base compliance in tests.

        Args:
            text_to_convert: Input text value ignored by this test double.
            voice: Optional voice selection ignored by this test double.
            audio_format: Optional audio format ignored by this test double.
            speaking_rate: Speaking-rate value ignored by this test double.
            use_ssml: SSML flag ignored by this test double.

        Returns:
            Fixed byte payload used only for abstract base compliance in tests.
        """
        # Normal return with a fixed byte payload for test compliance.
        return TEST_PROVIDER_AUDIO_BYTES

    def stream_audio(
        self,
        text: str,
        voice: AIVoiceSelectionBase | None = None,
    ) -> bytes:
        """
        Returns a fixed byte payload for abstract base compliance in tests.

        Args:
            text: Input text value ignored by this test double.
            voice: Optional voice selection ignored by this test double.

        Returns:
            Fixed byte payload used only for abstract base compliance in tests.
        """
        # Normal return with a fixed byte payload for test compliance.
        return TEST_PROVIDER_AUDIO_BYTES

    def speech_to_text(
        self,
        audio_bytes: bytes,
        language: str | None = None,
    ) -> str:
        """
        Returns a fixed transcript string for abstract base compliance in tests.

        Args:
            audio_bytes: Input audio bytes ignored by this test double.
            language: Optional language hint ignored by this test double.

        Returns:
            Fixed transcript string used only for abstract base compliance in tests.
        """
        # Normal return with a fixed transcript string for test compliance.
        return TEST_PROVIDER_RESULT


def _build_test_call_context(direction: str) -> AiApiCallContextModel:
    """
    Builds a deterministic call-context object for logger-backed middleware tests.

    Args:
        direction: Lifecycle direction assigned to the returned call-context object.

    Returns:
        Immutable call-context object with stable scalar metadata for assertions.
    """
    test_call_context: AiApiCallContextModel = AiApiCallContextModel(
        call_id=TEST_CALL_ID,
        event_time_utc=datetime(2026, 3, 19, 12, 0, tzinfo=timezone.utc),
        capability=TEST_CAPABILITY_COMPLETIONS,
        operation=TEST_OPERATION_SEND_PROMPT,
        provider_vendor=TEST_PROVIDER_VENDOR_OPENAI,
        provider_engine=TEST_PROVIDER_ENGINE_OPENAI,
        model_name=TEST_MODEL_NAME,
        model_version=TEST_MODEL_NAME,
        direction=direction,
        originating_caller_id=TEST_CALLER_ID,
        originating_caller_id_source=ORIGINATING_CALLER_ID_SOURCE_APPLICATION_CONTEXT,
        dict_metadata={"prompt_char_count": TEST_INPUT_PROMPT_CHAR_COUNT},
    )
    # Normal return with deterministic call-context metadata for tests.
    return test_call_context


def _build_test_result_summary() -> AiApiCallResultSummaryModel:
    """
    Builds a deterministic result-summary object for logger-backed middleware tests.

    Args:
        None

    Returns:
        Immutable result-summary object with stable scalar metadata for assertions.
    """
    test_result_summary: AiApiCallResultSummaryModel = AiApiCallResultSummaryModel(
        provider_elapsed_ms=TEST_PROVIDER_ELAPSED_MS,
        output_token_count=17,
        output_token_count_source="provider",
        provider_prompt_tokens=11,
        provider_completion_tokens=17,
        provider_total_tokens=28,
        finish_reason="stop",
        dict_metadata={"output_char_count": TEST_OUTPUT_CHAR_COUNT},
    )
    # Normal return with deterministic output metadata for tests.
    return test_result_summary


def test_observability_context_api_restores_previous_context() -> None:
    """
    Ensures nested request-scoped observability context values restore cleanly.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    outer_token = set_observability_context(
        caller_id=TEST_CALLER_ID,
        session_id=TEST_SESSION_ID,
    )
    inner_token = set_observability_context(
        caller_id=f"{TEST_CALLER_ID}-inner",
        workflow_id=TEST_WORKFLOW_ID,
    )
    try:
        active_context: ObservabilityContextModel = get_observability_context()
        assert active_context.caller_id == f"{TEST_CALLER_ID}-inner"
        assert active_context.session_id is None
        assert active_context.workflow_id == TEST_WORKFLOW_ID
    finally:
        reset_observability_context(inner_token)
        restored_context: ObservabilityContextModel = get_observability_context()
        assert restored_context.caller_id == TEST_CALLER_ID
        assert restored_context.session_id == TEST_SESSION_ID
        assert restored_context.workflow_id is None
        reset_observability_context(outer_token)

    default_context: ObservabilityContextModel = get_observability_context()
    assert default_context == ObservabilityContextModel()


def test_resolve_originating_caller_prefers_context_then_legacy() -> None:
    """
    Ensures request-scoped caller correlation overrides legacy caller settings.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    context_token = set_observability_context(caller_id=TEST_CALLER_ID)
    try:
        resolved_caller_id: str | None
        resolved_source: str
        resolved_caller_id, resolved_source = resolve_originating_caller(
            legacy_caller_id=TEST_LEGACY_CALLER_ID
        )
        assert resolved_caller_id == TEST_CALLER_ID
        assert resolved_source == ORIGINATING_CALLER_ID_SOURCE_APPLICATION_CONTEXT
    finally:
        reset_observability_context(context_token)

    resolved_caller_id, resolved_source = resolve_originating_caller(
        legacy_caller_id=TEST_LEGACY_CALLER_ID
    )
    assert resolved_caller_id == TEST_LEGACY_CALLER_ID
    assert resolved_source == ORIGINATING_CALLER_ID_SOURCE_LEGACY_SETTING


def test_get_observability_middleware_returns_logger_backed_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensures enabled observability settings resolve to the logger-backed implementation.

    Args:
        monkeypatch: Pytest fixture used to replace config lookup behavior.

    Returns:
        None for normal test completion.
    """

    def _get_enabled_settings(
        self: MiddlewareConfig,
    ) -> ObservabilitySettingsModel | None:
        """
        Returns enabled observability settings for middleware factory tests.

        Args:
            self: MiddlewareConfig instance ignored by this test helper.

        Returns:
            Enabled observability settings used to drive the logger-backed path.
        """
        # Normal return with enabled observability settings for the factory test.
        return ObservabilitySettingsModel()

    monkeypatch.setattr(
        MiddlewareConfig,
        "get_observability_settings",
        _get_enabled_settings,
    )

    middleware: AiApiObservabilityMiddleware = get_observability_middleware()

    assert isinstance(middleware, LoggerBackedObservabilityMiddleware)
    assert middleware.bool_enabled is True


def test_ai_base_wrapper_emits_input_and_output_with_shared_call_id() -> None:
    """
    Ensures the base wrapper emits input and output events around one provider call.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    test_client: FakeObservedClient = FakeObservedClient(model=TEST_MODEL_NAME)
    recording_middleware: RecordingObservabilityMiddleware = (
        RecordingObservabilityMiddleware()
    )
    test_client._observability_middleware = recording_middleware
    context_token = set_observability_context(
        caller_id=TEST_CALLER_ID,
        session_id=TEST_SESSION_ID,
        workflow_id=TEST_WORKFLOW_ID,
    )
    try:
        provider_result: str = test_client._execute_provider_call_with_observability(
            capability=TEST_CAPABILITY_COMPLETIONS,
            operation=TEST_OPERATION_SEND_PROMPT,
            dict_input_metadata={"prompt_char_count": TEST_INPUT_PROMPT_CHAR_COUNT},
            callable_execute=lambda: TEST_PROVIDER_RESULT,
            callable_build_result_summary=lambda result, elapsed_ms: AiApiCallResultSummaryModel(
                provider_elapsed_ms=elapsed_ms,
                dict_metadata={"output_char_count": len(result)},
            ),
        )
    finally:
        reset_observability_context(context_token)

    before_context: AiApiCallContextModel = (
        recording_middleware.list_call_contexts_before[0]
    )
    after_context, result_summary = recording_middleware.list_tuple_output_events[0]

    assert provider_result == TEST_PROVIDER_RESULT
    assert len(recording_middleware.list_call_contexts_before) == 1
    assert len(recording_middleware.list_tuple_output_events) == 1
    assert len(recording_middleware.list_tuple_error_events) == 0
    assert before_context.call_id == after_context.call_id
    assert before_context.direction == OBSERVABILITY_DIRECTION_INPUT
    assert after_context.direction == OBSERVABILITY_DIRECTION_OUTPUT
    assert before_context.originating_caller_id == TEST_CALLER_ID
    assert before_context.dict_metadata["session_id"] == TEST_SESSION_ID
    assert before_context.dict_metadata["workflow_id"] == TEST_WORKFLOW_ID
    assert result_summary.dict_metadata["output_char_count"] == TEST_OUTPUT_CHAR_COUNT
    assert result_summary.provider_elapsed_ms >= 0.0


def test_ai_base_wrapper_reraises_original_exception_and_emits_error() -> None:
    """
    Ensures wrapper error emission preserves the original provider exception behavior.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    test_client: FakeObservedClient = FakeObservedClient(model=TEST_MODEL_NAME)
    recording_middleware: RecordingObservabilityMiddleware = (
        RecordingObservabilityMiddleware()
    )
    test_client._observability_middleware = recording_middleware

    with pytest.raises(ValueError, match=TEST_EXCEPTION_MESSAGE):
        test_client._execute_provider_call_with_observability(
            capability=TEST_CAPABILITY_COMPLETIONS,
            operation=TEST_OPERATION_SEND_PROMPT,
            dict_input_metadata={"prompt_char_count": TEST_INPUT_PROMPT_CHAR_COUNT},
            callable_execute=lambda: (_ for _ in ()).throw(
                ValueError(TEST_EXCEPTION_MESSAGE)
            ),
            callable_build_result_summary=lambda result, elapsed_ms: AiApiCallResultSummaryModel(
                provider_elapsed_ms=elapsed_ms
            ),
        )

    before_context: AiApiCallContextModel = (
        recording_middleware.list_call_contexts_before[0]
    )
    error_context, exception, float_elapsed_ms = (
        recording_middleware.list_tuple_error_events[0]
    )

    assert len(recording_middleware.list_call_contexts_before) == 1
    assert len(recording_middleware.list_tuple_output_events) == 0
    assert len(recording_middleware.list_tuple_error_events) == 1
    assert before_context.call_id == error_context.call_id
    assert error_context.direction == OBSERVABILITY_DIRECTION_ERROR
    assert isinstance(exception, ValueError)
    assert float_elapsed_ms >= 0.0


def test_execute_observed_call_fails_open_when_observability_hooks_raise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Ensures broken lifecycle hooks do not change provider success behavior.

    Args:
        caplog: Pytest fixture used to capture warning logs for assertions.

    Returns:
        None for normal test completion.
    """
    caplog.set_level(logging.WARNING, logger=TEST_CONTEXT_WARNING_LOGGER)

    provider_result: str = execute_observed_call(
        observability_middleware=FailingObservabilityMiddleware(),
        callable_build_call_context=lambda: _build_test_call_context(
            OBSERVABILITY_DIRECTION_INPUT
        ),
        callable_execute=lambda: TEST_PROVIDER_RESULT,
        callable_build_result_summary=lambda result, elapsed_ms: AiApiCallResultSummaryModel(
            provider_elapsed_ms=elapsed_ms
        ),
    )

    assert provider_result == TEST_PROVIDER_RESULT
    assert any(
        log_record.message
        == OBSERVABILITY_CONTEXT_HOOK_FAILURE_LOG_MESSAGE
        % ("before_call", "RuntimeError")
        for log_record in caplog.records
    )
    assert any(
        log_record.message
        == OBSERVABILITY_CONTEXT_HOOK_FAILURE_LOG_MESSAGE
        % ("after_call", "RuntimeError")
        for log_record in caplog.records
    )


def test_execute_observed_call_fails_open_when_call_context_builder_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Ensures broken call-context construction does not change provider success behavior.

    Args:
        caplog: Pytest fixture used to capture warning logs for assertions.

    Returns:
        None for normal test completion.
    """
    caplog.set_level(logging.WARNING, logger=TEST_CONTEXT_WARNING_LOGGER)

    provider_result: str = execute_observed_call(
        observability_middleware=RecordingObservabilityMiddleware(),
        callable_build_call_context=lambda: (_ for _ in ()).throw(
            RuntimeError(TEST_EXCEPTION_MESSAGE)
        ),
        callable_execute=lambda: TEST_PROVIDER_RESULT,
        callable_build_result_summary=lambda result, elapsed_ms: AiApiCallResultSummaryModel(
            provider_elapsed_ms=elapsed_ms,
            dict_metadata={"output_char_count": len(result)},
        ),
    )

    assert provider_result == TEST_PROVIDER_RESULT
    assert any(
        log_record.message
        == OBSERVABILITY_CONTEXT_HOOK_FAILURE_LOG_MESSAGE
        % ("build_call_context", "RuntimeError")
        for log_record in caplog.records
    )


def test_execute_observed_call_fails_open_when_result_summary_builder_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Ensures broken result-summary construction does not change provider success behavior.

    Args:
        caplog: Pytest fixture used to capture warning logs for assertions.

    Returns:
        None for normal test completion.
    """
    caplog.set_level(logging.WARNING, logger=TEST_CONTEXT_WARNING_LOGGER)

    provider_result: str = execute_observed_call(
        observability_middleware=RecordingObservabilityMiddleware(),
        callable_build_call_context=lambda: _build_test_call_context(
            OBSERVABILITY_DIRECTION_INPUT
        ),
        callable_execute=lambda: TEST_PROVIDER_RESULT,
        callable_build_result_summary=lambda result, elapsed_ms: (_ for _ in ()).throw(
            RuntimeError(TEST_EXCEPTION_MESSAGE)
        ),
    )

    assert provider_result == TEST_PROVIDER_RESULT
    assert any(
        log_record.message
        == OBSERVABILITY_CONTEXT_HOOK_FAILURE_LOG_MESSAGE
        % ("build_result_summary", "RuntimeError")
        for log_record in caplog.records
    )


def test_voice_wrapper_emits_input_and_output_with_voice_metadata() -> None:
    """
    Ensures the voice wrapper emits metadata-only lifecycle events around one call.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    selected_model: AIVoiceModelBase = AIVoiceModelBase(name=TEST_MODEL_NAME_VOICE)
    test_voice_client: FakeVoiceClient = FakeVoiceClient(
        engine=TEST_PROVIDER_ENGINE_OPENAI,
        default_model_id=TEST_MODEL_NAME_VOICE,
        selected_model=selected_model,
    )
    recording_middleware: RecordingObservabilityMiddleware = (
        RecordingObservabilityMiddleware()
    )
    test_voice_client._observability_middleware = recording_middleware

    provider_audio_bytes: bytes = (
        test_voice_client._execute_voice_call_with_observability(
            operation=TEST_OPERATION_TEXT_TO_VOICE,
            dict_input_metadata={"input_char_count": TEST_INPUT_PROMPT_CHAR_COUNT},
            callable_execute=lambda: TEST_PROVIDER_AUDIO_BYTES,
            callable_build_result_summary=lambda audio_bytes, elapsed_ms: AiApiCallResultSummaryModel(
                provider_elapsed_ms=elapsed_ms,
                dict_metadata={"audio_byte_count": len(audio_bytes)},
            ),
        )
    )

    before_context: AiApiCallContextModel = (
        recording_middleware.list_call_contexts_before[0]
    )
    after_context, result_summary = recording_middleware.list_tuple_output_events[0]

    assert provider_audio_bytes == TEST_PROVIDER_AUDIO_BYTES
    assert before_context.capability == "tts"
    assert before_context.operation == TEST_OPERATION_TEXT_TO_VOICE
    assert before_context.model_name == TEST_MODEL_NAME_VOICE
    assert before_context.provider_engine == TEST_PROVIDER_ENGINE_OPENAI
    assert after_context.direction == OBSERVABILITY_DIRECTION_OUTPUT
    assert result_summary.dict_metadata["audio_byte_count"] == TEST_AUDIO_BYTE_COUNT


def test_observability_models_freeze_metadata_mappings() -> None:
    """
    Ensures call-context and result-summary metadata are copied and frozen.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    dict_context_metadata: dict[str, object] = {"prompt_char_count": 10}
    dict_result_metadata: dict[str, object] = {"output_char_count": 11}

    call_context: AiApiCallContextModel = AiApiCallContextModel(
        call_id=TEST_CALL_ID,
        event_time_utc=datetime(2026, 3, 19, 12, 0, tzinfo=timezone.utc),
        capability=TEST_CAPABILITY_COMPLETIONS,
        operation=TEST_OPERATION_SEND_PROMPT,
        provider_vendor=TEST_PROVIDER_VENDOR_OPENAI,
        provider_engine=TEST_PROVIDER_ENGINE_OPENAI,
        model_name=TEST_MODEL_NAME,
        model_version=TEST_MODEL_NAME,
        direction=OBSERVABILITY_DIRECTION_INPUT,
        dict_metadata=dict_context_metadata,
    )
    result_summary: AiApiCallResultSummaryModel = AiApiCallResultSummaryModel(
        provider_elapsed_ms=TEST_PROVIDER_ELAPSED_MS,
        dict_metadata=dict_result_metadata,
    )

    dict_context_metadata["prompt_char_count"] = 999
    dict_result_metadata["output_char_count"] = 999

    assert isinstance(call_context.dict_metadata, MappingProxyType)
    assert isinstance(result_summary.dict_metadata, MappingProxyType)
    assert call_context.dict_metadata["prompt_char_count"] == 10
    assert result_summary.dict_metadata["output_char_count"] == 11

    with pytest.raises(TypeError):
        call_context.dict_metadata["new_key"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        result_summary.dict_metadata["new_key"] = 1  # type: ignore[index]


def test_logger_backed_observability_middleware_logs_common_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Ensures logger-backed middleware emits metadata-only event and timing logs.

    Args:
        caplog: Pytest fixture used to capture event and metrics logs for assertions.

    Returns:
        None for normal test completion.
    """
    caplog.set_level(logging.INFO, logger=OBSERVABILITY_LOGGER_NAME)
    caplog.set_level(logging.INFO, logger=METRICS_LOGGER_NAME)
    middleware: LoggerBackedObservabilityMiddleware = (
        LoggerBackedObservabilityMiddleware(ObservabilitySettingsModel())
    )
    input_context: AiApiCallContextModel = _build_test_call_context(
        OBSERVABILITY_DIRECTION_INPUT
    )
    output_context: AiApiCallContextModel = _build_test_call_context(
        OBSERVABILITY_DIRECTION_OUTPUT
    )
    error_context: AiApiCallContextModel = _build_test_call_context(
        OBSERVABILITY_DIRECTION_ERROR
    )
    output_context = AiApiCallContextModel(
        call_id=output_context.call_id,
        event_time_utc=output_context.event_time_utc,
        capability=output_context.capability,
        operation=output_context.operation,
        provider_vendor=output_context.provider_vendor,
        provider_engine=output_context.provider_engine,
        model_name=output_context.model_name,
        model_version=output_context.model_version,
        direction=output_context.direction,
        originating_caller_id=output_context.originating_caller_id,
        originating_caller_id_source=output_context.originating_caller_id_source,
        dict_metadata={
            "prompt_char_count": TEST_INPUT_PROMPT_CHAR_COUNT,
            "call_id": TEST_COLLIDING_CALL_ID,
        },
    )
    result_summary: AiApiCallResultSummaryModel = AiApiCallResultSummaryModel(
        provider_elapsed_ms=TEST_PROVIDER_ELAPSED_MS,
        output_token_count=17,
        output_token_count_source="provider",
        provider_prompt_tokens=11,
        provider_completion_tokens=17,
        provider_total_tokens=28,
        finish_reason="stop",
        dict_metadata={
            "output_char_count": TEST_OUTPUT_CHAR_COUNT,
            "provider_elapsed_ms": TEST_COLLIDING_PROVIDER_ELAPSED_MS,
        },
    )

    middleware.before_call(input_context)
    middleware.after_call(output_context, result_summary)
    middleware.on_error(
        error_context,
        RuntimeError(TEST_EXCEPTION_MESSAGE),
        TEST_PROVIDER_ELAPSED_MS,
    )

    list_observability_records: list[logging.LogRecord] = [
        log_record
        for log_record in caplog.records
        if log_record.name == OBSERVABILITY_LOGGER_NAME
    ]
    list_metrics_records: list[logging.LogRecord] = [
        log_record
        for log_record in caplog.records
        if log_record.name == METRICS_LOGGER_NAME
    ]

    assert any(
        log_record.args[0] == OBSERVABILITY_EVENT_INPUT
        and log_record.args[1]["direction"] == OBSERVABILITY_DIRECTION_INPUT
        and log_record.args[1]["call_id"] == TEST_CALL_ID
        for log_record in list_observability_records
    )
    assert any(
        log_record.args[0] == OBSERVABILITY_EVENT_OUTPUT
        and log_record.args[1]["direction"] == OBSERVABILITY_DIRECTION_OUTPUT
        and log_record.args[1]["output_char_count"] == TEST_OUTPUT_CHAR_COUNT
        and log_record.args[1]["meta_call_id"] == TEST_COLLIDING_CALL_ID
        and log_record.args[1]["meta_provider_elapsed_ms"]
        == TEST_COLLIDING_PROVIDER_ELAPSED_MS
        and log_record.args[1]["provider_elapsed_ms"] == TEST_PROVIDER_ELAPSED_MS
        for log_record in list_observability_records
    )
    assert any(
        log_record.args[0] == OBSERVABILITY_EVENT_ERROR
        and log_record.args[1]["direction"] == OBSERVABILITY_DIRECTION_ERROR
        and log_record.args[1]["exception_type"] == "RuntimeError"
        for log_record in list_observability_records
    )
    assert any(
        "middleware_execution_timing middleware=observability direction=input"
        in log_record.message
        for log_record in list_metrics_records
    )


def test_logger_backed_observability_middleware_respects_optional_field_settings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Ensures optional observability settings suppress gated event fields at log emission time.

    Args:
        caplog: Pytest fixture used to capture emitted observability records.

    Returns:
        None for normal test completion.
    """
    caplog.set_level(logging.INFO, logger=OBSERVABILITY_LOGGER_NAME)
    middleware: LoggerBackedObservabilityMiddleware = (
        LoggerBackedObservabilityMiddleware(
            ObservabilitySettingsModel(
                include_media_details=False,
                include_provider_usage=False,
                include_audio_byte_count=False,
                include_image_byte_count=False,
                include_video_byte_count=False,
                token_count_mode="none",
            )
        )
    )
    input_context: AiApiCallContextModel = AiApiCallContextModel(
        call_id=TEST_CALL_ID,
        event_time_utc=datetime.now(timezone.utc),
        capability="tts",
        operation="text_to_voice",
        provider_vendor="openai",
        provider_engine="openai",
        model_name="tts-1",
        model_version=None,
        direction=OBSERVABILITY_DIRECTION_INPUT,
        dict_metadata={
            "has_media_attachments": True,
            "media_attachment_count": 2,
            "media_total_bytes": 99,
            "media_mime_types": ("image/png",),
            "input_token_count": 22,
            "input_token_count_source": "provider",
        },
    )
    output_context: AiApiCallContextModel = input_context.with_direction(
        OBSERVABILITY_DIRECTION_OUTPUT
    )
    result_summary: AiApiCallResultSummaryModel = AiApiCallResultSummaryModel(
        provider_elapsed_ms=TEST_PROVIDER_ELAPSED_MS,
        input_token_count=11,
        input_token_count_source="provider",
        output_token_count=17,
        output_token_count_source="provider",
        provider_prompt_tokens=11,
        provider_completion_tokens=17,
        provider_total_tokens=28,
        dict_metadata={
            "output_audio_byte_count": 1234,
            "total_output_bytes": 4321,
        },
    )

    middleware.before_call(input_context)
    middleware.after_call(output_context, result_summary)

    list_observability_records: list[logging.LogRecord] = [
        log_record
        for log_record in caplog.records
        if log_record.name == OBSERVABILITY_LOGGER_NAME
    ]
    dict_input_fields: dict[str, object] = list_observability_records[0].args[1]
    dict_output_fields: dict[str, object] = list_observability_records[1].args[1]

    assert "media_attachment_count" not in dict_input_fields
    assert "media_total_bytes" not in dict_input_fields
    assert "media_mime_types" not in dict_input_fields
    assert dict_input_fields["has_media_attachments"] is True
    assert "input_token_count" not in dict_output_fields
    assert "input_token_count_source" not in dict_output_fields
    assert "output_token_count" not in dict_output_fields
    assert "output_token_count_source" not in dict_output_fields
    assert "provider_prompt_tokens" not in dict_output_fields
    assert "provider_completion_tokens" not in dict_output_fields
    assert "provider_total_tokens" not in dict_output_fields
    assert "output_audio_byte_count" not in dict_output_fields
    assert dict_output_fields["total_output_bytes"] == 4321


def test_logger_backed_observability_middleware_hides_video_byte_count_when_disabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Ensures video byte-count suppression applies only to video-capability events.

    Args:
        caplog: Pytest fixture used to capture emitted observability records.

    Returns:
        None for normal test completion.
    """

    caplog.set_level(logging.INFO, logger=OBSERVABILITY_LOGGER_NAME)
    middleware: LoggerBackedObservabilityMiddleware = (
        LoggerBackedObservabilityMiddleware(
            ObservabilitySettingsModel(include_video_byte_count=False)
        )
    )
    input_context: AiApiCallContextModel = AiApiCallContextModel(
        call_id=TEST_CALL_ID,
        event_time_utc=datetime.now(timezone.utc),
        capability="videos",
        operation="generate_video",
        provider_vendor="google",
        provider_engine="google-gemini",
        model_name="veo-3.1-lite-generate-preview",
        model_version=None,
        direction=OBSERVABILITY_DIRECTION_INPUT,
        dict_metadata={},
    )
    output_context: AiApiCallContextModel = input_context.with_direction(
        OBSERVABILITY_DIRECTION_OUTPUT
    )
    result_summary: AiApiCallResultSummaryModel = AiApiCallResultSummaryModel(
        provider_elapsed_ms=TEST_PROVIDER_ELAPSED_MS,
        dict_metadata={"total_output_bytes": 4321},
    )

    middleware.before_call(input_context)
    middleware.after_call(output_context, result_summary)

    list_observability_records: list[logging.LogRecord] = [
        log_record
        for log_record in caplog.records
        if log_record.name == OBSERVABILITY_LOGGER_NAME
    ]
    dict_output_fields: dict[str, object] = list_observability_records[1].args[1]

    assert "total_output_bytes" not in dict_output_fields
