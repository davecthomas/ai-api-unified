# ruff: noqa: E402

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("boto3")
pytest.importorskip("google.genai")
pytest.importorskip("openai")

from ai_api_unified.ai_completions_exceptions import (
    StructuredResponseTokenLimitError,
)
from ai_api_unified.ai_base import AIStructuredPrompt
from ai_api_unified.completions.ai_bedrock_completions import (
    AiBedrockCompletions,
)
from ai_api_unified.completions.ai_google_gemini_completions import (
    GoogleGeminiCompletions,
)
from ai_api_unified.completions.ai_openai_completions import (
    AiOpenAICompletions,
)
from ai_api_unified.middleware.observability import (
    AiApiObservabilityMiddleware,
)
from ai_api_unified.middleware.observability_runtime import (
    AiApiCallContextModel,
    AiApiCallResultSummaryModel,
)

TEST_COMPLETIONS_MODEL: str = "test-model"
TEST_PROMPT: str = "Say hello"
TEST_INPUT_SUFFIX: str = " [input-redacted]"
TEST_OUTPUT_SUFFIX: str = " [output-redacted]"
TEST_OPENAI_USER: str = "legacy-user"
TEST_SYSTEM_PROMPT: str = "You are a helpful assistant."
TEST_OPERATION_SEND_PROMPT: str = "send_prompt"
TEST_OPERATION_STRICT_SCHEMA_PROMPT: str = "strict_schema_prompt"
TEST_TOO_SMALL_MAX_RESPONSE_TOKENS: int = 321


class FakeStructuredResponse(AIStructuredPrompt):
    """
    Minimal structured response model used for completions observability tests.

    Args:
        answer: Structured text field returned by the provider.

    Returns:
        Pydantic response model used for strict-schema test assertions.
    """

    answer: str

    @staticmethod
    def get_prompt() -> str:
        """
        Returns a placeholder prompt string required by the AIStructuredPrompt interface.

        Args:
            None

        Returns:
            Empty string because this test model is used only for response validation.
        """
        # Normal return because the test response model does not generate prompts directly.
        return ""


class RecordingObservabilityMiddleware(AiApiObservabilityMiddleware):
    """
    Records lifecycle events so provider tests can assert ordering and metadata.

    Args:
        list_event_order: Mutable event-order log shared with the test.

    Returns:
        Enabled middleware test double that stores before, after, and error events.
    """

    def __init__(self, list_event_order: list[str]) -> None:
        """
        Stores shared order-tracking state and initializes empty event collections.

        Args:
            list_event_order: Mutable event log shared with the surrounding test.

        Returns:
            None after the middleware test double is ready to record events.
        """
        self.list_event_order: list[str] = list_event_order
        self.list_before_contexts: list[AiApiCallContextModel] = []
        self.list_after_events: list[
            tuple[AiApiCallContextModel, AiApiCallResultSummaryModel]
        ] = []
        self.list_error_events: list[tuple[AiApiCallContextModel, Exception, float]] = (
            []
        )

    @property
    def bool_enabled(self) -> bool:
        """
        Indicates that the recording middleware is enabled for all tests.

        Args:
            None

        Returns:
            True because tests should exercise observability hooks.
        """
        # Normal return because the recording middleware is always enabled in tests.
        return True

    def before_call(self, call_context: AiApiCallContextModel) -> None:
        """
        Records one input lifecycle event.

        Args:
            call_context: Immutable call-context metadata emitted before provider execution.

        Returns:
            None after storing the input event payload.
        """
        self.list_event_order.append("before_call")
        self.list_before_contexts.append(call_context)
        # Normal return after recording the input lifecycle event.
        return None

    def after_call(
        self,
        call_context: AiApiCallContextModel,
        call_result_summary: AiApiCallResultSummaryModel,
    ) -> None:
        """
        Records one output lifecycle event.

        Args:
            call_context: Immutable call-context metadata emitted after provider execution.
            call_result_summary: Metadata-only summary derived from the provider output.

        Returns:
            None after storing the output event payload.
        """
        self.list_event_order.append("after_call")
        self.list_after_events.append((call_context, call_result_summary))
        # Normal return after recording the output lifecycle event.
        return None

    def on_error(
        self,
        call_context: AiApiCallContextModel,
        exception: Exception,
        float_elapsed_ms: float,
    ) -> None:
        """
        Records one error lifecycle event.

        Args:
            call_context: Immutable call-context metadata emitted for the error event.
            exception: Provider exception propagated through the wrapper.
            float_elapsed_ms: Elapsed milliseconds measured before the failure was surfaced.

        Returns:
            None after storing the error event payload.
        """
        self.list_event_order.append("on_error")
        self.list_error_events.append((call_context, exception, float_elapsed_ms))
        # Normal return after recording the error lifecycle event.
        return None


class FakePiiMiddleware:
    """
    Tracks input and output transformation calls while returning deterministic values.

    Args:
        list_event_order: Mutable event-order log shared with the test.
        callable_output_transform: Callable used to transform output text deterministically.

    Returns:
        Simple PII middleware test double for completions ordering tests.
    """

    def __init__(
        self,
        list_event_order: list[str],
        callable_output_transform: Callable[[str], str],
    ) -> None:
        """
        Stores shared order state and the output transform callable used in one test.

        Args:
            list_event_order: Mutable event log shared with the surrounding test.
            callable_output_transform: Callable that transforms output text for assertions.

        Returns:
            None after the test double is ready for input and output transformations.
        """
        self.list_event_order: list[str] = list_event_order
        self.callable_output_transform: Callable[[str], str] = callable_output_transform

    def process_input(self, text: str) -> str:
        """
        Records one input transformation and returns a deterministic sanitized prompt.

        Args:
            text: Raw prompt text supplied by the provider method under test.

        Returns:
            Sanitized prompt text used by the provider call.
        """
        self.list_event_order.append("process_input")
        # Normal return with deterministic sanitized input text for test assertions.
        return f"{text}{TEST_INPUT_SUFFIX}"

    def process_output(self, text: str) -> str:
        """
        Records one output transformation and returns the configured transformed value.

        Args:
            text: Raw output text supplied by the provider method under test.

        Returns:
            Transformed output text determined by the configured callable.
        """
        self.list_event_order.append("process_output")
        # Normal return with deterministic transformed output text for test assertions.
        return self.callable_output_transform(text)


def _build_openai_text_response(
    *,
    text: str,
    finish_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> Any:
    """
    Builds a minimal OpenAI chat-completions response object for unit tests.

    Args:
        text: Response text returned by the fake provider call.
        finish_reason: Finish reason reported by the fake provider response.
        prompt_tokens: Provider prompt token count.
        completion_tokens: Provider completion token count.
        total_tokens: Provider total token count.

    Returns:
        Simple namespace object matching the attributes accessed by the provider code.
    """
    response: Any = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=text, function_call=None),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )
    # Normal return with a fake OpenAI response object.
    return response


def _build_openai_structured_response(
    *,
    arguments: str,
    finish_reason: str = "stop",
    prompt_tokens: int = 9,
    completion_tokens: int = 4,
    total_tokens: int = 13,
) -> Any:
    """
    Builds a minimal OpenAI structured-response object for unit tests.

    Args:
        arguments: JSON string placed in the fake function-call arguments field.
        finish_reason: Finish reason reported by the fake provider response.
        prompt_tokens: Provider prompt token count.
        completion_tokens: Provider completion token count.
        total_tokens: Provider total token count.

    Returns:
        Simple namespace object matching the structured attributes accessed by the provider code.
    """
    response: Any = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=None,
                    function_call=SimpleNamespace(arguments=arguments),
                ),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )
    # Normal return with a fake OpenAI structured response object.
    return response


def test_openai_send_prompt_observability_wraps_entire_public_call() -> None:
    """
    Ensures OpenAI text completions emit one observability sequence across continuation calls.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    list_event_order: list[str] = []
    recording_middleware: RecordingObservabilityMiddleware = (
        RecordingObservabilityMiddleware(list_event_order)
    )
    list_responses: list[Any] = [
        _build_openai_text_response(
            text="Hello",
            finish_reason="length",
            prompt_tokens=10,
            completion_tokens=3,
            total_tokens=13,
        ),
        _build_openai_text_response(
            text=" world",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
        ),
    ]

    def _create_openai_completion(**kwargs: Any) -> Any:
        """
        Pops one queued OpenAI response while recording provider-call order.

        Args:
            kwargs: Provider request payload ignored by this test helper.

        Returns:
            Next queued fake OpenAI response object.
        """
        list_event_order.append("provider_call")
        # Normal return with the next fake provider response.
        return list_responses.pop(0)

    openai_client: AiOpenAICompletions = object.__new__(AiOpenAICompletions)
    openai_client.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create_openai_completion)
        )
    )
    openai_client.completions_model = TEST_COMPLETIONS_MODEL
    openai_client.model = TEST_COMPLETIONS_MODEL
    openai_client.user = TEST_OPENAI_USER
    openai_client._observability_middleware = recording_middleware
    openai_client.pii_middleware = FakePiiMiddleware(
        list_event_order=list_event_order,
        callable_output_transform=lambda text: f"{text}{TEST_OUTPUT_SUFFIX}",
    )

    response_text: str = openai_client.send_prompt(TEST_PROMPT)

    before_context: AiApiCallContextModel = recording_middleware.list_before_contexts[0]
    after_context, result_summary = recording_middleware.list_after_events[0]

    assert response_text == f"Hello world{TEST_OUTPUT_SUFFIX}"
    assert list_event_order == [
        "process_input",
        "before_call",
        "provider_call",
        "provider_call",
        "after_call",
        "process_output",
    ]
    assert before_context.originating_caller_id == TEST_OPENAI_USER
    assert before_context.originating_caller_id_source == "legacy_setting"
    assert before_context.dict_metadata["prompt_char_count"] == len(
        f"{TEST_PROMPT}{TEST_INPUT_SUFFIX}"
    )
    assert before_context.dict_metadata["response_mode"] == "text"
    assert after_context.call_id == before_context.call_id
    assert result_summary.dict_metadata["output_char_count"] == len("Hello world")
    assert result_summary.provider_prompt_tokens == 11
    assert result_summary.provider_completion_tokens == 5
    assert result_summary.provider_total_tokens == 16
    assert result_summary.finish_reason == "stop"
    assert result_summary.dict_metadata["continued_response"] is True
    assert recording_middleware.list_error_events == []


def test_openai_strict_schema_prompt_records_structured_metadata() -> None:
    """
    Ensures OpenAI structured completions include structured-mode observability metadata.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    list_event_order: list[str] = []
    recording_middleware: RecordingObservabilityMiddleware = (
        RecordingObservabilityMiddleware(list_event_order)
    )

    def _create_openai_completion(**kwargs: Any) -> Any:
        """
        Returns one fake structured OpenAI response while recording provider-call order.

        Args:
            kwargs: Provider request payload ignored by this test helper.

        Returns:
            Fake OpenAI structured response object.
        """
        list_event_order.append("provider_call")
        # Normal return with one fake structured provider response.
        return _build_openai_structured_response(arguments='{"answer":"Paris"}')

    openai_client: AiOpenAICompletions = object.__new__(AiOpenAICompletions)
    openai_client.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create_openai_completion)
        )
    )
    openai_client.completions_model = TEST_COMPLETIONS_MODEL
    openai_client.model = TEST_COMPLETIONS_MODEL
    openai_client.user = TEST_OPENAI_USER
    openai_client._observability_middleware = recording_middleware
    openai_client.pii_middleware = FakePiiMiddleware(
        list_event_order=list_event_order,
        callable_output_transform=lambda text: text,
    )

    response_model: FakeStructuredResponse = openai_client.strict_schema_prompt(
        TEST_PROMPT,
        FakeStructuredResponse,
    )

    before_context: AiApiCallContextModel = recording_middleware.list_before_contexts[0]
    _, result_summary = recording_middleware.list_after_events[0]

    assert response_model.answer == "Paris"
    assert before_context.dict_metadata["response_mode"] == "structured"
    assert (
        before_context.dict_metadata["max_response_tokens"]
        == openai_client.STRUCTURED_DEFAULT_MAX_RESPONSE_TOKENS
    )
    assert result_summary.dict_metadata["output_char_count"] == len(
        '{"answer":"Paris"}'
    )


def test_openai_strict_schema_prompt_rejects_small_token_cap() -> None:
    """
    Ensures OpenAI structured prompts fail fast on undersized token caps with a typed error.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    openai_client: AiOpenAICompletions = object.__new__(AiOpenAICompletions)
    openai_client.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: (_ for _ in ()).throw(
                    AssertionError("Provider call should not occur.")
                )
            )
        )
    )
    openai_client.completions_model = TEST_COMPLETIONS_MODEL
    openai_client.model = TEST_COMPLETIONS_MODEL
    openai_client.user = TEST_OPENAI_USER
    openai_client._observability_middleware = RecordingObservabilityMiddleware([])
    openai_client.pii_middleware = FakePiiMiddleware(
        list_event_order=[],
        callable_output_transform=lambda text: text,
    )

    with pytest.raises(StructuredResponseTokenLimitError) as exception_info:
        openai_client.strict_schema_prompt(
            TEST_PROMPT,
            FakeStructuredResponse,
            max_response_tokens=TEST_TOO_SMALL_MAX_RESPONSE_TOKENS,
        )

    assert exception_info.value.provider_name == "openai"
    assert (
        exception_info.value.max_response_tokens == TEST_TOO_SMALL_MAX_RESPONSE_TOKENS
    )
    assert "must be at least" in str(exception_info.value)


def test_openai_strict_schema_prompt_raises_typed_error_on_truncation() -> None:
    """
    Ensures OpenAI structured prompts raise a typed token-limit error before JSON parsing on truncation.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    list_event_order: list[str] = []
    recording_middleware: RecordingObservabilityMiddleware = (
        RecordingObservabilityMiddleware(list_event_order)
    )

    def _create_openai_completion(**kwargs: Any) -> Any:
        """
        Returns one fake truncated OpenAI structured response while recording provider-call order.

        Args:
            kwargs: Provider request payload ignored by this test helper.

        Returns:
            Fake OpenAI structured response object with a truncation finish reason.
        """
        list_event_order.append("provider_call")
        # Normal return with one fake truncated structured provider response.
        return _build_openai_structured_response(
            arguments='{"answer":"Par',
            finish_reason="length",
        )

    openai_client: AiOpenAICompletions = object.__new__(AiOpenAICompletions)
    openai_client.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create_openai_completion)
        )
    )
    openai_client.completions_model = TEST_COMPLETIONS_MODEL
    openai_client.model = TEST_COMPLETIONS_MODEL
    openai_client.user = TEST_OPENAI_USER
    openai_client._observability_middleware = recording_middleware
    openai_client.pii_middleware = FakePiiMiddleware(
        list_event_order=list_event_order,
        callable_output_transform=lambda text: text,
    )

    with pytest.raises(StructuredResponseTokenLimitError) as exception_info:
        openai_client.strict_schema_prompt(
            TEST_PROMPT,
            FakeStructuredResponse,
        )

    assert exception_info.value.finish_reason == "length"
    assert "Increase `max_response_tokens` and retry in client code." in str(
        exception_info.value
    )
    assert list_event_order == [
        "process_input",
        "before_call",
        "provider_call",
        "on_error",
    ]
    assert recording_middleware.list_after_events == []
    assert len(recording_middleware.list_error_events) == 1


def test_gemini_send_prompt_observability_uses_raw_output_for_summary() -> None:
    """
    Ensures Gemini text completions log raw output metadata before output redaction.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    list_event_order: list[str] = []
    recording_middleware: RecordingObservabilityMiddleware = (
        RecordingObservabilityMiddleware(list_event_order)
    )

    def _generate_content(**kwargs: Any) -> Any:
        """
        Returns one fake Gemini response while recording provider-call order.

        Args:
            kwargs: Provider request payload ignored by this test helper.

        Returns:
            Fake Gemini response object exposing text, candidates, and usage metadata.
        """
        list_event_order.append("provider_call")
        # Normal return with one fake Gemini response object.
        return SimpleNamespace(
            text="Gemini raw",
            candidates=[SimpleNamespace(finish_reason="STOP")],
            usage_metadata=SimpleNamespace(
                prompt_token_count=8,
                candidates_token_count=4,
                total_token_count=12,
            ),
        )

    gemini_client: GoogleGeminiCompletions = object.__new__(GoogleGeminiCompletions)
    gemini_client.client = SimpleNamespace(
        models=SimpleNamespace(generate_content=_generate_content)
    )
    gemini_client.completions_model = TEST_COMPLETIONS_MODEL
    gemini_client.model = TEST_COMPLETIONS_MODEL
    gemini_client._observability_middleware = recording_middleware
    gemini_client.pii_middleware = FakePiiMiddleware(
        list_event_order=list_event_order,
        callable_output_transform=lambda text: f"{text}{TEST_OUTPUT_SUFFIX}",
    )

    response_text: str = gemini_client.send_prompt(TEST_PROMPT)

    before_context: AiApiCallContextModel = recording_middleware.list_before_contexts[0]
    _, result_summary = recording_middleware.list_after_events[0]

    assert response_text == f"Gemini raw{TEST_OUTPUT_SUFFIX}"
    assert list_event_order == [
        "process_input",
        "before_call",
        "provider_call",
        "after_call",
        "process_output",
    ]
    assert before_context.dict_metadata["response_mode"] == "text"
    assert result_summary.dict_metadata["output_char_count"] == len("Gemini raw")
    assert result_summary.provider_prompt_tokens == 8
    assert result_summary.provider_completion_tokens == 4
    assert result_summary.provider_total_tokens == 12
    assert result_summary.finish_reason == "STOP"


def test_gemini_strict_schema_prompt_rejects_small_token_cap() -> None:
    """
    Ensures Gemini structured prompts fail fast on undersized token caps with a typed error.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    gemini_client: GoogleGeminiCompletions = object.__new__(GoogleGeminiCompletions)
    gemini_client.client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("Provider call should not occur.")
            )
        )
    )
    gemini_client.completions_model = TEST_COMPLETIONS_MODEL
    gemini_client.model = TEST_COMPLETIONS_MODEL
    gemini_client._observability_middleware = RecordingObservabilityMiddleware([])
    gemini_client.pii_middleware = FakePiiMiddleware(
        list_event_order=[],
        callable_output_transform=lambda text: text,
    )

    with pytest.raises(StructuredResponseTokenLimitError) as exception_info:
        gemini_client.strict_schema_prompt(
            TEST_PROMPT,
            FakeStructuredResponse,
            max_response_tokens=TEST_TOO_SMALL_MAX_RESPONSE_TOKENS,
        )

    assert exception_info.value.provider_name == "google-gemini"
    assert (
        exception_info.value.max_response_tokens == TEST_TOO_SMALL_MAX_RESPONSE_TOKENS
    )
    assert "must be at least" in str(exception_info.value)


def test_gemini_strict_schema_prompt_raises_typed_error_on_truncation() -> None:
    """
    Ensures Gemini structured prompts raise a typed token-limit error before JSON parsing on truncation.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    list_event_order: list[str] = []
    recording_middleware: RecordingObservabilityMiddleware = (
        RecordingObservabilityMiddleware(list_event_order)
    )

    def _generate_content(**kwargs: Any) -> Any:
        """
        Returns one fake truncated Gemini structured response while recording provider-call order.

        Args:
            kwargs: Provider request payload ignored by this test helper.

        Returns:
            Fake Gemini structured response object with a MAX_TOKENS finish reason.
        """
        list_event_order.append("provider_call")
        # Normal return with one fake truncated Gemini structured response.
        return SimpleNamespace(
            text='{"answer":"Par',
            candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
            usage_metadata=SimpleNamespace(
                prompt_token_count=8,
                candidates_token_count=4,
                total_token_count=12,
            ),
        )

    gemini_client: GoogleGeminiCompletions = object.__new__(GoogleGeminiCompletions)
    gemini_client.client = SimpleNamespace(
        models=SimpleNamespace(generate_content=_generate_content)
    )
    gemini_client.completions_model = TEST_COMPLETIONS_MODEL
    gemini_client.model = TEST_COMPLETIONS_MODEL
    gemini_client._observability_middleware = recording_middleware
    gemini_client.pii_middleware = FakePiiMiddleware(
        list_event_order=list_event_order,
        callable_output_transform=lambda text: text,
    )

    with pytest.raises(StructuredResponseTokenLimitError) as exception_info:
        gemini_client.strict_schema_prompt(
            TEST_PROMPT,
            FakeStructuredResponse,
        )

    assert exception_info.value.finish_reason == "MAX_TOKENS"
    assert list_event_order == [
        "process_input",
        "before_call",
        "provider_call",
        "on_error",
    ]
    assert recording_middleware.list_after_events == []
    assert len(recording_middleware.list_error_events) == 1


def test_bedrock_send_prompt_emits_one_observability_sequence_across_retries() -> None:
    """
    Ensures Bedrock retries do not produce one observability sequence per attempt.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    list_event_order: list[str] = []
    recording_middleware: RecordingObservabilityMiddleware = (
        RecordingObservabilityMiddleware(list_event_order)
    )
    list_attempts: list[Any] = [
        RuntimeError("transient failure"),
        {
            "stopReason": "end_turn",
            "usage": {"inputTokens": 7, "outputTokens": 3, "totalTokens": 10},
            "output": {"message": {"content": [{"text": "Bedrock raw"}]}},
        },
    ]

    def _converse(**kwargs: Any) -> Any:
        """
        Raises once, then returns a fake Bedrock response while recording provider-call order.

        Args:
            kwargs: Provider request payload ignored by this test helper.

        Returns:
            Either raises the queued exception or returns the queued Bedrock response.
        """
        list_event_order.append("provider_call")
        current_attempt: Any = list_attempts.pop(0)
        if isinstance(current_attempt, Exception):
            raise current_attempt
        # Normal return with the queued fake Bedrock response.
        return current_attempt

    bedrock_client: AiBedrockCompletions = object.__new__(AiBedrockCompletions)
    bedrock_client.client = SimpleNamespace(converse=_converse)
    bedrock_client.model = TEST_COMPLETIONS_MODEL
    bedrock_client.completions_model = TEST_COMPLETIONS_MODEL
    bedrock_client.backoff_delays = [0.0, 0.0]
    bedrock_client._observability_middleware = recording_middleware
    bedrock_client.pii_middleware = FakePiiMiddleware(
        list_event_order=list_event_order,
        callable_output_transform=lambda text: f"{text}{TEST_OUTPUT_SUFFIX}",
    )
    bedrock_client._sleep_with_backoff = lambda base_delay: None

    response_text: str = bedrock_client.send_prompt(TEST_PROMPT)

    before_context: AiApiCallContextModel = recording_middleware.list_before_contexts[0]
    _, result_summary = recording_middleware.list_after_events[0]

    assert response_text == f"Bedrock raw{TEST_OUTPUT_SUFFIX}"
    assert list_event_order == [
        "process_input",
        "before_call",
        "provider_call",
        "provider_call",
        "after_call",
        "process_output",
    ]
    assert before_context.dict_metadata["response_mode"] == "text"
    assert len(recording_middleware.list_before_contexts) == 1
    assert len(recording_middleware.list_after_events) == 1
    assert recording_middleware.list_error_events == []
    assert result_summary.dict_metadata["output_char_count"] == len("Bedrock raw")
    assert result_summary.provider_prompt_tokens == 7
    assert result_summary.provider_completion_tokens == 3
    assert result_summary.provider_total_tokens == 10


def test_bedrock_strict_schema_prompt_repair_path_sanitizes_and_keeps_metadata() -> (
    None
):
    """
    Ensures Bedrock JSON repair still applies output sanitization and preserves provider metadata.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    list_event_order: list[str] = []
    recording_middleware: RecordingObservabilityMiddleware = (
        RecordingObservabilityMiddleware(list_event_order)
    )

    def _converse(**kwargs: Any) -> Any:
        """
        Returns one fake Bedrock structured response while recording provider-call order.

        Args:
            kwargs: Provider request payload ignored by this test helper.

        Returns:
            Fake Bedrock structured response object.
        """
        list_event_order.append("provider_call")
        # Normal return with one fake structured Bedrock response.
        return {
            "stopReason": "end_turn",
            "usage": {"inputTokens": 9, "outputTokens": 4, "totalTokens": 13},
            "output": {"message": {"content": [{"text": '{"answer":"secret"'}]}},
        }

    bedrock_client: AiBedrockCompletions = object.__new__(AiBedrockCompletions)
    bedrock_client.client = SimpleNamespace(converse=_converse)
    bedrock_client.model = TEST_COMPLETIONS_MODEL
    bedrock_client.completions_model = TEST_COMPLETIONS_MODEL
    bedrock_client.backoff_delays = [0.0]
    bedrock_client._observability_middleware = recording_middleware
    bedrock_client.pii_middleware = FakePiiMiddleware(
        list_event_order=list_event_order,
        callable_output_transform=lambda text: text.replace("secret", "redacted"),
    )
    bedrock_client._sleep_with_backoff = lambda base_delay: None
    bedrock_client._repair_json = lambda raw_json: '{"answer":"secret"}'
    bedrock_client._extract_json_text_from_converse_response = (
        lambda response: response["output"]["message"]["content"][0]["text"]
    )

    response_model: FakeStructuredResponse = bedrock_client.strict_schema_prompt(
        TEST_PROMPT,
        FakeStructuredResponse,
    )

    before_context: AiApiCallContextModel = recording_middleware.list_before_contexts[0]
    _, result_summary = recording_middleware.list_after_events[0]

    assert response_model.answer == "redacted"
    assert list_event_order == [
        "process_input",
        "before_call",
        "provider_call",
        "process_output",
        "process_output",
        "after_call",
    ]
    assert before_context.dict_metadata["response_mode"] == "structured"
    assert result_summary.dict_metadata["output_char_count"] == len(
        '{"answer":"secret"'
    )
    assert result_summary.provider_prompt_tokens == 9
    assert result_summary.provider_completion_tokens == 4
    assert result_summary.provider_total_tokens == 13
    assert result_summary.finish_reason == "end_turn"


def test_bedrock_strict_schema_prompt_rejects_small_token_cap() -> None:
    """
    Ensures Bedrock structured prompts fail fast on undersized token caps with a typed error.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    bedrock_client: AiBedrockCompletions = object.__new__(AiBedrockCompletions)
    bedrock_client.client = SimpleNamespace(
        converse=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Provider call should not occur.")
        )
    )
    bedrock_client.model = TEST_COMPLETIONS_MODEL
    bedrock_client.completions_model = TEST_COMPLETIONS_MODEL
    bedrock_client.backoff_delays = [0.0]
    bedrock_client._observability_middleware = RecordingObservabilityMiddleware([])
    bedrock_client.pii_middleware = FakePiiMiddleware(
        list_event_order=[],
        callable_output_transform=lambda text: text,
    )
    bedrock_client._sleep_with_backoff = lambda base_delay: None

    with pytest.raises(StructuredResponseTokenLimitError) as exception_info:
        bedrock_client.strict_schema_prompt(
            TEST_PROMPT,
            FakeStructuredResponse,
            max_response_tokens=TEST_TOO_SMALL_MAX_RESPONSE_TOKENS,
        )

    assert exception_info.value.provider_name == "bedrock"
    assert (
        exception_info.value.max_response_tokens == TEST_TOO_SMALL_MAX_RESPONSE_TOKENS
    )
    assert "must be at least" in str(exception_info.value)


def test_bedrock_strict_schema_prompt_raises_typed_error_on_truncation() -> None:
    """
    Ensures Bedrock structured prompts raise a typed token-limit error before JSON parsing on truncation.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    list_event_order: list[str] = []
    recording_middleware: RecordingObservabilityMiddleware = (
        RecordingObservabilityMiddleware(list_event_order)
    )

    def _converse(**kwargs: Any) -> Any:
        """
        Returns one fake truncated Bedrock structured response while recording provider-call order.

        Args:
            kwargs: Provider request payload ignored by this test helper.

        Returns:
            Fake Bedrock structured response object with a max_tokens stop reason.
        """
        list_event_order.append("provider_call")
        # Normal return with one fake truncated Bedrock structured response.
        return {
            "stopReason": "max_tokens",
            "usage": {"inputTokens": 9, "outputTokens": 4, "totalTokens": 13},
            "output": {"message": {"content": [{"text": '{"answer":"Par'}]}},
        }

    bedrock_client: AiBedrockCompletions = object.__new__(AiBedrockCompletions)
    bedrock_client.client = SimpleNamespace(converse=_converse)
    bedrock_client.model = TEST_COMPLETIONS_MODEL
    bedrock_client.completions_model = TEST_COMPLETIONS_MODEL
    bedrock_client.backoff_delays = [0.0]
    bedrock_client._observability_middleware = recording_middleware
    bedrock_client.pii_middleware = FakePiiMiddleware(
        list_event_order=list_event_order,
        callable_output_transform=lambda text: text,
    )
    bedrock_client._sleep_with_backoff = lambda base_delay: None

    with pytest.raises(StructuredResponseTokenLimitError) as exception_info:
        bedrock_client.strict_schema_prompt(
            TEST_PROMPT,
            FakeStructuredResponse,
        )

    assert exception_info.value.finish_reason == "max_tokens"
    assert list_event_order == [
        "process_input",
        "before_call",
        "provider_call",
        "on_error",
    ]
    assert recording_middleware.list_after_events == []
    assert len(recording_middleware.list_error_events) == 1
