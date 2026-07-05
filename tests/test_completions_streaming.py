# test_completions_streaming.py
"""
Tests for streaming completions support.

Covers:
    - supports_streaming capability flags per provider
    - Base-class template gating (capability, PII configuration, empty prompt)
    - execute_observed_streaming_call lifecycle (events, abandonment, errors)
    - Provider stream call shape and chunk forwarding with mocked SDK clients
"""

import os
from typing import Any
from unittest.mock import Mock, patch

import pytest

from ai_api_unified.ai_base import (
    AIBaseCompletions,
    AICompletionsCapabilitiesBase,
    AICompletionsPromptParamsBase,
    AIStructuredPrompt,
)
from ai_api_unified.ai_provider_exceptions import (
    AiProviderCapabilityUnsupportedError,
    AiProviderConfigurationError,
)
from ai_api_unified.middleware.observability_runtime import (
    execute_observed_streaming_call,
)


class _StubCompletions(AIBaseCompletions):
    """Minimal concrete completions client relying on base-class defaults."""

    def __init__(self) -> None:
        super().__init__(model="stub-model")

    @property
    def list_model_names(self) -> list[str]:
        return ["stub-model"]

    @property
    def max_context_tokens(self) -> int:
        return 1000

    @property
    def price_per_1k_tokens(self) -> float:
        return 0.0

    def send_prompt(
        self, prompt: str, *, other_params: AICompletionsPromptParamsBase | None = None
    ) -> str:
        return "stub"

    def strict_schema_prompt(
        self,
        prompt: str,
        response_model: type[AIStructuredPrompt],
        max_response_tokens: int = AIBaseCompletions.STRUCTURED_DEFAULT_MAX_RESPONSE_TOKENS,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> AIStructuredPrompt:
        raise NotImplementedError


class _StreamingStubCompletions(_StubCompletions):
    """Stub whose capabilities declare streaming support."""

    @property
    def capabilities(self) -> AICompletionsCapabilitiesBase:
        return AICompletionsCapabilitiesBase(
            context_window_length=1000, supports_streaming=True
        )

    def _send_prompt_streaming_provider(self, prompt, *, other_params=None):
        yield "chunk-1"
        yield "chunk-2"


class TestStreamingCapabilityFlags:
    """supports_streaming values across capability descriptors."""

    def test_base_default_is_off(self) -> None:
        capabilities = AICompletionsCapabilitiesBase(context_window_length=1)
        assert capabilities.supports_streaming is False

    def test_google_models_support_streaming(self) -> None:
        pytest.importorskip("google.genai")
        from ai_api_unified.completions.ai_google_gemini_capabilities import (
            AICompletionsCapabilitiesGoogle,
        )

        assert (
            AICompletionsCapabilitiesGoogle.for_model(
                "gemini-2.5-flash"
            ).supports_streaming
            is True
        )

    def test_openai_models_support_streaming(self) -> None:
        pytest.importorskip("openai")
        from ai_api_unified.completions.ai_openai_completions import (
            AICompletionsCapabilitiesOpenAI,
        )

        assert (
            AICompletionsCapabilitiesOpenAI.for_model("gpt-4o-mini").supports_streaming
            is True
        )

    def test_bedrock_models_support_streaming(self) -> None:
        pytest.importorskip("boto3")
        from ai_api_unified.completions.ai_bedrock_completions import (
            AICompletionsCapabilitiesBedrock,
        )

        capabilities = AICompletionsCapabilitiesBedrock.for_model(
            "amazon.nova-lite-v1:0", context_window_length=8_192_000
        )
        assert capabilities.supports_streaming is True


class TestBaseStreamingGates:
    """Template-method gating on AIBaseCompletions.send_prompt_streaming."""

    def test_non_streaming_model_raises_capability_error(self) -> None:
        client = _StubCompletions()
        with pytest.raises(
            AiProviderCapabilityUnsupportedError, match="does not support streaming"
        ):
            client.send_prompt_streaming("hello")

    def test_empty_prompt_raises_value_error(self) -> None:
        client = _StreamingStubCompletions()
        with pytest.raises(ValueError, match="cannot be empty"):
            client.send_prompt_streaming("   ")

    def test_pii_enabled_raises_configuration_error(self) -> None:
        client = _StreamingStubCompletions()
        client.pii_middleware.bool_enabled = True
        with pytest.raises(AiProviderConfigurationError, match="PII redaction"):
            client.send_prompt_streaming("hello")

    def test_streaming_stub_yields_chunks(self) -> None:
        client = _StreamingStubCompletions()
        assert list(client.send_prompt_streaming("hello")) == ["chunk-1", "chunk-2"]

    def test_base_provider_hook_raises(self) -> None:
        client = _StubCompletions()
        with pytest.raises(AiProviderCapabilityUnsupportedError):
            list(client._send_prompt_streaming_provider("hello"))


class _RecordingMiddleware:
    """Fake observability middleware recording lifecycle events."""

    def __init__(self) -> None:
        self.bool_enabled = True
        self.list_events: list[str] = []
        self.list_summaries: list[Any] = []

    def before_call(self, *, call_context) -> None:
        self.list_events.append("before_call")

    def after_call(self, *, call_context, call_result_summary) -> None:
        self.list_events.append("after_call")
        self.list_summaries.append(call_result_summary)

    def on_error(self, *, call_context, exception, float_elapsed_ms) -> None:
        self.list_events.append("on_error")


class _FakeCallContext:
    def with_direction(self, direction: str) -> "_FakeCallContext":
        return self


class TestExecuteObservedStreamingCall:
    """Lifecycle semantics of the streaming observability runtime helper."""

    @staticmethod
    def _run(
        middleware: _RecordingMiddleware,
        chunks: list[str],
        *,
        fail_after: int | None = None,
    ):
        def _open_stream():
            for index, chunk in enumerate(chunks):
                if fail_after is not None and index == fail_after:
                    raise RuntimeError("stream failed")
                yield chunk

        return execute_observed_streaming_call(
            observability_middleware=middleware,
            callable_build_call_context=lambda: _FakeCallContext(),
            callable_open_stream=_open_stream,
            callable_build_result_summary=lambda elapsed_ms: {"elapsed_ms": elapsed_ms},
        )

    def test_disabled_middleware_passes_through(self) -> None:
        middleware = _RecordingMiddleware()
        middleware.bool_enabled = False
        assert list(self._run(middleware, ["a", "b"])) == ["a", "b"]
        assert middleware.list_events == []

    def test_exhaustion_emits_before_and_after(self) -> None:
        middleware = _RecordingMiddleware()
        assert list(self._run(middleware, ["a", "b"])) == ["a", "b"]
        assert middleware.list_events == ["before_call", "after_call"]
        assert middleware.list_summaries[0]["elapsed_ms"] >= 0

    def test_abandonment_emits_after_call_once(self) -> None:
        middleware = _RecordingMiddleware()
        stream = self._run(middleware, ["a", "b", "c"])
        assert next(stream) == "a"
        stream.close()
        assert middleware.list_events == ["before_call", "after_call"]

    def test_stream_error_emits_on_error_and_propagates(self) -> None:
        middleware = _RecordingMiddleware()
        stream = self._run(middleware, ["a", "b"], fail_after=1)
        assert next(stream) == "a"
        with pytest.raises(RuntimeError, match="stream failed"):
            next(stream)
        assert middleware.list_events == ["before_call", "on_error"]


class TestGoogleGeminiStreaming:
    """Mocked Gemini streaming call shape."""

    @staticmethod
    def _build_client(mock_client: Mock) -> Any:
        pytest.importorskip("google.genai")
        from ai_api_unified.completions.ai_google_gemini_completions import (
            GoogleGeminiCompletions,
        )

        with patch.object(
            GoogleGeminiCompletions,
            "_initialize_client",
            lambda self: setattr(self, "client", mock_client),
        ):
            # Normal return with a Gemini completions client whose SDK client is mocked.
            return GoogleGeminiCompletions(model="gemini-2.5-flash")

    def test_streaming_yields_chunk_text(self) -> None:
        mock_client = Mock()
        mock_client.models.generate_content_stream.return_value = iter(
            [Mock(text="Hello "), Mock(text=None), Mock(text="world")]
        )
        client = self._build_client(mock_client)

        list_chunks: list[str] = list(client.send_prompt_streaming("greet me"))

        assert list_chunks == ["Hello ", "world"]
        stream_kwargs = mock_client.models.generate_content_stream.call_args.kwargs
        assert stream_kwargs["model"] == "gemini-2.5-flash"
        assert stream_kwargs["contents"]


class TestOpenAIStreaming:
    """Mocked OpenAI streaming call shape."""

    @staticmethod
    def _build_client() -> Any:
        pytest.importorskip("openai")
        from ai_api_unified.completions.ai_openai_completions import (
            AiOpenAICompletions,
        )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = AiOpenAICompletions(model="gpt-4o-mini")
        client.client = Mock()
        # Normal return with an OpenAI completions client whose SDK client is mocked.
        return client

    def test_streaming_yields_delta_content(self) -> None:
        client = self._build_client()
        client.client.chat.completions.create.return_value = iter(
            [
                Mock(
                    usage=None,
                    choices=[Mock(delta=Mock(content="Hel"), finish_reason=None)],
                ),
                Mock(
                    usage=None,
                    choices=[Mock(delta=Mock(content="lo"), finish_reason="stop")],
                ),
                Mock(usage=Mock(), choices=[]),
            ]
        )

        list_chunks: list[str] = list(client.send_prompt_streaming("greet me"))

        assert list_chunks == ["Hel", "lo"]
        create_kwargs = client.client.chat.completions.create.call_args.kwargs
        assert create_kwargs["stream"] is True
        assert create_kwargs["stream_options"] == {"include_usage": True}
        assert create_kwargs["model"] == "gpt-4o-mini"


class TestBedrockStreaming:
    """Mocked Bedrock ConverseStream call shape."""

    @staticmethod
    def _build_client() -> Any:
        pytest.importorskip("boto3")
        with patch("ai_api_unified.ai_bedrock_base.boto3"):
            from ai_api_unified.completions.ai_bedrock_completions import (
                AiBedrockCompletions,
            )

            client = AiBedrockCompletions(model="amazon.nova-lite-v1:0")
        client.client = Mock()
        # Normal return with a Bedrock completions client whose SDK client is mocked.
        return client

    def test_streaming_yields_content_block_deltas(self) -> None:
        client = self._build_client()
        client.client.converse_stream.return_value = {
            "stream": iter(
                [
                    {"contentBlockDelta": {"delta": {"text": "Hel"}}},
                    {"contentBlockDelta": {"delta": {"text": "lo"}}},
                    {"messageStop": {"stopReason": "end_turn"}},
                    {"metadata": {"usage": {"inputTokens": 3, "outputTokens": 2}}},
                ]
            )
        }

        list_chunks: list[str] = list(client.send_prompt_streaming("greet me"))

        assert list_chunks == ["Hel", "lo"]
        converse_kwargs = client.client.converse_stream.call_args.kwargs
        assert converse_kwargs["modelId"] == "amazon.nova-lite-v1:0"
        assert converse_kwargs["messages"][0]["role"] == "user"
