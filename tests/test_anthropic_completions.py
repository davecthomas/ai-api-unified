# ruff: noqa: E402

# test_anthropic_completions.py
"""
Tests for the native Anthropic API completions engine (`claude`).

Covers factory resolution, capabilities, text send_prompt, structured output,
streaming, token counting, image content assembly, pricing registry entries,
and model lifecycle enforcement, all against a mocked SDK client.
"""

import os
from decimal import Decimal
from typing import Any
from unittest.mock import Mock, patch

import pytest

pytest.importorskip("anthropic")

from ai_api_unified.ai_base import (
    AIStructuredPrompt,
    SupportedDataType,
)
from ai_api_unified.ai_factory import AIFactory
from ai_api_unified.ai_provider_exceptions import AiProviderConfigurationError
from ai_api_unified.completions.ai_anthropic_completions import (
    AiAnthropicCompletions,
    AICompletionsCapabilitiesAnthropic,
)
from ai_api_unified.completions.ai_openai_completions import (
    AICompletionsPromptParamsOpenAI,
)
from ai_api_unified.pricing import pricing_registry
from ai_api_unified.pricing.pricing_registry import (
    PROVIDER_ANTHROPIC,
    get_model_pricing,
)


def _build_client(model: str = "claude-opus-4-8") -> AiAnthropicCompletions:
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        client = AiAnthropicCompletions(model=model)
    client.client = Mock()
    return client


def _text_response(
    text: str,
    *,
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> Mock:
    return Mock(
        content=[
            Mock(type="thinking", thinking=""),
            Mock(type="text", text=text),
        ],
        stop_reason=stop_reason,
        usage=Mock(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def test_factory_resolves_claude_engine() -> None:
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        client = AIFactory.get_ai_completions_client(
            completions_engine="claude", model_name="claude-opus-4-8"
        )
    assert isinstance(client, AiAnthropicCompletions)


def test_missing_api_key_raises() -> None:
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            AiAnthropicCompletions(model="claude-opus-4-8")


class TestCapabilities:
    def test_opus_capabilities(self) -> None:
        capabilities = AICompletionsCapabilitiesAnthropic.for_model("claude-opus-4-8")
        assert capabilities.context_window_length == 1_000_000
        assert capabilities.reasoning is True
        assert capabilities.supports_streaming is True
        assert capabilities.supports_token_counting is True
        assert capabilities.supported_data_types == [
            SupportedDataType.TEXT,
            SupportedDataType.IMAGE,
        ]
        assert capabilities.pricing is not None
        assert capabilities.pricing.token_rates.input_per_1m == Decimal("5.00")

    def test_haiku_context_window(self) -> None:
        capabilities = AICompletionsCapabilitiesAnthropic.for_model("claude-haiku-4-5")
        assert capabilities.context_window_length == 200_000

    def test_unknown_model_gets_conservative_default(self) -> None:
        capabilities = AICompletionsCapabilitiesAnthropic.for_model("claude-unknown")
        assert (
            capabilities.context_window_length
            == AICompletionsCapabilitiesAnthropic.DEFAULT_CONTEXT_WINDOW_LENGTH
        )
        assert capabilities.pricing is None


class TestSendPrompt:
    def test_send_prompt_uses_messages_create(self) -> None:
        client = _build_client()
        client.client.messages.create.return_value = _text_response("Hello there")

        result: str = client.send_prompt("Greet me")

        assert result == "Hello there"
        create_kwargs = client.client.messages.create.call_args.kwargs
        assert create_kwargs["model"] == "claude-opus-4-8"
        assert (
            create_kwargs["max_tokens"] == AiAnthropicCompletions.SEND_PROMPT_MAX_TOKENS
        )
        assert create_kwargs["system"]
        assert create_kwargs["messages"] == [{"role": "user", "content": "Greet me"}]

    def test_send_prompt_joins_only_text_blocks(self) -> None:
        client = _build_client()
        client.client.messages.create.return_value = Mock(
            content=[
                Mock(type="thinking", thinking="internal"),
                Mock(type="text", text="Part one. "),
                Mock(type="text", text="Part two."),
            ],
            stop_reason="end_turn",
            usage=Mock(input_tokens=1, output_tokens=2),
        )

        assert client.send_prompt("Go") == "Part one. Part two."

    def test_send_prompt_builds_image_content(self) -> None:
        client = _build_client()
        client.client.messages.create.return_value = _text_response("A red square")
        params = AICompletionsPromptParamsOpenAI(
            included_types=[SupportedDataType.IMAGE],
            included_data=[b"fake-image-bytes"],
            included_mime_types=["image/png"],
        )

        client.send_prompt("Describe this image", other_params=params)

        content = client.client.messages.create.call_args.kwargs["messages"][0][
            "content"
        ]
        assert content[0]["type"] == "image"
        assert content[0]["source"]["type"] == "base64"
        assert content[0]["source"]["media_type"] == "image/png"
        assert content[1] == {"type": "text", "text": "Describe this image"}


class TestStreaming:
    def test_streaming_yields_text_deltas(self) -> None:
        client = _build_client()
        client.client.messages.create.return_value = iter(
            [
                Mock(type="message_start", message=Mock(usage=Mock(input_tokens=7))),
                Mock(
                    type="content_block_delta",
                    delta=Mock(type="text_delta", text="Hel"),
                ),
                Mock(
                    type="content_block_delta",
                    delta=Mock(type="text_delta", text="lo"),
                ),
                Mock(
                    type="message_delta",
                    delta=Mock(stop_reason="end_turn"),
                    usage=Mock(output_tokens=2),
                ),
            ]
        )

        list_chunks: list[str] = list(client.send_prompt_streaming("Greet me"))

        assert list_chunks == ["Hel", "lo"]
        create_kwargs = client.client.messages.create.call_args.kwargs
        assert create_kwargs["stream"] is True
        assert (
            create_kwargs["max_tokens"] == AiAnthropicCompletions.STREAMING_MAX_TOKENS
        )


class _Person(AIStructuredPrompt):
    name: str | None = None
    age: int | None = None

    @staticmethod
    def get_prompt(input_text: str = "") -> str:
        return f"Extract name and age from: {input_text}"


class TestStrictSchemaPrompt:
    def test_strict_schema_prompt_parses_json_output(self) -> None:
        client = _build_client()
        client.client.messages.create.return_value = _text_response(
            '{"name": "Alice", "age": 30}'
        )

        result: Any = client.strict_schema_prompt("Extract", _Person)

        assert result.name == "Alice"
        assert result.age == 30
        create_kwargs = client.client.messages.create.call_args.kwargs
        dict_format = create_kwargs["output_config"]["format"]
        assert dict_format["type"] == "json_schema"
        assert dict_format["schema"]["additionalProperties"] is False

    def test_strict_schema_prompt_raises_on_invalid_json(self) -> None:
        client = _build_client()
        client.client.messages.create.return_value = _text_response("not json")

        with pytest.raises(ValueError, match="Invalid JSON response"):
            client.strict_schema_prompt("Extract", _Person)


class TestCountTokens:
    def test_count_tokens_uses_provider_endpoint(self) -> None:
        client = _build_client()
        client.client.messages.count_tokens.return_value = Mock(input_tokens=42)

        assert client.count_tokens("How many tokens?") == 42
        count_kwargs = client.client.messages.count_tokens.call_args.kwargs
        assert count_kwargs["model"] == "claude-opus-4-8"
        assert count_kwargs["messages"][0]["role"] == "user"


class TestObservabilityIdentity:
    def test_vendor_and_engine_resolution(self) -> None:
        # provider_vendor must match the pricing-registry provider label so
        # finops cost enrichment can price claude-engine calls.
        client = _build_client()
        assert client._resolve_observability_provider_vendor() == PROVIDER_ANTHROPIC
        assert client._resolve_observability_provider_engine() == "claude"


class TestPricingAndLifecycle:
    def test_compute_completion_cost(self) -> None:
        client = _build_client()
        # 1M in * 5.00/1M + 1M out * 25.00/1M = 30.00
        assert client.compute_completion_cost(
            input_tokens=1_000_000, output_tokens=1_000_000
        ) == Decimal("30.00")

    def test_registry_prices_current_models(self) -> None:
        pricing = get_model_pricing(PROVIDER_ANTHROPIC, "claude-opus-4-8")
        assert pricing is not None
        assert pricing.token_rates.input_per_1m == Decimal("5.00")
        assert pricing.token_rates.output_per_1m == Decimal("25.00")
        assert pricing.token_rates.cached_input_per_1m == Decimal("0.50")

        haiku = get_model_pricing(PROVIDER_ANTHROPIC, "claude-haiku-4-5")
        assert haiku is not None
        assert haiku.token_rates.input_per_1m == Decimal("1.00")

    def test_retired_model_fails_fast(self) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with pytest.raises(AiProviderConfigurationError, match="retired"):
                AiAnthropicCompletions(model="claude-3-opus-20240229")

    def test_deprecated_model_warns_once(self) -> None:
        pricing_registry._SET_WARNED_MODELS.discard(
            (PROVIDER_ANTHROPIC, "claude-opus-4-1")
        )
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with pytest.warns(DeprecationWarning, match="claude-opus-4-8"):
                AiAnthropicCompletions(model="claude-opus-4-1")
