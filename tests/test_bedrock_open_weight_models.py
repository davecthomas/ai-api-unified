# ruff: noqa: E402

# test_bedrock_open_weight_models.py
"""
Tests for the DeepSeek, Qwen, and GLM models served on Amazon Bedrock.

Covers the catalog and pricing entries, the per-family capability flags
(tool use, structured output, text-only input, no CountTokens, reasoning),
text extraction that skips reasoning blocks, the region hint on an invalid
model id, and the `bedrock` completions alias. All against a mocked
bedrock-runtime client.
"""

from decimal import Decimal
from typing import Any
from unittest.mock import Mock, patch

import pytest

pytest.importorskip("boto3")

from botocore.exceptions import ClientError

from ai_api_unified.ai_base import AIStructuredPrompt, AITool, SupportedDataType
from ai_api_unified.ai_factory import AIFactory
from ai_api_unified.ai_provider_exceptions import (
    AiProviderCapabilityUnsupportedError,
    AiProviderRequestError,
)
from ai_api_unified.completions.ai_bedrock_completions import AiBedrockCompletions
from ai_api_unified.pricing.pricing_registry import PROVIDER_BEDROCK, get_model_pricing

WEATHER_TOOL = AITool(
    name="get_weather",
    description="Get current weather for a city.",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)

LIST_TOOL_CAPABLE_MODELS: list[str] = [
    "deepseek.v3.2",
    "qwen.qwen3-next-80b-a3b",
    "qwen.qwen3-235b-a22b-2507-v1:0",
    "qwen.qwen3-coder-next",
    "qwen.qwen3-32b-v1:0",
    "zai.glm-5",
    "zai.glm-4.7",
    "zai.glm-4.7-flash",
]
LIST_ALL_MODELS: list[str] = [*LIST_TOOL_CAPABLE_MODELS, "us.deepseek.r1-v1:0"]


def _build_client(model: str) -> AiBedrockCompletions:
    with patch("ai_api_unified.ai_bedrock_base.boto3"):
        client = AiBedrockCompletions(model=model)
    client.client = Mock()
    client.backoff_delays = [0.0]
    client._sleep_with_backoff = lambda base_delay: None
    return client


def _converse_response(content: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 9, "outputTokens": 4, "totalTokens": 13},
    }


class TestCatalogAndPricing:
    @pytest.mark.parametrize("str_model", LIST_ALL_MODELS)
    def test_model_is_listed_with_context_window(self, str_model: str) -> None:
        client = _build_client(str_model)
        assert str_model in client.list_model_names
        assert client.max_context_tokens > 0

    @pytest.mark.parametrize("str_model", LIST_ALL_MODELS)
    def test_model_is_priced(self, str_model: str) -> None:
        pricing = get_model_pricing(PROVIDER_BEDROCK, str_model)
        assert pricing is not None
        assert pricing.token_rates.input_per_1m > 0
        assert pricing.token_rates.output_per_1m is not None

    def test_deepseek_v3_2_rates(self) -> None:
        pricing = get_model_pricing(PROVIDER_BEDROCK, "deepseek.v3.2")
        assert pricing is not None
        assert pricing.token_rates.input_per_1m == Decimal("0.62")
        assert pricing.token_rates.output_per_1m == Decimal("1.85")


class TestCapabilities:
    @pytest.mark.parametrize("str_model", LIST_TOOL_CAPABLE_MODELS)
    def test_tool_use_and_structured_output(self, str_model: str) -> None:
        capabilities = _build_client(str_model).capabilities
        assert capabilities.supports_tool_use is True
        assert capabilities.supports_structured_output is True

    @pytest.mark.parametrize("str_model", LIST_ALL_MODELS)
    def test_text_only_and_no_token_counting(self, str_model: str) -> None:
        capabilities = _build_client(str_model).capabilities
        assert capabilities.supported_data_types == [SupportedDataType.TEXT]
        assert capabilities.supports_token_counting is False
        assert capabilities.supports_streaming is True

    def test_deepseek_r1_has_no_tools_but_reasons(self) -> None:
        capabilities = _build_client("us.deepseek.r1-v1:0").capabilities
        assert capabilities.supports_tool_use is False
        assert capabilities.supports_structured_output is False
        assert capabilities.reasoning is True

    def test_nova_capabilities_unchanged(self) -> None:
        capabilities = _build_client("amazon.nova-lite-v1:0").capabilities
        assert capabilities.supports_token_counting is True
        assert SupportedDataType.IMAGE in capabilities.supported_data_types
        assert capabilities.reasoning is False

    def test_count_tokens_is_refused_before_the_request(self) -> None:
        client = _build_client("deepseek.v3.2")
        with pytest.raises(AiProviderCapabilityUnsupportedError):
            client.count_tokens("hello")
        client.client.count_tokens.assert_not_called()

    def test_tools_on_r1_are_refused_before_the_request(self) -> None:
        client = _build_client("us.deepseek.r1-v1:0")
        messages = [{"role": "user", "content": [{"text": "Weather?"}]}]
        with pytest.raises(AiProviderCapabilityUnsupportedError):
            client.send_conversation("sys", messages, tools=[WEATHER_TOOL])
        client.client.converse.assert_not_called()


class TestResponseHandling:
    def test_send_prompt_skips_a_leading_reasoning_block(self) -> None:
        client = _build_client("us.deepseek.r1-v1:0")
        client.client.converse.return_value = _converse_response(
            [
                {"reasoningContent": {"reasoningText": {"text": "thinking..."}}},
                {"text": "ok"},
            ]
        )
        assert client.send_prompt("Reply with ok") == "ok"

    def test_send_prompt_joins_split_text_blocks(self) -> None:
        client = _build_client("zai.glm-5")
        client.client.converse.return_value = _converse_response(
            [{"text": "hel"}, {"text": "lo"}]
        )
        assert client.send_prompt("Say hello") == "hello"

    def test_tool_call_turn_on_deepseek(self) -> None:
        client = _build_client("deepseek.v3.2")
        client.client.converse.return_value = {
            **_converse_response(
                [
                    {"text": "Checking."},
                    {
                        "toolUse": {
                            "toolUseId": "t1",
                            "name": "get_weather",
                            "input": {"city": "Paris"},
                        }
                    },
                ]
            ),
            "stopReason": "tool_use",
        }
        messages = [{"role": "user", "content": [{"text": "Weather in Paris?"}]}]
        turn = client.send_conversation("sys", messages, tools=[WEATHER_TOOL])
        assert turn.tool_calls[0].name == "get_weather"
        assert turn.tool_calls[0].input == {"city": "Paris"}


class TestRegionHint:
    def test_invalid_model_id_names_the_region(self) -> None:
        client = _build_client("qwen.qwen3-235b-a22b-2507-v1:0")
        client.client.converse.side_effect = ClientError(
            {
                "Error": {
                    "Code": "ValidationException",
                    "Message": "The provided model identifier is invalid.",
                },
                "ResponseMetadata": {"HTTPStatusCode": 400},
            },
            "Converse",
        )
        messages = [{"role": "user", "content": [{"text": "hi"}]}]
        with pytest.raises(AiProviderRequestError, match="AWS_REGION"):
            client.send_conversation("sys", messages)

    def test_other_errors_carry_no_hint(self) -> None:
        client = _build_client("zai.glm-5")
        client.client.converse.side_effect = ClientError(
            {
                "Error": {"Code": "AccessDeniedException", "Message": "denied"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "Converse",
        )
        messages = [{"role": "user", "content": [{"text": "hi"}]}]
        with pytest.raises(AiProviderRequestError) as exception_info:
            client.send_conversation("sys", messages)
        assert "AWS_REGION" not in str(exception_info.value)


class TestBedrockAlias:
    def test_bedrock_engine_resolves_to_the_bedrock_client(self) -> None:
        with patch("ai_api_unified.ai_bedrock_base.boto3"):
            client = AIFactory.get_ai_completions_client(
                model_name="zai.glm-5", completions_engine="bedrock"
            )
        assert isinstance(client, AiBedrockCompletions)
        assert client.completions_model == "zai.glm-5"


class TestRequestsPassBotocoreValidation:
    """
    Mocked tests replace the client wholesale, so botocore never checks the
    request shape. These use a real bedrock-runtime client with a Stubber,
    which runs botocore's own parameter validation before returning the
    canned response. That is what caught the structured-output schema being
    sent as a dict instead of the JSON string Converse requires.
    """

    @staticmethod
    def _stubbed_client(model: str) -> tuple[AiBedrockCompletions, Any]:
        import boto3
        from botocore.stub import Stubber

        runtime = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
        )
        stubber = Stubber(runtime)
        client = AiBedrockCompletions(model=model, bedrock_client=runtime)
        client.backoff_delays = [0.0]
        client._sleep_with_backoff = lambda base_delay: None
        return client, stubber

    def test_structured_output_request_is_valid(self) -> None:
        import json

        client, stubber = self._stubbed_client("zai.glm-5")
        stubber.add_response(
            "converse",
            {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": json.dumps({"city": "Paris"})}],
                    }
                },
                "stopReason": "end_turn",
                "usage": {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8},
                "metrics": {"latencyMs": 10},
            },
        )
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        }
        with stubber:
            result = client.send_structured_output("Capital?", response_schema=schema)
        assert result.data == {"city": "Paris"}

    def test_tool_conversation_request_is_valid(self) -> None:
        client, stubber = self._stubbed_client("deepseek.v3.2")
        stubber.add_response(
            "converse",
            {
                "output": {
                    "message": {"role": "assistant", "content": [{"text": "ok"}]}
                },
                "stopReason": "end_turn",
                "usage": {"inputTokens": 5, "outputTokens": 1, "totalTokens": 6},
                "metrics": {"latencyMs": 10},
            },
        )
        messages = [{"role": "user", "content": [{"text": "hi"}]}]
        with stubber:
            turn = client.send_conversation("sys", messages, tools=[WEATHER_TOOL])
        assert turn.text == "ok"


class _Capital(AIStructuredPrompt):
    city: str | None = None

    @staticmethod
    def get_prompt(input_text: str = "") -> str:
        return input_text


class TestStrictSchemaPrompt:
    def test_open_weight_models_use_native_structured_output(self) -> None:
        client = _build_client("zai.glm-5")
        client.client.converse.return_value = _converse_response(
            [{"text": '{"city": "Paris"}'}]
        )
        result = client.strict_schema_prompt("Capital of France?", _Capital)
        assert result.city == "Paris"
        kwargs = client.client.converse.call_args.kwargs
        assert "outputConfig" in kwargs
        # The legacy path's prefill and stop sequence are rejected by these models.
        assert "stopSequences" not in kwargs.get("inferenceConfig", {})
        assert kwargs["messages"][-1]["role"] == "user"

    def test_r1_uses_a_plain_request_and_extracts_the_json(self) -> None:
        client = _build_client("us.deepseek.r1-v1:0")
        client.client.converse.return_value = _converse_response(
            [
                {"text": 'Here it is:\n```json\n{"city": "Paris"}\n```'},
                {"reasoningContent": {"reasoningText": {"text": "thinking"}}},
            ]
        )
        result = client.strict_schema_prompt("Capital of France?", _Capital)
        assert result.city == "Paris"
        kwargs = client.client.converse.call_args.kwargs
        assert kwargs["messages"][-1]["role"] == "user"
        assert "stopSequences" not in kwargs["inferenceConfig"]
        assert "outputConfig" not in kwargs
