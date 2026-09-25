# ruff: noqa: E402

# test_bedrock_claude_strict_schema.py
"""
Tests for strict_schema_prompt request shapes on Claude via Bedrock.

Claude 4.6 and later reject an assistant prefill, and Claude 4.7 and later
reject a non-default temperature. strict_schema_prompt routes each model to
a request it accepts: native structured output where Bedrock offers it, a
plain request with the JSON extracted from the reply otherwise, and the
original prefill request for models that still accept it.
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest

pytest.importorskip("boto3")

from ai_api_unified.ai_base import AIStructuredPrompt
from ai_api_unified.completions.ai_bedrock_completions import AiBedrockCompletions


class _Capital(AIStructuredPrompt):
    city: str | None = None

    @staticmethod
    def get_prompt(input_text: str = "") -> str:
        return input_text


def _build_client(model: str, reply_text: str) -> AiBedrockCompletions:
    with patch("ai_api_unified.ai_bedrock_base.boto3"):
        client = AiBedrockCompletions(model=model)
    client.client = Mock()
    client.backoff_delays = [0.0]
    client._sleep_with_backoff = lambda base_delay: None
    client.client.converse.return_value = {
        "output": {"message": {"role": "assistant", "content": [{"text": reply_text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 9, "outputTokens": 4, "totalTokens": 13},
    }
    return client


def _request(client: AiBedrockCompletions) -> dict[str, Any]:
    return client.client.converse.call_args.kwargs


class TestClaudeStrictSchemaRouting:
    def test_claude_4_5_keeps_the_prefill_request(self) -> None:
        client = _build_client(
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0", '{"city": "Paris"}'
        )
        assert client.strict_schema_prompt("Capital?", _Capital).city == "Paris"
        kwargs = _request(client)
        assert kwargs["messages"][-1] == {
            "role": "assistant",
            "content": [{"text": "```json"}],
        }
        assert kwargs["inferenceConfig"]["stopSequences"] == ["```"]
        assert kwargs["inferenceConfig"]["temperature"] == 0.9

    def test_opus_4_6_uses_native_structured_output(self) -> None:
        client = _build_client("us.anthropic.claude-opus-4-6-v1", '{"city": "Paris"}')
        assert client.strict_schema_prompt("Capital?", _Capital).city == "Paris"
        kwargs = _request(client)
        assert "outputConfig" in kwargs
        assert kwargs["messages"][-1]["role"] == "user"

    def test_sonnet_4_6_sends_no_prefill_but_keeps_temperature(self) -> None:
        client = _build_client(
            "us.anthropic.claude-sonnet-4-6", '```json\n{"city": "Paris"}\n```'
        )
        assert client.strict_schema_prompt("Capital?", _Capital).city == "Paris"
        kwargs = _request(client)
        assert kwargs["messages"][-1]["role"] == "user"
        assert "stopSequences" not in kwargs["inferenceConfig"]
        assert kwargs["inferenceConfig"]["temperature"] == 0.9
        assert "outputConfig" not in kwargs

    @pytest.mark.parametrize(
        "str_model",
        [
            "us.anthropic.claude-opus-4-7",
            "us.anthropic.claude-opus-4-8",
            "us.anthropic.claude-opus-5",
            "us.anthropic.claude-opus-5-5",
            "us.anthropic.claude-sonnet-5",
            "us.anthropic.claude-fable-5-1",
        ],
    )
    def test_claude_4_7_and_later_send_no_prefill_and_no_temperature(
        self, str_model: str
    ) -> None:
        client = _build_client(str_model, 'The answer: {"city": "Paris"}')
        assert client.strict_schema_prompt("Capital?", _Capital).city == "Paris"
        kwargs = _request(client)
        assert kwargs["messages"][-1]["role"] == "user"
        assert kwargs["inferenceConfig"] == {"maxTokens": 2048}

    def test_nova_is_unchanged(self) -> None:
        client = _build_client("amazon.nova-lite-v1:0", '{"city": "Paris"}')
        client.strict_schema_prompt("Capital?", _Capital)
        kwargs = _request(client)
        assert kwargs["messages"][-1]["role"] == "assistant"
        assert kwargs["inferenceConfig"]["stopSequences"] == ["```"]


class TestExtractJsonPayload:
    @pytest.mark.parametrize(
        ("str_text", "str_expected"),
        [
            ('{"a": 1}', '{"a": 1}'),
            ('```json\n{"a": 1}\n```', '{"a": 1}'),
            ('```\n{"a": 1}\n```', '{"a": 1}'),
            ('Sure. {"a": {"b": 2}} Done.', '{"a": {"b": 2}}'),
            ("[1, 2]", "[1, 2]"),
            ("no json here", "no json here"),
        ],
    )
    def test_extracts_the_json_value(self, str_text: str, str_expected: str) -> None:
        assert AiBedrockCompletions._extract_json_payload(str_text) == str_expected


class TestRequestsPassBotocoreValidation:
    """Runs each routed request through botocore's real parameter validation."""

    @pytest.mark.parametrize(
        "str_model",
        [
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "us.anthropic.claude-sonnet-4-6",
            "us.anthropic.claude-opus-4-8",
        ],
    )
    def test_request_is_valid(self, str_model: str) -> None:
        import boto3
        from botocore.stub import Stubber

        runtime = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
        )
        stubber = Stubber(runtime)
        stubber.add_response(
            "converse",
            {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": '{"city": "Paris"}'}],
                    }
                },
                "stopReason": "end_turn",
                "usage": {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8},
                "metrics": {"latencyMs": 10},
            },
        )
        client = AiBedrockCompletions(model=str_model, bedrock_client=runtime)
        client.backoff_delays = [0.0]
        with stubber:
            result = client.strict_schema_prompt("Capital?", _Capital)
        assert result.city == "Paris"
