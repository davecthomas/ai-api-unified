# test_multi_engine_conversation_api.py
"""
Mocked tests for the conversation, structured-output, async, and retry
features on the openai, openai-responses, google-gemini, and bedrock engines
(claude engine coverage lives in test_completions_conversation_api.py).

Transport faking follows each engine's established repo pattern: construct
the real client with stubbed credentials, then replace the SDK client
attribute with Mock objects mimicking that SDK's object graph.
"""

import json
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ai_api_unified.ai_base import (
    AIFinishReason,
    AITool,
)
from ai_api_unified.ai_provider_exceptions import (
    AiProviderCapabilityUnsupportedError,
    AiProviderRequestError,
)

WEATHER_TOOL = AITool(
    name="get_weather",
    description="Get current weather for a city.",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
    strict=True,
)

GRAPH_SCHEMA: dict = {
    "type": "object",
    "properties": {"nodes": {"type": "array", "items": {"type": "object"}}},
    "required": ["nodes"],
}


# ── OpenAI Chat Completions engine ──────────────────────────────────────────

openai = pytest.importorskip("openai")

from ai_api_unified.completions.ai_openai_completions import (  # noqa: E402
    AiOpenAICompletions,
)
from ai_api_unified.completions.ai_openai_responses_completions import (  # noqa: E402
    AiOpenAIResponsesCompletions,
)


def _build_openai_client(**kwargs) -> AiOpenAICompletions:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        client = AiOpenAICompletions(model="gpt-4o-mini", **kwargs)
    client.client = Mock()
    return client


def _chat_usage(input_tokens: int = 10, output_tokens: int = 5) -> Mock:
    return Mock(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        prompt_tokens_details=Mock(cached_tokens=None),
    )


def _chat_message(
    content: str | None = None,
    tool_calls: list | None = None,
    refusal: str | None = None,
) -> Mock:
    message = Mock(spec=["content", "tool_calls", "refusal", "model_dump"])
    message.content = content
    message.tool_calls = tool_calls
    message.refusal = refusal
    message.model_dump = Mock(side_effect=TypeError("test double"))
    return message


def _chat_tool_call(call_id: str, name: str, arguments: str) -> Mock:
    function = Mock(spec=["name", "arguments"])
    function.name = name
    function.arguments = arguments
    tool_call = Mock(spec=["id", "function"])
    tool_call.id = call_id
    tool_call.function = function
    return tool_call


def _chat_response(message: Mock, finish_reason: str = "stop") -> Mock:
    return Mock(
        choices=[Mock(message=message, finish_reason=finish_reason)],
        usage=_chat_usage(),
    )


class TestOpenAIChatConversation:
    def test_full_tool_loop_cycle(self):
        client = _build_openai_client()
        turn1 = _chat_response(
            _chat_message(
                content=None,
                tool_calls=[_chat_tool_call("call_1", "get_weather", '{"city": "SF"}')],
            ),
            finish_reason="tool_calls",
        )
        turn2 = _chat_response(_chat_message(content="Sunny."), finish_reason="stop")
        client.client.chat.completions.create.side_effect = [turn1, turn2]

        messages = [{"role": "user", "content": "Weather in SF?"}]
        result1 = client.send_conversation("sys", messages, tools=[WEATHER_TOOL])
        assert result1.finish_reason is AIFinishReason.TOOL_USE
        assert result1.tool_calls[0].input == {"city": "SF"}
        # raw_content is the full assistant message; extend appends it as-is.
        client.extend_messages_with_turn(messages, result1)
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["tool_calls"][0]["id"] == "call_1"

        messages.append(
            client.build_tool_result_message(
                tool_call_id="call_1", result={"temp_f": 65}, is_error=False
            )
        )
        assert messages[-1]["role"] == "tool"
        assert messages[-1]["tool_call_id"] == "call_1"

        result2 = client.send_conversation("sys", messages, tools=[WEATHER_TOOL])
        assert result2.finish_reason is AIFinishReason.COMPLETE
        assert result2.text == "Sunny."
        assert result2.usage.input_tokens == 10

        first_kwargs = client.client.chat.completions.create.call_args_list[0].kwargs
        assert first_kwargs["messages"][0] == {"role": "system", "content": "sys"}
        assert first_kwargs["tools"][0]["function"]["name"] == "get_weather"
        assert first_kwargs["tools"][0]["function"]["strict"] is True

    def test_forced_tool_choice_shape(self):
        client = _build_openai_client()
        client.client.chat.completions.create.return_value = _chat_response(
            _chat_message(tool_calls=[_chat_tool_call("call_2", "get_weather", "{}")]),
            finish_reason="tool_calls",
        )
        client.send_conversation(
            "sys",
            [{"role": "user", "content": "hi"}],
            tools=[WEATHER_TOOL],
            tool_choice="get_weather",
        )
        kwargs = client.client.chat.completions.create.call_args.kwargs
        assert kwargs["tool_choice"] == {
            "type": "function",
            "function": {"name": "get_weather"},
        }

    def test_forced_tool_reports_tool_use_despite_stop_finish(self):
        # A forced tool_choice returns finish_reason "stop" on the live API
        # even though the message carries tool_calls; present tool calls win.
        client = _build_openai_client()
        client.client.chat.completions.create.return_value = _chat_response(
            _chat_message(
                tool_calls=[_chat_tool_call("call_3", "get_weather", '{"city": "SF"}')]
            ),
            finish_reason="stop",
        )
        turn = client.send_conversation(
            "sys",
            [{"role": "user", "content": "weather"}],
            tools=[WEATHER_TOOL],
            tool_choice="get_weather",
        )
        assert turn.finish_reason is AIFinishReason.TOOL_USE

    def test_structured_output_raw_schema(self):
        client = _build_openai_client()
        payload = {"nodes": [{"kind": "task"}]}
        client.client.chat.completions.create.return_value = _chat_response(
            _chat_message(content=json.dumps(payload))
        )
        result = client.send_structured_output("Compile.", response_schema=GRAPH_SCHEMA)
        assert result.data == payload
        assert result.finish_reason is AIFinishReason.COMPLETE
        kwargs = client.client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"]["type"] == "json_schema"
        assert kwargs["response_format"]["json_schema"]["schema"] == GRAPH_SCHEMA

    def test_structured_length_and_refusal(self):
        client = _build_openai_client()
        client.client.chat.completions.create.return_value = _chat_response(
            _chat_message(content='{"nodes": ['), finish_reason="length"
        )
        result = client.send_structured_output("Compile.", response_schema=GRAPH_SCHEMA)
        assert result.finish_reason is AIFinishReason.LENGTH
        assert result.data is None

        client.client.chat.completions.create.return_value = _chat_response(
            _chat_message(content=None, refusal="cannot help"), finish_reason="stop"
        )
        result = client.send_structured_output("Compile.", response_schema=GRAPH_SCHEMA)
        assert result.finish_reason is AIFinishReason.REFUSAL
        assert result.data is None

    def test_send_prompt_budget_and_timeout(self):
        client = _build_openai_client()
        client.client.with_options.return_value = client.client
        client.client.chat.completions.create.return_value = _chat_response(
            _chat_message(content="ok"), finish_reason="length"
        )
        text = client.send_prompt(
            "hi", max_response_tokens=500, request_timeout_seconds=15.0
        )
        # With an explicit budget the auto-continue-on-length loop is off.
        assert text == "ok"
        client.client.with_options.assert_called_once_with(timeout=15.0)
        kwargs = client.client.chat.completions.create.call_args.kwargs
        assert kwargs["max_completion_tokens"] == 500

    @pytest.mark.asyncio
    async def test_async_variants(self):
        client = _build_openai_client()
        async_client = Mock()
        async_client.chat.completions.create = AsyncMock(
            return_value=_chat_response(_chat_message(content="async ok"))
        )
        async_client.with_options.return_value = async_client
        client._async_client = async_client

        text = await client.asend_prompt("hi", max_response_tokens=64)
        assert text == "async ok"
        turn = await client.asend_conversation(
            "sys", [{"role": "user", "content": "hi"}]
        )
        assert turn.text == "async ok"

    def test_constructor_retry_policy_none(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("ai_api_unified.ai_openai_base.OpenAI") as mock_openai_cls:
                AiOpenAICompletions(model="gpt-4o-mini", retry_policy="none")
        assert mock_openai_cls.call_args.kwargs["max_retries"] == 0

    def test_status_error_wrapped(self):
        import httpx

        client = _build_openai_client()
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(429, request=request)
        client.client.chat.completions.create.side_effect = openai.APIStatusError(
            "rate limited", response=response, body=None
        )
        with pytest.raises(AiProviderRequestError) as exc_info:
            client.send_conversation("sys", [{"role": "user", "content": "hi"}])
        assert exc_info.value.status_code == 429
        assert exc_info.value.provider_engine == "openai"


# ── OpenAI Responses engine ─────────────────────────────────────────────────


def _build_responses_client() -> AiOpenAIResponsesCompletions:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        client = AiOpenAIResponsesCompletions(model="gpt-4o-mini")
    client.client = Mock()
    return client


def _responses_usage() -> Mock:
    return Mock(
        input_tokens=20,
        output_tokens=8,
        total_tokens=28,
        input_tokens_details=Mock(cached_tokens=None),
    )


def _function_call_item(call_id: str, name: str, arguments: str) -> Mock:
    item = Mock(spec=["type", "call_id", "name", "arguments"])
    item.type = "function_call"
    item.call_id = call_id
    item.name = name
    item.arguments = arguments
    return item


class TestResponsesConversation:
    def test_tool_turn_and_replay(self):
        client = _build_responses_client()
        response = Mock(
            output=[_function_call_item("fc_1", "get_weather", '{"city": "LA"}')],
            output_text="",
            status="completed",
            usage=_responses_usage(),
        )
        client.client.responses.create.return_value = response

        messages = [{"role": "user", "content": "Weather in LA?"}]
        turn = client.send_conversation(
            "sys", messages, tools=[WEATHER_TOOL], tool_choice="get_weather"
        )
        assert turn.finish_reason is AIFinishReason.TOOL_USE
        assert turn.tool_calls[0].id == "fc_1"
        assert turn.usage.input_tokens == 20

        kwargs = client.client.responses.create.call_args.kwargs
        assert kwargs["tools"][0] == {
            "type": "function",
            "name": "get_weather",
            "description": WEATHER_TOOL.description,
            "parameters": WEATHER_TOOL.input_schema,
            "strict": True,
        }
        assert kwargs["tool_choice"] == {"type": "function", "name": "get_weather"}

        # Replay extends the input item list, then appends the tool output.
        client.extend_messages_with_turn(messages, turn)
        assert messages[-1]["type"] == "function_call"
        tool_result = client.build_tool_result_message(
            tool_call_id="fc_1", result={"temp_f": 70}, is_error=False
        )
        assert tool_result == {
            "type": "function_call_output",
            "call_id": "fc_1",
            "output": json.dumps({"temp_f": 70}),
        }

    def test_structured_output_format_and_length(self):
        client = _build_responses_client()
        payload = {"nodes": []}
        client.client.responses.create.return_value = Mock(
            output=[],
            output_text=json.dumps(payload),
            status="completed",
            usage=_responses_usage(),
        )
        result = client.send_structured_output("Compile.", response_schema=GRAPH_SCHEMA)
        assert result.data == payload
        kwargs = client.client.responses.create.call_args.kwargs
        assert kwargs["text"]["format"]["type"] == "json_schema"
        assert kwargs["text"]["format"]["schema"] == GRAPH_SCHEMA

        client.client.responses.create.return_value = Mock(
            output=[],
            output_text='{"nodes"',
            status="incomplete",
            incomplete_details=Mock(reason="max_output_tokens"),
            usage=_responses_usage(),
        )
        result = client.send_structured_output("Compile.", response_schema=GRAPH_SCHEMA)
        assert result.finish_reason is AIFinishReason.LENGTH
        assert result.data is None

    @pytest.mark.asyncio
    async def test_async_conversation(self):
        client = _build_responses_client()
        async_client = Mock()
        async_client.responses.create = AsyncMock(
            return_value=Mock(
                output=[],
                output_text="done",
                status="completed",
                usage=_responses_usage(),
            )
        )
        async_client.with_options.return_value = async_client
        client._async_client = async_client
        turn = await client.asend_conversation(
            "sys", [{"role": "user", "content": "hi"}]
        )
        assert turn.finish_reason is AIFinishReason.COMPLETE
        assert turn.text == "done"


# ── Google Gemini engine ────────────────────────────────────────────────────

genai_module = pytest.importorskip("google.genai")

from ai_api_unified.completions.ai_google_gemini_completions import (  # noqa: E402
    GoogleGeminiCompletions,
)


def _build_gemini_client(mock_client: Mock, **kwargs) -> GoogleGeminiCompletions:
    with patch.object(
        GoogleGeminiCompletions,
        "_initialize_client",
        lambda self: setattr(self, "client", mock_client),
    ):
        return GoogleGeminiCompletions(model="gemini-2.5-flash", **kwargs)


def _gemini_usage() -> Mock:
    return Mock(
        prompt_token_count=30,
        candidates_token_count=12,
        total_token_count=42,
        cached_content_token_count=None,
    )


def _gemini_function_call_part(name: str, args: dict) -> Mock:
    function_call = Mock(spec=["name", "args"])
    function_call.name = name
    function_call.args = args
    part = Mock(spec=["function_call", "text", "model_dump"])
    part.function_call = function_call
    part.text = None
    part.model_dump = Mock(side_effect=TypeError("test double"))
    return part


def _gemini_text_part(text: str) -> Mock:
    part = Mock(spec=["function_call", "text", "model_dump"])
    part.function_call = None
    part.text = text
    part.model_dump = Mock(side_effect=TypeError("test double"))
    return part


def _gemini_response(parts: list, finish_reason: str = "FinishReason.STOP") -> Mock:
    return Mock(
        candidates=[Mock(content=Mock(parts=parts), finish_reason=finish_reason)],
        usage_metadata=_gemini_usage(),
        text="".join(getattr(p, "text", None) or "" for p in parts),
    )


class TestGeminiConversation:
    def test_tool_turn_forced_and_replay(self):
        mock_client = Mock()
        client = _build_gemini_client(mock_client)
        mock_client.models.generate_content.return_value = _gemini_response(
            [_gemini_function_call_part("get_weather", {"city": "NYC"})],
            finish_reason="FinishReason.STOP",
        )
        messages = [{"role": "user", "parts": [{"text": "Weather in NYC?"}]}]
        turn = client.send_conversation(
            "sys", messages, tools=[WEATHER_TOOL], tool_choice="get_weather"
        )
        assert turn.finish_reason is AIFinishReason.TOOL_USE
        # Gemini tool-call ids are the function name.
        assert turn.tool_calls[0].id == "get_weather"
        assert turn.tool_calls[0].input == {"city": "NYC"}
        assert turn.usage.input_tokens == 30

        config = mock_client.models.generate_content.call_args.kwargs["config"]
        declaration = config.tools[0].function_declarations[0]
        assert declaration.name == "get_weather"
        assert declaration.parameters_json_schema == WEATHER_TOOL.input_schema
        assert config.tool_config.function_calling_config.allowed_function_names == [
            "get_weather"
        ]

        client.extend_messages_with_turn(messages, turn)
        assert messages[-1]["role"] == "model"
        assert messages[-1]["parts"][0]["function_call"]["name"] == "get_weather"
        tool_result = client.build_tool_result_message(
            tool_call_id="get_weather", result={"temp_f": 55}, is_error=False
        )
        assert tool_result["parts"][0]["function_response"]["name"] == "get_weather"

    def test_structured_output_json_schema_and_length(self):
        mock_client = Mock()
        client = _build_gemini_client(mock_client)
        payload = {"nodes": []}
        mock_client.models.generate_content.return_value = _gemini_response(
            [_gemini_text_part(json.dumps(payload))]
        )
        result = client.send_structured_output("Compile.", response_schema=GRAPH_SCHEMA)
        assert result.data == payload
        config = mock_client.models.generate_content.call_args.kwargs["config"]
        assert config.response_json_schema == GRAPH_SCHEMA
        assert config.response_mime_type == "application/json"

        mock_client.models.generate_content.return_value = _gemini_response(
            [_gemini_text_part('{"nodes"')],
            finish_reason="FinishReason.MAX_TOKENS",
        )
        result = client.send_structured_output("Compile.", response_schema=GRAPH_SCHEMA)
        assert result.finish_reason is AIFinishReason.LENGTH
        assert result.data is None

    @pytest.mark.asyncio
    async def test_async_prompt(self):
        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=_gemini_response([_gemini_text_part("async ok")])
        )
        client = _build_gemini_client(mock_client)
        text = await client.asend_prompt("hi", max_response_tokens=128)
        assert text == "async ok"

    def test_retry_policy_none_single_attempt(self):
        mock_client = Mock()
        client = _build_gemini_client(mock_client, retry_policy="none")
        assert client._effective_max_retries() == 0
        assert client._effective_max_retries("none") == 0
        client_default = _build_gemini_client(Mock())
        assert client_default._effective_max_retries() is None
        assert client_default._effective_max_retries("none") == 0

    def test_capability_flags(self):
        client = _build_gemini_client(Mock())
        assert client.capabilities.supports_tool_use is True
        assert client.capabilities.supports_structured_output is True
        assert client.capabilities.supports_async is True


# ── Bedrock engine ──────────────────────────────────────────────────────────

pytest.importorskip("boto3")

from ai_api_unified.completions.ai_bedrock_completions import (  # noqa: E402
    AiBedrockCompletions,
)


def _build_bedrock_client(model: str = "amazon.nova-lite-v1:0", **kwargs):
    with patch("ai_api_unified.ai_bedrock_base.boto3"):
        client = AiBedrockCompletions(model=model, **kwargs)
    client.client = Mock()
    client.backoff_delays = [0.0]
    client._sleep_with_backoff = lambda base_delay: None
    return client


def _converse_response(content: list, stop_reason: str = "end_turn") -> dict:
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": stop_reason,
        "usage": {"inputTokens": 15, "outputTokens": 6, "totalTokens": 21},
    }


class TestBedrockConversation:
    def test_tool_turn_forced_and_replay(self):
        client = _build_bedrock_client()
        client.client.converse.return_value = _converse_response(
            [
                {
                    "toolUse": {
                        "toolUseId": "tu_1",
                        "name": "get_weather",
                        "input": {"city": "Denver"},
                    }
                }
            ],
            stop_reason="tool_use",
        )
        messages = [{"role": "user", "content": [{"text": "Weather in Denver?"}]}]
        turn = client.send_conversation(
            "sys", messages, tools=[WEATHER_TOOL], tool_choice="get_weather"
        )
        assert turn.finish_reason is AIFinishReason.TOOL_USE
        assert turn.tool_calls[0].id == "tu_1"
        assert turn.usage.input_tokens == 15

        kwargs = client.client.converse.call_args.kwargs
        tool_spec = kwargs["toolConfig"]["tools"][0]["toolSpec"]
        assert tool_spec["name"] == "get_weather"
        assert tool_spec["inputSchema"] == {"json": WEATHER_TOOL.input_schema}
        assert kwargs["toolConfig"]["toolChoice"] == {"tool": {"name": "get_weather"}}

        client.extend_messages_with_turn(messages, turn)
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"][0]["toolUse"]["toolUseId"] == "tu_1"
        tool_result = client.build_tool_result_message(
            tool_call_id="tu_1", result={"temp_f": 40}, is_error=True
        )
        assert tool_result["content"][0]["toolResult"]["status"] == "error"
        assert tool_result["content"][0]["toolResult"]["content"] == [
            {"json": {"temp_f": 40}}
        ]

    def test_finish_reason_mapping(self):
        client = _build_bedrock_client()
        for stop_reason, expected in (
            ("end_turn", AIFinishReason.COMPLETE),
            ("max_tokens", AIFinishReason.LENGTH),
            ("guardrail_intervened", AIFinishReason.REFUSAL),
        ):
            client.client.converse.return_value = _converse_response(
                [{"text": "x"}], stop_reason=stop_reason
            )
            turn = client.send_conversation(
                "sys", [{"role": "user", "content": [{"text": "hi"}]}]
            )
            assert turn.finish_reason is expected

    def test_structured_output_gated_per_model(self):
        # Nova is not in AWS's structured-outputs supported list.
        nova_client = _build_bedrock_client()
        with pytest.raises(AiProviderCapabilityUnsupportedError):
            nova_client.send_structured_output("Compile.", response_schema=GRAPH_SCHEMA)

        claude_client = _build_bedrock_client(model="us.anthropic.claude-opus-4-6-v1:0")
        payload = {"nodes": []}
        claude_client.client.converse.return_value = _converse_response(
            [{"text": json.dumps(payload)}]
        )
        result = claude_client.send_structured_output(
            "Compile.", response_schema=GRAPH_SCHEMA
        )
        assert result.data == payload
        kwargs = claude_client.client.converse.call_args.kwargs
        text_format = kwargs["outputConfig"]["textFormat"]
        assert text_format["type"] == "json_schema"
        assert text_format["structure"]["jsonSchema"]["schema"] == GRAPH_SCHEMA

    def test_timeout_and_async_stay_unimplemented(self):
        client = _build_bedrock_client()
        with pytest.raises(
            AiProviderCapabilityUnsupportedError, match="request_timeout_seconds"
        ):
            client.send_conversation(
                "sys",
                [{"role": "user", "content": [{"text": "hi"}]}],
                request_timeout_seconds=5.0,
            )
        assert client.capabilities.supports_async is False

    @pytest.mark.asyncio
    async def test_async_gate_raises(self):
        client = _build_bedrock_client()
        with pytest.raises(AiProviderCapabilityUnsupportedError):
            await client.asend_prompt("hi")

    def test_client_error_wrapped_with_status(self):
        from botocore.exceptions import ClientError as BotoClientError

        client = _build_bedrock_client()
        client.client.converse.side_effect = BotoClientError(
            {
                "Error": {"Code": "ThrottlingException", "Message": "slow down"},
                "ResponseMetadata": {"HTTPStatusCode": 429},
            },
            "Converse",
        )
        with pytest.raises(AiProviderRequestError) as exc_info:
            client.send_conversation(
                "sys", [{"role": "user", "content": [{"text": "hi"}]}]
            )
        assert exc_info.value.status_code == 429
        assert exc_info.value.provider_engine == "bedrock"

    def test_retry_policy_none_collapses_schedule(self):
        with patch("ai_api_unified.ai_bedrock_base.boto3"):
            client = AiBedrockCompletions(
                model="amazon.nova-lite-v1:0", retry_policy="none"
            )
        assert client.backoff_delays == [0.0]

    def test_send_prompt_maps_max_tokens(self):
        client = _build_bedrock_client()
        client.client.converse.return_value = _converse_response([{"text": "ok"}])
        client.send_prompt("hi", max_response_tokens=2048)
        kwargs = client.client.converse.call_args.kwargs
        assert kwargs["inferenceConfig"]["maxTokens"] == 2048
