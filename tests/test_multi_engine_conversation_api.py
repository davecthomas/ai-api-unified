# test_multi_engine_conversation_api.py
"""
Mocked tests for the conversation, structured-output, async, retry, and
model-listing features on the openai, openai-responses, google-gemini, and
bedrock engines
(claude engine coverage lives in test_completions_conversation_api.py).

Transport faking follows each engine's established repo pattern: construct
the real client with stubbed credentials, then replace the SDK client
attribute with Mock objects mimicking that SDK's object graph.
"""

import json
import os
import time
from typing import Any
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
    LIST_MODELS_CACHE_TTL_SECONDS,
    LIST_MODELS_FAILURE_TTL_SECONDS,
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


class TestInterfaceMessageShape:
    """The {role, content} shape ai_base documents reaches every surface."""

    DOCUMENTED: list[dict[str, Any]] = [{"role": "user", "content": "Ready?"}]

    def test_documented_shape_reaches_gemini_as_parts(self):
        mock_client = Mock()
        client = _build_gemini_client(mock_client)
        mock_client.models.generate_content.return_value = _gemini_response(
            [_gemini_text_part("Yes")]
        )
        client.send_conversation("sys", list(self.DOCUMENTED))
        contents = mock_client.models.generate_content.call_args.kwargs["contents"]
        assert contents == [{"role": "user", "parts": [{"text": "Ready?"}]}]

    def test_assistant_role_becomes_model(self):
        mock_client = Mock()
        client = _build_gemini_client(mock_client)
        mock_client.models.generate_content.return_value = _gemini_response(
            [_gemini_text_part("ok")]
        )
        client.send_conversation(
            "sys",
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        )
        contents = mock_client.models.generate_content.call_args.kwargs["contents"]
        assert [c["role"] for c in contents] == ["user", "model"]
        assert contents[1]["parts"] == [{"text": "hello"}]

    def test_gemini_shaped_entries_pass_through(self):
        """A replayed turn and a tool result must survive untranslated."""
        mock_client = Mock()
        client = _build_gemini_client(mock_client)
        mock_client.models.generate_content.return_value = _gemini_response(
            [_gemini_text_part("ok")]
        )
        replayed = {"role": "model", "parts": [{"text": "prior turn"}]}
        tool_result = client.build_tool_result_message(
            tool_call_id="get_weather", result={"temp_f": 55}, is_error=False
        )
        messages = [
            {"role": "user", "content": "hi"},
            dict(replayed),
            tool_result,
        ]
        client.send_conversation("sys", messages)
        contents = mock_client.models.generate_content.call_args.kwargs["contents"]
        assert contents[0] == {"role": "user", "parts": [{"text": "hi"}]}
        assert contents[1] == replayed
        # The helper's own object reaches the SDK, not a re-wrapped copy.
        assert contents[2] is tool_result

    def test_foreign_engine_helper_item_does_not_pass_through(self):
        """An entry with neither content nor parts is another engine's shape."""
        client = _build_gemini_client(Mock())
        # openai-responses helper output: no "content", no "parts".
        with pytest.raises(ValueError, match="build_tool_result_message"):
            client.send_conversation(
                "sys",
                [{"type": "function_call_output", "call_id": "c1", "output": "55"}],
            )

    def test_mixed_history_from_extend_messages_with_turn(self):
        """The two-shape history the interface produces is accepted."""
        mock_client = Mock()
        client = _build_gemini_client(mock_client)
        mock_client.models.generate_content.return_value = _gemini_response(
            [_gemini_function_call_part("get_weather", {"city": "NYC"})]
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": "Weather?"}]
        turn = client.send_conversation("sys", messages, tools=[WEATHER_TOOL])
        client.extend_messages_with_turn(messages, turn)
        messages.append(
            client.build_tool_result_message(
                tool_call_id="get_weather", result={"temp_f": 55}, is_error=False
            )
        )
        mock_client.models.generate_content.return_value = _gemini_response(
            [_gemini_text_part("55F")]
        )
        client.send_conversation("sys", messages, tools=[WEATHER_TOOL])
        contents = mock_client.models.generate_content.call_args.kwargs["contents"]
        assert contents[0] == {"role": "user", "parts": [{"text": "Weather?"}]}
        # Both helper outputs arrive untouched, not re-wrapped as text parts.
        assert contents[1] is messages[1]
        assert contents[2] is messages[2]
        assert len(contents) == 3

    def test_structured_output_accepts_documented_shape(self):
        mock_client = Mock()
        client = _build_gemini_client(mock_client)
        mock_client.models.generate_content.return_value = _gemini_response(
            [_gemini_text_part(json.dumps({"nodes": []}))]
        )
        result = client.send_structured_output(
            messages=list(self.DOCUMENTED), response_schema=GRAPH_SCHEMA
        )
        assert result.data == {"nodes": []}
        contents = mock_client.models.generate_content.call_args.kwargs["contents"]
        assert contents[0] == {"role": "user", "parts": [{"text": "Ready?"}]}

    @pytest.mark.asyncio
    async def test_async_conversation_accepts_documented_shape(self):
        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=_gemini_response([_gemini_text_part("Yes")])
        )
        client = _build_gemini_client(mock_client)
        await client.asend_conversation("sys", list(self.DOCUMENTED))
        contents = mock_client.aio.models.generate_content.call_args.kwargs["contents"]
        assert contents == [{"role": "user", "parts": [{"text": "Ready?"}]}]

    @pytest.mark.asyncio
    async def test_async_structured_output_accepts_documented_shape(self):
        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=_gemini_response(
                [_gemini_text_part(json.dumps({"nodes": []}))]
            )
        )
        client = _build_gemini_client(mock_client)
        result = await client.asend_structured_output(
            messages=list(self.DOCUMENTED), response_schema=GRAPH_SCHEMA
        )
        assert result.data == {"nodes": []}
        contents = mock_client.aio.models.generate_content.call_args.kwargs["contents"]
        assert contents[0] == {"role": "user", "parts": [{"text": "Ready?"}]}

    def test_untranslatable_content_names_the_helpers(self):
        """Another engine's raw_content blocks cannot be guessed at."""
        client = _build_gemini_client(Mock())
        with pytest.raises(ValueError, match="extend_messages_with_turn"):
            client.send_conversation(
                "sys", [{"role": "assistant", "content": [{"type": "text"}]}]
            )

    def test_base_default_leaves_messages_untouched(self):
        """Engines whose wire shape is already {role, content} are unaffected."""
        client = _build_openai_client()
        assert client._normalize_messages(self.DOCUMENTED) is self.DOCUMENTED
        assert client._normalize_messages(None) is None


class TestGeminiModelListing:
    def test_live_listing_intersects_specs_in_spec_order(self):
        mock_client = Mock()
        client = _build_gemini_client(mock_client)
        mock_client.models.list.return_value = [
            _gemini_catalogue_model("models/gemini-2.5-flash", ["generateContent"]),
            _gemini_catalogue_model(
                "publishers/google/models/gemini-2.5-pro", ["generateContent"]
            ),
            _gemini_catalogue_model("models/gemini-3.5-flash", ["generateContent"]),
            _gemini_catalogue_model("models/gemini-embedding-001", ["embedContent"]),
            _gemini_catalogue_model("models/gemini-9.9-unknown", ["generateContent"]),
        ]
        list_names = client.list_model_names
        # Spec order preserved; embedContent-only and non-spec names dropped;
        # spec entries absent from the catalogue (the 2.0 family) dropped.
        assert list_names == ["gemini-3.5-flash", "gemini-2.5-pro", "gemini-2.5-flash"]

    def test_catalogue_entry_without_actions_is_kept(self):
        mock_client = Mock()
        client = _build_gemini_client(mock_client)
        mock_client.models.list.return_value = [
            _gemini_catalogue_model("models/gemini-2.5-flash", None),
        ]
        assert client.list_model_names == ["gemini-2.5-flash"]

    def test_listing_failure_falls_back_to_static_specs(self, caplog):
        from ai_api_unified.completions.ai_google_gemini_completions import (
            GEMINI_MODEL_SPECS,
        )

        mock_client = Mock()
        client = _build_gemini_client(mock_client)
        mock_client.models.list.side_effect = RuntimeError("no network")
        with caplog.at_level("WARNING"):
            list_names = client.list_model_names
        assert list_names == list(GEMINI_MODEL_SPECS.keys())
        assert "falling back" in caplog.text

    def test_empty_intersection_falls_back_to_static_specs(self, caplog):
        from ai_api_unified.completions.ai_google_gemini_completions import (
            GEMINI_MODEL_SPECS,
        )

        mock_client = Mock()
        client = _build_gemini_client(mock_client)
        mock_client.models.list.return_value = [
            _gemini_catalogue_model("models/some-unrelated-model", ["generateContent"]),
        ]
        with caplog.at_level("WARNING"):
            list_names = client.list_model_names
        assert list_names == list(GEMINI_MODEL_SPECS.keys())
        # Failing open silently would present uncallable models as available.
        assert "named none of the" in caplog.text
        # A catalogue that answers but shares no names is a stable naming
        # mismatch, not a transient fault, so it holds the full window rather
        # than re-querying every minute for the life of the process.
        _, float_expires_at, _ = client._list_model_names_cache
        float_window = float_expires_at - time.monotonic()
        assert float_window > LIST_MODELS_FAILURE_TTL_SECONDS
        assert float_window == pytest.approx(LIST_MODELS_CACHE_TTL_SECONDS, abs=5.0)

    def test_successful_listing_is_cached_per_client(self):
        mock_client = Mock()
        client = _build_gemini_client(mock_client)
        mock_client.models.list.return_value = [
            _gemini_catalogue_model("models/gemini-2.5-flash", ["generateContent"]),
        ]
        first = client.list_model_names
        second = client.list_model_names
        assert first == second == ["gemini-2.5-flash"]
        assert mock_client.models.list.call_count == 1

    def test_failed_listing_is_cached_only_for_the_failure_window(self):
        from ai_api_unified.completions.ai_google_gemini_completions import (
            GEMINI_MODEL_SPECS,
        )

        mock_client = Mock()
        client = _build_gemini_client(mock_client)
        mock_client.models.list.side_effect = RuntimeError("no network")
        assert client.list_model_names == list(GEMINI_MODEL_SPECS.keys())
        # A provider that cannot answer costs one round trip per window, not
        # one per read.
        assert client.list_model_names == list(GEMINI_MODEL_SPECS.keys())
        assert mock_client.models.list.call_count == 1

        # Once the short failure window expires, a recovered provider is used.
        str_model, _, list_cached = client._list_model_names_cache
        client._list_model_names_cache = (str_model, 0.0, list_cached)
        mock_client.models.list.side_effect = None
        mock_client.models.list.return_value = [
            _gemini_catalogue_model("models/gemini-2.5-flash", ["generateContent"]),
        ]
        assert client.list_model_names == ["gemini-2.5-flash"]

    def test_expired_success_cache_is_refreshed(self):
        mock_client = Mock()
        client = _build_gemini_client(mock_client)
        mock_client.models.list.return_value = [
            _gemini_catalogue_model("models/gemini-2.5-flash", ["generateContent"]),
        ]
        assert client.list_model_names == ["gemini-2.5-flash"]
        str_model, _, list_cached = client._list_model_names_cache
        client._list_model_names_cache = (str_model, 0.0, list_cached)
        # Google publishes models over time, so a cached list cannot be final.
        mock_client.models.list.return_value = [
            _gemini_catalogue_model("models/gemini-2.5-flash", ["generateContent"]),
            _gemini_catalogue_model("models/gemini-2.5-pro", ["generateContent"]),
        ]
        assert client.list_model_names == ["gemini-2.5-pro", "gemini-2.5-flash"]

    def test_cache_is_keyed_on_the_configured_model(self):
        mock_client = Mock()
        client = _build_gemini_client(mock_client)
        mock_client.models.list.return_value = [
            _gemini_catalogue_model("models/gemini-2.5-flash", ["generateContent"]),
        ]
        assert client.list_model_names == ["gemini-2.5-flash"]
        # Repointing the client must not serve the previous model's answer.
        client.completions_model = "gemini-not-in-specs"  # outside the spec dict
        assert client.list_model_names == ["gemini-2.5-flash", "gemini-not-in-specs"]

    def test_configured_model_is_always_listed(self):
        mock_client = Mock()
        client = _build_gemini_client(mock_client)
        # A model outside the spec dict, which __init__ rewrites in practice.
        # The property must not depend on that distant guard.
        client.completions_model = "gemini-not-in-specs"
        mock_client.models.list.return_value = [
            _gemini_catalogue_model("models/gemini-2.5-flash", ["generateContent"]),
        ]
        list_names = client.list_model_names
        assert list_names == ["gemini-2.5-flash", "gemini-not-in-specs"]
        # The engine never omits the model it is configured to call.
        assert client.model_name in list_names

    def test_listing_goes_through_the_retry_wrapper(self):
        mock_client = Mock()
        client = _build_gemini_client(mock_client)
        mock_client.models.list.return_value = [
            _gemini_catalogue_model("models/gemini-2.5-flash", ["generateContent"]),
        ]
        with patch.object(
            client,
            "_retry_with_exponential_backoff",
            side_effect=lambda operation, **kwargs: operation(),
        ) as mock_retry:
            assert client.list_model_names == ["gemini-2.5-flash"]
        # A transient 429 must not silently downgrade to the static list.
        assert mock_retry.call_count == 1


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


DOCUMENTED_MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "Ready?"}]


def _outbound_gemini() -> Any:
    mock_client = Mock()
    client = _build_gemini_client(mock_client)
    mock_client.models.generate_content.return_value = _gemini_response(
        [_gemini_text_part("Yes")]
    )
    client.send_conversation("sys", list(DOCUMENTED_MESSAGES))
    return mock_client.models.generate_content.call_args.kwargs["contents"]


def _outbound_bedrock() -> Any:
    client = _build_bedrock_client()
    client.client.converse.return_value = _converse_response([{"text": "Yes"}])
    client.send_conversation("sys", list(DOCUMENTED_MESSAGES))
    return client.client.converse.call_args.kwargs["messages"]


def _outbound_openai_chat() -> Any:
    client = _build_openai_client()
    client.client.chat.completions.create.return_value = _chat_response(
        _chat_message(content="Yes")
    )
    client.send_conversation("sys", list(DOCUMENTED_MESSAGES))
    return client.client.chat.completions.create.call_args.kwargs["messages"]


def _outbound_openai_responses() -> Any:
    client = _build_responses_client()
    client.client.responses.create.return_value = Mock(
        output=[],
        output_text="Yes",
        status="completed",
        usage=_responses_usage(),
    )
    client.send_conversation("sys", list(DOCUMENTED_MESSAGES))
    return client.client.responses.create.call_args.kwargs["input"]


def _outbound_anthropic() -> Any:
    from ai_api_unified.completions.ai_anthropic_completions import (
        AiAnthropicCompletions,
    )

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        client = AiAnthropicCompletions(model="claude-opus-4-8")
    client.client = Mock()
    client.client.messages.create.return_value = Mock(
        content=[Mock(spec=["type", "text"], type="text", text="Yes")],
        stop_reason="end_turn",
        usage=Mock(input_tokens=10, output_tokens=5, cache_read_input_tokens=None),
    )
    client.send_conversation("sys", list(DOCUMENTED_MESSAGES))
    return client.client.messages.create.call_args.kwargs["messages"]


def _assert_gemini_valid(outbound: Any) -> None:
    """google-genai validates in-process; the real model is the oracle."""
    genai_module.types._GenerateContentParameters(model="m", contents=outbound)


def _assert_bedrock_valid(outbound: Any) -> None:
    """botocore validates Converse params client-side; use its own validator."""
    import botocore.session
    from botocore.validate import ParamValidator

    operation = (
        botocore.session.get_session()
        .get_service_model("bedrock-runtime")
        .operation_model("Converse")
    )
    report = ParamValidator().validate(
        {"modelId": "m", "messages": outbound, "system": [{"text": "s"}]},
        operation.input_shape,
    )
    assert not report.generate_report(), report.generate_report()


def _assert_string_content_preserved(outbound: Any) -> None:
    """These SDKs accept str content, so the identity default is correct.

    The engine may add its own entries (openai carries the system prompt as a
    message), so the property is that the caller's message survives verbatim,
    not that it is the only entry.
    """
    assert DOCUMENTED_MESSAGES[0] in outbound
    for entry in outbound:
        assert isinstance(entry["content"], str), entry


class TestBedrockMessageShape:
    """Bedrock's half of the class: Converse needs content blocks, not a string."""

    def test_documented_shape_becomes_content_blocks(self):
        client = _build_bedrock_client()
        client.client.converse.return_value = _converse_response([{"text": "Yes"}])
        client.send_conversation("sys", [{"role": "user", "content": "Ready?"}])
        sent = client.client.converse.call_args.kwargs["messages"]
        assert sent == [{"role": "user", "content": [{"text": "Ready?"}]}]

    def test_assistant_role_is_unchanged(self):
        """Converse already names the assistant role "assistant"."""
        client = _build_bedrock_client()
        client.client.converse.return_value = _converse_response([{"text": "ok"}])
        client.send_conversation(
            "sys",
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        )
        sent = client.client.converse.call_args.kwargs["messages"]
        assert [m["role"] for m in sent] == ["user", "assistant"]
        assert sent[1]["content"] == [{"text": "hello"}]

    def test_converse_shaped_helper_output_passes_through(self):
        client = _build_bedrock_client()
        client.client.converse.return_value = _converse_response([{"text": "ok"}])
        messages: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
        turn = client.send_conversation("sys", messages, tools=[WEATHER_TOOL])
        # Build the replayed turn with the helper rather than by hand, so the
        # test cannot drift from what the engine actually appends.
        client.extend_messages_with_turn(messages, turn)
        messages.append(
            client.build_tool_result_message(
                tool_call_id="tu_1", result={"temp_f": 55}, is_error=False
            )
        )
        client.client.converse.return_value = _converse_response([{"text": "55F"}])
        client.send_conversation("sys", messages, tools=[WEATHER_TOOL])
        sent = client.client.converse.call_args.kwargs["messages"]
        assert sent[0] == {"role": "user", "content": [{"text": "hi"}]}
        # Helper output arrives untouched, not re-wrapped as a text block.
        assert sent[1] is messages[1]
        assert sent[2] is messages[2]
        assert len(sent) == 3

    def test_foreign_engine_content_names_the_helpers(self):
        """Gemini-shaped history must not be forwarded to botocore."""
        client = _build_bedrock_client()
        with pytest.raises(ValueError, match="build_tool_result_message"):
            client.send_conversation(
                "sys", [{"role": "model", "parts": [{"text": "hi"}]}]
            )

    def test_type_tagged_block_list_does_not_pass_through(self):
        """Anthropic/OpenAI block lists are not Converse blocks."""
        client = _build_bedrock_client()
        # botocore rejects this shape; passing it through would reproduce the
        # in-process failure this override exists to prevent.
        with pytest.raises(ValueError, match="build_tool_result_message"):
            client.send_conversation(
                "sys",
                [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}],
            )

    def test_empty_content_list_does_not_pass_through(self):
        client = _build_bedrock_client()
        with pytest.raises(ValueError, match="build_tool_result_message"):
            client.send_conversation("sys", [{"role": "user", "content": []}])

    def test_structured_output_surface_also_wraps_content(self):
        """#49's surface list includes structured output, not just conversation."""
        # Structured output is gated to the Claude-family Bedrock models.
        client = _build_bedrock_client(model="us.anthropic.claude-opus-4-6-v1:0")
        client.client.converse.return_value = _converse_response(
            [{"text": json.dumps({"nodes": []})}]
        )
        client.send_structured_output(
            messages=[{"role": "user", "content": "Ready?"}],
            response_schema=GRAPH_SCHEMA,
        )
        sent = client.client.converse.call_args.kwargs["messages"]
        assert sent[0] == {"role": "user", "content": [{"text": "Ready?"}]}

    def test_block_keys_come_from_the_installed_service_model(self):
        """The allowlist tracks the SDK, so a botocore upgrade cannot stale it.

        Asserting equality against a frozen literal broke on botocore 1.43.80,
        which added toolAddition and toolRemoval: the suite went red on
        upgrade, and the engine rejected blocks Converse accepts.
        """
        import botocore.session

        shape = (
            botocore.session.get_session()
            .get_service_model("bedrock-runtime")
            .shape_for("ContentBlock")
        )
        assert AiBedrockCompletions._converse_content_block_keys() == frozenset(
            shape.members.keys()
        )

    def test_fallback_is_a_subset_of_the_service_model(self):
        """The built-in list must name only real members, never invent one."""
        import botocore.session

        shape = (
            botocore.session.get_session()
            .get_service_model("bedrock-runtime")
            .shape_for("ContentBlock")
        )
        assert AiBedrockCompletions.CONVERSE_CONTENT_BLOCK_KEYS_FALLBACK <= frozenset(
            shape.members.keys()
        )

    def test_block_type_added_by_a_newer_sdk_passes_through(self):
        """A member absent from the fallback must still be accepted."""
        added = "someFutureBlockType"
        assert added not in AiBedrockCompletions.CONVERSE_CONTENT_BLOCK_KEYS_FALLBACK
        with patch.object(
            AiBedrockCompletions,
            "_FROZENSET_CONVERSE_BLOCK_KEYS",
            AiBedrockCompletions.CONVERSE_CONTENT_BLOCK_KEYS_FALLBACK | {added},
        ):
            client = _build_bedrock_client()
            client.client.converse.return_value = _converse_response([{"text": "ok"}])
            block = {"role": "user", "content": [{added: {"x": 1}}]}
            client.send_conversation("sys", [block])
            sent = client.client.converse.call_args.kwargs["messages"]
            assert sent[0] is block

    def test_unreadable_service_model_falls_back(self):
        """A botocore whose model cannot be read must not break sending."""
        with (
            patch.object(AiBedrockCompletions, "_FROZENSET_CONVERSE_BLOCK_KEYS", None),
            patch("botocore.session.get_session", side_effect=RuntimeError("no model")),
        ):
            assert (
                AiBedrockCompletions._converse_content_block_keys()
                == AiBedrockCompletions.CONVERSE_CONTENT_BLOCK_KEYS_FALLBACK
            )
        AiBedrockCompletions._FROZENSET_CONVERSE_BLOCK_KEYS = None


class TestDocumentedShapeAcrossProviders:
    """These five engines accept the {role, content} shape ai_base documents.

    The rows are explicit, so this catches a wire-shape drift in an engine
    listed below, not the arrival of a sixth one. Adding an engine means adding
    a row here; that belongs in the engine checklist rather than being implied
    by this test. Where the SDK ships a client-side validator, the outbound
    payload is checked against that validator rather than against a shape this
    test asserts by hand.
    """

    @pytest.mark.parametrize(
        "engine, outbound_fn, assert_fn",
        [
            ("google-gemini", _outbound_gemini, _assert_gemini_valid),
            ("bedrock", _outbound_bedrock, _assert_bedrock_valid),
            ("openai", _outbound_openai_chat, _assert_string_content_preserved),
            (
                "openai-responses",
                _outbound_openai_responses,
                _assert_string_content_preserved,
            ),
            ("anthropic", _outbound_anthropic, _assert_string_content_preserved),
        ],
    )
    def test_documented_shape_is_accepted(self, engine, outbound_fn, assert_fn):
        assert_fn(outbound_fn())


def _gemini_catalogue_model(name: str, actions: list[str] | None) -> Mock:
    model_metadata = Mock(spec=["name", "supported_actions"])
    model_metadata.name = name
    model_metadata.supported_actions = actions
    return model_metadata
