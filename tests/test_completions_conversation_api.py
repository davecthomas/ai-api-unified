# test_completions_conversation_api.py
"""
Mocked tests for the engine-agnostic conversation, structured-output, async,
retry-policy, and observability-tag features (claude engine first).

Transport faking follows the repo pattern: construct the real client with a
stubbed ANTHROPIC_API_KEY, then replace `.client` (and `._async_client`) with
Mock objects whose responses mimic the Anthropic SDK object graph.
"""

import json
import logging
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

anthropic = pytest.importorskip("anthropic")

from ai_api_unified.ai_base import (  # noqa: E402
    AICompletionsPromptParamsBase,
    AIFinishReason,
    AIStructuredPrompt,
    AITool,
    AIBaseCompletions,
)
from ai_api_unified.ai_provider_exceptions import (  # noqa: E402
    AiProviderCapabilityUnsupportedError,
    AiProviderConfigurationError,
    AiProviderRequestError,
)
from ai_api_unified.completions.ai_anthropic_completions import (  # noqa: E402
    AiAnthropicCompletions,
)
from ai_api_unified.middleware.middleware_config import (  # noqa: E402
    ObservabilitySettingsModel,
)
from ai_api_unified.middleware.observability import (  # noqa: E402
    COST_LOGGER_NAME,
    LoggerBackedObservabilityMiddleware,
)
from ai_api_unified.middleware.observability_runtime import (  # noqa: E402
    AiApiCallContextModel,
    AiApiCallResultSummaryModel,
    get_observability_context,
    reset_observability_context,
    set_observability_context,
)

from datetime import datetime, timezone  # noqa: E402


# ── Builders ────────────────────────────────────────────────────────────────


def _build_client(model: str = "claude-opus-4-8", **kwargs) -> AiAnthropicCompletions:
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        client = AiAnthropicCompletions(model=model, **kwargs)
    client.client = Mock()
    return client


def _usage(
    input_tokens: int = 10,
    output_tokens: int = 5,
    cached: int | None = None,
) -> Mock:
    return Mock(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cached,
    )


def _text_block(text: str) -> Mock:
    return Mock(spec=["type", "text"], type="text", text=text)


def _tool_use_block(block_id: str, name: str, tool_input: dict) -> Mock:
    block = Mock(spec=["type", "id", "name", "input"], type="tool_use", id=block_id)
    block.name = name
    block.input = tool_input
    return block


def _response(blocks: list, stop_reason: str = "end_turn", usage: Mock | None = None):
    return Mock(
        content=blocks,
        stop_reason=stop_reason,
        usage=usage if usage is not None else _usage(),
    )


def _weather_tool(strict: bool = False) -> AITool:
    return AITool(
        name="get_weather",
        description="Get current weather for a city.",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        strict=strict,
    )


class _NodeModel(AIStructuredPrompt):
    node_id: str
    kind: str

    @staticmethod
    def get_prompt() -> str:
        return ""


class _StubCompletions(AIBaseCompletions):
    """Minimal engine with default (all-False) capability flags for gating tests."""

    def __init__(self) -> None:
        super().__init__(model="stub-model")

    @property
    def list_model_names(self) -> list[str]:
        return ["stub-model"]

    @property
    def max_context_tokens(self) -> int:
        return 1000

    def send_prompt(self, prompt: str, **kwargs) -> str:
        return ""

    def strict_schema_prompt(
        self, prompt, response_model, max_response_tokens=2048, *, other_params=None
    ):
        raise NotImplementedError


RAW_GRAPH_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "kind": {"const": "task"},
                            "name": {"type": "string"},
                        },
                        "required": ["kind", "name"],
                    },
                    {
                        "type": "object",
                        "properties": {
                            "kind": {"const": "gate"},
                            "condition": {"type": "string"},
                        },
                        "required": ["kind", "condition"],
                    },
                ]
            },
        }
    },
    "required": ["nodes"],
}


# ── send_conversation: full tool-loop turn cycle ────────────────────────────


class TestSendConversation:
    def test_full_tool_loop_cycle(self):
        client = _build_client()
        tool = _weather_tool(strict=True)
        turn1_response = _response(
            blocks=[
                _text_block("Checking the weather."),
                _tool_use_block("toolu_1", "get_weather", {"city": "SF"}),
            ],
            stop_reason="tool_use",
            usage=_usage(input_tokens=40, output_tokens=12),
        )
        turn2_response = _response(
            blocks=[_text_block("It is sunny in SF.")],
            stop_reason="end_turn",
            usage=_usage(input_tokens=60, output_tokens=9, cached=20),
        )
        client.client.messages.create.side_effect = [turn1_response, turn2_response]

        messages = [{"role": "user", "content": "What's the weather in SF?"}]
        turn1 = client.send_conversation("You are a helper.", messages, tools=[tool])

        assert turn1.finish_reason is AIFinishReason.TOOL_USE
        assert turn1.finish_reason == "tool_use"
        assert turn1.text == "Checking the weather."
        assert len(turn1.tool_calls) == 1
        assert turn1.tool_calls[0].id == "toolu_1"
        assert turn1.tool_calls[0].name == "get_weather"
        assert turn1.tool_calls[0].input == {"city": "SF"}
        assert turn1.usage.input_tokens == 40
        assert turn1.usage.output_tokens == 12
        # raw_content is replayable verbatim as the next assistant turn.
        assert turn1.raw_content[1]["type"] == "tool_use"
        assert turn1.raw_content[1]["input"] == {"city": "SF"}

        messages.append({"role": "assistant", "content": turn1.raw_content})
        messages.append(
            client.build_tool_result_message(
                tool_call_id="toolu_1", result={"temp_f": 65}, is_error=False
            )
        )
        turn2 = client.send_conversation("You are a helper.", messages, tools=[tool])

        assert turn2.finish_reason is AIFinishReason.COMPLETE
        assert turn2.text == "It is sunny in SF."
        assert turn2.tool_calls == []
        # Usage is present on every turn, cached folded into input per convention.
        assert turn2.usage.input_tokens == 80
        assert turn2.usage.cached_input_tokens == 20

        first_kwargs = client.client.messages.create.call_args_list[0].kwargs
        assert first_kwargs["system"] == "You are a helper."
        assert first_kwargs["tools"] == [
            {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "input_schema": tool.input_schema,
                "strict": True,
            }
        ]
        second_kwargs = client.client.messages.create.call_args_list[1].kwargs
        tool_result_entry = second_kwargs["messages"][2]
        assert tool_result_entry["role"] == "user"
        assert tool_result_entry["content"][0]["type"] == "tool_result"
        assert tool_result_entry["content"][0]["tool_use_id"] == "toolu_1"
        assert json.loads(tool_result_entry["content"][0]["content"]) == {"temp_f": 65}
        assert tool_result_entry["content"][0]["is_error"] is False

    def test_forced_tool_choice(self):
        client = _build_client()
        client.client.messages.create.return_value = _response(
            blocks=[_tool_use_block("toolu_2", "get_weather", {"city": "NYC"})],
            stop_reason="tool_use",
        )
        client.send_conversation(
            "sys",
            [{"role": "user", "content": "weather"}],
            tools=[_weather_tool()],
            tool_choice="get_weather",
        )
        kwargs = client.client.messages.create.call_args.kwargs
        assert kwargs["tool_choice"] == {"type": "tool", "name": "get_weather"}

    def test_tool_choice_must_name_supplied_tool(self):
        client = _build_client()
        with pytest.raises(ValueError, match="does not name a supplied tool"):
            client.send_conversation(
                "sys",
                [{"role": "user", "content": "hi"}],
                tools=[_weather_tool()],
                tool_choice="unknown_tool",
            )

    def test_empty_messages_rejected(self):
        client = _build_client()
        with pytest.raises(ValueError, match="messages cannot be empty"):
            client.send_conversation("sys", [], tools=[_weather_tool()])

    def test_capability_gate_raises_typed_error(self):
        stub = _StubCompletions()
        with pytest.raises(AiProviderCapabilityUnsupportedError):
            stub.send_conversation("sys", [{"role": "user", "content": "hi"}])
        with pytest.raises(AiProviderCapabilityUnsupportedError):
            stub.build_tool_result_message(tool_call_id="x", result={}, is_error=False)

    def test_pii_middleware_refuses_conversation(self):
        client = _build_client()
        client.pii_middleware = SimpleNamespace(bool_enabled=True)
        with pytest.raises(AiProviderConfigurationError, match="PII redaction"):
            client.send_conversation("sys", [{"role": "user", "content": "hi"}])

    def test_provider_options_merge_and_reserved_retry_key(self):
        client = _build_client()
        client.client.with_options.return_value = client.client
        client.client.messages.create.return_value = _response(
            blocks=[_text_block("ok")]
        )
        client.send_conversation(
            "sys",
            [{"role": "user", "content": "hi"}],
            tools=[_weather_tool()],
            provider_options={"metadata": {"user_id": "u1"}, "retry_policy": "none"},
        )
        client.client.with_options.assert_called_once_with(max_retries=0)
        kwargs = client.client.messages.create.call_args.kwargs
        assert kwargs["metadata"] == {"user_id": "u1"}
        assert "retry_policy" not in kwargs

    def test_request_timeout_maps_to_sdk_timeout(self):
        client = _build_client()
        client.client.with_options.return_value = client.client
        client.client.messages.create.return_value = _response(
            blocks=[_text_block("ok")]
        )
        client.send_conversation(
            "sys",
            [{"role": "user", "content": "hi"}],
            request_timeout_seconds=12.5,
        )
        client.client.with_options.assert_called_once_with(timeout=12.5)


# ── send_structured_output ──────────────────────────────────────────────────


class TestSendStructuredOutput:
    def test_raw_schema_returns_parsed_dict(self):
        client = _build_client()
        payload = {"nodes": [{"kind": "task", "name": "extract"}]}
        client.client.messages.create.return_value = _response(
            blocks=[_text_block(json.dumps(payload))],
            stop_reason="end_turn",
            usage=_usage(input_tokens=100, output_tokens=50),
        )
        result = client.send_structured_output(
            "Compile this prose into a graph.",
            response_schema=RAW_GRAPH_SCHEMA,
            system_prompt="You are a compiler.",
        )
        assert result.finish_reason is AIFinishReason.COMPLETE
        assert result.data == payload
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50

        kwargs = client.client.messages.create.call_args.kwargs
        assert kwargs["system"] == "You are a compiler."
        schema = kwargs["output_config"]["format"]["schema"]
        assert kwargs["output_config"]["format"]["type"] == "json_schema"
        # anyOf variants survive; object nodes are closed.
        assert schema["additionalProperties"] is False
        any_of = schema["properties"]["nodes"]["items"]["anyOf"]
        assert all(variant["additionalProperties"] is False for variant in any_of)
        # The caller-supplied schema object is not mutated.
        assert "additionalProperties" not in RAW_GRAPH_SCHEMA

    def test_response_model_path_validates(self):
        client = _build_client()
        client.client.messages.create.return_value = _response(
            blocks=[_text_block(json.dumps({"node_id": "n1", "kind": "task"}))]
        )
        result = client.send_structured_output(
            "Extract the node.", response_model=_NodeModel
        )
        assert result.data == {"node_id": "n1", "kind": "task"}

    def test_response_model_validation_failure_raises(self):
        client = _build_client()
        client.client.messages.create.return_value = _response(
            blocks=[_text_block(json.dumps({"node_id": "n1"}))]
        )
        with pytest.raises(ValueError, match="validation errors"):
            client.send_structured_output(
                "Extract the node.", response_model=_NodeModel
            )

    def test_length_truncation_returns_finish_reason_not_exception(self):
        client = _build_client()
        client.client.messages.create.return_value = _response(
            blocks=[_text_block('{"nodes": [')], stop_reason="max_tokens"
        )
        result = client.send_structured_output(
            "Compile.", response_schema=RAW_GRAPH_SCHEMA, max_response_tokens=4096
        )
        assert result.finish_reason is AIFinishReason.LENGTH
        assert result.data is None
        assert result.raw_text == '{"nodes": ['

    def test_refusal_returns_finish_reason_not_exception(self):
        client = _build_client()
        client.client.messages.create.return_value = _response(
            blocks=[], stop_reason="refusal"
        )
        result = client.send_structured_output(
            "Compile.", response_schema=RAW_GRAPH_SCHEMA
        )
        assert result.finish_reason is AIFinishReason.REFUSAL
        assert result.data is None

    def test_messages_replay_with_prompt_appended(self):
        client = _build_client()
        client.client.messages.create.return_value = _response(
            blocks=[_text_block(json.dumps({"nodes": []}))]
        )
        prior = [
            {"role": "user", "content": "compile this"},
            {"role": "assistant", "content": '{"nodes": [{"kind": "bogus"}]}'},
        ]
        client.send_structured_output(
            "The kind 'bogus' is invalid; use 'task' or 'gate'.",
            response_schema=RAW_GRAPH_SCHEMA,
            messages=prior,
        )
        kwargs = client.client.messages.create.call_args.kwargs
        assert kwargs["messages"][0] == prior[0]
        assert kwargs["messages"][1] == prior[1]
        assert kwargs["messages"][2]["role"] == "user"
        assert "bogus" in kwargs["messages"][2]["content"]

    def test_messages_only_without_prompt_is_allowed(self):
        client = _build_client()
        client.client.messages.create.return_value = _response(
            blocks=[_text_block(json.dumps({"nodes": []}))]
        )
        result = client.send_structured_output(
            response_schema=RAW_GRAPH_SCHEMA,
            messages=[{"role": "user", "content": "compile"}],
        )
        assert result.data == {"nodes": []}

    def test_exactly_one_schema_source_required(self):
        client = _build_client()
        with pytest.raises(ValueError, match="exactly one"):
            client.send_structured_output("p")
        with pytest.raises(ValueError, match="exactly one"):
            client.send_structured_output(
                "p", response_model=_NodeModel, response_schema=RAW_GRAPH_SCHEMA
            )

    def test_prompt_or_messages_required(self):
        client = _build_client()
        with pytest.raises(ValueError, match="prompt, messages"):
            client.send_structured_output(response_schema=RAW_GRAPH_SCHEMA)

    def test_max_response_tokens_capped_at_context_window(self):
        client = _build_client()
        with pytest.raises(ValueError, match="context window"):
            client.send_structured_output(
                "p",
                response_schema=RAW_GRAPH_SCHEMA,
                max_response_tokens=2_000_000,
            )

    def test_large_budget_streams_and_accumulates(self):
        client = _build_client()
        payload = {"nodes": [{"kind": "gate", "condition": "x > 1"}]}
        text = json.dumps(payload)
        events = [
            Mock(
                type="message_start",
                message=Mock(
                    usage=Mock(input_tokens=500, cache_read_input_tokens=None)
                ),
            ),
            Mock(
                type="content_block_delta",
                delta=Mock(type="text_delta", text=text[: len(text) // 2]),
            ),
            Mock(
                type="content_block_delta",
                delta=Mock(type="text_delta", text=text[len(text) // 2 :]),
            ),
            Mock(
                type="message_delta",
                delta=Mock(stop_reason="end_turn"),
                usage=Mock(output_tokens=900),
            ),
        ]
        client.client.messages.create.return_value = iter(events)
        result = client.send_structured_output(
            "Compile.",
            response_schema=RAW_GRAPH_SCHEMA,
            max_response_tokens=64_000,
        )
        assert result.data == payload
        assert result.usage.input_tokens == 500
        assert result.usage.output_tokens == 900
        kwargs = client.client.messages.create.call_args.kwargs
        assert kwargs["stream"] is True
        assert kwargs["max_tokens"] == 64_000

    def test_invalid_json_on_complete_raises(self):
        client = _build_client()
        client.client.messages.create.return_value = _response(
            blocks=[_text_block("not json")]
        )
        with pytest.raises(ValueError, match="Invalid JSON response"):
            client.send_structured_output("p", response_schema=RAW_GRAPH_SCHEMA)

    def test_capability_gate_raises_typed_error(self):
        stub = _StubCompletions()
        with pytest.raises(AiProviderCapabilityUnsupportedError):
            stub.send_structured_output("p", response_schema=RAW_GRAPH_SCHEMA)


# ── Extended send_prompt parameters ─────────────────────────────────────────


class TestSendPromptExtendedParams:
    def test_claude_maps_all_parameters(self):
        client = _build_client()
        client.client.with_options.return_value = client.client
        client.client.messages.create.return_value = _response(
            blocks=[_text_block("doc text")]
        )
        text = client.send_prompt(
            "Generate a workflow document.",
            system_prompt="You write workflow documents.",
            max_response_tokens=9000,
            request_timeout_seconds=30.0,
        )
        assert text == "doc text"
        client.client.with_options.assert_called_once_with(timeout=30.0)
        kwargs = client.client.messages.create.call_args.kwargs
        assert kwargs["system"] == "You write workflow documents."
        assert kwargs["max_tokens"] == 9000

    def test_omitting_new_parameters_preserves_default_behavior(self):
        client = _build_client()
        client.client.messages.create.return_value = _response(
            blocks=[_text_block("ok")]
        )
        client.send_prompt("hello")
        client.client.with_options.assert_not_called()
        kwargs = client.client.messages.create.call_args.kwargs
        assert kwargs["max_tokens"] == AiAnthropicCompletions.SEND_PROMPT_MAX_TOKENS

    def test_explicit_system_prompt_overrides_other_params(self):
        client = _build_client()
        client.client.messages.create.return_value = _response(
            blocks=[_text_block("ok")]
        )

        class _Params(AICompletionsPromptParamsBase):
            pass

        client.send_prompt(
            "hello",
            system_prompt="explicit wins",
            other_params=_Params(system_prompt="params value"),
        )
        assert (
            client.client.messages.create.call_args.kwargs["system"] == "explicit wins"
        )

    def test_unmapped_engine_rejects_new_parameters(self):
        stub = _StubCompletions()
        with pytest.raises(AiProviderCapabilityUnsupportedError):
            stub._reject_unsupported_send_prompt_params(
                max_response_tokens=100, request_timeout_seconds=None
            )
        # No parameters supplied is fine.
        stub._reject_unsupported_send_prompt_params(
            max_response_tokens=None, request_timeout_seconds=None
        )


# ── Async variants ──────────────────────────────────────────────────────────


def _wire_async(client: AiAnthropicCompletions, response) -> Mock:
    async_client = Mock()
    async_client.messages.create = AsyncMock(return_value=response)
    async_client.with_options.return_value = async_client
    client._async_client = async_client
    return async_client


class TestAsyncVariants:
    @pytest.mark.asyncio
    async def test_asend_prompt(self):
        client = _build_client()
        _wire_async(client, _response(blocks=[_text_block("async hello")]))
        text = await client.asend_prompt(
            "hi", system_prompt="s", max_response_tokens=64
        )
        assert text == "async hello"
        kwargs = client._async_client.messages.create.call_args.kwargs
        assert kwargs["system"] == "s"
        assert kwargs["max_tokens"] == 64

    @pytest.mark.asyncio
    async def test_asend_conversation_turn(self):
        client = _build_client()
        _wire_async(
            client,
            _response(
                blocks=[_tool_use_block("toolu_9", "get_weather", {"city": "LA"})],
                stop_reason="tool_use",
            ),
        )
        turn = await client.asend_conversation(
            "sys",
            [{"role": "user", "content": "weather"}],
            tools=[_weather_tool()],
            tool_choice="get_weather",
        )
        assert turn.finish_reason is AIFinishReason.TOOL_USE
        assert turn.tool_calls[0].input == {"city": "LA"}
        kwargs = client._async_client.messages.create.call_args.kwargs
        assert kwargs["tool_choice"] == {"type": "tool", "name": "get_weather"}

    @pytest.mark.asyncio
    async def test_asend_structured_output(self):
        client = _build_client()
        payload = {"nodes": []}
        _wire_async(client, _response(blocks=[_text_block(json.dumps(payload))]))
        result = await client.asend_structured_output(
            "compile", response_schema=RAW_GRAPH_SCHEMA
        )
        assert result.data == payload
        assert result.finish_reason is AIFinishReason.COMPLETE

    @pytest.mark.asyncio
    async def test_async_capability_gate(self):
        stub = _StubCompletions()
        with pytest.raises(AiProviderCapabilityUnsupportedError):
            await stub.asend_prompt("hi")


# ── Retry policy and typed request errors ───────────────────────────────────


class TestRetryPolicyAndErrors:
    def test_constructor_retry_policy_none_disables_sdk_retries(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch(
                "ai_api_unified.ai_anthropic_base.Anthropic"
            ) as mock_anthropic_cls:
                AiAnthropicCompletions(model="claude-opus-4-8", retry_policy="none")
        assert mock_anthropic_cls.call_args.kwargs["max_retries"] == 0

    def test_default_retry_policy_keeps_sdk_retries(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch(
                "ai_api_unified.ai_anthropic_base.Anthropic"
            ) as mock_anthropic_cls:
                AiAnthropicCompletions(model="claude-opus-4-8")
        assert "max_retries" not in mock_anthropic_cls.call_args.kwargs

    def test_env_retry_policy_none(self):
        with patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "test-key", "COMPLETIONS_RETRY_POLICY": "none"},
        ):
            with patch(
                "ai_api_unified.ai_anthropic_base.Anthropic"
            ) as mock_anthropic_cls:
                AiAnthropicCompletions(model="claude-opus-4-8")
        assert mock_anthropic_cls.call_args.kwargs["max_retries"] == 0

    def test_invalid_retry_policy_rejected(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with pytest.raises(ValueError, match="Unsupported retry policy"):
                AiAnthropicCompletions(model="claude-opus-4-8", retry_policy="maybe")

    def test_status_error_wrapped_with_status_code(self):
        client = _build_client()
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(429, request=request)
        client.client.messages.create.side_effect = anthropic.APIStatusError(
            "rate limited", response=response, body=None
        )
        with pytest.raises(AiProviderRequestError) as exc_info:
            client.send_conversation("sys", [{"role": "user", "content": "hi"}])
        assert exc_info.value.status_code == 429
        assert exc_info.value.provider_engine == "claude"

    def test_connection_error_wrapped_without_status_code(self):
        client = _build_client()
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        client.client.messages.create.side_effect = anthropic.APIConnectionError(
            request=request
        )
        with pytest.raises(AiProviderRequestError) as exc_info:
            client.send_prompt("hi")
        assert exc_info.value.status_code is None


# ── Observability tags ──────────────────────────────────────────────────────


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class TestObservabilityTags:
    def test_set_context_normalizes_tags(self):
        token = set_observability_context(
            caller_id="workflow-service",
            tags={"run_id": " r-42 ", "node_id": "n-7", " ": "dropped", "empty": " "},
        )
        try:
            context = get_observability_context()
            assert dict(context.tags) == {"run_id": "r-42", "node_id": "n-7"}
        finally:
            reset_observability_context(token)
        assert dict(get_observability_context().tags) == {}

    def test_call_context_carries_tags(self):
        client = _build_client()
        token = set_observability_context(
            caller_id="workflow-service",
            tags={"run_id": "r-42", "workflow": "intake"},
        )
        try:
            call_context = client._build_observability_call_context(
                capability="completions", operation="send_conversation"
            )
        finally:
            reset_observability_context(token)
        assert dict(call_context.dict_tags) == {"run_id": "r-42", "workflow": "intake"}

    def test_tags_emitted_on_lifecycle_events(self):
        middleware = LoggerBackedObservabilityMiddleware(ObservabilitySettingsModel())
        call_context = AiApiCallContextModel(
            call_id="call-1",
            event_time_utc=datetime.now(timezone.utc),
            capability="completions",
            operation="send_conversation",
            provider_vendor="anthropic",
            provider_engine="claude",
            model_name="claude-opus-4-8",
            model_version=None,
            direction="output",
            dict_tags={"run_id": "r-42", "node_id": "n-7"},
        )
        dict_fields = middleware._build_shared_event_fields(call_context=call_context)
        assert dict_fields["tag_run_id"] == "r-42"
        assert dict_fields["tag_node_id"] == "n-7"

    def test_tags_emitted_on_cost_events(self):
        handler = _CaptureHandler()
        cost_logger = logging.getLogger(COST_LOGGER_NAME)
        cost_logger.addHandler(handler)
        cost_logger.setLevel(logging.DEBUG)
        try:
            middleware = LoggerBackedObservabilityMiddleware(
                ObservabilitySettingsModel(emit_cost=True)
            )
            call_context = AiApiCallContextModel(
                call_id="call-2",
                event_time_utc=datetime.now(timezone.utc),
                capability="completions",
                operation="send_conversation",
                provider_vendor="anthropic",
                provider_engine="claude",
                model_name="claude-opus-4-8",
                model_version=None,
                direction="output",
                originating_caller_id="workflow-service",
                dict_tags={"run_id": "r-42"},
            )
            summary = AiApiCallResultSummaryModel(
                provider_elapsed_ms=5.0,
                provider_prompt_tokens=1000,
                provider_completion_tokens=100,
            )
            middleware._maybe_emit_cost_event(
                call_context=call_context, call_result_summary=summary
            )
            assert len(handler.records) == 1
            dict_cost_fields = handler.records[0].args[1]
            assert dict_cost_fields["tag_run_id"] == "r-42"
            assert dict_cost_fields["caller_id"] == "workflow-service"
        finally:
            cost_logger.removeHandler(handler)
