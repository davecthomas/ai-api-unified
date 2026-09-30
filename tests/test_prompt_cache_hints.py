# ruff: noqa: E402
# test_prompt_cache_hints.py
"""
Tests for the provider-neutral prompt cache hint (AIPromptCacheHint).

Covers the capability flags each engine declares, the request shape each
engine sends when a hint is supplied (Anthropic cache_control, Bedrock
cachePoint, OpenAI prompt_cache_key and prompt_cache_retention), the
unchanged request shape when no hint is supplied, per-model gating of
unsupported fields, and that send_conversation threads the hint to the
provider hook. All against mocked SDK clients.
"""

import os
from typing import Any
from unittest.mock import Mock, patch

import pytest

pytest.importorskip("anthropic")
pytest.importorskip("openai")
pytest.importorskip("boto3")

from ai_api_unified import AIPromptCacheHint, AIPromptCacheRetention
from ai_api_unified.ai_base import (
    AIBatchRequestItem,
    AIStructuredPrompt,
    AICompletionsPromptParamsBase,
    AITurnResult,
)
from ai_api_unified.completions.ai_anthropic_completions import (
    AiAnthropicCompletions,
)
from ai_api_unified.completions.ai_bedrock_completions import AiBedrockCompletions
from ai_api_unified.completions.ai_google_gemini_capabilities import (
    AICompletionsCapabilitiesGoogle,
)
from ai_api_unified.completions.ai_openai_compatible_completions import (
    AiOpenAICompatibleCompletions,
)
from ai_api_unified.completions.ai_openai_completions import AiOpenAICompletions
from ai_api_unified.completions.ai_openai_responses_completions import (
    AiOpenAIResponsesCompletions,
)

HINT_DEFAULT: AIPromptCacheHint = AIPromptCacheHint()
HINT_EXTENDED: AIPromptCacheHint = AIPromptCacheHint(
    retention=AIPromptCacheRetention.EXTENDED, key="tenant-42"
)
SYSTEM_PROMPT: str = "You are a stable, cacheable system prompt."


class _PromptParams(AICompletionsPromptParamsBase):
    """Concrete prompt params for tests (the base class is abstract)."""


def _params(prompt_cache: AIPromptCacheHint | None) -> _PromptParams:
    return _PromptParams(system_prompt=SYSTEM_PROMPT, prompt_cache=prompt_cache)


# ── Builders ────────────────────────────────────────────────────────────────


def _anthropic(model: str = "claude-opus-4-8") -> AiAnthropicCompletions:
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        client = AiAnthropicCompletions(model=model)
    client.client = Mock()
    client.client.messages.create.return_value = Mock(
        content=[Mock(type="text", text="ok")],
        stop_reason="end_turn",
        usage=Mock(input_tokens=10, output_tokens=2, cache_read_input_tokens=None),
    )
    return client


def _bedrock(model: str) -> AiBedrockCompletions:
    with patch("ai_api_unified.ai_bedrock_base.boto3"):
        client = AiBedrockCompletions(model=model)
    client.client = Mock()
    client.backoff_delays = [0.0]
    client._sleep_with_backoff = lambda base_delay: None
    client.client.converse.return_value = {
        "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 9, "outputTokens": 2, "totalTokens": 11},
    }
    return client


def _openai(
    model: str = "gpt-5.1", cls: type[AiOpenAICompletions] = AiOpenAICompletions
) -> AiOpenAICompletions:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        client = cls(model=model)
    client.client = Mock()
    return client


def _openai_compatible() -> AiOpenAICompatibleCompletions:
    dict_settings: dict[str, str] = {
        "OPENAI_COMPATIBLE_BASE_URL": "http://localhost:8000/v1",
        "COMPLETIONS_MODEL_NAME": "local-model",
    }
    with patch(
        "ai_api_unified.util.env_settings.EnvSettings.get_setting",
        side_effect=lambda str_key, default=None: dict_settings.get(str_key, default),
    ):
        client = AiOpenAICompatibleCompletions()
    client.client = Mock()
    return client


def _chat_response() -> Mock:
    message = Mock(spec=["content", "tool_calls", "refusal", "model_dump"])
    message.content = "ok"
    message.tool_calls = None
    message.refusal = None
    return Mock(
        choices=[Mock(message=message, finish_reason="stop")],
        usage=Mock(
            prompt_tokens=10,
            completion_tokens=2,
            total_tokens=12,
            prompt_tokens_details=Mock(cached_tokens=None),
        ),
    )


# ── Capability flags ────────────────────────────────────────────────────────


class TestCapabilityFlags:
    def test_base_defaults_are_off(self) -> None:
        capabilities = _openai_compatible().capabilities
        assert capabilities.implicit_prompt_caching is False
        assert capabilities.supports_prompt_cache_hint is False

    def test_anthropic_honors_hint(self) -> None:
        assert _anthropic().capabilities.supports_prompt_cache_hint is True

    def test_openai_is_implicit_and_honors_hint(self) -> None:
        capabilities = _openai().capabilities
        assert capabilities.implicit_prompt_caching is True
        assert capabilities.supports_prompt_cache_hint is True

    def test_responses_engine_keeps_the_flags(self) -> None:
        capabilities = _openai(cls=AiOpenAIResponsesCompletions).capabilities
        assert capabilities.supports_prompt_cache_hint is True

    @pytest.mark.parametrize("str_model", ["gemini-3.8-flash", "gemini-2.5-pro"])
    def test_gemini_caches_implicitly_without_a_hint(self, str_model: str) -> None:
        capabilities = AICompletionsCapabilitiesGoogle.for_model(str_model)
        assert capabilities.implicit_prompt_caching is True
        assert capabilities.supports_prompt_cache_hint is False

    @pytest.mark.parametrize(
        ("str_model", "bool_hint"),
        [
            ("us.anthropic.claude-opus-5", True),
            ("us.anthropic.claude-3-7-sonnet-20250219-v1:0", True),
            ("amazon.nova-lite-v1:0", True),
            ("us.anthropic.claude-3-5-haiku-20241022-v1:0", False),
            ("deepseek.v3.2", False),
        ],
    )
    def test_bedrock_hint_support_is_per_model(
        self, str_model: str, bool_hint: bool
    ) -> None:
        assert _bedrock(str_model).capabilities.supports_prompt_cache_hint is bool_hint


# ── Anthropic ───────────────────────────────────────────────────────────────


class TestAnthropic:
    def test_no_hint_keeps_plain_system_string(self) -> None:
        client = _anthropic()
        client.send_prompt("hi", other_params=_params(None))
        kwargs: dict[str, Any] = client.client.messages.create.call_args.kwargs
        assert kwargs["system"] == SYSTEM_PROMPT
        assert "cache_control" not in kwargs

    def test_hint_marks_system_block(self) -> None:
        client = _anthropic()
        client.send_prompt("hi", other_params=_params(HINT_DEFAULT))
        kwargs: dict[str, Any] = client.client.messages.create.call_args.kwargs
        assert kwargs["system"] == [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        # A one-shot prompt ends in unique content; automatic caching there
        # would pay the write premium on bytes never read back.
        assert "cache_control" not in kwargs

    def test_extended_retention_uses_one_hour_ttl(self) -> None:
        client = _anthropic()
        client.send_prompt("hi", other_params=_params(HINT_EXTENDED))
        kwargs: dict[str, Any] = client.client.messages.create.call_args.kwargs
        assert kwargs["system"][0]["cache_control"] == {
            "type": "ephemeral",
            "ttl": "1h",
        }

    def test_conversation_adds_matching_top_level_cache_control(self) -> None:
        client = _anthropic()
        dict_kwargs: dict[str, Any] = client._build_conversation_request_kwargs(
            system_prompt=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            tool_choice=None,
            max_response_tokens=None,
            dict_merge_options={},
            prompt_cache=HINT_EXTENDED,
        )
        dict_expected: dict[str, str] = {"type": "ephemeral", "ttl": "1h"}
        assert dict_kwargs["cache_control"] == dict_expected
        assert dict_kwargs["system"][0]["cache_control"] == dict_expected

    def test_conversation_without_hint_is_unchanged(self) -> None:
        client = _anthropic()
        dict_kwargs: dict[str, Any] = client._build_conversation_request_kwargs(
            system_prompt=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            tool_choice=None,
            max_response_tokens=None,
            dict_merge_options={},
        )
        assert dict_kwargs["system"] == SYSTEM_PROMPT
        assert "cache_control" not in dict_kwargs


# ── Bedrock ─────────────────────────────────────────────────────────────────


class TestBedrock:
    def test_no_hint_sends_text_only(self) -> None:
        client = _bedrock("us.anthropic.claude-opus-5")
        client.send_prompt("hi", other_params=_params(None))
        kwargs: dict[str, Any] = client.client.converse.call_args.kwargs
        assert kwargs["system"] == [{"text": SYSTEM_PROMPT}]

    def test_hint_appends_cache_point(self) -> None:
        client = _bedrock("us.anthropic.claude-opus-5")
        client.send_prompt("hi", other_params=_params(HINT_DEFAULT))
        kwargs: dict[str, Any] = client.client.converse.call_args.kwargs
        assert kwargs["system"] == [
            {"text": SYSTEM_PROMPT},
            {"cachePoint": {"type": "default"}},
        ]

    def test_extended_retention_on_supported_model(self) -> None:
        client = _bedrock("us.anthropic.claude-opus-5")
        client.send_prompt("hi", other_params=_params(HINT_EXTENDED))
        kwargs: dict[str, Any] = client.client.converse.call_args.kwargs
        assert kwargs["system"][1] == {"cachePoint": {"type": "default", "ttl": "1h"}}

    def test_extended_retention_falls_back_on_nova(self) -> None:
        client = _bedrock("amazon.nova-lite-v1:0")
        client.send_prompt("hi", other_params=_params(HINT_EXTENDED))
        kwargs: dict[str, Any] = client.client.converse.call_args.kwargs
        assert kwargs["system"][1] == {"cachePoint": {"type": "default"}}

    def test_unsupported_model_ignores_hint(self) -> None:
        client = _bedrock("deepseek.v3.2")
        client.send_prompt("hi", other_params=_params(HINT_EXTENDED))
        kwargs: dict[str, Any] = client.client.converse.call_args.kwargs
        assert kwargs["system"] == [{"text": SYSTEM_PROMPT}]


# ── OpenAI ──────────────────────────────────────────────────────────────────


class TestOpenAI:
    def test_no_hint_sends_no_cache_fields(self) -> None:
        client = _openai()
        client.client.chat.completions.create.return_value = _chat_response()
        client.send_prompt("hi", other_params=_params(None))
        kwargs: dict[str, Any] = client.client.chat.completions.create.call_args.kwargs
        assert "prompt_cache_key" not in kwargs
        assert "prompt_cache_retention" not in kwargs

    def test_hint_sends_key_and_extended_retention(self) -> None:
        client = _openai()
        client.client.chat.completions.create.return_value = _chat_response()
        client.send_prompt("hi", other_params=_params(HINT_EXTENDED))
        kwargs: dict[str, Any] = client.client.chat.completions.create.call_args.kwargs
        assert kwargs["prompt_cache_key"] == "tenant-42"
        assert kwargs["prompt_cache_retention"] == "24h"

    def test_extended_retention_skipped_on_models_without_it(self) -> None:
        client = _openai(model="gpt-4o-mini")
        assert client._build_prompt_cache_kwargs(HINT_EXTENDED) == {
            "prompt_cache_key": "tenant-42"
        }

    def test_default_hint_without_key_sends_nothing(self) -> None:
        assert _openai()._build_prompt_cache_kwargs(HINT_DEFAULT) == {}

    def test_conversation_request_carries_cache_fields(self) -> None:
        client = _openai()
        dict_kwargs: dict[str, Any] = client._build_chat_conversation_request_kwargs(
            system_prompt=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            tool_choice=None,
            max_response_tokens=None,
            dict_merge_options={},
            prompt_cache=HINT_EXTENDED,
        )
        assert dict_kwargs["prompt_cache_key"] == "tenant-42"
        assert dict_kwargs["prompt_cache_retention"] == "24h"

    def test_responses_request_carries_cache_fields(self) -> None:
        client = _openai(cls=AiOpenAIResponsesCompletions)
        dict_kwargs: dict[str, Any] = (
            client._build_responses_conversation_request_kwargs(
                system_prompt=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
                tool_choice=None,
                max_response_tokens=None,
                dict_merge_options={},
                prompt_cache=HINT_EXTENDED,
            )
        )
        assert dict_kwargs["prompt_cache_key"] == "tenant-42"
        assert dict_kwargs["prompt_cache_retention"] == "24h"

    def test_compatible_vendor_never_sends_cache_fields(self) -> None:
        assert _openai_compatible()._build_prompt_cache_kwargs(HINT_EXTENDED) == {}


# ── Base template method ────────────────────────────────────────────────────


class TestSendConversationThreading:
    def test_hint_reaches_provider_hook(self) -> None:
        client = _anthropic()
        mock_hook: Mock = Mock(return_value=Mock(spec=AITurnResult))
        with patch.object(client, "_send_conversation_provider", mock_hook):
            client.send_conversation(
                SYSTEM_PROMPT,
                [{"role": "user", "content": "hi"}],
                prompt_cache=HINT_DEFAULT,
            )
        assert mock_hook.call_args.kwargs["prompt_cache"] is HINT_DEFAULT

    def test_hook_defaults_to_no_hint(self) -> None:
        client = _anthropic()
        mock_hook: Mock = Mock(return_value=Mock(spec=AITurnResult))
        with patch.object(client, "_send_conversation_provider", mock_hook):
            client.send_conversation(SYSTEM_PROMPT, [{"role": "user", "content": "hi"}])
        assert mock_hook.call_args.kwargs["prompt_cache"] is None


# ── Structured output and batches ───────────────────────────────────────────


class TestStructuredOutputAndBatch:
    SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    def test_structured_output_hint_reaches_provider_hook(self) -> None:
        client = _anthropic()
        mock_hook: Mock = Mock(return_value=Mock(data=None))
        with patch.object(client, "_send_structured_output_provider", mock_hook):
            client.send_structured_output(
                "hi", response_schema=self.SCHEMA, prompt_cache=HINT_DEFAULT
            )
        assert mock_hook.call_args.kwargs["prompt_cache"] is HINT_DEFAULT

    def test_anthropic_structured_request_marks_system_block(self) -> None:
        client = _anthropic()
        dict_kwargs: dict[str, Any] = client._build_structured_request_kwargs(
            response_schema=self.SCHEMA,
            system_prompt=SYSTEM_PROMPT,
            prompt="hi",
            messages=None,
            max_response_tokens=1024,
            dict_merge_options={},
            prompt_cache=HINT_EXTENDED,
        )
        assert dict_kwargs["system"][0]["cache_control"] == {
            "type": "ephemeral",
            "ttl": "1h",
        }

    def test_openai_structured_request_carries_cache_fields(self) -> None:
        client = _openai()
        dict_kwargs: dict[str, Any] = client._build_chat_structured_request_kwargs(
            response_schema=self.SCHEMA,
            system_prompt=SYSTEM_PROMPT,
            prompt="hi",
            messages=None,
            max_response_tokens=1024,
            dict_merge_options={},
            prompt_cache=HINT_EXTENDED,
        )
        assert dict_kwargs["prompt_cache_key"] == "tenant-42"

    def test_anthropic_batch_item_marks_system_block(self) -> None:
        client = _anthropic()
        client.client.messages.batches.create.return_value = Mock(
            id="batch_1",
            processing_status="in_progress",
            request_counts=Mock(
                processing=2, succeeded=0, errored=0, canceled=0, expired=0
            ),
            created_at=None,
            ended_at=None,
            expires_at=None,
        )
        client.submit_batch(
            [
                AIBatchRequestItem(
                    custom_id="a", prompt="hi", prompt_cache=HINT_EXTENDED
                ),
                AIBatchRequestItem(custom_id="b", prompt="hi"),
            ]
        )
        list_requests: list[dict[str, Any]] = (
            client.client.messages.batches.create.call_args.kwargs["requests"]
        )
        assert list_requests[0]["params"]["system"][0]["cache_control"] == {
            "type": "ephemeral",
            "ttl": "1h",
        }
        assert isinstance(list_requests[1]["params"]["system"], str)


class TestBlankSystemPrompt:
    def test_anthropic_keeps_blank_system_as_string(self) -> None:
        assert AiAnthropicCompletions._build_system_param("", HINT_DEFAULT) == ""

    def test_bedrock_skips_cache_point_for_blank_system(self) -> None:
        client = _bedrock("us.anthropic.claude-opus-5")
        assert client._build_converse_system("  ", HINT_DEFAULT) == [{"text": "  "}]


class TestAnthropicBreakpointLimit:
    @staticmethod
    def _messages(int_markers: int) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"part {index}",
                        "cache_control": {"type": "ephemeral"},
                    }
                    for index in range(int_markers)
                ],
            }
        ]

    def _kwargs(self, int_markers: int) -> dict[str, Any]:
        return _anthropic()._build_conversation_request_kwargs(
            system_prompt=SYSTEM_PROMPT,
            messages=self._messages(int_markers),
            tools=[],
            tool_choice=None,
            max_response_tokens=None,
            dict_merge_options={},
            prompt_cache=HINT_DEFAULT,
        )

    def test_no_caller_breakpoints_adds_both(self) -> None:
        dict_kwargs = self._kwargs(0)
        assert "cache_control" in dict_kwargs
        assert isinstance(dict_kwargs["system"], list)

    def test_caller_breakpoints_take_precedence(self) -> None:
        dict_kwargs = self._kwargs(1)
        assert "cache_control" not in dict_kwargs
        assert dict_kwargs["system"] == SYSTEM_PROMPT

    def test_structured_output_defers_to_caller_breakpoints(self) -> None:
        dict_kwargs = _anthropic()._build_structured_request_kwargs(
            response_schema={"type": "object"},
            system_prompt=SYSTEM_PROMPT,
            prompt=None,
            messages=self._messages(4),
            max_response_tokens=1024,
            dict_merge_options={},
            prompt_cache=HINT_DEFAULT,
        )
        assert dict_kwargs["system"] == SYSTEM_PROMPT


class TestOlderSdkGating:
    def test_openai_drops_fields_the_sdk_does_not_accept(self) -> None:
        client = _openai()
        with patch.object(
            AiOpenAICompletions,
            "_known_provider_option_keys",
            return_value=frozenset({"model", "messages", "prompt_cache_key"}),
        ):
            assert client._build_prompt_cache_kwargs(HINT_EXTENDED) == {
                "prompt_cache_key": "tenant-42"
            }

    def test_bedrock_omits_ttl_when_botocore_lacks_it(self) -> None:
        client = _bedrock("us.anthropic.claude-opus-5")
        with patch.object(
            AiBedrockCompletions,
            "_converse_cache_point_support",
            return_value=(True, False),
        ):
            assert client._build_converse_system(SYSTEM_PROMPT, HINT_EXTENDED)[1] == {
                "cachePoint": {"type": "default"}
            }

    def test_bedrock_skips_cache_point_when_botocore_lacks_it(self) -> None:
        client = _bedrock("us.anthropic.claude-opus-5")
        with patch.object(
            AiBedrockCompletions,
            "_converse_cache_point_support",
            return_value=(False, False),
        ):
            assert client._build_converse_system(SYSTEM_PROMPT, HINT_EXTENDED) == [
                {"text": SYSTEM_PROMPT}
            ]


class TestOpenAIDatedSnapshots:
    def test_dated_snapshot_gets_extended_retention(self) -> None:
        client = _openai(model="gpt-4.1-2025-04-14")
        assert (
            client._build_prompt_cache_kwargs(HINT_EXTENDED)["prompt_cache_retention"]
            == "24h"
        )


class TestBedrockImplicitFlag:
    @pytest.mark.parametrize(
        ("str_model", "bool_implicit"),
        [
            ("us.anthropic.claude-opus-5", True),
            ("amazon.nova-lite-v1:0", True),
            ("us.anthropic.claude-3-5-haiku-20241022-v1:0", False),
            ("deepseek.v3.2", False),
        ],
    )
    def test_implicit_flag_follows_documented_cache_support(
        self, str_model: str, bool_implicit: bool
    ) -> None:
        assert _bedrock(str_model).capabilities.implicit_prompt_caching is bool_implicit


class TestBedrockPrecedenceAndGating:
    def test_caller_cache_points_take_precedence(self) -> None:
        client = _bedrock("us.anthropic.claude-opus-5")
        list_messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [{"text": "hi"}, {"cachePoint": {"type": "default"}}],
            }
        ]
        assert client._build_converse_system(
            SYSTEM_PROMPT, HINT_DEFAULT, list_messages
        ) == [{"text": SYSTEM_PROMPT}]

    def test_no_hint_skips_the_botocore_lookup(self) -> None:
        client = _bedrock("us.anthropic.claude-opus-5")
        with patch.object(
            AiBedrockCompletions, "_converse_cache_point_support"
        ) as mock_support:
            client._build_converse_system(SYSTEM_PROMPT, None)
        mock_support.assert_not_called()

    @pytest.mark.parametrize(
        ("str_model", "bool_1h"),
        [
            ("us.anthropic.claude-opus-5", True),
            ("us.anthropic.claude-3-7-sonnet-20250219-v1:0", False),
            ("amazon.nova-lite-v1:0", False),
        ],
    )
    def test_one_hour_ttl_capability(self, str_model: str, bool_1h: bool) -> None:
        assert _bedrock(str_model).capabilities.supports_prompt_cache_1h_ttl is bool_1h

    def test_strict_schema_via_structured_output_forwards_hint(self) -> None:
        client = _bedrock("us.anthropic.claude-opus-4-6-v1")
        mock_structured: Mock = Mock(return_value=Mock(data={"answer": "x"}))

        class _Answer(AIStructuredPrompt):
            answer: str

            @staticmethod
            def get_prompt() -> str:
                return "hi"

        with patch.object(client, "send_structured_output", mock_structured):
            client.strict_schema_prompt(
                "hi", _Answer, other_params=_params(HINT_DEFAULT)
            )
        assert mock_structured.call_args.kwargs["prompt_cache"] is HINT_DEFAULT
