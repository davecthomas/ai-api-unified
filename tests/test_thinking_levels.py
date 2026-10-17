# tests/test_thinking_levels.py
"""
Thinking levels: one provider-neutral level (off..max) from configuration,
the factory, the client, or a call; each engine applies it through its
provider's own control; a model that cannot honor the level applies the
nearest one it supports and records both.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from unittest.mock import Mock, patch

import pytest

pytest.importorskip("anthropic")
pytest.importorskip("openai")
pytest.importorskip("boto3")
pytest.importorskip("google.genai")

from ai_api_unified import (
    AIFactory,
    AIStructuredPrompt,
    AIThinkingCapabilities,
    AIThinkingLevel,
    AIThinkingResolution,
)
from ai_api_unified.ai_base import AICompletionsPromptParamsBase
from ai_api_unified.completions.ai_anthropic_completions import (
    AICompletionsCapabilitiesAnthropic,
)
from ai_api_unified.completions.ai_bedrock_completions import (
    AICompletionsCapabilitiesBedrock,
)
from ai_api_unified.completions.ai_google_gemini_capabilities import (
    AICompletionsCapabilitiesGoogle,
)
from ai_api_unified.completions.ai_openai_completions import (
    AICompletionsCapabilitiesOpenAI,
)
from ai_api_unified.completions.ai_openai_responses_completions import (
    AiOpenAIResponsesCompletions,
)
from ai_api_unified.ai_models_config import load_models_config
from ai_api_unified.thinking import (
    THINKING_CALL_OVERRIDE,
    claude_thinking_class,
    claude_thinking_request_fields,
    gemini_thinking_config,
    parse_thinking_level,
    resolve_thinking_level,
)
from tests.test_multi_engine_conversation_api import _build_gemini_client
from tests.test_prompt_cache_hints import (
    _anthropic,
    _bedrock,
    _chat_response,
    _openai,
    _openai_compatible,
)

L = AIThinkingLevel


class _Params(AICompletionsPromptParamsBase):
    """Concrete prompt params (the base is abstract)."""


class _Word(AIStructuredPrompt):
    word: str

    @staticmethod
    def get_prompt(input_text: str = "") -> str:
        return input_text


@pytest.fixture(autouse=True)
def _no_configured_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """The suite's conftest blanks AI_MODELS_CONFIG; the level then defaults."""
    monkeypatch.delenv("COMPLETIONS_THINKING", raising=False)


# ── The level, the capabilities, the resolution ─────────────────────────────


class TestLevelAndResolution:
    def test_parse_accepts_values_any_case_and_blank_means_unset(self) -> None:
        assert parse_thinking_level(" High ") is L.HIGH
        assert parse_thinking_level(L.OFF) is L.OFF
        assert parse_thinking_level("") is None and parse_thinking_level(None) is None
        with pytest.raises(ValueError, match="Unknown thinking level 'turbo'"):
            parse_thinking_level("turbo")

    def test_exact_level_is_exact(self) -> None:
        caps = AIThinkingCapabilities(levels=[L.OFF, L.LOW, L.MEDIUM, L.HIGH, L.MAX])
        resolution = resolve_thinking_level(L.HIGH, caps)
        assert resolution.applied is L.HIGH and resolution.exact
        assert resolution.reason is None

    def test_nearest_level_ties_go_lower(self) -> None:
        caps = AIThinkingCapabilities(levels=[L.OFF, L.LOW, L.HIGH])
        # MEDIUM sits between LOW and HIGH: lower wins.
        assert caps.nearest(L.MEDIUM) is L.LOW
        # MINIMAL sits between OFF and LOW: lower wins.
        assert caps.nearest(L.MINIMAL) is L.OFF
        assert caps.nearest(L.MAX) is L.HIGH
        resolution = resolve_thinking_level(L.MEDIUM, caps)
        assert resolution.applied is L.LOW and not resolution.exact
        assert resolution.reason == "unsupported_level"

    def test_always_on_model_applies_its_lowest_level_for_off(self) -> None:
        caps = AIThinkingCapabilities(levels=[L.LOW, L.MEDIUM, L.HIGH, L.MAX])
        assert caps.always_on
        assert resolve_thinking_level(L.OFF, caps).applied is L.LOW

    def test_no_control_applies_nothing(self) -> None:
        resolution = resolve_thinking_level(L.MEDIUM, None)
        assert resolution.applied is None and not resolution.exact
        assert resolution.reason == "no_thinking_control"


# ── Provider tables ─────────────────────────────────────────────────────────


class TestClaudeMapping:
    @pytest.mark.parametrize(
        "model, cls",
        [
            ("claude-fable-5-1", "always_on"),
            ("claude-opus-5-5", "always_on"),
            ("claude-opus-5", "adaptive"),
            ("claude-haiku-5-5", "adaptive"),
            ("claude-sonnet-5-5", "between_tools"),
            ("claude-opus-4-8", "adaptive"),
            ("claude-sonnet-4-6", "adaptive"),
            ("claude-haiku-4-5", "budget"),
            ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", "budget"),
            ("anthropic.claude-sonnet-4-20250514-v1:0", "budget"),
            ("claude-3-7-sonnet-latest", "budget"),
            ("claude-3-5-haiku-20241022", "none"),
            ("us.anthropic.claude-3-5-haiku-20241022-v1:0", "none"),
            ("claude-3-opus-20240229", "none"),
        ],
    )
    def test_model_classes(self, model: str, cls: str) -> None:
        assert claude_thinking_class(model) == cls

    def test_claude_3_without_extended_thinking_has_no_control(self) -> None:
        assert (
            AICompletionsCapabilitiesAnthropic.for_model(
                "claude-3-5-haiku-20241022"
            ).thinking
            is None
        )
        fields, reason = claude_thinking_request_fields(
            "claude-3-5-haiku-20241022", L.MEDIUM, max_tokens=16000
        )
        assert fields == {} and reason == "no_thinking_control"
        # The native engine retires this id; Bedrock still catalogs it.
        client = _bedrock("us.anthropic.claude-3-5-haiku-20241022-v1:0")
        client.send_prompt("hi")
        assert (
            "additionalModelRequestFields"
            not in client.client.converse.call_args.kwargs
        )
        assert client.last_thinking.applied is None

    def test_always_on_sends_effort_only(self) -> None:
        fields, reason = claude_thinking_request_fields(
            "claude-opus-5-5", L.HIGH, max_tokens=16000
        )
        assert fields == {"output_config": {"effort": "high"}} and reason is None
        caps = AICompletionsCapabilitiesAnthropic.for_model("claude-opus-5-5").thinking
        assert caps is not None and caps.always_on
        assert caps.nearest(L.OFF) is L.LOW

    def test_adaptive_sends_adaptive_plus_effort_and_disabled_for_off(self) -> None:
        fields, _ = claude_thinking_request_fields("claude-opus-5", L.MAX, max_tokens=1)
        assert fields == {
            "output_config": {"effort": "max"},
            "thinking": {"type": "adaptive"},
        }
        off, _ = claude_thinking_request_fields("claude-opus-5", L.OFF, max_tokens=1)
        assert off == {"thinking": {"type": "disabled"}}

    def test_sonnet_5_5_off_is_between_tools(self) -> None:
        off, _ = claude_thinking_request_fields(
            "claude-sonnet-5-5", L.OFF, max_tokens=1
        )
        assert off == {"thinking": {"type": "between_tools"}}

    def test_budget_model_tiers_and_clamp(self) -> None:
        fields, _ = claude_thinking_request_fields(
            "claude-haiku-4-5", L.MEDIUM, max_tokens=16000
        )
        assert fields == {"thinking": {"type": "enabled", "budget_tokens": 4096}}
        # The cap keeps 1024 tokens free for the answer.
        clamped, _ = claude_thinking_request_fields(
            "claude-haiku-4-5", L.MAX, max_tokens=8000
        )
        assert clamped["thinking"]["budget_tokens"] == 8000 - 1024
        at_default, _ = claude_thinking_request_fields(
            "claude-haiku-4-5", L.MEDIUM, max_tokens=2048
        )
        assert at_default["thinking"]["budget_tokens"] == 1024
        none, reason = claude_thinking_request_fields(
            "claude-haiku-4-5", L.LOW, max_tokens=2047
        )
        assert none == {} and reason == "max_tokens_too_small"
        off, _ = claude_thinking_request_fields("claude-haiku-4-5", L.OFF, max_tokens=1)
        assert off == {}


class TestOpenAIAndGeminiTables:
    def test_openai_families(self) -> None:
        gpt6 = AICompletionsCapabilitiesOpenAI.for_model("gpt-6-sol").thinking
        assert gpt6 is not None and L.OFF in gpt6.levels and L.MAX in gpt6.levels
        astra = AICompletionsCapabilitiesOpenAI.for_model("gpt-6-astra").thinking
        assert astra is not None and L.OFF not in astra.levels
        gpt5 = AICompletionsCapabilitiesOpenAI.for_model("gpt-5").thinking
        assert gpt5 is not None and gpt5.levels[0] is L.MINIMAL
        assert gpt5.nearest(L.MAX) is L.HIGH
        # Snapshots and siblings of the launch family match by prefix.
        for name in ("gpt-5-mini-2025-08-07", "gpt-5-pro", "gpt-5-codex"):
            snapshot = AICompletionsCapabilitiesOpenAI.for_model(name).thinking
            assert snapshot is not None and snapshot.levels == gpt5.levels, name
        assert AICompletionsCapabilitiesOpenAI.for_model("gpt-4.1").thinking is None
        o4 = AICompletionsCapabilitiesOpenAI.for_model("o4-mini").thinking
        assert o4 is not None and o4.levels == [L.LOW, L.MEDIUM, L.HIGH]

    def test_gemini_families(self) -> None:
        flash3 = AICompletionsCapabilitiesGoogle.for_model("gemini-3.7-flash").thinking
        assert flash3 is not None and flash3.always_on
        assert gemini_thinking_config("gemini-3.7-flash", L.MEDIUM) == {
            "thinking_level": "MEDIUM"
        }
        pro3 = AICompletionsCapabilitiesGoogle.for_model("gemini-3-pro").thinking
        assert pro3 is not None and pro3.levels == [L.LOW, L.HIGH]
        flash25 = AICompletionsCapabilitiesGoogle.for_model("gemini-2.5-flash").thinking
        assert flash25 is not None and L.OFF in flash25.levels
        assert gemini_thinking_config("gemini-2.5-flash", L.OFF) == {
            "thinking_budget": 0
        }
        assert gemini_thinking_config("gemini-2.5-pro", L.MAX) == {
            "thinking_budget": 32768
        }
        pro25 = AICompletionsCapabilitiesGoogle.for_model("gemini-2.5-pro").thinking
        assert pro25 is not None and L.OFF not in pro25.levels
        assert (
            AICompletionsCapabilitiesGoogle.for_model("gemini-2.0-flash").thinking
            is None
        )

    def test_bedrock_only_claude_has_a_control(self) -> None:
        claude = AICompletionsCapabilitiesBedrock.for_model(
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0", context_window_length=1
        )
        assert claude.thinking is not None and L.OFF in claude.thinking.levels
        nova = AICompletionsCapabilitiesBedrock.for_model(
            "amazon.nova-lite-v1:0", context_window_length=1
        )
        assert nova.thinking is None


# ── The level reaches each provider's request ───────────────────────────────


class TestAnthropicRequests:
    def test_default_medium_sends_adaptive_and_effort(self) -> None:
        client = _anthropic("claude-opus-4-8")
        client.send_prompt("hi")
        kwargs = client.client.messages.create.call_args.kwargs
        assert kwargs["thinking"] == {"type": "adaptive"}
        assert kwargs["output_config"] == {"effort": "medium"}
        assert client.last_thinking == AIThinkingResolution(
            requested=L.MEDIUM, applied=L.MEDIUM, exact=True
        )

    def test_per_call_params_override_the_client_level(self) -> None:
        client = _anthropic("claude-opus-4-8")
        client.thinking_level = "high"
        client.send_prompt("hi", other_params=_Params(thinking=L.OFF))
        kwargs = client.client.messages.create.call_args.kwargs
        assert kwargs["thinking"] == {"type": "disabled"}
        assert "output_config" not in kwargs
        client.send_prompt("hi")
        assert client.client.messages.create.call_args.kwargs["output_config"] == {
            "effort": "high"
        }

    def test_always_on_model_records_nearest_level_and_logs_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _anthropic("claude-opus-5-5")
        client.thinking_level = L.OFF
        with caplog.at_level(logging.DEBUG, logger="ai_api_unified.ai_base"):
            client.send_prompt("hi")
            client.send_prompt("hi again")
        kwargs = client.client.messages.create.call_args.kwargs
        assert "thinking" not in kwargs
        assert kwargs["output_config"] == {"effort": "low"}
        assert client.last_thinking.applied is L.LOW
        assert client.last_thinking.reason == "unsupported_level"
        events = [r for r in caplog.records if hasattr(r, "ai_thinking_event")]
        # One line per client per distinct resolution, at INFO.
        assert len(events) == 1 and events[0].levelno == logging.INFO
        assert events[0].thinking_requested == "off"
        assert events[0].thinking_applied == "low"

    def test_budget_model_sends_budget_tokens(self) -> None:
        client = _anthropic("claude-haiku-4-5")
        client.send_prompt("hi", max_response_tokens=8000)
        kwargs = client.client.messages.create.call_args.kwargs
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4096}
        assert "output_config" not in kwargs

    def test_conversation_provider_option_override_is_stripped_and_applied(
        self,
    ) -> None:
        client = _anthropic("claude-opus-4-8")
        turn = client.send_conversation(
            "sys",
            [{"role": "user", "content": "hi"}],
            provider_options={"thinking_level": "max", "metadata": {"k": "v"}},
        )
        kwargs = client.client.messages.create.call_args.kwargs
        assert kwargs["output_config"] == {"effort": "max"}
        assert "thinking_level" not in kwargs and kwargs["metadata"] == {"k": "v"}
        assert turn.thinking is not None and turn.thinking.requested is L.MAX
        # The override does not leak past the call.
        assert THINKING_CALL_OVERRIDE.get() is None

    def test_forced_tool_choice_turns_thinking_off_and_records_why(self) -> None:
        client = _anthropic("claude-opus-4-8")
        tool = Mock()
        tool.name = "lookup"
        tool.description = "d"
        tool.input_schema = {"type": "object", "properties": {}}
        with patch.object(
            client, "_build_provider_tools", return_value=[{"name": "lookup"}]
        ):
            with patch.object(
                client, "_validate_conversation_request", return_value=[tool]
            ):
                client.send_conversation(
                    "sys",
                    [{"role": "user", "content": "hi"}],
                    tools=[tool],
                    tool_choice="lookup",
                )
        kwargs = client.client.messages.create.call_args.kwargs
        assert kwargs["thinking"] == {"type": "disabled"}
        assert client.last_thinking.applied is L.OFF
        assert client.last_thinking.reason == "forced_tool_choice"

    def test_native_provider_option_still_wins(self) -> None:
        client = _anthropic("claude-opus-4-8")
        client.send_conversation(
            "sys",
            [{"role": "user", "content": "hi"}],
            provider_options={"thinking": {"type": "disabled"}},
        )
        kwargs = client.client.messages.create.call_args.kwargs
        assert kwargs["thinking"] == {"type": "disabled"}

    def test_structured_output_keeps_format_and_adds_effort(self) -> None:
        client = _anthropic("claude-opus-4-8")
        client.client.messages.create.return_value = Mock(
            content=[Mock(type="text", text='{"word": "pong"}')],
            stop_reason="end_turn",
            usage=Mock(input_tokens=10, output_tokens=2, cache_read_input_tokens=None),
        )
        result = client.send_structured_output("q", response_model=_Word)
        kwargs = client.client.messages.create.call_args.kwargs
        assert kwargs["output_config"]["format"]["type"] == "json_schema"
        assert kwargs["output_config"]["effort"] == "medium"
        assert result.thinking is not None and result.thinking.exact


class TestOpenAIRequests:
    def test_chat_sends_reasoning_effort(self) -> None:
        client = _openai("gpt-5.4")
        client.client.chat.completions.create.return_value = _chat_response()
        client.send_prompt("hi")
        kwargs = client.client.chat.completions.create.call_args.kwargs
        assert kwargs["reasoning_effort"] == "medium"
        client.send_prompt("hi", other_params=_Params(thinking=L.MAX))
        assert (
            client.client.chat.completions.create.call_args.kwargs["reasoning_effort"]
            == "xhigh"
        )
        client.send_prompt("hi", other_params=_Params(thinking=L.OFF))
        assert (
            client.client.chat.completions.create.call_args.kwargs["reasoning_effort"]
            == "none"
        )

    def test_model_without_a_control_sends_nothing_and_logs_at_debug_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _openai("gpt-4.1")
        client.client.chat.completions.create.return_value = _chat_response()
        with caplog.at_level(logging.DEBUG, logger="ai_api_unified.ai_base"):
            client.send_prompt("hi")
            client.send_prompt("hi")
        assert (
            "reasoning_effort"
            not in client.client.chat.completions.create.call_args.kwargs
        )
        assert client.last_thinking.applied is None
        assert client.last_thinking.reason == "no_thinking_control"
        events = [r for r in caplog.records if hasattr(r, "ai_thinking_event")]
        assert len(events) == 1 and events[0].levelno == logging.DEBUG

    def test_compatible_server_without_catalog_sends_nothing(self) -> None:
        client = _openai_compatible()
        client.client.chat.completions.create.return_value = _chat_response()
        client.send_prompt("hi")
        assert (
            "reasoning_effort"
            not in client.client.chat.completions.create.call_args.kwargs
        )

    def test_responses_nests_the_effort(self) -> None:
        client = _openai("gpt-5.4", cls=AiOpenAIResponsesCompletions)
        client.client.responses.create.return_value = Mock(
            output_text="ok",
            output=[],
            status="completed",
            usage=Mock(
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                input_tokens_details=None,
            ),
        )
        client.send_prompt("hi", other_params=_Params(thinking=L.HIGH))
        assert client.client.responses.create.call_args.kwargs["reasoning"] == {
            "effort": "high"
        }


class TestGeminiRequests:
    def test_conversation_config_carries_thinking_budget(self) -> None:
        mock_client = Mock()
        client = _build_gemini_client(mock_client)
        kwargs = client._build_gemini_conversation_config_kwargs(
            system_prompt="s",
            tools=[],
            tool_choice=None,
            max_response_tokens=100,
            request_timeout_seconds=None,
            dict_merge_options={},
        )
        assert kwargs["thinking_config"] == {"thinking_budget": 8192}
        token = THINKING_CALL_OVERRIDE.set(L.OFF)
        try:
            kwargs_off = client._build_gemini_conversation_config_kwargs(
                system_prompt="s",
                tools=[],
                tool_choice=None,
                max_response_tokens=100,
                request_timeout_seconds=None,
                dict_merge_options={},
            )
        finally:
            THINKING_CALL_OVERRIDE.reset(token)
        assert kwargs_off["thinking_config"] == {"thinking_budget": 0}

    def test_caller_thinking_config_wins(self) -> None:
        client = _build_gemini_client(Mock())
        kwargs = client._build_gemini_conversation_config_kwargs(
            system_prompt="s",
            tools=[],
            tool_choice=None,
            max_response_tokens=100,
            request_timeout_seconds=None,
            dict_merge_options={"thinking_config": {"thinking_budget": 512}},
        )
        assert kwargs["thinking_config"] == {"thinking_budget": 512}


class TestBedrockRequests:
    def test_budget_claude_sends_fields_and_drops_sampling(self) -> None:
        client = _bedrock("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        client.send_prompt("hi", max_response_tokens=8000)
        kwargs = client.client.converse.call_args.kwargs
        assert kwargs["additionalModelRequestFields"] == {
            "thinking": {"type": "enabled", "budget_tokens": 4096}
        }
        assert "temperature" not in kwargs["inferenceConfig"]
        assert "topP" not in kwargs["inferenceConfig"]

    def test_adaptive_claude_keeps_sampling_and_sends_effort(self) -> None:
        client = _bedrock("us.anthropic.claude-sonnet-4-6-v1")
        client.send_prompt("hi", max_response_tokens=8000)
        kwargs = client.client.converse.call_args.kwargs
        assert kwargs["additionalModelRequestFields"] == {
            "output_config": {"effort": "medium"},
            "thinking": {"type": "adaptive"},
        }
        assert kwargs["inferenceConfig"]["temperature"] == 0.2

    def test_budget_claude_default_cap_rises_so_the_level_applies(self) -> None:
        client = _bedrock("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        client.send_prompt("hi")
        kwargs = client.client.converse.call_args.kwargs
        assert kwargs["inferenceConfig"]["maxTokens"] == 4096
        assert kwargs["additionalModelRequestFields"]["thinking"] == {
            "type": "enabled",
            "budget_tokens": 3072,
        }
        assert client.last_thinking.exact
        # An explicit small cap is the caller's and stays; thinking is off.
        client.send_prompt("hi", max_response_tokens=512)
        kwargs = client.client.converse.call_args.kwargs
        assert kwargs["inferenceConfig"]["maxTokens"] == 512
        assert "additionalModelRequestFields" not in kwargs
        assert client.last_thinking.reason == "max_tokens_too_small"

    def test_nova_sends_nothing(self) -> None:
        client = _bedrock("amazon.nova-lite-v1:0")
        client.send_prompt("hi")
        assert (
            "additionalModelRequestFields"
            not in client.client.converse.call_args.kwargs
        )
        assert client.last_thinking.applied is None


# ── Configuration and the factory ───────────────────────────────────────────


class TestConfiguration:
    def test_models_json_sets_the_level_and_the_default_is_medium(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _anthropic("claude-opus-4-8")
        assert client.thinking_level is L.MEDIUM
        monkeypatch.setenv("AI_MODELS_CONFIG", '{"completions": {"thinking": "high"}}')
        client.thinking_level = None  # re-read configuration
        assert client.thinking_level is L.HIGH

    def test_models_json_field_feeds_the_setting(self) -> None:
        config = load_models_config('{"completions": {"thinking": "low"}}')
        assert config.completions.thinking == "low"

    def test_sample_file_sets_a_level(self) -> None:
        from pathlib import Path

        config = load_models_config(
            str(Path(__file__).resolve().parents[1] / "models_sample.json")
        )
        assert parse_thinking_level(config.completions.thinking) is L.MEDIUM

    def test_factory_argument_levels_primary_and_fallbacks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "AI_MODELS_CONFIG",
            '{"completions": {"engine": "claude", "model": "claude-opus-4-8"},'
            ' "model_check": {"mode": "off"}}',
        )
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k", "OPENAI_API_KEY": "k"}):
            chain = AIFactory.get_ai_completions_client(
                fallbacks=[("openai", "gpt-5.4")], thinking="high"
            )
            assert chain.primary.thinking_level is L.HIGH
            fallback = chain._client_at(1)
            assert fallback is not None and fallback.thinking_level is L.HIGH
            plain = AIFactory.get_ai_completions_client(fallbacks=[])
            assert plain.thinking_level is L.MEDIUM

    def test_wrappers_expose_last_thinking(self) -> None:
        client = _anthropic("claude-opus-4-8")
        from ai_api_unified.completions.ai_fallback_completions import (
            AIFallbackCandidate,
            AiFallbackCompletions,
        )

        chain = AiFallbackCompletions(
            primary=client,
            primary_candidate=AIFallbackCandidate(
                engine="claude", model="claude-opus-4-8"
            ),
            fallback_candidates=[],
            client_builder=lambda c: client,
        )
        chain.send_prompt("hi")
        assert chain.last_thinking is not None and chain.last_thinking.exact

    def test_observability_events_carry_this_calls_levels(self) -> None:
        """The call context is built before the provider call runs."""
        client = _anthropic("claude-opus-5-5")
        seen: list[dict[str, Any]] = []
        spy = Mock()
        spy.bool_enabled = True
        spy.before_call = lambda call_context: seen.append(
            dict(call_context.dict_metadata)
        )
        spy.after_call = lambda **kwargs: None
        spy.on_error = lambda **kwargs: None
        client._observability_middleware = spy
        client.send_prompt("hi", other_params=_Params(thinking=L.OFF))
        client.send_prompt("hi", other_params=_Params(thinking=L.HIGH))
        assert seen[0]["thinking_requested"] == "off"
        assert seen[0]["thinking_applied"] == "low"
        assert seen[0]["thinking_exact"] is False
        assert seen[1]["thinking_requested"] == "high"
        assert seen[1]["thinking_exact"] is True

    def test_wrapper_setter_reaches_every_candidate(self) -> None:
        from ai_api_unified.completions.ai_fallback_completions import (
            AIFallbackCandidate,
            AiFallbackCompletions,
        )

        primary = _anthropic("claude-opus-4-8")
        fallback = _anthropic("claude-sonnet-5")
        chain = AiFallbackCompletions(
            primary=primary,
            primary_candidate=AIFallbackCandidate(
                engine="claude", model="claude-opus-4-8"
            ),
            fallback_candidates=[
                AIFallbackCandidate(engine="claude", model="claude-sonnet-5")
            ],
            client_builder=lambda c: fallback,
        )
        chain.thinking_level = "off"
        assert chain.thinking_level is L.OFF and primary.thinking_level is L.OFF
        # Built lazily after the assignment, the fallback still gets the level.
        assert chain._client_at(1) is fallback and fallback.thinking_level is L.OFF
        chain.send_prompt("hi")
        assert primary.client.messages.create.call_args.kwargs["thinking"] == {
            "type": "disabled"
        }

    def test_router_setter_reaches_default_route_and_tiers(self) -> None:
        from tests.test_model_router import _FakeClassifier, _engines, _router

        engines = _engines("e", "h")
        router = _router(engines, classifier=_FakeClassifier("hard"))
        router.thinking_level = "high"
        assert router.thinking_level is L.HIGH
        assert router.default_client.thinking_level is L.HIGH
        router.send_prompt("p")
        assert engines["claude:h"].thinking_level is L.HIGH


# ── Native thinking fields, the classifier's level, batches ─────────────────


class TestNativeThinkingOptions:
    @pytest.mark.parametrize(
        ("options", "path"),
        [
            ({"thinking": {"type": "disabled"}}, "thinking"),
            ({"output_config": {"effort": "low"}}, "output_config.effort"),
            ({"output_config": {"format": {}}}, None),
            ({"temperature": 0}, None),
            (None, None),
        ],
    )
    def test_claude_paths(self, options: Any, path: str | None) -> None:
        assert _anthropic("claude-opus-4-8")._native_thinking_option(options) == path

    def test_every_engine_declares_its_fields(self) -> None:
        assert _openai("gpt-5.4")._native_thinking_option(
            {"reasoning_effort": "low"}
        ) == ("reasoning_effort")
        assert (
            _openai(
                "gpt-5.4", cls=AiOpenAIResponsesCompletions
            )._native_thinking_option({"reasoning": {"effort": "low"}})
            == "reasoning"
        )
        assert (
            _build_gemini_client(Mock())._native_thinking_option(
                {"thinking_config": {"thinking_budget": 0}}
            )
            == "thinking_config"
        )
        assert (
            _bedrock(
                "us.anthropic.claude-haiku-4-5-20251001-v1:0"
            )._native_thinking_option(
                {"additionalModelRequestFields": {"thinking": {"type": "disabled"}}}
            )
            == "additionalModelRequestFields.thinking"
        )

    def test_native_field_is_kept_recorded_as_caller_set_and_warned_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _anthropic("claude-opus-4-8")
        with caplog.at_level(logging.WARNING, logger="ai_api_unified.ai_base"):
            for _ in range(2):
                client.send_conversation(
                    "sys",
                    [{"role": "user", "content": "hi"}],
                    provider_options={"thinking": {"type": "disabled"}},
                )
        kwargs = client.client.messages.create.call_args.kwargs
        assert kwargs["thinking"] == {"type": "disabled"}
        # The layer added nothing of its own beside the caller's field.
        assert "effort" not in (kwargs.get("output_config") or {})
        resolution = client.last_thinking
        assert resolution.requested is L.MEDIUM
        assert resolution.applied is None and not resolution.exact
        assert resolution.reason == "provider_option"
        warned = [
            r
            for r in caplog.records
            if getattr(r, "ai_thinking_event", "") == "provider_option"
        ]
        assert len(warned) == 1 and warned[0].provider_option == "thinking"
        assert "thinking_level" in warned[0].getMessage()

    def test_the_next_call_without_it_applies_the_level_again(self) -> None:
        client = _anthropic("claude-opus-4-8")
        client.send_conversation(
            "sys",
            [{"role": "user", "content": "hi"}],
            provider_options={"thinking": {"type": "disabled"}},
        )
        client.send_conversation("sys", [{"role": "user", "content": "hi"}])
        assert client.last_thinking.reason is None
        assert client.last_thinking.applied is L.MEDIUM

    def test_neutral_option_is_not_a_native_field(self) -> None:
        client = _anthropic("claude-opus-4-8")
        client.send_conversation(
            "sys",
            [{"role": "user", "content": "hi"}],
            provider_options={"thinking_level": "high"},
        )
        assert client.last_thinking.applied is L.HIGH


class TestClassifierThinking:
    def _build(self, **kwargs: Any) -> Any:
        from ai_api_unified.classification.ai_structured_output_classifier import (
            AiStructuredOutputClassifier,
        )

        engine = Mock()
        engine.capabilities.supports_structured_output = True
        engine.model_name = "m"
        engine.thinking_level = "untouched"
        with patch(
            "ai_api_unified.ai_factory.AIFactory._build_completions_client",
            return_value=engine,
        ):
            AiStructuredOutputClassifier(engine="claude", **kwargs)
        return engine

    def test_a_built_client_classifies_with_thinking_off_by_default(self) -> None:
        assert self._build().thinking_level is L.OFF

    def test_explicit_and_configured_levels(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert self._build(thinking="low").thinking_level is L.LOW
        monkeypatch.setenv(
            "AI_MODELS_CONFIG", '{"classifier": {"thinking": "minimal"}}'
        )
        from ai_api_unified.ai_models_config import clear_models_config_cache

        clear_models_config_cache()
        try:
            assert self._build().thinking_level is L.MINIMAL
        finally:
            clear_models_config_cache()

    def test_a_callers_client_keeps_its_level_unless_one_is_passed(self) -> None:
        from ai_api_unified.classification.ai_structured_output_classifier import (
            AiStructuredOutputClassifier,
        )

        engine = Mock()
        engine.capabilities.supports_structured_output = True
        engine.model_name = "m"
        engine.thinking_level = L.HIGH
        AiStructuredOutputClassifier(completions_client=engine)
        assert engine.thinking_level is L.HIGH
        AiStructuredOutputClassifier(completions_client=engine, thinking="off")
        assert engine.thinking_level is L.OFF

    def test_bad_configured_level_fails_at_load(self) -> None:
        from ai_api_unified.ai_models_config import AIModelsConfig

        with pytest.raises(ValueError):
            AIModelsConfig.model_validate({"classifier": {"thinking": "lots"}})


class TestBatchThinking:
    def test_each_item_gets_the_mapping_and_the_job_records_it(self) -> None:
        from types import SimpleNamespace

        from ai_api_unified.ai_base import AIBatchRequestItem

        client = _anthropic("claude-opus-4-8")
        client.client.messages.batches.create.return_value = SimpleNamespace(
            id="msgbatch_1",
            processing_status="in_progress",
            request_counts=SimpleNamespace(
                processing=2, succeeded=0, errored=0, canceled=0, expired=0
            ),
            created_at=None,
            ended_at=None,
        )
        job = client.submit_batch(
            [
                AIBatchRequestItem(custom_id="a", prompt="x"),
                AIBatchRequestItem(custom_id="b", prompt="y", thinking=L.HIGH),
            ]
        )
        list_requests = client.client.messages.batches.create.call_args.kwargs[
            "requests"
        ]
        dict_params = {r["custom_id"]: r["params"] for r in list_requests}
        assert dict_params["a"]["output_config"]["effort"] == "medium"
        assert dict_params["b"]["output_config"]["effort"] == "high"
        assert job.thinking["a"].requested is L.MEDIUM
        assert job.thinking["b"].applied is L.HIGH
