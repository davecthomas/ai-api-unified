# ruff: noqa: E402

# test_model_catalog_nov_2026.py
"""
Tests for the 2026-11-15 provider model sweep.

Covers the newly cataloged Claude, GPT-6.1, Gemini 4, and image models with
their registry pricing, the forced-tool_choice guard on Sonnet 5.5 and Fable
5.2, the Chat Completions tool guard on GPT-6 Astra and GPT-6.1, Gemini
thinking levels on 3.8 Flash and Gemini 4, the default moves, and the
lifecycle retirements and deprecations.
"""

import os
from datetime import date
from decimal import Decimal
from typing import Any
from unittest.mock import Mock, patch

import pytest

from ai_api_unified.ai_base import AITool
from ai_api_unified.ai_provider_exceptions import AiProviderConfigurationError
from ai_api_unified.pricing.model_pricing import ModelLifecycleStatus
from ai_api_unified.pricing.pricing_registry import (
    enforce_model_lifecycle,
    get_model_info,
    get_model_pricing,
)
from ai_api_unified.thinking import (
    AIThinkingLevel,
    claude_thinking_class,
    gemini_thinking_capabilities,
    gemini_thinking_config,
)

WEATHER_TOOL = AITool(
    name="get_weather",
    description="Get current weather for a city.",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


class TestRegistryAdditions:
    """New models carry sourced rates."""

    @pytest.mark.parametrize(
        ("str_model", "str_input", "str_output", "str_cached"),
        [
            ("claude-sonnet-5-5", "2.00", "10.00", "0.10"),
            ("claude-haiku-5-5", "0.10", "0.50", "0.01"),
            ("claude-fable-5-2", "8.00", "40.00", "0.20"),
        ],
    )
    def test_claude_rates(
        self, str_model: str, str_input: str, str_output: str, str_cached: str
    ) -> None:
        pricing = get_model_pricing("anthropic", str_model)
        assert pricing is not None
        assert pricing.token_rates.input_per_1m == Decimal(str_input)
        assert pricing.token_rates.output_per_1m == Decimal(str_output)
        assert pricing.token_rates.cached_input_per_1m == Decimal(str_cached)

    def test_haiku_5_5_long_prompt_tier(self) -> None:
        pricing = get_model_pricing("anthropic", "claude-haiku-5-5")
        assert pricing is not None and pricing.tiers
        tier = pricing.tiers[0]
        assert tier.label == "context>100k"
        assert tier.token_rates.input_per_1m == Decimal("0.50")
        assert tier.token_rates.output_per_1m == Decimal("2.50")

    @pytest.mark.parametrize(
        ("str_model", "str_input", "str_cached", "str_long_input"),
        [
            ("gpt-6.1-sol", "2.00", "0.10", "4.00"),
            ("gpt-6.1-terra", "1.00", "0.05", "2.00"),
        ],
    )
    def test_gpt_6_1_rates_and_long_context_tier(
        self, str_model: str, str_input: str, str_cached: str, str_long_input: str
    ) -> None:
        pricing = get_model_pricing("openai", str_model)
        assert pricing is not None
        assert pricing.token_rates.input_per_1m == Decimal(str_input)
        assert pricing.token_rates.cached_input_per_1m == Decimal(str_cached)
        assert pricing.tiers is not None
        assert pricing.tiers[0].token_rates.input_per_1m == Decimal(str_long_input)

    def test_gemini_4_rates(self) -> None:
        for str_model in ("gemini-4-carbon", "gemini-4-argon"):
            pricing = get_model_pricing("google", str_model)
            assert pricing is not None
            assert pricing.token_rates.input_per_1m == Decimal("4.00")
            assert pricing.token_rates.output_per_1m == Decimal("20.00")
        hydrogen = get_model_pricing("google", "gemini-4-hydrogen")
        assert hydrogen is not None
        assert hydrogen.token_rates.input_per_1m == Decimal("1.50")
        assert hydrogen.token_rates.cached_input_per_1m == Decimal("0.075")


class TestLifecycleChanges:
    @pytest.mark.parametrize(
        ("str_provider", "str_model"),
        [
            ("openai", "o4-mini"),
            ("openai", "gpt-4.1-nano"),
            ("openai", "gpt-5.1-codex-max"),
            ("openai", "gpt-image-1"),
        ],
    )
    def test_retired_models_raise(self, str_provider: str, str_model: str) -> None:
        with pytest.raises(AiProviderConfigurationError, match="retired"):
            enforce_model_lifecycle(str_provider, str_model)

    @pytest.mark.parametrize(
        ("str_provider", "str_model", "sunset", "str_replacement"),
        [
            ("anthropic", "claude-sonnet-4-5", date(2026, 11, 30), "claude-sonnet-5-5"),
            ("google", "gemini-2.5-pro", date(2026, 12, 1), "gemini-3.8-flash"),
            ("google", "gemini-2.5-flash", date(2026, 12, 1), "gemini-3.5-flash-lite"),
            ("google", "gemini-3.7-flash", None, "gemini-3.8-flash"),
            ("google", "gemini-3.5-flash", None, "gemini-3.6-flash"),
            (
                "google",
                "gemini-3.1-flash-lite",
                date(2027, 5, 7),
                "gemini-3.5-flash-lite",
            ),
            ("google", "gemini-3.1-flash-image", None, "gemini-nano-banana-2.1"),
            ("openai", "gpt-4o", date(2027, 1, 14), "gpt-6-luna"),
            ("openai", "gpt-4.1-mini", date(2027, 1, 14), "gpt-6-luna"),
            ("openai", "gpt-5.4-nano", date(2027, 4, 1), "gpt-6-luna"),
            ("openai", "gpt-image-2", date(2027, 2, 10), "gpt-image-2.6-sunburst"),
        ],
    )
    def test_deprecations(
        self,
        str_provider: str,
        str_model: str,
        sunset: date | None,
        str_replacement: str,
    ) -> None:
        info = get_model_info(str_provider, str_model)
        assert info is not None
        assert info.status is ModelLifecycleStatus.DEPRECATED
        assert info.sunset_date == sunset
        assert info.recommended_replacement == str_replacement


class TestThinking:
    def test_claude_classes(self) -> None:
        assert claude_thinking_class("claude-fable-5-2") == "always_on"
        assert claude_thinking_class("claude-sonnet-5-5") == "between_tools"
        assert claude_thinking_class("claude-haiku-5-5") == "adaptive"
        assert claude_thinking_class("us.anthropic.claude-sonnet-5-5") == (
            "between_tools"
        )

    @pytest.mark.parametrize(
        "str_model",
        ["gemini-4-carbon", "gemini-4-argon", "gemini-4-hydrogen", "gemini-3.8-flash"],
    )
    def test_gemini_minimal_not_offered(self, str_model: str) -> None:
        capabilities = gemini_thinking_capabilities(str_model)
        assert capabilities is not None
        assert capabilities.levels == [
            AIThinkingLevel.LOW,
            AIThinkingLevel.MEDIUM,
            AIThinkingLevel.HIGH,
        ]
        assert gemini_thinking_config(str_model, AIThinkingLevel.LOW) == {
            "thinking_level": "LOW"
        }

    def test_gemini_3_7_keeps_minimal(self) -> None:
        capabilities = gemini_thinking_capabilities("gemini-3.7-flash")
        assert capabilities is not None
        assert AIThinkingLevel.MINIMAL in capabilities.levels


# ── Anthropic engine ────────────────────────────────────────────────────────

pytest.importorskip("anthropic")

from ai_api_unified.completions.ai_anthropic_completions import (
    AiAnthropicCompletions,
)


def _build_claude_client(model: str) -> AiAnthropicCompletions:
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        client = AiAnthropicCompletions(model=model)
    client.client = Mock()
    return client


class TestClaudeEngine:
    @pytest.mark.parametrize(
        "str_model", ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-2"]
    )
    def test_new_models_listed_with_1m_context(self, str_model: str) -> None:
        client = _build_claude_client(str_model)
        assert str_model in client.list_model_names
        assert client.max_context_tokens == 1_000_000
        assert client.capabilities.pricing is not None

    @pytest.mark.parametrize("str_model", ["claude-sonnet-5-5", "claude-fable-5-2"])
    def test_forced_tool_choice_fails_before_the_request(self, str_model: str) -> None:
        client = _build_claude_client(str_model)
        messages = [{"role": "user", "content": "Weather in SF?"}]
        with pytest.raises(ValueError, match="forced tool_choice"):
            client.send_conversation(
                "sys", messages, tools=[WEATHER_TOOL], tool_choice="get_weather"
            )
        client.client.messages.create.assert_not_called()

    def test_haiku_5_5_still_sends_forced_tool_choice(self) -> None:
        client = _build_claude_client("claude-haiku-5-5")
        kwargs: dict[str, Any] = client._build_conversation_request_kwargs(
            system_prompt="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=[WEATHER_TOOL],
            tool_choice="get_weather",
            max_response_tokens=None,
            dict_merge_options={},
        )
        assert kwargs["tool_choice"] == {"type": "tool", "name": "get_weather"}


# ── OpenAI engines ──────────────────────────────────────────────────────────

pytest.importorskip("openai")

from ai_api_unified.completions.ai_openai_completions import (
    AICompletionsCapabilitiesOpenAI,
    AiOpenAICompletions,
)
from ai_api_unified.images.ai_openai_images import AIOpenAIImages


def _build_openai_client(model: str) -> AiOpenAICompletions:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        client = AiOpenAICompletions(model=model)
    client.client = Mock()
    return client


class TestOpenAIEngines:
    @pytest.mark.parametrize("str_model", ["gpt-6.1-sol", "gpt-6.1-terra"])
    def test_gpt_6_1_capabilities(self, str_model: str) -> None:
        capabilities = AICompletionsCapabilitiesOpenAI.for_model(str_model)
        assert capabilities.context_window_length == 1_050_000
        assert capabilities.reasoning is True
        assert capabilities.pricing is not None

    def test_gpt_6_1_sol_rejects_off_terra_allows_it(self) -> None:
        sol = AICompletionsCapabilitiesOpenAI.for_model("gpt-6.1-sol").thinking
        terra = AICompletionsCapabilitiesOpenAI.for_model("gpt-6.1-terra").thinking
        assert sol is not None and AIThinkingLevel.OFF not in sol.levels
        assert terra is not None and AIThinkingLevel.OFF in terra.levels

    def test_catalog(self) -> None:
        client = _build_openai_client("gpt-6.1-sol")
        for str_model in ("gpt-6.1-sol", "gpt-6.1-terra"):
            assert str_model in client.list_model_names
        for str_model in ("o4-mini", "gpt-4.1-nano", "gpt-5.1-codex-max"):
            assert str_model not in client.list_model_names

    @pytest.mark.parametrize("str_model", ["gpt-6-astra", "gpt-6.1-sol"])
    def test_chat_tools_raise_before_the_request(self, str_model: str) -> None:
        client = _build_openai_client(str_model)
        with pytest.raises(ValueError, match="openai-responses"):
            client._build_chat_conversation_request_kwargs(
                system_prompt="sys",
                messages=[{"role": "user", "content": "hi"}],
                tools=[WEATHER_TOOL],
                tool_choice=None,
                max_response_tokens=None,
                dict_merge_options={},
            )

    def test_chat_tools_still_sent_for_gpt_6_luna(self) -> None:
        client = _build_openai_client("gpt-6-luna")
        kwargs: dict[str, Any] = client._build_chat_conversation_request_kwargs(
            system_prompt="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=[WEATHER_TOOL],
            tool_choice=None,
            max_response_tokens=None,
            dict_merge_options={},
        )
        assert kwargs["tools"]

    def test_images_default_and_catalog(self) -> None:
        assert AIOpenAIImages.DEFAULT_IMAGE_MODEL == "gpt-image-2.5-sunburst"
        assert "gpt-image-2.6-sunburst" in AIOpenAIImages.SUPPORTED_IMAGE_MODELS
        assert "gpt-image-1" not in AIOpenAIImages.SUPPORTED_IMAGE_MODELS


# ── Bedrock engine ──────────────────────────────────────────────────────────

pytest.importorskip("boto3")

from ai_api_unified.completions.ai_bedrock_completions import (
    AiBedrockCompletions,
)


def _build_bedrock_client(model: str) -> AiBedrockCompletions:
    with patch("ai_api_unified.ai_bedrock_base.boto3"):
        client = AiBedrockCompletions(model=model)
    client.client = Mock()
    return client


class TestBedrockEngine:
    @pytest.mark.parametrize(
        "str_model",
        [
            "us.anthropic.claude-sonnet-5-5",
            "us.anthropic.claude-haiku-5-5",
            "us.anthropic.claude-fable-5-2",
        ],
    )
    def test_new_claude_profiles_cataloged(self, str_model: str) -> None:
        client = _build_bedrock_client(str_model)
        assert str_model in client.list_model_names
        assert client.max_context_tokens == 1_000_000

    @pytest.mark.parametrize(
        "str_model", ["us.anthropic.claude-sonnet-5-5", "us.anthropic.claude-fable-5-2"]
    )
    def test_forced_tool_choice_rejected(self, str_model: str) -> None:
        client = _build_bedrock_client(str_model)
        messages = [{"role": "user", "content": [{"text": "Weather?"}]}]
        with pytest.raises(ValueError, match="forced tool_choice"):
            client.send_conversation(
                "sys", messages, tools=[WEATHER_TOOL], tool_choice="get_weather"
            )
        client.client.converse.assert_not_called()


# ── Gemini engines ──────────────────────────────────────────────────────────

pytest.importorskip("google.genai")

from ai_api_unified.completions.ai_google_gemini_capabilities import (
    AICompletionsCapabilitiesGoogle,
)
from ai_api_unified.completions.ai_google_gemini_completions import (
    DEFAULT_COMPLETIONS_MODEL,
    GEMINI_MODEL_SPECS,
)
from ai_api_unified.images.ai_google_gemini_images import AIGoogleGeminiImages


class TestGeminiEngines:
    @pytest.mark.parametrize(
        "str_model", ["gemini-4-carbon", "gemini-4-argon", "gemini-4-hydrogen"]
    )
    def test_gemini_4_cataloged(self, str_model: str) -> None:
        assert GEMINI_MODEL_SPECS[str_model]["max_context_tokens"] == 1_048_576
        capabilities = AICompletionsCapabilitiesGoogle.for_model(str_model)
        assert capabilities.context_window_length == 1_048_576
        assert capabilities.reasoning is True
        assert capabilities.thinking is not None

    def test_default_moves_off_deprecated_3_7(self) -> None:
        assert DEFAULT_COMPLETIONS_MODEL == "gemini-3.8-flash"

    def test_images_default(self) -> None:
        assert AIGoogleGeminiImages.DEFAULT_IMAGE_MODEL == "gemini-nano-banana-2.1"
