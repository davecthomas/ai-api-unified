# ruff: noqa: E402

# test_model_catalog_sep_2026.py
"""
Tests for the 2026-09-25 provider model sweep.

Covers the newly catalogued models and their registry pricing, the lifecycle
retirements (Sora, Imagen 4, Veo 2/3.0, dall-e) and deprecations, the engine
default moves, the forced-tool_choice guard on Claude models that reject it,
and the Gemini images engine's native generate_content path.
"""

import os
from datetime import date
from decimal import Decimal
from types import SimpleNamespace
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

    def test_claude_opus_5_5_and_fable_5_1_rates(self) -> None:
        opus = get_model_pricing("anthropic", "claude-opus-5-5")
        assert opus is not None
        assert opus.token_rates.input_per_1m == Decimal("4.00")
        assert opus.token_rates.output_per_1m == Decimal("20.00")
        # Opus 5.5 reads cache at 0.05x, not the usual 0.1x.
        assert opus.token_rates.cached_input_per_1m == Decimal("0.20")

        fable = get_model_pricing("anthropic", "claude-fable-5-1")
        assert fable is not None
        assert fable.token_rates.input_per_1m == Decimal("10.00")
        # Fable 5.1 reads cache at 0.025x.
        assert fable.token_rates.cached_input_per_1m == Decimal("0.25")

    def test_gpt_6_family_carries_long_context_tier(self) -> None:
        for str_model, str_input in (
            ("gpt-6-astra", "10.00"),
            ("gpt-6-sol", "2.00"),
            ("gpt-6-luna", "0.10"),
            ("gpt-5.6-sol", "4.00"),
            ("gpt-5.6-terra", "2.00"),
            ("gpt-5.6-luna", "0.20"),
        ):
            pricing = get_model_pricing("openai", str_model)
            assert pricing is not None, str_model
            assert pricing.token_rates.input_per_1m == Decimal(str_input)
            assert pricing.tiers is not None and len(pricing.tiers) == 1
            tier_rates = pricing.tiers[0].token_rates
            assert tier_rates is not None
            # Above 272K input the whole request bills at 2x input.
            assert tier_rates.input_per_1m == Decimal(str_input) * 2

    def test_gemini_3_7_and_3_8_flash(self) -> None:
        for str_model in ("gemini-3.8-flash", "gemini-3.7-flash"):
            pricing = get_model_pricing("google", str_model)
            assert pricing is not None
            assert pricing.token_rates.input_per_1m == Decimal("1.50")
            assert pricing.token_rates.output_per_1m == Decimal("7.50")

    def test_voyage_4_family(self) -> None:
        assert get_model_pricing("voyage", "voyage-4-large") is not None
        pricing = get_model_pricing("voyage", "voyage-4-lite")
        assert pricing is not None
        assert pricing.token_rates.input_per_1m == Decimal("0.02")


class TestLifecycleChanges:
    """Retired models raise; deprecated ones warn with the shutdown date."""

    @pytest.mark.parametrize(
        ("str_provider", "str_model"),
        [
            ("openai", "sora-2"),
            ("openai", "sora-2-pro"),
            ("openai", "dall-e-3"),
            ("google", "imagen-4.0-generate-001"),
            ("google", "gemini-3-pro-image-preview"),
            ("google", "veo-3.0-generate-001"),
            ("google", "veo-2.0-generate-001"),
        ],
    )
    def test_retired_models_raise(self, str_provider: str, str_model: str) -> None:
        with pytest.raises(AiProviderConfigurationError, match="retired"):
            enforce_model_lifecycle(str_provider, str_model)

    @pytest.mark.parametrize(
        ("str_model", "sunset", "str_replacement"),
        [
            ("o4-mini", date(2026, 10, 23), "gpt-5.6-luna"),
            ("gpt-4.1-nano", date(2026, 10, 23), "gpt-5.6-luna"),
            ("gpt-image-1", date(2026, 10, 23), "gpt-image-2"),
            ("gpt-image-1.5", date(2026, 12, 1), "gpt-image-2"),
        ],
    )
    def test_openai_deprecations(
        self, str_model: str, sunset: date, str_replacement: str
    ) -> None:
        info = get_model_info("openai", str_model)
        assert info is not None
        assert info.status is ModelLifecycleStatus.DEPRECATED
        assert info.sunset_date == sunset
        assert info.recommended_replacement == str_replacement


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
    def test_new_models_listed_with_1m_context(self) -> None:
        client = _build_claude_client("claude-opus-5-5")
        assert "claude-opus-5-5" in client.list_model_names
        assert "claude-fable-5-1" in client.list_model_names
        assert client.max_context_tokens == 1_000_000

    def test_default_is_one_generation_behind(self) -> None:
        assert AiAnthropicCompletions.DEFAULT_COMPLETIONS_MODEL == "claude-opus-5"

    @pytest.mark.parametrize("str_model", ["claude-opus-5-5", "claude-fable-5-1"])
    def test_forced_tool_choice_fails_before_the_request(self, str_model: str) -> None:
        client = _build_claude_client(str_model)
        messages = [{"role": "user", "content": "Weather in SF?"}]
        with pytest.raises(ValueError, match="forced tool_choice"):
            client.send_conversation(
                "sys", messages, tools=[WEATHER_TOOL], tool_choice="get_weather"
            )
        client.client.messages.create.assert_not_called()

    def test_forced_tool_choice_still_sent_for_opus_5(self) -> None:
        client = _build_claude_client("claude-opus-5")
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
from ai_api_unified.videos.ai_openai_videos import AIOpenAIVideos


class TestOpenAIEngines:
    def test_gpt_6_capabilities(self) -> None:
        capabilities = AICompletionsCapabilitiesOpenAI.for_model("gpt-6-sol")
        assert capabilities.context_window_length == 1_050_000
        assert capabilities.reasoning is True
        assert capabilities.knowledge_cutoff_date == date(2026, 4, 30)

    def test_gpt_5_6_does_not_inherit_the_gpt_5_cutoff(self) -> None:
        capabilities = AICompletionsCapabilitiesOpenAI.for_model("gpt-5.6-luna")
        assert capabilities.knowledge_cutoff_date == date(2026, 2, 16)

    def test_default_model(self) -> None:
        def _get_setting(str_key: str, default: Any = None) -> Any:
            # Only the API key is configured; every model setting is unset.
            return "test-key" if str_key == "OPENAI_API_KEY" else default

        with patch(
            "ai_api_unified.util.env_settings.EnvSettings.get_setting",
            side_effect=_get_setting,
        ):
            client = AiOpenAICompletions()
        assert client.completions_model == "gpt-5.6-luna"
        assert "gpt-6-astra" in client.list_model_names

    def test_images_default_and_catalogue(self) -> None:
        assert AIOpenAIImages.DEFAULT_IMAGE_MODEL == "gpt-image-2"
        assert "dall-e-3" not in AIOpenAIImages.SUPPORTED_IMAGE_MODELS

    def test_retired_image_model_rejected_at_construction(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with pytest.raises(AiProviderConfigurationError, match="retired"):
                AIOpenAIImages(model="dall-e-3")

    def test_sora_rejected_at_construction(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with pytest.raises(AiProviderConfigurationError, match="retired"):
                AIOpenAIVideos(model="sora-2")


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
    def test_nova_2_lite_catalogued(self) -> None:
        client = _build_bedrock_client("us.amazon.nova-2-lite-v1:0")
        assert "us.amazon.nova-2-lite-v1:0" in client.list_model_names
        assert client.max_context_tokens == 1_000_000
        assert client.capabilities.supports_tool_use is True

    def test_nova_v1_context_windows_match_model_cards(self) -> None:
        assert _build_bedrock_client("amazon.nova-lite-v1:0").max_context_tokens == (
            300_000
        )

    def test_forced_tool_choice_rejected_for_opus_5_5(self) -> None:
        client = _build_bedrock_client("us.anthropic.claude-opus-5-5")
        messages = [{"role": "user", "content": [{"text": "Weather?"}]}]
        with pytest.raises(ValueError, match="forced tool_choice"):
            client.send_conversation(
                "sys", messages, tools=[WEATHER_TOOL], tool_choice="get_weather"
            )
        client.client.converse.assert_not_called()


# ── Voyage embeddings ───────────────────────────────────────────────────────

from ai_api_unified.embeddings.ai_voyage_embeddings import (
    AIEmbeddingsCapabilitiesVoyage,
)


class TestVoyageCatalogue:
    def test_voyage_4_accepts_custom_dimensions(self) -> None:
        capabilities = AIEmbeddingsCapabilitiesVoyage.for_model("voyage-4")
        assert capabilities.default_dimensions == 1024
        assert capabilities.recommended_dimensions == [256, 512, 1024, 2048]


# ── Gemini images engine ────────────────────────────────────────────────────

pytest.importorskip("google.genai")

from ai_api_unified.images.ai_google_gemini_images import (
    AIGoogleGeminiImageProperties,
    AIGoogleGeminiImages,
)
from ai_api_unified.middleware.observability import NoOpObservabilityMiddleware


def _gemini_image_response(image_bytes: bytes) -> SimpleNamespace:
    part = SimpleNamespace(
        inline_data=SimpleNamespace(data=image_bytes, mime_type="image/jpeg")
    )
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))],
        usage_metadata=SimpleNamespace(prompt_token_count=7, total_token_count=1_300),
    )


def _build_gemini_images_client(
    generate_content: Mock, *, vertexai: bool = False
) -> AIGoogleGeminiImages:
    client: AIGoogleGeminiImages = AIGoogleGeminiImages.__new__(AIGoogleGeminiImages)
    client.image_model_name = "gemini-3.1-flash-image"
    client.client = SimpleNamespace(
        vertexai=vertexai,
        models=SimpleNamespace(generate_content=generate_content),
    )
    client._retry_with_exponential_backoff = lambda operation, **_: operation()
    client._observability_middleware = NoOpObservabilityMiddleware()
    return client


class TestGeminiImagesEngine:
    def test_default_is_a_native_gemini_image_model(self) -> None:
        assert AIGoogleGeminiImages.DEFAULT_IMAGE_MODEL == "gemini-3.1-flash-image"
        assert not any(
            str_model.startswith("imagen")
            for str_model in AIGoogleGeminiImages.SUPPORTED_IMAGE_MODELS
        )

    def test_generates_one_request_per_image(self) -> None:
        generate_content = Mock(
            side_effect=[
                _gemini_image_response(b"one"),
                _gemini_image_response(b"two"),
            ]
        )
        client = _build_gemini_images_client(generate_content)
        properties = AIGoogleGeminiImageProperties(
            width=1024, height=1024, num_images=2
        )

        list_images = client.generate_images("a lighthouse", properties)

        assert list_images == [b"one", b"two"]
        assert generate_content.call_count == 2
        kwargs = generate_content.call_args.kwargs
        assert kwargs["model"] == "gemini-3.1-flash-image"
        assert kwargs["config"].response_modalities == ["IMAGE"]
        assert kwargs["config"].image_config.aspect_ratio == "1:1"
        # The Developer API rejects person_generation, so it is omitted.
        assert kwargs["config"].image_config.person_generation is None

    def test_vertex_mode_sends_person_generation(self) -> None:
        generate_content = Mock(return_value=_gemini_image_response(b"img"))
        client = _build_gemini_images_client(generate_content, vertexai=True)
        properties = AIGoogleGeminiImageProperties(width=1024, height=1024)

        client.generate_images("a lighthouse", properties)

        image_config = generate_content.call_args.kwargs["config"].image_config
        assert image_config.person_generation is not None
        assert image_config.person_generation == "ALLOW_ADULT"

    def test_response_without_an_image_raises(self) -> None:
        empty = SimpleNamespace(candidates=[], usage_metadata=None)
        client = _build_gemini_images_client(Mock(return_value=empty))
        properties = AIGoogleGeminiImageProperties(width=1024, height=1024)
        with pytest.raises(ValueError, match="returned no images"):
            client.generate_images("a lighthouse", properties)
