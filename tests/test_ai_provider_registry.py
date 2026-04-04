"""Tests for AiProvider registry lookup and spec validation."""

from __future__ import annotations

import pytest

from ai_api_unified.ai_provider_exceptions import (
    AiProviderConfigurationError,
)
from ai_api_unified.ai_provider_registry import (
    AiProviderSpec,
    AI_PROVIDER_CAPABILITY_COMPLETIONS,
    AI_PROVIDER_CAPABILITY_IMAGES,
    AI_PROVIDER_CAPABILITY_VIDEOS,
    get_ai_provider_spec,
)


class TestAiProviderRegistry:
    """Validate AiProvider registry lookup and data normalization behavior."""

    def test_get_ai_provider_spec_normalizes_engine_value(self) -> None:
        """Registry lookup should normalize mixed-case engine strings before matching."""
        ai_provider_spec: AiProviderSpec = get_ai_provider_spec(
            AI_PROVIDER_CAPABILITY_COMPLETIONS,
            " OpenAI ",
        )

        assert ai_provider_spec.str_engine == "openai"
        assert ai_provider_spec.str_required_extra == "openai"

    def test_get_ai_provider_spec_raises_for_unknown_engine(self) -> None:
        """Registry lookup should raise configuration error when engine is unregistered."""
        with pytest.raises(
            AiProviderConfigurationError,
            match="Unsupported completions engine",
        ):
            get_ai_provider_spec(
                AI_PROVIDER_CAPABILITY_COMPLETIONS,
                "unknown-engine",
            )

    def test_ai_provider_spec_normalizes_engine_during_model_validation(self) -> None:
        """AiProviderSpec should trim and lowercase engine values at model validation time."""
        ai_provider_spec: AiProviderSpec = AiProviderSpec(
            str_capability="completions",
            str_engine="  GOOGLE-GEMINI  ",
            str_module_path=("ai_api_unified.completions.ai_google_gemini_completions"),
            str_class_name="GoogleGeminiCompletions",
            str_required_extra="google_gemini",
            str_consumer_install_command=("poetry add 'ai-api-unified[google_gemini]'"),
            str_local_install_command='poetry install --extras "google_gemini"',
            set_str_dependency_roots={"google"},
        )

        assert ai_provider_spec.str_engine == "google-gemini"

    def test_google_gemini_images_provider_is_registered(self) -> None:
        """Gemini images should participate in the centralized lazy-loading registry."""
        ai_provider_spec: AiProviderSpec = get_ai_provider_spec(
            AI_PROVIDER_CAPABILITY_IMAGES,
            "google-gemini",
        )

        assert ai_provider_spec.str_module_path == (
            "ai_api_unified.images.ai_google_gemini_images"
        )
        assert ai_provider_spec.str_class_name == "AIGoogleGeminiImages"
        assert ai_provider_spec.str_required_extra == "google_gemini"

    def test_google_gemini_video_provider_is_registered(self) -> None:
        """Gemini videos should participate in the centralized lazy-loading registry."""
        ai_provider_spec: AiProviderSpec = get_ai_provider_spec(
            AI_PROVIDER_CAPABILITY_VIDEOS,
            "google-gemini",
        )

        assert ai_provider_spec.str_module_path == (
            "ai_api_unified.videos.ai_google_gemini_videos"
        )
        assert ai_provider_spec.str_class_name == "AIGoogleGeminiVideos"
        assert ai_provider_spec.str_required_extra == "google_gemini"
