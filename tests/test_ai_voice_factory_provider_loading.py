"""Tests for AIVoiceFactory provider loading orchestration."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from ai_api_unified.ai_provider_exceptions import (
    AiProviderConfigurationError,
    AiProviderDependencyUnavailableError,
)
from ai_api_unified.ai_provider_registry import (
    AI_PROVIDER_CAPABILITY_VOICE,
)
from ai_api_unified.voice.ai_voice_factory import AIVoiceFactory


class FakeVoiceClient:
    """Minimal fake voice client constructor target for voice factory tests."""

    def __init__(self, engine: str) -> None:
        self.engine: str = engine


class TestAiVoiceFactoryProviderLoading:
    """Validate AIVoiceFactory selection + lazy-loader integration behavior."""

    def test_create_loads_selected_ai_provider(self) -> None:
        """Voice factory should resolve spec and instantiate voice class for selected engine."""
        mock_ai_provider_spec: Mock = Mock()
        mock_env_settings: Mock = Mock()
        mock_env_settings.get_setting.return_value = "google"

        with patch(
            "ai_api_unified.voice.ai_voice_factory.EnvSettings",
            return_value=mock_env_settings,
        ):
            with patch(
                "ai_api_unified.voice.ai_voice_factory.get_ai_provider_spec",
                return_value=mock_ai_provider_spec,
            ) as mock_get_ai_provider_spec:
                with patch(
                    "ai_api_unified.voice.ai_voice_factory.load_ai_provider_class",
                    return_value=FakeVoiceClient,
                ) as mock_load_ai_provider_class:
                    voice_client: FakeVoiceClient = AIVoiceFactory.create()

        assert isinstance(voice_client, FakeVoiceClient)
        assert voice_client.engine == "google"
        mock_get_ai_provider_spec.assert_called_once_with(
            AI_PROVIDER_CAPABILITY_VOICE,
            "google",
        )
        mock_load_ai_provider_class.assert_called_once()

    def test_create_translates_unknown_engine_to_value_error(self) -> None:
        """Unsupported voice engines should preserve legacy ValueError contract."""
        mock_env_settings: Mock = Mock()
        mock_env_settings.get_setting.return_value = "unknown"

        with patch(
            "ai_api_unified.voice.ai_voice_factory.EnvSettings",
            return_value=mock_env_settings,
        ):
            with patch(
                "ai_api_unified.voice.ai_voice_factory.get_ai_provider_spec",
                side_effect=AiProviderConfigurationError("unknown engine"),
            ):
                with pytest.raises(
                    ValueError,
                    match="Unsupported AI_VOICE_ENGINE",
                ):
                    AIVoiceFactory.create()

    def test_create_raises_dependency_error_when_extra_missing(self) -> None:
        """Dependency unavailable errors should propagate unchanged from loader."""
        mock_ai_provider_spec: Mock = Mock()
        mock_env_settings: Mock = Mock()
        mock_env_settings.get_setting.return_value = "google"

        with patch(
            "ai_api_unified.voice.ai_voice_factory.EnvSettings",
            return_value=mock_env_settings,
        ):
            with patch(
                "ai_api_unified.voice.ai_voice_factory.get_ai_provider_spec",
                return_value=mock_ai_provider_spec,
            ):
                with patch(
                    "ai_api_unified.voice.ai_voice_factory.load_ai_provider_class",
                    side_effect=AiProviderDependencyUnavailableError("missing extra"),
                ):
                    with pytest.raises(
                        AiProviderDependencyUnavailableError,
                        match="missing extra",
                    ):
                        AIVoiceFactory.create()

    def test_create_raises_when_voice_engine_is_missing(self) -> None:
        """Voice factory should fail fast when no voice engine is configured."""
        mock_env_settings: Mock = Mock()
        mock_env_settings.get_setting.return_value = ""

        with patch(
            "ai_api_unified.voice.ai_voice_factory.EnvSettings",
            return_value=mock_env_settings,
        ):
            with pytest.raises(
                ValueError,
                match="AI_VOICE_ENGINE must be configured explicitly",
            ):
                AIVoiceFactory.create()
