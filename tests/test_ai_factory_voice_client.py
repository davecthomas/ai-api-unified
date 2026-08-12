"""Tests for AIFactory.get_ai_voice_client delegation to AIVoiceFactory."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from ai_api_unified.ai_factory import AIFactory
from ai_api_unified.ai_provider_exceptions import AiProviderConfigurationError
from ai_api_unified.voice.ai_voice_factory import AIVoiceFactory


class FakeVoiceClient:
    """Minimal fake voice client constructor target for factory delegation tests."""

    def __init__(self, engine: str) -> None:
        self.engine: str = engine


class TestAiFactoryVoiceClient:
    """Validate voice is reachable from AIFactory alongside the other capabilities."""

    def test_voice_constructor_is_a_sibling_of_the_other_capabilities(self) -> None:
        """AIFactory should expose a voice constructor next to the other four."""
        for str_method_name in (
            "get_ai_completions_client",
            "get_ai_embedding_client",
            "get_ai_images_client",
            "get_ai_video_client",
            "get_ai_voice_client",
        ):
            assert callable(getattr(AIFactory, str_method_name))

    def test_explicit_engine_override_is_passed_through(self) -> None:
        """An explicit engine argument should reach AIVoiceFactory unchanged."""
        with patch.object(
            AIVoiceFactory, "create", return_value=FakeVoiceClient(engine="openai")
        ) as mock_create:
            voice_client: FakeVoiceClient = AIFactory.get_ai_voice_client(
                voice_engine="OpenAI"
            )

        assert voice_client.engine == "openai"
        mock_create.assert_called_once_with("openai")

    def test_engine_falls_back_to_environment_configuration(self) -> None:
        """With no override the engine should come from AI_VOICE_ENGINE."""
        mock_env_settings: Mock = Mock()
        mock_env_settings.get_setting.return_value = "elevenlabs"

        with patch(
            "ai_api_unified.ai_factory.EnvSettings", return_value=mock_env_settings
        ):
            with patch.object(
                AIVoiceFactory,
                "create",
                return_value=FakeVoiceClient(engine="elevenlabs"),
            ) as mock_create:
                AIFactory.get_ai_voice_client()

        mock_create.assert_called_once_with("elevenlabs")

    def test_unconfigured_engine_raises_value_error(self) -> None:
        """An unset AI_VOICE_ENGINE should raise the shared required-engine error."""
        mock_env_settings: Mock = Mock()
        mock_env_settings.get_setting.return_value = ""

        with patch(
            "ai_api_unified.ai_factory.EnvSettings", return_value=mock_env_settings
        ):
            with pytest.raises(
                ValueError,
                match="AI_VOICE_ENGINE must be configured explicitly",
            ):
                AIFactory.get_ai_voice_client()

    def test_unsupported_engine_error_matches_the_voice_factory(self) -> None:
        """Both entry points should report an unsupported engine identically."""
        with patch(
            "ai_api_unified.voice.ai_voice_factory.get_ai_provider_spec",
            side_effect=AiProviderConfigurationError("unknown engine"),
        ):
            with pytest.raises(ValueError) as factory_exception_info:
                AIFactory.get_ai_voice_client(voice_engine="nope")
            with pytest.raises(ValueError) as voice_factory_exception_info:
                AIVoiceFactory.create("nope")

        assert str(factory_exception_info.value) == str(
            voice_factory_exception_info.value
        )
        assert "Unsupported AI_VOICE_ENGINE: nope" == str(factory_exception_info.value)

    def test_voice_factory_create_remains_callable_without_arguments(self) -> None:
        """The pre-existing no-argument AIVoiceFactory.create contract must still hold."""
        mock_env_settings: Mock = Mock()
        mock_env_settings.get_setting.return_value = "google"

        with patch(
            "ai_api_unified.voice.ai_voice_factory.EnvSettings",
            return_value=mock_env_settings,
        ):
            with patch(
                "ai_api_unified.voice.ai_voice_factory.get_ai_provider_spec",
                return_value=Mock(),
            ):
                with patch(
                    "ai_api_unified.voice.ai_voice_factory.load_ai_provider_class",
                    return_value=FakeVoiceClient,
                ):
                    voice_client: FakeVoiceClient = AIVoiceFactory.create()

        assert voice_client.engine == "google"
