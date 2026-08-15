"""Tests for AIFactory.get_ai_voice_client delegation to AIVoiceFactory."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from ai_api_unified.ai_factory import AIFactory
from ai_api_unified.ai_provider_exceptions import (
    AiProviderCapabilityUnsupportedError,
    AiProviderConfigurationError,
)
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

    def test_base_url_and_retry_policy_reach_the_provider(self) -> None:
        """Provider kwargs should be forwarded like the sibling constructors do."""
        with patch.object(
            AIVoiceFactory, "create", return_value=FakeVoiceClient(engine="openai")
        ) as mock_create:
            AIFactory.get_ai_voice_client(
                voice_engine="openai",
                base_url="https://gateway.invalid/v1",
                retry_policy="none",
            )

        mock_create.assert_called_once_with(
            "openai",
            base_url="https://gateway.invalid/v1",
            retry_policy="none",
        )

    @pytest.mark.parametrize(
        "str_unsupported_engine", ["google", "azure", "elevenlabs"]
    )
    @pytest.mark.parametrize(
        "str_kwarg_name, object_kwarg_value",
        [("base_url", "https://gateway.invalid/v1"), ("retry_policy", "none")],
    )
    def test_provider_kwargs_rejected_for_engines_that_cannot_honor_them(
        self,
        str_unsupported_engine: str,
        str_kwarg_name: str,
        object_kwarg_value: object,
    ) -> None:
        """Engines without an AIOpenAIBase constructor must reject these arguments."""
        with pytest.raises(AiProviderCapabilityUnsupportedError) as exception_info:
            AIFactory.get_ai_voice_client(
                voice_engine=str_unsupported_engine,
                **{str_kwarg_name: object_kwarg_value},
            )

        # The message must name the argument the caller actually passed.
        assert str_kwarg_name in str(exception_info.value)

    @pytest.mark.parametrize(
        "str_unsupported_engine", ["google", "azure", "elevenlabs"]
    )
    def test_voice_factory_create_rejects_unsupported_provider_kwargs(
        self, str_unsupported_engine: str
    ) -> None:
        """The guard belongs to the factory that accepts the kwargs, not only AIFactory.

        AIVoiceBase ignores unknown model fields, so an unguarded override here
        would be dropped in silence.
        """
        with pytest.raises(AiProviderCapabilityUnsupportedError):
            AIVoiceFactory.create(
                str_unsupported_engine, base_url="https://gateway.invalid/v1"
            )

    def test_base_url_rejection_reports_its_own_reason(self) -> None:
        """A rejected base URL must not surface as an unsupported-engine error.

        Provider construction raises AiProviderConfigurationError for reasons
        unrelated to engine resolution, and the plaintext-credential rejection
        is one of them.
        """
        with pytest.raises(AiProviderConfigurationError) as exception_info:
            AIVoiceFactory.create("openai", base_url="http://insecure.invalid/v1")

        assert "https://" in str(exception_info.value)
        assert "Unsupported AI_VOICE_ENGINE" not in str(exception_info.value)

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
