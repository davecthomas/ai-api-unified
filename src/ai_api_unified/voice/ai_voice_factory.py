"""
Factory for creating voice provider clients through centralized lazy loading.
"""

from __future__ import annotations

import logging

from ai_api_unified.ai_provider_exceptions import (
    AiProviderConfigurationError,
    AiProviderDependencyUnavailableError,
    AiProviderRuntimeError,
)
from ai_api_unified.ai_provider_loader import load_ai_provider_class
from ai_api_unified.ai_provider_registry import (
    AiProviderSpec,
    AI_PROVIDER_CAPABILITY_VOICE,
    get_ai_provider_spec,
)
from ai_api_unified.util.env_settings import EnvSettings
from ai_api_unified.voice.ai_voice_base import AIVoiceBase

_LOGGER: logging.Logger = logging.getLogger(__name__)

AI_VOICE_ENGINE_ENV_KEY: str = "AI_VOICE_ENGINE"


class AIVoiceFactory:
    """
    Factory to create AI voice clients based on environment configuration.
    """

    @staticmethod
    def create() -> AIVoiceBase:
        """
        Creates a voice provider client based on the configured voice engine.

        Args:
            None

        Returns:
            Concrete AIVoiceBase implementation for the configured voice engine.
            Raises ValueError for unsupported engines and RuntimeError-derived
            provider exceptions for dependency/runtime loading failures.
        """
        env_settings: EnvSettings = EnvSettings()
        object_engine_value: object = env_settings.get_setting(AI_VOICE_ENGINE_ENV_KEY, "")
        str_engine: str = (
            str(object_engine_value).strip().lower()
            if object_engine_value is not None
            else ""
        )
        if not str_engine:
            raise ValueError(
                "AI_VOICE_ENGINE must be configured explicitly; there is no default provider."
            )

        try:
            ai_provider_spec: AiProviderSpec = get_ai_provider_spec(
                AI_PROVIDER_CAPABILITY_VOICE, str_engine
            )
            class_ai_voice_provider: type[AIVoiceBase] = load_ai_provider_class(
                ai_provider_spec,
                AIVoiceBase,
            )
            voice_provider_client: AIVoiceBase = class_ai_voice_provider(
                engine=str_engine
            )
            # Normal return with resolved voice provider client.
            return voice_provider_client
        except AiProviderConfigurationError as exception:
            _LOGGER.error("Unsupported AI_VOICE_ENGINE: %s", str_engine)
            raise ValueError(
                f"Unsupported AI_VOICE_ENGINE: {str_engine}"
            ) from exception
        except AiProviderDependencyUnavailableError as exception:
            _LOGGER.warning(str(exception))
            raise
        except AiProviderRuntimeError:
            raise
