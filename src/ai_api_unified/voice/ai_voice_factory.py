"""
Factory for creating voice provider clients through centralized lazy loading.
"""

from __future__ import annotations

import logging
from typing import Any

from ai_api_unified.ai_provider_exceptions import (
    AiProviderCapabilityUnsupportedError,
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
# Only the OpenAI voice provider shares AIOpenAIBase, so it is the one voice
# engine whose constructor accepts these arguments.
FROZENSET_AI_OPENAI_BASE_VOICE_ENGINES: frozenset[str] = frozenset({"openai"})
FROZENSET_AI_OPENAI_BASE_VOICE_KWARGS: frozenset[str] = frozenset(
    {"base_url", "retry_policy"}
)


class AIVoiceFactory:
    """
    Factory to create AI voice clients based on environment configuration.
    """

    @staticmethod
    def create(voice_engine: str | None = None, **kwargs: Any) -> AIVoiceBase:
        """
        Creates a voice provider client based on the configured voice engine.

        Args:
            voice_engine: Optional engine override; falls back to AI_VOICE_ENGINE config.
            **kwargs: Provider-specific constructor arguments such as base_url
                and retry_policy, forwarded to the resolved voice provider.

        Returns:
            Concrete AIVoiceBase implementation for the requested voice engine.
            Raises ValueError for unsupported engines and RuntimeError-derived
            provider exceptions for dependency/runtime loading failures.
        """
        str_engine: str
        if voice_engine is not None and voice_engine.strip():
            str_engine = voice_engine.strip().lower()
        else:
            env_settings: EnvSettings = EnvSettings()
            object_engine_value: object = env_settings.get_setting(
                AI_VOICE_ENGINE_ENV_KEY, ""
            )
            str_engine = (
                str(object_engine_value).strip().lower()
                if object_engine_value is not None
                else ""
            )
        if not str_engine:
            raise ValueError(
                "AI_VOICE_ENGINE must be configured explicitly; there is no default provider."
            )

        # Scope this catch to the registry lookup alone. Provider construction
        # raises the same exception type for unrelated reasons -- a rejected
        # plaintext base URL among them -- and reporting those as an unsupported
        # engine sends the caller after the wrong problem.
        try:
            ai_provider_spec: AiProviderSpec = get_ai_provider_spec(
                AI_PROVIDER_CAPABILITY_VOICE, str_engine
            )
        except AiProviderConfigurationError as exception:
            _LOGGER.error("Unsupported AI_VOICE_ENGINE: %s", str_engine)
            raise ValueError(
                f"Unsupported AI_VOICE_ENGINE: {str_engine}"
            ) from exception

        AIVoiceFactory._require_provider_kwargs_support(
            str_engine=str_engine,
            dict_provider_kwargs=kwargs,
        )

        try:
            class_ai_voice_provider: type[AIVoiceBase] = load_ai_provider_class(
                ai_provider_spec,
                AIVoiceBase,
            )
            voice_provider_client: AIVoiceBase = class_ai_voice_provider(
                engine=str_engine, **kwargs
            )
            # Normal return with resolved voice provider client.
            return voice_provider_client
        except AiProviderDependencyUnavailableError as exception:
            _LOGGER.warning(str(exception))
            raise
        except AiProviderRuntimeError:
            raise

    @staticmethod
    def _require_provider_kwargs_support(
        *,
        str_engine: str,
        dict_provider_kwargs: dict[str, Any],
    ) -> None:
        """
        Rejects provider kwargs for voice engines whose constructor cannot honor them.

        This lives here rather than in AIFactory because this is the factory that
        accepts the kwargs: AIVoiceBase ignores unknown model fields, so an
        unguarded override would be dropped in silence.

        Args:
            str_engine: Resolved voice engine selector token.
            dict_provider_kwargs: Provider constructor arguments supplied by the caller.

        Raises:
            AiProviderCapabilityUnsupportedError: When the engine cannot honor a
                supplied argument.
        """
        for str_kwarg_name in sorted(dict_provider_kwargs):
            frozenset_supported: frozenset[str] = (
                FROZENSET_AI_OPENAI_BASE_VOICE_ENGINES
                if str_kwarg_name in FROZENSET_AI_OPENAI_BASE_VOICE_KWARGS
                else frozenset()
            )
            if str_engine in frozenset_supported:
                continue
            raise AiProviderCapabilityUnsupportedError(
                f"Voice engine {str_engine!r} does not support {str_kwarg_name!r}. "
                + (
                    f"Supported engines: {', '.join(sorted(frozenset_supported))}."
                    if frozenset_supported
                    else "No voice engine accepts that argument."
                )
            )
