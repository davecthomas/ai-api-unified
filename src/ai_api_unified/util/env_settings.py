from __future__ import annotations

import logging
import os
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

_LOGGER: logging.Logger = logging.getLogger(__name__)


class EnvSettings(BaseSettings):
    """Small wrapper around environment configuration."""

    EMBEDDING_ENGINE: str | None = None
    COMPLETIONS_ENGINE: str | None = None
    OPENAI_API_KEY: str | None = None
    OPENAI_BASE_URL: str | None = None
    EMBEDDING_MODEL_NAME: str | None = None
    COMPLETIONS_MODEL_NAME: str | None = None
    EMBEDDING_DIMENSIONS: int | None = None
    VECTOR_METRIC: str = "cosine"
    AWS_REGION: str = "us-east-1"
    AI_API_GEO_RESIDENCY: str | None = None
    _SUPPORTED_DATA_RESIDENCY_GEOS: tuple[str, ...] = ("us", "usa", "united states")
    IMAGE_MODEL_NAME: str | None = None
    IMAGE_ENGINE: str | None = None
    VIDEO_MODEL_NAME: str | None = None
    VIDEO_ENGINE: str | None = None
    VIDEO_OUTPUT_DIR: str | None = None
    VIDEO_POLL_INTERVAL_SECONDS: int | None = None
    VIDEO_TIMEOUT_SECONDS: int | None = None
    BEDROCK_VIDEO_OUTPUT_S3_URI: str | None = None
    AI_VOICE_ENGINE: str | None = None
    GOOGLE_GEMINI_API_KEY: str | None = None
    GOOGLE_AUTH_METHOD: str | None = None
    GOOGLE_PROJECT_ID: str | None = None
    GOOGLE_LOCATION: str | None = None
    DEFAULT_GEMINI_TTS_MODEL: str | None = None
    AI_MIDDLEWARE_CONFIG_PATH: str | None = None
    MICROSOFT_COGNITIVE_SERVICES_API_KEY: str | None = None
    MICROSOFT_COGNITIVE_SERVICES_REGION: str | None = None
    ELEVEN_LABS_API_KEY: str | None = None
    IS_HEX_ENABLED: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="allow", frozen=False)

    def get_setting(self, setting: str, default: Any = None) -> Any:
        """Retrieve a setting value with an optional default."""
        if hasattr(self, setting):
            setting_value: Any = getattr(self, setting)
            if setting_value is not None:
                return setting_value
        if self.model_extra and setting in self.model_extra:
            return self.model_extra[setting]
        return os.environ.get(setting, default)

    # Alias
    def get(self, setting: str, default: Any = None) -> Any:
        return self.get_setting(setting, default)

    def is_setting_on(self, setting: str) -> bool:
        return bool(self.get_setting(setting))

    def is_configured(self, setting: str) -> bool:
        return self.get_setting(setting) is not None

    def override_setting(self, setting: str, value: Any) -> None:
        setattr(self, setting, value)
        import os

        os.environ[setting] = str(value)

    def get_geo_residency(self) -> str | None:
        """Return normalized geo residency constraint if configured."""

        # If value is blank or None, return None
        value: str | None = self.get_setting("AI_API_GEO_RESIDENCY")
        if value is None or not value.strip():
            return None

        normalized_value: str = value.strip().lower()
        if normalized_value in self._SUPPORTED_DATA_RESIDENCY_GEOS:
            return "US"

        _LOGGER.warning(
            "Unsupported AI_API_GEO_RESIDENCY value '%s'. Falling back to default provider settings.",
            value,
        )
        return None
