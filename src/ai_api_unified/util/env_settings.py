from __future__ import annotations

import logging
import os
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

_LOGGER: logging.Logger = logging.getLogger(__name__)


class EnvSettings(BaseSettings):
    """Small wrapper around environment configuration."""

    EMBEDDING_ENGINE: str = "openai"
    COMPLETIONS_ENGINE: str = "openai"
    OPENAI_API_KEY: str | None = None
    OPENAI_BASE_URL: str | None = None
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
    COMPLETIONS_MODEL_NAME: str = "gpt-4o-mini"
    EMBEDDING_DIMENSIONS: int = 1536
    VECTOR_METRIC: str = "cosine"
    AWS_REGION: str = "us-east-1"
    AI_API_GEO_RESIDENCY: str | None = None
    _SUPPORTED_DATA_RESIDENCY_GEOS: tuple[str, ...] = ("us", "usa", "united states")
    IMAGE_MODEL_NAME: str | None = "gpt-image-1"
    IMAGE_ENGINE: str = "openai"
    GOOGLE_GEMINI_API_KEY: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="allow", frozen=False)

    def get_setting(self, setting: str, default: Any = None) -> Any:
        """Retrieve a setting value with an optional default."""
        if hasattr(self, setting):
            return getattr(self, setting)
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
