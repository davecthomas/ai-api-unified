from __future__ import annotations

from typing import Any
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSettings(BaseSettings):
    """Small wrapper around environment configuration."""

    EMBEDDING_ENGINE: str = "openai"
    COMPLETIONS_ENGINE: str = "openai"
    OPENAI_API_KEY: str | None = None
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
    COMPLETIONS_MODEL_NAME: str = "gpt-4o-mini"
    EMBEDDING_DIMENSIONS: int = 1536
    VECTOR_METRIC: str = "cosine"
    AWS_REGION: str = "us-east-1"

    model_config = SettingsConfigDict(env_file=".env", extra="allow", frozen=False)

    def get_setting(self, setting: str, default: Any = None) -> Any:
        """Retrieve a setting value with an optional default."""
        return getattr(self, setting, default)

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
