from __future__ import annotations

import logging
import os
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

_LOGGER: logging.Logger = logging.getLogger(__name__)
# Selection keys already reported this process; one warning per key.
_SET_WARNED_DEPRECATED_SELECTIONS: set[str] = set()


def reset_deprecated_selection_warnings() -> None:
    """Forgets which deprecated selection keys were reported (tests)."""
    _SET_WARNED_DEPRECATED_SELECTIONS.clear()
    # Normal return after clearing the once-per-process record.
    return None


class EnvSettings(BaseSettings):
    """Small wrapper around environment configuration."""

    EMBEDDING_ENGINE: str | None = None
    COMPLETIONS_ENGINE: str | None = None
    OPENAI_API_KEY: str | None = None
    OPENAI_BASE_URL: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    # Admin API key for organization-level finops attribution (optional).
    ANTHROPIC_ADMIN_KEY: str | None = None
    EMBEDDING_MODEL_NAME: str | None = None
    COMPLETIONS_MODEL_NAME: str | None = None
    EMBEDDING_DIMENSIONS: int | None = None
    VECTOR_METRIC: str = "cosine"
    # No default here: an unset region falls through to models.json
    # (providers.aws.region), and callers default to us-east-1.
    AWS_REGION: str | None = None
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
                self._warn_deprecated_selection(setting, setting_value)
                return setting_value
        if setting in os.environ:
            # Early return: the process environment beats .env for undeclared
            # keys, as it does for declared fields.
            self._warn_deprecated_selection(setting, os.environ[setting])
            return os.environ[setting]
        # pydantic-settings stores undeclared .env keys lowercased, so a setting
        # read by its documented upper-case name must match the lowercased key.
        # Exact-case extras (init kwargs, override_setting) are attributes and
        # were already served by the hasattr branch above.
        str_dotenv_key: str = setting.lower()
        if self.model_extra and str_dotenv_key in self.model_extra:
            # Early return with the .env value.
            self._warn_deprecated_selection(setting, self.model_extra[str_dotenv_key])
            return self.model_extra[str_dotenv_key]
        str_from_file: str | None = self._models_config_setting(setting)
        if str_from_file is not None:
            # Early return with the models-file value.
            return str_from_file
        # Normal return with the default.
        return default

    @staticmethod
    def _warn_deprecated_selection(setting: str, value: Any) -> None:
        """
        Warns once per process when a model selection is still set in the
        environment or .env. Selections live in models.json; the key keeps
        overriding the file so a deployment does not break, and the warning
        names the field and the migrator that moves it.

        Args:
            setting: The settings key that was read.
            value: The value found; blank means unset and draws no warning.

        Returns:
            None after logging, or when the key is not a selection or was
            already reported.
        """
        from ..ai_models_config import DICT_SETTING_FIELD_NAMES

        str_field: str | None = DICT_SETTING_FIELD_NAMES.get(setting)
        if str_field is None or not str(value).strip():
            # Early return: not a selection key, or set to blank.
            return None
        if setting in _SET_WARNED_DEPRECATED_SELECTIONS:
            # Early return: reported once already.
            return None
        _SET_WARNED_DEPRECATED_SELECTIONS.add(setting)
        _LOGGER.warning(
            "%s is set in the environment or .env; settings other than secrets "
            "live in models.json (%s). The value still overrides the file. Move "
            "it with `poetry run python scripts/migrate2json.py --comment-env`.",
            setting,
            str_field,
            extra={
                "ai_config_event": "deprecated_selection_key",
                "setting": setting,
                "models_config_field": str_field,
            },
        )
        # Normal return after the one-time warning.
        return None

    def _models_config_setting(self, setting: str) -> str | None:
        """
        The value models.json supplies for a selection key, consulted after
        the process environment and .env.

        Args:
            setting: A settings key.

        Returns:
            The file's value, or None when the file does not set it, there
            is no file, or the key is not file-backed.
        """
        # Local import: the config module imports the router config, whose
        # imports reach back to this module.
        from ..ai_models_config import FROZENSET_FILE_BACKED_SETTINGS

        if setting not in FROZENSET_FILE_BACKED_SETTINGS:
            # Early return: not a key the file can supply.
            return None
        config = self.get_models_config()
        if config is None:
            # Early return: no file.
            return None
        # Normal return with the file's value for this key.
        return config.setting(setting)

    def get_models_config(self) -> Any:
        """
        The loaded models config (AIModelsConfig), or None when there is none.

        AI_MODELS_CONFIG from the process environment or .env names the file
        (or holds inline JSON); unset means ./models.json when it exists;
        blank means no file.
        """
        from ..ai_models_config import AI_MODELS_CONFIG_KEY, resolve_models_config

        str_dotenv_key: str = AI_MODELS_CONFIG_KEY.lower()
        bool_in_dotenv: bool = bool(
            self.model_extra and str_dotenv_key in self.model_extra
        )
        # An exact-case extra (init kwarg or override_setting) is an attribute,
        # as get_setting treats it.
        object_attribute: Any = getattr(self, AI_MODELS_CONFIG_KEY, None)
        bool_configured: bool = (
            object_attribute is not None
            or AI_MODELS_CONFIG_KEY in os.environ
            or bool_in_dotenv
        )
        str_configured: str | None
        if object_attribute is not None:
            str_configured = str(object_attribute)
        elif AI_MODELS_CONFIG_KEY in os.environ:
            str_configured = os.environ[AI_MODELS_CONFIG_KEY]
        else:
            str_configured = (
                str(self.model_extra[str_dotenv_key]) if bool_in_dotenv else None
            )
        # Normal return with the resolved config, or None.
        return resolve_models_config(str_configured, bool_configured=bool_configured)

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
