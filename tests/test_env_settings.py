from __future__ import annotations

from pathlib import Path

from ai_api_unified.util.env_settings import EnvSettings
from ai_api_unified.util.utils import is_hex_enabled


def test_get_setting_reads_model_extra_for_undeclared_values() -> None:
    """
    Undeclared settings captured in model_extra should resolve before os.environ.
    """
    settings: EnvSettings = EnvSettings.model_validate(
        {"UNDECLARED_EXTRA_SETTING": "from-model-extra"}
    )

    assert settings.model_extra is not None
    assert settings.model_extra["UNDECLARED_EXTRA_SETTING"] == "from-model-extra"
    assert settings.get_setting("UNDECLARED_EXTRA_SETTING") == "from-model-extra"


def test_get_setting_uses_default_when_declared_field_is_unset() -> None:
    """
    Declared optional settings should still honor explicit fallback defaults when unset.
    """
    settings: EnvSettings = EnvSettings()

    assert (
        settings.get_setting("COMPLETIONS_ENGINE", "google-gemini") == "google-gemini"
    )


def test_get_setting_reads_declared_values_from_env_file(tmp_path: Path) -> None:
    """Declared settings should load from .env files and be returned by get_setting."""
    env_file: Path = tmp_path / ".env"
    env_file.write_text("GOOGLE_AUTH_METHOD=api_key\n", encoding="utf-8")

    settings: EnvSettings = EnvSettings(_env_file=env_file)

    assert settings.get_setting("GOOGLE_AUTH_METHOD") == "api_key"


def test_is_hex_enabled_reads_env_file_value(monkeypatch, tmp_path: Path) -> None:
    """Hex detection should honor values loaded from a local .env file."""
    monkeypatch.delenv("IS_HEX_ENABLED", raising=False)
    env_file: Path = tmp_path / ".env"
    env_file.write_text("IS_HEX_ENABLED=true\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert is_hex_enabled() is True
