# tests/test_env_settings_extra_keys.py
"""
Undeclared settings read from .env (COMPLETIONS_FALLBACKS, the router and
classifier keys, ...) must resolve by their documented upper-case names.
pydantic-settings stores them lowercased in model_extra.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_api_unified.util.env_settings import EnvSettings


@pytest.fixture
def dotenv_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / ".env").write_text(
        "COMPLETIONS_ROUTER_CONFIG=./router.json\n"
        "COMPLETIONS_FALLBACKS=openai:gpt-4o-mini\n"
        "CLASSIFIER_ENGINE=claude\n"
    )
    monkeypatch.chdir(tmp_path)
    for str_key in (
        "COMPLETIONS_ROUTER_CONFIG",
        "COMPLETIONS_FALLBACKS",
        "CLASSIFIER_ENGINE",
    ):
        monkeypatch.delenv(str_key, raising=False)
    return tmp_path


def test_undeclared_dotenv_keys_resolve_by_upper_case_name(dotenv_dir: Path) -> None:
    env_settings = EnvSettings()
    assert env_settings.get_setting("COMPLETIONS_ROUTER_CONFIG", "") == "./router.json"
    assert env_settings.get_setting("COMPLETIONS_FALLBACKS", "") == "openai:gpt-4o-mini"
    assert env_settings.get_setting("CLASSIFIER_ENGINE", "") == "claude"


def test_process_environment_still_wins_over_default(
    dotenv_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COMPLETIONS_ROUTER_CLASSIFIER", "none")
    assert EnvSettings().get_setting("COMPLETIONS_ROUTER_CLASSIFIER", "") == "none"
    assert EnvSettings().get_setting("NOT_A_SETTING", "fallback") == "fallback"
