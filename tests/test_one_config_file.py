# tests/test_one_config_file.py
"""
Everything but secrets in models.json: provider settings and the middleware
section are served from the file, a pre-models.json environment key still
wins and warns once, AWS_REGION never warns, the middleware YAML path still
works, and the migrator moves the new keys and the YAML in.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ai_api_unified import config_schema
from ai_api_unified.ai_base import resolve_base_url_override
from ai_api_unified.ai_models_config import clear_models_config_cache
from ai_api_unified.middleware.middleware_config import MiddlewareConfig
from ai_api_unified.util.env_settings import (
    EnvSettings,
    reset_deprecated_selection_warnings,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
LOGGER_ENV: str = "ai_api_unified.util.env_settings"

DICT_PROVIDERS: dict[str, Any] = {
    "openai": {"base_url": "https://gw.example.com/v1", "user": "svc-a"},
    "anthropic": {"base_url": "https://gw.example.com/anthropic", "user": "svc-b"},
    "google": {"auth_method": "service_account", "project_id": "p1", "location": "eu"},
    "aws": {"region": "eu-west-1"},
    "openai_compatible": {
        "base_url": "http://localhost:8000/v1",
        "structured_output": "json_object",
        "context_window": 32000,
    },
    "azure_speech": {"region": "westus"},
}
DICT_EXPECTED: dict[str, str] = {
    "OPENAI_BASE_URL_OVERRIDE": "https://gw.example.com/v1",
    "OPENAI_USER": "svc-a",
    "ANTHROPIC_BASE_URL_OVERRIDE": "https://gw.example.com/anthropic",
    "ANTHROPIC_USER": "svc-b",
    "GOOGLE_AUTH_METHOD": "service_account",
    "GOOGLE_PROJECT_ID": "p1",
    "GOOGLE_LOCATION": "eu",
    "AWS_REGION": "eu-west-1",
    "OPENAI_COMPATIBLE_BASE_URL": "http://localhost:8000/v1",
    "OPENAI_COMPATIBLE_STRUCTURED_OUTPUT": "json_object",
    "OPENAI_COMPATIBLE_CONTEXT_WINDOW": "32000",
    "MICROSOFT_COGNITIVE_SERVICES_REGION": "westus",
    "BEDROCK_VIDEO_OUTPUT_S3_URI": "s3://bucket/out/",
    "AI_STRICT_DEPRECATIONS": "True",
}


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A working directory with no .env and none of the moved keys set."""
    monkeypatch.chdir(tmp_path)
    for str_key in [*DICT_EXPECTED, "AI_MIDDLEWARE_CONFIG_PATH"]:
        monkeypatch.delenv(str_key, raising=False)
    clear_models_config_cache()
    reset_deprecated_selection_warnings()
    yield tmp_path
    clear_models_config_cache()
    reset_deprecated_selection_warnings()


def _use(monkeypatch: pytest.MonkeyPatch, dict_config: dict[str, Any]) -> None:
    monkeypatch.setenv("AI_MODELS_CONFIG", json.dumps(dict_config))
    clear_models_config_cache()


class TestProviderSettings:
    def test_every_moved_key_is_served_from_the_file(
        self, isolated: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _use(
            monkeypatch,
            {
                "providers": DICT_PROVIDERS,
                "videos": {"bedrock_output_s3_uri": "s3://bucket/out/"},
                "strict_deprecations": True,
            },
        )
        env = EnvSettings()
        assert {k: env.get_setting(k) for k in DICT_EXPECTED} == DICT_EXPECTED

    def test_env_key_still_wins_and_warns_once(
        self,
        isolated: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _use(monkeypatch, {"providers": DICT_PROVIDERS})
        monkeypatch.setenv("GOOGLE_PROJECT_ID", "from-env")
        with caplog.at_level(logging.WARNING, logger=LOGGER_ENV):
            assert EnvSettings().get_setting("GOOGLE_PROJECT_ID") == "from-env"
            assert EnvSettings().get_setting("GOOGLE_PROJECT_ID") == "from-env"
        records = [r for r in caplog.records if hasattr(r, "ai_config_event")]
        assert [r.models_config_field for r in records] == [
            "providers.google.project_id"
        ]

    def test_aws_region_in_the_environment_wins_without_a_warning(
        self,
        isolated: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _use(monkeypatch, {"providers": DICT_PROVIDERS})
        monkeypatch.setenv("AWS_REGION", "ap-south-1")
        with caplog.at_level(logging.WARNING, logger=LOGGER_ENV):
            assert EnvSettings().get_setting("AWS_REGION") == "ap-south-1"
        assert not [r for r in caplog.records if hasattr(r, "ai_config_event")]

    def test_unset_region_falls_back_to_the_callers_default(
        self, isolated: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _use(monkeypatch, {})
        assert EnvSettings().get_setting("AWS_REGION", "us-east-1") == "us-east-1"

    def test_a_base_url_from_the_file_is_still_validated(
        self, isolated: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _use(
            monkeypatch,
            {"providers": {"openai": {"base_url": "http://gw.example.com"}}},
        )
        from ai_api_unified.ai_provider_exceptions import AiProviderConfigurationError

        with pytest.raises(
            AiProviderConfigurationError, match=r"providers\.openai\.base_url.*https"
        ):
            resolve_base_url_override(
                EnvSettings(), str_env_key="OPENAI_BASE_URL_OVERRIDE", str_explicit=None
            )

    @pytest.mark.parametrize(
        "dict_bad",
        [
            {"providers": {"google": {"auth_method": "oauth"}}},
            {"providers": {"openai_compatible": {"context_window": 0}}},
            {"providers": {"cohere": {}}},
        ],
        ids=["bad-auth-method", "zero-context", "unknown-provider"],
    )
    def test_bad_provider_settings_fail_at_load(self, dict_bad: dict[str, Any]) -> None:
        from ai_api_unified.ai_models_config import AIModelsConfig

        with pytest.raises(ValueError):
            AIModelsConfig.model_validate(dict_bad)


class TestMiddlewareSection:
    LIST_MIDDLEWARE: list[dict[str, Any]] = [
        {
            "name": "pii_redaction",
            "enabled": True,
            "settings": {"direction": "input_only"},
        },
        {"name": "observability", "enabled": True, "settings": {"log_level": "INFO"}},
        {"name": "observability", "enabled": "not-a-bool-or-anything"},
    ]

    def test_models_json_middleware_loads_when_no_path_is_set(
        self, isolated: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _use(monkeypatch, {"middleware": self.LIST_MIDDLEWARE})
        config = MiddlewareConfig()
        assert config.get_pii_redaction_settings() is not None
        # The invalid third entry is skipped, as the YAML loader skips it.
        assert [e.name for e in config._configuration.list_middleware] == [
            "pii_redaction",
            "observability",
        ]

    def test_yaml_path_still_wins_and_warns(
        self,
        isolated: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _use(monkeypatch, {"middleware": self.LIST_MIDDLEWARE[:1]})
        path_yaml = isolated / "middleware.yaml"
        path_yaml.write_text(
            "middleware:\n  - name: observability\n    enabled: true\n"
        )
        monkeypatch.setenv("AI_MIDDLEWARE_CONFIG_PATH", str(path_yaml))
        with caplog.at_level(logging.WARNING, logger=LOGGER_ENV):
            config = MiddlewareConfig()
        assert [e.name for e in config._configuration.list_middleware] == [
            "observability"
        ]
        assert any(
            getattr(r, "setting", "") == "AI_MIDDLEWARE_CONFIG_PATH"
            for r in caplog.records
        )

    def test_no_middleware_anywhere_is_empty(
        self, isolated: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _use(monkeypatch, {})
        assert MiddlewareConfig()._configuration.list_middleware == []

    def test_validate_checks_the_section_entries(self, tmp_path: Path) -> None:
        path_good = tmp_path / "good.json"
        path_good.write_text(
            json.dumps({"middleware": self.LIST_MIDDLEWARE[:2], "completions": {}})
        )
        path_bad = tmp_path / "bad.json"
        path_bad.write_text(
            json.dumps(
                {"completions": {}, "middleware": [{"name": "spam", "enabled": True}]}
            )
        )
        assert config_schema.validate_config_file(str(path_good)) == []
        assert config_schema.validate_config_file(str(path_bad)) == [
            "unknown middleware 'spam'"
        ]


class TestMigrator:
    def _script(self) -> Any:
        spec = importlib.util.spec_from_file_location(
            "migrate2json", REPO_ROOT / "scripts" / "migrate2json.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_moves_provider_keys_middleware_and_strict_but_not_secrets_or_region(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "middleware.yaml").write_text(
            "middleware:\n  - name: pii_redaction\n    enabled: true\n"
        )
        (tmp_path / ".env").write_text(
            "OPENAI_API_KEY=secret\nAWS_REGION=eu-west-1\n"
            "OPENAI_BASE_URL_OVERRIDE=https://gw.example.com/v1\n"
            "GOOGLE_PROJECT_ID=p1\nOPENAI_COMPATIBLE_CONTEXT_WINDOW=32000\n"
            "BEDROCK_VIDEO_OUTPUT_S3_URI=s3://bucket/out/\n"
            "AI_MIDDLEWARE_CONFIG_PATH=./middleware.yaml\nAI_STRICT_DEPRECATIONS=true\n"
        )
        out = tmp_path / "models.json"
        assert (
            self._script().main(
                ["--env", str(tmp_path / ".env"), "--out", str(out), "--comment-env"]
            )
            == 0
        )
        dict_out = json.loads(out.read_text())
        assert dict_out["providers"] == {
            "openai": {"base_url": "https://gw.example.com/v1"},
            "google": {"project_id": "p1"},
            "openai_compatible": {"context_window": 32000},
        }
        assert dict_out["videos"] == {"bedrock_output_s3_uri": "s3://bucket/out/"}
        assert dict_out["middleware"] == [{"name": "pii_redaction", "enabled": True}]
        assert dict_out["strict_deprecations"] is True
        str_env = (tmp_path / ".env").read_text()
        assert "\nOPENAI_API_KEY=secret" in "\n" + str_env
        assert "\nAWS_REGION=eu-west-1" in "\n" + str_env
        assert "# moved to models.json: GOOGLE_PROJECT_ID=p1" in str_env
        assert config_schema.validate_config_file(str(out)) == []
