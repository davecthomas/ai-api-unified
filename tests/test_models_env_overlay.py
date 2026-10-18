# tests/test_models_env_overlay.py
"""
Per-environment overrides: `environments` in models.json holds partial
configs, and AI_MODELS_ENV picks one to deep-merge over the file.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from ai_api_unified import AIFactory
from ai_api_unified.ai_models_config import (
    clear_models_config_cache,
    load_models_config,
    merge_config_overlay,
)
from ai_api_unified.util.env_settings import EnvSettings

BASE: dict[str, Any] = {
    "completions": {
        "engine": "claude",
        "model": "claude-haiku-4-5",
        "fallbacks": ["openai:gpt-5.4-mini"],
    },
    "embeddings": {"engine": "openai", "model": "text-embedding-3-small"},
    "environments": {
        "prod": {
            "completions": {"model": "claude-sonnet-4-5", "fallbacks": []},
            "service": "billing",
        },
        "dev": {"embeddings": {"model": "text-embedding-3-large"}},
    },
}


@pytest.fixture
def models_file(tmp_path: Path) -> Iterator[Any]:
    def _write(dict_config: dict[str, Any]) -> Path:
        path = tmp_path / "models.json"
        path.write_text(json.dumps(dict_config))
        clear_models_config_cache()
        return path

    yield _write
    clear_models_config_cache()


class TestMerge:
    def test_objects_merge_and_lists_replace(self) -> None:
        merged = merge_config_overlay(
            {"a": {"x": 1, "y": [1, 2]}, "b": 1}, {"a": {"y": [3]}, "c": 2}
        )
        assert merged == {"a": {"x": 1, "y": [3]}, "b": 1, "c": 2}


class TestLoad:
    def test_environment_overrides_and_keeps_the_rest(self, models_file: Any) -> None:
        config = load_models_config(str(models_file(BASE)), "prod")
        assert config.completions.model == "claude-sonnet-4-5"
        assert config.completions.engine == "claude"
        assert config.completions.fallbacks == []
        assert config.embeddings.model == "text-embedding-3-small"
        assert config.service == "billing" and config.environment == "prod"

    def test_no_environment_is_the_base(self, models_file: Any) -> None:
        config = load_models_config(str(models_file(BASE)))
        assert config.completions.model == "claude-haiku-4-5"
        assert config.environment is None

    def test_unknown_environment_lists_the_defined_ones(self, models_file: Any) -> None:
        with pytest.raises(
            ValueError, match=r"AI_MODELS_ENV='stage'.*\['dev', 'prod'\]"
        ):
            load_models_config(str(models_file(BASE)), "stage")

    def test_environment_without_environments_section(self, models_file: Any) -> None:
        with pytest.raises(ValueError, match="defined: none"):
            load_models_config(str(models_file({"service": "x"})), "prod")

    @pytest.mark.parametrize(
        "bad",
        [
            {"environments": {"prod": "nope"}},
            {"environments": {"prod": {"environments": {}}}},
            {"environment": "prod"},
            {"environments": ["prod"]},
        ],
    )
    def test_malformed_environments_fail(self, models_file: Any, bad: dict) -> None:
        with pytest.raises(ValueError):
            load_models_config(str(models_file(bad)))

    def test_an_invalid_override_fails_validation(self, models_file: Any) -> None:
        bad = {"environments": {"prod": {"completions": {"thinking": "lots"}}}}
        with pytest.raises(ValueError, match="invalid"):
            load_models_config(str(models_file(bad)), "prod")


class TestSelection:
    def test_ai_models_env_picks_the_overlay(self, models_file: Any) -> None:
        path = models_file(BASE)
        with patch.dict(
            os.environ, {"AI_MODELS_CONFIG": str(path), "AI_MODELS_ENV": "dev"}
        ):
            config = EnvSettings().get_models_config()
        assert config.embeddings.model == "text-embedding-3-large"
        assert config.environment == "dev"

    def test_switching_environment_is_not_served_from_cache(
        self, models_file: Any
    ) -> None:
        path = models_file(BASE)
        with patch.dict(
            os.environ, {"AI_MODELS_CONFIG": str(path), "AI_MODELS_ENV": "prod"}
        ):
            prod = EnvSettings().get_models_config()
        with patch.dict(
            os.environ, {"AI_MODELS_CONFIG": str(path), "AI_MODELS_ENV": ""}
        ):
            base = EnvSettings().get_models_config()
        assert prod.completions.model != base.completions.model

    def test_clients_carry_the_environment_tag(self, models_file: Any) -> None:
        path = models_file(BASE)
        with patch.dict(
            os.environ,
            {
                "AI_MODELS_CONFIG": str(path),
                "AI_MODELS_ENV": "prod",
                "ANTHROPIC_API_KEY": "test-key",
            },
        ):
            prod = AIFactory.get_ai_completions_client()
        with patch.dict(
            os.environ,
            {
                "AI_MODELS_CONFIG": str(path),
                "AI_MODELS_ENV": "",
                "ANTHROPIC_API_KEY": "test-key",
                "OPENAI_API_KEY": "test-key",
            },
        ):
            base = AIFactory.get_ai_completions_client()
        assert prod.model_name == "claude-sonnet-4-5"
        assert prod._dict_config_tags["config_env"] == "prod"
        assert prod._dict_config_tags["service"] == "billing"
        assert "config_env" not in base._dict_config_tags
        assert prod.config_id != base.config_id
