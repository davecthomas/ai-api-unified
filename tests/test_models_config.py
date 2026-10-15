# tests/test_models_config.py
"""
models.json: parsing (JSON, YAML, inline), the allow/deny policy, the
settings it supplies through EnvSettings and their precedence, the sample
file, and policy enforcement in the factory.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_api_unified import AIFactory, AiFallbackCompletions, AiRouterCompletions
from ai_api_unified.ai_models_config import (
    AIModelPolicy,
    AIModelsConfig,
    clear_models_config_cache,
    load_models_config,
)
from ai_api_unified.util.env_settings import EnvSettings

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
LOGGER_FACTORY: str = "ai_api_unified.ai_factory"


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    clear_models_config_cache()
    yield
    clear_models_config_cache()


class TestPolicy:
    def test_deny_wins_allow_lists_and_bare_engines(self) -> None:
        policy = AIModelPolicy(
            allow=["claude:*", "openai:*"], deny=["openai:gpt-4*", "bedrock"]
        )
        assert policy.is_allowed("claude", "claude-opus-5-5")
        assert policy.is_allowed("openai", "gpt-5.6-luna")
        assert (
            policy.denial_reason("openai", "gpt-4.1-mini")
            == "denied by policy pattern 'openai:gpt-4*'"
        )
        assert (
            policy.denial_reason("bedrock", "amazon.nova-lite-v1:0")
            == "denied by policy pattern 'bedrock:*'"
        )
        assert (
            policy.denial_reason("google-gemini", "gemini-2.5-flash")
            == "not on the allow list"
        )

    def test_empty_allow_allows_everything(self) -> None:
        assert AIModelPolicy().is_allowed("anything", "at-all")
        assert AIModelPolicy(deny=["x:*"]).is_allowed("y", "z")

    def test_engine_default_model_matches_engine_wide_patterns_only(self) -> None:
        policy = AIModelPolicy(deny=["claude:claude-opus-*"])
        assert policy.is_allowed("claude", "")
        assert not AIModelPolicy(deny=["claude"]).is_allowed("claude", "")


class TestLoad:
    def test_json_yaml_and_inline(self, tmp_path: Path) -> None:
        dict_payload = {
            "completions": {"engine": "claude", "model": "m"},
            "deny": ["openai"],
        }
        (tmp_path / "m.json").write_text(json.dumps(dict_payload))
        (tmp_path / "m.yaml").write_text(
            "completions:\n  engine: claude\n  model: m\ndeny: [openai]\n"
        )
        for value in (
            str(tmp_path / "m.json"),
            str(tmp_path / "m.yaml"),
            json.dumps(dict_payload),
        ):
            config = load_models_config(value)
            assert config.completions.engine == "claude" and config.deny == ["openai:*"]

    def test_errors(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="blank"):
            load_models_config(" ")
        with pytest.raises(ValueError, match="not found"):
            load_models_config(str(tmp_path / "nope.json"))
        with pytest.raises(ValueError, match="valid JSON"):
            load_models_config("{nope")
        with pytest.raises(ValueError, match="invalid"):
            load_models_config('{"unknown_section": {}}')
        with pytest.raises(ValueError, match="object"):
            load_models_config("[]")

    def test_sample_file_validates_and_sets_every_section(self) -> None:
        config = load_models_config(str(REPO_ROOT / "models_sample.json"))
        assert config.router is not None and [t.name for t in config.router.tiers] == [
            "easy",
            "medium",
            "hard",
        ]
        assert config.policy.denial_reason("openai", "gpt-4o-mini")
        assert (
            config.setting("COMPLETIONS_FALLBACKS")
            == "claude:claude-sonnet-5,openai:gpt-5.6-luna"
        )
        assert config.setting("EMBEDDING_MODEL_NAME") == "gemini-embedding-001"
        assert config.setting("AI_VOICE_ENGINE") == "google"
        assert (
            config.setting("COMPLETIONS_ROUTER_CLASSIFIER")
            == "openai-decisions:gpt-6-luna"
        )
        assert config.setting("AI_MODEL_CHECK") == "prune"

    def test_fixture_file_validates_and_denies_one_of_its_own_tier_models(self) -> None:
        config = load_models_config(
            str(REPO_ROOT / "tests" / "fixtures" / "models_test.json")
        )
        assert config.comment and config.router is not None
        list_denied = [
            c.label
            for tier in config.router.tiers
            for c in tier.candidates
            if config.policy.denial_reason(c.engine, c.model)
        ]
        assert list_denied == ["claude:claude-fable-5-1"]
        assert config.policy.is_allowed("claude", "claude-opus-5-5")

    def test_setting_mapping_and_unset_keys(self) -> None:
        config = AIModelsConfig.model_validate(
            {
                "completions": {"engine": "claude"},
                "model_check": {"timeout_seconds": 3.5},
            }
        )
        assert config.setting("COMPLETIONS_ENGINE") == "claude"
        assert config.setting("COMPLETIONS_MODEL_NAME") is None
        assert config.setting("COMPLETIONS_FALLBACKS") is None
        assert config.setting("COMPLETIONS_ROUTER_CONFIG") is None
        assert config.setting("AI_MODEL_CHECK_TIMEOUT_SECONDS") == "3.5"
        assert config.setting("OPENAI_API_KEY") is None


class TestPrecedence:
    @pytest.fixture
    def workdir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        (tmp_path / "models.json").write_text(
            json.dumps(
                {
                    "completions": {
                        "engine": "claude",
                        "model": "from-file",
                        "fallbacks": ["openai:gpt-5.6-luna"],
                    },
                    "router": {
                        "tiers": [
                            {"name": "easy", "models": ["claude:claude-haiku-4-5"]}
                        ]
                    },
                }
            )
        )
        monkeypatch.chdir(tmp_path)
        for key in (
            "AI_MODELS_CONFIG",
            "COMPLETIONS_MODEL_NAME",
            "COMPLETIONS_ENGINE",
            "COMPLETIONS_FALLBACKS",
            "COMPLETIONS_ROUTER_CONFIG",
        ):
            monkeypatch.delenv(key, raising=False)
        return tmp_path

    def test_default_file_is_used_when_unset(self, workdir: Path) -> None:
        env = EnvSettings()
        assert env.get_setting("COMPLETIONS_MODEL_NAME", "") == "from-file"
        assert env.get_setting("COMPLETIONS_ENGINE", "") == "claude"
        assert env.get_setting("COMPLETIONS_FALLBACKS", "") == "openai:gpt-5.6-luna"
        assert (
            json.loads(env.get_setting("COMPLETIONS_ROUTER_CONFIG", ""))["tiers"][0][
                "name"
            ]
            == "easy"
        )

    def test_process_environment_and_dotenv_beat_the_file(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("COMPLETIONS_MODEL_NAME", "from-env")
        assert EnvSettings().get_setting("COMPLETIONS_MODEL_NAME") == "from-env"
        (workdir / ".env").write_text("COMPLETIONS_FALLBACKS=claude:claude-sonnet-5\n")
        assert (
            EnvSettings().get_setting("COMPLETIONS_FALLBACKS")
            == "claude:claude-sonnet-5"
        )

    def test_blank_setting_turns_the_file_off_and_path_points_elsewhere(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AI_MODELS_CONFIG", "")
        assert EnvSettings().get_setting("COMPLETIONS_MODEL_NAME", "<none>") == "<none>"
        (workdir / "other.json").write_text(
            json.dumps({"completions": {"model": "from-other"}})
        )
        monkeypatch.setenv("AI_MODELS_CONFIG", str(workdir / "other.json"))
        assert EnvSettings().get_setting("COMPLETIONS_MODEL_NAME") == "from-other"
        monkeypatch.setenv(
            "AI_MODELS_CONFIG", json.dumps({"completions": {"model": "inline"}})
        )
        assert EnvSettings().get_setting("COMPLETIONS_MODEL_NAME") == "inline"

    def test_file_edits_are_picked_up_by_mtime(self, workdir: Path) -> None:
        assert EnvSettings().get_setting("COMPLETIONS_MODEL_NAME") == "from-file"
        path = workdir / "models.json"
        path.write_text(json.dumps({"completions": {"model": "edited"}}))
        os.utime(path, (path.stat().st_atime + 5, path.stat().st_mtime + 5))
        assert EnvSettings().get_setting("COMPLETIONS_MODEL_NAME") == "edited"


BASE: dict[str, str] = {
    "ANTHROPIC_API_KEY": "test-key",
    "COMPLETIONS_ENGINE": "claude",
    "COMPLETIONS_MODEL_NAME": "claude-haiku-4-5",
}


@contextmanager
def _settings(models: dict | None = None, **values: str) -> Iterator[None]:
    dict_all = {**BASE, **values}
    config = AIModelsConfig.model_validate(models) if models is not None else None
    if models is not None:
        dict_all["AI_MODELS_CONFIG"] = json.dumps(models)

    def _get(key: str, default=None):
        # Mirrors the real precedence: explicit values, then the file.
        if key in dict_all:
            return dict_all[key]
        str_file = config.setting(key) if config is not None else None
        return str_file if str_file is not None else default

    with (
        patch(
            "ai_api_unified.util.env_settings.EnvSettings.get_setting",
            side_effect=_get,
        ),
        patch.dict(os.environ, dict_all, clear=False),
    ):
        yield


class TestFactoryPolicy:
    def test_denied_primary_raises(self) -> None:
        pytest.importorskip("anthropic")
        with (
            _settings({"deny": ["claude:claude-haiku-*"]}),
            pytest.raises(
                ValueError, match="completions model claude:claude-haiku-4-5 is denied"
            ),
        ):
            AIFactory.get_ai_completions_client()

    def test_explicit_denied_model_argument_raises(self) -> None:
        pytest.importorskip("anthropic")
        with (
            _settings({"allow": ["claude:claude-haiku-4-5"]}),
            pytest.raises(ValueError, match="not on the allow list"),
        ):
            AIFactory.get_ai_completions_client(model_name="claude-opus-5-5")

    def test_denied_fallback_is_pruned_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        pytest.importorskip("anthropic")
        with (
            _settings(
                {"deny": ["claude:claude-opus-*"]},
                COMPLETIONS_FALLBACKS="claude:claude-opus-5-5,claude:claude-sonnet-5",
            ),
            caplog.at_level(logging.WARNING, logger=LOGGER_FACTORY),
        ):
            client = AIFactory.get_ai_completions_client()
        assert isinstance(client, AiFallbackCompletions)
        assert [c.label for c in client.candidates] == [
            "claude:claude-haiku-4-5",
            "claude:claude-sonnet-5",
        ]
        denied = [
            r
            for r in caplog.records
            if getattr(r, "ai_model_policy_event", "") == "model_denied"
        ]
        assert (
            "claude:claude-opus-5-5 is denied by policy pattern"
            in denied[0].getMessage()
        )

    def test_denied_tier_models_prune_and_remap_upward(self) -> None:
        pytest.importorskip("anthropic")
        router = {
            "tiers": [
                {"name": "easy", "models": ["claude:claude-haiku-4-5"]},
                {"name": "medium", "models": ["claude:claude-sonnet-5"]},
                {"name": "hard", "models": ["claude:claude-opus-5-5"]},
            ],
            "default_tier": "medium",
        }
        with _settings(
            {
                "deny": ["claude:claude-sonnet-*"],
                "router": router,
                "classifier": {"engine": "none"},
            }
        ):
            client = AIFactory.get_ai_completions_client()
        assert isinstance(client, AiRouterCompletions)
        assert [t.name for t in client.tiers] == ["easy", "hard"]
        assert (
            client.config.pruned_tiers == {"medium": "hard"}
            and client.config.default_tier == "hard"
        )

    def test_denied_classifier_degrades_the_router(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        pytest.importorskip("anthropic")
        router = {"tiers": [{"name": "easy", "models": ["claude:claude-haiku-4-5"]}]}
        with (
            _settings(
                {
                    "deny": ["claude:claude-opus-*"],
                    "router": router,
                    "classifier": {"engine": "claude", "model": "claude-opus-5-5"},
                }
            ),
            caplog.at_level(logging.WARNING),
        ):
            client = AIFactory.get_ai_completions_client()
        assert isinstance(client, AiRouterCompletions) and client.classifier is None
        unavailable = [
            r
            for r in caplog.records
            if getattr(r, "ai_router_event", "") == "classifier_unavailable"
        ]
        assert "denied by policy" in unavailable[0].reason

    def test_denied_embeddings_and_image_models_raise(self) -> None:
        pytest.importorskip("openai")
        with _settings(
            {"deny": ["openai"]}, OPENAI_API_KEY="test-key", IMAGE_ENGINE="openai"
        ):
            with pytest.raises(
                ValueError,
                match="embeddings model openai:text-embedding-3-small is denied",
            ):
                AIFactory.get_ai_embedding_client(
                    embedding_engine="openai", model_name="text-embedding-3-small"
                )
            with pytest.raises(ValueError, match="image model openai:"):
                AIFactory.get_ai_images_client(image_model="gpt-image-1")

    def test_open_policy_changes_nothing(self) -> None:
        pytest.importorskip("anthropic")
        with _settings(
            {"allow": [], "deny": []}, COMPLETIONS_FALLBACKS="claude:claude-sonnet-5"
        ):
            client = AIFactory.get_ai_completions_client()
        assert isinstance(client, AiFallbackCompletions) and len(client.candidates) == 2

    def test_file_backed_router_builds_without_env_keys(self) -> None:
        pytest.importorskip("anthropic")
        models = {
            "completions": {"engine": "claude", "model": "claude-haiku-4-5"},
            "router": {
                "tiers": [{"name": "easy", "models": ["claude:claude-haiku-4-5"]}]
            },
            "classifier": {"engine": "none"},
        }
        dict_env = {
            "ANTHROPIC_API_KEY": "test-key",
            "AI_MODELS_CONFIG": json.dumps(models),
            "COMPLETIONS_ROUTER_CONFIG": "",
            "COMPLETIONS_ROUTER_CLASSIFIER": "",
            "COMPLETIONS_ENGINE": "",
            "COMPLETIONS_MODEL_NAME": "",
        }
        with patch.dict(os.environ, dict_env, clear=False):
            # Blank env keys fall through to the file for file-backed settings.
            for key in (
                "COMPLETIONS_ROUTER_CONFIG",
                "COMPLETIONS_ROUTER_CLASSIFIER",
                "COMPLETIONS_ENGINE",
                "COMPLETIONS_MODEL_NAME",
            ):
                os.environ.pop(key)
            client = AIFactory.get_ai_completions_client()
        assert (
            isinstance(client, AiRouterCompletions)
            and client.default_client.model_name == "claude-haiku-4-5"
        )
