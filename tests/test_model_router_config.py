# tests/test_model_router_config.py
"""
AIRouterConfig parsing: JSON from a path or inline, the `models` spelling,
built-in tier descriptions, validation, and classification_text_for.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_api_unified import AIRouterConfig, AIRouterTier, load_router_config
from ai_api_unified.ai_provider_exceptions import AiFallbackReason as R
from ai_api_unified.routing import (
    DICT_DEFAULT_TIER_DESCRIPTIONS,
    classification_text_for,
)

DICT_CONFIG: dict = {
    "tiers": [
        {
            "name": "easy",
            "models": ["openai:gpt-4.1-mini", "google-gemini:gemini-2.5-flash"],
        },
        {"name": "medium", "models": ["claude:claude-sonnet-5"]},
        {
            "name": "hard",
            "description": "Deep reasoning.",
            "models": [
                {
                    "engine": "claude",
                    "model": "claude-opus-5-5",
                    "base_url": "https://x.test",
                }
            ],
        },
    ],
    "classifier": {
        "engine": "openai-decisions",
        "model": "gpt-6-luna",
        "timeout_seconds": 3,
    },
    "default_tier": "Medium",
    "on_exhausted": "raise",
    "fallback_on": "unavailable, timeout",
}


class TestLoad:
    def test_inline_json(self) -> None:
        config = load_router_config(json.dumps(DICT_CONFIG))
        assert [tier.name for tier in config.tiers] == ["easy", "medium", "hard"]
        assert config.tiers[0].candidates[1].label == "google-gemini:gemini-2.5-flash"
        assert config.tiers[2].candidates[0].base_url == "https://x.test"
        assert config.classifier is not None and config.classifier.timeout_seconds == 3
        assert config.default_tier == "medium"
        assert config.on_exhausted == "raise"
        assert config.fallback_on == {R.UNAVAILABLE, R.TIMEOUT}

    def test_file_path(self, tmp_path: Path) -> None:
        path = tmp_path / "router.json"
        path.write_text(json.dumps(DICT_CONFIG))
        assert load_router_config(str(path)).tier_index("hard") == 2

    def test_missing_file(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            load_router_config("/nonexistent/router.json")

    def test_invalid_json_and_blank(self) -> None:
        with pytest.raises(ValueError, match="valid JSON"):
            load_router_config("{not json")
        with pytest.raises(ValueError, match="blank"):
            load_router_config("  ")
        with pytest.raises(ValueError, match="object"):
            load_router_config("[]")

    def test_bedrock_ids_keep_their_colons(self) -> None:
        config = load_router_config(
            '{"tiers": [{"name": "easy", "models": ["nova:amazon.nova-lite-v1:0"]}]}'
        )
        assert config.tiers[0].candidates[0].model == "amazon.nova-lite-v1:0"


class TestValidation:
    def test_built_in_descriptions(self) -> None:
        tier = AIRouterTier(name="easy", models=["openai:gpt-4.1-mini"])  # type: ignore[call-arg]
        assert tier.description == DICT_DEFAULT_TIER_DESCRIPTIONS["easy"]

    def test_custom_tier_needs_a_description(self) -> None:
        with pytest.raises(ValueError, match="needs a description"):
            AIRouterTier(name="expert", models=["openai:gpt-4.1-mini"])  # type: ignore[call-arg]

    @pytest.mark.parametrize("name", ["Easy Tier", "1st", "-x", ""])
    def test_tier_name_pattern(self, name: str) -> None:
        with pytest.raises(ValueError):
            AIRouterTier(name=name, description="d", models=["openai:gpt-4.1-mini"])  # type: ignore[call-arg]

    def test_name_is_normalized(self) -> None:
        tier = AIRouterTier(name=" HARD ", description="d", models=["openai:x"])  # type: ignore[call-arg]
        assert tier.name == "hard"

    def test_empty_models_rejected(self) -> None:
        with pytest.raises(ValueError):
            AIRouterTier(name="easy", models=[])  # type: ignore[call-arg]

    def test_duplicate_tier_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            AIRouterConfig(tiers=[AIRouterTier(name="easy", models=["openai:a"]), AIRouterTier(name="easy", models=["openai:b"])])  # type: ignore[call-arg]

    def test_unknown_default_tier_rejected(self) -> None:
        with pytest.raises(ValueError, match="default_tier"):
            AIRouterConfig(tiers=[AIRouterTier(name="easy", models=["openai:a"])], default_tier="hard")  # type: ignore[call-arg]

    def test_no_tiers_rejected(self) -> None:
        with pytest.raises(ValueError):
            AIRouterConfig(tiers=[])

    def test_classifier_none_is_disabled(self) -> None:
        config = load_router_config(
            '{"tiers": [{"name": "easy", "models": ["openai:a"]}], "classifier": {"engine": "NONE"}}'
        )
        assert config.classifier is not None and config.classifier.disabled


class TestClassificationText:
    def test_latest_user_message_wins(self) -> None:
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "latest"},
        ]
        assert classification_text_for("prompt", messages) == "latest"

    def test_text_blocks_are_joined(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "text", "text": "b"},
                ],
            }
        ]
        assert classification_text_for(None, messages) == "a\nb"

    def test_prompt_when_no_user_message(self) -> None:
        assert (
            classification_text_for("prompt", [{"role": "assistant", "content": "x"}])
            == "prompt"
        )
        assert classification_text_for("prompt", None) == "prompt"
        assert classification_text_for(None, None) == ""
