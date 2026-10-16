# tests/test_ai_factory_classifier.py
"""
AIFactory.get_ai_classifier_client: engine resolution from arguments and
settings, the loud error when nothing is configured, registry coverage, and
an offline build of a real structured-output classifier.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest

from ai_api_unified import AIBaseClassifier, AIFactory
from ai_api_unified.ai_provider_registry import (
    AI_PROVIDER_CAPABILITY_CLASSIFICATION,
    DICT_TUPLE_AI_PROVIDER_REGISTRY,
    get_ai_provider_spec,
)
from ai_api_unified.classification.ai_structured_output_classifier import (
    AiStructuredOutputClassifier,
)

BASE_SETTINGS: dict[str, str] = {
    "ANTHROPIC_API_KEY": "test-key",
}


@contextmanager
def _settings(dict_settings: dict[str, str]) -> Iterator[None]:
    dict_all: dict[str, str] = {**BASE_SETTINGS, **dict_settings}
    with (
        patch(
            "ai_api_unified.util.env_settings.EnvSettings.get_setting",
            side_effect=lambda key, default=None: dict_all.get(key, default),
        ),
        patch.dict(os.environ, dict_all, clear=False),
    ):
        yield


class TestRegistry:
    @pytest.mark.parametrize(
        "token",
        [
            "openai",
            "openai-responses",
            "openai-compatible",
            "claude",
            "google-gemini",
            "bedrock",
            "anthropic",
        ],
    )
    def test_structured_output_engines_are_registered(self, token: str) -> None:
        spec = get_ai_provider_spec(AI_PROVIDER_CAPABILITY_CLASSIFICATION, token)
        assert spec.str_class_name == "AiStructuredOutputClassifier"
        completions = get_ai_provider_spec("completions", token)
        assert spec.str_required_extra == completions.str_required_extra

    @pytest.mark.parametrize("token", ["nova", "llama", "mistral", "cohere", "ai21"])
    def test_vendor_tokens_without_structured_output_are_absent(
        self, token: str
    ) -> None:
        assert (
            AI_PROVIDER_CAPABILITY_CLASSIFICATION,
            token,
        ) not in DICT_TUPLE_AI_PROVIDER_REGISTRY


class TestResolution:
    def test_unset_engine_without_openai_key_fails_loudly(self) -> None:
        with _settings({}), pytest.raises(ValueError, match="classifier.engine"):
            AIFactory.get_ai_classifier_client()

    def test_unset_engine_with_openai_key_uses_decisions(self) -> None:
        pytest.importorskip("openai")
        with _settings({"OPENAI_API_KEY": "test-key"}):
            client = AIFactory.get_ai_classifier_client()
        assert type(client).__name__ == "AiOpenAIDecisionsClassifier"
        assert client.model_name == "gpt-6-luna"

    def test_configured_engine_wins_over_the_openai_key(self) -> None:
        pytest.importorskip("anthropic")
        with _settings({"OPENAI_API_KEY": "test-key", "CLASSIFIER_ENGINE": "claude"}):
            client = AIFactory.get_ai_classifier_client()
        assert isinstance(client, AiStructuredOutputClassifier)

    def test_decisions_is_registered(self) -> None:
        spec = get_ai_provider_spec(
            AI_PROVIDER_CAPABILITY_CLASSIFICATION, "openai-decisions"
        )
        assert spec.str_class_name == "AiOpenAIDecisionsClassifier"

    def test_unknown_engine_fails_loudly(self) -> None:
        with (
            _settings({"CLASSIFIER_ENGINE": "nova"}),
            pytest.raises(ValueError, match="nova"),
        ):
            AIFactory.get_ai_classifier_client()

    def test_engine_and_model_from_settings(self) -> None:
        pytest.importorskip("anthropic")
        with _settings(
            {"CLASSIFIER_ENGINE": "claude", "CLASSIFIER_MODEL_NAME": "claude-haiku-4-5"}
        ):
            client = AIFactory.get_ai_classifier_client()
        assert isinstance(client, AiStructuredOutputClassifier)
        assert isinstance(client, AIBaseClassifier)
        assert client.model_name == "claude-haiku-4-5"
        assert client.str_engine == "claude"

    def test_arguments_override_settings(self) -> None:
        pytest.importorskip("anthropic")
        with _settings({"CLASSIFIER_ENGINE": "nova"}):
            client = AIFactory.get_ai_classifier_client(
                classifier_engine="claude", model_name="claude-sonnet-5"
            )
        assert client.model_name == "claude-sonnet-5"

    def test_blank_model_uses_the_engine_default(self) -> None:
        pytest.importorskip("anthropic")
        with _settings({"CLASSIFIER_ENGINE": "claude", "CLASSIFIER_MODEL_NAME": ""}):
            client = AIFactory.get_ai_classifier_client()
        assert client.model_name

    def test_classifier_capabilities_follow_the_engine(self) -> None:
        pytest.importorskip("anthropic")
        with _settings({"CLASSIFIER_ENGINE": "claude"}):
            caps: Any = AIFactory.get_ai_classifier_client().capabilities
        assert caps.supports_choice and caps.supports_predicate and caps.supports_score
