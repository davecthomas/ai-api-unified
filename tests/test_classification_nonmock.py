# tests/test_classification_nonmock.py
"""
Live classification tests against real providers.

Structured-output classifiers run on Claude and Gemini when their keys are
set. The Decisions tests need an OpenAI org with credits; a quota error
skips them with a message rather than failing, since the shape of a live
Decisions response (notably whether it carries `usage`) is what they exist
to record.

Run with:

    poetry run pytest -m nonmock tests/test_classification_nonmock.py -q
"""

from __future__ import annotations

import logging
import os
import socket

import pytest

from ai_api_unified import (
    AIClassificationChoice,
    AIClassificationLevel,
    AIFactory,
    AiFallbackReason,
    AiProviderRequestError,
)
from ai_api_unified.middleware.middleware_config import ObservabilitySettingsModel
from ai_api_unified.middleware.observability import (
    COST_LOGGER_NAME,
    LoggerBackedObservabilityMiddleware,
)

pytestmark = pytest.mark.nonmock

CHOICES: list[AIClassificationChoice] = [
    AIClassificationChoice(
        value="easy", description="A short factual or formulaic request."
    ),
    AIClassificationChoice(
        value="hard", description="Multi-step reasoning or expert judgment."
    ),
]
LEVELS: list[AIClassificationLevel] = [
    AIClassificationLevel(label="low", description="Trivial."),
    AIClassificationLevel(label="mid", description="Some thought."),
    AIClassificationLevel(label="high", description="Deep reasoning."),
]
STR_EASY: str = "What is the capital of France?"
STR_HARD: str = (
    "Design a sharded, multi-region inventory system that never double-sells "
    "an item under network partitions, and prove the invariant."
)


def _skip_without(str_key: str, str_host: str) -> None:
    if str_key not in os.environ:
        pytest.skip(f"Skipping: {str_key} not set")
    try:
        socket.gethostbyname(str_host)
    except OSError as exception:
        pytest.skip(f"Skipping: DNS unavailable for {str_host}: {exception}")


def _skip_on_quota(exception: AiProviderRequestError) -> None:
    if exception.fallback_reason in (
        AiFallbackReason.QUOTA_EXHAUSTED,
        AiFallbackReason.AUTH_REJECTED,
    ):
        pytest.skip(f"Skipping: provider account cannot serve ({exception})")
    raise exception


@pytest.mark.parametrize(
    ("str_engine", "str_model", "str_key", "str_host"),
    [
        ("claude", "claude-haiku-4-5", "ANTHROPIC_API_KEY", "api.anthropic.com"),
        (
            "google-gemini",
            "gemini-2.5-flash",
            "GOOGLE_GEMINI_API_KEY",
            "generativelanguage.googleapis.com",
        ),
    ],
    ids=["claude", "gemini"],
)
def test_structured_output_classifier_separates_easy_from_hard(
    str_engine: str, str_model: str, str_key: str, str_host: str
) -> None:
    _skip_without(str_key, str_host)
    classifier = AIFactory.get_ai_classifier_client(
        classifier_engine=str_engine, model_name=str_model
    )
    try:
        easy = classifier.classify_choice(
            STR_EASY, CHOICES, instructions="Rate the difficulty."
        )
        hard = classifier.classify_choice(
            STR_HARD, CHOICES, instructions="Rate the difficulty."
        )
        predicate = classifier.classify_predicate(
            STR_EASY, instructions="Is this a question about geography?"
        )
        scored = classifier.classify_score(STR_HARD, LEVELS)
    except AiProviderRequestError as exception:
        _skip_on_quota(exception)
    assert easy.choice == "easy" and hard.choice == "hard"
    assert easy.refused is False
    assert easy.token_source == "provider" and easy.usage.input_tokens
    assert easy.provider_engine == str_engine and easy.model_name == str_model
    assert predicate.probability == 1.0
    assert scored.score is not None and scored.score >= 0.5


def test_decisions_classifier_choice_predicate_score_and_usage_shape() -> None:
    _skip_without("OPENAI_API_KEY", "api.openai.com")
    classifier = AIFactory.get_ai_classifier_client(
        classifier_engine="openai-decisions"
    )
    try:
        easy = classifier.classify_choice(
            STR_EASY, CHOICES, instructions="Rate the difficulty."
        )
        hard = classifier.classify_choice(
            STR_HARD, CHOICES, instructions="Rate the difficulty."
        )
        predicate = classifier.classify_predicate(
            STR_EASY, instructions="Is this a question about geography?"
        )
        scored = classifier.classify_score(STR_HARD, LEVELS)
    except AiProviderRequestError as exception:
        _skip_on_quota(exception)
    assert easy.choice == "easy" and hard.choice == "hard"
    assert easy.confidence is not None and easy.probabilities
    assert predicate.probability is not None and predicate.probability > 0.5
    assert scored.score is not None and scored.score >= 0.5
    # Records whether the live response carries usage: "provider" means it
    # does, "estimated" means the engine counted. Either is metered.
    print(
        f"\nDecisions token_source={easy.token_source} input_tokens={easy.usage.input_tokens}"
    )
    assert easy.token_source in ("provider", "estimated")
    assert easy.usage.input_tokens


def test_decisions_classification_emits_a_cost_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _skip_without("OPENAI_API_KEY", "api.openai.com")
    classifier = AIFactory.get_ai_classifier_client(
        classifier_engine="openai-decisions"
    )
    classifier._observability_middleware = LoggerBackedObservabilityMiddleware(
        ObservabilitySettingsModel(emit_cost=True)
    )
    with caplog.at_level(logging.DEBUG, logger=COST_LOGGER_NAME):
        try:
            classifier.classify_choice(STR_EASY, CHOICES)
        except AiProviderRequestError as exception:
            _skip_on_quota(exception)
    list_cost = [r for r in caplog.records if r.name == COST_LOGGER_NAME]
    assert list_cost, "no cost event"
    fields = list_cost[-1].args[1]
    assert fields["model"] == "decisions/gpt-6-luna"
    assert fields["operation"] == "classify_choice"
    assert fields["input_tokens"] and float(fields["usd_cost"]) > 0
