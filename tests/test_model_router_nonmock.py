# tests/test_model_router_nonmock.py
"""
Live model-router tests: a two-tier router on real providers, classified by
a real classifier, with the routed events checked.

Claude classifies by default here, since every environment with an
Anthropic key can run it; the Decisions-classified variant skips on a quota
error until the OpenAI org has credits.

Run with:

    poetry run pytest -m nonmock tests/test_model_router_nonmock.py -q
"""

from __future__ import annotations

import logging
import os
import socket

import pytest

from ai_api_unified import (
    AIClassificationChoice,
    AIFactory,
    AIRouterClassifierConfig,
    AIRouterConfig,
    AIRouterTier,
    AiFallbackReason,
    AiProviderRequestError,
    AiRouterCompletions,
)

pytestmark = pytest.mark.nonmock

LOGGER_NAME: str = "ai_api_unified.completions.ai_router_completions"
STR_EASY: str = "Reply with the single word: pong"
PROBE_CHOICES: list[AIClassificationChoice] = [
    AIClassificationChoice(value="easy", description="A short request."),
    AIClassificationChoice(value="hard", description="A demanding request."),
]
STR_HARD: str = (
    "Explain, step by step, how to prove that the halting problem is "
    "undecidable, then relate it to Rice's theorem in two paragraphs."
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


def _config(classifier: AIRouterClassifierConfig | None) -> AIRouterConfig:
    return AIRouterConfig(
        tiers=[
            AIRouterTier(name="easy", models=["claude:claude-haiku-4-5"]),  # type: ignore[call-arg]
            AIRouterTier(name="hard", models=["claude:claude-sonnet-5"]),  # type: ignore[call-arg]
        ],
        classifier=classifier,
        default_tier="easy",
    )


def _run_router(
    router: AiRouterCompletions, caplog: pytest.LogCaptureFixture
) -> list[logging.LogRecord]:
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        easy_text = router.send_prompt(STR_EASY, max_response_tokens=32)
        str_easy_tier = router.last_tier
        router.send_prompt(STR_HARD, max_response_tokens=64)
        str_hard_tier = router.last_tier
    assert easy_text and "pong" in easy_text.lower()
    assert str_easy_tier == "easy" and str_hard_tier == "hard"
    return [r for r in caplog.records if getattr(r, "ai_router_event", "") == "routed"]


def test_router_classified_by_claude_routes_easy_and_hard(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _skip_without("ANTHROPIC_API_KEY", "api.anthropic.com")
    monkeypatch.setenv("COMPLETIONS_ENGINE", "claude")
    monkeypatch.setenv("COMPLETIONS_MODEL_NAME", "claude-haiku-4-5")
    client = AIFactory.get_ai_completions_client(
        router=_config(
            AIRouterClassifierConfig(engine="claude", model="claude-haiku-4-5")
        )
    )
    assert isinstance(client, AiRouterCompletions)
    list_routed = _run_router(client, caplog)
    assert [r.method for r in list_routed] == ["classifier", "classifier"]
    assert [r.tier for r in list_routed] == ["easy", "hard"]
    assert list_routed[0].classifier_engine == "claude"
    assert list_routed[0].classifier_input_tokens
    assert list_routed[0].classifier_token_source == "provider"


def test_router_classified_by_decisions(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _skip_without("ANTHROPIC_API_KEY", "api.anthropic.com")
    _skip_without("OPENAI_API_KEY", "api.openai.com")
    monkeypatch.setenv("COMPLETIONS_ENGINE", "claude")
    monkeypatch.setenv("COMPLETIONS_MODEL_NAME", "claude-haiku-4-5")
    client = AIFactory.get_ai_completions_client(
        router=_config(AIRouterClassifierConfig(engine="openai-decisions"))
    )
    assert isinstance(client, AiRouterCompletions)
    if client.classifier is None:
        pytest.skip("Skipping: the Decisions classifier could not be built")
    # Probe once so a quota error skips instead of degrading silently.
    try:
        client.classifier.classify_choice(STR_EASY, PROBE_CHOICES)
    except AiProviderRequestError as exception:
        _skip_on_quota(exception)
    list_routed = _run_router(client, caplog)
    assert [r.tier for r in list_routed] == ["easy", "hard"]
    assert list_routed[0].classifier_engine == "openai-decisions"
    assert list_routed[0].classifier_usd_cost is not None
