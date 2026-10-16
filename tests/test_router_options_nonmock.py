# tests/test_router_options_nonmock.py
"""
Live router options: the real Decisions classifier with min_confidence,
per-tier thinking levels reaching Claude, and a route_floor.

    poetry run pytest -m nonmock tests/test_router_options_nonmock.py -q -s
"""

from __future__ import annotations

import logging
import os
import socket

import pytest

from ai_api_unified import (
    AIFactory,
    AIRouterConfig,
    AIStructuredPrompt,
    AIThinkingLevel,
    AiRouterCompletions,
)

pytestmark = pytest.mark.nonmock

LOGGER_ROUTER: str = "ai_api_unified.completions.ai_router_completions"
FLOAT_MIN_CONFIDENCE: float = 0.97


class _Word(AIStructuredPrompt):
    word: str

    @staticmethod
    def get_prompt(input_text: str = "") -> str:
        return input_text


@pytest.fixture
def router() -> AiRouterCompletions:
    for str_key, str_host in (
        ("OPENAI_API_KEY", "api.openai.com"),
        ("ANTHROPIC_API_KEY", "api.anthropic.com"),
    ):
        if not os.environ.get(str_key):
            pytest.skip(f"Skipping: {str_key} not set")
        try:
            socket.gethostbyname(str_host)
        except OSError as exception:
            pytest.skip(f"Skipping: DNS unavailable for {str_host}: {exception}")
    config = AIRouterConfig.model_validate(
        {
            "tiers": [
                {
                    "name": "easy",
                    "models": ["claude:claude-haiku-4-5"],
                    "thinking": "off",
                },
                {
                    "name": "medium",
                    "models": ["claude:claude-haiku-4-5"],
                    "thinking": "low",
                },
                {
                    "name": "hard",
                    "models": ["claude:claude-haiku-4-5"],
                    "thinking": "high",
                },
            ],
            "classifier": {
                "engine": "openai-decisions",
                "model": "gpt-6-luna",
                "min_confidence": FLOAT_MIN_CONFIDENCE,
            },
        }
    )
    client = AIFactory.get_ai_completions_client(
        completions_engine="claude", model_name="claude-haiku-4-5", router=config
    )
    assert isinstance(client, AiRouterCompletions)
    return client


def test_low_confidence_moves_up_and_tier_thinking_applies(
    router: AiRouterCompletions, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER_ROUTER):
        result = router.send_structured_output(
            "Is a tomato a fruit or a vegetable? Answer with one word.",
            response_model=_Word,
        )
    routed = [
        r for r in caplog.records if getattr(r, "ai_router_event", "") == "routed"
    ][-1]
    print(
        f"\n  routed to {routed.tier} by {routed.method} "
        f"(confidence {routed.confidence}); thinking {result.thinking}"
    )
    if routed.confidence is not None and routed.confidence < FLOAT_MIN_CONFIDENCE:
        assert routed.method == "low_confidence" or routed.tier == "hard"
    else:
        assert routed.method == "classifier"
    dict_levels = {
        "easy": AIThinkingLevel.OFF,
        "medium": AIThinkingLevel.LOW,
        "hard": AIThinkingLevel.HIGH,
    }
    assert result.thinking is not None
    assert result.thinking.requested is dict_levels[routed.tier]


def test_route_floor_lifts_an_easy_request(router: AiRouterCompletions) -> None:
    result = router.send_structured_output(
        "Reply with the word pong.",
        response_model=_Word,
        provider_options={"route_floor": "hard"},
    )
    print(f"\n  floored to {router.last_tier}; thinking {result.thinking}")
    assert router.last_tier == "hard"
    assert result.thinking is not None
    assert result.thinking.requested is AIThinkingLevel.HIGH
