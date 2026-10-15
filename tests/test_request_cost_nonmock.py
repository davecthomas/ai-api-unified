# tests/test_request_cost_nonmock.py
"""
Live per-request cost: a two-tier router on real providers, classified by a
real classifier, reports the classifier call and the served attempt on one
AIRequestCostReport, and a plain engine call reports a one-call request.

Run with:

    poetry run pytest -m nonmock tests/test_request_cost_nonmock.py -q
"""

from __future__ import annotations

import os
import socket
from decimal import Decimal

import pytest

from ai_api_unified import (
    AIFactory,
    AIRouterClassifierConfig,
    AIRouterConfig,
    AIRouterTier,
    AiRouterCompletions,
    request_cost_scope,
)

pytestmark = pytest.mark.nonmock

STR_EASY: str = "Reply with the single word: pong"


def _skip_without(str_key: str, str_host: str) -> None:
    if str_key not in os.environ:
        pytest.skip(f"Skipping: {str_key} not set")
    try:
        socket.gethostbyname(str_host)
    except OSError as exception:
        pytest.skip(f"Skipping: DNS unavailable for {str_host}: {exception}")


def test_plain_engine_call_reports_a_one_call_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _skip_without("ANTHROPIC_API_KEY", "api.anthropic.com")
    monkeypatch.setenv("COMPLETIONS_ENGINE", "claude")
    monkeypatch.setenv("COMPLETIONS_MODEL_NAME", "claude-haiku-4-5")
    client = AIFactory.get_ai_completions_client()
    text = client.send_prompt(STR_EASY, max_response_tokens=32)
    assert text and "pong" in text.lower()
    report = client.last_request_cost
    assert report is not None and report.status == "ok"
    assert len(report.calls) == 1 and report.calls[0].served
    assert report.billed_calls == 1 and report.failed_calls == 0
    assert report.total_usd > Decimal("0")
    assert report.served_by == "claude:claude-haiku-4-5"


def test_routed_request_reports_classifier_and_served_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _skip_without("ANTHROPIC_API_KEY", "api.anthropic.com")
    monkeypatch.setenv("COMPLETIONS_ENGINE", "claude")
    monkeypatch.setenv("COMPLETIONS_MODEL_NAME", "claude-haiku-4-5")
    client = AIFactory.get_ai_completions_client(
        router=AIRouterConfig(
            tiers=[
                AIRouterTier(name="easy", models=["claude:claude-haiku-4-5"]),  # type: ignore[call-arg]
                AIRouterTier(name="hard", models=["claude:claude-sonnet-5"]),  # type: ignore[call-arg]
            ],
            classifier=AIRouterClassifierConfig(
                engine="claude", model="claude-haiku-4-5"
            ),
            default_tier="easy",
        )
    )
    assert isinstance(client, AiRouterCompletions)
    with request_cost_scope("live-step") as scope:
        client.send_prompt(STR_EASY, max_response_tokens=32)
    report = scope.report
    assert report is not None and report.operation == "live-step"
    assert [item.role for item in report.calls] == ["classifier", "attempt"]
    assert all(item.outcome == "ok" for item in report.calls)
    assert report.classifier_usd > Decimal("0")
    assert report.total_usd > report.classifier_usd
    assert report.wasted_usd == Decimal("0")
    assert report.served_by_tier == client.last_tier
    assert report.calls[1].served and not report.calls[0].served
    # The router did not own the request, so it keeps no report of its own.
    assert client.last_request_cost is None
