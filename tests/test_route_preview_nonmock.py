# tests/test_route_preview_nonmock.py
"""
Live route preview: the real Decisions classifier picks a tier, the preview
prices the tier's model, and no answering model is called.

    poetry run pytest -m nonmock tests/test_route_preview_nonmock.py -q -s
"""

from __future__ import annotations

import os
import socket
from decimal import Decimal

import pytest

from ai_api_unified import AIFactory, AIRouterConfig, AiRouterCompletions

pytestmark = pytest.mark.nonmock


def test_preview_classifies_and_prices_without_answering() -> None:
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
                {"name": "easy", "models": ["claude:claude-haiku-4-5"]},
                {"name": "hard", "models": ["claude:claude-opus-5-5"]},
            ],
            "classifier": {"engine": "openai-decisions", "model": "gpt-6-luna"},
        }
    )
    client = AIFactory.get_ai_completions_client(
        completions_engine="claude", model_name="claude-haiku-4-5", router=config
    )
    assert isinstance(client, AiRouterCompletions)
    preview = client.preview_route(
        "Prove that there are infinitely many primes, then generalize the "
        "argument to primes of the form 4k+3.",
        max_response_tokens=2000,
    )
    print(f"\n  {preview.model_dump(mode='json')}")
    assert preview.tier in ("easy", "hard") and preview.method == "classifier"
    assert preview.model in ("claude-haiku-4-5", "claude-opus-5-5")
    assert preview.usd_max is not None and preview.usd_max > preview.usd_input
    assert preview.classifier_usd > Decimal("0")
    # The preview sent nothing to the tier's model.
    assert client.last_request_cost is None or all(
        item.role == "classifier" for item in client.last_request_cost.calls
    )
