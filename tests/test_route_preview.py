# tests/test_route_preview.py
"""
Route preview: where a request would go and what it would cost, without
sending it to the answering model. A plain engine previews itself, a
fallback chain its first ready candidate, and a router runs its classifier.
"""

from __future__ import annotations

import math
from decimal import Decimal
from unittest.mock import Mock

import pytest

pytest.importorskip("anthropic")

from ai_api_unified import AIRoutePreview, AIThinkingLevel  # noqa: E402
from ai_api_unified.ai_base import INT_PREVIEW_CHARS_PER_TOKEN  # noqa: E402
from ai_api_unified.pricing.pricing_registry import get_model_pricing  # noqa: E402
from tests.test_model_fallback import _FakeEngine as _FbFake  # noqa: E402
from tests.test_model_fallback import _chain, _err  # noqa: E402
from tests.test_model_router import (  # noqa: E402
    _FakeClassifier,
    _router,
    _three,
    _three_tiers,
)
from tests.test_prompt_cache_hints import _anthropic  # noqa: E402

STR_PROMPT: str = "Summarize the attached contract in three bullet points."


class TestEnginePreview:
    def test_estimate_cost_and_no_call(self) -> None:
        client = _anthropic("claude-opus-4-8")
        preview = client.preview_route(STR_PROMPT, system_prompt="Be brief.")
        assert isinstance(preview, AIRoutePreview)
        assert preview.engine == "claude" and preview.model == "claude-opus-4-8"
        assert preview.method == "direct" and preview.tier is None
        int_expected = math.ceil(
            len("Be brief.\n" + STR_PROMPT) / INT_PREVIEW_CHARS_PER_TOKEN
        )
        assert preview.input_tokens == int_expected
        assert preview.input_token_source == "estimated"
        assert preview.max_output_tokens == client.SEND_PROMPT_MAX_TOKENS
        rates = get_model_pricing("anthropic", "claude-opus-4-8").token_rates
        assert preview.usd_input == Decimal(
            int_expected
        ) * rates.input_per_1m / Decimal(1_000_000)
        assert preview.usd_max == preview.usd_input + Decimal(
            preview.max_output_tokens
        ) * rates.output_per_1m / Decimal(1_000_000)
        assert preview.pricing_source and preview.pricing_effective_date
        assert preview.thinking is not None
        assert preview.thinking.requested is AIThinkingLevel.MEDIUM
        client.client.messages.create.assert_not_called()

    def test_provider_count_when_asked(self) -> None:
        client = _anthropic("claude-opus-4-8")
        client.client.messages.count_tokens.return_value = Mock(input_tokens=42)
        preview = client.preview_route(
            STR_PROMPT, max_response_tokens=500, bool_count_tokens=True
        )
        assert preview.input_tokens == 42 and preview.input_token_source == "provider"
        assert preview.max_output_tokens == 500

    def test_conversation_text_counts(self) -> None:
        client = _anthropic("claude-opus-4-8")
        preview = client.preview_route(
            messages=[
                {"role": "user", "content": "a" * 40},
                {"role": "assistant", "content": [{"type": "text", "text": "b" * 40}]},
            ]
        )
        assert preview.input_tokens == 21  # 80 characters plus the joining newline


class TestChainPreview:
    def test_primary_unless_resting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ai_api_unified.completions.ai_fallback_cooldown import (
            AiCandidateCooldowns,
        )

        primary = _FbFake("p", fail=_err(None))
        chain = _chain(primary, _FbFake("f"))
        assert chain.preview_route("hi").model == "p"
        AiCandidateCooldowns.bench(
            str_engine="fake",
            str_model="p",
            base_url="https://p.test",
            reason=__import__("ai_api_unified").AiFallbackReason.UNAVAILABLE,
            bool_account_wide=False,
            float_seconds=30,
            str_label="fake:p",
        )
        assert chain.preview_route("hi").model == "f"
        assert primary.calls == []


class TestRouterPreview:
    def test_classifier_picks_the_tier_and_no_tier_model_is_called(self) -> None:
        engines = _three()
        classifier = _FakeClassifier("hard", confidence=0.91)
        router = _router(engines, tiers=_three_tiers(), classifier=classifier)
        preview = router.preview_route(STR_PROMPT)
        assert preview.tier == "hard" and preview.method == "classifier"
        assert preview.confidence == 0.91 and preview.model == "h"
        assert len(classifier.calls) == 1
        assert all(engine.calls == [] for engine in engines.values())

    def test_override_and_floor_apply_without_classifying_twice(self) -> None:
        engines = _three()
        classifier = _FakeClassifier("easy")
        router = _router(engines, tiers=_three_tiers(), classifier=classifier)
        pinned = router.preview_route(STR_PROMPT, provider_options={"route": "medium"})
        assert pinned.tier == "medium" and pinned.method == "override"
        floored = router.preview_route(
            STR_PROMPT, provider_options={"route_floor": "hard"}
        )
        assert floored.tier == "hard" and floored.method == "floor"

    def test_no_classifier_previews_the_default_route(self) -> None:
        router = _router(_three(), tiers=_three_tiers(), classifier=None)
        preview = router.preview_route(STR_PROMPT)
        assert preview.tier is None and preview.method == "default"
        assert preview.model == "d"
        assert preview.classifier_usd == Decimal("0")
