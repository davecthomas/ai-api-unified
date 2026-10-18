# tests/test_prompt_result.py
"""
send_prompt_with_result: the response text plus which provider and model
answered, the router tier and method, the thinking level, the token usage,
and the cost, all for this call alone.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import pytest

from ai_api_unified import (
    AiFallbackReason as R,
    AIPromptResult,
    AIThinkingLevel,
    request_cost_scope,
)
from ai_api_unified.middleware.request_cost import (
    REQUEST_CALL_CAPTURES,
    capture_request_calls,
    note_call_thinking,
)
from ai_api_unified.middleware.observability_runtime import push_observability_tags
from ai_api_unified.thinking import AIThinkingResolution
from tests.test_model_router import _tier
from tests.test_request_cost import (
    CLASSIFIER_COST,
    GPT_COST,
    OPUS_COST,
    _chain,
    _err,
    _PricedClassifier,
    _PricedEngine,
)
from ai_api_unified.completions.ai_fallback_completions import (
    DEFAULT_FALLBACK_REASONS,
)
from ai_api_unified.completions.ai_router_completions import AiRouterCompletions
from ai_api_unified.routing.router_config import AIRouterConfig


class _ThinkingEngine(_PricedEngine):
    """A priced engine that resolves a thinking level per call, as engines do."""

    def send_prompt(self, prompt: str, **kwargs: Any) -> str:
        self._thinking_for_call(kwargs.get("other_params"))
        return super().send_prompt(prompt, **kwargs)


def _priced_router(
    engines: dict[str, _PricedEngine], *, classifier: Any = None
) -> AiRouterCompletions:
    return AiRouterCompletions(
        default_client=_PricedEngine("d"),
        config=AIRouterConfig(
            tiers=[_tier("easy", "claude:e"), _tier("hard", "claude:h")]
        ),
        classifier=classifier,
        client_builder=lambda candidate: engines[candidate.label],
        fallback_on=DEFAULT_FALLBACK_REASONS,
        cooldown=False,
    )


class TestEnvelope:
    def test_single_engine_reports_model_usage_and_cost(self) -> None:
        engine = _PricedEngine("p")
        result: AIPromptResult = engine.send_prompt_with_result("hi")
        assert result.text == "p:ok"
        assert result.provider == "openai" and result.provider_engine == "openai"
        assert result.model_name == "gpt-5.4"
        assert result.usage.input_tokens == 1000
        assert result.usage.output_tokens == 500
        assert result.usage.total_tokens == 1500
        assert result.total_usd == GPT_COST
        assert result.route_tier is None and result.route_method is None
        assert result.model_pin is None
        assert result.request_id == engine.last_request_cost.request_id

    def test_send_prompt_still_returns_plain_text(self) -> None:
        assert _PricedEngine("p").send_prompt("hi") == "p:ok"

    def test_fallback_reports_the_serving_candidate_and_the_wasted_attempt(
        self,
    ) -> None:
        chain = _chain(
            _PricedEngine("p", fail=_err(R.UNAVAILABLE, 529)),
            _PricedEngine(
                "f", vendor="anthropic", engine="claude", model="claude-opus-5"
            ),
        )
        result = chain.send_prompt_with_result("hi")
        assert result.text == "f:ok"
        assert result.provider == "anthropic"
        # The chain's candidate token, as on turn and structured results.
        assert result.provider_engine == "fake"
        assert result.model_name == "claude-opus-5"
        assert result.total_usd == OPUS_COST
        assert result.cost.failed_calls == 1 and len(result.cost.calls) == 2

    def test_router_reports_tier_method_and_classifier_cost(self) -> None:
        router = _priced_router(
            {
                "claude:e": _PricedEngine("e"),
                "claude:h": _PricedEngine("h"),
            },
            classifier=_PricedClassifier("hard"),
        )
        result = router.send_prompt_with_result("prove it")
        assert result.text == "h:ok"
        assert result.route_tier == "hard" and result.route_method == "classifier"
        assert result.cost.classifier_usd == CLASSIFIER_COST
        assert result.total_usd == GPT_COST + CLASSIFIER_COST
        # The router's route id stays the request id.
        assert result.request_id == router.last_request_cost.request_id

    def test_config_identity_rides_on_the_result(self) -> None:
        engine = _PricedEngine("p")
        engine.set_config_identity("abc123", "release-7")
        result = engine.send_prompt_with_result("hi")
        assert result.config_id == "abc123"
        assert result.config_version == "release-7"

    def test_result_is_immutable(self) -> None:
        result = _PricedEngine("p").send_prompt_with_result("hi")
        with pytest.raises(Exception):
            result.text = "changed"  # type: ignore[misc]


class TestIsolation:
    def test_calls_inside_a_callers_request_each_report_only_their_own(
        self,
    ) -> None:
        cheap = _PricedEngine("c")
        dear = _PricedEngine("x", vendor="anthropic", model="claude-opus-5")
        with request_cost_scope("workflow-step") as scope:
            first = cheap.send_prompt_with_result("a")
            second = dear.send_prompt_with_result("b")
        assert first.total_usd == GPT_COST and len(first.cost.calls) == 1
        assert second.total_usd == OPUS_COST and len(second.cost.calls) == 1
        assert first.request_id == second.request_id == scope.request_id
        assert scope.report.total_usd == GPT_COST + OPUS_COST

    def test_concurrent_async_calls_do_not_share_a_ledger(self) -> None:
        chain = _chain(
            _PricedEngine("p"),
            _PricedEngine("f", vendor="anthropic", model="claude-opus-5"),
        )

        async def _both() -> list[AIPromptResult]:
            return list(
                await asyncio.gather(
                    chain.asend_prompt_with_result("one"),
                    chain.asend_prompt_with_result("two", model="f"),
                )
            )

        first, second = asyncio.run(_both())
        assert first.model_name == "gpt-5.4" and len(first.cost.calls) == 1
        assert second.model_name == "claude-opus-5" and len(second.cost.calls) == 1
        assert first.request_id != second.request_id

    def test_capture_is_removed_after_the_call(self) -> None:
        _PricedEngine("p").send_prompt_with_result("hi")
        assert REQUEST_CALL_CAPTURES.get() == ()

    def test_capture_is_removed_when_the_call_raises(self) -> None:
        engine = _PricedEngine("p", fail=_err(R.UNAVAILABLE, 529))
        with pytest.raises(Exception):
            engine.send_prompt_with_result("hi")
        assert REQUEST_CALL_CAPTURES.get() == ()


class TestThinking:
    def test_result_carries_the_served_calls_level(self) -> None:
        engine = _ThinkingEngine("p")
        engine.thinking_level = AIThinkingLevel.HIGH
        result = engine.send_prompt_with_result("hi")
        assert result.thinking is not None
        assert result.thinking.requested is AIThinkingLevel.HIGH

    def test_the_classifiers_level_is_not_the_answers(self) -> None:
        resolution = AIThinkingResolution(
            requested=AIThinkingLevel.OFF, applied=AIThinkingLevel.OFF, exact=True
        )
        with capture_request_calls("x") as capture:
            token = push_observability_tags({"router_role": "classifier"})
            try:
                note_call_thinking(resolution)
            finally:
                from ai_api_unified.middleware.observability_runtime import (
                    reset_observability_context,
                )

                reset_observability_context(token)
        assert capture.thinking is None
        with capture_request_calls("y") as capture:
            note_call_thinking(resolution)
        assert capture.thinking == resolution


class TestAsync:
    def test_async_twin_matches_the_sync_envelope(self) -> None:
        engine = _PricedEngine("p")
        result = asyncio.run(engine.asend_prompt_with_result("hi"))
        assert result.text == "p:ok"
        assert result.total_usd == GPT_COST
        assert result.usage.total_tokens == 1500
        assert isinstance(result.total_usd, Decimal)
