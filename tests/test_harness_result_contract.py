"""
The result contract a harness engine relies on (3.29.0): the RUN_FAILED
fallback reason, reported costs, the cost report on a raised exception,
route_confidence on results, and the candidate_index observability tag.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

import pytest

from ai_api_unified import (
    AiFallbackReason as R,
    AiProviderRequestError,
    request_cost_scope,
)
from ai_api_unified.completions.ai_fallback_completions import (
    DEFAULT_FALLBACK_REASONS,
    TAG_CANDIDATE_INDEX,
    AIFallbackCandidate,
    AiFallbackCompletions,
)
from ai_api_unified.completions.ai_fallback_cooldown import (
    DICT_DEFAULT_COOLDOWN_SECONDS,
    AiCandidateCooldowns,
    AIFallbackCooldownPolicy,
)
from ai_api_unified.middleware.middleware_config import ObservabilitySettingsModel
from ai_api_unified.middleware.observability import (
    PRICING_CONFIDENCE_REPORTED,
    LoggerBackedObservabilityMiddleware,
)
from ai_api_unified.middleware.observability_runtime import (
    AiApiCallResultSummaryModel,
)
from ai_api_unified.middleware.request_cost import build_line_item
from tests.test_prompt_result import _priced_router
from tests.test_request_cost import (
    GPT_COST,
    _CaptureHandler,
    _chain,
    _context,
    _err,
    _PricedClassifier,
    _PricedEngine,
)
from ai_api_unified.middleware.observability import COST_LOGGER_NAME

RUN_USD: Decimal = Decimal("0.4200")
GATEWAY: str = "http_gateway"


def _reported_summary(result: Any, ms: float) -> AiApiCallResultSummaryModel:
    return AiApiCallResultSummaryModel(
        provider_elapsed_ms=ms,
        provider_prompt_tokens=90_000,
        provider_completion_tokens=4_000,
        provider_reported_usd_cost=RUN_USD,
        provider_reported_cost_source=GATEWAY,
    )


class _RunEngine(_PricedEngine):
    """A priced fake whose calls report their own dollars, as a harness run does."""

    def __init__(self, name: str, **kwargs: Any) -> None:
        super().__init__(name, model="harness-run", **kwargs)

    def send_prompt(self, prompt: str, **kwargs: Any) -> str:
        def _execute() -> str:
            self._hit("send_prompt")
            return f"{self.model}:ok"

        return self._execute_provider_call_with_observability(
            capability="completions",
            operation="send_prompt",
            dict_input_metadata=None,
            callable_execute=_execute,
            callable_build_result_summary=_reported_summary,
        )


def _run_failed(usd: Decimal | None = RUN_USD) -> AiProviderRequestError:
    return AiProviderRequestError(
        "run ended in error",
        fallback_reason=R.RUN_FAILED,
        reported_usd_cost=usd,
        reported_cost_source=GATEWAY,
    )


def _chain_on(
    reasons: frozenset[R], *engines: _PricedEngine, cooldown: bool = True
) -> AiFallbackCompletions:
    dict_by_label = {f"fake:{engine.model}": engine for engine in engines[1:]}
    return AiFallbackCompletions(
        primary=engines[0],
        primary_candidate=AIFallbackCandidate(
            engine="fake", model=str(engines[0].model), base_url="https://p.test"
        ),
        fallback_candidates=[
            AIFallbackCandidate(
                engine="fake",
                model=str(engine.model),
                base_url=f"https://{engine.model}.test",
            )
            for engine in engines[1:]
        ],
        client_builder=lambda candidate: dict_by_label[candidate.label],
        fallback_on=reasons,
        cooldown=cooldown,
    )


@pytest.fixture
def cost_capture() -> Any:
    handler = _CaptureHandler()
    logger = logging.getLogger(COST_LOGGER_NAME)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield handler
    logger.removeHandler(handler)


@pytest.fixture(autouse=True)
def _no_cooldowns() -> Any:
    AiCandidateCooldowns.reset()
    yield
    AiCandidateCooldowns.reset()


# ── RUN_FAILED ──────────────────────────────────────────────────────────────


class TestRunFailed:
    def test_is_opt_in_and_starts_no_cooldown(self) -> None:
        assert R("run_failed") is R.RUN_FAILED
        assert R.RUN_FAILED not in DEFAULT_FALLBACK_REASONS
        assert DICT_DEFAULT_COOLDOWN_SECONDS[R.RUN_FAILED] == 0.0
        assert AIFallbackCooldownPolicy().seconds_for(R.RUN_FAILED) == 0.0

    def test_off_by_default_the_chain_raises_the_failed_run(self) -> None:
        second = _PricedEngine("b")
        chain = _chain(_PricedEngine("a", fail=_run_failed()), second)
        with pytest.raises(AiProviderRequestError) as info:
            chain.send_prompt("fix the bug")
        assert info.value.fallback_reason is R.RUN_FAILED
        assert second.calls == []

    def test_opted_in_the_chain_moves_on_and_rests_nobody(self) -> None:
        chain = _chain_on(
            DEFAULT_FALLBACK_REASONS | {R.RUN_FAILED},
            _PricedEngine("a", fail=_run_failed()),
            _PricedEngine("b"),
        )
        assert chain.send_prompt("fix the bug") == "b:ok"
        assert not AiCandidateCooldowns.resting(
            str_engine="fake", str_model="a", base_url="https://p.test"
        )


# ── Reported costs ──────────────────────────────────────────────────────────


class TestReportedCost:
    def test_line_item_uses_the_reported_dollars_and_source(self) -> None:
        item = build_line_item(
            call_context=_context(model="harness-run"),
            call_result_summary=_reported_summary(None, 1.0),
            float_elapsed_ms=1.0,
        )
        assert item.usd_cost == RUN_USD and item.billed
        assert item.pricing_source == GATEWAY
        assert item.pricing_effective_date is None

    def test_reported_cost_wins_over_registry_pricing(self) -> None:
        item = build_line_item(
            call_context=_context(model="gpt-5.4"),
            call_result_summary=_reported_summary(None, 1.0),
            float_elapsed_ms=1.0,
        )
        assert item.usd_cost == RUN_USD and item.pricing_source == GATEWAY

    def test_cost_event_agrees_with_the_line_item(
        self, cost_capture: _CaptureHandler
    ) -> None:
        middleware = LoggerBackedObservabilityMiddleware(
            ObservabilitySettingsModel(emit_cost=True)
        )
        context = _context(model="harness-run")
        middleware.after_call(context, _reported_summary(None, 1.0))
        dict_fields: dict[str, Any] = cost_capture.records[-1].args[1]
        assert dict_fields["usd_cost"] == str(RUN_USD)
        assert dict_fields["pricing_source"] == GATEWAY
        assert dict_fields["pricing_effective_date"] is None
        assert dict_fields["pricing_confidence"] == PRICING_CONFIDENCE_REPORTED
        assert dict_fields["currency"] == "USD"

    def test_a_failed_run_with_a_reported_cost_is_billed(self) -> None:
        item = build_line_item(
            call_context=_context(model="harness-run"),
            call_result_summary=None,
            float_elapsed_ms=1.0,
            exception=_run_failed(),
        )
        assert item.outcome == "error" and item.fallback_reason == "run_failed"
        assert item.usd_cost == RUN_USD and item.billed

    def test_a_served_run_totals_its_reported_cost(self) -> None:
        engine = _RunEngine("run")
        with request_cost_scope("unit") as scope:
            engine.send_prompt("fix the bug")
        assert scope.report.total_usd == RUN_USD


# ── exception.cost ──────────────────────────────────────────────────────────


class TestExceptionCost:
    def test_exhausted_chain_raises_with_every_try_on_cost(self) -> None:
        chain = _chain_on(
            DEFAULT_FALLBACK_REASONS | {R.RUN_FAILED},
            _PricedEngine("a", model="harness-run", fail=_run_failed()),
            _PricedEngine("b", model="harness-run", fail=_run_failed()),
        )
        with pytest.raises(AiProviderRequestError) as info:
            chain.send_prompt("fix the bug")
        report = info.value.cost
        assert report is not None and report.status == "error"
        assert len(report.calls) == 2
        assert [item.fallback_reason for item in report.calls] == [
            "run_failed",
            "run_failed",
        ]
        assert report.total_usd == RUN_USD * 2
        assert report is chain.last_request_cost

    def test_unbilled_failure_still_carries_a_report(self) -> None:
        engine = _PricedEngine("p", fail=_err(R.UNAVAILABLE, 529))
        with pytest.raises(AiProviderRequestError) as info:
            engine.send_prompt("hi")
        assert info.value.cost is not None
        assert info.value.cost.failed_calls == 1
        assert info.value.cost.total_usd == Decimal("0")

    def test_caller_scope_attaches_its_own_report(self) -> None:
        served = _PricedEngine("ok")
        failing = _PricedEngine("p", fail=_err(R.UNAVAILABLE, 529))
        with pytest.raises(AiProviderRequestError) as info:
            with request_cost_scope("workflow-step") as scope:
                served.send_prompt("one")
                failing.send_prompt("two")
        assert info.value.cost is scope.report
        assert len(info.value.cost.calls) == 2
        assert info.value.cost.total_usd == GPT_COST

    def test_cost_is_none_on_a_fresh_error(self) -> None:
        assert _err(R.UNAVAILABLE).cost is None


# ── route_confidence ────────────────────────────────────────────────────────


class TestRouteConfidence:
    def test_router_result_carries_the_classifier_confidence(self) -> None:
        router = _priced_router(
            {"claude:e": _PricedEngine("e"), "claude:h": _PricedEngine("h")},
            classifier=_PricedClassifier("hard"),
        )
        result = router.send_prompt_with_result("prove it")
        assert result.route_method == "classifier"
        assert result.route_confidence == pytest.approx(0.8)

    def test_no_classifier_means_no_confidence(self) -> None:
        router = _priced_router(
            {"claude:e": _PricedEngine("e"), "claude:h": _PricedEngine("h")},
        )
        assert router.send_prompt_with_result("hi").route_confidence is None

    def test_plain_engine_has_no_confidence(self) -> None:
        assert _PricedEngine("p").send_prompt_with_result("hi").route_confidence is None


# ── candidate_index ─────────────────────────────────────────────────────────


class TestCandidateIndex:
    def test_each_try_sees_its_chain_position(self) -> None:
        first = _PricedEngine("a", fail=_err(R.UNAVAILABLE, 503))
        second = _PricedEngine("b")
        chain = _chain_on(DEFAULT_FALLBACK_REASONS, first, second, cooldown=False)
        assert chain.send_prompt("hi") == "b:ok"
        assert first.tags_seen[-1][TAG_CANDIDATE_INDEX] == "0"
        assert second.tags_seen[-1][TAG_CANDIDATE_INDEX] == "1"

    def test_position_does_not_follow_cooldown_order(self) -> None:
        first = _PricedEngine("a", fail=_err(R.UNAVAILABLE, 503))
        second = _PricedEngine("b")
        chain = _chain_on(DEFAULT_FALLBACK_REASONS, first, second)
        chain.send_prompt("hi")
        # The primary now rests, so the next call starts at "b".
        chain.send_prompt("hi again")
        assert second.tags_seen[-1][TAG_CANDIDATE_INDEX] == "1"

    def test_tag_is_gone_after_the_call(self) -> None:
        from ai_api_unified.middleware.observability_runtime import (
            get_observability_context,
        )

        chain = _chain_on(
            DEFAULT_FALLBACK_REASONS, _PricedEngine("a"), _PricedEngine("b")
        )
        chain.send_prompt("hi")
        assert TAG_CANDIDATE_INDEX not in get_observability_context().tags

    def test_every_tier_numbers_from_zero_so_the_tier_tag_tells_them_apart(
        self,
    ) -> None:
        easy = _PricedEngine("e", fail=_err(R.UNAVAILABLE, 503))
        hard = _PricedEngine("h")
        router = _priced_router(
            {"claude:e": easy, "claude:h": hard},
            classifier=_PricedClassifier("easy"),
        )
        assert router.send_prompt("summarize") == "h:ok"
        easy_tags, hard_tags = easy.tags_seen[-1], hard.tags_seen[-1]
        assert easy_tags[TAG_CANDIDATE_INDEX] == hard_tags[TAG_CANDIDATE_INDEX] == "0"
        assert (easy_tags["route_tier"], hard_tags["route_tier"]) == ("easy", "hard")
