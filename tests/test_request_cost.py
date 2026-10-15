# tests/test_request_cost.py
"""
Per-request cost: the request scope records one line item per provider call
(classifier, failed attempts, fallbacks, escalations), totals them into an
AIRequestCostReport, exposes it as `client.last_request_cost`, and emits one
`ai_api_request_cost` event on the cost topic when `emit_cost` is on.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from decimal import Decimal
from typing import Any
from unittest.mock import patch

import pytest

from ai_api_unified import (
    AIRequestCostReport,
    AiFallbackReason as R,
    AiProviderRequestError,
    request_cost_scope,
)
from ai_api_unified.completions.ai_fallback_completions import (
    DEFAULT_FALLBACK_REASONS,
    AIFallbackCandidate,
    AiFallbackCompletions,
)
from ai_api_unified.middleware.middleware_config import ObservabilitySettingsModel
from ai_api_unified.middleware.observability import (
    COST_LOGGER_NAME,
    LoggerBackedObservabilityMiddleware,
    NoOpObservabilityMiddleware,
)
from ai_api_unified.middleware.observability_runtime import (
    AiApiCallContextModel,
    AiApiCallResultSummaryModel,
    execute_observed_call,
    get_observability_context,
    set_observability_context,
    reset_observability_context,
)
from ai_api_unified.middleware.request_cost import (
    OBSERVABILITY_EVENT_REQUEST_COST,
    RequestCostScope,
    build_line_item,
    current_request_cost_scope,
)
from tests.test_model_router import (
    _FakeClassifier,
    _FakeEngine,
    _router,
    _tier,
)

# gpt-5.4: 1000 in * 2.50/1M + 500 out * 15.00/1M = 0.0100
GPT_COST: Decimal = Decimal("0.0100")
# claude-opus-5: 1000 in * 5.00/1M + 500 out * 25.00/1M = 0.0175
OPUS_COST: Decimal = Decimal("0.0175")
# The fake classifier reports 12 input tokens on gpt-5.4: 12 * 2.50/1M
CLASSIFIER_COST: Decimal = Decimal("0.00003")


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def cost_capture() -> Iterator[_CaptureHandler]:
    handler = _CaptureHandler()
    logger = logging.getLogger(COST_LOGGER_NAME)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield handler
    logger.removeHandler(handler)


def _err(reason: R | None, status: int | None = 503) -> AiProviderRequestError:
    return AiProviderRequestError(
        f"fail:{reason}", status_code=status, fallback_reason=reason
    )


def _summary(result: Any, ms: float) -> AiApiCallResultSummaryModel:
    return AiApiCallResultSummaryModel(
        provider_elapsed_ms=ms,
        provider_prompt_tokens=1000,
        provider_completion_tokens=500,
    )


class _PricedEngine(_FakeEngine):
    """
    A fake engine whose calls run through the observed-call path with a
    priced identity, so each call becomes a line item with a USD cost.
    """

    def __init__(
        self,
        name: str,
        *,
        vendor: str = "openai",
        engine: str = "openai",
        model: str = "gpt-5.4",
        fail: Exception | None = None,
    ) -> None:
        super().__init__(name, fail=fail, engine=engine)
        self.vendor = vendor
        self.engine_label = engine
        self.priced_model = model
        self._observability_middleware = NoOpObservabilityMiddleware()

    def _resolve_observability_provider_vendor(self) -> str:
        return self.vendor

    def _resolve_observability_provider_engine(self) -> str:
        return self.engine_label

    def _resolve_observability_model_name(self) -> str | None:
        return self.priced_model

    def _observed(self, op: str, result: Any) -> Any:
        def _execute() -> Any:
            self._hit(op)
            return result

        return self._execute_provider_call_with_observability(
            capability="completions",
            operation=op,
            dict_input_metadata=None,
            callable_execute=_execute,
            callable_build_result_summary=_summary,
        )

    def send_prompt(self, prompt: str, **kwargs: Any) -> str:
        return self._observed("send_prompt", f"{self.model}:ok")

    async def asend_prompt(self, prompt: str, **kwargs: Any) -> str:
        async def _execute() -> str:
            self._hit("asend_prompt")
            return f"{self.model}:ok"

        return await self._execute_provider_acall_with_observability(
            capability="completions",
            operation="asend_prompt",
            dict_input_metadata=None,
            callable_execute=_execute,
            callable_build_result_summary=_summary,
        )

    def send_prompt_streaming(
        self, prompt: str, *, other_params: Any = None
    ) -> Iterator[str]:
        def _open() -> Iterator[str]:
            self._hit("send_prompt_streaming")
            yield f"{self.model}-1"
            yield f"{self.model}-2"

        return self._execute_streaming_provider_call_with_observability(
            operation="send_prompt_streaming",
            dict_input_metadata=None,
            callable_open_stream=_open,
            callable_build_result_summary=lambda ms: _summary(None, ms),
        )


class _PricedClassifier(_FakeClassifier):
    """The router test classifier, priced as an OpenAI gpt-5.4 call."""

    def __init__(self, choice: str | None = "hard") -> None:
        super().__init__(choice)
        self._observability_middleware = NoOpObservabilityMiddleware()

    def _resolve_observability_provider_vendor(self) -> str:
        return "openai"

    def _resolve_observability_provider_engine(self) -> str:
        return "openai"

    def _resolve_observability_model_name(self) -> str | None:
        return "gpt-5.4"


def _chain(*engines: _PricedEngine) -> AiFallbackCompletions:
    """A fallback chain over priced engines, each on its own account."""
    dict_by_label: dict[str, _PricedEngine] = {
        f"fake:{engine.model}": engine for engine in engines[1:]
    }
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
        fallback_on=DEFAULT_FALLBACK_REASONS,
    )


def _context(model: str = "gpt-5.4", tags: dict[str, str] | None = None):
    from datetime import datetime, timezone

    return AiApiCallContextModel(
        call_id="call-1",
        event_time_utc=datetime.now(timezone.utc),
        capability="completions",
        operation="send_prompt",
        provider_vendor="openai",
        provider_engine="openai",
        model_name=model,
        model_version=None,
        direction="input",
        dict_tags=tags or {},
    )


# ── Line items from the observed-call path ──────────────────────────────────


class TestLineItems:
    def test_observed_call_records_ok_and_error_items_without_logger_middleware(
        self,
    ) -> None:
        with request_cost_scope("unit") as scope:
            execute_observed_call(
                observability_middleware=NoOpObservabilityMiddleware(),
                callable_build_call_context=lambda: _context(),
                callable_execute=lambda: "ok",
                callable_build_result_summary=_summary,
            )
            with pytest.raises(AiProviderRequestError):
                execute_observed_call(
                    observability_middleware=NoOpObservabilityMiddleware(),
                    callable_build_call_context=lambda: _context(),
                    callable_execute=lambda: (_ for _ in ()).throw(
                        _err(R.UNAVAILABLE, 529)
                    ),
                    callable_build_result_summary=_summary,
                )
        report: AIRequestCostReport = scope.report
        ok, failed = report.calls
        assert ok.outcome == "ok" and ok.usd_cost == GPT_COST and ok.billed
        assert failed.outcome == "error" and failed.usd_cost is None
        assert failed.error_type == "AiProviderRequestError"
        assert failed.fallback_reason == "unavailable" and not failed.billed
        assert report.billed_calls == 1 and report.failed_calls == 1
        assert report.total_usd == GPT_COST

    def test_nothing_is_recorded_outside_a_scope(self) -> None:
        assert current_request_cost_scope() is None
        execute_observed_call(
            observability_middleware=NoOpObservabilityMiddleware(),
            callable_build_call_context=lambda: _context(),
            callable_execute=lambda: "ok",
            callable_build_result_summary=_summary,
        )
        assert current_request_cost_scope() is None

    def test_line_item_cost_matches_the_per_call_cost_event(
        self, cost_capture: _CaptureHandler
    ) -> None:
        mw = LoggerBackedObservabilityMiddleware(
            ObservabilitySettingsModel(emit_cost=True)
        )
        ctx = _context(model="claude-opus-5")
        ctx = ctx.__class__(
            **{**ctx.__dict__, "provider_vendor": "anthropic", "dict_tags": {}}
        )
        summary = _summary(None, 1.0)
        mw.after_call(ctx, summary)
        item = build_line_item(
            call_context=ctx, call_result_summary=summary, float_elapsed_ms=1.0
        )
        assert str(item.usd_cost) == cost_capture.records[-1].args[1]["usd_cost"]
        assert item.usd_cost == OPUS_COST

    def test_role_tier_and_method_come_from_tags(self) -> None:
        item = build_line_item(
            call_context=_context(
                tags={
                    "router_role": "classifier",
                    "route_tier": "medium",
                    "route_method": "classifier",
                }
            ),
            call_result_summary=_summary(None, 1.0),
            float_elapsed_ms=1.0,
        )
        assert item.role == "classifier"
        assert item.route_tier == "medium" and item.route_method == "classifier"

    def test_unpriced_model_is_billed_but_unpriced(self) -> None:
        item = build_line_item(
            call_context=_context(model="no-such-model"),
            call_result_summary=_summary(None, 1.0),
            float_elapsed_ms=1.0,
        )
        assert item.billed and item.usd_cost is None
        scope = RequestCostScope("x")
        scope.record(item)
        assert scope.build_report("ok").unpriced_calls == 1


# ── Wrappers ────────────────────────────────────────────────────────────────


class TestFallbackChain:
    def test_failed_primary_and_served_fallback_are_two_items_one_billed(
        self,
    ) -> None:
        chain = _chain(
            _PricedEngine("p", fail=_err(R.UNAVAILABLE, 529)),
            _PricedEngine(
                "f", vendor="anthropic", engine="claude", model="claude-opus-5"
            ),
        )
        assert chain.send_prompt("hi") == "f:ok"
        report = chain.last_request_cost
        assert report is not None and report.status == "ok"
        assert [item.outcome for item in report.calls] == ["error", "ok"]
        assert report.billed_calls == 1 and report.failed_calls == 1
        assert report.total_usd == OPUS_COST and report.wasted_usd == Decimal("0")
        assert report.served_by == "claude:claude-opus-5"
        assert report.calls[1].served and not report.calls[0].served
        # The engines were nested, so they own no report of their own.
        assert chain.primary.last_request_cost is None

    def test_exhausted_chain_reports_error_status(self) -> None:
        chain = _chain(
            _PricedEngine("p", fail=_err(R.UNAVAILABLE)),
            _PricedEngine("f", fail=_err(R.UNAVAILABLE)),
        )
        with pytest.raises(AiProviderRequestError):
            chain.send_prompt("hi")
        report = chain.last_request_cost
        assert report.status == "error" and report.failed_calls == 2
        assert report.served_by is None and report.total_usd == Decimal("0")

    def test_async_chain(self) -> None:
        chain = _chain(_PricedEngine("p", fail=_err(R.UNAVAILABLE)), _PricedEngine("f"))
        assert asyncio.run(chain.asend_prompt("hi")) == "f:ok"
        assert chain.last_request_cost.billed_calls == 1
        assert chain.last_request_cost.operation == "asend_prompt"

    def test_streaming_chain_reports_after_consumption(self) -> None:
        chain = _chain(_PricedEngine("p", fail=_err(R.UNAVAILABLE)), _PricedEngine("f"))
        stream = chain.send_prompt_streaming("hi")
        assert chain.last_request_cost is None
        assert list(stream) == ["f-1", "f-2"]
        report = chain.last_request_cost
        assert report.operation == "send_prompt_streaming"
        assert report.failed_calls == 1 and report.billed_calls == 1
        assert report.calls[1].served
        assert current_request_cost_scope() is None


class TestRouter:
    def _example_router(self) -> tuple[Any, dict[str, _PricedEngine]]:
        """
        The issue's example: the classifier picks medium; medium's first
        candidate returns a 529, its fallback fails too, and the hard tier
        answers.
        """
        engines = {
            "claude:m": _PricedEngine(
                "m",
                vendor="anthropic",
                engine="claude",
                model="claude-opus-5",
                fail=_err(R.UNAVAILABLE, 529),
            ),
            "google-gemini:g": _PricedEngine(
                "g",
                vendor="google",
                engine="google-gemini",
                model="gemini-3-pro",
                fail=_err(R.UNAVAILABLE, 503),
            ),
            "claude:h": _PricedEngine(
                "h", vendor="anthropic", engine="claude", model="claude-opus-5"
            ),
        }
        router = _router(
            engines,
            tiers=[
                _tier("medium", "claude:m", "google-gemini:g"),
                _tier("hard", "claude:h"),
            ],
            classifier=_PricedClassifier("medium"),
        )
        return router, engines

    def test_classifier_failed_attempts_and_escalation_are_one_request(self) -> None:
        router, _ = self._example_router()
        assert router.send_prompt("summarize this contract") == "h:ok"
        report = router.last_request_cost
        assert report is not None
        assert [item.role for item in report.calls] == [
            "classifier",
            "attempt",
            "attempt",
            "attempt",
        ]
        assert [item.outcome for item in report.calls] == ["ok", "error", "error", "ok"]
        assert len(report.calls) == 4
        assert report.billed_calls == 2 and report.failed_calls == 2
        assert report.classifier_usd == CLASSIFIER_COST
        assert report.total_usd == CLASSIFIER_COST + OPUS_COST
        assert report.wasted_usd == Decimal("0")
        assert report.served_by == "claude:claude-opus-5"
        assert (
            report.served_by_tier == "hard" and report.served_by_method == "classifier"
        )
        assert [item.route_tier for item in report.calls] == [
            None,
            "medium",
            "medium",
            "hard",
        ]

    def test_request_id_is_the_route_id_on_every_call(self) -> None:
        router, engines = self._example_router()
        router.send_prompt("x")
        report = router.last_request_cost
        for engine in engines.values():
            for tags in engine.tags_seen:
                assert tags["request_id"] == report.request_id
                assert tags["route_id"] == report.request_id
        assert router.classifier.tags_seen[0]["request_id"] == report.request_id
        assert dict(get_observability_context().tags) == {}

    def test_routed_log_prices_the_classifier_from_the_ledger(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        router, _ = self._example_router()
        with caplog.at_level(
            logging.INFO, logger="ai_api_unified.completions.ai_router_completions"
        ):
            router.send_prompt("x")
        routed = [
            r for r in caplog.records if getattr(r, "ai_router_event", "") == "routed"
        ]
        assert routed and routed[0].classifier_usd_cost == float(CLASSIFIER_COST)

    def test_async_router(self) -> None:
        router, _ = self._example_router()
        assert asyncio.run(router.asend_prompt("x")) == "h:ok"
        assert router.last_request_cost.billed_calls == 2

    def test_streaming_router(self) -> None:
        router, _ = self._example_router()
        assert list(router.send_prompt_streaming("x")) == ["h-1", "h-2"]
        report = router.last_request_cost
        assert report.operation == "send_prompt_streaming"
        assert len(report.calls) == 4 and report.calls[-1].served

    def test_wasted_is_a_billed_answer_nobody_used(self) -> None:
        scope = RequestCostScope("x")
        ok = build_line_item(
            call_context=_context(),
            call_result_summary=_summary(None, 1),
            float_elapsed_ms=1,
        )
        scope.record(ok)
        scope.record(ok.model_copy(update={"call_id": "call-2"}))
        scope.mark_served()
        report = scope.build_report("ok")
        assert report.total_usd == GPT_COST * 2
        assert report.wasted_usd == GPT_COST
        assert [item.served for item in report.calls] == [False, True]


class TestPlainEngineAndCallerScope:
    def test_plain_engine_call_is_a_one_call_request(self) -> None:
        engine = _PricedEngine("e")
        engine.send_prompt("hi")
        report = engine.last_request_cost
        assert report is not None and len(report.calls) == 1
        assert report.calls[0].served and report.total_usd == GPT_COST
        assert report.operation == "send_prompt"
        assert current_request_cost_scope() is None

    def test_caller_scope_composes_several_calls(self) -> None:
        engine = _PricedEngine("e")
        chain = _chain(_PricedEngine("p"))
        with request_cost_scope("workflow-step") as scope:
            assert scope.bool_owner
            engine.send_prompt("a")
            chain.send_prompt("b")
            with request_cost_scope("inner") as inner:
                assert inner is scope
        assert scope.report.operation == "workflow-step"
        assert len(scope.report.calls) == 2
        assert scope.report.total_usd == GPT_COST * 2
        assert [item.served for item in scope.report.calls] == [True, True]
        # Nested clients did not own the request, so they keep no report.
        assert engine.last_request_cost is None
        assert chain.last_request_cost is None

    def test_caller_scope_keeps_context_ids_and_tags(self) -> None:
        token = set_observability_context(
            caller_id="svc", session_id="s1", workflow_id="w1", tags={"team": "x"}
        )
        try:
            with request_cost_scope("step", request_id="req-1") as scope:
                assert get_observability_context().tags["request_id"] == "req-1"
            assert get_observability_context().tags == {"team": "x"}
        finally:
            reset_observability_context(token)
        report = scope.report
        assert report.request_id == "req-1"
        assert (report.caller_id, report.session_id, report.workflow_id) == (
            "svc",
            "s1",
            "w1",
        )
        assert report.tags == {"team": "x"}

    def test_scope_closes_with_error_status_on_exception(self) -> None:
        with pytest.raises(RuntimeError):
            with request_cost_scope("step") as scope:
                raise RuntimeError("boom")
        assert scope.report.status == "error"
        assert current_request_cost_scope() is None


# ── Emission ────────────────────────────────────────────────────────────────


class TestEmission:
    def test_request_cost_event_on_the_cost_topic(
        self, cost_capture: _CaptureHandler
    ) -> None:
        mw = LoggerBackedObservabilityMiddleware(
            ObservabilitySettingsModel(emit_cost=True)
        )
        chain = _chain(
            _PricedEngine("p", fail=_err(R.UNAVAILABLE, 529)), _PricedEngine("f")
        )
        for engine in (chain.primary, chain._client_at(1)):
            engine._observability_middleware = mw
        with patch(
            "ai_api_unified.middleware.observability.get_observability_middleware",
            return_value=mw,
        ):
            chain.send_prompt("hi")
        events = {r.args[0]: r.args[1] for r in cost_capture.records}
        assert set(events) == {"ai_api_call_cost", OBSERVABILITY_EVENT_REQUEST_COST}
        request = events[OBSERVABILITY_EVENT_REQUEST_COST]
        assert request["status"] == "ok" and request["operation"] == "send_prompt"
        assert request["total_usd"] == str(GPT_COST)
        assert request["classifier_usd"] == "0" and request["wasted_usd"] == "0"
        assert request["calls_count"] == 2
        assert request["billed_calls"] == 1 and request["failed_calls"] == 1
        assert request["served_by"] == "openai:gpt-5.4"
        assert [c["outcome"] for c in request["calls"]] == ["error", "ok"]
        assert request["calls"][0]["fallback_reason"] == "unavailable"
        assert request["calls"][1]["usd_cost"] == str(GPT_COST)
        assert request["elapsed_ms"] >= 0
        # The per-call cost event correlates with the request.
        per_call = events["ai_api_call_cost"]
        assert per_call["tag_request_id"] == request["request_id"]
        assert per_call["call_id"] == request["calls"][1]["call_id"]

    def test_no_request_event_when_emit_cost_is_off(
        self, cost_capture: _CaptureHandler
    ) -> None:
        mw = LoggerBackedObservabilityMiddleware(ObservabilitySettingsModel())
        engine = _PricedEngine("e")
        engine._observability_middleware = mw
        with patch(
            "ai_api_unified.middleware.observability.get_observability_middleware",
            return_value=mw,
        ):
            engine.send_prompt("hi")
        assert cost_capture.records == []
        # The report is still available programmatically.
        assert engine.last_request_cost.total_usd == GPT_COST

    def test_emission_failure_does_not_reach_the_caller(self) -> None:
        engine = _PricedEngine("e")
        with patch(
            "ai_api_unified.middleware.observability.get_observability_middleware",
            side_effect=RuntimeError("sink down"),
        ):
            assert engine.send_prompt("hi") == "e:ok"
        assert engine.last_request_cost is not None
