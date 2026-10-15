# tests/test_billed_failures.py
"""
Billed failures: a structured answer the library could not use raises
StructuredOutputValidationError carrying the provider's usage, the request
ledger and the cost topic bill it, and AiFallbackReason.INVALID_OUTPUT (opt-in)
lets the chain try the next model and the router escalate, leaving the
discarded answer as a wasted line item.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from decimal import Decimal
from typing import Any
from unittest.mock import Mock, patch

import pytest

from ai_api_unified import (
    AIBaseCompletions,
    AICompletionsCapabilitiesBase,
    AIFinishReason,
    AIRouterConfig,
    AIStructuredOutputResult,
    AIStructuredPrompt,
    AITokenUsage,
    AiFallbackReason as R,
    AiProviderRequestError,
    AiRouterCompletions,
    StructuredOutputValidationError,
    StructuredResponseTokenLimitError,
    get_model_pricing,
)
from ai_api_unified.completions.ai_fallback_completions import (
    DEFAULT_FALLBACK_REASONS,
    AIFallbackCandidate,
    AiFallbackCompletions,
    parse_fallback_reasons,
)
from ai_api_unified.middleware.middleware_config import ObservabilitySettingsModel
from ai_api_unified.middleware.observability import (
    COST_LOGGER_NAME,
    LoggerBackedObservabilityMiddleware,
)
from ai_api_unified.middleware.request_cost import (
    build_line_item,
    summary_from_exception_usage,
)
from tests.test_model_router import _tier
from tests.test_request_cost import (
    GPT_COST,
    OPUS_COST,
    _CaptureHandler,
    _PricedClassifier,
    _PricedEngine,
    _context,
    _err,
)

USAGE: AITokenUsage = AITokenUsage(input_tokens=1000, output_tokens=500)
WITH_INVALID_OUTPUT: frozenset[R] = DEFAULT_FALLBACK_REASONS | {R.INVALID_OUTPUT}


@pytest.fixture
def cost_capture() -> Iterator[_CaptureHandler]:
    handler = _CaptureHandler()
    logger = logging.getLogger(COST_LOGGER_NAME)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield handler
    logger.removeHandler(handler)


class _Summary(AIStructuredPrompt):
    title: str

    @staticmethod
    def get_prompt(input_text: str = "") -> str:
        return input_text


class _StructuredEngine(_PricedEngine):
    """
    A priced engine whose structured output runs through the observed call.
    `answer` is the data to return; `bad_json=True` raises the billed
    invalid-output error inside the observed call, the way a real engine's
    result builder does.
    """

    def __init__(
        self,
        name: str,
        *,
        answer: dict[str, Any] | None = None,
        bad_json: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.answer: dict[str, Any] = answer if answer is not None else {"title": name}
        self.bad_json = bad_json

    @property
    def capabilities(self) -> AICompletionsCapabilitiesBase:
        return self._caps.model_copy(update={"context_window_length": 100_000})

    def send_structured_output(self, prompt: Any = None, **kwargs: Any) -> Any:
        # The router test fake short-circuits this; run the real template so
        # capability gating and response-model validation apply.
        return AIBaseCompletions.send_structured_output(self, prompt, **kwargs)

    def _send_structured_output_provider(self, **kwargs: Any) -> Any:
        def _execute() -> AIStructuredOutputResult:
            self._hit("send_structured_output")
            if self.bad_json:
                raise StructuredOutputValidationError(
                    "Invalid JSON response: boom",
                    usage=USAGE,
                    raw_text="not json",
                    finish_reason="complete",
                )
            return AIStructuredOutputResult(
                data=self.answer, finish_reason=AIFinishReason.COMPLETE, usage=USAGE
            )

        from tests.test_request_cost import _summary

        return self._execute_provider_call_with_observability(
            capability="completions",
            operation="send_structured_output",
            dict_input_metadata=None,
            callable_execute=_execute,
            callable_build_result_summary=_summary,
        )


def _chain(
    *engines: _StructuredEngine, fallback_on: frozenset[R] = DEFAULT_FALLBACK_REASONS
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
        fallback_on=fallback_on,
    )


# ── The error and the ledger ────────────────────────────────────────────────


class TestBilledFailureLineItem:
    def test_error_is_a_value_error_and_an_invalid_output_trigger(self) -> None:
        error = StructuredOutputValidationError("bad", usage=USAGE, raw_text="x")
        assert isinstance(error, ValueError)
        assert isinstance(error, AiProviderRequestError)
        assert error.fallback_reason is R.INVALID_OUTPUT
        assert error.status_code is None and not error.account_wide
        assert error.usage is USAGE and error.raw_text == "x"

    def test_line_item_from_exception_usage_is_billed_and_priced(self) -> None:
        item = build_line_item(
            call_context=_context(),
            call_result_summary=None,
            float_elapsed_ms=1.0,
            exception=StructuredOutputValidationError("bad", usage=USAGE),
        )
        assert item.outcome == "error" and item.billed
        assert item.usd_cost == GPT_COST
        assert item.input_tokens == 1000 and item.output_tokens == 500
        assert item.error_type == "StructuredOutputValidationError"
        assert item.fallback_reason == "invalid_output"

    def test_transport_failure_stays_unbilled(self) -> None:
        item = build_line_item(
            call_context=_context(),
            call_result_summary=None,
            float_elapsed_ms=1.0,
            exception=_err(R.UNAVAILABLE, 529),
        )
        assert not item.billed and item.usd_cost is None
        assert summary_from_exception_usage(_err(R.UNAVAILABLE), 1.0) is None

    def test_token_limit_error_carries_usage(self) -> None:
        error = StructuredResponseTokenLimitError(
            message="truncated",
            provider_name="openai",
            model_name="gpt-5.4",
            max_response_tokens=64,
            minimum_supported_tokens=16,
            usage=AITokenUsage(input_tokens=10, output_tokens=64),
        )
        summary = summary_from_exception_usage(error, 2.0)
        assert summary is not None
        assert (summary.provider_prompt_tokens, summary.provider_completion_tokens) == (
            10,
            64,
        )
        assert (
            summary_from_exception_usage(
                StructuredOutputValidationError("x", usage=AITokenUsage()), 1.0
            )
            is None
        )

    def test_invalid_output_is_a_known_reason_and_not_a_default(self) -> None:
        assert parse_fallback_reasons("unavailable,invalid_output") == {
            R.UNAVAILABLE,
            R.INVALID_OUTPUT,
        }
        assert R.INVALID_OUTPUT not in DEFAULT_FALLBACK_REASONS


class TestCostTopic:
    def test_billed_failure_emits_a_cost_event_marked_error(
        self, cost_capture: _CaptureHandler
    ) -> None:
        mw = LoggerBackedObservabilityMiddleware(
            ObservabilitySettingsModel(emit_cost=True)
        )
        engine = _StructuredEngine("e", bad_json=True)
        engine._observability_middleware = mw
        with patch(
            "ai_api_unified.middleware.observability.get_observability_middleware",
            return_value=mw,
        ):
            with pytest.raises(StructuredOutputValidationError):
                engine.send_structured_output("p", response_model=_Summary)
        events = {r.args[0]: r.args[1] for r in cost_capture.records}
        per_call = events["ai_api_call_cost"]
        assert per_call["outcome"] == "error"
        assert per_call["error_type"] == "StructuredOutputValidationError"
        assert per_call["usd_cost"] == str(GPT_COST)
        request = events["ai_api_request_cost"]
        assert request["status"] == "error"
        assert request["billed_calls"] == 1 and request["failed_calls"] == 1
        assert request["total_usd"] == str(GPT_COST)
        assert request["calls"][0]["outcome"] == "error"

    def test_successful_cost_event_says_ok(self, cost_capture: _CaptureHandler) -> None:
        mw = LoggerBackedObservabilityMiddleware(
            ObservabilitySettingsModel(emit_cost=True)
        )
        from tests.test_request_cost import _summary

        mw.after_call(_context(), _summary(None, 1.0))
        fields = cost_capture.records[-1].args[1]
        assert fields["outcome"] == "ok" and fields["error_type"] is None


# ── Fallback and router behavior ────────────────────────────────────────────


class TestFallbackOnInvalidOutput:
    def test_off_by_default_the_error_propagates_and_is_billed(self) -> None:
        chain = _chain(_StructuredEngine("p", bad_json=True), _StructuredEngine("f"))
        with pytest.raises(ValueError, match="Invalid JSON"):
            chain.send_structured_output("p", response_model=_Summary)
        report = chain.last_request_cost
        assert report.status == "error" and len(report.calls) == 1
        assert report.billed_calls == 1 and report.failed_calls == 1
        assert report.total_usd == GPT_COST and report.wasted_usd == GPT_COST
        assert chain._client_at(1).calls == []

    def test_opt_in_tries_the_next_model_and_wastes_the_first(self) -> None:
        chain = _chain(
            _StructuredEngine("p", bad_json=True),
            _StructuredEngine("f"),
            fallback_on=WITH_INVALID_OUTPUT,
        )
        result = chain.send_structured_output("p", response_model=_Summary)
        assert result.data == {"title": "f"}
        report = chain.last_request_cost
        assert [item.outcome for item in report.calls] == ["error", "ok"]
        assert report.billed_calls == 2 and report.failed_calls == 1
        assert report.total_usd == GPT_COST * 2
        assert report.wasted_usd == GPT_COST
        assert report.served_by == "openai:gpt-5.4" and report.calls[1].served
        assert report.calls[0].fallback_reason == "invalid_output"

    def test_response_model_failure_after_the_call_is_also_a_trigger(self) -> None:
        """
        Validation runs after the observed call, so the first item is a
        successful, billed call that nobody used.
        """
        chain = _chain(
            _StructuredEngine("p", answer={"wrong": 1}),
            _StructuredEngine("f"),
            fallback_on=WITH_INVALID_OUTPUT,
        )
        result = chain.send_structured_output("p", response_model=_Summary)
        assert result.data == {"title": "f"}
        report = chain.last_request_cost
        assert [item.outcome for item in report.calls] == ["ok", "ok"]
        assert [item.served for item in report.calls] == [False, True]
        assert report.wasted_usd == GPT_COST and report.total_usd == GPT_COST * 2

    def test_response_model_failure_raises_a_typed_error_with_usage(self) -> None:
        engine = _StructuredEngine("p", answer={"wrong": 1})
        with pytest.raises(StructuredOutputValidationError) as info:
            engine.send_structured_output("p", response_model=_Summary)
        assert info.value.usage == USAGE
        assert info.value.finish_reason == "complete"
        assert "validation errors" in str(info.value)


class TestRouterEscalation:
    def _example_router(self, fallback_on: frozenset[R]) -> AiRouterCompletions:
        """
        The issue's example: medium's first candidate returns a 529, its
        fallback answers with invalid JSON (billed, unused), and hard answers.
        """
        gemini_model = "gemini-3.5-flash"
        engines = {
            "claude:m": _StructuredEngine(
                "m",
                vendor="anthropic",
                engine="claude",
                model="claude-opus-5",
                fail=_err(R.UNAVAILABLE, 529),
            ),
            "google-gemini:g": _StructuredEngine(
                "g",
                vendor="google",
                engine="google-gemini",
                model=gemini_model,
                bad_json=True,
            ),
            "claude:h": _StructuredEngine(
                "h", vendor="anthropic", engine="claude", model="claude-opus-5"
            ),
        }
        config = AIRouterConfig(
            tiers=[
                _tier("medium", "claude:m", "google-gemini:g"),
                _tier("hard", "claude:h"),
            ]
        )
        return AiRouterCompletions(
            default_client=_StructuredEngine("d"),
            config=config,
            classifier=_PricedClassifier("medium"),
            client_builder=Mock(side_effect=lambda c: engines[c.label]),
            fallback_on=fallback_on,
        )

    def test_the_issue_example_four_calls_three_billed_one_wasted(self) -> None:
        router = self._example_router(WITH_INVALID_OUTPUT)
        result = router.send_structured_output("summarize", response_model=_Summary)
        assert result.data == {"title": "h"}
        report = router.last_request_cost
        gemini_cost = get_model_pricing(
            "google", "gemini-3.5-flash"
        ).compute_token_cost(
            input_tokens=1000,
            output_tokens=500,
            cached_input_tokens=0,
            cache_write_5m_tokens=0,
            cache_write_1h_tokens=0,
        )
        assert [item.role for item in report.calls] == [
            "classifier",
            "attempt",
            "attempt",
            "attempt",
        ]
        assert [item.outcome for item in report.calls] == ["ok", "error", "error", "ok"]
        assert len(report.calls) == 4
        assert report.billed_calls == 3 and report.failed_calls == 2
        assert report.total_usd == report.classifier_usd + gemini_cost + OPUS_COST
        assert report.wasted_usd == gemini_cost
        assert report.served_by_tier == "hard"
        assert report.calls[2].error_type == "StructuredOutputValidationError"
        assert report.calls[1].usd_cost is None and not report.calls[1].billed

    def test_without_opt_in_the_router_raises_the_billed_failure(self) -> None:
        router = self._example_router(DEFAULT_FALLBACK_REASONS)
        with pytest.raises(StructuredOutputValidationError):
            router.send_structured_output("summarize", response_model=_Summary)
        report = router.last_request_cost
        assert report.status == "error" and len(report.calls) == 3
        assert report.billed_calls == 2 and report.wasted_usd > Decimal("0")
