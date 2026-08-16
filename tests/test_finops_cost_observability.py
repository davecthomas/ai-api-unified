# test_finops_cost_observability.py
"""
Tests for financial-ops cost enrichment on the observability middleware.

Cost is a config-gated (emit_cost) enrichment emitted on a dedicated cost topic
logger, computed from provider-reported token counts and registry pricing.
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from ai_api_unified.middleware.middleware_config import ObservabilitySettingsModel
from ai_api_unified.middleware.observability import (
    COST_LOGGER_NAME,
    LoggerBackedObservabilityMiddleware,
)
from ai_api_unified.middleware.observability_runtime import (
    AiApiCallContextModel,
    AiApiCallResultSummaryModel,
)


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def cost_capture():
    handler = _CaptureHandler()
    logger = logging.getLogger(COST_LOGGER_NAME)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield handler
    logger.removeHandler(handler)


def _context(
    model_name: str = "gpt-5.4", provider: str = "openai"
) -> AiApiCallContextModel:
    return AiApiCallContextModel(
        call_id="call-1",
        event_time_utc=datetime.now(timezone.utc),
        capability="completions",
        operation="send_prompt",
        provider_vendor=provider,
        provider_engine=provider,
        model_name=model_name,
        model_version=None,
        direction="output",
        originating_caller_id="team-x",
    )


def _summary(
    prompt: int | None = 1000,
    completion: int | None = 500,
    cached: int | None = None,
    cache_write_5m: int | None = None,
    cache_write_1h: int | None = None,
):
    return AiApiCallResultSummaryModel(
        provider_elapsed_ms=10.0,
        provider_prompt_tokens=prompt,
        provider_completion_tokens=completion,
        provider_cached_input_tokens=cached,
        provider_cache_write_5m_tokens=cache_write_5m,
        provider_cache_write_1h_tokens=cache_write_1h,
    )


def _cost_fields(handler: _CaptureHandler) -> dict:
    assert len(handler.records) == 1
    return handler.records[-1].args[1]


class TestCostEmission:
    def test_emits_cost_with_provenance(self, cost_capture) -> None:
        mw = LoggerBackedObservabilityMiddleware(
            ObservabilitySettingsModel(emit_cost=True)
        )
        mw.after_call(_context(), _summary())
        fields = _cost_fields(cost_capture)
        # gpt-5.4: 1000in*2.50/1M + 500out*15.00/1M = 0.01
        assert fields["usd_cost"] == "0.0100"
        assert fields["model"] == "gpt-5.4"
        assert fields["provider"] == "openai"
        assert fields["input_tokens"] == 1000
        assert fields["output_tokens"] == 500
        assert fields["caller_id"] == "team-x"
        assert fields["currency"] == "USD"
        assert fields["pricing_confidence"] == "high"
        assert fields["pricing_source"].startswith("https://")
        assert fields["pricing_effective_date"] == "2026-07-07"

    def test_cached_input_tokens_priced_at_cached_rate(self, cost_capture) -> None:
        mw = LoggerBackedObservabilityMiddleware(
            ObservabilitySettingsModel(emit_cost=True)
        )
        # gpt-5.4: 1000 prompt tokens of which 400 are cache reads, 500 output.
        # Non-cached input 600*2.50/1M + cached 400*0.25/1M + output 500*15.00/1M
        # = 0.0015 + 0.0001 + 0.0075 = 0.0091.
        mw.after_call(_context(), _summary(prompt=1000, completion=500, cached=400))
        fields = _cost_fields(cost_capture)
        assert fields["usd_cost"] == "0.0091"
        assert fields["input_tokens"] == 1000
        assert fields["cached_input_tokens"] == 400

    def test_cache_writes_priced_per_ttl_and_emitted(self, cost_capture) -> None:
        mw = LoggerBackedObservabilityMiddleware(
            ObservabilitySettingsModel(emit_cost=True)
        )
        # claude-opus-5: 1000 prompt, 500 output, plus 2000 tokens written to a
        # 5-minute cache (6.25/1M) and 1000 to a 1-hour cache (10.00/1M).
        # 1000*5.00/1M + 500*25.00/1M + 2000*6.25/1M + 1000*10.00/1M
        # = 0.005 + 0.0125 + 0.0125 + 0.01 = 0.0400
        mw.after_call(
            _context(model_name="claude-opus-5", provider="anthropic"),
            _summary(cache_write_5m=2000, cache_write_1h=1000),
        )
        fields = _cost_fields(cost_capture)
        assert fields["usd_cost"] == "0.0400"
        assert fields["cache_write_5m_tokens"] == 2000
        assert fields["cache_write_1h_tokens"] == 1000

    def test_cache_writes_are_additive_not_a_prompt_subset(self, cost_capture) -> None:
        """A cache write adds cost rather than re-slicing the prompt count."""
        mw = LoggerBackedObservabilityMiddleware(
            ObservabilitySettingsModel(emit_cost=True)
        )
        mw.after_call(_context(), _summary(prompt=1000, completion=500))
        cost_without = Decimal(_cost_fields(cost_capture)["usd_cost"])

        cost_capture.records.clear()
        mw.after_call(
            _context(), _summary(prompt=1000, completion=500, cache_write_5m=1000)
        )
        fields = _cost_fields(cost_capture)

        # gpt-5.4 has no cache-write rate, so the write falls back to base
        # input (2.50/1M): the prompt cost is unchanged and the write adds.
        assert Decimal(fields["usd_cost"]) == cost_without + Decimal("0.0025")
        assert fields["input_tokens"] == 1000

    def test_cache_write_only_call_still_costs(self, cost_capture) -> None:
        """A call reporting only cache writes must not be skipped as no-usage."""
        mw = LoggerBackedObservabilityMiddleware(
            ObservabilitySettingsModel(emit_cost=True)
        )
        mw.after_call(
            _context(), _summary(prompt=None, completion=None, cache_write_5m=1000)
        )
        fields = _cost_fields(cost_capture)

        assert Decimal(fields["usd_cost"]) > 0
        assert fields["cache_write_5m_tokens"] == 1000

    def test_cached_none_matches_uncached_cost(self, cost_capture) -> None:
        mw = LoggerBackedObservabilityMiddleware(
            ObservabilitySettingsModel(emit_cost=True)
        )
        mw.after_call(_context(), _summary(prompt=1000, completion=500, cached=None))
        fields = _cost_fields(cost_capture)
        # Unchanged from the no-cache case: 1000*2.50/1M + 500*15.00/1M = 0.0100.
        assert fields["usd_cost"] == "0.0100"
        assert fields["cached_input_tokens"] is None

    def test_disabled_by_default_emits_nothing(self, cost_capture) -> None:
        mw = LoggerBackedObservabilityMiddleware(ObservabilitySettingsModel())
        mw.after_call(_context(), _summary())
        assert cost_capture.records == []

    def test_unpriced_model_skips(self, cost_capture) -> None:
        mw = LoggerBackedObservabilityMiddleware(
            ObservabilitySettingsModel(emit_cost=True)
        )
        mw.after_call(_context(model_name="not-a-real-model"), _summary())
        assert cost_capture.records == []

    def test_no_token_usage_skips(self, cost_capture) -> None:
        mw = LoggerBackedObservabilityMiddleware(
            ObservabilitySettingsModel(emit_cost=True)
        )
        mw.after_call(_context(), _summary(prompt=None, completion=None))
        assert cost_capture.records == []

    def test_cost_fires_even_when_output_events_disabled(self, cost_capture) -> None:
        # input_only disables the output event; cost enrichment is independent.
        mw = LoggerBackedObservabilityMiddleware(
            ObservabilitySettingsModel(emit_cost=True, direction="input_only")
        )
        mw.after_call(_context(), _summary())
        assert len(cost_capture.records) == 1

    def test_embedding_input_only_cost(self, cost_capture) -> None:
        mw = LoggerBackedObservabilityMiddleware(
            ObservabilitySettingsModel(emit_cost=True)
        )
        ctx = AiApiCallContextModel(
            call_id="e1",
            event_time_utc=datetime.now(timezone.utc),
            capability="embedding",
            operation="generate_embeddings",
            provider_vendor="openai",
            provider_engine="openai",
            model_name="text-embedding-3-small",
            model_version=None,
            direction="output",
            originating_caller_id=None,
        )
        mw.after_call(ctx, _summary(prompt=1_000_000, completion=None))
        fields = _cost_fields(cost_capture)
        # 1M tokens * 0.02/1M = 0.02
        assert fields["usd_cost"] == "0.02"
        assert fields["output_tokens"] is None


class TestCostTopicOverride:
    def test_custom_topic_routes_events(self) -> None:
        handler = _CaptureHandler()
        topic = "custom.finops.topic"
        logger = logging.getLogger(topic)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            mw = LoggerBackedObservabilityMiddleware(
                ObservabilitySettingsModel(emit_cost=True, emit_cost_topic=topic)
            )
            mw.after_call(_context(), _summary())
            assert len(handler.records) == 1
        finally:
            logger.removeHandler(handler)
