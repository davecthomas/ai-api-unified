# tests/test_metering.py
"""
Token and cost metering: Gemini output counts include thinking tokens (billed
at the output rate), line items carry the rates' source and date, and the
request report attributes cost by service, tenant, and purpose.
"""

from __future__ import annotations

import json
import os
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

genai = pytest.importorskip("google.genai")
pytest.importorskip("anthropic")

from ai_api_unified import AIFactory  # noqa: E402
from ai_api_unified.ai_models_config import clear_models_config_cache  # noqa: E402
from ai_api_unified.completions.ai_google_gemini_completions import (  # noqa: E402
    GoogleGeminiCompletions,
)
from ai_api_unified.middleware.observability_runtime import (  # noqa: E402
    reset_observability_context,
    set_observability_context,
)
from ai_api_unified.pricing.pricing_registry import (  # noqa: E402
    PROVIDER_GOOGLE,
    get_model_pricing,
)
from tests.test_prompt_cache_hints import _anthropic  # noqa: E402


def _usage(**counts: Any) -> SimpleNamespace:
    return SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=counts.get("prompt", 15),
            candidates_token_count=counts.get("candidates"),
            thoughts_token_count=counts.get("thoughts"),
            total_token_count=counts.get("total"),
            cached_content_token_count=None,
        )
    )


class TestGeminiThinkingTokens:
    @pytest.mark.parametrize(
        ("candidates", "thoughts", "output", "reasoning"),
        [
            (3, 336, 339, 336),
            (12, None, 12, None),
            (None, 40, 40, 40),
            (None, None, None, None),
        ],
    )
    def test_output_is_answer_plus_thinking(
        self, candidates: Any, thoughts: Any, output: Any, reasoning: Any
    ) -> None:
        response = _usage(candidates=candidates, thoughts=thoughts)
        assert (
            GoogleGeminiCompletions._extract_gemini_completion_tokens(response)
            == output
        )
        assert (
            GoogleGeminiCompletions._extract_gemini_thought_tokens(response)
            == reasoning
        )

    def test_non_integer_fields_are_ignored(self) -> None:
        # Test doubles (Mock) answer any attribute; only ints count.
        response = SimpleNamespace(usage_metadata=Mock(candidates_token_count=5))
        assert GoogleGeminiCompletions._extract_gemini_completion_tokens(response) == 5

    def test_a_thinking_call_is_priced_on_all_output_tokens(self) -> None:
        with patch.object(
            GoogleGeminiCompletions,
            "_initialize_client",
            lambda self: setattr(self, "client", Mock()),
        ):
            engine = GoogleGeminiCompletions(model="gemini-2.5-flash")
        engine.client.models.generate_content.return_value = Mock(
            candidates=[
                Mock(
                    content=Mock(parts=[genai.types.Part(text="391")]),
                    finish_reason="FinishReason.STOP",
                )
            ],
            usage_metadata=SimpleNamespace(
                prompt_token_count=15,
                candidates_token_count=3,
                thoughts_token_count=336,
                total_token_count=354,
                cached_content_token_count=None,
            ),
            text="391",
        )
        engine.send_prompt("What is 17*23?")
        item = engine.last_request_cost.calls[0]
        assert item.output_tokens == 339 and item.reasoning_tokens == 336
        rates = get_model_pricing(PROVIDER_GOOGLE, "gemini-2.5-flash").token_rates
        expected = (
            Decimal(15) * rates.input_per_1m + Decimal(339) * rates.output_per_1m
        ) / Decimal(1_000_000)
        assert item.usd_cost == expected
        assert item.pricing_source and item.pricing_source.startswith("http")
        assert item.pricing_effective_date is not None
        fields = item.to_event_fields()
        assert fields["reasoning_tokens"] == 336
        assert (
            fields["pricing_effective_date"] == item.pricing_effective_date.isoformat()
        )


class TestAttribution:
    def test_service_from_models_json_tags_every_request(self, tmp_path: Any) -> None:
        path = tmp_path / "models.json"
        path.write_text(
            json.dumps(
                {
                    "service": "billing-api",
                    "completions": {"engine": "claude", "model": "claude-haiku-4-5"},
                }
            )
        )
        clear_models_config_cache()
        try:
            with patch.dict(
                os.environ,
                {"AI_MODELS_CONFIG": str(path), "ANTHROPIC_API_KEY": "test-key"},
            ):
                client = AIFactory.get_ai_completions_client()
        finally:
            clear_models_config_cache()
        client.client = Mock()
        client.client.messages.create.return_value = Mock(
            content=[Mock(type="text", text="hi")],
            stop_reason="end_turn",
            usage=Mock(input_tokens=5, output_tokens=1, cache_read_input_tokens=None),
        )
        client.send_conversation("sys", [{"role": "user", "content": "hi"}])
        assert client.last_request_cost.service == "billing-api"

    def test_tenant_and_purpose_reach_the_report_and_purpose_is_stripped(self) -> None:
        client = _anthropic("claude-opus-4-8")
        token = set_observability_context(tags={"tenant": "acme"})
        try:
            client.send_conversation(
                "sys",
                [{"role": "user", "content": "hi"}],
                provider_options={"purpose": "summarize"},
            )
        finally:
            reset_observability_context(token)
        report = client.last_request_cost
        assert report.tenant == "acme" and report.purpose == "summarize"
        fields = report.to_event_fields()
        assert fields["tenant"] == "acme" and fields["purpose"] == "summarize"
        assert "purpose" not in client.client.messages.create.call_args.kwargs

    def test_purpose_through_a_fallback_chain_reaches_the_report(self) -> None:
        from ai_api_unified import AIFallbackCandidate, AiFallbackCompletions

        primary = _anthropic("claude-opus-4-8")
        chain = AiFallbackCompletions(
            primary=primary,
            primary_candidate=AIFallbackCandidate(
                engine="claude", model="claude-opus-4-8"
            ),
            fallback_candidates=[AIFallbackCandidate(engine="claude", model="x")],
            client_builder=Mock(),
        )
        chain.send_conversation(
            "sys",
            [{"role": "user", "content": "hi"}],
            provider_options={"purpose": "triage"},
        )
        assert chain.last_request_cost.purpose == "triage"
        assert chain.last_request_cost.calls[0].served

    def test_no_attribution_is_none(self) -> None:
        client = _anthropic("claude-opus-4-8")
        client.send_conversation("sys", [{"role": "user", "content": "hi"}])
        report = client.last_request_cost
        assert (
            report.service is None and report.tenant is None and report.purpose is None
        )
