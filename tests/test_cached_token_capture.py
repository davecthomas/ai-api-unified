# test_cached_token_capture.py
"""
Provider-level cached-input-token extraction for finops cost accuracy.

Each provider reports prompt-cache reads differently. These tests verify the
per-provider extractors and the library-wide invariant that
provider_prompt_tokens includes the cached subset (so the cost middleware can
split it out and bill cache reads at the cached rate).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


class TestOpenAIChatCached:
    def test_extracts_cached_tokens_subset_of_prompt(self) -> None:
        pytest.importorskip("openai")
        from ai_api_unified.completions.ai_openai_completions import (
            AiOpenAICompletions,
        )

        completion = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=1000,
                completion_tokens=200,
                total_tokens=1200,
                prompt_tokens_details=SimpleNamespace(cached_tokens=400),
            )
        )
        # prompt_tokens already includes cached, so it is unchanged; cached is
        # extracted separately as the subset.
        assert AiOpenAICompletions._extract_openai_prompt_tokens(completion) == 1000
        assert AiOpenAICompletions._extract_openai_cached_tokens(completion) == 400

    def test_missing_cache_details_returns_none(self) -> None:
        pytest.importorskip("openai")
        from ai_api_unified.completions.ai_openai_completions import (
            AiOpenAICompletions,
        )

        completion = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=2, total_tokens=12
            )
        )
        assert AiOpenAICompletions._extract_openai_cached_tokens(completion) is None


class TestOpenAIResponsesCached:
    def test_usage_tuple_includes_cached(self) -> None:
        pytest.importorskip("openai")
        from ai_api_unified.completions.ai_openai_responses_completions import (
            AiOpenAIResponsesCompletions,
        )

        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=900,
                output_tokens=100,
                total_tokens=1000,
                input_tokens_details=SimpleNamespace(cached_tokens=300),
            )
        )
        prompt, output, total, cached = (
            AiOpenAIResponsesCompletions._extract_responses_usage(response)
        )
        assert (prompt, output, total, cached) == (900, 100, 1000, 300)

    def test_no_usage_returns_all_none(self) -> None:
        pytest.importorskip("openai")
        from ai_api_unified.completions.ai_openai_responses_completions import (
            AiOpenAIResponsesCompletions,
        )

        assert AiOpenAIResponsesCompletions._extract_responses_usage(
            SimpleNamespace(usage=None)
        ) == (None, None, None, None)


class TestAnthropicCachedFold:
    def test_fold_adds_cache_reads_to_prompt(self) -> None:
        pytest.importorskip("anthropic")
        from ai_api_unified.completions.ai_anthropic_completions import (
            AiAnthropicCompletions,
        )

        # Anthropic input_tokens EXCLUDES cache reads, so the fold adds them.
        prompt, _, cached = AiAnthropicCompletions._fold_anthropic_prompt_tokens(
            600, 400
        )
        assert prompt == 1000
        assert cached == 400

    def test_fold_all_none(self) -> None:
        pytest.importorskip("anthropic")
        from ai_api_unified.completions.ai_anthropic_completions import (
            AiAnthropicCompletions,
        )

        assert AiAnthropicCompletions._fold_anthropic_prompt_tokens(None, None) == (
            None,
            None,
            None,
        )

    def test_extract_usage_folds_and_reports_total(self) -> None:
        pytest.importorskip("anthropic")
        from ai_api_unified.completions.ai_anthropic_completions import (
            AiAnthropicCompletions,
        )

        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=600,
                output_tokens=100,
                cache_read_input_tokens=400,
            )
        )
        prompt, output, total, cached = AiAnthropicCompletions._extract_anthropic_usage(
            response
        )
        # prompt folds in cache reads (1000), total = prompt + output (1100).
        assert (prompt, output, total, cached) == (1000, 100, 1100, 400)


class TestAnthropicCacheWriteCapture:
    """Cache writes bill at a premium and are reported per cache lifetime."""

    @staticmethod
    def _extract(usage_obj):
        pytest.importorskip("anthropic")
        from ai_api_unified.completions.ai_anthropic_completions import (
            AiAnthropicCompletions,
        )

        return AiAnthropicCompletions._extract_anthropic_cache_write_tokens(
            SimpleNamespace(usage=usage_obj)
        )

    def test_extracts_per_ttl_breakdown(self) -> None:
        assert self._extract(
            SimpleNamespace(
                input_tokens=600,
                cache_creation_input_tokens=3000,
                cache_creation=SimpleNamespace(
                    ephemeral_5m_input_tokens=2000,
                    ephemeral_1h_input_tokens=1000,
                ),
            )
        ) == (2000, 1000)

    def test_unknown_ttl_remainder_attributed_to_five_minute(self) -> None:
        """A TTL tier this code does not know must not bill as free.

        Its tokens appear in the aggregate but not the known split; the
        remainder is attributed to the 5-minute tier (the lowest premium) so
        it under-reports rather than vanishes.
        """
        assert self._extract(
            SimpleNamespace(
                input_tokens=600,
                cache_creation_input_tokens=3500,  # 500 from an unknown tier
                cache_creation=SimpleNamespace(
                    ephemeral_5m_input_tokens=2000,
                    ephemeral_1h_input_tokens=1000,
                ),
            )
        ) == (2500, 1000)

    def test_split_matching_aggregate_is_unchanged(self) -> None:
        assert self._extract(
            SimpleNamespace(
                input_tokens=600,
                cache_creation_input_tokens=3000,
                cache_creation=SimpleNamespace(
                    ephemeral_5m_input_tokens=2000,
                    ephemeral_1h_input_tokens=1000,
                ),
            )
        ) == (2000, 1000)

    def test_falls_back_to_aggregate_as_five_minute(self) -> None:
        """Older payloads report only the aggregate; 5m is the default TTL."""
        assert self._extract(
            SimpleNamespace(input_tokens=600, cache_creation_input_tokens=2500)
        ) == (2500, None)

    def test_no_cache_write_usage_reports_none(self) -> None:
        assert self._extract(SimpleNamespace(input_tokens=600)) == (None, None)

    def test_missing_usage_block_reports_none(self) -> None:
        pytest.importorskip("anthropic")
        from ai_api_unified.completions.ai_anthropic_completions import (
            AiAnthropicCompletions,
        )

        assert AiAnthropicCompletions._extract_anthropic_cache_write_tokens(
            SimpleNamespace()
        ) == (None, None)

    def test_cache_writes_are_not_folded_into_prompt_tokens(self) -> None:
        """Writes are billed separately, so they must stay out of the prompt count.

        Cache reads fold in (they are a discounted slice of the prompt); folding
        writes in too would double-bill them at the input rate.
        """
        pytest.importorskip("anthropic")
        from ai_api_unified.completions.ai_anthropic_completions import (
            AiAnthropicCompletions,
        )

        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=600,
                output_tokens=100,
                cache_read_input_tokens=400,
                cache_creation_input_tokens=5000,
            )
        )
        prompt, _, _, _ = AiAnthropicCompletions._extract_anthropic_usage(response)

        # 600 input + 400 cache reads; the 5000 written tokens are excluded.
        assert prompt == 1000


class TestBedrockCacheWriteCapture:
    def test_attributes_converse_writes_to_five_minute_tier(self) -> None:
        pytest.importorskip("boto3")
        from ai_api_unified.completions.ai_bedrock_completions import (
            AiBedrockCompletions,
        )

        assert AiBedrockCompletions._cache_write_kwargs(
            {"usage": {"inputTokens": 600, "cacheWriteInputTokens": 1500}}
        ) == {
            "provider_cache_write_5m_tokens": 1500,
            "provider_cache_write_1h_tokens": None,
        }

    def test_absent_cache_write_usage_reports_none(self) -> None:
        pytest.importorskip("boto3")
        from ai_api_unified.completions.ai_bedrock_completions import (
            AiBedrockCompletions,
        )

        assert AiBedrockCompletions._cache_write_kwargs({"usage": {}}) == {
            "provider_cache_write_5m_tokens": None,
            "provider_cache_write_1h_tokens": None,
        }


class TestTurnResultCacheWriteExposure:
    """Result objects must expose the same write counts the cost stream bills.

    Consumers pricing calls from AITurnResult.usage rather than the cost log
    would otherwise have no way to see cache-priming tokens.
    """

    def test_anthropic_turn_usage_carries_cache_writes(self) -> None:
        pytest.importorskip("anthropic")
        from ai_api_unified.completions.ai_anthropic_completions import (
            AiAnthropicCompletions,
        )

        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=600,
                output_tokens=100,
                cache_read_input_tokens=400,
                cache_creation_input_tokens=3000,
                cache_creation=SimpleNamespace(
                    ephemeral_5m_input_tokens=2000,
                    ephemeral_1h_input_tokens=1000,
                ),
            )
        )
        usage = AiAnthropicCompletions._usage_from_tuple(
            AiAnthropicCompletions,  # unbound instance-method call in a test
            AiAnthropicCompletions._extract_anthropic_usage(response),
            dict_cache_write=AiAnthropicCompletions._cache_write_kwargs(response),
        )

        assert usage.cache_write_5m_tokens == 2000
        assert usage.cache_write_1h_tokens == 1000
        # Writes stay out of the prompt-side counts.
        assert usage.input_tokens == 1000

    def test_bedrock_turn_usage_carries_cache_writes(self) -> None:
        pytest.importorskip("boto3")
        from ai_api_unified.completions.ai_bedrock_completions import (
            AiBedrockCompletions,
        )

        usage = AiBedrockCompletions._usage_from_converse(
            AiBedrockCompletions,
            {
                "usage": {
                    "inputTokens": 600,
                    "outputTokens": 100,
                    "totalTokens": 700,
                    "cacheWriteInputTokens": 1500,
                }
            },
        )

        assert usage.cache_write_5m_tokens == 1500
        assert usage.cache_write_1h_tokens is None

    def test_public_cost_api_accepts_cache_writes(self) -> None:
        """compute_completion_cost prices writes like the cost middleware does."""
        from decimal import Decimal
        from datetime import date
        from ai_api_unified.pricing import (
            AIModelPricing,
            AITokenRates,
            PricingUnit,
        )

        pricing = AIModelPricing(
            unit=PricingUnit.TOKEN,
            effective_date=date(2026, 7, 7),
            source="x",
            token_rates=AITokenRates(
                input_per_1m=Decimal("5.00"),
                output_per_1m=Decimal("25.00"),
                cache_write_5m_per_1m=Decimal("6.25"),
                cache_write_1h_per_1m=Decimal("10.00"),
            ),
        )

        class _Capabilities:
            def __init__(self) -> None:
                self.pricing = pricing

        class _Client:
            capabilities = _Capabilities()

        from ai_api_unified.ai_base import AIBaseCompletions

        cost = AIBaseCompletions.compute_completion_cost(
            _Client(),
            input_tokens=1000,
            output_tokens=500,
            cache_write_5m_tokens=2000,
            cache_write_1h_tokens=1000,
        )

        # 1000*5/1M + 500*25/1M + 2000*6.25/1M + 1000*10/1M = 0.0400
        assert cost == 0.0400


class TestResultModelPositionalCompatibility:
    """New dataclass fields append after the pre-2.24 layout.

    External middleware or test doubles constructing these positionally must
    keep their original argument mapping in a minor release.
    """

    def test_call_result_summary_field_order_prefix_unchanged(self) -> None:
        import dataclasses
        from ai_api_unified.middleware.observability_runtime import (
            AiApiCallResultSummaryModel,
        )

        names = [f.name for f in dataclasses.fields(AiApiCallResultSummaryModel)]
        assert names[: len(_PRE_224_SUMMARY_FIELDS)] == _PRE_224_SUMMARY_FIELDS

    def test_observed_completions_field_order_prefix_unchanged(self) -> None:
        import dataclasses
        from ai_api_unified.ai_base import AiApiObservedCompletionsResultModel

        names = [
            f.name for f in dataclasses.fields(AiApiObservedCompletionsResultModel)
        ]
        assert names[: len(_PRE_224_OBSERVED_FIELDS)] == _PRE_224_OBSERVED_FIELDS


_PRE_224_SUMMARY_FIELDS = [
    "provider_elapsed_ms",
    "input_token_count",
    "input_token_count_source",
    "output_token_count",
    "output_token_count_source",
    "provider_prompt_tokens",
    "provider_completion_tokens",
    "provider_cached_input_tokens",
    "provider_total_tokens",
    "finish_reason",
    "dict_metadata",
]

_PRE_224_OBSERVED_FIELDS = [
    "return_value",
    "raw_output_text",
    "finish_reason",
    "provider_prompt_tokens",
    "provider_completion_tokens",
    "provider_cached_input_tokens",
    "provider_total_tokens",
    "dict_metadata",
]


class TestCacheWriteWiringCompleteness:
    """Every path that reports cache reads must also report cache writes.

    The two are extracted from the same usage payload, so a result-summary
    construction that reports one and not the other is an oversight: the call
    bills its cache discount but silently drops the write premium. This caught
    two Bedrock paths (the Converse tool-loop turn and the structured-output
    provider) that a first pass missed.
    """

    @staticmethod
    def _unwired_sites(module_filename: str) -> list[int]:
        """Return line numbers of result constructions missing cache-write kwargs.

        Walks the AST rather than grepping source text: a formatter reflow must
        not fail this, and a nearby comment mentioning cache_write must not make
        a genuinely unwired site pass.
        """
        import ast
        import importlib
        import inspect

        module = importlib.import_module(
            f"ai_api_unified.completions.{module_filename}"
        )
        tree = ast.parse(inspect.getsource(module))
        list_unwired: list[int] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            str_name = getattr(func, "id", None) or getattr(func, "attr", None)
            if str_name != "AiApiObservedCompletionsResultModel":
                continue
            set_kwargs = {kw.arg for kw in node.keywords if kw.arg is not None}
            if "provider_cached_input_tokens" not in set_kwargs:
                continue
            # A **splat carries arg=None; those sites pass the kwargs as a dict.
            bool_has_splat = any(kw.arg is None for kw in node.keywords)
            bool_has_explicit = "provider_cache_write_5m_tokens" in set_kwargs
            if not (bool_has_splat or bool_has_explicit):
                list_unwired.append(node.lineno)
        return list_unwired

    def test_anthropic_reports_writes_wherever_it_reports_reads(self) -> None:
        pytest.importorskip("anthropic")

        assert self._unwired_sites("ai_anthropic_completions") == []

    def test_bedrock_reports_writes_wherever_it_reports_reads(self) -> None:
        pytest.importorskip("boto3")

        assert self._unwired_sites("ai_bedrock_completions") == []


class TestBedrockCachedFold:
    def test_prompt_folds_cache_reads(self) -> None:
        pytest.importorskip("boto3")
        from ai_api_unified.completions.ai_bedrock_completions import (
            AiBedrockCompletions,
        )

        response = {
            "usage": {
                "inputTokens": 600,
                "outputTokens": 100,
                "totalTokens": 700,
                "cacheReadInputTokens": 400,
            }
        }
        assert AiBedrockCompletions._extract_bedrock_prompt_tokens(response) == 1000
        assert AiBedrockCompletions._extract_bedrock_cached_tokens(response) == 400
        # Total is recomputed from the folded prompt + output so the emitted
        # triple stays consistent (prompt + completion = total), even though the
        # raw provider totalTokens (700) excludes cache reads.
        assert AiBedrockCompletions._extract_bedrock_total_tokens(response) == 1100

    def test_no_cache_leaves_prompt_unchanged(self) -> None:
        pytest.importorskip("boto3")
        from ai_api_unified.completions.ai_bedrock_completions import (
            AiBedrockCompletions,
        )

        response = {
            "usage": {"inputTokens": 600, "outputTokens": 100, "totalTokens": 700}
        }
        assert AiBedrockCompletions._extract_bedrock_prompt_tokens(response) == 600
        assert AiBedrockCompletions._extract_bedrock_cached_tokens(response) is None
        # With no cache reads the recomputed total equals the provider total.
        assert AiBedrockCompletions._extract_bedrock_total_tokens(response) == 700


class TestGeminiCached:
    def test_extracts_cached_content_token_count(self) -> None:
        pytest.importorskip("google.genai")
        from ai_api_unified.completions.ai_google_gemini_completions import (
            GoogleGeminiCompletions,
        )

        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=1000,
                candidates_token_count=200,
                total_token_count=1200,
                cached_content_token_count=350,
            )
        )
        # prompt_token_count already includes cached; unchanged.
        assert GoogleGeminiCompletions._extract_gemini_prompt_tokens(response) == 1000
        assert GoogleGeminiCompletions._extract_gemini_cached_tokens(response) == 350

    def test_no_usage_metadata_returns_none(self) -> None:
        pytest.importorskip("google.genai")
        from ai_api_unified.completions.ai_google_gemini_completions import (
            GoogleGeminiCompletions,
        )

        assert (
            GoogleGeminiCompletions._extract_gemini_cached_tokens(
                SimpleNamespace(usage_metadata=None)
            )
            is None
        )
