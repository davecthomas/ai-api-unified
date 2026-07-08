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
