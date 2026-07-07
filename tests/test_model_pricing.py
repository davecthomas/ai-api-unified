# test_model_pricing.py
"""
Tests for the structured pricing descriptor, registry, and lifecycle policy.
"""

import os
import warnings
from decimal import Decimal
from unittest.mock import patch

import pytest

from ai_api_unified.ai_provider_exceptions import AiProviderConfigurationError
from ai_api_unified.pricing import (
    AIModelPricing,
    AITokenRates,
    ModelLifecycleStatus,
    PricingUnit,
    enforce_model_lifecycle,
    get_model_info,
    get_model_pricing,
)
from ai_api_unified.pricing import pricing_registry


class TestPricingMath:
    """AIModelPricing cost computation and blended shim."""

    @staticmethod
    def _pricing() -> AIModelPricing:
        return AIModelPricing(
            unit=PricingUnit.TOKEN,
            effective_date=__import__("datetime").date(2026, 7, 7),
            source="https://example.test",
            token_rates=AITokenRates(
                input_per_1m=Decimal("2.50"),
                output_per_1m=Decimal("15.00"),
                cached_input_per_1m=Decimal("0.25"),
            ),
        )

    def test_input_output_cost(self) -> None:
        # 1000 input * 2.50/1M + 500 output * 15.00/1M = 0.0025 + 0.0075
        assert self._pricing().compute_token_cost(
            input_tokens=1000, output_tokens=500
        ) == Decimal("0.0100")

    def test_cached_input_billed_at_cached_rate(self) -> None:
        # 1000 cached * 0.25/1M = 0.00025
        assert self._pricing().compute_token_cost(
            input_tokens=0, cached_input_tokens=1000
        ) == Decimal("0.00025")

    def test_blended_per_1k(self) -> None:
        # mean(2.50, 15.00) = 8.75 per 1M -> 0.00875 per 1K
        assert self._pricing().blended_per_1k_tokens() == pytest.approx(0.00875)

    def test_compute_requires_token_rates(self) -> None:
        image_pricing = AIModelPricing(
            unit=PricingUnit.IMAGE,
            effective_date=__import__("datetime").date(2026, 7, 7),
            source="x",
            per_image_usd=Decimal("0.04"),
        )
        with pytest.raises(ValueError, match="requires token_rates"):
            image_pricing.compute_token_cost(input_tokens=10)


class TestRegistry:
    """Registry lookups return the researched rates."""

    def test_openai_gpt_5_4(self) -> None:
        pricing = get_model_pricing("openai", "gpt-5.4")
        assert pricing is not None
        assert pricing.token_rates.input_per_1m == Decimal("2.50")
        assert pricing.token_rates.output_per_1m == Decimal("15.00")

    def test_new_codex_model_present(self) -> None:
        assert get_model_pricing("openai", "gpt-5.1-codex-max") is not None

    def test_uncatalogued_model_returns_none(self) -> None:
        assert get_model_pricing("openai", "does-not-exist") is None


class TestLifecycle:
    """enforce_model_lifecycle policy per status."""

    def setup_method(self) -> None:
        pricing_registry._SET_WARNED_MODELS.clear()

    def test_active_model_passes_silently(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            enforce_model_lifecycle("openai", "gpt-5.4")
        assert caught == []

    def test_uncatalogued_model_passes_silently(self) -> None:
        enforce_model_lifecycle("openai", "unknown-model")  # no raise

    def test_retired_model_raises(self) -> None:
        with pytest.raises(AiProviderConfigurationError, match="retired"):
            enforce_model_lifecycle("google", "gemini-1.5-pro-002")

    def test_deprecated_model_warns_once(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            enforce_model_lifecycle("google", "gemini-2.0-flash")
            enforce_model_lifecycle("google", "gemini-2.0-flash")
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep) == 1  # deduped per process
        assert "gemini-2.5-flash" in str(dep[0].message)  # names replacement

    def test_strict_mode_escalates_deprecated_to_error(self) -> None:
        with patch.dict(os.environ, {"AI_STRICT_DEPRECATIONS": "1"}):
            with pytest.raises(AiProviderConfigurationError, match="deprecated"):
                enforce_model_lifecycle("google", "gemini-2.0-flash")


class TestClientCostApi:
    """Cost API surfaces on completions and embeddings clients."""

    def test_openai_completion_cost(self) -> None:
        pytest.importorskip("openai")
        from ai_api_unified.completions.ai_openai_completions import AiOpenAICompletions

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = AiOpenAICompletions(model="gpt-5.4")
        assert client.capabilities.pricing is not None
        # 1000 in + 500 out on gpt-5.4 = 0.01
        assert client.compute_completion_cost(
            input_tokens=1000, output_tokens=500
        ) == pytest.approx(0.01)

    def test_openai_embedding_cost_and_shim(self) -> None:
        pytest.importorskip("openai")
        from ai_api_unified.embeddings.ai_openai_embeddings import AiOpenAIEmbeddings

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = AiOpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
        # 1M tokens * 0.02/1M = 0.02
        assert client.compute_embedding_cost(input_tokens=1_000_000) == pytest.approx(
            0.02
        )
        # deprecated calculate_cost shim delegates to the same value
        assert client.calculate_cost(1_000_000) == pytest.approx(0.02)

    def test_info_carries_lifecycle(self) -> None:
        info = get_model_info("google", "gemini-2.0-flash")
        assert info is not None
        assert info.status is ModelLifecycleStatus.DEPRECATED
        assert info.recommended_replacement == "gemini-2.5-flash"
