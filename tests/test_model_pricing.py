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
from ai_api_unified.pricing.pricing_registry import (
    DICT_MODEL_INFO,
    PROVIDER_ANTHROPIC,
    PROVIDER_BEDROCK,
)


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

    @staticmethod
    def _pricing_with_cache_writes() -> AIModelPricing:
        return AIModelPricing(
            unit=PricingUnit.TOKEN,
            effective_date=__import__("datetime").date(2026, 7, 7),
            source="https://example.test",
            token_rates=AITokenRates(
                input_per_1m=Decimal("2.50"),
                output_per_1m=Decimal("15.00"),
                cached_input_per_1m=Decimal("0.25"),
                cache_write_5m_per_1m=Decimal("3.125"),
                cache_write_1h_per_1m=Decimal("5.00"),
            ),
        )

    def test_cache_writes_billed_per_ttl(self) -> None:
        # 1000 * 3.125/1M + 2000 * 5.00/1M = 0.003125 + 0.010
        assert self._pricing_with_cache_writes().compute_token_cost(
            input_tokens=0, cache_write_5m_tokens=1000, cache_write_1h_tokens=2000
        ) == Decimal("0.013125")

    def test_cache_writes_add_to_input_rather_than_subset(self) -> None:
        """Cache writes are billed on top of the prompt, not carved out of it.

        Unlike cache reads, the provider reports writes separately from the
        input count, so they must not be subtracted from input_tokens.
        """
        pricing = self._pricing_with_cache_writes()
        base = pricing.compute_token_cost(input_tokens=1000)
        with_write = pricing.compute_token_cost(
            input_tokens=1000, cache_write_5m_tokens=1000
        )

        assert with_write == base + Decimal("0.003125")

    def test_cache_write_falls_back_to_input_rate(self) -> None:
        """A write still costs at least base input when no premium is configured."""
        # _pricing() has no cache-write rates: 1000 * 2.50/1M
        assert self._pricing().compute_token_cost(
            input_tokens=0, cache_write_5m_tokens=1000
        ) == Decimal("0.0025")

    def test_no_cache_writes_leaves_cost_unchanged(self) -> None:
        """Callers that report no writes bill exactly as before."""
        pricing = self._pricing_with_cache_writes()

        assert pricing.compute_token_cost(
            input_tokens=1000, output_tokens=500
        ) == pricing.compute_token_cost(
            input_tokens=1000,
            output_tokens=500,
            cache_write_5m_tokens=0,
            cache_write_1h_tokens=0,
        )

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

    def test_every_anthropic_model_carries_cache_write_rates(self) -> None:
        """Anthropic bills every model's cache writes, so none may be unrated.

        An unrated write silently falls back to the base input rate and
        under-reports the premium, so a new model added without rates is a
        finops defect rather than a missing nice-to-have.
        """
        list_missing: list[str] = [
            model
            for (provider, model), info in DICT_MODEL_INFO.items()
            if provider == PROVIDER_ANTHROPIC
            and info.pricing is not None
            and info.pricing.token_rates is not None
            and (
                info.pricing.token_rates.cache_write_5m_per_1m is None
                or info.pricing.token_rates.cache_write_1h_per_1m is None
            )
        ]

        assert list_missing == []

    def test_anthropic_cache_write_rates_match_documented_multipliers(self) -> None:
        """Anthropic prices cache writes at 1.25x base input (5m) and 2x (1h).

        The rates are stored explicitly per model rather than derived, so this
        pins them to the documented relationship: a typo in one entry, or a
        provider change to the multipliers, surfaces here.
        """
        for (provider, model), info in DICT_MODEL_INFO.items():
            if provider != PROVIDER_ANTHROPIC or info.pricing is None:
                continue
            rates = info.pricing.token_rates
            if rates is None or rates.cache_write_5m_per_1m is None:
                continue
            assert rates.cache_write_5m_per_1m == rates.input_per_1m * Decimal(
                "1.25"
            ), model
            assert rates.cache_write_1h_per_1m == rates.input_per_1m * Decimal(
                "2"
            ), model

    def test_bedrock_models_are_rated_or_explicitly_noted(self) -> None:
        """Bedrock bills cache writes, so an unrated entry must say why.

        The engine reports cacheWriteInputTokens for every Bedrock completions
        model, not only the Claude ones, so this covers them all. An unrated
        entry silently bills writes at base input; requiring a note keeps that
        a recorded decision rather than an oversight the next model inherits.
        Embeddings models are excluded — they have no prompt cache to prime,
        which an absent output rate identifies.
        """
        for (provider, model), info in DICT_MODEL_INFO.items():
            if provider != PROVIDER_BEDROCK:
                continue
            if (
                info.pricing is not None
                and info.pricing.token_rates is not None
                and info.pricing.token_rates.output_per_1m is None
            ):
                continue
            if info.pricing is None or info.pricing.token_rates is None:
                continue
            if info.pricing.token_rates.cache_write_5m_per_1m is not None:
                continue
            assert info.pricing.notes and "cache write" in info.pricing.notes, (
                f"{model} bills cache writes but carries no rate and no note "
                "explaining the gap"
            )

    def test_free_write_providers_carry_an_explicit_zero_rate(self) -> None:
        """Free-by-design must be zero, not unset.

        Unset means charged-but-unrated and falls back to the base input rate,
        which would bill a free write at 100% of input.
        """
        for provider, model in (
            ("openai", "gpt-5.4"),
            ("openai", "gpt-5"),
            ("google", "gemini-3.5-flash"),
            ("google", "gemini-2.5-pro"),
        ):
            pricing = get_model_pricing(provider, model)
            assert pricing is not None
            assert pricing.token_rates.cache_write_5m_per_1m == Decimal("0"), model
            assert pricing.token_rates.cache_write_1h_per_1m == Decimal("0"), model

    def test_every_caching_capable_openai_google_model_is_free_rated(self) -> None:
        """A caching-capable model left unset would silently bill writes."""
        list_unset: list[str] = [
            f"{provider}/{model}"
            for (provider, model), info in DICT_MODEL_INFO.items()
            if provider in ("openai", "google")
            and info.pricing is not None
            and info.pricing.token_rates is not None
            and info.pricing.token_rates.cached_input_per_1m is not None
            and info.pricing.token_rates.cache_write_5m_per_1m is None
        ]

        assert list_unset == []

    def test_conditional_tier_rates_carry_cache_write_columns(self) -> None:
        """Tier rates must not drop the write columns their base entry carries.

        `compute_token_cost` does not consult tiers yet, so a gap here is
        latent rather than live — but tier-aware pricing is exactly what the
        tiers exist for, and a free-write model whose tier rates are unset
        would bill writes at the tier input rate once it lands.
        """
        list_unset: list[str] = [
            f"{provider}/{model} tier={tier.label!r}"
            for (provider, model), info in DICT_MODEL_INFO.items()
            if info.pricing is not None and info.pricing.tiers
            for tier in info.pricing.tiers
            if tier.token_rates is not None
            and tier.token_rates.cached_input_per_1m is not None
            and tier.token_rates.cache_write_5m_per_1m is None
        ]

        assert list_unset == []

    def test_free_writes_cost_nothing(self) -> None:
        """The whole point of the zero rate: a free write bills zero."""
        pricing = get_model_pricing("openai", "gpt-5.4")
        assert pricing is not None

        assert pricing.compute_token_cost(
            input_tokens=0, cache_write_5m_tokens=1_000_000
        ) == Decimal("0")

    def test_unrated_writes_still_fall_back_to_input(self) -> None:
        """Charged-but-unrated keeps the safe under-reporting fallback."""
        pricing = get_model_pricing(
            "bedrock", "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        )
        assert pricing is not None
        assert pricing.token_rates.cache_write_5m_per_1m is None

        # Falls back to the 0.80/1M input rate rather than billing zero.
        assert pricing.compute_token_cost(
            input_tokens=0, cache_write_5m_tokens=1_000_000
        ) == Decimal("0.80")

    def test_anthropic_claude_5_generation(self) -> None:
        # Added 2026-08-03 from the live models API; opus-5 matches opus-4-8
        # rates, sonnet-5 is registered at list (not introductory) rates.
        opus = get_model_pricing("anthropic", "claude-opus-5")
        assert opus is not None
        assert opus.token_rates.input_per_1m == Decimal("5.00")
        assert opus.token_rates.output_per_1m == Decimal("25.00")
        assert opus.token_rates.cached_input_per_1m == Decimal("0.50")

        sonnet = get_model_pricing("anthropic", "claude-sonnet-5")
        assert sonnet is not None
        assert sonnet.token_rates.input_per_1m == Decimal("3.00")
        assert sonnet.token_rates.output_per_1m == Decimal("15.00")

    def test_openai_gpt_5_5_present(self) -> None:
        pricing = get_model_pricing("openai", "gpt-5.5")
        assert pricing is not None
        assert pricing.token_rates.input_per_1m == Decimal("5.00")
        assert pricing.token_rates.output_per_1m == Decimal("30.00")

    def test_gemini_3_generation(self) -> None:
        flash = get_model_pricing("google", "gemini-3.5-flash")
        assert flash is not None
        assert flash.token_rates.input_per_1m == Decimal("1.50")
        assert flash.token_rates.output_per_1m == Decimal("9.00")

        # The pro preview carries a >200K pricing tier like gemini-2.5-pro.
        pro = get_model_pricing("google", "gemini-3.1-pro-preview")
        assert pro is not None
        assert pro.token_rates.input_per_1m == Decimal("2.00")
        assert pro.tiers is not None and len(pro.tiers) == 1
        assert pro.tiers[0].token_rates is not None
        assert pro.tiers[0].token_rates.input_per_1m == Decimal("4.00")


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

    def test_sunset_date_phrasing_matches_status(self) -> None:
        # A sunset date on a retired entry describes a past withdrawal; on a
        # deprecated entry it is still a schedule.
        with pytest.raises(AiProviderConfigurationError) as excinfo:
            enforce_model_lifecycle("anthropic", "claude-opus-4-1")
        retired_message = str(excinfo.value)
        assert "withdrawn on 2026-08-05" in retired_message
        assert "scheduled" not in retired_message

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            enforce_model_lifecycle("google", "gemini-2.0-flash")
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep) == 1
        assert "scheduled for withdrawal on 2026-06-01" in str(dep[0].message)

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
