# model_pricing.py

"""
Structured, sourced, per-modality model pricing and lifecycle types.

Replaces the single blended ``price_per_1k_tokens`` float with descriptors that
carry input/output/cached rates separately, a provenance stamp (effective date
and source), and a lifecycle status. These are the inputs the planned
financial-ops observability layer consumes; rates use Decimal because they feed
a cost ledger.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel

# One million, the normalization denominator for token and character rates.
TOKENS_PER_RATE_UNIT: int = 1_000_000


class ModelLifecycleStatus(str, Enum):
    """Provider lifecycle state for a model this library can target."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"  # still serves requests; scheduled for withdrawal
    RETIRED = "retired"  # provider rejects requests (e.g. hard 404)


class PricingUnit(str, Enum):
    """Canonical billing unit per modality; there is no cross-modal unit."""

    TOKEN = "token"  # per 1M tokens (completions, embeddings)
    CHARACTER = "character"  # per 1M characters (character-billed TTS)
    IMAGE = "image"  # per generated image
    SECOND = "second"  # per second of generated video
    MINUTE = "minute"  # per audio minute (transcription)


class AITokenRates(BaseModel):
    """Per-1M-token rates. output/cached are None where a modality lacks them."""

    input_per_1m: Decimal
    output_per_1m: Decimal | None = None  # None for embeddings (input only)
    cached_input_per_1m: Decimal | None = None


class AIPricingTier(BaseModel):
    """A conditional rate (context-length, quality, or resolution tier)."""

    label: str  # e.g. "context>200k", "quality:high", "1080p"
    token_rates: AITokenRates | None = None
    per_unit_usd: Decimal | None = None


class AIModelPricing(BaseModel):
    """
    Provenance-stamped pricing for one model in one canonical unit.

    Exactly one rate shape is populated to match ``unit``: token_rates for
    token/character units, or the matching per_* field for image/second/minute.
    """

    unit: PricingUnit
    currency: str = "USD"
    effective_date: date
    source: str  # official pricing URL
    confidence: Literal["high", "medium", "low"] = "high"
    token_rates: AITokenRates | None = None
    per_image_usd: Decimal | None = None
    per_second_usd: Decimal | None = None
    per_1m_characters_usd: Decimal | None = None
    per_minute_usd: Decimal | None = None
    tiers: list[AIPricingTier] | None = None
    notes: str | None = None

    def compute_token_cost(
        self,
        *,
        input_tokens: int,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
    ) -> Decimal:
        """
        Return the USD cost for measured token usage using the split rates.

        Cached input tokens are billed at the cached rate and are treated as a
        subset of input_tokens (so pass the non-cached remainder as
        input_tokens). Falls back to the input rate when no cached rate exists.

        Args:
            input_tokens: Non-cached input tokens billed at the input rate.
            output_tokens: Output tokens billed at the output rate.
            cached_input_tokens: Input tokens billed at the cached-input rate.

        Returns:
            USD cost as a Decimal.

        Raises:
            ValueError: When this pricing has no token_rates (wrong modality).
        """
        if self.token_rates is None:
            raise ValueError(
                "compute_token_cost requires token_rates; this pricing uses "
                f"unit={self.unit.value}."
            )
        rates: AITokenRates = self.token_rates
        cost: Decimal = (
            Decimal(input_tokens) * rates.input_per_1m / TOKENS_PER_RATE_UNIT
        )
        if output_tokens and rates.output_per_1m is not None:
            cost += Decimal(output_tokens) * rates.output_per_1m / TOKENS_PER_RATE_UNIT
        if cached_input_tokens:
            cached_rate: Decimal = (
                rates.cached_input_per_1m
                if rates.cached_input_per_1m is not None
                else rates.input_per_1m
            )
            cost += Decimal(cached_input_tokens) * cached_rate / TOKENS_PER_RATE_UNIT
        # Normal return with the computed USD cost.
        return cost

    def blended_per_1k_tokens(self) -> float:
        """
        Return a back-compat blended input+output rate per 1K tokens.

        Supports the deprecated ``price_per_1k_tokens`` surface. Uses the mean of
        input and output where both exist; input only for embeddings.
        """
        if self.token_rates is None:
            # Early return because non-token modalities have no per-1k-token rate.
            return 0.0
        rates: AITokenRates = self.token_rates
        if rates.output_per_1m is None:
            per_1m: Decimal = rates.input_per_1m
        else:
            per_1m = (rates.input_per_1m + rates.output_per_1m) / 2
        # Normal return with the blended rate converted to per-1K tokens.
        return float(per_1m / 1000)


class AIModelInfo(BaseModel):
    """Registry entry: lifecycle status and optional pricing for one model."""

    provider: str
    model: str
    status: ModelLifecycleStatus = ModelLifecycleStatus.ACTIVE
    sunset_date: date | None = None
    recommended_replacement: str | None = None
    pricing: AIModelPricing | None = None
