# pricing_registry.py

"""
Central model pricing and lifecycle registry.

Data only, plus a lifecycle chokepoint. Rates were compiled 2026-07-07 from
official provider pricing pages (see docs/pricing_research.md for the full
table, sources, and confidence). Kept separate from the provider classes so
prices update on the provider's schedule without touching provider code.

Lifecycle policy (see enforce_model_lifecycle):
    - RETIRED: raise AiProviderConfigurationError (the provider would reject it
      anyway; a clear early error beats an opaque downstream failure).
    - DEPRECATED: warn once per process (logging.warning + DeprecationWarning)
      with the sunset date and replacement, unless AI_STRICT_DEPRECATIONS is
      truthy, in which case escalate to the same error as RETIRED.
"""

from __future__ import annotations

import logging
import warnings
from datetime import date
from decimal import Decimal

from ..ai_provider_exceptions import AiProviderConfigurationError
from ..util.env_settings import EnvSettings
from .model_pricing import (
    AIModelInfo,
    AIModelPricing,
    AIPricingTier,
    AITokenRates,
    ModelLifecycleStatus,
    PricingUnit,
)

_LOGGER: logging.Logger = logging.getLogger(__name__)

# Provenance shared by this compilation.
_EFFECTIVE: date = date(2026, 7, 7)
# Later compilation date for models added after the initial 2026-07-07 sweep.
_EFFECTIVE_AUG: date = date(2026, 8, 3)
_SRC_OPENAI: str = "https://developers.openai.com/api/docs/pricing"
_SRC_GOOGLE: str = "https://ai.google.dev/gemini-api/docs/pricing"
_SRC_GOOGLE_DEP: str = "https://ai.google.dev/gemini-api/docs/deprecations"
_SRC_BEDROCK: str = "https://aws.amazon.com/bedrock/pricing/"
_SRC_ANTHROPIC: str = "https://platform.claude.com/docs/en/about-claude/models/overview"
_SRC_VOYAGE: str = "https://docs.voyageai.com/docs/pricing"

# Cache-write rate for providers that charge nothing to populate a cache.
# Distinct from None, which means "charged but not yet rated" and falls
# back to the base input rate. Zero must be explicit so a free-write model
# is never billed at that fallback.
_FREE_WRITES: str = "0"

# Provider labels used as the first half of registry keys.
PROVIDER_OPENAI: str = "openai"
PROVIDER_GOOGLE: str = "google"
PROVIDER_BEDROCK: str = "bedrock"
PROVIDER_ANTHROPIC: str = "anthropic"
PROVIDER_VOYAGE: str = "voyage"


def _tok(
    input_r: str,
    output_r: str | None,
    cached_r: str | None,
    source: str,
    confidence: str = "high",
    tiers: list[AIPricingTier] | None = None,
    notes: str | None = None,
    effective: date = _EFFECTIVE,
    write_5m_r: str | None = None,
    write_1h_r: str | None = None,
) -> AIModelPricing:
    """Build a token-unit AIModelPricing from string decimals (per 1M tokens)."""
    return AIModelPricing(
        unit=PricingUnit.TOKEN,
        effective_date=effective,
        source=source,
        confidence=confidence,  # type: ignore[arg-type]
        token_rates=AITokenRates(
            input_per_1m=Decimal(input_r),
            output_per_1m=Decimal(output_r) if output_r is not None else None,
            cached_input_per_1m=Decimal(cached_r) if cached_r is not None else None,
            cache_write_5m_per_1m=(
                Decimal(write_5m_r) if write_5m_r is not None else None
            ),
            cache_write_1h_per_1m=(
                Decimal(write_1h_r) if write_1h_r is not None else None
            ),
        ),
        tiers=tiers,
        notes=notes,
    )


def _info(
    provider: str,
    model: str,
    pricing: AIModelPricing | None = None,
    *,
    status: ModelLifecycleStatus = ModelLifecycleStatus.ACTIVE,
    sunset: date | None = None,
    replacement: str | None = None,
) -> tuple[tuple[str, str], AIModelInfo]:
    """Build a ((provider, model), AIModelInfo) registry pair."""
    return (provider, model), AIModelInfo(
        provider=provider,
        model=model,
        status=status,
        sunset_date=sunset,
        recommended_replacement=replacement,
        pricing=pricing,
    )


# Registry keyed by (provider, model). Non-token modalities are represented for
# lifecycle only here; their per-unit pricing lands with the finops layer.
DICT_MODEL_INFO: dict[tuple[str, str], AIModelInfo] = dict(
    [
        # ── OpenAI completions ──────────────────────────────────────────────
        _info(
            PROVIDER_OPENAI,
            "gpt-5.5",
            _tok(
                "5.00",
                "30.00",
                "0.50",
                _SRC_OPENAI,
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
            ),
        ),
        _info(
            PROVIDER_OPENAI,
            "gpt-5.4",
            _tok(
                "2.50",
                "15.00",
                "0.25",
                _SRC_OPENAI,
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
            ),
        ),
        _info(
            PROVIDER_OPENAI,
            "gpt-5.4-mini",
            _tok(
                "0.75",
                "4.50",
                "0.075",
                _SRC_OPENAI,
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
            ),
        ),
        _info(
            PROVIDER_OPENAI,
            "gpt-5.4-nano",
            _tok(
                "0.20",
                "1.25",
                "0.02",
                _SRC_OPENAI,
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
            ),
        ),
        _info(
            PROVIDER_OPENAI,
            "gpt-5.2",
            _tok(
                "1.75",
                "14.00",
                "0.175",
                _SRC_OPENAI,
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
            ),
        ),
        _info(
            PROVIDER_OPENAI,
            "gpt-5.1-codex-max",
            _tok(
                "1.25",
                "10.00",
                "0.125",
                _SRC_OPENAI,
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
            ),
        ),
        _info(
            PROVIDER_OPENAI,
            "gpt-5",
            _tok(
                "1.25",
                "10.00",
                "0.125",
                _SRC_OPENAI,
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
            ),
        ),
        _info(
            PROVIDER_OPENAI,
            "gpt-5-mini",
            _tok(
                "0.25",
                "2.00",
                "0.025",
                _SRC_OPENAI,
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
            ),
        ),
        _info(
            PROVIDER_OPENAI,
            "gpt-5-nano",
            _tok(
                "0.05",
                "0.40",
                "0.005",
                _SRC_OPENAI,
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
            ),
        ),
        _info(
            PROVIDER_OPENAI,
            "gpt-4.1",
            _tok(
                "2.00",
                "8.00",
                "0.50",
                _SRC_OPENAI,
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
            ),
        ),
        _info(
            PROVIDER_OPENAI,
            "gpt-4.1-mini",
            _tok(
                "0.40",
                "1.60",
                "0.10",
                _SRC_OPENAI,
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
            ),
        ),
        _info(
            PROVIDER_OPENAI,
            "gpt-4.1-nano",
            _tok(
                "0.10",
                "0.40",
                "0.025",
                _SRC_OPENAI,
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
            ),
        ),
        _info(
            PROVIDER_OPENAI,
            "o4-mini",
            _tok(
                "1.10",
                "4.40",
                "0.275",
                _SRC_OPENAI,
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
            ),
        ),
        _info(
            PROVIDER_OPENAI,
            "gpt-4o",
            _tok(
                "2.50",
                "10.00",
                "1.25",
                _SRC_OPENAI,
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
            ),
        ),
        _info(
            PROVIDER_OPENAI,
            "gpt-4o-mini",
            _tok(
                "0.15",
                "0.60",
                "0.075",
                _SRC_OPENAI,
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
            ),
        ),
        # ── OpenAI embeddings (input only) ──────────────────────────────────
        _info(
            PROVIDER_OPENAI,
            "text-embedding-3-small",
            _tok("0.02", None, None, _SRC_OPENAI),
        ),
        _info(
            PROVIDER_OPENAI,
            "text-embedding-3-large",
            _tok("0.13", None, None, _SRC_OPENAI),
        ),
        _info(
            PROVIDER_OPENAI,
            "text-embedding-ada-002",
            _tok("0.10", None, None, _SRC_OPENAI),
        ),
        # ── Voyage AI embeddings (input only) ───────────────────────────────
        _info(
            PROVIDER_VOYAGE,
            "voyage-3-lite",
            _tok("0.02", None, None, _SRC_VOYAGE),
        ),
        _info(
            PROVIDER_VOYAGE,
            "voyage-3",
            _tok("0.06", None, None, _SRC_VOYAGE),
        ),
        _info(
            PROVIDER_VOYAGE,
            "voyage-3-large",
            _tok("0.18", None, None, _SRC_VOYAGE),
        ),
        _info(
            PROVIDER_VOYAGE,
            "voyage-code-3",
            _tok("0.18", None, None, _SRC_VOYAGE),
        ),
        _info(
            PROVIDER_VOYAGE,
            "voyage-finance-2",
            _tok("0.12", None, None, _SRC_VOYAGE),
        ),
        _info(
            PROVIDER_VOYAGE,
            "voyage-law-2",
            _tok("0.12", None, None, _SRC_VOYAGE),
        ),
        # ── Google completions (active) ─────────────────────────────────────
        _info(
            PROVIDER_GOOGLE,
            "gemini-3.6-flash",
            _tok(
                "1.50",
                "7.50",
                "0.15",
                _SRC_GOOGLE,
                effective=_EFFECTIVE_AUG,
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
            ),
        ),
        _info(
            PROVIDER_GOOGLE,
            "gemini-3.5-flash",
            _tok(
                "1.50",
                "9.00",
                "0.15",
                _SRC_GOOGLE,
                effective=_EFFECTIVE_AUG,
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
            ),
        ),
        _info(
            PROVIDER_GOOGLE,
            "gemini-3.5-flash-lite",
            _tok(
                "0.30",
                "2.50",
                "0.03",
                _SRC_GOOGLE,
                effective=_EFFECTIVE_AUG,
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
            ),
        ),
        _info(
            PROVIDER_GOOGLE,
            "gemini-3.1-flash-lite",
            _tok(
                "0.25",
                "1.50",
                "0.025",
                _SRC_GOOGLE,
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
                effective=_EFFECTIVE_AUG,
                notes="Audio input priced higher ($0.50/1M).",
            ),
        ),
        _info(
            PROVIDER_GOOGLE,
            "gemini-3.1-pro-preview",
            _tok(
                "2.00",
                "12.00",
                "0.20",
                _SRC_GOOGLE,
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
                effective=_EFFECTIVE_AUG,
                tiers=[
                    AIPricingTier(
                        label="context>200k",
                        token_rates=AITokenRates(
                            input_per_1m=Decimal("4.00"),
                            output_per_1m=Decimal("18.00"),
                            cached_input_per_1m=Decimal("0.40"),
                            cache_write_5m_per_1m=Decimal(_FREE_WRITES),
                            cache_write_1h_per_1m=Decimal(_FREE_WRITES),
                        ),
                    )
                ],
                notes="Base rates apply to <=200K input tokens. Preview model; "
                "latest pro tier served by the Gemini API.",
            ),
        ),
        _info(
            PROVIDER_GOOGLE,
            "gemini-2.5-pro",
            _tok(
                "1.25",
                "10.00",
                "0.13",
                _SRC_GOOGLE,
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
                tiers=[
                    AIPricingTier(
                        label="context>200k",
                        token_rates=AITokenRates(
                            input_per_1m=Decimal("2.50"),
                            output_per_1m=Decimal("15.00"),
                            cached_input_per_1m=Decimal("0.25"),
                            cache_write_5m_per_1m=Decimal(_FREE_WRITES),
                            cache_write_1h_per_1m=Decimal(_FREE_WRITES),
                        ),
                    )
                ],
                notes="Base rates apply to <=200K input tokens.",
            ),
        ),
        _info(
            PROVIDER_GOOGLE,
            "gemini-2.5-flash",
            _tok(
                "0.30",
                "2.50",
                "0.03",
                _SRC_GOOGLE,
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
                notes="Audio input priced higher ($1.00/1M).",
            ),
        ),
        _info(
            PROVIDER_GOOGLE,
            "gemini-2.5-flash-lite",
            _tok(
                "0.10",
                "0.40",
                "0.01",
                _SRC_GOOGLE,
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
            ),
        ),
        # ── Google completions (deprecated: shutdown date passed) ───────────
        _info(
            PROVIDER_GOOGLE,
            "gemini-2.0-flash",
            _tok(
                "0.10",
                "0.40",
                None,
                _SRC_GOOGLE,
                confidence="medium",
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
            ),
            status=ModelLifecycleStatus.DEPRECATED,
            sunset=date(2026, 6, 1),
            replacement="gemini-2.5-flash",
        ),
        _info(
            PROVIDER_GOOGLE,
            "gemini-2.0-flash-001",
            _tok(
                "0.10",
                "0.40",
                None,
                _SRC_GOOGLE,
                confidence="medium",
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
            ),
            status=ModelLifecycleStatus.DEPRECATED,
            sunset=date(2026, 6, 1),
            replacement="gemini-2.5-flash",
        ),
        _info(
            PROVIDER_GOOGLE,
            "gemini-2.0-flash-lite",
            _tok(
                "0.075",
                "0.30",
                None,
                _SRC_GOOGLE,
                confidence="medium",
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
            ),
            status=ModelLifecycleStatus.DEPRECATED,
            sunset=date(2026, 6, 1),
            replacement="gemini-2.5-flash-lite",
        ),
        _info(
            PROVIDER_GOOGLE,
            "gemini-2.0-flash-lite-001",
            _tok(
                "0.075",
                "0.30",
                None,
                _SRC_GOOGLE,
                confidence="medium",
                write_5m_r=_FREE_WRITES,
                write_1h_r=_FREE_WRITES,
            ),
            status=ModelLifecycleStatus.DEPRECATED,
            sunset=date(2026, 6, 1),
            replacement="gemini-2.5-flash-lite",
        ),
        # ── Google completions (retired: hard 404) ──────────────────────────
        _info(
            PROVIDER_GOOGLE,
            "gemini-1.5-pro-002",
            status=ModelLifecycleStatus.RETIRED,
            replacement="gemini-2.5-pro",
        ),
        _info(
            PROVIDER_GOOGLE,
            "gemini-1.5-flash-002",
            status=ModelLifecycleStatus.RETIRED,
            replacement="gemini-2.5-flash",
        ),
        # ── Google embeddings ───────────────────────────────────────────────
        _info(
            PROVIDER_GOOGLE,
            "gemini-embedding-001",
            _tok("0.15", None, None, _SRC_GOOGLE, notes="Batch $0.075/1M."),
        ),
        _info(
            PROVIDER_GOOGLE,
            "gemini-embedding-2",
            _tok(
                "0.20",
                None,
                None,
                _SRC_GOOGLE,
                notes="Text rate; multimodal input priced separately "
                "(image $0.45/1M, audio $6.50/1M, video $12.00/1M).",
            ),
        ),
        # ── Google images (deprecated) ──────────────────────────────────────
        _info(
            PROVIDER_GOOGLE,
            "imagen-4.0-generate-001",
            AIModelPricing(
                unit=PricingUnit.IMAGE,
                effective_date=_EFFECTIVE,
                source=_SRC_GOOGLE,
                confidence="high",
                per_image_usd=Decimal("0.04"),
                tiers=[
                    AIPricingTier(label="fast", per_unit_usd=Decimal("0.02")),
                    AIPricingTier(label="standard", per_unit_usd=Decimal("0.04")),
                    AIPricingTier(label="ultra", per_unit_usd=Decimal("0.06")),
                ],
            ),
            status=ModelLifecycleStatus.DEPRECATED,
            sunset=date(2026, 8, 17),
            replacement="a current-generation Gemini image model",
        ),
        # ── Bedrock completions (the engine reports cacheWriteInputTokens for
        # every model here, so each carries either an explicit write rate or a
        # note recording why it is unrated; unrated writes bill at base input,
        # which under-reports a premium rather than dropping the cost) ───────
        _info(
            PROVIDER_BEDROCK,
            "amazon.nova-micro-v1:0",
            _tok(
                "0.035",
                "0.14",
                None,
                _SRC_BEDROCK,
                notes="Bedrock reports cacheWriteInputTokens for this model but is partner-priced, and no authoritative AWS cache-write rate has been sourced. Writes fall back to the base input rate; if AWS charges a premium, that under-reports it. Add write_5m_r once the rate is confirmed.",
            ),
        ),
        _info(
            PROVIDER_BEDROCK,
            "amazon.nova-lite-v1:0",
            _tok(
                "0.06",
                "0.24",
                None,
                _SRC_BEDROCK,
                notes="Bedrock reports cacheWriteInputTokens for this model but is partner-priced, and no authoritative AWS cache-write rate has been sourced. Writes fall back to the base input rate; if AWS charges a premium, that under-reports it. Add write_5m_r once the rate is confirmed.",
            ),
        ),
        _info(
            PROVIDER_BEDROCK,
            "amazon.nova-pro-v1:0",
            _tok(
                "0.80",
                "3.20",
                None,
                _SRC_BEDROCK,
                notes="Bedrock reports cacheWriteInputTokens for this model but is partner-priced, and no authoritative AWS cache-write rate has been sourced. Writes fall back to the base input rate; if AWS charges a premium, that under-reports it. Add write_5m_r once the rate is confirmed.",
            ),
        ),
        _info(
            PROVIDER_BEDROCK,
            "amazon.nova-premier-v1:0",
            _tok(
                "2.50",
                "12.50",
                None,
                _SRC_BEDROCK,
                notes="Bedrock reports cacheWriteInputTokens for this model but is partner-priced, and no authoritative AWS cache-write rate has been sourced. Writes fall back to the base input rate; if AWS charges a premium, that under-reports it. Add write_5m_r once the rate is confirmed.",
            ),
        ),
        _info(
            PROVIDER_BEDROCK,
            "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            _tok(
                "0.80",
                "4.00",
                None,
                _SRC_BEDROCK,
                notes="Bedrock bills cache writes (cacheWriteInputTokens) but is "
                "partner-priced, and no authoritative AWS cache-write rate has "
                "been sourced. Writes therefore fall back to the base input "
                "rate, under-reporting the premium rather than dropping it. "
                "Add write_5m_r once the AWS rate is confirmed.",
            ),
        ),
        # ── Anthropic completions (native API; cached rate is the documented
        # 0.1x prompt-cache read multiplier. Cache writes bill above base input
        # and are priced per cache lifetime: 1.25x for the 5-minute TTL and 2x
        # for the 1-hour TTL) ────────────────────────────────────────────────
        _info(
            PROVIDER_ANTHROPIC,
            "claude-fable-5",
            _tok(
                "10.00",
                "50.00",
                "1.00",
                _SRC_ANTHROPIC,
                write_5m_r="12.50",
                write_1h_r="20.00",
            ),
        ),
        _info(
            PROVIDER_ANTHROPIC,
            "claude-opus-5",
            _tok(
                "5.00",
                "25.00",
                "0.50",
                _SRC_ANTHROPIC,
                effective=_EFFECTIVE_AUG,
                write_5m_r="6.25",
                write_1h_r="10.00",
            ),
        ),
        _info(
            PROVIDER_ANTHROPIC,
            "claude-sonnet-5",
            _tok(
                "3.00",
                "15.00",
                "0.30",
                _SRC_ANTHROPIC,
                effective=_EFFECTIVE_AUG,
                write_5m_r="3.75",
                write_1h_r="6.00",
                notes="List rates; introductory pricing ($2.00 in / $10.00 out) "
                "runs through 2026-08-31.",
            ),
        ),
        _info(
            PROVIDER_ANTHROPIC,
            "claude-opus-4-8",
            _tok(
                "5.00",
                "25.00",
                "0.50",
                _SRC_ANTHROPIC,
                write_5m_r="6.25",
                write_1h_r="10.00",
            ),
        ),
        _info(
            PROVIDER_ANTHROPIC,
            "claude-opus-4-7",
            _tok(
                "5.00",
                "25.00",
                "0.50",
                _SRC_ANTHROPIC,
                write_5m_r="6.25",
                write_1h_r="10.00",
            ),
        ),
        _info(
            PROVIDER_ANTHROPIC,
            "claude-opus-4-6",
            _tok(
                "5.00",
                "25.00",
                "0.50",
                _SRC_ANTHROPIC,
                write_5m_r="6.25",
                write_1h_r="10.00",
            ),
        ),
        _info(
            PROVIDER_ANTHROPIC,
            "claude-sonnet-4-6",
            _tok(
                "3.00",
                "15.00",
                "0.30",
                _SRC_ANTHROPIC,
                write_5m_r="3.75",
                write_1h_r="6.00",
            ),
        ),
        _info(
            PROVIDER_ANTHROPIC,
            "claude-haiku-4-5",
            _tok(
                "1.00",
                "5.00",
                "0.10",
                _SRC_ANTHROPIC,
                write_5m_r="1.25",
                write_1h_r="2.00",
            ),
        ),
        # ── Anthropic completions (retired: hard 404) ───────────────────────
        # Rates are retained on entries that were priced while active so
        # historical cost enrichment can still price calls made before the
        # withdrawal date; lifecycle enforcement is independent of pricing.
        _info(
            PROVIDER_ANTHROPIC,
            "claude-opus-4-1",
            _tok(
                "15.00",
                "75.00",
                "1.50",
                _SRC_ANTHROPIC,
                write_5m_r="18.75",
                write_1h_r="30.00",
            ),
            status=ModelLifecycleStatus.RETIRED,
            sunset=date(2026, 8, 5),
            replacement="claude-opus-5",
        ),
        _info(
            PROVIDER_ANTHROPIC,
            "claude-3-7-sonnet-20250219",
            status=ModelLifecycleStatus.RETIRED,
            replacement="claude-sonnet-4-6",
        ),
        _info(
            PROVIDER_ANTHROPIC,
            "claude-3-5-haiku-20241022",
            status=ModelLifecycleStatus.RETIRED,
            replacement="claude-haiku-4-5",
        ),
        _info(
            PROVIDER_ANTHROPIC,
            "claude-3-opus-20240229",
            status=ModelLifecycleStatus.RETIRED,
            replacement="claude-opus-4-8",
        ),
        # ── Bedrock / Titan embeddings ──────────────────────────────────────
        _info(
            PROVIDER_BEDROCK,
            "amazon.titan-embed-text-v2:0",
            _tok("0.02", None, None, _SRC_BEDROCK, confidence="medium"),
        ),
        _info(
            PROVIDER_BEDROCK,
            "amazon.titan-embed-text-v1",
            _tok("0.10", None, None, _SRC_BEDROCK, confidence="medium"),
        ),
    ]
)

# Deduped per (provider, model) so a deprecation warns once per process.
_SET_WARNED_MODELS: set[tuple[str, str]] = set()

_STRICT_DEPRECATIONS_ENV: str = "AI_STRICT_DEPRECATIONS"


def get_model_info(provider: str, model: str) -> AIModelInfo | None:
    """Return the registry entry for a model, or None when not catalogued."""
    return DICT_MODEL_INFO.get((provider, model))


def get_model_pricing(provider: str, model: str) -> AIModelPricing | None:
    """Return pricing for a model, or None when not catalogued or priced."""
    info: AIModelInfo | None = get_model_info(provider, model)
    return info.pricing if info is not None else None


def _strict_deprecations_enabled() -> bool:
    """Return True when AI_STRICT_DEPRECATIONS is set to a truthy value."""
    raw: str | int | float | bool | None = EnvSettings().get_setting(
        _STRICT_DEPRECATIONS_ENV, None
    )
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _format_lifecycle_message(info: AIModelInfo) -> str:
    """Build a human-readable lifecycle message with replacement and sunset."""
    parts: list[str] = [
        f"Model '{info.model}' ({info.provider}) is {info.status.value}"
    ]
    if info.sunset_date is not None:
        # Retired models are already withdrawn, so the date reads as history;
        # deprecated ones are still served until it arrives.
        if info.status is ModelLifecycleStatus.RETIRED:
            parts.append(f"withdrawn on {info.sunset_date.isoformat()}")
        else:
            parts.append(f"scheduled for withdrawal on {info.sunset_date.isoformat()}")
    if info.recommended_replacement is not None:
        parts.append(f"use '{info.recommended_replacement}' instead")
    return "; ".join(parts) + "."


def enforce_model_lifecycle(provider: str, model: str) -> None:
    """
    Apply the lifecycle policy for a resolved (provider, model).

    Call once when a client resolves its model. Active or uncatalogued models
    pass silently.

    Args:
        provider: Provider label (openai, google, bedrock).
        model: Concrete model identifier the client will use.

    Raises:
        AiProviderConfigurationError: When the model is retired, or deprecated
            and AI_STRICT_DEPRECATIONS is enabled.
    """
    info: AIModelInfo | None = get_model_info(provider, model)
    if info is None or info.status is ModelLifecycleStatus.ACTIVE:
        # Early return: active or uncatalogued models need no notification.
        return None

    message: str = _format_lifecycle_message(info)

    if info.status is ModelLifecycleStatus.RETIRED:
        raise AiProviderConfigurationError(message)

    # Deprecated: escalate to an error only under strict mode.
    if _strict_deprecations_enabled():
        raise AiProviderConfigurationError(
            message + " (AI_STRICT_DEPRECATIONS is enabled)"
        )

    key: tuple[str, str] = (provider, model)
    if key not in _SET_WARNED_MODELS:
        _SET_WARNED_MODELS.add(key)
        _LOGGER.warning(message)
        warnings.warn(message, DeprecationWarning, stacklevel=3)
    # Normal return after emitting the one-time deprecation notice.
    return None
