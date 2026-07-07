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
_SRC_OPENAI: str = "https://developers.openai.com/api/docs/pricing"
_SRC_GOOGLE: str = "https://ai.google.dev/gemini-api/docs/pricing"
_SRC_GOOGLE_DEP: str = "https://ai.google.dev/gemini-api/docs/deprecations"
_SRC_BEDROCK: str = "https://aws.amazon.com/bedrock/pricing/"

# Provider labels used as the first half of registry keys.
PROVIDER_OPENAI: str = "openai"
PROVIDER_GOOGLE: str = "google"
PROVIDER_BEDROCK: str = "bedrock"


def _tok(
    input_r: str,
    output_r: str | None,
    cached_r: str | None,
    source: str,
    confidence: str = "high",
    tiers: list[AIPricingTier] | None = None,
    notes: str | None = None,
) -> AIModelPricing:
    """Build a token-unit AIModelPricing from string decimals (per 1M tokens)."""
    return AIModelPricing(
        unit=PricingUnit.TOKEN,
        effective_date=_EFFECTIVE,
        source=source,
        confidence=confidence,  # type: ignore[arg-type]
        token_rates=AITokenRates(
            input_per_1m=Decimal(input_r),
            output_per_1m=Decimal(output_r) if output_r is not None else None,
            cached_input_per_1m=Decimal(cached_r) if cached_r is not None else None,
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
        _info(PROVIDER_OPENAI, "gpt-5.5", _tok("5.00", "30.00", "0.50", _SRC_OPENAI)),
        _info(PROVIDER_OPENAI, "gpt-5.4", _tok("2.50", "15.00", "0.25", _SRC_OPENAI)),
        _info(
            PROVIDER_OPENAI, "gpt-5.4-mini", _tok("0.75", "4.50", "0.075", _SRC_OPENAI)
        ),
        _info(
            PROVIDER_OPENAI, "gpt-5.4-nano", _tok("0.20", "1.25", "0.02", _SRC_OPENAI)
        ),
        _info(PROVIDER_OPENAI, "gpt-5.2", _tok("1.75", "14.00", "0.175", _SRC_OPENAI)),
        _info(
            PROVIDER_OPENAI,
            "gpt-5.1-codex-max",
            _tok("1.25", "10.00", "0.125", _SRC_OPENAI),
        ),
        _info(PROVIDER_OPENAI, "gpt-5", _tok("1.25", "10.00", "0.125", _SRC_OPENAI)),
        _info(
            PROVIDER_OPENAI, "gpt-5-mini", _tok("0.25", "2.00", "0.025", _SRC_OPENAI)
        ),
        _info(
            PROVIDER_OPENAI, "gpt-5-nano", _tok("0.05", "0.40", "0.005", _SRC_OPENAI)
        ),
        _info(PROVIDER_OPENAI, "gpt-4.1", _tok("2.00", "8.00", "0.50", _SRC_OPENAI)),
        _info(
            PROVIDER_OPENAI, "gpt-4.1-mini", _tok("0.40", "1.60", "0.10", _SRC_OPENAI)
        ),
        _info(
            PROVIDER_OPENAI, "gpt-4.1-nano", _tok("0.10", "0.40", "0.025", _SRC_OPENAI)
        ),
        _info(PROVIDER_OPENAI, "o4-mini", _tok("1.10", "4.40", "0.275", _SRC_OPENAI)),
        _info(PROVIDER_OPENAI, "gpt-4o", _tok("2.50", "10.00", "1.25", _SRC_OPENAI)),
        _info(
            PROVIDER_OPENAI, "gpt-4o-mini", _tok("0.15", "0.60", "0.075", _SRC_OPENAI)
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
        # ── Google completions (active) ─────────────────────────────────────
        _info(
            PROVIDER_GOOGLE,
            "gemini-2.5-pro",
            _tok(
                "1.25",
                "10.00",
                "0.13",
                _SRC_GOOGLE,
                tiers=[
                    AIPricingTier(
                        label="context>200k",
                        token_rates=AITokenRates(
                            input_per_1m=Decimal("2.50"),
                            output_per_1m=Decimal("15.00"),
                            cached_input_per_1m=Decimal("0.25"),
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
                notes="Audio input priced higher ($1.00/1M).",
            ),
        ),
        _info(
            PROVIDER_GOOGLE,
            "gemini-2.5-flash-lite",
            _tok("0.10", "0.40", "0.01", _SRC_GOOGLE),
        ),
        # ── Google completions (deprecated: shutdown date passed) ───────────
        _info(
            PROVIDER_GOOGLE,
            "gemini-2.0-flash",
            _tok("0.10", "0.40", None, _SRC_GOOGLE, confidence="medium"),
            status=ModelLifecycleStatus.DEPRECATED,
            sunset=date(2026, 6, 1),
            replacement="gemini-2.5-flash",
        ),
        _info(
            PROVIDER_GOOGLE,
            "gemini-2.0-flash-001",
            _tok("0.10", "0.40", None, _SRC_GOOGLE, confidence="medium"),
            status=ModelLifecycleStatus.DEPRECATED,
            sunset=date(2026, 6, 1),
            replacement="gemini-2.5-flash",
        ),
        _info(
            PROVIDER_GOOGLE,
            "gemini-2.0-flash-lite",
            _tok("0.075", "0.30", None, _SRC_GOOGLE, confidence="medium"),
            status=ModelLifecycleStatus.DEPRECATED,
            sunset=date(2026, 6, 1),
            replacement="gemini-2.5-flash-lite",
        ),
        _info(
            PROVIDER_GOOGLE,
            "gemini-2.0-flash-lite-001",
            _tok("0.075", "0.30", None, _SRC_GOOGLE, confidence="medium"),
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
        # ── Bedrock completions ─────────────────────────────────────────────
        _info(
            PROVIDER_BEDROCK,
            "amazon.nova-micro-v1:0",
            _tok("0.035", "0.14", None, _SRC_BEDROCK),
        ),
        _info(
            PROVIDER_BEDROCK,
            "amazon.nova-lite-v1:0",
            _tok("0.06", "0.24", None, _SRC_BEDROCK),
        ),
        _info(
            PROVIDER_BEDROCK,
            "amazon.nova-pro-v1:0",
            _tok("0.80", "3.20", None, _SRC_BEDROCK),
        ),
        _info(
            PROVIDER_BEDROCK,
            "amazon.nova-premier-v1:0",
            _tok("2.50", "12.50", None, _SRC_BEDROCK),
        ),
        _info(
            PROVIDER_BEDROCK,
            "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            _tok("0.80", "4.00", None, _SRC_BEDROCK),
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
