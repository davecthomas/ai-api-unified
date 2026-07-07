"""Model pricing and lifecycle: structured rates plus deprecation policy."""

from __future__ import annotations

from .model_pricing import (
    AIModelInfo,
    AIModelPricing,
    AIPricingTier,
    AITokenRates,
    ModelLifecycleStatus,
    PricingUnit,
)
from .pricing_registry import (
    DICT_MODEL_INFO,
    enforce_model_lifecycle,
    get_model_info,
    get_model_pricing,
)

__all__: list[str] = [
    "AIModelInfo",
    "AIModelPricing",
    "AIPricingTier",
    "AITokenRates",
    "ModelLifecycleStatus",
    "PricingUnit",
    "DICT_MODEL_INFO",
    "enforce_model_lifecycle",
    "get_model_info",
    "get_model_pricing",
]
