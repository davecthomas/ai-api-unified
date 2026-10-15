"""
Model routing: classify a prompt's complexity and send it to a tier of models.

Import the public types from the package root.
"""

from .router_config import (
    DICT_DEFAULT_TIER_DESCRIPTIONS,
    AIRouterClassifierConfig,
    AIRouterConfig,
    AIRouterTier,
    classification_text_for,
    load_router_config,
)

__all__ = [
    "DICT_DEFAULT_TIER_DESCRIPTIONS",
    "AIRouterClassifierConfig",
    "AIRouterConfig",
    "AIRouterTier",
    "classification_text_for",
    "load_router_config",
]
