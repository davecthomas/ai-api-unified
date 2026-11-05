# route_decision.py
"""
Where the router would send a request, as data: the tier, how it was chosen,
and that tier's candidates in order. Produced by
AiRouterCompletions.classify_route and AiRouteClassifier, which build and
call none of the candidates.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from ..completions.ai_fallback_completions import AIFallbackCandidate
from ..thinking import AIThinkingLevel


class AIRouteDecision(BaseModel):
    """
    The router's pick for one request.

    Attributes:
        tier: The tier the request routes to; None for the default chain.
        method: How the tier was chosen: classifier, low_confidence, floor,
            ceiling, override, pinned, model_pin, or default.
        confidence: The classifier's confidence, when it ran and reported one.
        thinking: The tier's thinking level, else the router's; None when
            neither is set.
        candidates: The tier's candidates in order, exactly as configured.
            On the default chain, the default chain's candidates.
        classifier_usd: What the classifier call cost.
        config_id: The routing configuration that made the pick.
    """

    model_config = ConfigDict(frozen=True)

    tier: str | None
    method: str
    confidence: float | None = None
    thinking: AIThinkingLevel | None = None
    candidates: list[AIFallbackCandidate]
    classifier_usd: Decimal = Decimal("0")
    config_id: str | None = None
