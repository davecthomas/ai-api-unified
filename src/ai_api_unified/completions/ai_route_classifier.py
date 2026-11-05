# ai_route_classifier.py
"""
A router that only classifies.

AiRouteClassifier picks a request's tier exactly as AiRouterCompletions does
(the classifier, min_confidence, default_tier, route, route_floor,
route_ceiling) and returns the tier's candidates as configured. It answers
no completions call, and it never builds or probes a candidate. So a tier
candidate may name something the library has no engine for, such as
`harness:pi-lean/claude:claude-sonnet-5`, and the caller runs the candidates
itself. The meta-harness runs each one as a coding run.

Build it with AIFactory.get_route_classifier. Because there is no default
chain to fall back to, the config must name a default_tier, and a pick that
lands on the default chain (a `route: default` option, or a model pin no
tier holds) raises.
"""

from __future__ import annotations

from typing import Any, Type

from ..ai_base import (
    AIBaseClassifier,
    AIBaseCompletions,
    AIStructuredPrompt,
)
from ..ai_provider_exceptions import AiFallbackReason, AiProviderConfigurationError
from ..routing.route_decision import AIRouteDecision
from ..routing.router_config import AIRouterConfig
from .ai_fallback_completions import AIFallbackCandidate
from .ai_router_completions import AiRouterCompletions

STR_NO_COMPLETIONS: str = (
    "A classify-only router answers no completions calls. Call classify_route "
    "and run the candidates it returns."
)


class _NoDefaultChain(AIBaseCompletions):
    """Stands in for the default chain a classify-only router does not have."""

    def __init__(self) -> None:
        super().__init__(model="")

    @property
    def list_model_names(self) -> list[str]:
        # Normal return: there is no model behind this stand-in.
        return []

    @property
    def max_context_tokens(self) -> int:
        # Normal return: there is no model behind this stand-in.
        return 0

    def strict_schema_prompt(
        self,
        prompt: str,
        response_model: Type[AIStructuredPrompt],
        max_response_tokens: int = 0,
        **kwargs: Any,
    ) -> AIStructuredPrompt:
        raise AiProviderConfigurationError(STR_NO_COMPLETIONS)

    def send_prompt(self, prompt: str, **kwargs: Any) -> str:
        raise AiProviderConfigurationError(STR_NO_COMPLETIONS)


def _refuse_build(candidate: AIFallbackCandidate) -> AIBaseCompletions:
    """The candidate builder: a classify-only router never builds a candidate."""
    raise AiProviderConfigurationError(STR_NO_COMPLETIONS)


class AiRouteClassifier:
    """
    Picks a tier and returns its candidates, without building or calling any.

    Construct through AIFactory.get_route_classifier. Direct construction
    suits tests.
    """

    def __init__(
        self,
        *,
        config: AIRouterConfig,
        classifier: AIBaseClassifier | None,
        str_classifier_unavailable_reason: str | None = None,
        config_id: str | None = None,
        config_version: str | None = None,
    ) -> None:
        """
        Args:
            config: Tiers and policy. Must name a default_tier.
            classifier: The classification engine, or None to route every
                request to the default tier.
            str_classifier_unavailable_reason: Why no classifier was built,
                logged once; None when one was.
            config_id: The routing configuration's id, carried on each decision.
            config_version: The configuration's label from models.json.

        Raises:
            ValueError: The config names no default_tier.
        """
        if config.default_tier is None:
            raise ValueError(
                "A classify-only router needs router.default_tier: it has no "
                "default chain for a request the classifier cannot place."
            )
        self._router: AiRouterCompletions = AiRouterCompletions(
            default_client=_NoDefaultChain(),
            config=config,
            classifier=classifier,
            client_builder=_refuse_build,
            fallback_on=frozenset[AiFallbackReason](),
            str_classifier_unavailable_reason=str_classifier_unavailable_reason,
            cooldown=None,
        )
        self._router.set_config_identity(config_id, config_version)

    @property
    def config(self) -> AIRouterConfig:
        """Tiers and policy."""
        # Normal return with the config.
        return self._router.config

    @property
    def classifier(self) -> AIBaseClassifier | None:
        """The classification engine, or None."""
        # Normal return with the classifier.
        return self._router.classifier

    @property
    def config_id(self) -> str | None:
        """The routing configuration's id."""
        # Normal return with the id.
        return self._router.config_id

    def classify_route(
        self,
        prompt: str | None = None,
        *,
        messages: list[dict[str, Any]] | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> AIRouteDecision:
        """
        Picks the request's tier and returns it with the tier's candidates.

        Args:
            prompt: The request text, when it is a single prompt.
            messages: The request history, when it is a conversation.
            provider_options: `route`, `route_floor` and `route_ceiling`.

        Returns:
            AIRouteDecision: The tier, how it was chosen, and its candidates.

        Raises:
            ValueError: A route option names no tier, route_floor sits above
                route_ceiling, or the pick is the default chain.
        """
        # Normal return with the checked decision.
        return self._on_a_tier(
            self._router.classify_route(
                prompt, messages=messages, provider_options=provider_options
            )
        )

    async def aclassify_route(
        self,
        prompt: str | None = None,
        *,
        messages: list[dict[str, Any]] | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> AIRouteDecision:
        """Async twin of classify_route."""
        # Normal return with the checked decision.
        return self._on_a_tier(
            await self._router.aclassify_route(
                prompt, messages=messages, provider_options=provider_options
            )
        )

    @staticmethod
    def _on_a_tier(decision: AIRouteDecision) -> AIRouteDecision:
        """Raises when the pick is the default chain, which this router lacks."""
        if decision.tier is None:
            raise ValueError(
                f"The request routed to the default chain (method "
                f"{decision.method}), and a classify-only router has none. Name "
                "a tier instead of `route: default`, or pin a model a tier holds."
            )
        # Normal return with the decision.
        return decision
