# ai_router_completions.py
"""
Completions client that routes each request to a tier of models by the
prompt's classified complexity.

AiRouterCompletions wraps the default client the factory builds today (the
configured completions LLM and its fallback chain) plus an ordered list of
tiers, each an engine:model chain of its own. On a request it classifies the
latest user text with a classification engine (see AIBaseClassifier), picks
the tier the classifier names, and runs the call on that tier's chain. The
chain is an AiFallbackCompletions, so provider failures, engine families,
and same-account skips behave exactly as they do without a router.

The router adds one failure behavior of its own: when a tier's chain is
exhausted on a trigger reason, it escalates to the next tier up (or raises,
when configured). The classifier can never fail a user's call: no
classifier, a classifier error, or a refusal routes to the default tier or
the default chain, and the reason is logged.

Every call logs a `routed` event before it runs, and pushes route tags
(route_id, route_tier, route_method) onto the observability context so the
engine's own input, output, and cost events carry them.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from decimal import Decimal
from typing import Any, ClassVar, Type, TypeVar

from ..ai_base import (
    AIBaseClassifier,
    AIBaseCompletions,
    AIBatchJob,
    AIBatchRequestItem,
    AIBatchResultItem,
    AIClassificationChoice,
    AIClassificationResult,
    AICompletionsCapabilitiesBase,
    AICompletionsPromptParamsBase,
    AIPromptCacheHint,
    AIProviderOrgInfoBase,
    AIProviderOrgInfoCapability,
    AIStructuredOutputResult,
    AIStructuredPrompt,
    AITool,
    AITurnResult,
)
from ..ai_provider_exceptions import AiFallbackReason, AiProviderRequestError
from ..middleware.observability_runtime import (
    ObservabilityContextModel,
    get_observability_context,
    reset_observability_context,
    set_observability_context,
)
from ..middleware.request_cost import RequestCostScope, current_request_cost_scope
from ..routing.router_config import (
    AIRouterConfig,
    AIRouterTier,
    classification_text_for,
)
from .ai_fallback_completions import (
    DICT_ENGINE_FAMILY_BY_TOKEN,
    AIFallbackCandidate,
    AiFallbackCompletions,
    history_family_of,
)

_LOGGER: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")

# Values of the `ai_router_event` field on this module's log records.
ROUTER_EVENT_ROUTED: str = "routed"
ROUTER_EVENT_CLASSIFIER_UNAVAILABLE: str = "classifier_unavailable"
ROUTER_EVENT_CLASSIFIER_FAILED: str = "classifier_failed"
ROUTER_EVENT_TIER_EXHAUSTED: str = "tier_exhausted"
ROUTER_EVENT_TIER_UNBUILDABLE: str = "tier_unbuildable"

# Values of the `method` field on a routed event.
ROUTE_METHOD_CLASSIFIER: str = "classifier"
ROUTE_METHOD_DEFAULT: str = "default"
ROUTE_METHOD_PINNED: str = "pinned"
ROUTE_METHOD_OVERRIDE: str = "override"

# Observability tags pushed around every routed call.
TAG_ROUTE_ID: str = "route_id"
TAG_ROUTE_TIER: str = "route_tier"
TAG_ROUTE_METHOD: str = "route_method"
TAG_ROUTER_ROLE: str = "router_role"
TAG_VALUE_CLASSIFIER_ROLE: str = "classifier"

# Tier label used in tags and logs for the default chain.
STR_DEFAULT_CHAIN_LABEL: str = "default"

STR_ROUTING_INSTRUCTIONS: str = (
    "Classify how demanding this request is for a language model, so it can "
    "be sent to a model of matching capability. Judge the request itself, "
    "not its length. Pick the one tier whose description fits best."
)


class AiRouterCompletions(AIBaseCompletions):
    """
    Completions client that picks a tier of models per request.

    Construct through AIFactory.get_ai_completions_client(router=...) or the
    COMPLETIONS_ROUTER_CONFIG setting. Direct construction suits tests.
    """

    # Reserved provider option: a tier name, or "default", pins one call.
    PROVIDER_OPTION_ROUTE: ClassVar[str] = "route"

    def __init__(
        self,
        *,
        default_client: AIBaseCompletions,
        config: AIRouterConfig,
        classifier: AIBaseClassifier | None,
        client_builder: Callable[[AIFallbackCandidate], AIBaseCompletions],
        fallback_on: frozenset[AiFallbackReason],
        str_classifier_unavailable_reason: str | None = None,
    ) -> None:
        """
        Args:
            default_client: The configured completions LLM, with its own
                fallback chain when one is configured.
            config: Tiers and policy.
            classifier: The classification engine, or None to route every
                call to the default tier or chain.
            client_builder: Builds one engine client from a candidate.
            fallback_on: Trigger reasons for tiers that set none.
            str_classifier_unavailable_reason: Why no classifier was built,
                logged once; None when one was.
        """
        super().__init__(model=default_client.model_name)
        self._default_client: AIBaseCompletions = default_client
        self._config: AIRouterConfig = config
        self._classifier: AIBaseClassifier | None = classifier
        self._client_builder: Callable[[AIFallbackCandidate], AIBaseCompletions] = (
            client_builder
        )
        self._fallback_on: frozenset[AiFallbackReason] = frozenset(fallback_on)
        self._tier_clients: dict[int, AiFallbackCompletions | None] = {}
        self._last_tier_index: int | None = None
        self._last_route_client: AIBaseCompletions = default_client
        self._last_decision: AIClassificationResult | None = None
        self._int_default_tier_index: int | None = (
            config.tier_index(config.default_tier)
            if config.default_tier is not None
            else None
        )
        if classifier is None:
            _LOGGER.warning(
                "ROUTER: no classifier (%s); every call routes to %s.",
                str_classifier_unavailable_reason or "not configured",
                self._tier_label(self._int_default_tier_index),
                extra={
                    "ai_router_event": ROUTER_EVENT_CLASSIFIER_UNAVAILABLE,
                    "reason": str_classifier_unavailable_reason or "not_configured",
                    "routed_to": self._tier_label(self._int_default_tier_index),
                },
            )

    # ── Introspection ───────────────────────────────────────────────────────

    @property
    def default_client(self) -> AIBaseCompletions:
        """The configured completions LLM (and its chain)."""
        # Normal return with the default client.
        return self._default_client

    @property
    def tiers(self) -> list[AIRouterTier]:
        """The tiers, ascending capability."""
        # Normal return with a copy.
        return list(self._config.tiers)

    @property
    def config(self) -> AIRouterConfig:
        """The router configuration."""
        # Normal return with the config.
        return self._config

    @property
    def classifier(self) -> AIBaseClassifier | None:
        """The classification engine, or None."""
        # Normal return with the classifier.
        return self._classifier

    @property
    def last_decision(self) -> AIClassificationResult | None:
        """The classifier's result on the most recent classified call."""
        # Normal return with the decision.
        return self._last_decision

    @property
    def last_tier(self) -> str | None:
        """Name of the tier that served the most recent call; None for the default chain."""
        # Normal return with the tier name.
        return (
            None
            if self._last_tier_index is None
            else self._config.tiers[self._last_tier_index].name
        )

    @property
    def last_route_client(self) -> AIBaseCompletions:
        """The client that served the most recent call."""
        # Normal return with the client.
        return self._last_route_client

    def for_tier(self, str_name: str) -> AIBaseCompletions:
        """
        The chain client for one tier, for calls that must bypass routing.

        Args:
            str_name: Tier name, or "default" for the default chain.

        Returns:
            That tier's AiFallbackCompletions, or the default client.

        Raises:
            ValueError: No such tier.
        """
        str_resolved: str | None = self._config.resolve_tier_name(
            str_name.strip().lower()
        )
        if str_resolved is None or str_resolved == STR_DEFAULT_CHAIN_LABEL:
            # Early return with the default chain (also a pruned tier with
            # nothing above it).
            return self._default_client
        int_index: int | None = self._config.tier_index(str_resolved)
        if int_index is None:
            raise ValueError(
                f"Unknown router tier {str_name!r}; expected one of "
                f"{[tier.name for tier in self._config.tiers]} or 'default'."
            )
        client: AiFallbackCompletions | None = self._tier_client(int_index)
        if client is None:
            raise AiProviderRequestError(
                f"Router tier {str_name!r} could not be built.",
                status_code=None,
                fallback_reason=None,
            )
        # Normal return with the tier's chain.
        return client

    # ── Pass-through to the default client ──────────────────────────────────

    @property
    def capabilities(self) -> AICompletionsCapabilitiesBase:
        # Normal return with the default client's capabilities.
        return self._default_client.capabilities

    @property
    def list_model_names(self) -> list[str]:
        # Normal return with the default client's catalog.
        return self._default_client.list_model_names

    @property
    def max_context_tokens(self) -> int:
        # Normal return with the default client's limit.
        return self._default_client.max_context_tokens

    def get_org_info(self) -> AIProviderOrgInfoBase:
        # Normal return with the default client's org info.
        return self._default_client.get_org_info()

    def get_org_info_capability(self) -> AIProviderOrgInfoCapability:
        # Normal return with the default client's capability.
        return self._default_client.get_org_info_capability()

    @property
    def price_per_1k_tokens(self) -> float:
        # Normal return priced at the last route.
        return self._last_route_client.price_per_1k_tokens

    def compute_completion_cost(
        self,
        *,
        input_tokens: int,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        cache_write_5m_tokens: int = 0,
        cache_write_1h_tokens: int = 0,
    ) -> float:
        # Normal return priced at the last route.
        return self._last_route_client.compute_completion_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_write_5m_tokens=cache_write_5m_tokens,
            cache_write_1h_tokens=cache_write_1h_tokens,
        )

    def count_tokens(
        self,
        prompt: str,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> int:
        # Normal return with the default client's count.
        return self._default_client.count_tokens(prompt, other_params=other_params)

    def strict_schema_prompt(
        self,
        prompt: str,
        response_model: Type[AIStructuredPrompt],
        max_response_tokens: int = AIBaseCompletions.STRUCTURED_DEFAULT_MAX_RESPONSE_TOKENS,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> AIStructuredPrompt:
        """Legacy path; not routed."""
        # Normal return from the default client.
        return self._default_client.strict_schema_prompt(
            prompt, response_model, max_response_tokens, other_params=other_params
        )

    def submit_batch(self, requests: list[AIBatchRequestItem]) -> AIBatchJob:
        # Normal return from the default client.
        return self._default_client.submit_batch(requests)

    def get_batch(self, batch: str | AIBatchJob) -> AIBatchJob:
        # Normal return from the default client.
        return self._default_client.get_batch(batch)

    def get_batch_results(self, batch: str | AIBatchJob) -> list[AIBatchResultItem]:
        # Normal return from the default client.
        return self._default_client.get_batch_results(batch)

    def cancel_batch(self, batch: str | AIBatchJob) -> AIBatchJob:
        # Normal return from the default client.
        return self._default_client.cancel_batch(batch)

    def run_batch(
        self,
        requests: list[AIBatchRequestItem],
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float | None = None,
    ) -> list[AIBatchResultItem]:
        # Normal return from the default client.
        return self._default_client.run_batch(
            requests,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    # ── Tier clients ────────────────────────────────────────────────────────

    def _tier_client(self, int_index: int) -> AiFallbackCompletions | None:
        """
        Builds a tier's chain the first time it is needed.

        Args:
            int_index: Tier position.

        Returns:
            The chain, or None when its primary could not be built (logged
            once; the tier stays out for the rest of the process).
        """
        if int_index in self._tier_clients:
            # Early return with the cached chain or its tombstone.
            return self._tier_clients[int_index]
        tier: AIRouterTier = self._config.tiers[int_index]
        try:
            primary: AIBaseCompletions = self._client_builder(tier.candidates[0])
        except Exception as exception:
            _LOGGER.error(
                "ROUTER tier %s could not be built and is out of the router "
                "for this process: %s",
                tier.label,
                exception,
                extra={
                    "ai_router_event": ROUTER_EVENT_TIER_UNBUILDABLE,
                    "tier": tier.name,
                    "error": type(exception).__name__,
                },
            )
            self._tier_clients[int_index] = None
            # Early return: unbuildable.
            return None
        client: AiFallbackCompletions = AiFallbackCompletions(
            primary=primary,
            primary_candidate=tier.candidates[0],
            fallback_candidates=tier.candidates[1:],
            client_builder=self._client_builder,
            fallback_on=tier.fallback_on or self._fallback_on,
        )
        self._tier_clients[int_index] = client
        # Normal return with the new chain.
        return client

    def _client_at(self, int_index: int | None) -> AIBaseCompletions | None:
        """The default client for None, else the tier's chain."""
        if int_index is None:
            # Early return with the default chain.
            return self._default_client
        # Normal return with the tier's chain, or None if unbuildable.
        return self._tier_client(int_index)

    def _tier_label(self, int_index: int | None) -> str:
        # Normal return with the tier name or the default label.
        return (
            STR_DEFAULT_CHAIN_LABEL
            if int_index is None
            else self._config.tiers[int_index].name
        )

    @staticmethod
    def _candidate_family(candidate: AIFallbackCandidate) -> str | None:
        # Normal return with the token's family, or None for an unknown token.
        return DICT_ENGINE_FAMILY_BY_TOKEN.get(candidate.engine)

    # ── Selection ───────────────────────────────────────────────────────────

    def _strip_route_option(
        self, provider_options: dict[str, Any] | None
    ) -> tuple[dict[str, Any] | None, str | None]:
        """
        Removes the reserved route key before options reach a chain.

        Args:
            provider_options: The call's provider_options.

        Returns:
            (remaining options or None, the route override or None).
        """
        if not provider_options:
            # Early return: nothing to strip.
            return None, None
        dict_copy: dict[str, Any] = dict(provider_options)
        raw_route: Any = dict_copy.pop(self.PROVIDER_OPTION_ROUTE, None)
        str_route: str | None = (
            str(raw_route).strip().lower() if raw_route not in (None, "") else None
        )
        # Normal return with the stripped options and the override.
        return dict_copy or None, str_route

    def _select(
        self,
        *,
        str_text: str,
        messages: list[dict[str, Any]] | None,
        str_route: str | None,
        str_operation: str,
        str_route_id: str,
    ) -> tuple[int | None, str, AIClassificationResult | None]:
        """
        Decides which tier serves one call.

        Args:
            str_text: The text to classify.
            messages: The call's history, if any.
            str_route: A per-call override, if any.
            str_operation: Method name, for logs.
            str_route_id: Correlation id for this call's events.

        Returns:
            (tier index or None for the default chain, route method, the
            classifier's result if it ran).

        Raises:
            ValueError: The override names no tier.
        """
        if str_route is not None:
            str_resolved: str | None = self._config.resolve_tier_name(str_route)
            if str_resolved is None or str_resolved == STR_DEFAULT_CHAIN_LABEL:
                # Early return: pinned to the default chain (or a pruned tier
                # with nothing above it).
                return None, ROUTE_METHOD_OVERRIDE, None
            int_override: int | None = self._config.tier_index(str_resolved)
            if int_override is None:
                raise ValueError(
                    f"Unknown router tier {str_route!r} in provider_options "
                    f"'{self.PROVIDER_OPTION_ROUTE}'; expected one of "
                    f"{[tier.name for tier in self._config.tiers]} or 'default'."
                )
            # Early return: pinned to the named tier.
            return int_override, ROUTE_METHOD_OVERRIDE, None
        str_family: str | None = history_family_of(messages or [])
        if str_family is not None:
            # Early return: engine-shaped history stays in its family.
            return self._pinned_tier(str_family), ROUTE_METHOD_PINNED, None
        if self._classifier is None:
            # Early return: no classifier, so the default route.
            return self._int_default_tier_index, ROUTE_METHOD_DEFAULT, None
        if not str_text.strip():
            # Early return: nothing to classify.
            return self._int_default_tier_index, ROUTE_METHOD_DEFAULT, None
        float_started: float = time.perf_counter()
        try:
            decision: AIClassificationResult = self._classify(str_text, str_route_id)
        except Exception as exception:
            self._log_classifier_failed(
                str_operation,
                str_route_id,
                error=type(exception).__name__,
                fallback_reason=getattr(exception, "fallback_reason", None),
                float_started=float_started,
            )
            # Early return: the classifier failed, so the default route.
            return self._int_default_tier_index, ROUTE_METHOD_DEFAULT, None
        # Normal return with the classified tier, or the default route.
        return self._tier_for_decision(
            decision, str_operation, str_route_id, float_started
        )

    def _tier_for_decision(
        self,
        decision: AIClassificationResult,
        str_operation: str,
        str_route_id: str,
        float_started: float,
    ) -> tuple[int | None, str, AIClassificationResult | None]:
        """Maps a classifier answer to a tier, honoring pruned tiers."""
        self._last_decision = decision
        if decision.refused or not decision.choice:
            self._log_classifier_failed(
                str_operation,
                str_route_id,
                error="refusal" if decision.refused else "unknown_tier",
                fallback_reason=None,
                float_started=float_started,
                decision=decision,
            )
            # Early return: no usable answer, so the default route.
            return self._int_default_tier_index, ROUTE_METHOD_DEFAULT, decision
        if decision.choice in self._config.pruned_tiers:
            str_next: str | None = self._config.pruned_tiers[decision.choice]
            # Early return: a pruned tier goes to the next higher survivor.
            return (
                self._config.tier_index(str_next) if str_next else None,
                ROUTE_METHOD_CLASSIFIER,
                decision,
            )
        int_index: int | None = self._config.tier_index(decision.choice)
        if int_index is None:
            self._log_classifier_failed(
                str_operation,
                str_route_id,
                error="unknown_tier",
                fallback_reason=None,
                float_started=float_started,
                decision=decision,
            )
            # Early return: the classifier named no tier we know.
            return self._int_default_tier_index, ROUTE_METHOD_DEFAULT, decision
        # Normal return with the classified tier.
        return int_index, ROUTE_METHOD_CLASSIFIER, decision

    def _pinned_tier(self, str_family: str) -> int | None:
        """
        The tier that serves an engine-shaped history: the tier that served
        the last call when its family matches, else the first tier with a
        candidate of that family, else the default chain.
        """
        list_eligible: list[int] = [
            int_index
            for int_index, tier in enumerate(self._config.tiers)
            if any(
                self._candidate_family(candidate) == str_family
                for candidate in tier.candidates
            )
        ]
        if self._last_tier_index in list_eligible:
            # Early return: stay where the conversation was.
            return self._last_tier_index
        # Normal return with the first eligible tier, or the default chain.
        return list_eligible[0] if list_eligible else None

    def _classify(self, str_text: str, str_route_id: str) -> AIClassificationResult:
        """Runs the classifier with the router role and route id tagged on its call."""
        assert self._classifier is not None
        list_choices: list[AIClassificationChoice] = [
            AIClassificationChoice(value=tier.name, description=tier.description)
            for tier in self._config.tiers
        ]
        str_instructions: str = STR_ROUTING_INSTRUCTIONS
        if self._config.classifier and self._config.classifier.instructions:
            str_instructions = f"{self._config.classifier.instructions.strip()}\n{STR_ROUTING_INSTRUCTIONS}"
        float_timeout: float | None = (
            self._config.classifier.timeout_seconds if self._config.classifier else None
        )
        token = self._push_tags(
            {TAG_ROUTER_ROLE: TAG_VALUE_CLASSIFIER_ROLE, TAG_ROUTE_ID: str_route_id}
        )
        try:
            # Normal return with the classifier's decision.
            return self._classifier.classify_choice(
                str_text,
                list_choices,
                instructions=str_instructions,
                request_timeout_seconds=float_timeout,
            )
        finally:
            reset_observability_context(token)

    async def _aclassify(
        self, str_text: str, str_route_id: str
    ) -> AIClassificationResult:
        """Async twin of _classify; falls back to the sync path without async support."""
        assert self._classifier is not None
        if not self._classifier.capabilities.supports_async:
            # Early return via the sync path.
            return self._classify(str_text, str_route_id)
        list_choices: list[AIClassificationChoice] = [
            AIClassificationChoice(value=tier.name, description=tier.description)
            for tier in self._config.tiers
        ]
        str_instructions: str = STR_ROUTING_INSTRUCTIONS
        if self._config.classifier and self._config.classifier.instructions:
            str_instructions = f"{self._config.classifier.instructions.strip()}\n{STR_ROUTING_INSTRUCTIONS}"
        float_timeout: float | None = (
            self._config.classifier.timeout_seconds if self._config.classifier else None
        )
        token = self._push_tags(
            {TAG_ROUTER_ROLE: TAG_VALUE_CLASSIFIER_ROLE, TAG_ROUTE_ID: str_route_id}
        )
        try:
            # Normal return with the classifier's decision.
            return await self._classifier.aclassify_choice(
                str_text,
                list_choices,
                instructions=str_instructions,
                request_timeout_seconds=float_timeout,
            )
        finally:
            reset_observability_context(token)

    async def _aselect(
        self,
        *,
        str_text: str,
        messages: list[dict[str, Any]] | None,
        str_route: str | None,
        str_operation: str,
        str_route_id: str,
    ) -> tuple[int | None, str, AIClassificationResult | None]:
        """Async twin of _select."""
        if str_route is not None or history_family_of(messages or []) is not None:
            # Early return: no classifier call is involved.
            return self._select(
                str_text=str_text,
                messages=messages,
                str_route=str_route,
                str_operation=str_operation,
                str_route_id=str_route_id,
            )
        if self._classifier is None or not str_text.strip():
            # Early return: the default route.
            return self._int_default_tier_index, ROUTE_METHOD_DEFAULT, None
        float_started: float = time.perf_counter()
        try:
            decision: AIClassificationResult = await self._aclassify(
                str_text, str_route_id
            )
        except Exception as exception:
            self._log_classifier_failed(
                str_operation,
                str_route_id,
                error=type(exception).__name__,
                fallback_reason=getattr(exception, "fallback_reason", None),
                float_started=float_started,
            )
            # Early return: the classifier failed, so the default route.
            return self._int_default_tier_index, ROUTE_METHOD_DEFAULT, None
        # Normal return with the classified tier, or the default route.
        return self._tier_for_decision(
            decision, str_operation, str_route_id, float_started
        )

    # ── Dispatch ────────────────────────────────────────────────────────────

    def _next_tier_up(self, int_index: int, str_family: str | None) -> int | None:
        """
        The next tier above one that may serve this call, or None.

        A family-pinned history can only replay on a tier with a candidate
        of that family, so tiers of other families are skipped.
        """
        # Loop upward for the first tier that can take the call.
        for int_next in range(int_index + 1, len(self._config.tiers)):
            if str_family is None or any(
                self._candidate_family(candidate) == str_family
                for candidate in self._config.tiers[int_next].candidates
            ):
                # Early return with the eligible tier.
                return int_next
        # Normal return: nothing above may serve the call.
        return None

    def _should_escalate(
        self,
        int_index: int | None,
        exception: BaseException,
        str_family: str | None,
    ) -> bool:
        """
        True when a failed tier should hand the call to the next tier up.

        Only a trigger failure on a tier that has an eligible tier above it
        escalates; the default chain, "raise" mode, a non-trigger error, and
        the top tier all re-raise.
        """
        if int_index is None or self._config.on_exhausted == "raise":
            # Early return: nothing above the default chain, or raise mode.
            return False
        if self._next_tier_up(int_index, str_family) is None:
            # Early return: no tier above may serve the call.
            return False
        reason: Any = getattr(exception, "fallback_reason", None)
        client: AiFallbackCompletions | None = self._tier_clients.get(int_index)
        frozenset_reasons: frozenset[AiFallbackReason] = (
            client.fallback_on if client is not None else self._fallback_on
        )
        # Normal return: escalate only on a trigger reason.
        return reason is not None and reason in frozenset_reasons

    def _run(
        self,
        str_operation: str,
        call: Callable[[AIBaseCompletions], T],
        *,
        int_index: int | None,
        str_method: str,
        decision: AIClassificationResult | None,
        str_route_id: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> T:
        """
        Runs one call on the selected tier, escalating on exhaustion.

        Args:
            str_operation: Method name, for logs.
            call: Invokes the operation on one client.
            int_index: Selected tier, or None for the default chain.
            str_method: How the tier was chosen, for logs.
            decision: The classifier's result, if it ran.
            str_route_id: Correlation id for this call's events.
            messages: The call's history, so escalation stays in its family.

        Returns:
            The first successful result.
        """
        self._log_routed(str_operation, str_route_id, int_index, str_method, decision)
        str_family: str | None = history_family_of(messages or [])
        int_current: int | None = int_index
        while True:
            client: AIBaseCompletions | None = self._client_at(int_current)
            if client is None:
                int_current = self._next_after_unbuildable(
                    int_current, str_operation, str_route_id, str_family
                )
                continue
            token = self._push_tags(
                {
                    TAG_ROUTE_ID: str_route_id,
                    TAG_ROUTE_TIER: self._tier_label(int_current),
                    TAG_ROUTE_METHOD: str_method,
                }
            )
            try:
                result: T = call(client)
            except Exception as exception:
                if not self._should_escalate(int_current, exception, str_family):
                    raise
                assert int_current is not None
                int_next: int | None = self._next_tier_up(int_current, str_family)
                assert int_next is not None
                self._log_tier_exhausted(
                    str_operation, str_route_id, int_current, int_next, exception
                )
                int_current = int_next
                continue
            finally:
                reset_observability_context(token)
            self._last_tier_index = int_current
            self._last_route_client = client
            # Normal return with the result.
            return result

    async def _arun(
        self,
        str_operation: str,
        call: Callable[[AIBaseCompletions], Awaitable[T]],
        *,
        int_index: int | None,
        str_method: str,
        decision: AIClassificationResult | None,
        str_route_id: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> T:
        """Async twin of _run."""
        self._log_routed(str_operation, str_route_id, int_index, str_method, decision)
        str_family: str | None = history_family_of(messages or [])
        int_current: int | None = int_index
        while True:
            client: AIBaseCompletions | None = self._client_at(int_current)
            if client is None:
                int_current = self._next_after_unbuildable(
                    int_current, str_operation, str_route_id, str_family
                )
                continue
            token = self._push_tags(
                {
                    TAG_ROUTE_ID: str_route_id,
                    TAG_ROUTE_TIER: self._tier_label(int_current),
                    TAG_ROUTE_METHOD: str_method,
                }
            )
            try:
                result: T = await call(client)
            except Exception as exception:
                if not self._should_escalate(int_current, exception, str_family):
                    raise
                assert int_current is not None
                int_next: int | None = self._next_tier_up(int_current, str_family)
                assert int_next is not None
                self._log_tier_exhausted(
                    str_operation, str_route_id, int_current, int_next, exception
                )
                int_current = int_next
                continue
            finally:
                reset_observability_context(token)
            self._last_tier_index = int_current
            self._last_route_client = client
            # Normal return with the result.
            return result

    def _next_after_unbuildable(
        self,
        int_index: int | None,
        str_operation: str,
        str_route_id: str,
        str_family: str | None,
    ) -> int | None:
        """
        Picks where a call goes when its tier could not be built: the next
        eligible tier up in escalate mode, else the error propagates.
        """
        if int_index is None:
            # Should not happen: the default client is always built.
            raise AiProviderRequestError(
                "Router default chain is unavailable.",
                status_code=None,
                fallback_reason=None,
            )
        int_next: int | None = (
            None
            if self._config.on_exhausted == "raise"
            else self._next_tier_up(int_index, str_family)
        )
        if int_next is None:
            raise AiProviderRequestError(
                f"Router tier {self._config.tiers[int_index].name!r} could not be "
                "built and no tier above it may serve this call.",
                status_code=None,
                fallback_reason=None,
            )
        _LOGGER.error(
            "ROUTER: %s cannot use tier %s (unbuildable); escalating to %s.",
            str_operation,
            self._config.tiers[int_index].name,
            self._config.tiers[int_next].name,
            extra={
                "ai_router_event": ROUTER_EVENT_TIER_EXHAUSTED,
                "route_id": str_route_id,
                "operation": str_operation,
                "tier": self._config.tiers[int_index].name,
                "fallback_reason": "unbuildable",
                "escalate_to": self._config.tiers[int_next].name,
            },
        )
        # Normal return with the next eligible tier.
        return int_next

    @staticmethod
    def _push_tags(dict_tags: dict[str, str]) -> Any:
        """Replaces the observability context with the same ids and merged tags."""
        context: ObservabilityContextModel = get_observability_context()
        # Normal return with the reset token.
        return set_observability_context(
            caller_id=context.caller_id,
            session_id=context.session_id,
            workflow_id=context.workflow_id,
            tags={**dict(context.tags), **dict_tags},
        )

    # ── Logging ─────────────────────────────────────────────────────────────

    def _log_routed(
        self,
        str_operation: str,
        str_route_id: str,
        int_index: int | None,
        str_method: str,
        decision: AIClassificationResult | None,
    ) -> None:
        context: ObservabilityContextModel = get_observability_context()
        _LOGGER.info(
            "ROUTED: %s -> %s (%s%s).",
            str_operation,
            self._tier_label(int_index),
            str_method,
            (
                f", confidence {decision.confidence}"
                if decision is not None and decision.confidence is not None
                else ""
            ),
            extra={
                "ai_router_event": ROUTER_EVENT_ROUTED,
                "route_id": str_route_id,
                "operation": str_operation,
                "tier": self._tier_label(int_index),
                "tier_index": int_index,
                "method": str_method,
                "classifier_engine": (
                    decision.provider_engine if decision is not None else None
                ),
                "classifier_model": (
                    decision.model_name if decision is not None else None
                ),
                "confidence": decision.confidence if decision is not None else None,
                "probabilities": (
                    decision.probabilities if decision is not None else None
                ),
                "classifier_latency_ms": (
                    decision.latency_ms if decision is not None else None
                ),
                "classifier_input_tokens": (
                    decision.usage.input_tokens if decision is not None else None
                ),
                "classifier_token_source": (
                    decision.token_source if decision is not None else None
                ),
                "classifier_usd_cost": (
                    self._classifier_cost() if decision is not None else None
                ),
                "caller_id": context.caller_id,
                "session_id": context.session_id,
                "workflow_id": context.workflow_id,
            },
        )

    @staticmethod
    def _classifier_cost() -> float | None:
        """
        USD cost of this request's classifier call, read from the request
        ledger so it matches the classifier's own cost event (cache tokens
        included), or None when unpriced or no request is open.
        """
        scope: RequestCostScope | None = current_request_cost_scope()
        if scope is None:
            # Early return: no request ledger to read.
            return None
        usd_cost: Decimal | None = scope.classifier_cost()
        # Normal return with the ledger's classifier spend.
        return float(usd_cost) if usd_cost is not None else None

    def _log_classifier_failed(
        self,
        str_operation: str,
        str_route_id: str,
        *,
        error: str,
        fallback_reason: Any,
        float_started: float,
        decision: AIClassificationResult | None = None,
    ) -> None:
        _LOGGER.warning(
            "ROUTER: classifier failed on %s (%s); routing to %s.",
            str_operation,
            error,
            self._tier_label(self._int_default_tier_index),
            extra={
                "ai_router_event": ROUTER_EVENT_CLASSIFIER_FAILED,
                "route_id": str_route_id,
                "operation": str_operation,
                "classifier_engine": (
                    decision.provider_engine
                    if decision is not None
                    else (
                        self._classifier._resolve_observability_provider_engine()
                        if self._classifier is not None
                        else None
                    )
                ),
                "classifier_model": (
                    decision.model_name
                    if decision is not None
                    else (self._classifier.model_name if self._classifier else None)
                ),
                "error": error,
                "fallback_reason": (
                    fallback_reason.value
                    if isinstance(fallback_reason, AiFallbackReason)
                    else fallback_reason
                ),
                "latency_ms": (time.perf_counter() - float_started) * 1000.0,
                "routed_to": self._tier_label(self._int_default_tier_index),
            },
        )

    def _log_tier_exhausted(
        self,
        str_operation: str,
        str_route_id: str,
        int_index: int,
        int_next: int,
        exception: BaseException,
    ) -> None:
        reason: Any = getattr(exception, "fallback_reason", None)
        _LOGGER.error(
            "ROUTER: %s exhausted tier %s (%s); escalating to %s: %s",
            str_operation,
            self._config.tiers[int_index].name,
            reason.value if isinstance(reason, AiFallbackReason) else reason,
            self._config.tiers[int_next].name,
            exception,
            extra={
                "ai_router_event": ROUTER_EVENT_TIER_EXHAUSTED,
                "route_id": str_route_id,
                "operation": str_operation,
                "tier": self._config.tiers[int_index].name,
                "fallback_reason": (
                    reason.value if isinstance(reason, AiFallbackReason) else reason
                ),
                "escalate_to": self._config.tiers[int_next].name,
            },
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def send_prompt(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_response_tokens: int | None = None,
        request_timeout_seconds: float | None = None,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> str:
        str_route_id: str = uuid.uuid4().hex

        def _routed() -> str:
            int_index, str_method, decision = self._select(
                str_text=prompt,
                messages=None,
                str_route=None,
                str_operation="send_prompt",
                str_route_id=str_route_id,
            )
            # Normal return with the serving tier's text.
            return self._run(
                "send_prompt",
                lambda client: client.send_prompt(
                    prompt,
                    system_prompt=system_prompt,
                    max_response_tokens=max_response_tokens,
                    request_timeout_seconds=request_timeout_seconds,
                    other_params=other_params,
                ),
                int_index=int_index,
                str_method=str_method,
                decision=decision,
                str_route_id=str_route_id,
            )

        # Normal return: the classifier and every tier attempt are charged
        # to one request keyed by the route id.
        return self._with_request_cost_scope(
            "send_prompt", _routed, request_id=str_route_id
        )

    async def asend_prompt(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_response_tokens: int | None = None,
        request_timeout_seconds: float | None = None,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> str:
        str_route_id: str = uuid.uuid4().hex

        async def _routed() -> str:
            int_index, str_method, decision = await self._aselect(
                str_text=prompt,
                messages=None,
                str_route=None,
                str_operation="asend_prompt",
                str_route_id=str_route_id,
            )
            # Normal return with the serving tier's text.
            return await self._arun(
                "asend_prompt",
                lambda client: client.asend_prompt(
                    prompt,
                    system_prompt=system_prompt,
                    max_response_tokens=max_response_tokens,
                    request_timeout_seconds=request_timeout_seconds,
                    other_params=other_params,
                ),
                int_index=int_index,
                str_method=str_method,
                decision=decision,
                str_route_id=str_route_id,
            )

        # Normal return with the request-scoped result.
        return await self._awith_request_cost_scope(
            "asend_prompt", _routed, request_id=str_route_id
        )

    def send_prompt_streaming(
        self,
        prompt: str,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> Iterator[str]:
        """Classifies once, then streams from the chosen tier; escalation only before the first chunk."""
        str_route_id: str = uuid.uuid4().hex
        # Normal return with the routing generator, charged to one request.
        return self._stream_with_request_cost_scope(
            "send_prompt_streaming",
            lambda: self._stream(prompt, other_params, str_route_id),
            request_id=str_route_id,
        )

    def _stream(
        self,
        prompt: str,
        other_params: AICompletionsPromptParamsBase | None,
        str_route_id: str,
    ) -> Iterator[str]:
        int_index, str_method, decision = self._select(
            str_text=prompt,
            messages=None,
            str_route=None,
            str_operation="send_prompt_streaming",
            str_route_id=str_route_id,
        )
        self._log_routed(
            "send_prompt_streaming", str_route_id, int_index, str_method, decision
        )
        int_current: int | None = int_index
        while True:
            client: AIBaseCompletions | None = self._client_at(int_current)
            if client is None:
                int_current = self._next_after_unbuildable(
                    int_current, "send_prompt_streaming", str_route_id, None
                )
                continue
            token = self._push_tags(
                {
                    TAG_ROUTE_ID: str_route_id,
                    TAG_ROUTE_TIER: self._tier_label(int_current),
                    TAG_ROUTE_METHOD: str_method,
                }
            )
            try:
                iterator: Iterator[str] = client.send_prompt_streaming(
                    prompt, other_params=other_params
                )
                str_first: str = next(iterator)
            except StopIteration:
                reset_observability_context(token)
                self._last_tier_index = int_current
                self._last_route_client = client
                # Early return: an empty stream.
                return
            except Exception as exception:
                reset_observability_context(token)
                if not self._should_escalate(int_current, exception, None):
                    raise
                assert int_current is not None
                int_next: int | None = self._next_tier_up(int_current, None)
                assert int_next is not None
                self._log_tier_exhausted(
                    "send_prompt_streaming",
                    str_route_id,
                    int_current,
                    int_next,
                    exception,
                )
                int_current = int_next
                continue
            self._last_tier_index = int_current
            self._last_route_client = client
            try:
                yield str_first
                yield from iterator
            finally:
                reset_observability_context(token)
            # Early return: the stream completed on this tier.
            return

    def send_structured_output(
        self,
        prompt: str | None = None,
        *,
        response_model: Type[AIStructuredPrompt] | None = None,
        response_schema: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_response_tokens: int = AIBaseCompletions.STRUCTURED_DEFAULT_MAX_RESPONSE_TOKENS,
        request_timeout_seconds: float | None = None,
        provider_options: dict[str, Any] | None = None,
        prompt_cache: AIPromptCacheHint | None = None,
    ) -> AIStructuredOutputResult:
        str_route_id: str = uuid.uuid4().hex
        dict_options, str_route = self._strip_route_option(provider_options)

        def _routed() -> AIStructuredOutputResult:
            int_index, str_method, decision = self._select(
                str_text=classification_text_for(prompt, messages),
                messages=messages,
                str_route=str_route,
                str_operation="send_structured_output",
                str_route_id=str_route_id,
            )
            # Normal return with the serving tier's result.
            return self._run(
                "send_structured_output",
                lambda client: client.send_structured_output(
                    prompt,
                    response_model=response_model,
                    response_schema=response_schema,
                    system_prompt=system_prompt,
                    messages=messages,
                    max_response_tokens=max_response_tokens,
                    request_timeout_seconds=request_timeout_seconds,
                    provider_options=dict_options,
                    prompt_cache=prompt_cache,
                ),
                int_index=int_index,
                str_method=str_method,
                decision=decision,
                str_route_id=str_route_id,
                messages=messages,
            )

        # Normal return with the request-scoped result.
        return self._with_request_cost_scope(
            "send_structured_output", _routed, request_id=str_route_id
        )

    async def asend_structured_output(
        self,
        prompt: str | None = None,
        *,
        response_model: Type[AIStructuredPrompt] | None = None,
        response_schema: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_response_tokens: int = AIBaseCompletions.STRUCTURED_DEFAULT_MAX_RESPONSE_TOKENS,
        request_timeout_seconds: float | None = None,
        provider_options: dict[str, Any] | None = None,
        prompt_cache: AIPromptCacheHint | None = None,
    ) -> AIStructuredOutputResult:
        str_route_id: str = uuid.uuid4().hex
        dict_options, str_route = self._strip_route_option(provider_options)

        async def _routed() -> AIStructuredOutputResult:
            int_index, str_method, decision = await self._aselect(
                str_text=classification_text_for(prompt, messages),
                messages=messages,
                str_route=str_route,
                str_operation="asend_structured_output",
                str_route_id=str_route_id,
            )
            # Normal return with the serving tier's result.
            return await self._arun(
                "asend_structured_output",
                lambda client: client.asend_structured_output(
                    prompt,
                    response_model=response_model,
                    response_schema=response_schema,
                    system_prompt=system_prompt,
                    messages=messages,
                    max_response_tokens=max_response_tokens,
                    request_timeout_seconds=request_timeout_seconds,
                    provider_options=dict_options,
                    prompt_cache=prompt_cache,
                ),
                int_index=int_index,
                str_method=str_method,
                decision=decision,
                str_route_id=str_route_id,
                messages=messages,
            )

        # Normal return with the request-scoped result.
        return await self._awith_request_cost_scope(
            "asend_structured_output", _routed, request_id=str_route_id
        )

    def send_conversation(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[AITool] | None = None,
        tool_choice: str | None = None,
        max_response_tokens: int | None = None,
        request_timeout_seconds: float | None = None,
        provider_options: dict[str, Any] | None = None,
        prompt_cache: AIPromptCacheHint | None = None,
    ) -> AITurnResult:
        str_route_id: str = uuid.uuid4().hex
        dict_options, str_route = self._strip_route_option(provider_options)

        def _routed() -> AITurnResult:
            int_index, str_method, decision = self._select(
                str_text=classification_text_for(None, messages),
                messages=messages,
                str_route=str_route,
                str_operation="send_conversation",
                str_route_id=str_route_id,
            )
            # Normal return with the serving tier's turn.
            return self._run(
                "send_conversation",
                lambda client: client.send_conversation(
                    system_prompt,
                    messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    max_response_tokens=max_response_tokens,
                    request_timeout_seconds=request_timeout_seconds,
                    provider_options=dict_options,
                    prompt_cache=prompt_cache,
                ),
                int_index=int_index,
                str_method=str_method,
                decision=decision,
                str_route_id=str_route_id,
                messages=messages,
            )

        # Normal return with the request-scoped turn.
        return self._with_request_cost_scope(
            "send_conversation", _routed, request_id=str_route_id
        )

    async def asend_conversation(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[AITool] | None = None,
        tool_choice: str | None = None,
        max_response_tokens: int | None = None,
        request_timeout_seconds: float | None = None,
        provider_options: dict[str, Any] | None = None,
        prompt_cache: AIPromptCacheHint | None = None,
    ) -> AITurnResult:
        str_route_id: str = uuid.uuid4().hex
        dict_options, str_route = self._strip_route_option(provider_options)

        async def _routed() -> AITurnResult:
            int_index, str_method, decision = await self._aselect(
                str_text=classification_text_for(None, messages),
                messages=messages,
                str_route=str_route,
                str_operation="asend_conversation",
                str_route_id=str_route_id,
            )
            # Normal return with the serving tier's turn.
            return await self._arun(
                "asend_conversation",
                lambda client: client.asend_conversation(
                    system_prompt,
                    messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    max_response_tokens=max_response_tokens,
                    request_timeout_seconds=request_timeout_seconds,
                    provider_options=dict_options,
                    prompt_cache=prompt_cache,
                ),
                int_index=int_index,
                str_method=str_method,
                decision=decision,
                str_route_id=str_route_id,
                messages=messages,
            )

        # Normal return with the request-scoped turn.
        return await self._awith_request_cost_scope(
            "asend_conversation", _routed, request_id=str_route_id
        )

    # ── History helpers ─────────────────────────────────────────────────────

    def _client_for_history(
        self, messages: list[dict[str, Any]] | None
    ) -> AIBaseCompletions:
        """The client whose family shaped a history, else the last route."""
        str_family: str | None = history_family_of(messages or [])
        if str_family is None:
            # Early return with the last route.
            return self._last_route_client
        client: AIBaseCompletions | None = self._client_at(
            self._pinned_tier(str_family)
        )
        # Normal return with the pinned client, or the last route if unbuildable.
        return client if client is not None else self._last_route_client

    def build_tool_result_message(
        self,
        *,
        tool_call_id: str,
        result: dict[str, Any],
        is_error: bool = False,
        messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        client: AIBaseCompletions = self._client_for_history(messages)
        if isinstance(client, AiFallbackCompletions):
            # Early return via the chain, which routes by the history itself.
            return client.build_tool_result_message(
                tool_call_id=tool_call_id,
                result=result,
                is_error=is_error,
                messages=messages,
            )
        # Normal return via a plain engine, whose base takes no messages.
        return client.build_tool_result_message(
            tool_call_id=tool_call_id, result=result, is_error=is_error
        )

    def extend_messages_with_turn(
        self,
        messages: list[dict[str, Any]],
        turn: AITurnResult,
    ) -> list[dict[str, Any]]:
        """Appends the turn in the shape of the engine that produced it."""
        # Loop over tiers for the candidate that produced the turn.
        for int_index, tier in enumerate(self._config.tiers):
            for candidate in tier.candidates:
                if (
                    turn.provider_engine == candidate.engine
                    and turn.model_name
                    and turn.model_name == candidate.model
                ):
                    client: AIBaseCompletions | None = self._client_at(int_index)
                    if client is not None:
                        # Early return via the producing tier's chain.
                        return client.extend_messages_with_turn(messages, turn)
        # Normal return via the last route, else the default client.
        return self._last_route_client.extend_messages_with_turn(messages, turn)
