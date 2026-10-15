"""
Per-request cost: one total across classification, failed attempts, fallbacks,
and escalations.

A request is the outermost public call: a router or fallback wrapper around
its attempt loop, a plain engine around its one provider call, or a caller's
own `request_cost_scope()` composing several library calls into one logical
unit. The scope lives in a contextvar; every provider call that runs under it
(through `execute_observed_call` and its async and streaming twins) adds one
line item, whether or not the logger-backed observability middleware is
enabled, so the report is always available programmatically. When the
outermost scope closes, one `ai_api_request_cost` event goes to the cost topic
if `emit_cost` is on. Per-call `ai_api_call_cost` events are unchanged and
gain a `tag_request_id` field for correlation.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ai_api_unified.pricing.model_pricing import AIModelPricing
from ai_api_unified.pricing.pricing_registry import get_model_pricing

if TYPE_CHECKING:
    from ai_api_unified.middleware.observability_runtime import (
        AiApiCallContextModel,
        AiApiCallResultSummaryModel,
    )

_LOGGER: logging.Logger = logging.getLogger(__name__)

OBSERVABILITY_EVENT_REQUEST_COST: str = "ai_api_request_cost"
TAG_REQUEST_ID: str = "request_id"
TAG_ROUTER_ROLE: str = "router_role"
TAG_VALUE_CLASSIFIER_ROLE: str = "classifier"
TAG_ROUTE_TIER: str = "route_tier"
TAG_ROUTE_METHOD: str = "route_method"

LINE_ITEM_ROLE_CLASSIFIER: str = "classifier"
LINE_ITEM_ROLE_ATTEMPT: str = "attempt"
LINE_ITEM_OUTCOME_OK: str = "ok"
LINE_ITEM_OUTCOME_ERROR: str = "error"
REQUEST_STATUS_OK: str = "ok"
REQUEST_STATUS_ERROR: str = "error"

DECIMAL_ZERO: Decimal = Decimal("0")


class AIRequestCostLineItem(BaseModel):
    """
    One provider call inside a request, priced the way its per-call event is.

    Attributes:
        call_id: The per-call event id, so a line item joins its
            `ai_api_call_cost` and `ai_api_call_error` events.
        provider: Provider vendor (`anthropic`, `openai`, ...).
        provider_engine: Engine name the client used (`claude`, `bedrock`, ...).
        model: Model the call went to, when known.
        operation: Public operation name (`send_prompt`, ...).
        role: `classifier` when the call ran under the router's classifier
            tag, else `attempt`.
        route_tier: Router tier label from the call's tags, if routed.
        route_method: How the router chose the tier, if routed.
        input_tokens: Provider-reported prompt tokens (cached subset included).
        output_tokens: Provider-reported completion tokens.
        cached_input_tokens: Cache-read subset of `input_tokens`.
        cache_write_5m_tokens: Tokens written to a 5-minute cache.
        cache_write_1h_tokens: Tokens written to a 1-hour cache.
        usd_cost: Cost from registry pricing; None when the call failed, the
            model is unpriced, or the provider reported no usage.
        outcome: `ok` or `error`.
        error_type: Exception class name when the call failed.
        fallback_reason: The failure's `AiFallbackReason` value, if any.
        served: True for the call whose answer the request returned.
        elapsed_ms: Wall-clock provider time for this call.
    """

    model_config = ConfigDict(frozen=True)

    call_id: str
    provider: str
    provider_engine: str
    model: str | None = None
    operation: str
    role: str = LINE_ITEM_ROLE_ATTEMPT
    route_tier: str | None = None
    route_method: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_5m_tokens: int | None = None
    cache_write_1h_tokens: int | None = None
    usd_cost: Decimal | None = None
    outcome: str = LINE_ITEM_OUTCOME_OK
    error_type: str | None = None
    fallback_reason: str | None = None
    served: bool = False
    elapsed_ms: float = 0.0

    @property
    def billed(self) -> bool:
        """
        True when the provider reported usage, priced or not. A failed call
        whose exception carried usage (a billed answer the library could not
        use) is billed too; a 529 or a connection error is not.
        """
        # Normal return: any usage means the provider charged for the call.
        return any(
            value is not None
            for value in (
                self.input_tokens,
                self.output_tokens,
                self.cache_write_5m_tokens,
                self.cache_write_1h_tokens,
            )
        )

    def to_event_fields(self) -> dict[str, Any]:
        """Compact form for the `calls` list on the request-cost event."""
        # Normal return with the scalar fields a log consumer needs.
        return {
            "call_id": self.call_id,
            "role": self.role,
            "provider_engine": self.provider_engine,
            "model": self.model,
            "route_tier": self.route_tier,
            "outcome": self.outcome,
            "error_type": self.error_type,
            "fallback_reason": self.fallback_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "usd_cost": str(self.usd_cost) if self.usd_cost is not None else None,
            "served": self.served,
        }


class AIRequestCostReport(BaseModel):
    """
    What one request cost, across every call it made.

    Attributes:
        request_id: Correlation id; the router's `route_id` when routed.
        operation: Public operation the request started with.
        status: `ok` when the request returned a result, else `error`.
        started_at_utc: When the scope opened.
        elapsed_ms: Wall-clock time from open to close.
        caller_id: From the observability context when the scope opened.
        session_id: From the observability context when the scope opened.
        workflow_id: From the observability context when the scope opened.
        tags: Caller tags on the observability context when the scope opened.
        served_by_engine: Engine of the call whose answer was returned.
        served_by_model: Model of the call whose answer was returned.
        served_by_tier: Router tier of the served call, if routed.
        served_by_method: Router method of the served call, if routed.
        total_usd: Sum of every priced line item, classifier included.
        classifier_usd: Sum of priced classifier calls.
        wasted_usd: Priced attempts that answered but were not served.
        billed_calls: Calls that answered with usage, priced or not.
        failed_calls: Calls that raised.
        unpriced_calls: Billed calls on a model with no registry pricing, so
            `total_usd` understates the request by their cost.
        calls: Every line item, in call order.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str
    operation: str
    status: str
    started_at_utc: datetime
    elapsed_ms: float
    caller_id: str | None = None
    session_id: str | None = None
    workflow_id: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    served_by_engine: str | None = None
    served_by_model: str | None = None
    served_by_tier: str | None = None
    served_by_method: str | None = None
    total_usd: Decimal = DECIMAL_ZERO
    classifier_usd: Decimal = DECIMAL_ZERO
    wasted_usd: Decimal = DECIMAL_ZERO
    billed_calls: int = 0
    failed_calls: int = 0
    unpriced_calls: int = 0
    calls: list[AIRequestCostLineItem] = Field(default_factory=list)

    @property
    def served_by(self) -> str | None:
        """`engine:model` of the served call, or None when nothing was served."""
        if self.served_by_engine is None:
            # Early return: nothing served.
            return None
        # Normal return with the engine:model label.
        return f"{self.served_by_engine}:{self.served_by_model}"

    def to_event_fields(self) -> dict[str, Any]:
        """Fields for the `ai_api_request_cost` event; money as strings."""
        dict_fields: dict[str, Any] = {
            "request_id": self.request_id,
            "operation": self.operation,
            "status": self.status,
            "event_time_utc": datetime.now(timezone.utc).isoformat(),
            "started_at_utc": self.started_at_utc.isoformat(),
            "elapsed_ms": self.elapsed_ms,
            "caller_id": self.caller_id,
            "session_id": self.session_id,
            "workflow_id": self.workflow_id,
            "served_by": self.served_by,
            "served_by_tier": self.served_by_tier,
            "served_by_method": self.served_by_method,
            "total_usd": str(self.total_usd),
            "classifier_usd": str(self.classifier_usd),
            "wasted_usd": str(self.wasted_usd),
            "currency": "USD",
            "calls_count": len(self.calls),
            "billed_calls": self.billed_calls,
            "failed_calls": self.failed_calls,
            "unpriced_calls": self.unpriced_calls,
            "calls": [item.to_event_fields() for item in self.calls],
        }
        # Loop over caller tags so request events correlate like per-call ones.
        for str_key, str_value in self.tags.items():
            dict_fields[f"tag_{str_key}"] = str_value
        # Normal return with the event payload.
        return dict_fields


def resolve_call_pricing(call_context: AiApiCallContextModel) -> AIModelPricing | None:
    """
    Looks up token pricing for one call's provider and model.

    Args:
        call_context: Provider-boundary call metadata.

    Returns:
        The registry pricing with token rates, or None when the model is
        unknown or has no token pricing on record.
    """
    if call_context.model_name is None:
        # Early return because a model is required to resolve pricing.
        return None
    pricing: AIModelPricing | None = get_model_pricing(
        call_context.provider_vendor, call_context.model_name
    )
    if pricing is None or pricing.token_rates is None:
        # Early return because this model has no token pricing on record.
        return None
    # Normal return with the priced entry.
    return pricing


def compute_call_cost(
    pricing: AIModelPricing,
    call_result_summary: AiApiCallResultSummaryModel,
) -> Decimal | None:
    """
    Prices one call from its provider-reported usage; the per-call cost event
    and the request line item share this math.

    Cached input tokens are a subset of `provider_prompt_tokens` billed at the
    cached rate; the remainder bills at the full input rate. Cache writes are
    reported separately by the provider and are not part of the prompt count,
    so they are added rather than subtracted.

    Args:
        pricing: Registry pricing for the call's model, with token rates.
        call_result_summary: Provider-result usage for the call.

    Returns:
        USD cost, or None when the provider reported no usage at all.
    """
    int_input_tokens: int | None = call_result_summary.provider_prompt_tokens
    int_output_tokens: int | None = call_result_summary.provider_completion_tokens
    int_cache_write_5m_tokens: int | None = (
        call_result_summary.provider_cache_write_5m_tokens
    )
    int_cache_write_1h_tokens: int | None = (
        call_result_summary.provider_cache_write_1h_tokens
    )
    if (
        int_input_tokens is None
        and int_output_tokens is None
        and int_cache_write_5m_tokens is None
        and int_cache_write_1h_tokens is None
    ):
        # Early return because there is no usage to cost.
        return None
    int_cached: int = call_result_summary.provider_cached_input_tokens or 0
    int_non_cached_input: int = max((int_input_tokens or 0) - int_cached, 0)
    # Normal return with the priced call.
    return pricing.compute_token_cost(
        input_tokens=int_non_cached_input,
        output_tokens=int_output_tokens or 0,
        cached_input_tokens=int_cached,
        cache_write_5m_tokens=int_cache_write_5m_tokens or 0,
        cache_write_1h_tokens=int_cache_write_1h_tokens or 0,
    )


def build_line_item(
    *,
    call_context: AiApiCallContextModel,
    call_result_summary: AiApiCallResultSummaryModel | None,
    float_elapsed_ms: float,
    exception: BaseException | None = None,
) -> AIRequestCostLineItem:
    """
    Builds the request line item for one provider call.

    Args:
        call_context: Provider-boundary call metadata (tags carry the router
            role, tier, and method).
        call_result_summary: Usage summary when the call answered; None when
            it raised.
        float_elapsed_ms: Wall-clock provider time.
        exception: What the call raised, if it failed.

    Returns:
        The line item, priced when the call answered on a priced model.
    """
    dict_tags: Mapping[str, str] = call_context.dict_tags
    str_role: str = (
        LINE_ITEM_ROLE_CLASSIFIER
        if dict_tags.get(TAG_ROUTER_ROLE) == TAG_VALUE_CLASSIFIER_ROLE
        else LINE_ITEM_ROLE_ATTEMPT
    )
    str_outcome: str = LINE_ITEM_OUTCOME_OK
    str_error_type: str | None = None
    str_fallback_reason: str | None = None
    if exception is not None:
        reason: Any = getattr(exception, "fallback_reason", None)
        str_outcome = LINE_ITEM_OUTCOME_ERROR
        str_error_type = type(exception).__name__
        str_fallback_reason = getattr(reason, "value", reason)
        if call_result_summary is None:
            # A billed failure (an answer the library could not use) carries
            # the provider's usage on the exception; a 529 carries none.
            call_result_summary = summary_from_exception_usage(
                exception, float_elapsed_ms
            )
    if call_result_summary is None:
        # Early return with a failed attempt: a call, but nothing billed.
        return AIRequestCostLineItem(
            call_id=call_context.call_id,
            provider=call_context.provider_vendor,
            provider_engine=call_context.provider_engine,
            model=call_context.model_name,
            operation=call_context.operation,
            role=str_role,
            route_tier=dict_tags.get(TAG_ROUTE_TIER),
            route_method=dict_tags.get(TAG_ROUTE_METHOD),
            outcome=str_outcome,
            error_type=str_error_type,
            fallback_reason=str_fallback_reason,
            elapsed_ms=float_elapsed_ms,
        )
    pricing: AIModelPricing | None = resolve_call_pricing(call_context)
    usd_cost: Decimal | None = (
        compute_call_cost(pricing, call_result_summary) if pricing else None
    )
    # Normal return with the priced (or unpriced) billed call.
    return AIRequestCostLineItem(
        call_id=call_context.call_id,
        provider=call_context.provider_vendor,
        provider_engine=call_context.provider_engine,
        model=call_context.model_name,
        operation=call_context.operation,
        role=str_role,
        route_tier=dict_tags.get(TAG_ROUTE_TIER),
        route_method=dict_tags.get(TAG_ROUTE_METHOD),
        input_tokens=call_result_summary.provider_prompt_tokens,
        output_tokens=call_result_summary.provider_completion_tokens,
        cached_input_tokens=call_result_summary.provider_cached_input_tokens,
        cache_write_5m_tokens=call_result_summary.provider_cache_write_5m_tokens,
        cache_write_1h_tokens=call_result_summary.provider_cache_write_1h_tokens,
        usd_cost=usd_cost,
        outcome=str_outcome,
        error_type=str_error_type,
        fallback_reason=str_fallback_reason,
        elapsed_ms=float_elapsed_ms,
    )


def summary_from_exception_usage(
    exception: BaseException,
    float_elapsed_ms: float,
) -> AiApiCallResultSummaryModel | None:
    """
    Builds a usage summary from the `usage` an exception carries, when a
    billed answer was discarded (invalid structured output, a truncated
    structured answer).

    Args:
        exception: The failure; its `usage` attribute, when present, is an
            AITokenUsage-shaped object (input_tokens, output_tokens, ...).
        float_elapsed_ms: Wall-clock provider time for the call.

    Returns:
        The summary, or None when the exception carries no usage or the usage
        reports no tokens at all.
    """
    from ai_api_unified.middleware.observability_runtime import (
        AiApiCallResultSummaryModel,
    )

    usage: Any = getattr(exception, "usage", None)
    if usage is None:
        # Early return: a transport failure carries nothing billable.
        return None
    tuple_counts: tuple[int | None, ...] = tuple(
        getattr(usage, str_field, None)
        for str_field in (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "total_tokens",
            "cache_write_5m_tokens",
            "cache_write_1h_tokens",
        )
    )
    if all(value is None for value in tuple_counts):
        # Early return: usage present but empty, so nothing was billed.
        return None
    # Normal return with the discarded answer's usage.
    return AiApiCallResultSummaryModel(
        provider_elapsed_ms=float_elapsed_ms,
        provider_prompt_tokens=tuple_counts[0],
        provider_completion_tokens=tuple_counts[1],
        provider_cached_input_tokens=tuple_counts[2],
        provider_total_tokens=tuple_counts[3],
        provider_cache_write_5m_tokens=tuple_counts[4],
        provider_cache_write_1h_tokens=tuple_counts[5],
    )


class RequestCostScope:
    """
    The mutable accumulator behind one request; `request_cost_scope()` yields it.

    Attributes:
        request_id: Correlation id pushed as the `request_id` tag.
        operation: Public operation the request started with.
        bool_owner: True on the scope itself, the outermost one that closes
            and emits. A nested `request_cost_scope()` call receives a
            JoinedRequestCostScope view with this False, so only the owner
            stores the report.
        report: The finished report once the owner's scope has closed.
    """

    def __init__(self, operation: str, request_id: str | None = None) -> None:
        from ai_api_unified.middleware.observability_runtime import (
            get_observability_context,
        )

        context = get_observability_context()
        self.request_id: str = request_id or uuid.uuid4().hex
        self.operation: str = operation
        self.bool_owner: bool = True
        self.report: AIRequestCostReport | None = None
        self._caller_id: str | None = context.caller_id
        self._session_id: str | None = context.session_id
        self._workflow_id: str | None = context.workflow_id
        self._tags: dict[str, str] = dict(context.tags)
        self._started_at: datetime = datetime.now(timezone.utc)
        self._perf_started: float = time.perf_counter()
        self._items: list[AIRequestCostLineItem] = []

    @property
    def items(self) -> list[AIRequestCostLineItem]:
        """Line items recorded so far, in call order."""
        # Normal return with a copy so callers cannot reorder the ledger.
        return list(self._items)

    def record(self, item: AIRequestCostLineItem) -> None:
        """Appends one provider call to the request."""
        self._items.append(item)
        # Normal return after recording.
        return None

    def mark_served(self) -> None:
        """
        Marks the most recent successful attempt as the one the request
        returned. Classifier calls are never served.
        """
        # Loop backward for the last answer that could have been returned.
        for int_index in range(len(self._items) - 1, -1, -1):
            item: AIRequestCostLineItem = self._items[int_index]
            if (
                item.outcome == LINE_ITEM_OUTCOME_OK
                and item.role != LINE_ITEM_ROLE_CLASSIFIER
            ):
                if not item.served:
                    self._items[int_index] = item.model_copy(update={"served": True})
                # Early return after marking the served call.
                return None
        # Normal return: no successful attempt to mark.
        return None

    def discard_last_served(self) -> None:
        """
        Clears the served mark on the most recent served item: the answer
        was marked when the provider call returned, and the library then
        rejected it (a failed response-model validation), so it is wasted.
        """
        # Loop backward for the item most recently marked served.
        for int_index in range(len(self._items) - 1, -1, -1):
            if self._items[int_index].served:
                self._items[int_index] = self._items[int_index].model_copy(
                    update={"served": False}
                )
                # Early return after clearing the mark.
                return None
        # Normal return: nothing was marked.
        return None

    def classifier_cost(self) -> Decimal | None:
        """Sum of priced classifier calls so far, or None when none ran."""
        list_costs: list[Decimal] = [
            item.usd_cost
            for item in self._items
            if item.role == LINE_ITEM_ROLE_CLASSIFIER and item.usd_cost is not None
        ]
        if not list_costs:
            # Early return: no priced classifier call.
            return None
        # Normal return with the classifier spend.
        return sum(list_costs, DECIMAL_ZERO)

    def build_report(self, status: str) -> AIRequestCostReport:
        """Totals the line items into the request report."""
        served: AIRequestCostLineItem | None = next(
            (item for item in reversed(self._items) if item.served), None
        )
        total_usd: Decimal = sum(
            (item.usd_cost for item in self._items if item.usd_cost is not None),
            DECIMAL_ZERO,
        )
        classifier_usd: Decimal = self.classifier_cost() or DECIMAL_ZERO
        wasted_usd: Decimal = sum(
            (
                item.usd_cost
                for item in self._items
                if item.usd_cost is not None
                and not item.served
                and item.role != LINE_ITEM_ROLE_CLASSIFIER
            ),
            DECIMAL_ZERO,
        )
        # Normal return with the totals.
        return AIRequestCostReport(
            request_id=self.request_id,
            operation=self.operation,
            status=status,
            started_at_utc=self._started_at,
            elapsed_ms=(time.perf_counter() - self._perf_started) * 1000.0,
            caller_id=self._caller_id,
            session_id=self._session_id,
            workflow_id=self._workflow_id,
            tags=dict(self._tags),
            served_by_engine=served.provider_engine if served else None,
            served_by_model=served.model if served else None,
            served_by_tier=served.route_tier if served else None,
            served_by_method=served.route_method if served else None,
            total_usd=total_usd,
            classifier_usd=classifier_usd,
            wasted_usd=wasted_usd,
            billed_calls=sum(1 for item in self._items if item.billed),
            failed_calls=sum(
                1 for item in self._items if item.outcome == LINE_ITEM_OUTCOME_ERROR
            ),
            unpriced_calls=sum(
                1 for item in self._items if item.billed and item.usd_cost is None
            ),
            calls=list(self._items),
        )


class JoinedRequestCostScope:
    """
    What a nested `request_cost_scope()` call receives: a view of the outer
    scope that records into it and reads from it, with `bool_owner` False so
    the joiner knows it neither closes the request nor stores the report.
    """

    def __init__(self, outer: RequestCostScope) -> None:
        self._outer: RequestCostScope = outer
        self.bool_owner: bool = False

    def __getattr__(self, str_name: str) -> Any:
        # Normal return: everything but ownership comes from the outer scope.
        return getattr(self._outer, str_name)


REQUEST_COST_SCOPE: ContextVar[RequestCostScope | None] = ContextVar(
    "ai_api_unified_request_cost_scope", default=None
)


def current_request_cost_scope() -> RequestCostScope | None:
    """The active request scope, or None outside any request."""
    # Normal return with the contextvar value.
    return REQUEST_COST_SCOPE.get()


def record_call_line_item(
    *,
    call_context: AiApiCallContextModel,
    call_result_summary: AiApiCallResultSummaryModel | None,
    float_elapsed_ms: float,
    exception: BaseException | None = None,
) -> None:
    """
    Adds one provider call to the active request scope, if any. Fails open:
    a pricing or bookkeeping error is logged and never reaches provider code.
    """
    scope: RequestCostScope | None = REQUEST_COST_SCOPE.get()
    if scope is None:
        # Early return: no request is open.
        return None
    try:
        scope.record(
            build_line_item(
                call_context=call_context,
                call_result_summary=call_result_summary,
                float_elapsed_ms=float_elapsed_ms,
                exception=exception,
            )
        )
    except Exception as record_exception:
        _LOGGER.warning(
            "request_cost_line_item_failed error_type=%s",
            record_exception.__class__.__name__,
        )
    # Normal return after recording (or logging the failure).
    return None


def _emit_request_cost(report: AIRequestCostReport) -> None:
    """Hands the finished report to the configured observability middleware."""
    from ai_api_unified.middleware.observability import get_observability_middleware

    try:
        get_observability_middleware().emit_request_cost(report)
    except Exception as exception:
        _LOGGER.warning(
            "request_cost_emit_failed error_type=%s",
            exception.__class__.__name__,
        )
    # Normal return: emission is best effort.
    return None


@contextmanager
def request_cost_scope(
    operation: str,
    *,
    request_id: str | None = None,
) -> Iterator[RequestCostScope | JoinedRequestCostScope]:
    """
    Opens (or joins) the request that provider calls are charged to.

    The outermost call owns the scope: on exit it totals the line items into
    `scope.report`, emits `ai_api_request_cost` when cost events are on, and
    restores the observability context. A nested call joins the outer scope
    and receives a view of it with `bool_owner` False. Use it directly to
    roll several library calls into one logical request, such as a workflow
    step:

        with request_cost_scope("summarize-contract") as scope:
            client.send_structured_output(...)
            client.send_prompt(...)
        print(scope.report.total_usd)

    Args:
        operation: Label for the request, such as the public method name or
            a workflow step.
        request_id: Correlation id; generated when omitted. A router passes
            its `route_id`.

    Yields:
        The active scope, or a joined view of it. `scope.bool_owner` tells
        the caller whether it opened the scope; `scope.report` is set when
        the owner's block ends.
    """
    from ai_api_unified.middleware.observability_runtime import (
        get_observability_context,
        reset_observability_context,
        set_observability_context,
    )

    existing: RequestCostScope | None = REQUEST_COST_SCOPE.get()
    if existing is not None:
        # Early yield: join the outer request; its owner closes it.
        yield JoinedRequestCostScope(existing)
        return
    scope: RequestCostScope = RequestCostScope(operation, request_id=request_id)
    token_scope: Token[RequestCostScope | None] = REQUEST_COST_SCOPE.set(scope)
    context = get_observability_context()
    token_context = set_observability_context(
        caller_id=context.caller_id,
        session_id=context.session_id,
        workflow_id=context.workflow_id,
        tags={**dict(context.tags), TAG_REQUEST_ID: scope.request_id},
    )
    str_status: str = REQUEST_STATUS_ERROR
    try:
        yield scope
        str_status = REQUEST_STATUS_OK
    finally:
        try:
            reset_observability_context(token_context)
            REQUEST_COST_SCOPE.reset(token_scope)
        except ValueError:
            # A generator closed from another context cannot reset its
            # tokens; clear the scope directly so it does not leak.
            REQUEST_COST_SCOPE.set(None)
        try:
            scope.report = scope.build_report(str_status)
        except Exception as exception:
            _LOGGER.warning(
                "request_cost_report_failed error_type=%s",
                exception.__class__.__name__,
            )
        if scope.report is not None:
            _emit_request_cost(scope.report)
