# ai_fallback_completions.py
"""
Completions client that retries a failed request on another model.

AiFallbackCompletions wraps an ordered list of (engine, model) candidates.
The first is the primary, constructed up front as usual. The rest are built
lazily, the first time a request needs them, so a fallback whose optional
extra is missing or whose credentials are wrong cannot break startup.

A request moves to the next candidate only when the engine raises
AiProviderRequestError with a fallback_reason in the configured set (by
default UNAVAILABLE, RATE_LIMITED, and QUOTA_EXHAUSTED; MODEL_UNAVAILABLE is
off because it usually means a configuration typo that should fail loudly).
Every other exception propagates: validation errors, capability errors, a
refusal, or a timeout would not go better on a different model.

Three limits follow from the engines' request shapes:

- A conversation whose history already holds engine-shaped entries
  (assistant raw_content, tool results) cannot replay on another engine.
  The wrapper reads which engine family shaped the history and routes the
  turn to a candidate of that family; a failure there propagates. Routing
  is a function of the history, so one client can serve many conversations.
- A streaming call fails over only if the error arrives before the first
  chunk; after that the caller already holds partial output.
- Batches, token counting, and capabilities always go to the primary. Cost
  helpers price at the candidate that served the most recent call.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable, Iterator, Sequence
from typing import Any, ClassVar, Type, TypeVar

from pydantic import BaseModel, ConfigDict

from ..ai_base import (
    AIBaseCompletions,
    AIBatchJob,
    AIBatchRequestItem,
    AIBatchResultItem,
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

_LOGGER: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")

# Values of the `ai_fallback_event` field on this module's log records, so a
# log processor can count and alert on each without parsing the message.
FALLBACK_EVENT_FAILOVER: str = "failover"
FALLBACK_EVENT_SERVED: str = "served_by_fallback"
FALLBACK_EVENT_EXHAUSTED: str = "chain_exhausted"
FALLBACK_EVENT_UNBUILDABLE: str = "candidate_unbuildable"
FALLBACK_EVENT_SKIPPED: str = "candidate_skipped"

# Reasons a fallback client acts on unless configured otherwise.
DEFAULT_FALLBACK_REASONS: frozenset[AiFallbackReason] = frozenset(
    {
        AiFallbackReason.UNAVAILABLE,
        AiFallbackReason.RATE_LIMITED,
        AiFallbackReason.QUOTA_EXHAUSTED,
    }
)

# Roles a provider-neutral conversation history may use. Anything else
# (tool results, engine-shaped assistant turns) is engine-specific.
FROZENSET_NEUTRAL_ROLES: frozenset[str] = frozenset({"user", "assistant", "system"})

# Engine families, by the request shape their conversation history takes.
# Ordered so a subclass module (openai_responses) matches before its base
# (openai) when walking a class's MRO.
TUPLE_ENGINE_FAMILY_MODULE_MARKERS: tuple[tuple[str, str], ...] = (
    ("ai_openai_responses_completions", "openai-responses"),
    ("ai_openai_completions", "openai"),
    ("ai_anthropic_completions", "anthropic"),
    ("ai_bedrock_completions", "bedrock"),
    ("ai_google_gemini_completions", "gemini"),
)

# Top-level item types the Responses API uses in its input list. A message
# item carries a role too, so this check runs before the role-based ones.
FROZENSET_RESPONSES_ITEM_TYPES: frozenset[str] = frozenset(
    {"message", "function_call", "function_call_output", "reasoning"}
)

# Keys a Chat Completions assistant message may carry beyond role and
# content. Any of them marks the history as OpenAI-shaped; other providers
# reject them.
FROZENSET_OPENAI_ASSISTANT_KEYS: frozenset[str] = frozenset(
    {"tool_calls", "function_call", "annotations", "refusal", "audio"}
)

# Keys a provider-neutral message may carry. Exactly these two: OpenAI also
# accepts "name", but Anthropic and Bedrock reject it as an unknown key.
FROZENSET_NEUTRAL_MESSAGE_KEYS: frozenset[str] = frozenset({"role", "content"})

# Content-block keys Converse uses; a block with one of these and no "type"
# is Bedrock-shaped.
FROZENSET_CONVERSE_BLOCK_KEYS: frozenset[str] = frozenset(
    {"text", "toolUse", "toolResult", "image", "document", "cachePoint"}
)


class AIFallbackCandidate(BaseModel):
    """
    One (engine, model) pair in a fallback chain.

    Attributes:
        engine: Completions engine token, as the factory accepts it.
        model: Model name for that engine; blank means the engine default.
        base_url: Optional API base-URL override for engines that accept one.
    """

    model_config = ConfigDict(frozen=True)

    engine: str
    model: str = ""
    base_url: str | None = None

    @property
    def label(self) -> str:
        """Short engine:model form for logs."""
        # Normal return with the log label.
        return f"{self.engine}:{self.model or '<default>'}"


def parse_fallback_candidates(str_value: str) -> list[AIFallbackCandidate]:
    """
    Parses the COMPLETIONS_FALLBACKS setting.

    The format is comma-separated `engine:model` pairs. The split is on the
    first colon only, since Bedrock model ids contain colons
    (`bedrock:amazon.nova-lite-v1:0`). A pair with no colon names an engine
    and takes that engine's default model.

    Args:
        str_value: Raw setting text; blank means no fallbacks.

    Returns:
        Candidates in the order given.
    """
    list_candidates: list[AIFallbackCandidate] = []
    # Loop over each comma-separated pair.
    for str_item in str_value.split(","):
        str_pair: str = str_item.strip()
        if not str_pair:
            continue
        str_engine, _, str_model = str_pair.partition(":")
        list_candidates.append(
            AIFallbackCandidate(
                engine=str_engine.strip().lower(), model=str_model.strip()
            )
        )
    # Normal return with the parsed chain.
    return list_candidates


def parse_fallback_reasons(
    value: Iterable[AiFallbackReason | str] | str,
) -> frozenset[AiFallbackReason]:
    """
    Normalizes a fallback-reason set from configuration or code.

    Args:
        value: A comma-separated string of reason values, or an iterable of
            reasons or reason values.

    Returns:
        The reasons as a frozenset.

    Raises:
        ValueError: When a value is not an AiFallbackReason.
    """
    list_raw: list[AiFallbackReason | str] = (
        [item for item in value.split(",")] if isinstance(value, str) else list(value)
    )
    set_reasons: set[AiFallbackReason] = set()
    # Loop over each raw entry so a typo names itself.
    for raw in list_raw:
        if isinstance(raw, AiFallbackReason):
            set_reasons.add(raw)
            continue
        str_token: str = str(raw).strip().lower()
        if not str_token:
            continue
        try:
            set_reasons.add(AiFallbackReason(str_token))
        except ValueError as exception:
            raise ValueError(
                f"Unknown fallback reason {str_token!r}; expected one of "
                f"{', '.join(reason.value for reason in AiFallbackReason)}."
            ) from exception
    # Normal return with the normalized set.
    return frozenset(set_reasons)


def engine_family_of(client: AIBaseCompletions) -> str | None:
    """
    Names the request-shape family of one engine client.

    An engine may declare its family in FALLBACK_ENGINE_FAMILY; otherwise the
    class MRO is walked so vendor subclasses (an OpenAI-compatible vendor
    engine, a Responses engine) resolve to the family whose shapes they use.

    Args:
        client: A completions engine.

    Returns:
        The family name, or None for an engine outside the known set.
    """
    str_declared: Any = getattr(client, "FALLBACK_ENGINE_FAMILY", None)
    if isinstance(str_declared, str):
        # Early return with the engine's own declaration.
        return str_declared
    # Loop over the MRO so the most derived module wins.
    for klass in type(client).__mro__:
        str_module: str = klass.__module__
        for str_marker, str_family in TUPLE_ENGINE_FAMILY_MODULE_MARKERS:
            if str_marker in str_module:
                # Early return with the first family matched.
                return str_family
    # Normal return: unknown family.
    return None


def history_family_of(messages: list[dict[str, Any]]) -> str | None:
    """
    Names the engine family that shaped a conversation history.

    Args:
        messages: Caller-managed message history.

    Returns:
        The family name, or None when every message is provider-neutral.
    """
    # Loop over messages until one carries an engine-specific shape.
    for message in messages:
        if not isinstance(message, dict):
            continue
        if "parts" in message:
            # Early return: Gemini content objects.
            return "gemini"
        if message.get("type") in FROZENSET_RESPONSES_ITEM_TYPES or (
            "role" not in message and "type" in message
        ):
            # Early return: Responses API input items, including message
            # items, which carry a role as well as a type.
            return "openai-responses"
        if message.get("role") == "tool" or (
            FROZENSET_OPENAI_ASSISTANT_KEYS & message.keys()
        ):
            # Early return: Chat Completions tool results, tool calls, or an
            # assistant message carrying SDK fields.
            return "openai"
        content: Any = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict):
                if "type" in block:
                    # Early return: Messages API content blocks.
                    return "anthropic"
                if FROZENSET_CONVERSE_BLOCK_KEYS & block.keys():
                    # Early return: Converse content blocks.
                    return "bedrock"
            elif getattr(block, "type", None) is not None:
                # Early return: replayed Messages API SDK objects.
                return "anthropic"
    # Normal return: nothing engine-specific found.
    return None


class AiFallbackCompletions(AIBaseCompletions):
    """
    Completions client that fails over across an ordered candidate chain.

    Construct through AIFactory.get_ai_completions_client(fallbacks=...) or
    the COMPLETIONS_FALLBACKS setting. See the module docstring for the
    failover rules and their limits.
    """

    # Reserved per-call provider_options key; "none" keeps the call on its
    # starting candidate. Stripped before the options reach an engine.
    PROVIDER_OPTION_FALLBACK: ClassVar[str] = "fallback"

    # Marker for candidates whose construction failed; never retried.
    _UNBUILDABLE: ClassVar[object] = object()

    def __init__(
        self,
        *,
        primary: AIBaseCompletions,
        primary_candidate: AIFallbackCandidate,
        fallback_candidates: Sequence[AIFallbackCandidate],
        client_builder: Callable[[AIFallbackCandidate], AIBaseCompletions],
        fallback_on: Iterable[AiFallbackReason] = DEFAULT_FALLBACK_REASONS,
    ) -> None:
        """
        Args:
            primary: The already-constructed primary engine client.
            primary_candidate: The primary's engine and model, for logs and
                result stamping.
            fallback_candidates: Candidates to try, in order, after the primary.
            client_builder: Builds a client for one candidate; called lazily.
            fallback_on: Reasons that move a request to the next candidate.
        """
        super().__init__(model=primary.model)
        self._candidates: list[AIFallbackCandidate] = [
            primary_candidate,
            *fallback_candidates,
        ]
        self._clients: dict[int, Any] = {0: primary}
        self._client_builder: Callable[[AIFallbackCandidate], AIBaseCompletions] = (
            client_builder
        )
        self._fallback_on: frozenset[AiFallbackReason] = frozenset(fallback_on)
        # Index of the candidate that served the most recent call of any kind.
        self._last_route_index: int = 0

    # ── Introspection and primary-only delegation ───────────────────────────

    @property
    def primary(self) -> AIBaseCompletions:
        """The primary engine client."""
        # Normal return with the eagerly built primary.
        return self._clients[0]

    @property
    def candidates(self) -> list[AIFallbackCandidate]:
        """The chain, primary first."""
        # Normal return with a copy so callers cannot reorder the chain.
        return list(self._candidates)

    @property
    def fallback_on(self) -> frozenset[AiFallbackReason]:
        """Reasons that move a request to the next candidate."""
        # Normal return with the configured reason set.
        return self._fallback_on

    @property
    def last_route(self) -> AIFallbackCandidate:
        """The candidate that served the most recent call."""
        # Normal return with the last serving candidate.
        return self._candidates[self._last_route_index]

    @property
    def last_route_client(self) -> AIBaseCompletions:
        """The engine client that served the most recent call."""
        client: AIBaseCompletions | None = self._client_at(self._last_route_index)
        # Normal return: a candidate that served a call was built.
        return client if client is not None else self.primary

    @property
    def capabilities(self) -> AICompletionsCapabilitiesBase:
        """The primary's capabilities."""
        # Normal return with the primary descriptor.
        return self.primary.capabilities

    @property
    def list_model_names(self) -> list[str]:
        """The primary's model catalog."""
        # Normal return with the primary catalog.
        return self.primary.list_model_names

    @property
    def max_context_tokens(self) -> int:
        """The primary's context window."""
        # Normal return with the primary context window.
        return self.primary.max_context_tokens

    def get_org_info(self) -> AIProviderOrgInfoBase:
        """The primary's organization identity."""
        # Normal return with the primary's identity.
        return self.primary.get_org_info()

    def get_org_info_capability(self) -> AIProviderOrgInfoCapability:
        """The primary's organization-identity capability."""
        # Normal return with the primary's capability.
        return self.primary.get_org_info_capability()

    def price_per_1k_tokens(self) -> float:
        """Blended rate of the candidate that served the most recent call."""
        # Normal return priced at the last route.
        return self.last_route_client.price_per_1k_tokens()

    def compute_completion_cost(
        self,
        *,
        input_tokens: int,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        cache_write_5m_tokens: int = 0,
        cache_write_1h_tokens: int = 0,
    ) -> float:
        """
        Cost at the rates of the candidate that served the most recent call.

        Usage on a turn or structured result names its route in
        provider_engine and model_name; call this right after that call.
        """
        # Normal return priced at the last route.
        return self.last_route_client.compute_completion_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_write_5m_tokens=cache_write_5m_tokens,
            cache_write_1h_tokens=cache_write_1h_tokens,
        )

    # ── Chain mechanics ─────────────────────────────────────────────────────

    def _client_at(self, int_index: int) -> AIBaseCompletions | None:
        """
        Returns the client for one candidate, building it on first use.

        A candidate whose construction fails, for any reason, is logged once
        and skipped on every later request: a missing extra or bad
        configuration does not clear on its own, and a build error must not
        replace the outage error that started the failover.

        Args:
            int_index: Position in the chain.

        Returns:
            The client, or None when the candidate cannot be built.
        """
        if int_index in self._clients:
            cached: Any = self._clients[int_index]
            # Early return with the cached client, or None if unbuildable.
            return None if cached is self._UNBUILDABLE else cached
        candidate: AIFallbackCandidate = self._candidates[int_index]
        try:
            client: AIBaseCompletions = self._client_builder(candidate)
        except Exception as exception:
            _LOGGER.error(
                "FALLBACK candidate %s cannot be built and will be skipped for "
                "the rest of the process: %s",
                candidate.label,
                exception,
                extra={
                    "ai_fallback_event": FALLBACK_EVENT_UNBUILDABLE,
                    "fallback_candidate": candidate.label,
                },
            )
            self._clients[int_index] = self._UNBUILDABLE
            # Early return: this candidate is out of the chain.
            return None
        self._clients[int_index] = client
        # Normal return with the newly built client.
        return client

    def _built_clients(self) -> list[tuple[int, AIBaseCompletions]]:
        """Candidates already built, in chain order."""
        # Normal return with (index, client) for each usable built client.
        return [
            (int_index, client)
            for int_index, client in sorted(self._clients.items())
            if client is not self._UNBUILDABLE
        ]

    def _route_for_history(self, messages: list[dict[str, Any]]) -> int | None:
        """
        Picks the candidate a conversation history can replay on.

        Args:
            messages: Caller-managed message history.

        Returns:
            None when the history is provider-neutral (any candidate will
            do), otherwise the first built candidate of the family that
            shaped it. Only a built candidate can have produced history. A
            history that is not neutral but matches no known family (a key
            a provider's SDK added that this module does not list) pins to
            the candidate that served the most recent call, since that is
            the engine most likely to have produced it; guessing the primary
            would replay a foreign shape there.
        """
        if self._history_is_engine_neutral(messages):
            # Early return: neutral history.
            return None
        str_family: str | None = history_family_of(messages)
        if str_family is None:
            # Early return: engine-shaped but unrecognized; stay where the
            # last call was served.
            return self._last_route_index
        # Loop over built candidates for one of the shaping family.
        for int_index, client in self._built_clients():
            if engine_family_of(client) == str_family:
                # Early return with the matching candidate.
                return int_index
        # Normal return: no built candidate matches; the primary will report
        # the shape mismatch as its own error.
        return 0

    def _strip_reserved_options(
        self, provider_options: dict[str, Any] | None
    ) -> tuple[dict[str, Any] | None, bool]:
        """
        Removes the reserved fallback key before options reach an engine.

        Args:
            provider_options: The call's provider_options.

        Returns:
            (options without the key, or None when nothing remains; whether
            fallback is allowed for this call).
        """
        if not provider_options:
            # Early return: nothing to strip.
            return provider_options, True
        dict_copy: dict[str, Any] = dict(provider_options)
        str_value: str = str(dict_copy.pop(self.PROVIDER_OPTION_FALLBACK, "") or "")
        # Normal return with the stripped copy and the opt-out flag.
        return dict_copy or None, str_value.strip().lower() != "none"

    def _attempts(
        self,
        *,
        int_start: int,
        bool_allow_fallback: bool,
        tuple_capabilities: tuple[str, ...],
    ) -> Iterator[tuple[int, AIBaseCompletions, bool]]:
        """
        Yields the candidates one call may try, in order.

        The starting candidate is never skipped: the engine's own template
        method raises the typed capability error there, as it would without
        a chain. A later candidate that lacks a required capability is
        skipped with a warning.

        Args:
            int_start: Chain position to start from.
            bool_allow_fallback: False yields only the starting candidate.
            tuple_capabilities: Capability flags a fallback candidate needs.

        Yields:
            (index, client, is_last) for each candidate to try.
        """
        list_indexes: list[int] = (
            list(range(int_start, len(self._candidates)))
            if bool_allow_fallback
            else [int_start]
        )
        # Loop over the chain positions this call may use.
        for int_index in list_indexes:
            client: AIBaseCompletions | None = self._client_at(int_index)
            if client is None:
                continue
            if int_index != int_start:
                list_missing: list[str] = [
                    str_capability
                    for str_capability in tuple_capabilities
                    if not getattr(client.capabilities, str_capability, False)
                ]
                if list_missing:
                    _LOGGER.warning(
                        "FALLBACK candidate %s skipped: %s not supported.",
                        self._candidates[int_index].label,
                        ", ".join(list_missing),
                        extra={
                            "ai_fallback_event": FALLBACK_EVENT_SKIPPED,
                            "fallback_candidate": self._candidates[int_index].label,
                        },
                    )
                    continue
            yield int_index, client, int_index == list_indexes[-1]

    def _should_continue(
        self,
        error: AiProviderRequestError,
        *,
        int_index: int,
        bool_is_last: bool,
        str_operation: str,
    ) -> bool:
        """
        Decides whether a failed attempt moves the call to the next candidate.

        Args:
            error: The typed request error the candidate raised.
            int_index: Chain position that failed.
            bool_is_last: True when no candidate follows.
            str_operation: Method name, for logs.

        Returns:
            True to try the next candidate; False to re-raise.
        """
        if error.fallback_reason not in self._fallback_on:
            # Early return: not a fallback trigger, so the error propagates.
            return False
        str_reason: str = error.fallback_reason.value if error.fallback_reason else ""
        if bool_is_last:
            self._log_exhausted(str_operation, error, self._candidates[int_index].label)
            # Early return: nothing left to try; the caller re-raises.
            return False
        str_next: str = self._candidates[int_index + 1].label
        _LOGGER.error(
            "FALLBACK: %s failed on %s (%s, status %s); trying %s next: %s",
            str_operation,
            self._candidates[int_index].label,
            str_reason,
            error.status_code,
            str_next,
            error,
            extra={
                "ai_fallback_event": FALLBACK_EVENT_FAILOVER,
                "operation": str_operation,
                "fallback_from": self._candidates[int_index].label,
                "fallback_to": str_next,
                "fallback_reason": str_reason,
                "status_code": error.status_code,
            },
        )
        # Normal return: move on.
        return True

    def _log_exhausted(
        self,
        str_operation: str,
        last_error: AiProviderRequestError | None,
        str_candidates: str,
    ) -> None:
        """
        Logs that no candidate could serve a request.

        Args:
            str_operation: Method name, for logs.
            last_error: The failure that ended the chain, if any candidate ran.
            str_candidates: Labels of the candidates that were tried or skipped.
        """
        _LOGGER.error(
            "FALLBACK EXHAUSTED: %s failed on every candidate (%s); raising %s.",
            str_operation,
            str_candidates,
            last_error if last_error is not None else "a request error",
            extra={
                "ai_fallback_event": FALLBACK_EVENT_EXHAUSTED,
                "operation": str_operation,
                "fallback_reason": (
                    last_error.fallback_reason.value
                    if last_error is not None and last_error.fallback_reason
                    else ""
                ),
            },
        )

    def _log_served_by_fallback(
        self, int_index: int, str_operation: str, error: AiProviderRequestError
    ) -> None:
        """
        Logs that a fallback served a request the primary could not.

        Emitted once per request that actually moved, so a dashboard can
        count degraded requests without also counting every later turn of a
        conversation that is pinned to a fallback engine.

        Args:
            int_index: Chain position that served the call.
            str_operation: Method name, for logs.
            error: The failure that started the failover.
        """
        _LOGGER.warning(
            "FALLBACK SERVED: %s answered by %s after %s failed (%s).",
            str_operation,
            self._candidates[int_index].label,
            self._candidates[0].label,
            error.fallback_reason.value if error.fallback_reason else "",
            extra={
                "ai_fallback_event": FALLBACK_EVENT_SERVED,
                "operation": str_operation,
                "fallback_from": self._candidates[0].label,
                "fallback_to": self._candidates[int_index].label,
                "fallback_reason": (
                    error.fallback_reason.value if error.fallback_reason else ""
                ),
            },
        )

    @staticmethod
    def _typed_error(
        client: AIBaseCompletions, exception: Exception
    ) -> AiProviderRequestError | None:
        """
        Returns the typed form of an exception, via the engine's own mapping.

        Streaming providers surface raw SDK errors at the first `next()`, so
        the wrapper asks the engine to classify them the way its blocking
        paths do.

        Args:
            client: The engine that raised.
            exception: What it raised.

        Returns:
            The typed request error, or None when the engine maps nothing.
        """
        if isinstance(exception, AiProviderRequestError):
            # Early return: already typed.
            return exception
        try:
            client._raise_request_error(exception)
        except AiProviderRequestError as typed:
            # Early return with the engine's classification.
            return typed
        # Normal return: not a transport error.
        return None

    def _stamp_route(self, result: T, int_index: int) -> T:
        """
        Records which candidate served a call, on the result where it fits.

        Args:
            result: The value the engine returned.
            int_index: Chain position that served the call.

        Returns:
            The same result; turn and structured results gain route fields.
        """
        self._last_route_index = int_index
        if isinstance(result, (AITurnResult, AIStructuredOutputResult)):
            client: AIBaseCompletions | None = self._client_at(int_index)
            result.provider_engine = self._candidates[int_index].engine
            result.model_name = client.model_name if client is not None else None
        # Normal return with the (possibly stamped) result.
        return result

    def _run(
        self,
        str_operation: str,
        call: Callable[[AIBaseCompletions], T],
        *,
        tuple_capabilities: tuple[str, ...] = (),
        bool_allow_fallback: bool = True,
        int_start: int = 0,
    ) -> T:
        """
        Runs one call across the chain until a candidate serves it.

        Args:
            str_operation: Method name, for logs.
            call: Invokes the operation on one client.
            tuple_capabilities: Capability flags a fallback candidate needs.
            bool_allow_fallback: False keeps the call on the starting candidate.
            int_start: Chain position to start from.

        Returns:
            The first successful result.

        Raises:
            AiProviderRequestError: The last eligible failure when no later
                candidate could be tried, or the first ineligible one.
        """
        last_error: AiProviderRequestError | None = None
        # Loop over the chain until a candidate serves the call.
        for int_index, client, bool_is_last in self._attempts(
            int_start=int_start,
            bool_allow_fallback=bool_allow_fallback,
            tuple_capabilities=tuple_capabilities,
        ):
            try:
                result: T = call(client)
            except AiProviderRequestError as error:
                if not self._should_continue(
                    error,
                    int_index=int_index,
                    bool_is_last=bool_is_last,
                    str_operation=str_operation,
                ):
                    raise
                last_error = error
                continue
            if last_error is not None:
                self._log_served_by_fallback(int_index, str_operation, last_error)
            # Early return with the first result.
            return self._stamp_route(result, int_index)
        # Normal exit: every later candidate was skipped after a failure.
        raise self._exhausted(last_error, int_start, str_operation)

    async def _arun(
        self,
        str_operation: str,
        call: Callable[[AIBaseCompletions], Awaitable[T]],
        *,
        tuple_capabilities: tuple[str, ...] = (),
        bool_allow_fallback: bool = True,
        int_start: int = 0,
    ) -> T:
        """Async twin of _run."""
        last_error: AiProviderRequestError | None = None
        # Loop over the chain until a candidate serves the call.
        for int_index, client, bool_is_last in self._attempts(
            int_start=int_start,
            bool_allow_fallback=bool_allow_fallback,
            tuple_capabilities=tuple_capabilities,
        ):
            try:
                result: T = await call(client)
            except AiProviderRequestError as error:
                if not self._should_continue(
                    error,
                    int_index=int_index,
                    bool_is_last=bool_is_last,
                    str_operation=str_operation,
                ):
                    raise
                last_error = error
                continue
            if last_error is not None:
                self._log_served_by_fallback(int_index, str_operation, last_error)
            # Early return with the first result.
            return self._stamp_route(result, int_index)
        # Normal exit: every later candidate was skipped after a failure.
        raise self._exhausted(last_error, int_start, str_operation)

    def _exhausted(
        self,
        last_error: AiProviderRequestError | None,
        int_start: int,
        str_operation: str,
    ) -> Exception:
        """
        Builds the error to raise when the chain produced no result.

        Args:
            last_error: The last eligible failure, if any candidate ran.
            int_start: The chain position the call started from.
            str_operation: Method name, for logs.

        Returns:
            The last error, or a request error naming an unusable start.
        """
        self._log_exhausted(
            str_operation,
            last_error,
            ", ".join(candidate.label for candidate in self._candidates[int_start:]),
        )
        if last_error is not None:
            # Normal return with the failure that ended the chain.
            return last_error
        # Normal return for the case where the starting candidate could not
        # be built, so nothing ran at all.
        return AiProviderRequestError(
            f"Fallback candidate {self._candidates[int_start].label} could not "
            "be built and no other candidate was eligible for this call.",
            status_code=None,
            fallback_reason=None,
        )

    @staticmethod
    def _history_is_engine_neutral(messages: list[dict[str, Any]]) -> bool:
        """
        Reports whether a conversation history can replay on any engine.

        Neutral history is user, assistant, and system messages whose
        content is plain text and that carry no other keys, since a key one
        provider's SDK adds (annotations, refusal) is one another rejects.

        Args:
            messages: Caller-managed message history.

        Returns:
            True when every message is provider-neutral.
        """
        # Normal return after checking every message.
        return all(
            isinstance(message, dict)
            and str(message.get("role", "")) in FROZENSET_NEUTRAL_ROLES
            and isinstance(message.get("content"), str)
            and set(message) <= FROZENSET_NEUTRAL_MESSAGE_KEYS
            for message in messages
        )

    def _routing_for(
        self,
        messages: list[dict[str, Any]] | None,
        provider_options: dict[str, Any] | None,
    ) -> tuple[int, bool, dict[str, Any] | None]:
        """
        Resolves where a history-bearing call starts and whether it may move.

        Args:
            messages: Caller-managed history, if the call carries one.
            provider_options: The call's provider_options.

        Returns:
            (start index, fallback allowed, provider_options for the engine).
        """
        dict_options, bool_allowed = self._strip_reserved_options(provider_options)
        int_route: int | None = self._route_for_history(messages or [])
        if int_route is None:
            # Normal return: neutral history starts at the primary.
            return 0, bool_allowed, dict_options
        # Normal return: engine-shaped history is pinned to its family.
        return int_route, False, dict_options

    # ── Text prompts ────────────────────────────────────────────────────────

    def send_prompt(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_response_tokens: int | None = None,
        request_timeout_seconds: float | None = None,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> str:
        """Sends a text prompt, failing over across the chain."""
        # Normal return with the first candidate's text.
        return self._run(
            "send_prompt",
            lambda client: client.send_prompt(
                prompt,
                system_prompt=system_prompt,
                max_response_tokens=max_response_tokens,
                request_timeout_seconds=request_timeout_seconds,
                other_params=other_params,
            ),
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
        """Async twin of send_prompt."""
        # Normal return with the first candidate's text.
        return await self._arun(
            "asend_prompt",
            lambda client: client.asend_prompt(
                prompt,
                system_prompt=system_prompt,
                max_response_tokens=max_response_tokens,
                request_timeout_seconds=request_timeout_seconds,
                other_params=other_params,
            ),
            tuple_capabilities=("supports_async",),
        )

    def send_prompt_streaming(
        self,
        prompt: str,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> Iterator[str]:
        """
        Streams a text prompt, failing over only before the first chunk.

        Once a candidate has yielded output the caller holds partial text,
        so a later failure propagates rather than restarting elsewhere.
        """
        # Normal return with the failover-aware generator.
        return self._stream_with_fallback(prompt, other_params)

    def _stream_with_fallback(
        self,
        prompt: str,
        other_params: AICompletionsPromptParamsBase | None,
    ) -> Iterator[str]:
        """Generator behind send_prompt_streaming."""
        last_error: AiProviderRequestError | None = None
        # Loop over the chain until a candidate yields its first chunk.
        for int_index, client, bool_is_last in self._attempts(
            int_start=0,
            bool_allow_fallback=True,
            tuple_capabilities=("supports_streaming",),
        ):
            iterator: Iterator[str] = client.send_prompt_streaming(
                prompt, other_params=other_params
            )
            try:
                str_first: str = next(iterator)
            except StopIteration:
                self._last_route_index = int_index
                # Early return: the candidate served an empty stream.
                return
            except Exception as exception:
                # Streaming providers raise the SDK's own error at the first
                # next(); classify it the way the engine's blocking paths do.
                error: AiProviderRequestError | None = self._typed_error(
                    client, exception
                )
                if error is None or not self._should_continue(
                    error,
                    int_index=int_index,
                    bool_is_last=bool_is_last,
                    str_operation="send_prompt_streaming",
                ):
                    raise error if error is not None else exception
                last_error = error
                continue
            if last_error is not None:
                self._log_served_by_fallback(
                    int_index, "send_prompt_streaming", last_error
                )
            self._last_route_index = int_index
            yield str_first
            yield from iterator
            # Early return: the stream completed on this candidate.
            return
        # Normal exit: every later candidate was skipped after a failure.
        raise self._exhausted(last_error, 0, "send_prompt_streaming")

    def count_tokens(
        self,
        prompt: str,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> int:
        """Counts tokens on the primary; counts are model-specific."""
        # Normal return with the primary's count.
        return self.primary.count_tokens(prompt, other_params=other_params)

    # ── Structured output ───────────────────────────────────────────────────

    def strict_schema_prompt(
        self,
        prompt: str,
        response_model: Type[AIStructuredPrompt],
        max_response_tokens: int = AIBaseCompletions.STRUCTURED_DEFAULT_MAX_RESPONSE_TOKENS,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> AIStructuredPrompt:
        """Runs strict_schema_prompt, failing over across the chain."""
        # Normal return with the first candidate's parsed model.
        return self._run(
            "strict_schema_prompt",
            lambda client: client.strict_schema_prompt(
                prompt,
                response_model,
                max_response_tokens,
                other_params=other_params,
            ),
        )

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
        """
        Runs send_structured_output, failing over while messages are neutral.

        An engine-shaped history routes to a candidate of the family that
        shaped it, with fallback off.
        """
        int_start, bool_allowed, dict_options = self._routing_for(
            messages, provider_options
        )
        # Normal return with the serving candidate's structured result.
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
            tuple_capabilities=("supports_structured_output",),
            bool_allow_fallback=bool_allowed,
            int_start=int_start,
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
        """Async twin of send_structured_output."""
        int_start, bool_allowed, dict_options = self._routing_for(
            messages, provider_options
        )
        # Normal return with the serving candidate's structured result.
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
            tuple_capabilities=("supports_async", "supports_structured_output"),
            bool_allow_fallback=bool_allowed,
            int_start=int_start,
        )

    # ── Conversations ───────────────────────────────────────────────────────

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
        """
        Sends one conversation turn, failing over while the history is neutral.

        Once the history holds engine-shaped entries, the turn goes to a
        candidate of the family that shaped it, and a failure there
        propagates.
        """
        int_start, bool_allowed, dict_options = self._routing_for(
            messages, provider_options
        )
        # Normal return with the serving candidate's turn.
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
            tuple_capabilities=("supports_tool_use",),
            bool_allow_fallback=bool_allowed,
            int_start=int_start,
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
        """Async twin of send_conversation."""
        int_start, bool_allowed, dict_options = self._routing_for(
            messages, provider_options
        )
        # Normal return with the serving candidate's turn.
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
            tuple_capabilities=("supports_async", "supports_tool_use"),
            bool_allow_fallback=bool_allowed,
            int_start=int_start,
        )

    def _client_for_history(self, messages: list[dict[str, Any]]) -> AIBaseCompletions:
        """
        The engine whose shapes a history uses.

        Args:
            messages: Caller-managed message history.

        Returns:
            The candidate of the shaping family; for neutral history, the
            candidate that served the most recent call, since that is the
            one whose turn the caller is about to append.
        """
        int_route: int | None = self._route_for_history(messages)
        if int_route is None:
            # Early return: the turn just served shapes what comes next.
            return self.last_route_client
        client: AIBaseCompletions | None = self._client_at(int_route)
        # Normal return with the shaping family's client.
        return client if client is not None else self.primary

    def build_tool_result_message(
        self,
        *,
        tool_call_id: str,
        result: dict[str, Any],
        is_error: bool = False,
        messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Builds the tool-result message in the shape of the serving engine.

        Args:
            tool_call_id: Id of the tool call being answered.
            result: The tool's output.
            is_error: True when the tool failed.
            messages: The conversation's history. Pass it whenever one
                client serves several conversations; it names the engine
                whose shape the result must take. Without it, the candidate
                that served the most recent call is used.
        """
        client: AIBaseCompletions = (
            self._client_for_history(messages)
            if messages is not None
            else self.last_route_client
        )
        # Normal return with the engine-shaped tool result.
        return client.build_tool_result_message(
            tool_call_id=tool_call_id, result=result, is_error=is_error
        )

    def extend_messages_with_turn(
        self,
        messages: list[dict[str, Any]],
        turn: AITurnResult,
    ) -> list[dict[str, Any]]:
        """Appends the assistant turn in the shape of the engine that made it."""
        client: AIBaseCompletions = self._client_for_turn(turn, messages)
        # Normal return with the extended history.
        return client.extend_messages_with_turn(messages, turn)

    def _client_for_turn(
        self, turn: AITurnResult, messages: list[dict[str, Any]]
    ) -> AIBaseCompletions:
        """
        The engine that produced a turn, read from its route stamp.

        Args:
            turn: A turn this client returned.
            messages: The conversation's history, used when the turn carries
                no stamp.

        Returns:
            The candidate named by the stamp, else the history's engine.
        """
        if turn.provider_engine is not None:
            # Loop over built candidates for the one the stamp names.
            for int_index, client in self._built_clients():
                candidate: AIFallbackCandidate = self._candidates[int_index]
                if (
                    candidate.engine == turn.provider_engine
                    and client.model_name == turn.model_name
                ):
                    # Early return with the stamped candidate.
                    return client
        # Normal return by history shape.
        return self._client_for_history(messages)

    # ── Batches: primary only ───────────────────────────────────────────────

    def submit_batch(self, requests: list[AIBatchRequestItem]) -> AIBatchJob:
        """Submits on the primary; batch jobs are provider-specific."""
        # Normal return with the primary's job.
        return self.primary.submit_batch(requests)

    def get_batch(self, batch: str | AIBatchJob) -> AIBatchJob:
        """Polls on the primary."""
        # Normal return with the primary's job state.
        return self.primary.get_batch(batch)

    def get_batch_results(self, batch: str | AIBatchJob) -> list[AIBatchResultItem]:
        """Fetches results on the primary."""
        # Normal return with the primary's results.
        return self.primary.get_batch_results(batch)

    def cancel_batch(self, batch: str | AIBatchJob) -> AIBatchJob:
        """Cancels on the primary."""
        # Normal return with the primary's job state.
        return self.primary.cancel_batch(batch)

    def run_batch(
        self,
        requests: list[AIBatchRequestItem],
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float | None = None,
    ) -> list[AIBatchResultItem]:
        """Runs a batch end to end on the primary."""
        # Normal return with the primary's batch results.
        return self.primary.run_batch(
            requests,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
