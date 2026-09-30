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
  Such a turn stays on the engine that served the conversation's last
  turn, and a failure there propagates.
- A streaming call fails over only if the error arrives before the first
  chunk; after that the caller already holds partial output.
- Batches, token counting, and capabilities always go to the primary.
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
    AIStructuredOutputResult,
    AIStructuredPrompt,
    AITool,
    AITurnResult,
)
from ..ai_provider_exceptions import (
    AiFallbackReason,
    AiProviderConfigurationError,
    AiProviderDependencyUnavailableError,
    AiProviderRequestError,
)

_LOGGER: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")

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


class AiFallbackCompletions(AIBaseCompletions):
    """
    Completions client that fails over across an ordered candidate chain.

    Construct through AIFactory.get_ai_completions_client(fallbacks=...) or
    the COMPLETIONS_FALLBACKS setting. See the module docstring for the
    failover rules and their limits.
    """

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
        # Index of the candidate that served the most recent conversation
        # turn; engine-shaped history and tool-result messages follow it.
        self._conversation_route_index: int = 0

    # ── Introspection ───────────────────────────────────────────────────────

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

    # ── Chain mechanics ─────────────────────────────────────────────────────

    def _client_at(self, int_index: int) -> AIBaseCompletions | None:
        """
        Returns the client for one candidate, building it on first use.

        A candidate whose construction fails (missing extra, bad
        configuration) is logged once and skipped on every later request,
        since neither clears on its own.

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
        except (
            AiProviderDependencyUnavailableError,
            AiProviderConfigurationError,
            ValueError,
        ) as exception:
            _LOGGER.warning(
                "Fallback candidate %s cannot be built and will be skipped: %s",
                candidate.label,
                exception,
            )
            self._clients[int_index] = self._UNBUILDABLE
            # Early return: this candidate is out of the chain.
            return None
        self._clients[int_index] = client
        # Normal return with the newly built client.
        return client

    def _fallback_allowed(self, provider_options: dict[str, Any] | None) -> bool:
        """
        Reports whether a call may leave the primary.

        Args:
            provider_options: The call's provider_options, which may carry
                the reserved "fallback" key.

        Returns:
            False when provider_options sets fallback to "none".
        """
        if not provider_options:
            # Early return: nothing disables it.
            return True
        str_value: str = str(
            provider_options.get(self.PROVIDER_OPTION_FALLBACK, "") or ""
        )
        # Normal return: only an explicit "none" disables fallback.
        return str_value.strip().lower() != "none"

    def _eligible(self, error: AiProviderRequestError) -> bool:
        """True when the error's reason is in the configured set."""
        # Normal return with the reason-set membership.
        return error.fallback_reason in self._fallback_on

    def _candidate_indexes(
        self, *, int_start: int, bool_allow_fallback: bool
    ) -> list[int]:
        """The chain positions one call may try, in order."""
        if not bool_allow_fallback:
            # Early return with the single starting candidate.
            return [int_start]
        # Normal return with the starting candidate and everything after it.
        return list(range(int_start, len(self._candidates)))

    def _skip_for_capability(
        self, int_index: int, client: AIBaseCompletions, str_capability: str | None
    ) -> bool:
        """
        Reports whether a fallback candidate lacks the capability a call needs.

        The starting candidate is never skipped: the engine's own template
        method raises the typed capability error there, as it would without
        a fallback chain.
        """
        if str_capability is None or int_index == 0:
            # Early return: nothing to check.
            return False
        if getattr(client.capabilities, str_capability, False):
            # Early return: the candidate can serve the call.
            return False
        _LOGGER.warning(
            "Fallback candidate %s skipped: %s is not supported.",
            self._candidates[int_index].label,
            str_capability,
        )
        # Normal return: skip this candidate.
        return True

    def _log_failover(
        self, int_index: int, str_operation: str, error: AiProviderRequestError
    ) -> None:
        """Logs one failover at warning level, naming engine and reason."""
        str_reason: str = error.fallback_reason.value if error.fallback_reason else ""
        _LOGGER.warning(
            "%s failed on %s (%s); trying the next fallback candidate: %s",
            str_operation,
            self._candidates[int_index].label,
            str_reason,
            error,
        )

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
        str_capability: str | None = None,
        bool_allow_fallback: bool = True,
        int_start: int = 0,
    ) -> T:
        """
        Runs one call across the chain until a candidate serves it.

        Args:
            str_operation: Method name, for logs.
            call: Invokes the operation on one client.
            str_capability: Capability flag a fallback candidate must have.
            bool_allow_fallback: False keeps the call on the starting candidate.
            int_start: Chain position to start from.

        Returns:
            The first successful result.

        Raises:
            AiProviderRequestError: The last eligible failure when no
                candidate served the call, or the first ineligible one.
        """
        last_error: AiProviderRequestError | None = None
        list_indexes: list[int] = self._candidate_indexes(
            int_start=int_start, bool_allow_fallback=bool_allow_fallback
        )
        # Loop over the chain until a candidate serves the call.
        for int_index in list_indexes:
            client: AIBaseCompletions | None = self._client_at(int_index)
            if client is None or self._skip_for_capability(
                int_index, client, str_capability
            ):
                continue
            try:
                result: T = call(client)
            except AiProviderRequestError as error:
                if (
                    not bool_allow_fallback
                    or not self._eligible(error)
                    or int_index == list_indexes[-1]
                ):
                    raise
                self._log_failover(int_index, str_operation, error)
                last_error = error
                continue
            # Early return with the first result.
            return self._stamp_route(result, int_index)
        assert last_error is not None  # every path that gets here recorded one
        # Normal exit: every remaining candidate was skipped after a failure.
        raise last_error

    async def _arun(
        self,
        str_operation: str,
        call: Callable[[AIBaseCompletions], Awaitable[T]],
        *,
        str_capability: str | None = None,
        bool_allow_fallback: bool = True,
        int_start: int = 0,
    ) -> T:
        """Async twin of _run."""
        last_error: AiProviderRequestError | None = None
        list_indexes: list[int] = self._candidate_indexes(
            int_start=int_start, bool_allow_fallback=bool_allow_fallback
        )
        # Loop over the chain until a candidate serves the call.
        for int_index in list_indexes:
            client: AIBaseCompletions | None = self._client_at(int_index)
            if client is None or self._skip_for_capability(
                int_index, client, str_capability
            ):
                continue
            try:
                result: T = await call(client)
            except AiProviderRequestError as error:
                if (
                    not bool_allow_fallback
                    or not self._eligible(error)
                    or int_index == list_indexes[-1]
                ):
                    raise
                self._log_failover(int_index, str_operation, error)
                last_error = error
                continue
            # Early return with the first result.
            return self._stamp_route(result, int_index)
        assert last_error is not None  # every path that gets here recorded one
        # Normal exit: every remaining candidate was skipped after a failure.
        raise last_error

    @staticmethod
    def _history_is_engine_neutral(messages: list[dict[str, Any]]) -> bool:
        """
        Reports whether a conversation history can replay on any engine.

        Neutral history is user, assistant, and system messages whose
        content is plain text. Engine-shaped entries (replayed raw_content,
        tool results) bind the conversation to the engine that produced
        them.

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
            for message in messages
        )

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
            str_capability="supports_async",
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
        int_last: int = len(self._candidates) - 1
        # Loop over the chain until a candidate yields its first chunk.
        for int_index in range(len(self._candidates)):
            client: AIBaseCompletions | None = self._client_at(int_index)
            if client is None or self._skip_for_capability(
                int_index, client, "supports_streaming"
            ):
                continue
            iterator: Iterator[str] = client.send_prompt_streaming(
                prompt, other_params=other_params
            )
            try:
                str_first: str = next(iterator)
            except StopIteration:
                self._last_route_index = int_index
                # Early return: the candidate served an empty stream.
                return
            except AiProviderRequestError as error:
                if not self._eligible(error) or int_index == int_last:
                    raise
                self._log_failover(int_index, "send_prompt_streaming", error)
                last_error = error
                continue
            self._last_route_index = int_index
            yield str_first
            yield from iterator
            # Early return: the stream completed on this candidate.
            return
        assert last_error is not None  # every path that gets here recorded one
        # Normal exit: every remaining candidate was skipped after a failure.
        raise last_error

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
        """Runs send_structured_output, failing over across the chain."""
        bool_neutral: bool = self._history_is_engine_neutral(messages or [])
        # Normal return with the first candidate's structured result.
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
                provider_options=provider_options,
                prompt_cache=prompt_cache,
            ),
            str_capability="supports_structured_output",
            bool_allow_fallback=bool_neutral
            and self._fallback_allowed(provider_options),
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
        bool_neutral: bool = self._history_is_engine_neutral(messages or [])
        # Normal return with the first candidate's structured result.
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
                provider_options=provider_options,
                prompt_cache=prompt_cache,
            ),
            str_capability="supports_async",
            bool_allow_fallback=bool_neutral
            and self._fallback_allowed(provider_options),
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

        Once the history holds engine-shaped entries, the turn goes to the
        engine that served the previous turn and a failure there propagates.
        """
        bool_neutral: bool = self._history_is_engine_neutral(messages)
        int_start: int = 0 if bool_neutral else self._conversation_route_index
        turn: AITurnResult = self._run(
            "send_conversation",
            lambda client: client.send_conversation(
                system_prompt,
                messages,
                tools=tools,
                tool_choice=tool_choice,
                max_response_tokens=max_response_tokens,
                request_timeout_seconds=request_timeout_seconds,
                provider_options=provider_options,
                prompt_cache=prompt_cache,
            ),
            str_capability="supports_tool_use",
            bool_allow_fallback=bool_neutral
            and self._fallback_allowed(provider_options),
            int_start=int_start,
        )
        self._conversation_route_index = self._last_route_index
        # Normal return with the serving candidate's turn.
        return turn

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
        bool_neutral: bool = self._history_is_engine_neutral(messages)
        int_start: int = 0 if bool_neutral else self._conversation_route_index
        turn: AITurnResult = await self._arun(
            "asend_conversation",
            lambda client: client.asend_conversation(
                system_prompt,
                messages,
                tools=tools,
                tool_choice=tool_choice,
                max_response_tokens=max_response_tokens,
                request_timeout_seconds=request_timeout_seconds,
                provider_options=provider_options,
                prompt_cache=prompt_cache,
            ),
            str_capability="supports_async",
            bool_allow_fallback=bool_neutral
            and self._fallback_allowed(provider_options),
            int_start=int_start,
        )
        self._conversation_route_index = self._last_route_index
        # Normal return with the serving candidate's turn.
        return turn

    def _conversation_client(self) -> AIBaseCompletions:
        """The engine that served the most recent conversation turn."""
        client: AIBaseCompletions | None = self._client_at(
            self._conversation_route_index
        )
        # Normal return: a candidate that served a turn was built, so this
        # is only None before any turn, when it falls back to the primary.
        return client if client is not None else self.primary

    def build_tool_result_message(
        self,
        *,
        tool_call_id: str,
        result: dict[str, Any],
        is_error: bool = False,
    ) -> dict[str, Any]:
        """Builds the tool-result message in the serving engine's shape."""
        # Normal return with the engine-shaped tool result.
        return self._conversation_client().build_tool_result_message(
            tool_call_id=tool_call_id, result=result, is_error=is_error
        )

    def extend_messages_with_turn(
        self,
        messages: list[dict[str, Any]],
        turn: AITurnResult,
    ) -> list[dict[str, Any]]:
        """Appends the assistant turn in the serving engine's shape."""
        # Normal return with the extended history.
        return self._conversation_client().extend_messages_with_turn(messages, turn)

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
