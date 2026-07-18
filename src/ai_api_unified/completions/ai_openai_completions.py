# ai_openai_completions.py

import base64
import json
import logging
import time
from collections.abc import Iterator
from datetime import date
from typing import Any, ClassVar, Type

from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import ValidationError, model_validator

from ai_api_unified.ai_completions_exceptions import (
    StructuredResponseTokenLimitError,
)
from ai_api_unified.ai_openai_base import AIOpenAIBase

from ..ai_base import (
    AIBaseCompletions,
    AiApiObservedCompletionsResultModel,
    AIFinishReason,
    AIStructuredOutputResult,
    AIStructuredPrompt,
    AICompletionsCapabilitiesBase,
    AICompletionsPromptParamsBase,
    AITokenUsage,
    AITool,
    AIToolCall,
    AITurnResult,
    RETRY_POLICY_NONE,
    SupportedDataType,
)
from ..ai_provider_exceptions import AiProviderRequestError
from ..middleware.observability_runtime import (
    AiApiCallResultSummaryModel,
    ObservabilityMetadataValue,
)
from ..pricing.pricing_registry import (
    PROVIDER_OPENAI,
    enforce_model_lifecycle,
    get_model_pricing,
)


_LOGGER: logging.Logger = logging.getLogger(__name__)


class AICompletionsCapabilitiesOpenAI(AICompletionsCapabilitiesBase):
    """OpenAI-specific completions capabilities.

    Based on https://platform.openai.com/docs/models and public announcements.
    """

    _DEFAULT_CAPABILITIES: ClassVar[dict[str, Any]] = {
        "context_window_length": 128_000,
        "knowledge_cutoff_date": None,
        "reasoning": False,
        "supported_data_types": [
            SupportedDataType.TEXT,
            SupportedDataType.IMAGE,
            SupportedDataType.AUDIO,
        ],
        "supports_data_residency_constraint": True,
        # All supported chat models stream via chat.completions stream=True.
        "supports_streaming": True,
        # All catalogued chat models accept tools/tool_choice and the
        # json_schema response_format, and the SDK ships AsyncOpenAI.
        "supports_tool_use": True,
        "supports_structured_output": True,
        "supports_async": True,
    }

    # Context window sizes (max tokens each model can handle)
    DICT_OPENAI_CONTEXT_WINDOWS: ClassVar[dict[str, int]] = {
        # --- GPT-5.x current family (API) ---
        "gpt-5.4": 400_000,
        "gpt-5.1-codex-max": 400_000,
        # --- GPT-5 Family (API) ---
        "gpt-5": 400_000,
        "gpt-5-mini": 400_000,
        "gpt-5-nano": 400_000,
        # --- GPT-4.1 Family (API) ---
        "gpt-4.1": 1_000_000,
        "gpt-4.1-mini": 1_000_000,
        "gpt-4.1-nano": 1_000_000,
        # --- o4 Reasoning Series (API & Help Center) ---
        "o4-mini": 200_000,
        "o4-mini-high": 200_000,  # "high" is a reasoning effort mode, not a larger context model
        # --- GPT-4o Omni Series (launch blog) ---
        "gpt-4o": 128_000,
        "gpt-4o-mini": 128_000,
    }

    _OPENAI_FAMILY_PROPERTIES: ClassVar[dict[str, dict[str, Any]]] = {
        # Reasoning models with documented 400k total context; cutoff explicitly published.
        "gpt-5": {
            "reasoning": True,
            "knowledge_cutoff_date": date(2024, 9, 30),
        },
        # Non-reasoning models; cutoff stated as June 2024.
        "gpt-4.1": {
            "reasoning": False,
            "knowledge_cutoff_date": date(2024, 6, 1),
        },
        # Reasoning o-series; OpenAI docs discuss o4 as reasoning models. Cutoff date not explicitly
        # published in every source; we align with 2024-06 for consistency with adjacent releases.
        "o4": {
            "reasoning": True,
            "knowledge_cutoff_date": date(2024, 6, 1),
        },
        # Non-reasoning GPT-4o family; launch material cites October 2023 knowledge.
        "gpt-4o": {
            "reasoning": False,
            "knowledge_cutoff_date": date(2023, 10, 1),
        },
    }

    # Pricing now lives in the pricing registry (single source of truth), keyed
    # by (provider, model) with split input/output/cached rates and provenance.

    @classmethod
    def for_model(cls, model_name: str) -> "AICompletionsCapabilitiesOpenAI":
        """Create capabilities instance for the requested OpenAI model.

        Looks up the context window from DICT_OPENAI_CONTEXT_WINDOWS and applies
        family-level defaults (reasoning + knowledge cutoff) without duplicating logic.
        """
        normalized_name: str = model_name.strip().lower()
        capabilities: dict[str, Any] = dict(cls._DEFAULT_CAPABILITIES)

        # 1) Context window: direct lookup with safe fallback to default
        ctx_len: int | None = cls.DICT_OPENAI_CONTEXT_WINDOWS.get(normalized_name)
        if ctx_len is not None:
            capabilities["context_window_length"] = ctx_len

        # 2) Pricing from the central registry (single source of truth)
        capabilities["pricing"] = get_model_pricing(PROVIDER_OPENAI, normalized_name)

        # 3) Family properties: map by prefix to avoid duplicating per-model conditionals
        if normalized_name.startswith("gpt-5"):
            capabilities.update(cls._OPENAI_FAMILY_PROPERTIES["gpt-5"])
        elif normalized_name.startswith("gpt-4.1"):
            capabilities.update(cls._OPENAI_FAMILY_PROPERTIES["gpt-4.1"])
        elif normalized_name.startswith("o4"):
            capabilities.update(cls._OPENAI_FAMILY_PROPERTIES["o4"])
        elif normalized_name.startswith("gpt-4o"):
            capabilities.update(cls._OPENAI_FAMILY_PROPERTIES["gpt-4o"])
        # else: keep _DEFAULT_CAPABILITIES

        return cls(**capabilities)


class AICompletionsPromptParamsOpenAI(AICompletionsPromptParamsBase):
    """OpenAI-specific prompt parameters."""

    # this is an OpenAI-specific hint for image inputs, allowing for control of
    # the level of detail in the model's analysis of the image
    DETAIL_VALUE_AUTO: ClassVar[str] = "auto"
    DETAIL_VALUE_LOW: ClassVar[str] = "low"
    DETAIL_VALUE_HIGH: ClassVar[str] = "high"
    DETAIL_ALLOWED_VALUES: ClassVar[set[str]] = {
        DETAIL_VALUE_AUTO,
        DETAIL_VALUE_LOW,
        DETAIL_VALUE_HIGH,
    }

    detail_hints: list[str] | None = None

    @model_validator(mode="after")
    def _validate_detail_hints(self) -> "AICompletionsPromptParamsOpenAI":
        list_types: list[SupportedDataType] = list(self.included_types or [])

        if not list_types:
            if self.detail_hints not in (None, []):
                raise ValueError(
                    "detail_hints cannot be provided without included media items."
                )
            return self.model_copy(update={"detail_hints": None})

        detail_hints: list[str] = list(self.detail_hints or [])

        # If no hints provided, default all to 'auto'
        if not detail_hints:
            return self.model_copy(
                update={
                    "detail_hints": [
                        self.DETAIL_VALUE_AUTO for _ in range(len(list_types))
                    ]
                }
            )

        if len(detail_hints) != len(list_types):
            raise ValueError(
                "detail_hints must have the same length as included media attachments."
            )

        normalized_hints: list[str] = []
        for index, hint in enumerate(detail_hints):
            normalized_hint = hint.lower()
            if normalized_hint not in self.DETAIL_ALLOWED_VALUES:
                raise ValueError(
                    f"detail_hints[{index}] must be one of {sorted(self.DETAIL_ALLOWED_VALUES)}."
                )
            normalized_hints.append(normalized_hint)

        return self.model_copy(update={"detail_hints": normalized_hints})

    def get_detail_hint(self, index: int) -> str:
        """
        Return the OpenAI-specific detail hint for the given media index, falling
        back to 'auto' when unspecified.
        """

        if not self.detail_hints or index >= len(self.detail_hints):
            return self.DETAIL_VALUE_AUTO
        return self.detail_hints[index]


class AiOpenAICompletions(AIOpenAIBase, AIBaseCompletions):
    def __init__(self, model: str = "4o-mini", **kwargs: Any):
        """
        Initializes the AiOpenAICompletions class, setting the model and related configuration.

        Args:
            model (str): The embedding model to use.
        """
        AIOpenAIBase.__init__(self, **kwargs)
        explicit_model: str = model.strip() if model else ""
        self.completions_model = explicit_model or self.env.get_setting(
            "COMPLETIONS_MODEL_NAME", "gpt-4o-mini"
        )
        AIBaseCompletions.__init__(self, model=self.completions_model, **kwargs)
        self.model = self.completions_model
        enforce_model_lifecycle(PROVIDER_OPENAI, self.completions_model)
        self._capabilities: AICompletionsCapabilitiesOpenAI = (
            AICompletionsCapabilitiesOpenAI.for_model(self.completions_model)
        )

    @property
    def capabilities(self) -> AICompletionsCapabilitiesOpenAI:
        """Return the resolved capabilities for the current OpenAI model."""

        return self._capabilities

    @property
    def list_model_names(self) -> list[str]:
        # Updated as of Aug 2025 based on OpenAI announcements and docs
        return [
            # --- GPT-5.x current family ---
            "gpt-5.4",  # current workhorse
            "gpt-5.1-codex-max",  # coding-optimized, large context
            # --- GPT-5 Family (previous generation, still available) ---
            "gpt-5",  # flagship
            "gpt-5-mini",  # smaller, cheaper variant
            "gpt-5-nano",  # lowest cost, optimized for speed
            # --- GPT-4.1 Family (still supported, introduced Apr 2025) ---
            "gpt-4.1",  # high context (1M tokens), not deprecated yet
            "gpt-4.1-mini",  # lower-cost variant
            "gpt-4.1-nano",  # smallest variant
            # --- o4 Reasoning Series (still active) ---
            "o4-mini",  # reasoning-focused model
            "o4-mini-high",  # higher reasoning capacity variant
            # --- GPT-4o Omni Series (superseded by GPT-5 but still available) ---
            "gpt-4o",  # multimodal model (text, vision, audio)
            "gpt-4o-mini",  # cheaper multimodal variant
            # Note: OpenAI has not yet announced deprecation, but GPT-5 is the successor.
        ]

    @property
    def max_context_tokens(self) -> int:
        """
        Look up the OpenAI context window for the current model_name.
        Falls back to 0 if unknown (i.e. no guard will occur).
        """
        return AICompletionsCapabilitiesOpenAI.DICT_OPENAI_CONTEXT_WINDOWS.get(
            self.completions_model, 0
        )

    # ── Conversation, structured output, async, retries (2.15.0) ────────────

    # Engine token carried on typed request errors.
    PROVIDER_ENGINE_TOKEN: ClassVar[str] = "openai"

    DICT_FINISH_REASON_MAP: ClassVar[dict[str, AIFinishReason]] = {
        "stop": AIFinishReason.COMPLETE,
        "length": AIFinishReason.LENGTH,
        "tool_calls": AIFinishReason.TOOL_USE,
        "function_call": AIFinishReason.TOOL_USE,
        "content_filter": AIFinishReason.REFUSAL,
    }

    def _client_for_call(
        self,
        *,
        request_timeout_seconds: float | None = None,
        retry_policy_override: str | None = None,
    ) -> Any:
        """
        Returns the sync client, applying per-call timeout and retry options.
        """
        dict_options: dict[str, Any] = {}
        if request_timeout_seconds is not None:
            dict_options["timeout"] = request_timeout_seconds
        if retry_policy_override is not None:
            if retry_policy_override.strip().lower() == RETRY_POLICY_NONE:
                dict_options["max_retries"] = 0
        if not dict_options:
            # Early return with the shared client when no per-call options apply.
            return self.client
        # Normal return with a per-call options view of the shared client.
        return self.client.with_options(**dict_options)

    def _async_client_for_call(
        self,
        *,
        request_timeout_seconds: float | None = None,
        retry_policy_override: str | None = None,
    ) -> Any:
        """
        Async twin of _client_for_call.
        """
        dict_options: dict[str, Any] = {}
        if request_timeout_seconds is not None:
            dict_options["timeout"] = request_timeout_seconds
        if retry_policy_override is not None:
            if retry_policy_override.strip().lower() == RETRY_POLICY_NONE:
                dict_options["max_retries"] = 0
        if not dict_options:
            # Early return with the shared async client when no per-call options apply.
            return self.async_client
        # Normal return with a per-call options view of the shared async client.
        return self.async_client.with_options(**dict_options)

    def _raise_request_error(self, exception: Exception) -> None:
        """
        Re-raises one OpenAI SDK transport error as the typed request error.

        Carries the HTTP status code (when available) so caller-owned backoff
        can classify 429/5xx uniformly across engines. Non-transport
        exceptions propagate unchanged.
        """
        if isinstance(exception, APIStatusError):
            raise AiProviderRequestError(
                f"OpenAI request failed with status "
                f"{exception.status_code}: {exception.message}",
                status_code=exception.status_code,
                provider_engine=self.PROVIDER_ENGINE_TOKEN,
            ) from exception
        if isinstance(exception, (APITimeoutError, APIConnectionError)):
            raise AiProviderRequestError(
                f"OpenAI request failed before a status was available: " f"{exception}",
                status_code=None,
                provider_engine=self.PROVIDER_ENGINE_TOKEN,
            ) from exception
        # Normal return so non-transport exceptions propagate unchanged.
        return None

    @staticmethod
    def _build_chat_provider_tools(list_tools: list[AITool]) -> list[dict[str, Any]]:
        """
        Maps provider-neutral AITool definitions onto Chat Completions tools.
        """
        list_provider_tools: list[dict[str, Any]] = []
        # Loop over tool definitions so each maps to the chat function shape.
        for ai_tool in list_tools:
            dict_function: dict[str, Any] = {
                "name": ai_tool.name,
                "description": ai_tool.description,
                "parameters": ai_tool.input_schema,
            }
            if ai_tool.strict:
                dict_function["strict"] = True
            list_provider_tools.append({"type": "function", "function": dict_function})
        # Normal return with the chat-shaped tool list.
        return list_provider_tools

    def _usage_from_chat_response(self, response: Any) -> AITokenUsage:
        """
        Builds the provider-neutral usage model from one chat completion.
        """
        # Normal return with the provider-neutral token usage model.
        return AITokenUsage(
            input_tokens=self._extract_openai_prompt_tokens(response),
            output_tokens=self._extract_openai_completion_tokens(response),
            cached_input_tokens=self._extract_openai_cached_tokens(response),
            total_tokens=self._extract_openai_total_tokens(response),
        )

    @staticmethod
    def _serialize_chat_assistant_message(message: Any) -> dict[str, Any]:
        """
        Serializes one chat assistant message into a replayable dictionary.

        Uses the SDK model_dump when available; otherwise reconstructs the
        message from typed attributes so raw_content stays replayable as the
        next assistant message.
        """
        model_dump = getattr(message, "model_dump", None)
        if callable(model_dump):
            try:
                dict_message: Any = model_dump(exclude_none=True)
                if isinstance(dict_message, dict):
                    # Early return with the SDK-serialized assistant message.
                    return dict_message
            except TypeError:
                # Fall through to attribute-based reconstruction (test doubles).
                pass
        dict_rebuilt: dict[str, Any] = {"role": "assistant"}
        str_content: Any = getattr(message, "content", None)
        if str_content is not None:
            dict_rebuilt["content"] = str_content
        list_tool_calls: list[dict[str, Any]] = []
        # Loop over tool calls so replayed messages carry the full call shape.
        for tool_call in getattr(message, "tool_calls", None) or []:
            function = getattr(tool_call, "function", None)
            list_tool_calls.append(
                {
                    "id": str(getattr(tool_call, "id", "") or ""),
                    "type": "function",
                    "function": {
                        "name": str(getattr(function, "name", "") or ""),
                        "arguments": str(getattr(function, "arguments", "") or "{}"),
                    },
                }
            )
        if list_tool_calls:
            dict_rebuilt["tool_calls"] = list_tool_calls
        # Normal return with the reconstructed assistant message.
        return dict_rebuilt

    def _build_turn_result_from_chat(self, response: Any) -> AITurnResult:
        """
        Maps one Chat Completions response onto the provider-neutral turn.
        """
        choice = response.choices[0]
        message = choice.message
        list_tool_calls: list[AIToolCall] = []
        # Loop over tool calls so callers receive parsed inputs per call.
        for tool_call in getattr(message, "tool_calls", None) or []:
            function = getattr(tool_call, "function", None)
            str_arguments: str = str(getattr(function, "arguments", "") or "{}")
            try:
                dict_input: dict[str, Any] = json.loads(str_arguments)
            except json.JSONDecodeError as json_error:
                raise ValueError(
                    f"Tool call arguments were not valid JSON: {json_error}"
                ) from json_error
            list_tool_calls.append(
                AIToolCall(
                    id=str(getattr(tool_call, "id", "") or ""),
                    name=str(getattr(function, "name", "") or ""),
                    input=dict_input,
                )
            )
        str_finish_reason: str | None = getattr(choice, "finish_reason", None)
        finish_reason: AIFinishReason = self._normalize_finish_reason(
            str(str_finish_reason) if str_finish_reason is not None else None,
            self.DICT_FINISH_REASON_MAP,
        )
        # Present tool calls are authoritative: a forced tool_choice reports
        # finish_reason "stop" even though the message carries tool_calls.
        if list_tool_calls:
            finish_reason = AIFinishReason.TOOL_USE
        # The refusal field is authoritative even when finish_reason is stop.
        if getattr(message, "refusal", None):
            finish_reason = AIFinishReason.REFUSAL
        str_text: str | None = getattr(message, "content", None) or None
        # Normal return with the provider-neutral turn result.
        return AITurnResult(
            text=str_text,
            tool_calls=list_tool_calls,
            finish_reason=finish_reason,
            raw_content=self._serialize_chat_assistant_message(message),
            usage=self._usage_from_chat_response(response),
        )

    def _observed_chat_turn_result(
        self, response: Any
    ) -> AiApiObservedCompletionsResultModel[AITurnResult]:
        """
        Wraps one chat response as an observed conversation-turn result.
        """
        turn_result: AITurnResult = self._build_turn_result_from_chat(response)
        str_finish_reason: Any = getattr(response.choices[0], "finish_reason", None)
        # Normal return with the observed turn result and usage metadata.
        return AiApiObservedCompletionsResultModel(
            return_value=turn_result,
            raw_output_text=turn_result.text or "",
            finish_reason=str(str_finish_reason) if str_finish_reason else None,
            provider_prompt_tokens=turn_result.usage.input_tokens,
            provider_completion_tokens=turn_result.usage.output_tokens,
            provider_cached_input_tokens=turn_result.usage.cached_input_tokens,
            provider_total_tokens=turn_result.usage.total_tokens,
            dict_metadata={"tool_call_count": len(turn_result.tool_calls)},
        )

    def _build_chat_conversation_request_kwargs(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[AITool],
        tool_choice: str | None,
        max_response_tokens: int | None,
        dict_merge_options: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Builds the Chat Completions request kwargs for one conversation turn.
        """
        dict_request_kwargs: dict[str, Any] = {
            "model": self.completions_model,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
        }
        if max_response_tokens is not None:
            dict_request_kwargs["max_completion_tokens"] = max_response_tokens
        if tools:
            dict_request_kwargs["tools"] = self._build_chat_provider_tools(tools)
        if tool_choice is not None:
            dict_request_kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice},
            }
        # provider_options merge last so callers can extend the raw request.
        dict_request_kwargs.update(dict_merge_options)
        # Normal return with the chat-shaped conversation request.
        return dict_request_kwargs

    def _build_conversation_observability_metadata(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[AITool],
        tool_choice: str | None,
        max_response_tokens: int | None,
    ) -> dict[str, ObservabilityMetadataValue]:
        """
        Builds metadata-only observability fields for one conversation turn.
        """
        dict_metadata: dict[str, ObservabilityMetadataValue] = {
            "response_mode": self.RESPONSE_MODE_TEXT,
            "message_count": len(messages),
            "tool_count": len(tools),
            "system_prompt_char_count": len(system_prompt),
        }
        if tool_choice is not None:
            dict_metadata["forced_tool"] = tool_choice
        if max_response_tokens is not None:
            dict_metadata["max_response_tokens"] = max_response_tokens
        # Normal return with scalar conversation metadata.
        return dict_metadata

    def _send_conversation_provider(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[AITool],
        tool_choice: str | None,
        max_response_tokens: int | None,
        request_timeout_seconds: float | None,
        provider_options: dict[str, Any] | None,
    ) -> AITurnResult:
        """
        Sends one conversation turn to the Chat Completions API.
        """
        dict_merge_options, str_retry_override = self._split_provider_options(
            provider_options
        )
        call_client: Any = self._client_for_call(
            request_timeout_seconds=request_timeout_seconds,
            retry_policy_override=str_retry_override,
        )
        dict_request_kwargs: dict[str, Any] = (
            self._build_chat_conversation_request_kwargs(
                system_prompt=system_prompt,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_response_tokens=max_response_tokens,
                dict_merge_options=dict_merge_options,
            )
        )
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = (
            self._build_conversation_observability_metadata(
                system_prompt=system_prompt,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_response_tokens=max_response_tokens,
            )
        )

        def _execute_turn() -> AiApiObservedCompletionsResultModel[AITurnResult]:
            try:
                response = call_client.chat.completions.create(**dict_request_kwargs)
            except Exception as exception:
                self._raise_request_error(exception)
                raise
            # Normal return with the observed conversation turn.
            return self._observed_chat_turn_result(response)

        observed_result: AiApiObservedCompletionsResultModel[AITurnResult] = (
            self._execute_provider_call_with_observability(
                capability=self.CLIENT_TYPE_COMPLETIONS,
                operation="send_conversation",
                dict_input_metadata=dict_input_metadata,
                callable_execute=_execute_turn,
                callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_completions_observability_result_summary(
                    observed_result=result,
                    provider_elapsed_ms=provider_elapsed_ms,
                ),
                legacy_caller_id=self.user,
            )
        )
        # Normal return with the caller-facing conversation turn.
        return observed_result.return_value

    async def _asend_conversation_provider(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[AITool],
        tool_choice: str | None,
        max_response_tokens: int | None,
        request_timeout_seconds: float | None,
        provider_options: dict[str, Any] | None,
    ) -> AITurnResult:
        """
        Async twin of _send_conversation_provider using AsyncOpenAI.
        """
        dict_merge_options, str_retry_override = self._split_provider_options(
            provider_options
        )
        call_client: Any = self._async_client_for_call(
            request_timeout_seconds=request_timeout_seconds,
            retry_policy_override=str_retry_override,
        )
        dict_request_kwargs: dict[str, Any] = (
            self._build_chat_conversation_request_kwargs(
                system_prompt=system_prompt,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_response_tokens=max_response_tokens,
                dict_merge_options=dict_merge_options,
            )
        )
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = (
            self._build_conversation_observability_metadata(
                system_prompt=system_prompt,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_response_tokens=max_response_tokens,
            )
        )

        async def _execute_turn() -> AiApiObservedCompletionsResultModel[AITurnResult]:
            try:
                response = await call_client.chat.completions.create(
                    **dict_request_kwargs
                )
            except Exception as exception:
                self._raise_request_error(exception)
                raise
            # Normal return with the observed conversation turn.
            return self._observed_chat_turn_result(response)

        observed_result: AiApiObservedCompletionsResultModel[AITurnResult] = (
            await self._execute_provider_acall_with_observability(
                capability=self.CLIENT_TYPE_COMPLETIONS,
                operation="asend_conversation",
                dict_input_metadata=dict_input_metadata,
                callable_execute=_execute_turn,
                callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_completions_observability_result_summary(
                    observed_result=result,
                    provider_elapsed_ms=provider_elapsed_ms,
                ),
                legacy_caller_id=self.user,
            )
        )
        # Normal return with the caller-facing conversation turn.
        return observed_result.return_value

    def _build_tool_result_message_provider(
        self,
        *,
        tool_call_id: str,
        result: dict[str, Any],
        is_error: bool,
    ) -> dict[str, Any]:
        """
        Builds one Chat Completions tool-role result message.

        The chat wire format has no error flag on tool messages, so failures
        are encoded inside the JSON payload under an "error" key.
        """
        dict_payload: dict[str, Any] = {"error": result} if is_error else result
        # Normal return with the chat-shaped tool-result entry.
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(dict_payload),
        }

    def _extend_messages_with_turn_provider(
        self,
        *,
        messages: list[dict[str, Any]],
        turn: AITurnResult,
    ) -> None:
        """
        Appends one chat assistant message; raw_content is the full message.
        """
        messages.append(turn.raw_content)
        # Normal return after appending the assistant turn.
        return None

    def _build_structured_observability_metadata(
        self,
        *,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        max_response_tokens: int,
    ) -> dict[str, ObservabilityMetadataValue]:
        """
        Builds metadata-only observability fields for one structured call.
        """
        # Normal return with scalar structured-call metadata.
        return {
            "response_mode": self.RESPONSE_MODE_STRUCTURED,
            "prompt_char_count": len(prompt) if prompt else 0,
            "message_count": len(messages) if messages else 0,
            "max_response_tokens": max_response_tokens,
        }

    def _build_structured_output_result_from_parts(
        self,
        *,
        raw_output_text: str,
        finish_reason: AIFinishReason,
        usage: AITokenUsage,
    ) -> AIStructuredOutputResult:
        """
        Maps structured response parts onto the provider-neutral result model.
        """
        if finish_reason in (AIFinishReason.LENGTH, AIFinishReason.REFUSAL):
            # Early return so callers branch on finish_reason instead of parsing.
            return AIStructuredOutputResult(
                data=None,
                finish_reason=finish_reason,
                usage=usage,
                raw_text=self.pii_middleware.process_output(raw_output_text),
            )
        str_content: str = self.pii_middleware.process_output(raw_output_text)
        if not str_content:
            raise ValueError("Empty response from OpenAI API")
        try:
            parsed_data: Any = json.loads(str_content)
        except json.JSONDecodeError as json_error:
            raise ValueError(f"Invalid JSON response: {json_error}") from json_error
        if not isinstance(parsed_data, dict):
            raise ValueError(
                "Structured response was valid JSON but not a JSON object."
            )
        # Normal return with the parsed structured output result.
        return AIStructuredOutputResult(
            data=parsed_data,
            finish_reason=finish_reason,
            usage=usage,
            raw_text=str_content,
        )

    def _send_structured_output_provider(
        self,
        *,
        response_schema: dict[str, Any],
        system_prompt: str | None,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        max_response_tokens: int,
        request_timeout_seconds: float | None,
        provider_options: dict[str, Any] | None,
    ) -> AIStructuredOutputResult:
        """
        Generates structured output via the chat json_schema response format.

        Runs in schema-guided mode (strict: false), matching the engine's
        existing strict_schema_prompt behavior; parsed output is validated
        client-side by the base template.
        """
        dict_merge_options, str_retry_override = self._split_provider_options(
            provider_options
        )
        call_client: Any = self._client_for_call(
            request_timeout_seconds=request_timeout_seconds,
            retry_policy_override=str_retry_override,
        )
        dict_request_kwargs: dict[str, Any] = (
            self._build_chat_structured_request_kwargs(
                response_schema=response_schema,
                system_prompt=system_prompt,
                prompt=prompt,
                messages=messages,
                max_response_tokens=max_response_tokens,
                dict_merge_options=dict_merge_options,
            )
        )
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = (
            self._build_structured_observability_metadata(
                prompt=prompt,
                messages=messages,
                max_response_tokens=max_response_tokens,
            )
        )

        def _execute_structured() -> (
            AiApiObservedCompletionsResultModel[AIStructuredOutputResult]
        ):
            try:
                response = call_client.chat.completions.create(**dict_request_kwargs)
            except Exception as exception:
                self._raise_request_error(exception)
                raise
            # Normal return with the observed structured output result.
            return self._observed_chat_structured_result(response)

        observed_result: AiApiObservedCompletionsResultModel[
            AIStructuredOutputResult
        ] = self._execute_provider_call_with_observability(
            capability=self.CLIENT_TYPE_COMPLETIONS,
            operation="send_structured_output",
            dict_input_metadata=dict_input_metadata,
            callable_execute=_execute_structured,
            callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_completions_observability_result_summary(
                observed_result=result,
                provider_elapsed_ms=provider_elapsed_ms,
            ),
            legacy_caller_id=self.user,
        )
        # Normal return with the caller-facing structured output result.
        return observed_result.return_value

    async def _asend_structured_output_provider(
        self,
        *,
        response_schema: dict[str, Any],
        system_prompt: str | None,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        max_response_tokens: int,
        request_timeout_seconds: float | None,
        provider_options: dict[str, Any] | None,
    ) -> AIStructuredOutputResult:
        """
        Async twin of _send_structured_output_provider.
        """
        dict_merge_options, str_retry_override = self._split_provider_options(
            provider_options
        )
        call_client: Any = self._async_client_for_call(
            request_timeout_seconds=request_timeout_seconds,
            retry_policy_override=str_retry_override,
        )
        dict_request_kwargs: dict[str, Any] = (
            self._build_chat_structured_request_kwargs(
                response_schema=response_schema,
                system_prompt=system_prompt,
                prompt=prompt,
                messages=messages,
                max_response_tokens=max_response_tokens,
                dict_merge_options=dict_merge_options,
            )
        )
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = (
            self._build_structured_observability_metadata(
                prompt=prompt,
                messages=messages,
                max_response_tokens=max_response_tokens,
            )
        )

        async def _execute_structured() -> (
            AiApiObservedCompletionsResultModel[AIStructuredOutputResult]
        ):
            try:
                response = await call_client.chat.completions.create(
                    **dict_request_kwargs
                )
            except Exception as exception:
                self._raise_request_error(exception)
                raise
            # Normal return with the observed structured output result.
            return self._observed_chat_structured_result(response)

        observed_result: AiApiObservedCompletionsResultModel[
            AIStructuredOutputResult
        ] = await self._execute_provider_acall_with_observability(
            capability=self.CLIENT_TYPE_COMPLETIONS,
            operation="asend_structured_output",
            dict_input_metadata=dict_input_metadata,
            callable_execute=_execute_structured,
            callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_completions_observability_result_summary(
                observed_result=result,
                provider_elapsed_ms=provider_elapsed_ms,
            ),
            legacy_caller_id=self.user,
        )
        # Normal return with the caller-facing structured output result.
        return observed_result.return_value

    def _build_chat_structured_request_kwargs(
        self,
        *,
        response_schema: dict[str, Any],
        system_prompt: str | None,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        max_response_tokens: int,
        dict_merge_options: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Builds the Chat Completions request kwargs for one structured call.
        """
        str_system_prompt: str = self._resolve_system_prompt(
            system_prompt,
            None,
            AICompletionsPromptParamsBase.DEFAULT_STRICT_SCHEMA_SYSTEM_PROMPT,
        )
        list_messages: list[dict[str, Any]] = [
            {"role": "system", "content": str_system_prompt},
            *(messages or []),
        ]
        if prompt is not None and prompt.strip():
            str_redacted_prompt: str = self.pii_middleware.process_input(prompt)
            list_messages.append({"role": "user", "content": str_redacted_prompt})
        dict_request_kwargs: dict[str, Any] = {
            "model": self.completions_model,
            "messages": list_messages,
            "max_completion_tokens": max_response_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": response_schema,
                    "strict": False,
                },
            },
        }
        dict_request_kwargs.update(dict_merge_options)
        # Normal return with the chat-shaped structured request.
        return dict_request_kwargs

    def _observed_chat_structured_result(
        self, response: Any
    ) -> AiApiObservedCompletionsResultModel[AIStructuredOutputResult]:
        """
        Wraps one chat structured response as an observed result.
        """
        choice = response.choices[0]
        str_finish_reason: Any = getattr(choice, "finish_reason", None)
        finish_reason: AIFinishReason = self._normalize_finish_reason(
            str(str_finish_reason) if str_finish_reason is not None else None,
            self.DICT_FINISH_REASON_MAP,
        )
        if getattr(choice.message, "refusal", None):
            finish_reason = AIFinishReason.REFUSAL
        structured_result: AIStructuredOutputResult = (
            self._build_structured_output_result_from_parts(
                raw_output_text=getattr(choice.message, "content", None) or "",
                finish_reason=finish_reason,
                usage=self._usage_from_chat_response(response),
            )
        )
        # Normal return with the observed structured output result.
        return AiApiObservedCompletionsResultModel(
            return_value=structured_result,
            raw_output_text=structured_result.raw_text,
            finish_reason=str(str_finish_reason) if str_finish_reason else None,
            provider_prompt_tokens=structured_result.usage.input_tokens,
            provider_completion_tokens=structured_result.usage.output_tokens,
            provider_cached_input_tokens=structured_result.usage.cached_input_tokens,
            provider_total_tokens=structured_result.usage.total_tokens,
        )

    async def _asend_prompt_provider(
        self,
        prompt: str,
        *,
        system_prompt: str | None,
        max_response_tokens: int | None,
        request_timeout_seconds: float | None,
        other_params: AICompletionsPromptParamsBase | None,
    ) -> str:
        """
        Async twin of send_prompt using AsyncOpenAI (single-shot, no
        auto-continue loop).
        """
        str_redacted_prompt: str = self.pii_middleware.process_input(prompt)
        str_system_prompt: str = self._resolve_system_prompt(
            system_prompt,
            other_params,
            AICompletionsPromptParamsBase.DEFAULT_SYSTEM_PROMPT,
        )
        call_client: Any = self._async_client_for_call(
            request_timeout_seconds=request_timeout_seconds
        )
        user_content = self._build_user_message_content(
            str_redacted_prompt, other_params
        )
        dict_request_kwargs: dict[str, Any] = {
            "model": self.completions_model,
            "messages": [
                {"role": "system", "content": str_system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        if max_response_tokens is not None:
            dict_request_kwargs["max_completion_tokens"] = max_response_tokens
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = (
            self._build_completions_observability_input_metadata(
                prompt=str_redacted_prompt,
                system_prompt=str_system_prompt,
                other_params=other_params,
                response_mode=self.RESPONSE_MODE_TEXT,
                max_response_tokens=max_response_tokens,
            )
        )

        async def _execute_text_prompt() -> AiApiObservedCompletionsResultModel[str]:
            try:
                response = await call_client.chat.completions.create(
                    **dict_request_kwargs
                )
            except Exception as exception:
                self._raise_request_error(exception)
                raise
            raw_output_text: str = response.choices[0].message.content or ""
            # Normal return with the chat text output and usage metadata.
            return AiApiObservedCompletionsResultModel(
                return_value=raw_output_text,
                raw_output_text=raw_output_text,
                finish_reason=str(response.choices[0].finish_reason or ""),
                provider_prompt_tokens=self._extract_openai_prompt_tokens(response),
                provider_completion_tokens=self._extract_openai_completion_tokens(
                    response
                ),
                provider_cached_input_tokens=self._extract_openai_cached_tokens(
                    response
                ),
                provider_total_tokens=self._extract_openai_total_tokens(response),
            )

        observed_result: AiApiObservedCompletionsResultModel[str] = (
            await self._execute_provider_acall_with_observability(
                capability=self.CLIENT_TYPE_COMPLETIONS,
                operation="asend_prompt",
                dict_input_metadata=dict_input_metadata,
                callable_execute=_execute_text_prompt,
                callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_completions_observability_result_summary(
                    observed_result=result,
                    provider_elapsed_ms=provider_elapsed_ms,
                ),
                legacy_caller_id=self.user,
            )
        )
        # Normal return with sanitized chat text after observability wrapping.
        return self.pii_middleware.process_output(observed_result.return_value)

    def _build_user_message_content(
        self,
        prompt: str,
        other_params: AICompletionsPromptParamsBase | None,
    ) -> str | list[dict[str, Any]]:
        """
        Construct OpenAI message content that supports multimodal inputs.
        Returns a plain string for text-only prompts to preserve backwards
        compatibility with earlier provider implementations.
        """

        if other_params is None or not other_params.has_included_media:
            return prompt

        content_parts: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
        ]

        for (
            index,
            media_type,
            media_bytes,
            mime_type,
        ) in other_params.iter_included_media():
            if media_type is not SupportedDataType.IMAGE:
                continue

            if len(media_bytes) > AICompletionsPromptParamsBase.MAX_IMAGE_BYTES:
                raise ValueError(
                    f"Image attachment {index} exceeds the maximum allowed size of "
                    f"{AICompletionsPromptParamsBase.MAX_IMAGE_BYTES} bytes."
                )

            encoded_bytes: str = base64.b64encode(media_bytes).decode("ascii")
            image_payload: dict[str, Any] = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{encoded_bytes}",
                },
            }

            if isinstance(other_params, AICompletionsPromptParamsOpenAI):
                detail_hint: str = other_params.get_detail_hint(index)
                if detail_hint != AICompletionsPromptParamsOpenAI.DETAIL_VALUE_AUTO:
                    image_payload["image_url"]["detail"] = detail_hint

            content_parts.append(image_payload)

        if len(content_parts) == 1:
            return prompt

        return content_parts

    def strict_schema_prompt(
        self,
        prompt: str,
        response_model: Type[AIStructuredPrompt],
        max_response_tokens: int = AIBaseCompletions.STRUCTURED_DEFAULT_MAX_RESPONSE_TOKENS,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> AIStructuredPrompt:
        """
        Sends a prompt to the OpenAI API using function calling to enforce a JSON schema
        and parses the response into the specified Pydantic model.

        Args:
            prompt (str): The prompt string to send.
            response_model (Type[PromptResult]): The Pydantic model to parse the response into.
            max_response_tokens (int): The maximum number of response tokens requested.
            other_params: Optional provider-specific parameters, including custom system prompt.

        Returns:
            PromptResult: An instance of the specified Pydantic model containing the parsed response.
        """
        prompt = self.pii_middleware.process_input(prompt)
        self._validate_structured_max_response_tokens(
            provider_name=self.PROVIDER_VENDOR_OPENAI,
            model_name=self.completions_model,
            max_response_tokens=max_response_tokens,
        )

        # Include a brief system instruction to nudge the model toward JSON-only output
        system_prompt: str = (
            AICompletionsPromptParamsBase.DEFAULT_STRICT_SCHEMA_SYSTEM_PROMPT
        )
        if other_params is not None and other_params.system_prompt is not None:
            system_prompt = other_params.system_prompt
        user_content = self._build_user_message_content(prompt, other_params)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Define a dummy “function” whose parameters are your JSON schema
        functions = [
            {
                "name": "strict_schema_response",
                "description": "Enforce the given JSON schema in the response.",
                "parameters": response_model.model_json_schema(),
            }
        ]

        max_retries = 3
        retry_delay = 2
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = (
            self._build_completions_observability_input_metadata(
                prompt=prompt,
                system_prompt=system_prompt,
                other_params=other_params,
                response_mode=self.RESPONSE_MODE_STRUCTURED,
                max_response_tokens=max_response_tokens,
            )
        )

        def _execute_structured_prompt() -> (
            AiApiObservedCompletionsResultModel[AIStructuredPrompt]
        ):
            current_retry_delay: int = retry_delay
            # Loop through structured-prompt retry attempts while preserving existing backoff behavior.
            for attempt in range(max_retries):
                try:
                    completion = self.client.chat.completions.create(
                        model=self.completions_model,
                        messages=messages,
                        functions=functions,
                        function_call={"name": "strict_schema_response"},
                        max_completion_tokens=max_response_tokens,
                    )
                    str_finish_reason: str = str(completion.choices[0].finish_reason)
                    choice_msg = completion.choices[0].message

                    if choice_msg.function_call and choice_msg.function_call.arguments:
                        raw_output_text: str = choice_msg.function_call.arguments
                    else:
                        raw_output_text = choice_msg.content or ""

                    if str_finish_reason == "length":
                        self._raise_structured_token_limit_error(
                            provider_name=self.PROVIDER_VENDOR_OPENAI,
                            model_name=self.completions_model,
                            max_response_tokens=max_response_tokens,
                            finish_reason=str_finish_reason,
                            raw_output_text=raw_output_text,
                        )

                    content_str: str = self.pii_middleware.process_output(
                        raw_output_text
                    )
                    parsed_json: dict[str, Any] = json.loads(content_str)
                    validated_response: AIStructuredPrompt = (
                        response_model.model_validate(parsed_json)
                    )
                    observed_result: AiApiObservedCompletionsResultModel[
                        AIStructuredPrompt
                    ] = AiApiObservedCompletionsResultModel(
                        return_value=validated_response,
                        raw_output_text=raw_output_text,
                        finish_reason=str_finish_reason,
                        provider_prompt_tokens=self._extract_openai_prompt_tokens(
                            completion
                        ),
                        provider_completion_tokens=self._extract_openai_completion_tokens(
                            completion
                        ),
                        provider_cached_input_tokens=self._extract_openai_cached_tokens(
                            completion
                        ),
                        provider_total_tokens=self._extract_openai_total_tokens(
                            completion
                        ),
                    )
                    # Normal return with validated structured output and raw provider metadata.
                    return observed_result

                except ValidationError as validation_error:
                    _LOGGER.exception(
                        "Validation error in strict_schema_prompt: %s",
                        validation_error,
                    )
                    raise

                except StructuredResponseTokenLimitError:
                    raise

                except Exception as exception:
                    if "MAX_TOKENS" in str(exception).upper():
                        _LOGGER.error(
                            "OpenAI structured prompt failed due to MAX_TOKENS for model=%s with max_response_tokens=%s.",
                            self.completions_model,
                            max_response_tokens,
                        )
                    if attempt < max_retries - 1:
                        time.sleep(current_retry_delay)
                        current_retry_delay *= 2
                        continue
                    raise RuntimeError(
                        f"strict_schema_prompt failed after {max_retries} attempts: {exception}"
                    ) from exception

            raise RuntimeError("strict_schema_prompt exhausted retries unexpectedly.")

        observed_result: AiApiObservedCompletionsResultModel[AIStructuredPrompt] = (
            self._execute_provider_call_with_observability(
                capability=self.CLIENT_TYPE_COMPLETIONS,
                operation="strict_schema_prompt",
                dict_input_metadata=dict_input_metadata,
                callable_execute=_execute_structured_prompt,
                callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_completions_observability_result_summary(
                    observed_result=result,
                    provider_elapsed_ms=provider_elapsed_ms,
                ),
                legacy_caller_id=self.user,
            )
        )
        # Normal return with validated structured output after observability wrapping.
        return observed_result.return_value

    def send_prompt(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_response_tokens: int | None = None,
        request_timeout_seconds: float | None = None,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> str:
        """
        Sends a prompt to the latest version of the OpenAI API for chat and returns the completion result.

        Args:
            prompt (str): The prompt string to send.
            system_prompt: Optional persistent instructions; overrides
                other_params.system_prompt when both are supplied.
            max_response_tokens: Optional response token budget; maps to the
                chat max_completion_tokens field. Supplying it also disables
                the legacy auto-continue-on-length loop so the budget holds.
            request_timeout_seconds: Optional per-call timeout; maps to the
                OpenAI SDK request timeout.
            other_params: Optional provider-specific parameters (not used for OpenAI currently)

        Returns:
            str: The completion result as a string.
        """
        call_client: Any = self._client_for_call(
            request_timeout_seconds=request_timeout_seconds
        )
        dict_token_kwargs: dict[str, Any] = (
            {"max_completion_tokens": max_response_tokens}
            if max_response_tokens is not None
            else {}
        )
        try:
            prompt = self.pii_middleware.process_input(prompt)

            system_prompt = self._resolve_system_prompt(
                system_prompt,
                other_params,
                AICompletionsPromptParamsBase.DEFAULT_SYSTEM_PROMPT,
            )

            user_content = self._build_user_message_content(prompt, other_params)
            dict_input_metadata: dict[str, ObservabilityMetadataValue] = (
                self._build_completions_observability_input_metadata(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    other_params=other_params,
                    response_mode=self.RESPONSE_MODE_TEXT,
                    max_response_tokens=max_response_tokens,
                )
            )

            def _execute_text_prompt() -> AiApiObservedCompletionsResultModel[str]:
                try:
                    response = call_client.chat.completions.create(
                        model=self.completions_model,
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {"role": "user", "content": user_content},
                        ],
                        **dict_token_kwargs,
                    )
                except Exception as exception:
                    self._raise_request_error(exception)
                    raise

                raw_output_text: str = response.choices[0].message.content or ""
                int_provider_prompt_tokens: int | None = (
                    self._extract_openai_prompt_tokens(response)
                )
                int_provider_completion_tokens: int | None = (
                    self._extract_openai_completion_tokens(response)
                )
                int_provider_total_tokens: int | None = (
                    self._extract_openai_total_tokens(response)
                )
                int_provider_cached_tokens: int | None = (
                    self._extract_openai_cached_tokens(response)
                )
                bool_continued_response: bool = False

                # An explicit caller budget disables the legacy auto-continue
                # loop; continuing past "length" would defeat the budget.
                while (
                    max_response_tokens is None
                    and response.choices[0].finish_reason == "length"
                ):
                    bool_continued_response = True
                    response = call_client.chat.completions.create(
                        model=self.completions_model,
                        messages=[
                            {"role": "system", "content": "Continue."},
                        ],
                    )
                    raw_output_text += response.choices[0].message.content or ""
                    int_provider_prompt_tokens = self._sum_optional_ints(
                        int_provider_prompt_tokens,
                        self._extract_openai_prompt_tokens(response),
                    )
                    int_provider_completion_tokens = self._sum_optional_ints(
                        int_provider_completion_tokens,
                        self._extract_openai_completion_tokens(response),
                    )
                    int_provider_total_tokens = self._sum_optional_ints(
                        int_provider_total_tokens,
                        self._extract_openai_total_tokens(response),
                    )
                    int_provider_cached_tokens = self._sum_optional_ints(
                        int_provider_cached_tokens,
                        self._extract_openai_cached_tokens(response),
                    )

                observed_result: AiApiObservedCompletionsResultModel[str] = (
                    AiApiObservedCompletionsResultModel(
                        return_value=raw_output_text,
                        raw_output_text=raw_output_text,
                        finish_reason=str(response.choices[0].finish_reason),
                        provider_prompt_tokens=int_provider_prompt_tokens,
                        provider_completion_tokens=int_provider_completion_tokens,
                        provider_cached_input_tokens=int_provider_cached_tokens,
                        provider_total_tokens=int_provider_total_tokens,
                        dict_metadata={"continued_response": bool_continued_response},
                    )
                )
                # Normal return with raw completion text and aggregated provider usage metadata.
                return observed_result

            observed_result: AiApiObservedCompletionsResultModel[str] = (
                self._execute_provider_call_with_observability(
                    capability=self.CLIENT_TYPE_COMPLETIONS,
                    operation="send_prompt",
                    dict_input_metadata=dict_input_metadata,
                    callable_execute=_execute_text_prompt,
                    callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_completions_observability_result_summary(
                        observed_result=result,
                        provider_elapsed_ms=provider_elapsed_ms,
                    ),
                    legacy_caller_id=self.user,
                )
            )
            completion: str = self.pii_middleware.process_output(
                observed_result.return_value
            )
            # Normal return with sanitized completion text after observability wrapping.
            return completion

        except Exception as e:
            _LOGGER.exception("An error occurred while sending the prompt: %s", e)
            raise

    def _send_prompt_streaming_provider(
        self,
        prompt: str,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> Iterator[str]:
        """
        Stream a chat completion response from OpenAI chunk by chunk.

        Capability and PII gating already ran in the base template method.
        Unlike send_prompt, streaming does not auto-continue on a `length`
        finish reason: the caller sees exactly one provider stream. There is
        no retry wrapper: retrying a partially consumed stream would duplicate
        output.

        Args:
            prompt: Validated text prompt to send.
            other_params: Optional provider-specific parameters.

        Returns:
            Iterator of response text chunks in provider order.
        """
        system_prompt: str = (
            other_params.system_prompt
            if other_params is not None and other_params.system_prompt is not None
            else AICompletionsPromptParamsBase.DEFAULT_SYSTEM_PROMPT
        )
        user_content = self._build_user_message_content(prompt, other_params)
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = (
            self._build_completions_observability_input_metadata(
                prompt=prompt,
                system_prompt=system_prompt,
                other_params=other_params,
                response_mode=self.RESPONSE_MODE_TEXT,
            )
        )
        dict_input_metadata["response_streaming"] = True

        dict_stream_state: dict[str, Any] = {
            "list_text_parts": [],
            "int_chunk_count": 0,
            "finish_reason": None,
            "usage_chunk": None,
            "bool_completed": False,
        }

        def _open_stream() -> Iterator[str]:
            stream = self.client.chat.completions.create(
                model=self.completions_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                stream=True,
                stream_options={"include_usage": True},
            )
            # Loop through provider chunks so callers see text as it arrives.
            for chunk in stream:
                if chunk.usage is not None:
                    dict_stream_state["usage_chunk"] = chunk
                if not chunk.choices:
                    # Usage-only terminal chunk carries no delta content.
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason is not None:
                    dict_stream_state["finish_reason"] = str(choice.finish_reason)
                str_chunk_text: str | None = choice.delta.content
                if str_chunk_text:
                    dict_stream_state["int_chunk_count"] += 1
                    dict_stream_state["list_text_parts"].append(str_chunk_text)
                    yield str_chunk_text
            dict_stream_state["bool_completed"] = True

        def _build_summary(provider_elapsed_ms: float) -> AiApiCallResultSummaryModel:
            usage_chunk = dict_stream_state["usage_chunk"]
            str_full_text: str = "".join(dict_stream_state["list_text_parts"])
            observed_result: AiApiObservedCompletionsResultModel[str] = (
                AiApiObservedCompletionsResultModel(
                    return_value=str_full_text,
                    raw_output_text=str_full_text,
                    finish_reason=dict_stream_state["finish_reason"],
                    provider_prompt_tokens=(
                        self._extract_openai_prompt_tokens(usage_chunk)
                        if usage_chunk is not None
                        else None
                    ),
                    provider_completion_tokens=(
                        self._extract_openai_completion_tokens(usage_chunk)
                        if usage_chunk is not None
                        else None
                    ),
                    provider_cached_input_tokens=(
                        self._extract_openai_cached_tokens(usage_chunk)
                        if usage_chunk is not None
                        else None
                    ),
                    provider_total_tokens=(
                        self._extract_openai_total_tokens(usage_chunk)
                        if usage_chunk is not None
                        else None
                    ),
                )
            )
            # Normal return with the streaming summary built from accumulated chunks.
            return self._build_streaming_completions_observability_result_summary(
                observed_result=observed_result,
                provider_elapsed_ms=provider_elapsed_ms,
                int_chunk_count=dict_stream_state["int_chunk_count"],
                bool_stream_completed=dict_stream_state["bool_completed"],
            )

        # Normal return with the observability-wrapped OpenAI stream.
        return self._execute_streaming_provider_call_with_observability(
            operation="send_prompt_streaming",
            dict_input_metadata=dict_input_metadata,
            callable_open_stream=_open_stream,
            callable_build_result_summary=_build_summary,
            legacy_caller_id=self.user,
        )

    @staticmethod
    def _extract_openai_prompt_tokens(completion: Any) -> int | None:
        """
        Returns provider-reported prompt token counts from one OpenAI completion response.

        Args:
            completion: OpenAI SDK response object returned by `chat.completions.create`.

        Returns:
            Provider-reported prompt token count when available, otherwise None.
        """
        try:
            if completion.usage is None:
                # Early return because the provider response did not include usage metadata.
                return None
            # Normal return with provider-reported prompt token usage.
            return completion.usage.prompt_tokens
        except AttributeError:
            # Early return because the SDK response did not expose usage metadata as expected.
            return None

    @staticmethod
    def _extract_openai_completion_tokens(completion: Any) -> int | None:
        """
        Returns provider-reported completion token counts from one OpenAI completion response.

        Args:
            completion: OpenAI SDK response object returned by `chat.completions.create`.

        Returns:
            Provider-reported completion token count when available, otherwise None.
        """
        try:
            if completion.usage is None:
                # Early return because the provider response did not include usage metadata.
                return None
            # Normal return with provider-reported completion token usage.
            return completion.usage.completion_tokens
        except AttributeError:
            # Early return because the SDK response did not expose usage metadata as expected.
            return None

    @staticmethod
    def _extract_openai_total_tokens(completion: Any) -> int | None:
        """
        Returns provider-reported total token counts from one OpenAI completion response.

        Args:
            completion: OpenAI SDK response object returned by `chat.completions.create`.

        Returns:
            Provider-reported total token count when available, otherwise None.
        """
        try:
            if completion.usage is None:
                # Early return because the provider response did not include usage metadata.
                return None
            # Normal return with provider-reported total token usage.
            return completion.usage.total_tokens
        except AttributeError:
            # Early return because the SDK response did not expose usage metadata as expected.
            return None

    @staticmethod
    def _extract_openai_cached_tokens(completion: Any) -> int | None:
        """
        Returns provider-reported cached prompt token counts from one OpenAI response.

        OpenAI reports prompt-cache reads in `usage.prompt_tokens_details.cached_tokens`;
        they are a subset of `usage.prompt_tokens`.

        Args:
            completion: OpenAI SDK response object returned by `chat.completions.create`.

        Returns:
            Provider-reported cached prompt token count when available, otherwise None.
        """
        try:
            if completion.usage is None:
                # Early return because the provider response did not include usage metadata.
                return None
            details: Any = getattr(completion.usage, "prompt_tokens_details", None)
            if details is None:
                # Early return because the SDK response did not include cache details.
                return None
            # Normal return with provider-reported cached prompt token usage.
            return getattr(details, "cached_tokens", None)
        except AttributeError:
            # Early return because the SDK response did not expose usage metadata as expected.
            return None

    @staticmethod
    def _sum_optional_ints(
        int_left: int | None,
        int_right: int | None,
    ) -> int | None:
        """
        Sums two optional integers while preserving None when both values are absent.

        Args:
            int_left: Existing optional integer accumulator.
            int_right: Optional integer value to add.

        Returns:
            Summed integer when either value is present, otherwise None.
        """
        if int_left is None and int_right is None:
            # Early return because neither side supplied a value to sum.
            return None
        # Normal return with the summed optional integer value.
        return (int_left or 0) + (int_right or 0)
