# ai_openai_responses_completions.py

"""
OpenAI completions via the Responses API (`client.responses`).

This is a text-first completions engine that targets OpenAI's Responses API,
the successor surface to Chat Completions with stronger reasoning-model support
and server-side streaming. It reuses AiOpenAICompletions for model resolution,
pricing, and context lookups, and overrides the provider calls to hit
`client.responses` instead of `client.chat.completions`.

Media inputs are not handled by this engine yet; its capabilities advertise
text only. Use the `openai` (Chat Completions) engine for image inputs.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any, ClassVar, Type

from pydantic import ValidationError

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
    SupportedDataType,
)
from ..middleware.observability_runtime import (
    AiApiCallResultSummaryModel,
    ObservabilityMetadataValue,
)
from .ai_openai_completions import AiOpenAICompletions

_LOGGER: logging.Logger = logging.getLogger(__name__)


class AiOpenAIResponsesCompletions(AiOpenAICompletions):
    """
    OpenAI completions client backed by the Responses API.

    Inherits model resolution, pricing, and context-window lookups from
    AiOpenAICompletions and overrides the provider calls to use
    `client.responses`.
    """

    @property
    def capabilities(self) -> AICompletionsCapabilitiesBase:
        """Return text-only capabilities for the Responses engine."""
        # This engine handles text input only; restrict the inherited model
        # capabilities so callers do not assume media support here.
        return self._capabilities.model_copy(
            update={"supported_data_types": [SupportedDataType.TEXT]}
        )

    @staticmethod
    def _extract_responses_usage(
        response: Any,
    ) -> tuple[int | None, int | None, int | None, int | None]:
        """Return (input, output, total, cached_input) token counts.

        The Responses API reports prompt-cache reads in
        `usage.input_tokens_details.cached_tokens`; they are a subset of
        `usage.input_tokens`.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return None, None, None, None
        details = getattr(usage, "input_tokens_details", None)
        cached_tokens = (
            getattr(details, "cached_tokens", None) if details is not None else None
        )
        return (
            getattr(usage, "input_tokens", None),
            getattr(usage, "output_tokens", None),
            getattr(usage, "total_tokens", None),
            cached_tokens,
        )

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
        Send a text prompt through the Responses API and return the text output.

        Args:
            prompt: The text prompt to send.
            system_prompt: Optional persistent instructions; overrides
                other_params.system_prompt when both are supplied.
            max_response_tokens: Optional response token budget; maps to the
                Responses max_output_tokens field.
            request_timeout_seconds: Optional per-call timeout; maps to the
                OpenAI SDK request timeout.
            other_params: Optional provider-specific parameters (system prompt only).

        Returns:
            Generated text response.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty or None")
        call_client: Any = self._client_for_call(
            request_timeout_seconds=request_timeout_seconds
        )
        dict_token_kwargs: dict[str, Any] = (
            {"max_output_tokens": max_response_tokens}
            if max_response_tokens is not None
            else {}
        )
        prompt = self.pii_middleware.process_input(prompt)
        system_prompt = self._resolve_system_prompt(
            system_prompt,
            other_params,
            AICompletionsPromptParamsBase.DEFAULT_SYSTEM_PROMPT,
        )
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
                response = call_client.responses.create(
                    model=self.completions_model,
                    input=prompt,
                    instructions=system_prompt,
                    **dict_token_kwargs,
                )
            except Exception as exception:
                self._raise_request_error(exception)
                raise
            raw_output_text: str = response.output_text or ""
            prompt_tokens, completion_tokens, total_tokens, cached_tokens = (
                self._extract_responses_usage(response)
            )
            # Normal return with the Responses text output and usage metadata.
            return AiApiObservedCompletionsResultModel(
                return_value=raw_output_text,
                raw_output_text=raw_output_text,
                finish_reason=str(getattr(response, "status", "") or ""),
                provider_prompt_tokens=prompt_tokens,
                provider_completion_tokens=completion_tokens,
                provider_cached_input_tokens=cached_tokens,
                provider_total_tokens=total_tokens,
            )

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
        # Normal return with sanitized Responses text after observability wrapping.
        return self.pii_middleware.process_output(observed_result.return_value)

    def strict_schema_prompt(
        self,
        prompt: str,
        response_model: Type[AIStructuredPrompt],
        max_response_tokens: int = AIBaseCompletions.STRUCTURED_DEFAULT_MAX_RESPONSE_TOKENS,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> AIStructuredPrompt:
        """
        Generate structured output via the Responses API json_schema text format.

        Args:
            prompt: The prompt string to send.
            response_model: Pydantic model class the response is validated into.
            max_response_tokens: Maximum tokens for the response.
            other_params: Optional provider-specific parameters (system prompt only).

        Returns:
            Structured response as an instance of response_model.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty or None")
        if not issubclass(response_model, AIStructuredPrompt):
            raise ValueError("response_model must be a subclass of AIStructuredPrompt")
        self._validate_structured_max_response_tokens(
            provider_name=self.PROVIDER_VENDOR_OPENAI,
            model_name=self.completions_model,
            max_response_tokens=max_response_tokens,
        )
        prompt = self.pii_middleware.process_input(prompt)
        system_prompt: str = (
            other_params.system_prompt
            if other_params is not None and other_params.system_prompt is not None
            else AICompletionsPromptParamsBase.DEFAULT_STRICT_SCHEMA_SYSTEM_PROMPT
        )
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = (
            self._build_completions_observability_input_metadata(
                prompt=prompt,
                system_prompt=system_prompt,
                other_params=other_params,
                response_mode=self.RESPONSE_MODE_STRUCTURED,
                max_response_tokens=max_response_tokens,
            )
        )
        text_format: dict[str, Any] = {
            "format": {
                "type": "json_schema",
                "name": response_model.__name__,
                "schema": response_model.model_json_schema(),
                "strict": False,
            }
        }

        def _generate_structured() -> (
            AiApiObservedCompletionsResultModel[AIStructuredPrompt]
        ):
            response = self.client.responses.create(
                model=self.completions_model,
                input=prompt,
                instructions=system_prompt,
                max_output_tokens=max_response_tokens,
                text=text_format,
            )
            raw_output_text: str = response.output_text or ""
            if not raw_output_text:
                raise ValueError("Empty response from OpenAI Responses API")
            prompt_tokens, completion_tokens, total_tokens, cached_tokens = (
                self._extract_responses_usage(response)
            )
            try:
                content_str: str = self.pii_middleware.process_output(raw_output_text)
                response_data: dict[str, Any] = json.loads(content_str)
                validated_response: AIStructuredPrompt = response_model(**response_data)
            except json.JSONDecodeError as json_error:
                raise ValueError(f"Invalid JSON response: {json_error}") from json_error
            except ValidationError as validation_error:
                raise ValueError(
                    "Response validation failed with "
                    f"{len(validation_error.errors())} validation errors."
                ) from None
            # Normal return with the validated structured response and usage metadata.
            return AiApiObservedCompletionsResultModel(
                return_value=validated_response,
                raw_output_text=raw_output_text,
                finish_reason=str(getattr(response, "status", "") or ""),
                provider_prompt_tokens=prompt_tokens,
                provider_completion_tokens=completion_tokens,
                provider_cached_input_tokens=cached_tokens,
                provider_total_tokens=total_tokens,
            )

        observed_result: AiApiObservedCompletionsResultModel[AIStructuredPrompt] = (
            self._execute_provider_call_with_observability(
                capability=self.CLIENT_TYPE_COMPLETIONS,
                operation="strict_schema_prompt",
                dict_input_metadata=dict_input_metadata,
                callable_execute=_generate_structured,
                callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_completions_observability_result_summary(
                    observed_result=result,
                    provider_elapsed_ms=provider_elapsed_ms,
                ),
                legacy_caller_id=self.user,
            )
        )
        # Normal return with the caller-facing structured response.
        return observed_result.return_value

    def _send_prompt_streaming_provider(
        self,
        prompt: str,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> Iterator[str]:
        """
        Stream a Responses API text response chunk by chunk.

        Capability and PII gating already ran in the base template method. No
        retry wrapper: retrying a partially consumed stream would duplicate
        output.

        Args:
            prompt: Validated text prompt to send.
            other_params: Optional provider-specific parameters (system prompt only).

        Returns:
            Iterator of response text chunks in provider order.
        """
        system_prompt: str = (
            other_params.system_prompt
            if other_params is not None and other_params.system_prompt is not None
            else AICompletionsPromptParamsBase.DEFAULT_SYSTEM_PROMPT
        )
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
            "final_response": None,
            "bool_completed": False,
        }

        def _open_stream() -> Iterator[str]:
            stream = self.client.responses.create(
                model=self.completions_model,
                input=prompt,
                instructions=system_prompt,
                stream=True,
            )
            # Loop through Responses stream events so callers see text as it arrives.
            for event in stream:
                str_event_type: str = getattr(event, "type", "")
                if str_event_type == "response.output_text.delta":
                    str_chunk_text: str = getattr(event, "delta", "") or ""
                    if str_chunk_text:
                        dict_stream_state["int_chunk_count"] += 1
                        dict_stream_state["list_text_parts"].append(str_chunk_text)
                        yield str_chunk_text
                elif str_event_type == "response.completed":
                    dict_stream_state["final_response"] = getattr(
                        event, "response", None
                    )
            dict_stream_state["bool_completed"] = True

        def _build_summary(provider_elapsed_ms: float) -> AiApiCallResultSummaryModel:
            final_response = dict_stream_state["final_response"]
            prompt_tokens, completion_tokens, total_tokens, cached_tokens = (
                self._extract_responses_usage(final_response)
                if final_response is not None
                else (None, None, None, None)
            )
            str_full_text: str = "".join(dict_stream_state["list_text_parts"])
            observed_result: AiApiObservedCompletionsResultModel[str] = (
                AiApiObservedCompletionsResultModel(
                    return_value=str_full_text,
                    raw_output_text=str_full_text,
                    finish_reason=(
                        str(getattr(final_response, "status", "") or "")
                        if final_response is not None
                        else None
                    ),
                    provider_prompt_tokens=prompt_tokens,
                    provider_completion_tokens=completion_tokens,
                    provider_cached_input_tokens=cached_tokens,
                    provider_total_tokens=total_tokens,
                )
            )
            # Normal return with the streaming summary built from accumulated chunks.
            return self._build_streaming_completions_observability_result_summary(
                observed_result=observed_result,
                provider_elapsed_ms=provider_elapsed_ms,
                int_chunk_count=dict_stream_state["int_chunk_count"],
                bool_stream_completed=dict_stream_state["bool_completed"],
            )

        # Normal return with the observability-wrapped Responses stream.
        return self._execute_streaming_provider_call_with_observability(
            operation="send_prompt_streaming",
            dict_input_metadata=dict_input_metadata,
            callable_open_stream=_open_stream,
            callable_build_result_summary=_build_summary,
            legacy_caller_id=self.user,
        )

    # ── Conversation, structured output, async (Responses wire shapes) ──────

    PROVIDER_ENGINE_TOKEN: ClassVar[str] = "openai-responses"

    RESPONSES_STATUS_INCOMPLETE: ClassVar[str] = "incomplete"
    RESPONSES_INCOMPLETE_MAX_TOKENS: ClassVar[str] = "max_output_tokens"

    @staticmethod
    def _build_responses_provider_tools(
        list_tools: list[AITool],
    ) -> list[dict[str, Any]]:
        """
        Maps provider-neutral AITool definitions onto Responses function tools.
        """
        list_provider_tools: list[dict[str, Any]] = []
        # Loop over tool definitions so each maps to the Responses tool shape.
        for ai_tool in list_tools:
            dict_tool: dict[str, Any] = {
                "type": "function",
                "name": ai_tool.name,
                "description": ai_tool.description,
                "parameters": ai_tool.input_schema,
            }
            if ai_tool.strict:
                dict_tool["strict"] = True
            list_provider_tools.append(dict_tool)
        # Normal return with the Responses-shaped tool list.
        return list_provider_tools

    @staticmethod
    def _serialize_responses_output_item(item: Any) -> dict[str, Any]:
        """
        Serializes one Responses output item into a replayable dictionary.
        """
        model_dump = getattr(item, "model_dump", None)
        if callable(model_dump):
            try:
                dict_item: Any = model_dump(exclude_none=True)
                if isinstance(dict_item, dict):
                    # Early return with the SDK-serialized output item.
                    return dict_item
            except TypeError:
                # Fall through to attribute-based reconstruction (test doubles).
                pass
        str_item_type: str = str(getattr(item, "type", "") or "")
        if str_item_type == "function_call":
            return {
                "type": "function_call",
                "call_id": str(getattr(item, "call_id", "") or ""),
                "name": str(getattr(item, "name", "") or ""),
                "arguments": str(getattr(item, "arguments", "") or "{}"),
            }
        if str_item_type == "message":
            return {
                "type": "message",
                "role": "assistant",
                "content": getattr(item, "content", None) or [],
            }
        # Normal return with a minimal typed placeholder for unknown items.
        return {"type": str_item_type}

    def _usage_from_responses(self, response: Any) -> AITokenUsage:
        """
        Builds the provider-neutral usage model from one Responses result.
        """
        tuple_usage = self._extract_responses_usage(response)
        # Normal return with the provider-neutral token usage model.
        return AITokenUsage(
            input_tokens=tuple_usage[0],
            output_tokens=tuple_usage[1],
            cached_input_tokens=tuple_usage[3],
            total_tokens=tuple_usage[2],
        )

    def _responses_finish_reason(
        self,
        response: Any,
        *,
        bool_has_tool_calls: bool,
        bool_has_refusal: bool,
    ) -> AIFinishReason:
        """
        Derives the normalized finish reason from one Responses result.

        The Responses API reports a run status rather than a chat-style finish
        reason: tool calls surface as output items, truncation as
        status="incomplete" with reason max_output_tokens, and refusals as
        refusal content parts.
        """
        if bool_has_refusal:
            # Early return because a refusal part is authoritative.
            return AIFinishReason.REFUSAL
        if bool_has_tool_calls:
            # Early return because requested tool calls define the turn.
            return AIFinishReason.TOOL_USE
        str_status: str = str(getattr(response, "status", "") or "")
        if str_status == self.RESPONSES_STATUS_INCOMPLETE:
            incomplete_details = getattr(response, "incomplete_details", None)
            str_reason: str = str(getattr(incomplete_details, "reason", "") or "")
            if str_reason == self.RESPONSES_INCOMPLETE_MAX_TOKENS:
                # Early return because output was truncated at the budget.
                return AIFinishReason.LENGTH
        # Normal return because the run completed normally.
        return AIFinishReason.COMPLETE

    def _build_turn_result_from_responses(self, response: Any) -> AITurnResult:
        """
        Maps one Responses result onto the provider-neutral turn.
        """
        list_tool_calls: list[AIToolCall] = []
        list_raw_items: list[dict[str, Any]] = []
        bool_has_refusal: bool = False
        # Loop over output items so tool calls and replay content align.
        for item in getattr(response, "output", None) or []:
            list_raw_items.append(self._serialize_responses_output_item(item))
            str_item_type: str = str(getattr(item, "type", "") or "")
            if str_item_type == "function_call":
                str_arguments: str = str(getattr(item, "arguments", "") or "{}")
                try:
                    dict_input: dict[str, Any] = json.loads(str_arguments)
                except json.JSONDecodeError as json_error:
                    raise ValueError(
                        f"Tool call arguments were not valid JSON: {json_error}"
                    ) from json_error
                list_tool_calls.append(
                    AIToolCall(
                        id=str(getattr(item, "call_id", "") or ""),
                        name=str(getattr(item, "name", "") or ""),
                        input=dict_input,
                    )
                )
            elif str_item_type == "message":
                # Loop over content parts so refusal parts are detected.
                for content_part in getattr(item, "content", None) or []:
                    if str(getattr(content_part, "type", "") or "") == "refusal":
                        bool_has_refusal = True
        str_text: str = str(getattr(response, "output_text", "") or "")
        finish_reason: AIFinishReason = self._responses_finish_reason(
            response,
            bool_has_tool_calls=bool(list_tool_calls),
            bool_has_refusal=bool_has_refusal,
        )
        # Normal return with the provider-neutral turn result.
        return AITurnResult(
            text=str_text if str_text else None,
            tool_calls=list_tool_calls,
            finish_reason=finish_reason,
            raw_content=list_raw_items,
            usage=self._usage_from_responses(response),
        )

    def _observed_responses_turn_result(
        self, response: Any
    ) -> AiApiObservedCompletionsResultModel[AITurnResult]:
        """
        Wraps one Responses result as an observed conversation-turn result.
        """
        turn_result: AITurnResult = self._build_turn_result_from_responses(response)
        # Normal return with the observed turn result and usage metadata.
        return AiApiObservedCompletionsResultModel(
            return_value=turn_result,
            raw_output_text=turn_result.text or "",
            finish_reason=str(getattr(response, "status", "") or "") or None,
            provider_prompt_tokens=turn_result.usage.input_tokens,
            provider_completion_tokens=turn_result.usage.output_tokens,
            provider_cached_input_tokens=turn_result.usage.cached_input_tokens,
            provider_total_tokens=turn_result.usage.total_tokens,
            dict_metadata={"tool_call_count": len(turn_result.tool_calls)},
        )

    def _build_responses_conversation_request_kwargs(
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
        Builds the Responses API request kwargs for one conversation turn.
        """
        dict_request_kwargs: dict[str, Any] = {
            "model": self.completions_model,
            "instructions": system_prompt,
            "input": messages,
        }
        if max_response_tokens is not None:
            dict_request_kwargs["max_output_tokens"] = max_response_tokens
        if tools:
            dict_request_kwargs["tools"] = self._build_responses_provider_tools(tools)
        if tool_choice is not None:
            dict_request_kwargs["tool_choice"] = {
                "type": "function",
                "name": tool_choice,
            }
        # provider_options merge last so callers can extend the raw request.
        dict_request_kwargs.update(dict_merge_options)
        # Normal return with the Responses-shaped conversation request.
        return dict_request_kwargs

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
        Sends one conversation turn to the Responses API.
        """
        dict_merge_options, str_retry_override = self._split_provider_options(
            provider_options
        )
        call_client: Any = self._client_for_call(
            request_timeout_seconds=request_timeout_seconds,
            retry_policy_override=str_retry_override,
        )
        dict_request_kwargs: dict[str, Any] = (
            self._build_responses_conversation_request_kwargs(
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
                response = call_client.responses.create(**dict_request_kwargs)
            except Exception as exception:
                self._raise_request_error(exception)
                raise
            # Normal return with the observed conversation turn.
            return self._observed_responses_turn_result(response)

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
            self._build_responses_conversation_request_kwargs(
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
                response = await call_client.responses.create(**dict_request_kwargs)
            except Exception as exception:
                self._raise_request_error(exception)
                raise
            # Normal return with the observed conversation turn.
            return self._observed_responses_turn_result(response)

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
        Builds one Responses function_call_output input item.

        The Responses wire format has no error flag on function outputs, so
        failures are encoded inside the JSON payload under an "error" key.
        """
        dict_payload: dict[str, Any] = {"error": result} if is_error else result
        # Normal return with the Responses-shaped tool-result entry.
        return {
            "type": "function_call_output",
            "call_id": tool_call_id,
            "output": json.dumps(dict_payload),
        }

    def _extend_messages_with_turn_provider(
        self,
        *,
        messages: list[dict[str, Any]],
        turn: AITurnResult,
    ) -> None:
        """
        Extends the input item list with the turn's output items.
        """
        messages.extend(turn.raw_content or [])
        # Normal return after extending the input item list.
        return None

    def _build_responses_structured_request_kwargs(
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
        Builds the Responses API request kwargs for one structured call.
        """
        str_system_prompt: str = self._resolve_system_prompt(
            system_prompt,
            None,
            AICompletionsPromptParamsBase.DEFAULT_STRICT_SCHEMA_SYSTEM_PROMPT,
        )
        list_input: list[dict[str, Any]] = list(messages or [])
        if prompt is not None and prompt.strip():
            str_redacted_prompt: str = self.pii_middleware.process_input(prompt)
            list_input.append({"role": "user", "content": str_redacted_prompt})
        dict_request_kwargs: dict[str, Any] = {
            "model": self.completions_model,
            "instructions": str_system_prompt,
            "input": list_input,
            "max_output_tokens": max_response_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "structured_output",
                    "schema": response_schema,
                    "strict": False,
                }
            },
        }
        dict_request_kwargs.update(dict_merge_options)
        # Normal return with the Responses-shaped structured request.
        return dict_request_kwargs

    def _observed_responses_structured_result(
        self, response: Any
    ) -> AiApiObservedCompletionsResultModel[AIStructuredOutputResult]:
        """
        Wraps one Responses structured result as an observed result.
        """
        bool_has_refusal: bool = False
        # Loop over output message parts so refusal parts are detected.
        for item in getattr(response, "output", None) or []:
            if str(getattr(item, "type", "") or "") == "message":
                for content_part in getattr(item, "content", None) or []:
                    if str(getattr(content_part, "type", "") or "") == "refusal":
                        bool_has_refusal = True
        finish_reason: AIFinishReason = self._responses_finish_reason(
            response,
            bool_has_tool_calls=False,
            bool_has_refusal=bool_has_refusal,
        )
        structured_result: AIStructuredOutputResult = (
            self._build_structured_output_result_from_parts(
                raw_output_text=str(getattr(response, "output_text", "") or ""),
                finish_reason=finish_reason,
                usage=self._usage_from_responses(response),
            )
        )
        # Normal return with the observed structured output result.
        return AiApiObservedCompletionsResultModel(
            return_value=structured_result,
            raw_output_text=structured_result.raw_text,
            finish_reason=str(getattr(response, "status", "") or "") or None,
            provider_prompt_tokens=structured_result.usage.input_tokens,
            provider_completion_tokens=structured_result.usage.output_tokens,
            provider_cached_input_tokens=structured_result.usage.cached_input_tokens,
            provider_total_tokens=structured_result.usage.total_tokens,
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
        Generates structured output via the Responses json_schema text format.
        """
        dict_merge_options, str_retry_override = self._split_provider_options(
            provider_options
        )
        call_client: Any = self._client_for_call(
            request_timeout_seconds=request_timeout_seconds,
            retry_policy_override=str_retry_override,
        )
        dict_request_kwargs: dict[str, Any] = (
            self._build_responses_structured_request_kwargs(
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
                response = call_client.responses.create(**dict_request_kwargs)
            except Exception as exception:
                self._raise_request_error(exception)
                raise
            # Normal return with the observed structured output result.
            return self._observed_responses_structured_result(response)

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
            self._build_responses_structured_request_kwargs(
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
                response = await call_client.responses.create(**dict_request_kwargs)
            except Exception as exception:
                self._raise_request_error(exception)
                raise
            # Normal return with the observed structured output result.
            return self._observed_responses_structured_result(response)

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
        Async twin of send_prompt using AsyncOpenAI on the Responses API.
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
        dict_request_kwargs: dict[str, Any] = {
            "model": self.completions_model,
            "input": str_redacted_prompt,
            "instructions": str_system_prompt,
        }
        if max_response_tokens is not None:
            dict_request_kwargs["max_output_tokens"] = max_response_tokens
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
                response = await call_client.responses.create(**dict_request_kwargs)
            except Exception as exception:
                self._raise_request_error(exception)
                raise
            raw_output_text: str = str(getattr(response, "output_text", "") or "")
            tuple_usage = self._extract_responses_usage(response)
            # Normal return with the Responses text output and usage metadata.
            return AiApiObservedCompletionsResultModel(
                return_value=raw_output_text,
                raw_output_text=raw_output_text,
                finish_reason=str(getattr(response, "status", "") or ""),
                provider_prompt_tokens=tuple_usage[0],
                provider_completion_tokens=tuple_usage[1],
                provider_cached_input_tokens=tuple_usage[3],
                provider_total_tokens=tuple_usage[2],
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
        # Normal return with sanitized Responses text after observability wrapping.
        return self.pii_middleware.process_output(observed_result.return_value)
