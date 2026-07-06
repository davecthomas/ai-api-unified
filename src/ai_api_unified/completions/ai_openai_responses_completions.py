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
from typing import Any, Type

from pydantic import ValidationError

from ..ai_base import (
    AIBaseCompletions,
    AiApiObservedCompletionsResultModel,
    AIStructuredPrompt,
    AICompletionsCapabilitiesBase,
    AICompletionsPromptParamsBase,
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
    ) -> tuple[int | None, int | None, int | None]:
        """Return (input, output, total) token counts from a Responses result."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return None, None, None
        return (
            getattr(usage, "input_tokens", None),
            getattr(usage, "output_tokens", None),
            getattr(usage, "total_tokens", None),
        )

    def send_prompt(
        self,
        prompt: str,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> str:
        """
        Send a text prompt through the Responses API and return the text output.

        Args:
            prompt: The text prompt to send.
            other_params: Optional provider-specific parameters (system prompt only).

        Returns:
            Generated text response.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty or None")
        prompt = self.pii_middleware.process_input(prompt)
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

        def _execute_text_prompt() -> AiApiObservedCompletionsResultModel[str]:
            response = self.client.responses.create(
                model=self.completions_model,
                input=prompt,
                instructions=system_prompt,
            )
            raw_output_text: str = response.output_text or ""
            prompt_tokens, completion_tokens, total_tokens = (
                self._extract_responses_usage(response)
            )
            # Normal return with the Responses text output and usage metadata.
            return AiApiObservedCompletionsResultModel(
                return_value=raw_output_text,
                raw_output_text=raw_output_text,
                finish_reason=str(getattr(response, "status", "") or ""),
                provider_prompt_tokens=prompt_tokens,
                provider_completion_tokens=completion_tokens,
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
            prompt_tokens, completion_tokens, total_tokens = (
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
            prompt_tokens, completion_tokens, total_tokens = (
                self._extract_responses_usage(final_response)
                if final_response is not None
                else (None, None, None)
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
