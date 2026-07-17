# ai_anthropic_completions.py

"""
Anthropic Claude completions via the native Anthropic API (api.anthropic.com).

This engine targets the Messages API (`client.messages`) using the official
`anthropic` SDK and ANTHROPIC_API_KEY authentication. It is registered as the
`claude` completions engine. Claude models remain reachable through Amazon
Bedrock via the `anthropic` engine; the two differ only in configuration
(native API key vs AWS credentials) and expose the same caller-facing API.

Structured output uses the Messages API `output_config` JSON-schema format,
which constrains the response to the supplied schema on all current Claude
models (including models whose always-on thinking is incompatible with forced
tool choice). Token counting uses the provider-side
`client.messages.count_tokens` endpoint.
"""

from __future__ import annotations

import base64
import copy
import json
import logging
from collections.abc import Iterator
from typing import Any, ClassVar, Type

from anthropic import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import ValidationError

from ..ai_anthropic_base import AIAnthropicBase
from ..ai_base import (
    AIBaseCompletions,
    AiApiObservedCompletionsResultModel,
    AIBatchItemStatus,
    AIBatchJob,
    AIBatchRequestItem,
    AIBatchResultItem,
    AIBatchStatus,
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
    PROVIDER_ANTHROPIC,
    enforce_model_lifecycle,
    get_model_pricing,
)

_LOGGER: logging.Logger = logging.getLogger(__name__)


class AICompletionsCapabilitiesAnthropic(AICompletionsCapabilitiesBase):
    """Anthropic-specific completions capabilities.

    Based on https://platform.claude.com/docs/en/about-claude/models/overview
    (current model lineup, context windows, and feature support).
    """

    # Context window sizes (max input tokens each model can handle).
    DICT_ANTHROPIC_CONTEXT_WINDOWS: ClassVar[dict[str, int]] = {
        "claude-fable-5": 1_000_000,
        "claude-opus-4-8": 1_000_000,
        "claude-opus-4-7": 1_000_000,
        "claude-opus-4-6": 1_000_000,
        "claude-sonnet-4-6": 1_000_000,
        "claude-haiku-4-5": 200_000,
    }

    # Conservative context window for models absent from the lookup table.
    DEFAULT_CONTEXT_WINDOW_LENGTH: ClassVar[int] = 200_000

    @classmethod
    def for_model(cls, model_name: str) -> "AICompletionsCapabilitiesAnthropic":
        """Create capabilities instance for the requested Claude model.

        Every current Claude model streams via the Messages API, counts input
        tokens via the provider-side count_tokens endpoint, accepts image
        inputs, and supports reasoning (adaptive or extended thinking).
        """
        normalized_name: str = model_name.strip().lower()
        # Normal return with per-model context window and registry pricing.
        return cls(
            context_window_length=cls.DICT_ANTHROPIC_CONTEXT_WINDOWS.get(
                normalized_name, cls.DEFAULT_CONTEXT_WINDOW_LENGTH
            ),
            reasoning=True,
            supported_data_types=[SupportedDataType.TEXT, SupportedDataType.IMAGE],
            supports_streaming=True,
            supports_token_counting=True,
            supports_batch=True,
            supports_tool_use=True,
            supports_structured_output=True,
            supports_async=True,
            pricing=get_model_pricing(PROVIDER_ANTHROPIC, normalized_name),
        )


class AiAnthropicCompletions(AIAnthropicBase, AIBaseCompletions):
    """
    Completion client for the native Anthropic API via the Messages API,
    with structured-output prompts, streaming, and token counting.
    """

    DEFAULT_COMPLETIONS_MODEL: ClassVar[str] = "claude-opus-4-8"
    # The Messages API requires max_tokens on every request. Non-streaming
    # requests stay under SDK HTTP-timeout guards at this size.
    SEND_PROMPT_MAX_TOKENS: ClassVar[int] = 16_000
    # Streaming has no timeout concern; give the model room.
    STREAMING_MAX_TOKENS: ClassVar[int] = 64_000
    # Anthropic rejects images above 5MB per image, below the library-wide
    # 20MB attachment cap, so this engine enforces the tighter limit locally.
    MAX_IMAGE_BYTES_ANTHROPIC: ClassVar[int] = 5_000_000
    STOP_REASON_REFUSAL: ClassVar[str] = "refusal"
    STOP_REASON_MAX_TOKENS: ClassVar[str] = "max_tokens"
    STOP_REASON_TOOL_USE: ClassVar[str] = "tool_use"
    # Above this response budget, non-streaming Messages requests risk the
    # SDK's long-request guard, so structured calls stream and accumulate
    # internally; the caller still sees one blocking call.
    NONSTREAMING_MAX_TOKENS_THRESHOLD: ClassVar[int] = 16_000
    # Reserved provider_options key honored by this engine: "retry_policy"
    # ("none" disables SDK retries for that call). All other keys are merged
    # verbatim into the Messages API request.
    PROVIDER_OPTION_RETRY_POLICY: ClassVar[str] = "retry_policy"

    DICT_FINISH_REASON_MAP: ClassVar[dict[str, AIFinishReason]] = {
        "end_turn": AIFinishReason.COMPLETE,
        "stop_sequence": AIFinishReason.COMPLETE,
        "pause_turn": AIFinishReason.COMPLETE,
        "max_tokens": AIFinishReason.LENGTH,
        "tool_use": AIFinishReason.TOOL_USE,
        "refusal": AIFinishReason.REFUSAL,
    }

    def __init__(self, model: str = "", **kwargs: Any):
        """
        Initializes the AiAnthropicCompletions class, setting the model and
        related configuration.

        Args:
            model: The Claude model to use; falls back to COMPLETIONS_MODEL_NAME.
        """
        AIAnthropicBase.__init__(self, **kwargs)
        raw_model: str = model or self.env.get_setting(
            "COMPLETIONS_MODEL_NAME", self.DEFAULT_COMPLETIONS_MODEL
        )
        # Normalize once so lifecycle, context, and pricing lookups agree.
        self.completions_model: str = raw_model.strip().lower()
        AIBaseCompletions.__init__(self, model=self.completions_model, **kwargs)
        self.model = self.completions_model
        enforce_model_lifecycle(PROVIDER_ANTHROPIC, self.completions_model)
        self._capabilities: AICompletionsCapabilitiesAnthropic = (
            AICompletionsCapabilitiesAnthropic.for_model(self.completions_model)
        )

    @property
    def capabilities(self) -> AICompletionsCapabilitiesAnthropic:
        """Return the resolved capabilities for the current Claude model."""
        return self._capabilities

    @property
    def list_model_names(self) -> list[str]:
        # Current Claude lineup on the native Anthropic API (aliases, not
        # date-suffixed snapshots).
        return [
            "claude-fable-5",  # most capable, premium tier
            "claude-opus-4-8",  # most capable Opus-tier model (default)
            "claude-opus-4-7",  # previous-generation Opus
            "claude-opus-4-6",  # older Opus
            "claude-sonnet-4-6",  # speed/intelligence balance
            "claude-haiku-4-5",  # fastest, most cost-effective
        ]

    @property
    def max_context_tokens(self) -> int:
        """
        Look up the Claude context window for the current model_name.
        Falls back to 0 if unknown (i.e. no guard will occur).
        """
        return AICompletionsCapabilitiesAnthropic.DICT_ANTHROPIC_CONTEXT_WINDOWS.get(
            self.completions_model, 0
        )

    def _build_user_message_content(
        self,
        prompt: str,
        other_params: AICompletionsPromptParamsBase | None,
    ) -> str | list[dict[str, Any]]:
        """
        Construct Messages API user content that supports image inputs.
        Returns a plain string for text-only prompts.
        """
        if other_params is None or not other_params.has_included_media:
            return prompt

        # Anthropic recommends placing media blocks before the text block.
        content_parts: list[dict[str, Any]] = []
        for (
            index,
            media_type,
            media_bytes,
            mime_type,
        ) in other_params.iter_included_media():
            if media_type is not SupportedDataType.IMAGE:
                continue
            if len(media_bytes) > self.MAX_IMAGE_BYTES_ANTHROPIC:
                raise ValueError(
                    f"Image attachment {index} exceeds Anthropic's per-image "
                    f"limit of {self.MAX_IMAGE_BYTES_ANTHROPIC} bytes."
                )
            encoded_bytes: str = base64.b64encode(media_bytes).decode("ascii")
            content_parts.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": encoded_bytes,
                    },
                }
            )

        if not content_parts:
            return prompt
        content_parts.append({"type": "text", "text": prompt})
        return content_parts

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """Join the text blocks of a Messages API response, skipping thinking."""
        list_text_parts: list[str] = []
        for block in getattr(response, "content", None) or []:
            if getattr(block, "type", "") == "text":
                list_text_parts.append(getattr(block, "text", "") or "")
        return "".join(list_text_parts)

    @staticmethod
    def _fold_anthropic_prompt_tokens(
        int_input_tokens: int | None,
        int_cached_tokens: int | None,
    ) -> tuple[int | None, int | None, int | None]:
        """Normalize Anthropic input/cache counts to the library's convention.

        Anthropic reports `input_tokens` exclusive of cache reads, with
        `cache_read_input_tokens` billed separately. The library convention is
        that prompt tokens include the cached subset, so this folds cache reads
        into the prompt count and returns (prompt, total_prompt_component,
        cached) where prompt is the combined billable input.
        """
        if int_input_tokens is None and int_cached_tokens is None:
            return None, None, None
        int_prompt_tokens: int = (int_input_tokens or 0) + (int_cached_tokens or 0)
        return int_prompt_tokens, int_prompt_tokens, int_cached_tokens

    @staticmethod
    def _extract_anthropic_usage(
        response: Any,
    ) -> tuple[int | None, int | None, int | None, int | None]:
        """Return (prompt, output, total, cached) token counts from a Messages result.

        `prompt` folds any cache reads into the input count so cached is a
        subset of it, matching the library-wide convention.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return None, None, None, None
        int_input_tokens: int | None = getattr(usage, "input_tokens", None)
        int_output_tokens: int | None = getattr(usage, "output_tokens", None)
        int_cached_tokens: int | None = getattr(usage, "cache_read_input_tokens", None)
        int_prompt_tokens, _, _ = AiAnthropicCompletions._fold_anthropic_prompt_tokens(
            int_input_tokens, int_cached_tokens
        )
        int_total_tokens: int | None = None
        if int_prompt_tokens is not None or int_output_tokens is not None:
            int_total_tokens = (int_prompt_tokens or 0) + (int_output_tokens or 0)
        return int_prompt_tokens, int_output_tokens, int_total_tokens, int_cached_tokens

    @staticmethod
    def _build_output_config_schema(
        response_model: Type[AIStructuredPrompt],
    ) -> dict[str, Any]:
        """
        Build a Messages API output_config JSON schema from a Pydantic model.

        The structured-output format requires additionalProperties to be false
        on every object node; Pydantic does not emit that, so this walks the
        generated schema and sets it. The walk recurses only through known
        schema-bearing keywords so property names like "properties" or
        "additionalProperties" on caller models are never mistaken for schema
        nodes.
        """
        dict_schema: dict[str, Any] = copy.deepcopy(response_model.model_json_schema())
        AiAnthropicCompletions._close_schema_objects(dict_schema)
        return dict_schema

    @staticmethod
    def _close_schema_objects(dict_schema: dict[str, Any]) -> dict[str, Any]:
        """
        Sets additionalProperties: false on every object node of one JSON schema.

        Mutates and returns the supplied schema. Applied to both pydantic-derived
        and caller-supplied raw schemas; compliant schemas pass through unchanged.
        """

        def _close_objects(node: Any) -> None:
            if not isinstance(node, dict):
                return
            if node.get("type") == "object":
                node.setdefault("additionalProperties", False)
            # Recurse only through schema-bearing keywords; "properties" and
            # "$defs" values are name->schema mappings, the rest are schemas
            # or schema lists.
            for mapping_key in ("properties", "$defs", "patternProperties"):
                mapping = node.get(mapping_key)
                if isinstance(mapping, dict):
                    for child_schema in mapping.values():
                        _close_objects(child_schema)
            for schema_key in ("items", "additionalProperties", "not"):
                _close_objects(node.get(schema_key))
            for list_key in ("anyOf", "allOf", "oneOf", "prefixItems"):
                list_schemas = node.get(list_key)
                if isinstance(list_schemas, list):
                    for child_schema in list_schemas:
                        _close_objects(child_schema)

        _close_objects(dict_schema)
        return dict_schema

    # ── Shared helpers for conversation, structured output, and async ───────

    def _split_provider_options(
        self, provider_options: dict[str, Any] | None
    ) -> tuple[dict[str, Any], str | None]:
        """
        Splits provider_options into request-merge keys and reserved keys.

        Args:
            provider_options: Optional engine-specific escape hatch supplied
                per call.

        Returns:
            Tuple of (kwargs merged verbatim into the Messages request,
            optional per-call retry policy override).
        """
        if not provider_options:
            # Early return because there is nothing to split.
            return {}, None
        dict_merge_options: dict[str, Any] = dict(provider_options)
        str_retry_policy: str | None = dict_merge_options.pop(
            self.PROVIDER_OPTION_RETRY_POLICY, None
        )
        # Normal return with merge options and the reserved retry override.
        return dict_merge_options, str_retry_policy

    def _client_for_call(
        self,
        *,
        request_timeout_seconds: float | None = None,
        retry_policy_override: str | None = None,
    ) -> Any:
        """
        Returns the sync client, applying per-call timeout and retry options.

        Args:
            request_timeout_seconds: Optional per-call provider timeout.
            retry_policy_override: Optional per-call retry policy token.

        Returns:
            Anthropic client (possibly a with_options view of it).
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
        Re-raises one Anthropic SDK transport error as the typed request error.

        Carries the HTTP status code (when available) so caller-owned backoff
        can classify 429/5xx/529 uniformly across engines. Non-transport
        exceptions propagate unchanged.

        Args:
            exception: Exception raised by the Anthropic SDK call.

        Raises:
            AiProviderRequestError: For SDK status, timeout, and connection errors.
        """
        if isinstance(exception, APIStatusError):
            raise AiProviderRequestError(
                f"Anthropic request failed with status "
                f"{exception.status_code}: {exception.message}",
                status_code=exception.status_code,
                provider_engine=self.PROVIDER_ENGINE_CLAUDE,
            ) from exception
        if isinstance(exception, (APITimeoutError, APIConnectionError)):
            raise AiProviderRequestError(
                f"Anthropic request failed before a status was available: "
                f"{exception}",
                status_code=None,
                provider_engine=self.PROVIDER_ENGINE_CLAUDE,
            ) from exception
        # Normal return so non-transport exceptions propagate unchanged.
        return None

    @staticmethod
    def _serialize_content_block(block: Any) -> dict[str, Any]:
        """
        Serializes one Messages content block into a replayable dictionary.

        Uses the SDK model_dump when available; otherwise reconstructs the
        block from its typed attributes so raw_content stays replayable as the
        next assistant turn.
        """
        model_dump = getattr(block, "model_dump", None)
        if callable(model_dump):
            try:
                dict_block: Any = model_dump()
                if isinstance(dict_block, dict):
                    # Early return with the SDK-serialized block.
                    return dict_block
            except TypeError:
                # Fall through to attribute-based reconstruction (test doubles).
                pass
        str_block_type: str = str(getattr(block, "type", "") or "")
        if str_block_type == "text":
            return {"type": "text", "text": getattr(block, "text", "") or ""}
        if str_block_type == "tool_use":
            return {
                "type": "tool_use",
                "id": str(getattr(block, "id", "") or ""),
                "name": str(getattr(block, "name", "") or ""),
                "input": dict(getattr(block, "input", None) or {}),
            }
        # Normal return with a minimal typed placeholder for unknown blocks.
        return {"type": str_block_type}

    @staticmethod
    def _build_provider_tools(list_tools: list[AITool]) -> list[dict[str, Any]]:
        """
        Maps provider-neutral AITool definitions onto Messages API tool dicts.
        """
        list_provider_tools: list[dict[str, Any]] = []
        # Loop over tool definitions so each maps to the Messages tool shape.
        for ai_tool in list_tools:
            dict_tool: dict[str, Any] = {
                "name": ai_tool.name,
                "description": ai_tool.description,
                "input_schema": ai_tool.input_schema,
            }
            if ai_tool.strict:
                dict_tool["strict"] = True
            list_provider_tools.append(dict_tool)
        # Normal return with the Messages-shaped tool list.
        return list_provider_tools

    def _usage_from_tuple(
        self,
        tuple_usage: tuple[int | None, int | None, int | None, int | None],
    ) -> AITokenUsage:
        """
        Builds the provider-neutral usage model from the engine usage tuple.
        """
        int_prompt_tokens, int_output_tokens, int_total_tokens, int_cached_tokens = (
            tuple_usage
        )
        # Normal return with the provider-neutral token usage model.
        return AITokenUsage(
            input_tokens=int_prompt_tokens,
            output_tokens=int_output_tokens,
            cached_input_tokens=int_cached_tokens,
            total_tokens=int_total_tokens,
        )

    def _build_turn_result(self, response: Any) -> AITurnResult:
        """
        Maps one Messages API response onto the provider-neutral AITurnResult.
        """
        list_text_parts: list[str] = []
        list_tool_calls: list[AIToolCall] = []
        list_raw_blocks: list[dict[str, Any]] = []
        # Loop over response blocks so text, tool calls, and raw replay content align.
        for block in getattr(response, "content", None) or []:
            list_raw_blocks.append(self._serialize_content_block(block))
            str_block_type: str = str(getattr(block, "type", "") or "")
            if str_block_type == "text":
                list_text_parts.append(getattr(block, "text", "") or "")
            elif str_block_type == "tool_use":
                list_tool_calls.append(
                    AIToolCall(
                        id=str(getattr(block, "id", "") or ""),
                        name=str(getattr(block, "name", "") or ""),
                        input=dict(getattr(block, "input", None) or {}),
                    )
                )
        str_stop_reason: str | None = getattr(response, "stop_reason", None)
        finish_reason: AIFinishReason = self._normalize_finish_reason(
            str(str_stop_reason) if str_stop_reason is not None else None,
            self.DICT_FINISH_REASON_MAP,
        )
        str_text: str = "".join(list_text_parts)
        # Normal return with the provider-neutral turn result.
        return AITurnResult(
            text=str_text if str_text else None,
            tool_calls=list_tool_calls,
            finish_reason=finish_reason,
            raw_content=list_raw_blocks,
            usage=self._usage_from_tuple(self._extract_anthropic_usage(response)),
        )

    def _build_structured_messages(
        self,
        *,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """
        Builds the Messages list for one structured-output call.

        Caller-managed messages are replayed first (for example a prior model
        output plus correction feedback); a supplied prompt is appended as the
        final user turn.
        """
        list_messages: list[dict[str, Any]] = list(messages or [])
        if prompt is not None and prompt.strip():
            str_redacted_prompt: str = self.pii_middleware.process_input(prompt)
            list_messages.append({"role": "user", "content": str_redacted_prompt})
        # Normal return with the combined message list.
        return list_messages

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
        Sends a prompt to the Anthropic Messages API and returns the text result.

        Args:
            prompt: The prompt string to send.
            system_prompt: Optional persistent instructions; maps to the
                Messages API system field and overrides
                other_params.system_prompt when both are supplied.
            max_response_tokens: Optional response token budget; maps to the
                Messages API max_tokens field (default SEND_PROMPT_MAX_TOKENS).
            request_timeout_seconds: Optional per-call timeout; maps to the
                Anthropic SDK request timeout.
            other_params: Optional provider-specific parameters (system prompt,
                image attachments).

        Returns:
            The completion result as a string.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty or None")
        prompt = self.pii_middleware.process_input(prompt)
        str_system_prompt: str = self._resolve_system_prompt(
            system_prompt,
            other_params,
            AICompletionsPromptParamsBase.DEFAULT_SYSTEM_PROMPT,
        )
        int_max_tokens: int = max_response_tokens or self.SEND_PROMPT_MAX_TOKENS
        call_client: Any = self._client_for_call(
            request_timeout_seconds=request_timeout_seconds
        )
        user_content = self._build_user_message_content(prompt, other_params)
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = (
            self._build_completions_observability_input_metadata(
                prompt=prompt,
                system_prompt=str_system_prompt,
                other_params=other_params,
                response_mode=self.RESPONSE_MODE_TEXT,
                max_response_tokens=max_response_tokens,
            )
        )

        def _execute_text_prompt() -> AiApiObservedCompletionsResultModel[str]:
            try:
                response = call_client.messages.create(
                    model=self.completions_model,
                    max_tokens=int_max_tokens,
                    system=str_system_prompt,
                    messages=[{"role": "user", "content": user_content}],
                )
            except Exception as exception:
                self._raise_request_error(exception)
                raise
            str_stop_reason: str = str(getattr(response, "stop_reason", "") or "")
            if str_stop_reason == self.STOP_REASON_REFUSAL:
                # Safety classifiers return HTTP 200 with an empty or partial
                # body; surfacing that as a successful completion would be
                # indistinguishable from a real answer.
                raise RuntimeError(
                    f"Anthropic declined the request for model "
                    f"'{self.completions_model}' (stop_reason=refusal)."
                )
            raw_output_text: str = self._extract_response_text(response)
            if str_stop_reason == self.STOP_REASON_MAX_TOKENS:
                _LOGGER.warning(
                    "Anthropic response for model '%s' was truncated at the "
                    "%s-token max_tokens cap.",
                    self.completions_model,
                    int_max_tokens,
                )
            prompt_tokens, completion_tokens, total_tokens, cached_tokens = (
                self._extract_anthropic_usage(response)
            )
            # Normal return with the Messages text output and usage metadata.
            return AiApiObservedCompletionsResultModel(
                return_value=raw_output_text,
                raw_output_text=raw_output_text,
                finish_reason=str_stop_reason,
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
        # Normal return with sanitized Messages text after observability wrapping.
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
        Generate structured output via the Messages API output_config JSON-schema
        format and parse the response into the specified Pydantic model.

        On models with always-on thinking (claude-fable-5), thinking tokens
        count against max_tokens; pass a max_response_tokens well above the
        2048 shared default there so the budget covers thinking plus the JSON
        body.

        Args:
            prompt: The prompt string to send.
            response_model: Pydantic model class the response is validated into.
            max_response_tokens: Maximum tokens for the response.
            other_params: Optional provider-specific parameters (system prompt,
                image attachments).

        Returns:
            Structured response as an instance of response_model.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty or None")
        if not issubclass(response_model, AIStructuredPrompt):
            raise ValueError("response_model must be a subclass of AIStructuredPrompt")
        self._validate_structured_max_response_tokens(
            provider_name=self.PROVIDER_VENDOR_ANTHROPIC,
            model_name=self.completions_model,
            max_response_tokens=max_response_tokens,
        )
        prompt = self.pii_middleware.process_input(prompt)
        system_prompt: str = (
            other_params.system_prompt
            if other_params is not None and other_params.system_prompt is not None
            else AICompletionsPromptParamsBase.DEFAULT_STRICT_SCHEMA_SYSTEM_PROMPT
        )
        user_content = self._build_user_message_content(prompt, other_params)
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = (
            self._build_completions_observability_input_metadata(
                prompt=prompt,
                system_prompt=system_prompt,
                other_params=other_params,
                response_mode=self.RESPONSE_MODE_STRUCTURED,
                max_response_tokens=max_response_tokens,
            )
        )
        dict_output_config: dict[str, Any] = {
            "format": {
                "type": "json_schema",
                "schema": self._build_output_config_schema(response_model),
            }
        }

        def _generate_structured() -> (
            AiApiObservedCompletionsResultModel[AIStructuredPrompt]
        ):
            response = self.client.messages.create(
                model=self.completions_model,
                max_tokens=max_response_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
                output_config=dict_output_config,
            )
            str_finish_reason: str = str(getattr(response, "stop_reason", "") or "")
            if str_finish_reason == self.STOP_REASON_REFUSAL:
                raise RuntimeError(
                    f"Anthropic declined the request for model "
                    f"'{self.completions_model}' (stop_reason=refusal)."
                )
            raw_output_text: str = self._extract_response_text(response)
            if str_finish_reason == self.STOP_REASON_MAX_TOKENS:
                self._raise_structured_token_limit_error(
                    provider_name=self.PROVIDER_VENDOR_ANTHROPIC,
                    model_name=self.completions_model,
                    max_response_tokens=max_response_tokens,
                    finish_reason=str_finish_reason,
                    raw_output_text=raw_output_text,
                )
            if not raw_output_text:
                raise ValueError("Empty response from Anthropic Messages API")
            prompt_tokens, completion_tokens, total_tokens, cached_tokens = (
                self._extract_anthropic_usage(response)
            )
            try:
                content_str: str = self.pii_middleware.process_output(raw_output_text)
                response_data: dict[str, Any] = json.loads(content_str)
                validated_response: AIStructuredPrompt = response_model.model_validate(
                    response_data
                )
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
                finish_reason=str_finish_reason,
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

    # ── Conversation turns and tool use ─────────────────────────────────────

    def _build_conversation_request_kwargs(
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
        Builds the Messages API request kwargs for one conversation turn.
        """
        dict_request_kwargs: dict[str, Any] = {
            "model": self.completions_model,
            "max_tokens": max_response_tokens or self.SEND_PROMPT_MAX_TOKENS,
            "system": system_prompt,
            "messages": messages,
        }
        if tools:
            dict_request_kwargs["tools"] = self._build_provider_tools(tools)
        if tool_choice is not None:
            dict_request_kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}
        # provider_options merge last so callers can extend the raw request;
        # unknown keys are the provider's to accept or reject.
        dict_request_kwargs.update(dict_merge_options)
        # Normal return with the Messages-shaped conversation request.
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

    def _observed_turn_result(
        self, response: Any
    ) -> AiApiObservedCompletionsResultModel[AITurnResult]:
        """
        Wraps one Messages response as an observed conversation-turn result.
        """
        turn_result: AITurnResult = self._build_turn_result(response)
        tuple_usage = self._extract_anthropic_usage(response)
        str_stop_reason: str | None = getattr(response, "stop_reason", None)
        # Normal return with the observed turn result and usage metadata.
        return AiApiObservedCompletionsResultModel(
            return_value=turn_result,
            raw_output_text=turn_result.text or "",
            finish_reason=str(str_stop_reason) if str_stop_reason else None,
            provider_prompt_tokens=tuple_usage[0],
            provider_completion_tokens=tuple_usage[1],
            provider_cached_input_tokens=tuple_usage[3],
            provider_total_tokens=tuple_usage[2],
            dict_metadata={"tool_call_count": len(turn_result.tool_calls)},
        )

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
        Sends one conversation turn to the Messages API.
        """
        dict_merge_options, str_retry_override = self._split_provider_options(
            provider_options
        )
        call_client: Any = self._client_for_call(
            request_timeout_seconds=request_timeout_seconds,
            retry_policy_override=str_retry_override,
        )
        dict_request_kwargs: dict[str, Any] = self._build_conversation_request_kwargs(
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_response_tokens=max_response_tokens,
            dict_merge_options=dict_merge_options,
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
                response = call_client.messages.create(**dict_request_kwargs)
            except Exception as exception:
                self._raise_request_error(exception)
                raise
            # Normal return with the observed conversation turn.
            return self._observed_turn_result(response)

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
        Async twin of _send_conversation_provider using the AsyncAnthropic client.
        """
        dict_merge_options, str_retry_override = self._split_provider_options(
            provider_options
        )
        call_client: Any = self._async_client_for_call(
            request_timeout_seconds=request_timeout_seconds,
            retry_policy_override=str_retry_override,
        )
        dict_request_kwargs: dict[str, Any] = self._build_conversation_request_kwargs(
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_response_tokens=max_response_tokens,
            dict_merge_options=dict_merge_options,
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
                response = await call_client.messages.create(**dict_request_kwargs)
            except Exception as exception:
                self._raise_request_error(exception)
                raise
            # Normal return with the observed conversation turn.
            return self._observed_turn_result(response)

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
        Builds one Messages API tool_result user message.
        """
        # Normal return with the Messages-shaped tool-result entry.
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": json.dumps(result),
                    "is_error": is_error,
                }
            ],
        }

    # ── Structured output (send_structured_output) ──────────────────────────

    def _build_structured_request_kwargs(
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
        Builds the Messages API request kwargs for one structured-output call.
        """
        str_system_prompt: str = self._resolve_system_prompt(
            system_prompt,
            None,
            AICompletionsPromptParamsBase.DEFAULT_STRICT_SCHEMA_SYSTEM_PROMPT,
        )
        dict_schema: dict[str, Any] = self._close_schema_objects(
            copy.deepcopy(response_schema)
        )
        dict_request_kwargs: dict[str, Any] = {
            "model": self.completions_model,
            "max_tokens": max_response_tokens,
            "system": str_system_prompt,
            "messages": self._build_structured_messages(
                prompt=prompt, messages=messages
            ),
            "output_config": {"format": {"type": "json_schema", "schema": dict_schema}},
        }
        dict_request_kwargs.update(dict_merge_options)
        # Normal return with the Messages-shaped structured request.
        return dict_request_kwargs

    def _execute_structured_request(
        self,
        call_client: Any,
        dict_request_kwargs: dict[str, Any],
        *,
        bool_stream: bool,
    ) -> tuple[str, str | None, tuple[int | None, int | None, int | None, int | None]]:
        """
        Executes one structured request, streaming and accumulating when asked.

        Large response budgets trip the SDK's non-streaming long-request guard,
        so calls above NONSTREAMING_MAX_TOKENS_THRESHOLD stream internally and
        accumulate; the caller still sees one blocking call.

        Returns:
            Tuple of (raw output text, provider stop reason, usage tuple).
        """
        if not bool_stream:
            try:
                response = call_client.messages.create(**dict_request_kwargs)
            except Exception as exception:
                self._raise_request_error(exception)
                raise
            # Normal return with the non-streaming structured response parts.
            return (
                self._extract_response_text(response),
                getattr(response, "stop_reason", None),
                self._extract_anthropic_usage(response),
            )

        list_text_parts: list[str] = []
        str_stop_reason: str | None = None
        int_input_tokens: int | None = None
        int_cached_tokens: int | None = None
        int_output_tokens: int | None = None
        try:
            stream = call_client.messages.create(**dict_request_kwargs, stream=True)
            # Loop through stream events so long structured responses accumulate.
            for event in stream:
                str_event_type: str = getattr(event, "type", "")
                if str_event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if getattr(delta, "type", "") == "text_delta":
                        list_text_parts.append(getattr(delta, "text", "") or "")
                elif str_event_type == "message_start":
                    usage = getattr(getattr(event, "message", None), "usage", None)
                    int_input_tokens = getattr(usage, "input_tokens", None)
                    int_cached_tokens = getattr(usage, "cache_read_input_tokens", None)
                elif str_event_type == "message_delta":
                    delta = getattr(event, "delta", None)
                    raw_stop_reason = getattr(delta, "stop_reason", None)
                    if raw_stop_reason is not None:
                        str_stop_reason = str(raw_stop_reason)
                    usage = getattr(event, "usage", None)
                    raw_output_tokens = getattr(usage, "output_tokens", None)
                    if raw_output_tokens is not None:
                        int_output_tokens = raw_output_tokens
        except Exception as exception:
            self._raise_request_error(exception)
            raise
        int_prompt_tokens, _, _ = self._fold_anthropic_prompt_tokens(
            int_input_tokens, int_cached_tokens
        )
        int_total_tokens: int | None = None
        if int_prompt_tokens is not None or int_output_tokens is not None:
            int_total_tokens = (int_prompt_tokens or 0) + (int_output_tokens or 0)
        # Normal return with the accumulated structured response parts.
        return (
            "".join(list_text_parts),
            str_stop_reason,
            (int_prompt_tokens, int_output_tokens, int_total_tokens, int_cached_tokens),
        )

    async def _aexecute_structured_request(
        self,
        call_client: Any,
        dict_request_kwargs: dict[str, Any],
        *,
        bool_stream: bool,
    ) -> tuple[str, str | None, tuple[int | None, int | None, int | None, int | None]]:
        """
        Async twin of _execute_structured_request.
        """
        if not bool_stream:
            try:
                response = await call_client.messages.create(**dict_request_kwargs)
            except Exception as exception:
                self._raise_request_error(exception)
                raise
            # Normal return with the non-streaming structured response parts.
            return (
                self._extract_response_text(response),
                getattr(response, "stop_reason", None),
                self._extract_anthropic_usage(response),
            )

        list_text_parts: list[str] = []
        str_stop_reason: str | None = None
        int_input_tokens: int | None = None
        int_cached_tokens: int | None = None
        int_output_tokens: int | None = None
        try:
            stream = await call_client.messages.create(
                **dict_request_kwargs, stream=True
            )
            # Loop through stream events so long structured responses accumulate.
            async for event in stream:
                str_event_type: str = getattr(event, "type", "")
                if str_event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if getattr(delta, "type", "") == "text_delta":
                        list_text_parts.append(getattr(delta, "text", "") or "")
                elif str_event_type == "message_start":
                    usage = getattr(getattr(event, "message", None), "usage", None)
                    int_input_tokens = getattr(usage, "input_tokens", None)
                    int_cached_tokens = getattr(usage, "cache_read_input_tokens", None)
                elif str_event_type == "message_delta":
                    delta = getattr(event, "delta", None)
                    raw_stop_reason = getattr(delta, "stop_reason", None)
                    if raw_stop_reason is not None:
                        str_stop_reason = str(raw_stop_reason)
                    usage = getattr(event, "usage", None)
                    raw_output_tokens = getattr(usage, "output_tokens", None)
                    if raw_output_tokens is not None:
                        int_output_tokens = raw_output_tokens
        except Exception as exception:
            self._raise_request_error(exception)
            raise
        int_prompt_tokens, _, _ = self._fold_anthropic_prompt_tokens(
            int_input_tokens, int_cached_tokens
        )
        int_total_tokens: int | None = None
        if int_prompt_tokens is not None or int_output_tokens is not None:
            int_total_tokens = (int_prompt_tokens or 0) + (int_output_tokens or 0)
        # Normal return with the accumulated structured response parts.
        return (
            "".join(list_text_parts),
            str_stop_reason,
            (int_prompt_tokens, int_output_tokens, int_total_tokens, int_cached_tokens),
        )

    def _build_structured_output_result(
        self,
        *,
        raw_output_text: str,
        str_stop_reason: str | None,
        tuple_usage: tuple[int | None, int | None, int | None, int | None],
    ) -> AIStructuredOutputResult:
        """
        Maps one structured response onto the provider-neutral result model.

        LENGTH and REFUSAL turns return data=None with the normalized finish
        reason so the caller distinguishes retryable truncation from an abort
        in code; COMPLETE turns parse the JSON body.
        """
        finish_reason: AIFinishReason = self._normalize_finish_reason(
            str_stop_reason, self.DICT_FINISH_REASON_MAP
        )
        usage: AITokenUsage = self._usage_from_tuple(tuple_usage)
        if finish_reason in (AIFinishReason.LENGTH, AIFinishReason.REFUSAL):
            # Early return so callers branch on finish_reason instead of parsing.
            return AIStructuredOutputResult(
                data=None,
                finish_reason=finish_reason,
                usage=usage,
                raw_text=raw_output_text,
            )
        str_content: str = self.pii_middleware.process_output(raw_output_text)
        if not str_content:
            raise ValueError("Empty response from Anthropic Messages API")
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
        Generates structured output via the Messages API output_config format.
        """
        dict_merge_options, str_retry_override = self._split_provider_options(
            provider_options
        )
        call_client: Any = self._client_for_call(
            request_timeout_seconds=request_timeout_seconds,
            retry_policy_override=str_retry_override,
        )
        dict_request_kwargs: dict[str, Any] = self._build_structured_request_kwargs(
            response_schema=response_schema,
            system_prompt=system_prompt,
            prompt=prompt,
            messages=messages,
            max_response_tokens=max_response_tokens,
            dict_merge_options=dict_merge_options,
        )
        bool_stream: bool = max_response_tokens > self.NONSTREAMING_MAX_TOKENS_THRESHOLD
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
            raw_output_text, str_stop_reason, tuple_usage = (
                self._execute_structured_request(
                    call_client, dict_request_kwargs, bool_stream=bool_stream
                )
            )
            structured_result: AIStructuredOutputResult = (
                self._build_structured_output_result(
                    raw_output_text=raw_output_text,
                    str_stop_reason=(
                        str(str_stop_reason) if str_stop_reason is not None else None
                    ),
                    tuple_usage=tuple_usage,
                )
            )
            # Normal return with the observed structured output result.
            return AiApiObservedCompletionsResultModel(
                return_value=structured_result,
                raw_output_text=structured_result.raw_text,
                finish_reason=(
                    str(str_stop_reason) if str_stop_reason is not None else None
                ),
                provider_prompt_tokens=tuple_usage[0],
                provider_completion_tokens=tuple_usage[1],
                provider_cached_input_tokens=tuple_usage[3],
                provider_total_tokens=tuple_usage[2],
            )

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
        dict_request_kwargs: dict[str, Any] = self._build_structured_request_kwargs(
            response_schema=response_schema,
            system_prompt=system_prompt,
            prompt=prompt,
            messages=messages,
            max_response_tokens=max_response_tokens,
            dict_merge_options=dict_merge_options,
        )
        bool_stream: bool = max_response_tokens > self.NONSTREAMING_MAX_TOKENS_THRESHOLD
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
            raw_output_text, str_stop_reason, tuple_usage = (
                await self._aexecute_structured_request(
                    call_client, dict_request_kwargs, bool_stream=bool_stream
                )
            )
            structured_result: AIStructuredOutputResult = (
                self._build_structured_output_result(
                    raw_output_text=raw_output_text,
                    str_stop_reason=(
                        str(str_stop_reason) if str_stop_reason is not None else None
                    ),
                    tuple_usage=tuple_usage,
                )
            )
            # Normal return with the observed structured output result.
            return AiApiObservedCompletionsResultModel(
                return_value=structured_result,
                raw_output_text=structured_result.raw_text,
                finish_reason=(
                    str(str_stop_reason) if str_stop_reason is not None else None
                ),
                provider_prompt_tokens=tuple_usage[0],
                provider_completion_tokens=tuple_usage[1],
                provider_cached_input_tokens=tuple_usage[3],
                provider_total_tokens=tuple_usage[2],
            )

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
        Async twin of send_prompt using the AsyncAnthropic client.
        """
        str_redacted_prompt: str = self.pii_middleware.process_input(prompt)
        str_system_prompt: str = self._resolve_system_prompt(
            system_prompt,
            other_params,
            AICompletionsPromptParamsBase.DEFAULT_SYSTEM_PROMPT,
        )
        int_max_tokens: int = max_response_tokens or self.SEND_PROMPT_MAX_TOKENS
        call_client: Any = self._async_client_for_call(
            request_timeout_seconds=request_timeout_seconds
        )
        user_content = self._build_user_message_content(
            str_redacted_prompt, other_params
        )
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
                response = await call_client.messages.create(
                    model=self.completions_model,
                    max_tokens=int_max_tokens,
                    system=str_system_prompt,
                    messages=[{"role": "user", "content": user_content}],
                )
            except Exception as exception:
                self._raise_request_error(exception)
                raise
            str_stop_reason: str = str(getattr(response, "stop_reason", "") or "")
            if str_stop_reason == self.STOP_REASON_REFUSAL:
                raise RuntimeError(
                    f"Anthropic declined the request for model "
                    f"'{self.completions_model}' (stop_reason=refusal)."
                )
            raw_output_text: str = self._extract_response_text(response)
            prompt_tokens, completion_tokens, total_tokens, cached_tokens = (
                self._extract_anthropic_usage(response)
            )
            # Normal return with the Messages text output and usage metadata.
            return AiApiObservedCompletionsResultModel(
                return_value=raw_output_text,
                raw_output_text=raw_output_text,
                finish_reason=str_stop_reason,
                provider_prompt_tokens=prompt_tokens,
                provider_completion_tokens=completion_tokens,
                provider_cached_input_tokens=cached_tokens,
                provider_total_tokens=total_tokens,
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
        # Normal return with sanitized Messages text after observability wrapping.
        return self.pii_middleware.process_output(observed_result.return_value)

    def _send_prompt_streaming_provider(
        self,
        prompt: str,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> Iterator[str]:
        """
        Stream a Messages API text response chunk by chunk.

        Capability and PII gating already ran in the base template method. No
        retry wrapper: retrying a partially consumed stream would duplicate
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
            "int_input_tokens": None,
            "int_cached_input_tokens": None,
            "int_output_tokens": None,
            "bool_completed": False,
        }

        def _open_stream() -> Iterator[str]:
            stream = self.client.messages.create(
                model=self.completions_model,
                max_tokens=self.STREAMING_MAX_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
                stream=True,
            )
            # Loop through Messages stream events so callers see text as it arrives.
            for event in stream:
                str_event_type: str = getattr(event, "type", "")
                if str_event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if getattr(delta, "type", "") == "text_delta":
                        str_chunk_text: str = getattr(delta, "text", "") or ""
                        if str_chunk_text:
                            dict_stream_state["int_chunk_count"] += 1
                            dict_stream_state["list_text_parts"].append(str_chunk_text)
                            yield str_chunk_text
                elif str_event_type == "message_start":
                    usage = getattr(getattr(event, "message", None), "usage", None)
                    int_input_tokens: int | None = getattr(usage, "input_tokens", None)
                    if int_input_tokens is not None:
                        dict_stream_state["int_input_tokens"] = int_input_tokens
                    int_cached_tokens: int | None = getattr(
                        usage, "cache_read_input_tokens", None
                    )
                    if int_cached_tokens is not None:
                        dict_stream_state["int_cached_input_tokens"] = int_cached_tokens
                elif str_event_type == "message_delta":
                    delta = getattr(event, "delta", None)
                    str_stop_reason: str | None = getattr(delta, "stop_reason", None)
                    if str_stop_reason is not None:
                        dict_stream_state["finish_reason"] = str(str_stop_reason)
                        if str(str_stop_reason) == self.STOP_REASON_REFUSAL:
                            _LOGGER.warning(
                                "Anthropic declined mid-stream for model '%s' "
                                "(stop_reason=refusal); streamed text is partial.",
                                self.completions_model,
                            )
                    usage = getattr(event, "usage", None)
                    int_output_tokens: int | None = getattr(
                        usage, "output_tokens", None
                    )
                    if int_output_tokens is not None:
                        dict_stream_state["int_output_tokens"] = int_output_tokens
            dict_stream_state["bool_completed"] = True

        def _build_summary(provider_elapsed_ms: float) -> AiApiCallResultSummaryModel:
            int_input_tokens: int | None = dict_stream_state["int_input_tokens"]
            int_cached_tokens: int | None = dict_stream_state["int_cached_input_tokens"]
            int_output_tokens: int | None = dict_stream_state["int_output_tokens"]
            # Fold cache reads into the prompt count so cached is a subset of it.
            int_prompt_tokens, _, _ = self._fold_anthropic_prompt_tokens(
                int_input_tokens, int_cached_tokens
            )
            int_total_tokens: int | None = None
            if int_prompt_tokens is not None or int_output_tokens is not None:
                int_total_tokens = (int_prompt_tokens or 0) + (int_output_tokens or 0)
            str_full_text: str = "".join(dict_stream_state["list_text_parts"])
            observed_result: AiApiObservedCompletionsResultModel[str] = (
                AiApiObservedCompletionsResultModel(
                    return_value=str_full_text,
                    raw_output_text=str_full_text,
                    finish_reason=dict_stream_state["finish_reason"],
                    provider_prompt_tokens=int_prompt_tokens,
                    provider_completion_tokens=int_output_tokens,
                    provider_cached_input_tokens=int_cached_tokens,
                    provider_total_tokens=int_total_tokens,
                )
            )
            # Normal return with the streaming summary built from accumulated chunks.
            return self._build_streaming_completions_observability_result_summary(
                observed_result=observed_result,
                provider_elapsed_ms=provider_elapsed_ms,
                int_chunk_count=dict_stream_state["int_chunk_count"],
                bool_stream_completed=dict_stream_state["bool_completed"],
            )

        # Normal return with the observability-wrapped Messages stream.
        return self._execute_streaming_provider_call_with_observability(
            operation="send_prompt_streaming",
            dict_input_metadata=dict_input_metadata,
            callable_open_stream=_open_stream,
            callable_build_result_summary=_build_summary,
            legacy_caller_id=self.user,
        )

    def _count_tokens_provider(
        self,
        prompt: str,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> int:
        """
        Count input tokens via the Messages API count_tokens endpoint.

        Builds the same request shape send_prompt would submit and asks the
        provider to count its input tokens without running inference.

        Args:
            prompt: Validated text prompt to measure.
            other_params: Optional provider-specific parameters.

        Returns:
            Provider-reported input token count.
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

        def _execute_count_tokens() -> AiApiObservedCompletionsResultModel[int]:
            response = self.client.messages.count_tokens(
                model=self.completions_model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            int_input_tokens: int = int(getattr(response, "input_tokens", 0))
            # Normal return with the provider-counted input token total.
            return AiApiObservedCompletionsResultModel(
                return_value=int_input_tokens,
                raw_output_text="",
                provider_prompt_tokens=int_input_tokens,
                provider_total_tokens=int_input_tokens,
                dict_metadata={"operation": "count_tokens"},
            )

        observed_result: AiApiObservedCompletionsResultModel[int] = (
            self._execute_provider_call_with_observability(
                capability=self.CLIENT_TYPE_COMPLETIONS,
                operation="count_tokens",
                dict_input_metadata=dict_input_metadata,
                callable_execute=_execute_count_tokens,
                callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_completions_observability_result_summary(
                    observed_result=result,
                    provider_elapsed_ms=provider_elapsed_ms,
                ),
                legacy_caller_id=self.user,
            )
        )
        # Normal return with the caller-facing input token count.
        return observed_result.return_value

    # ── Batch completions (Message Batches API) ─────────────────────────────

    BATCH_ID_PREFIX: ClassVar[str] = "claude:"

    DICT_BATCH_PROCESSING_STATUS: ClassVar[dict[str, AIBatchStatus]] = {
        "in_progress": AIBatchStatus.IN_PROGRESS,
        "canceling": AIBatchStatus.CANCELING,
        "ended": AIBatchStatus.ENDED,
    }

    DICT_BATCH_RESULT_TYPE: ClassVar[dict[str, AIBatchItemStatus]] = {
        "succeeded": AIBatchItemStatus.SUCCEEDED,
        "errored": AIBatchItemStatus.ERRORED,
        "canceled": AIBatchItemStatus.CANCELED,
        "expired": AIBatchItemStatus.EXPIRED,
    }

    def _to_provider_batch_id(self, batch_id: str) -> str:
        """Strip the engine namespace to recover the raw provider batch id."""
        if batch_id.startswith(self.BATCH_ID_PREFIX):
            return batch_id[len(self.BATCH_ID_PREFIX) :]
        return batch_id

    def _normalize_batch_job(self, batch: Any) -> AIBatchJob:
        """Map an Anthropic batch object to the provider-agnostic AIBatchJob."""
        counts = getattr(batch, "request_counts", None)
        int_processing: int | None = getattr(counts, "processing", None)
        int_succeeded: int | None = getattr(counts, "succeeded", None)
        int_errored: int | None = getattr(counts, "errored", None)
        int_canceled: int | None = getattr(counts, "canceled", None)
        int_expired: int | None = getattr(counts, "expired", None)
        list_counts: list[int | None] = [
            int_processing,
            int_succeeded,
            int_errored,
            int_canceled,
            int_expired,
        ]
        int_request_count: int | None = None
        if any(value is not None for value in list_counts):
            int_request_count = sum(value or 0 for value in list_counts)
        str_provider_batch_id: str = str(getattr(batch, "id", "") or "")
        str_processing_status: str = str(getattr(batch, "processing_status", "") or "")
        batch_status: AIBatchStatus | None = self.DICT_BATCH_PROCESSING_STATUS.get(
            str_processing_status
        )
        if batch_status is None:
            # An unmapped status is treated as still in progress (so results are
            # not fetched prematurely); log it since it can otherwise poll until
            # the run_batch timeout.
            _LOGGER.warning(
                "Unmapped Anthropic batch processing_status %r for batch %s; "
                "treating as in_progress.",
                str_processing_status,
                str_provider_batch_id,
            )
            batch_status = AIBatchStatus.IN_PROGRESS
        # Normal return with the normalized batch job handle.
        return AIBatchJob(
            batch_id=f"{self.BATCH_ID_PREFIX}{str_provider_batch_id}",
            provider_batch_id=str_provider_batch_id,
            status=batch_status,
            request_count=int_request_count,
            succeeded_count=int_succeeded,
            errored_count=int_errored,
            canceled_count=int_canceled,
            expired_count=int_expired,
            processing_count=int_processing,
            submitted_at_utc=getattr(batch, "created_at", None),
            ended_at_utc=getattr(batch, "ended_at", None),
            provider_engine=self.PROVIDER_ENGINE_CLAUDE,
            provider_model_name=self.completions_model,
        )

    def _normalize_batch_result_item(self, result: Any) -> AIBatchResultItem:
        """Map one Anthropic batch result entry to AIBatchResultItem."""
        str_custom_id: str = str(getattr(result, "custom_id", "") or "")
        inner = getattr(result, "result", None)
        str_result_type: str = str(getattr(inner, "type", "") or "")
        item_status: AIBatchItemStatus = self.DICT_BATCH_RESULT_TYPE.get(
            str_result_type, AIBatchItemStatus.ERRORED
        )
        str_text: str | None = None
        str_error: str | None = None
        int_prompt_tokens: int | None = None
        int_completion_tokens: int | None = None
        if item_status is AIBatchItemStatus.SUCCEEDED:
            message = getattr(inner, "message", None)
            str_text = self.pii_middleware.process_output(
                self._extract_response_text(message) if message is not None else ""
            )
            # Index rather than unpack so this survives the usage tuple gaining
            # a cached-token element (finops cached-token work); the first two
            # positions are always (prompt, completion).
            tuple_usage = self._extract_anthropic_usage(message)
            int_prompt_tokens = tuple_usage[0]
            int_completion_tokens = tuple_usage[1]
        else:
            error = getattr(inner, "error", None)
            if error is not None:
                str_error = str(getattr(error, "type", "") or "") or str(error)
        # Normal return with the normalized per-request result.
        return AIBatchResultItem(
            custom_id=str_custom_id,
            status=item_status,
            text=str_text,
            error_message=str_error,
            provider_prompt_tokens=int_prompt_tokens,
            provider_completion_tokens=int_completion_tokens,
        )

    def _submit_batch_provider(self, requests: list[AIBatchRequestItem]) -> AIBatchJob:
        """Create an Anthropic message batch from the request items."""
        list_batch_requests: list[dict[str, Any]] = []
        # Loop through request items so each prompt is redacted and shaped for the API.
        for item in requests:
            str_prompt: str = self.pii_middleware.process_input(item.prompt)
            str_system_prompt: str = (
                item.system_prompt
                if item.system_prompt is not None
                else AICompletionsPromptParamsBase.DEFAULT_SYSTEM_PROMPT
            )
            dict_params: dict[str, Any] = {
                "model": self.completions_model,
                "max_tokens": item.max_response_tokens or self.SEND_PROMPT_MAX_TOKENS,
                "system": str_system_prompt,
                "messages": [{"role": "user", "content": str_prompt}],
            }
            list_batch_requests.append(
                {"custom_id": item.custom_id, "params": dict_params}
            )
        batch = self.client.messages.batches.create(requests=list_batch_requests)
        # Normal return with the normalized submitted batch job.
        return self._normalize_batch_job(batch)

    def _get_batch_provider(self, batch_id: str) -> AIBatchJob:
        """Retrieve current status for an Anthropic message batch."""
        batch = self.client.messages.batches.retrieve(
            self._to_provider_batch_id(batch_id)
        )
        # Normal return with the refreshed batch job.
        return self._normalize_batch_job(batch)

    def _cancel_batch_provider(self, batch_id: str) -> AIBatchJob:
        """Request cancellation of an Anthropic message batch."""
        batch = self.client.messages.batches.cancel(
            self._to_provider_batch_id(batch_id)
        )
        # Normal return with the batch job after requesting cancellation.
        return self._normalize_batch_job(batch)

    def _get_batch_results_provider(self, batch_id: str) -> list[AIBatchResultItem]:
        """Stream and normalize results for an ended Anthropic message batch."""
        list_results: list[AIBatchResultItem] = []
        # Loop through the streamed result entries, keyed by custom_id.
        for result in self.client.messages.batches.results(
            self._to_provider_batch_id(batch_id)
        ):
            list_results.append(self._normalize_batch_result_item(result))
        # Normal return with the per-request batch results.
        return list_results
