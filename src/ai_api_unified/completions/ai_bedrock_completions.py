# ai_bedrock_completions.py

import json
import logging
from collections.abc import Iterator
from typing import Any, ClassVar, Type

from pydantic import ValidationError

from ..ai_completions_exceptions import StructuredResponseTokenLimitError
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
    RETRY_POLICY_DEFAULT,
    RETRY_POLICY_NONE,
    SupportedDataType,
    normalize_retry_policy,
)
from ..ai_bedrock_base import AIBedrockBase, BotoCoreError, ClientError
from ..ai_provider_exceptions import (
    AiProviderCapabilityUnsupportedError,
    AiProviderRequestError,
)
from ..middleware.observability_runtime import (
    AiApiCallResultSummaryModel,
    ObservabilityMetadataValue,
)
from ..pricing.pricing_registry import (
    PROVIDER_BEDROCK,
    enforce_model_lifecycle,
    get_model_pricing,
)
from ..util.env_settings import EnvSettings

_LOGGER: logging.Logger = logging.getLogger(__name__)


class AICompletionsCapabilitiesBedrock(AICompletionsCapabilitiesBase):
    """
    Bedrock-specific completions capabilities.

    Based on https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
    """

    # Model families with Converse toolConfig support that this engine maps
    # (Amazon Nova and Anthropic Claude on Bedrock).
    TUPLE_TOOL_USE_MODEL_MARKERS: ClassVar[tuple[str, ...]] = (
        "nova",
        "anthropic.claude",
    )
    # Models AWS lists as supporting native structured outputs
    # (Converse outputConfig.textFormat); see
    # https://docs.aws.amazon.com/bedrock/latest/userguide/structured-output.html
    TUPLE_STRUCTURED_OUTPUT_MODEL_MARKERS: ClassVar[tuple[str, ...]] = (
        "claude-haiku-4-5",
        "claude-sonnet-4-5",
        "claude-opus-4-5",
        "claude-opus-4-6",
    )

    @classmethod
    def for_model(
        cls,
        model_name: str,
        *,
        context_window_length: int,
    ) -> "AICompletionsCapabilitiesBedrock":
        """Create capabilities instance for the requested Bedrock model."""
        normalized_name: str = model_name.strip().lower()
        # Tool use and structured output are per-model on Bedrock: toolConfig
        # is mapped for Nova and Claude families; outputConfig structured
        # output only for the models AWS supports. Async stays False — boto3
        # has no official async client.
        bool_supports_tool_use: bool = any(
            marker in normalized_name for marker in cls.TUPLE_TOOL_USE_MODEL_MARKERS
        )
        bool_supports_structured_output: bool = any(
            marker in normalized_name
            for marker in cls.TUPLE_STRUCTURED_OUTPUT_MODEL_MARKERS
        )
        # Normal return with Bedrock capabilities; every chat model this client
        # targets streams via the ConverseStream API and supports the
        # provider-side CountTokens operation.
        return cls(
            context_window_length=context_window_length,
            supported_data_types=[SupportedDataType.TEXT, SupportedDataType.IMAGE],
            supports_streaming=True,
            supports_token_counting=True,
            supports_tool_use=bool_supports_tool_use,
            supports_structured_output=bool_supports_structured_output,
            pricing=get_model_pricing(PROVIDER_BEDROCK, model_name),
        )


class AiBedrockCompletions(AIBedrockBase, AIBaseCompletions):
    """
    Completion client for Amazon Bedrock via the Converse API, with
    structured-output prompts.
    """

    def __init__(
        self, model: str = "", *, retry_policy: str | None = None, **kwargs: Any
    ):
        settings = EnvSettings()
        resolved_model: str = (
            model
            if model
            else settings.get("COMPLETIONS_MODEL_NAME", "amazon.nova-lite-v1:0")
        )
        self.completions_model: str = resolved_model
        enforce_model_lifecycle(PROVIDER_BEDROCK, resolved_model)
        str_retry_candidate: str = (
            retry_policy
            if retry_policy is not None
            else str(settings.get("COMPLETIONS_RETRY_POLICY", RETRY_POLICY_DEFAULT))
        )
        self.retry_policy: str = normalize_retry_policy(str_retry_candidate)
        AIBedrockBase.__init__(self, model=resolved_model, **kwargs)
        AIBaseCompletions.__init__(self, model=resolved_model, **kwargs)
        # Reuse the existing retry schedule from the base but allow overrides via kwargs
        custom_backoff: list[float] | None = kwargs.get("backoff_delays")
        if custom_backoff is not None:
            self.backoff_delays = custom_backoff
        if self.retry_policy == RETRY_POLICY_NONE:
            # A single-entry schedule collapses every engine retry loop to one
            # attempt. botocore's own retry config is client-level; set the
            # standard AWS_MAX_ATTEMPTS=1 environment variable to disable it.
            self.backoff_delays = [0.0]

    @property
    def max_context_tokens(self) -> int:
        """
        Look up the Bedrock model context window for the current model_name.
        Falls back to 0 if unknown (i.e. no guard will occur).
        """
        return self.DICT_CONTEXT_WINDOWS.get(self.completions_model, 0)

    DICT_CONTEXT_WINDOWS: dict[str, int] = {
        "amazon.nova-micro-v1:0": 4_096_000,
        "amazon.nova-lite-v1:0": 8_192_000,
        "amazon.nova-pro-v1:0": 16_384_000,
        "amazon.nova-premier-v1:0": 32_768_000,
        "us.anthropic.claude-3-5-haiku-20241022-v1:0": 8_192_000,
    }

    @property
    def list_model_names(self) -> list[str]:
        # As of Oct 2025, aggregated from AWS release notes:
        return [
            "amazon.nova-micro-v1:0",
            "amazon.nova-lite-v1:0",
            "amazon.nova-pro-v1:0",
            "amazon.nova-premier-v1:0",
            "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        ]

    @property
    def capabilities(self) -> AICompletionsCapabilitiesBedrock:
        """Return model capabilities for the current Bedrock model."""
        return AICompletionsCapabilitiesBedrock.for_model(
            self.completions_model,
            context_window_length=self.max_context_tokens,
        )

    def _extract_json_text_from_converse_response(self, resp: dict[str, Any]) -> str:
        content = resp.get("output", {}).get("message", {}).get("content", [])
        if not content or "text" not in content[0]:
            raise RuntimeError("No text in response")
        return content[0]["text"]

    def strict_schema_prompt(
        self,
        prompt: str,
        response_model: Type[AIStructuredPrompt],
        max_response_tokens: int = AIBaseCompletions.STRUCTURED_DEFAULT_MAX_RESPONSE_TOKENS,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> AIStructuredPrompt:
        """
        Free-form JSON generation with Pydantic v2 post-validation.
        Guarantees variety by using sampling instead of tool-locking.
        """
        self._validate_structured_max_response_tokens(
            provider_name=self.PROVIDER_VENDOR_BEDROCK,
            model_name=self.completions_model,
            max_response_tokens=max_response_tokens,
        )

        prompt += self.generate_prompt_entropy_tag()

        # 1) Auto-append the generic JSON-schema instruction—no assumptions
        prompt_addendum: str = self.generate_prompt_addendum_json_schema_instruction(
            response_model, code_fence=False
        )
        full_prompt = f"{prompt}\n\n{prompt_addendum}"
        full_prompt = self.pii_middleware.process_input(full_prompt)

        system_prompt: str = (
            other_params.system_prompt
            if other_params is not None and other_params.system_prompt is not None
            else AICompletionsPromptParamsBase.DEFAULT_STRICT_SCHEMA_SYSTEM_PROMPT
        )

        # 2. Build plain messages
        user_content: list[dict[str, Any]] = self._build_user_content(
            full_prompt, other_params
        )
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"text": "```json"}]},
        ]

        # 3. Sampling settings for maximum variety
        inference_config = {
            "maxTokens": max_response_tokens,
            "temperature": 0.9,  # high randomness
            "stopSequences": ["```"],  # stop at JSON end marker
            # "topP": 0.9,  # nucleus sampling
        }
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = (
            self._build_completions_observability_input_metadata(
                prompt=full_prompt,
                system_prompt=system_prompt,
                other_params=other_params,
                response_mode=self.RESPONSE_MODE_STRUCTURED,
                max_response_tokens=max_response_tokens,
            )
        )

        def _execute_structured_prompt() -> (
            AiApiObservedCompletionsResultModel[AIStructuredPrompt]
        ):
            raw_json: str = ""
            parsed: dict[str, Any] | list[Any] | None = None
            # Loop through Bedrock retries while preserving existing JSON repair and validation behavior.
            for attempt, delay in enumerate(self.backoff_delays, start=1):
                try:
                    resp = self.client.converse(
                        modelId=self.model,
                        messages=messages,
                        system=[{"text": system_prompt}],
                        inferenceConfig=inference_config,
                    )
                    str_stop_reason: str = (
                        str(resp.get("stopReason", "")).strip().lower()
                    )
                    try:
                        raw_json = self._extract_json_text_from_converse_response(resp)
                    except RuntimeError:
                        raw_json = ""
                        if str_stop_reason == "max_tokens":
                            self._raise_structured_token_limit_error(
                                provider_name=self.PROVIDER_VENDOR_BEDROCK,
                                model_name=self.completions_model,
                                max_response_tokens=max_response_tokens,
                                finish_reason=str_stop_reason,
                                raw_output_text=raw_json,
                            )
                        raise
                    raw_json = raw_json.rstrip("```").strip()
                    if str_stop_reason == "max_tokens":
                        self._raise_structured_token_limit_error(
                            provider_name=self.PROVIDER_VENDOR_BEDROCK,
                            model_name=self.completions_model,
                            max_response_tokens=max_response_tokens,
                            finish_reason=str_stop_reason,
                            raw_output_text=raw_json,
                        )
                    sanitized_json: str = self.pii_middleware.process_output(raw_json)
                    parsed = json.loads(sanitized_json)
                    if isinstance(parsed, dict) and "properties" in parsed:
                        parsed = parsed["properties"]
                    validated_response: AIStructuredPrompt = (
                        response_model.model_validate(parsed)
                    )
                    observed_result: AiApiObservedCompletionsResultModel[
                        AIStructuredPrompt
                    ] = AiApiObservedCompletionsResultModel(
                        return_value=validated_response,
                        raw_output_text=raw_json,
                        finish_reason=str_stop_reason,
                        provider_prompt_tokens=self._extract_bedrock_prompt_tokens(
                            resp
                        ),
                        provider_completion_tokens=self._extract_bedrock_completion_tokens(
                            resp
                        ),
                        provider_cached_input_tokens=self._extract_bedrock_cached_tokens(
                            resp
                        ),
                        provider_total_tokens=self._extract_bedrock_total_tokens(resp),
                    )
                    # Normal return with validated Bedrock structured output and raw provider metadata.
                    return observed_result

                except json.JSONDecodeError as json_decode_error:
                    fixed: str = self._repair_json(raw_json)
                    try:
                        fixed_sanitized: str = self.pii_middleware.process_output(fixed)
                        parsed = json.loads(fixed_sanitized)
                        validated_response = response_model.model_validate(parsed)
                        return AiApiObservedCompletionsResultModel(
                            return_value=validated_response,
                            raw_output_text=raw_json,
                            finish_reason=str_stop_reason,
                            provider_prompt_tokens=self._extract_bedrock_prompt_tokens(
                                resp
                            ),
                            provider_completion_tokens=self._extract_bedrock_completion_tokens(
                                resp
                            ),
                            provider_cached_input_tokens=self._extract_bedrock_cached_tokens(
                                resp
                            ),
                            provider_total_tokens=self._extract_bedrock_total_tokens(
                                resp
                            ),
                        )
                    except json.JSONDecodeError:
                        if attempt < len(self.backoff_delays):
                            self._sleep_with_backoff(delay)
                            continue
                        raise RuntimeError(
                            "JSON parse (even after repair) failed after "
                            f"{attempt} tries: {json_decode_error}. "
                            f"raw_json_char_count={len(raw_json)}"
                        ) from json_decode_error

                except ValidationError as validation_error:
                    errors: list[dict[str, Any]] = validation_error.errors()
                    _LOGGER.error(
                        "Bedrock validation failed: model=%s raw_json_char_count=%s parsed_type=%s error_count=%s",
                        self.completions_model,
                        len(raw_json),
                        type(parsed).__name__,
                        len(errors),
                    )
                    raise RuntimeError(
                        f"{response_model.__name__}.model_validate() failed with "
                        f"{len(errors)} validation errors. "
                        f"parsed_type={type(parsed).__name__} "
                        f"raw_json_char_count={len(raw_json)}"
                    ) from None

                except StructuredResponseTokenLimitError:
                    raise

                except self.client.exceptions.ModelErrorException as model_error:
                    _LOGGER.warning(
                        "Bedrock model error on attempt %s: %s",
                        attempt,
                        model_error,
                    )
                    if attempt < len(self.backoff_delays):
                        self._sleep_with_backoff(delay)
                        continue
                    raise RuntimeError(
                        f"Bedrock ModelErrorException after {attempt} tries: {model_error}"
                    ) from model_error

                except ClientError as client_error:
                    if attempt < len(self.backoff_delays):
                        self._sleep_with_backoff(delay)
                        continue
                    raise RuntimeError(
                        f"Bedrock error after {attempt} tries: {client_error}"
                    ) from client_error

                except Exception as exception:
                    if "MAX_TOKENS" in str(exception).upper():
                        _LOGGER.error(
                            "Bedrock structured prompt failed due to MAX_TOKENS for model=%s with max_response_tokens=%s.",
                            self.completions_model,
                            max_response_tokens,
                        )
                    if attempt < len(self.backoff_delays):
                        self._sleep_with_backoff(delay)
                        continue
                    raise RuntimeError(
                        f"Unexpected failure after {attempt} tries: {exception}"
                    ) from exception

            raise RuntimeError(
                "Bedrock structured prompt exhausted retries unexpectedly."
            )

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
            )
        )
        # Normal return with validated Bedrock structured output after observability wrapping.
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
        # boto3 has no per-call timeout; that parameter stays unimplemented on
        # this engine and raises the typed capability error when supplied.
        self._reject_bedrock_timeout(request_timeout_seconds)
        prompt = self.pii_middleware.process_input(prompt)
        # AWS Bedrock expects messages in this format for the Converse API
        user_content: list[dict[str, Any]] = self._build_user_content(
            prompt, other_params
        )
        messages = [{"role": "user", "content": user_content}]

        system_prompt = self._resolve_system_prompt(
            system_prompt,
            other_params,
            AICompletionsPromptParamsBase.DEFAULT_SYSTEM_PROMPT,
        )

        inference_config = {
            "temperature": 0.2,
            "topP": 0.85,
            "maxTokens": max_response_tokens or 256,
        }
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
            # Loop through Bedrock text retries while preserving existing backoff behavior.
            for attempt, delay in enumerate(self.backoff_delays, start=1):
                try:
                    response = self.client.converse(
                        modelId=self.model,
                        messages=messages,
                        system=[{"text": system_prompt}],
                        inferenceConfig=inference_config,
                    )
                    content = (
                        response.get("output", {}).get("message", {}).get("content", [])
                    )
                    raw_output_text: str = ""
                    if content and "text" in content[0]:
                        raw_output_text = content[0]["text"]
                    observed_result: AiApiObservedCompletionsResultModel[str] = (
                        AiApiObservedCompletionsResultModel(
                            return_value=raw_output_text,
                            raw_output_text=raw_output_text,
                            finish_reason=str(response.get("stopReason", "")),
                            provider_prompt_tokens=self._extract_bedrock_prompt_tokens(
                                response
                            ),
                            provider_completion_tokens=self._extract_bedrock_completion_tokens(
                                response
                            ),
                            provider_cached_input_tokens=self._extract_bedrock_cached_tokens(
                                response
                            ),
                            provider_total_tokens=self._extract_bedrock_total_tokens(
                                response
                            ),
                        )
                    )
                    # Normal return with raw Bedrock text output and provider usage metadata.
                    return observed_result
                except Exception as exception:
                    if attempt == len(self.backoff_delays):
                        raise RuntimeError(
                            f"Bedrock converse failed: {exception}"
                        ) from exception
                    self._sleep_with_backoff(delay)

            raise RuntimeError("Bedrock text prompt exhausted retries unexpectedly.")

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
            )
        )
        sanitized_output: str = self.pii_middleware.process_output(
            observed_result.return_value
        )
        # Normal return with sanitized Bedrock text output after observability wrapping.
        return sanitized_output

    def _send_prompt_streaming_provider(
        self,
        prompt: str,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> Iterator[str]:
        """
        Stream a text prompt response from Bedrock's ConverseStream API chunk by chunk.

        Capability and PII gating already ran in the base template method.
        There is no retry wrapper: retrying a partially consumed stream would
        duplicate output.

        Args:
            prompt: Validated text prompt to send.
            other_params: Optional provider-specific parameters.

        Returns:
            Iterator of response text chunks in provider order.
        """
        user_content: list[dict[str, Any]] = self._build_user_content(
            prompt, other_params
        )
        messages = [{"role": "user", "content": user_content}]
        system_prompt: str = (
            other_params.system_prompt
            if other_params is not None and other_params.system_prompt is not None
            else AICompletionsPromptParamsBase.DEFAULT_SYSTEM_PROMPT
        )
        inference_config = {"temperature": 0.2, "topP": 0.85, "maxTokens": 256}
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
            "metadata_event": None,
            "bool_completed": False,
        }

        def _open_stream() -> Iterator[str]:
            response = self.client.converse_stream(
                modelId=self.model,
                messages=messages,
                system=[{"text": system_prompt}],
                inferenceConfig=inference_config,
            )
            # Loop through ConverseStream events so callers see text as it arrives.
            for event in response.get("stream", []):
                if "contentBlockDelta" in event:
                    str_chunk_text: str = (
                        event["contentBlockDelta"].get("delta", {}).get("text", "")
                    )
                    if str_chunk_text:
                        dict_stream_state["int_chunk_count"] += 1
                        dict_stream_state["list_text_parts"].append(str_chunk_text)
                        yield str_chunk_text
                elif "messageStop" in event:
                    dict_stream_state["finish_reason"] = str(
                        event["messageStop"].get("stopReason", "")
                    )
                elif "metadata" in event:
                    dict_stream_state["metadata_event"] = event["metadata"]
            dict_stream_state["bool_completed"] = True

        def _build_summary(provider_elapsed_ms: float) -> AiApiCallResultSummaryModel:
            # ConverseStream reports usage in the terminal metadata event using the
            # same shape as the synchronous converse response.
            dict_usage_response: dict[str, Any] = (
                dict_stream_state["metadata_event"] or {}
            )
            str_full_text: str = "".join(dict_stream_state["list_text_parts"])
            observed_result: AiApiObservedCompletionsResultModel[str] = (
                AiApiObservedCompletionsResultModel(
                    return_value=str_full_text,
                    raw_output_text=str_full_text,
                    finish_reason=dict_stream_state["finish_reason"],
                    provider_prompt_tokens=self._extract_bedrock_prompt_tokens(
                        dict_usage_response
                    ),
                    provider_completion_tokens=self._extract_bedrock_completion_tokens(
                        dict_usage_response
                    ),
                    provider_cached_input_tokens=self._extract_bedrock_cached_tokens(
                        dict_usage_response
                    ),
                    provider_total_tokens=self._extract_bedrock_total_tokens(
                        dict_usage_response
                    ),
                )
            )
            # Normal return with the streaming summary built from accumulated events.
            return self._build_streaming_completions_observability_result_summary(
                observed_result=observed_result,
                provider_elapsed_ms=provider_elapsed_ms,
                int_chunk_count=dict_stream_state["int_chunk_count"],
                bool_stream_completed=dict_stream_state["bool_completed"],
            )

        # Normal return with the observability-wrapped Bedrock stream.
        return self._execute_streaming_provider_call_with_observability(
            operation="send_prompt_streaming",
            dict_input_metadata=dict_input_metadata,
            callable_open_stream=_open_stream,
            callable_build_result_summary=_build_summary,
        )

    def _count_tokens_provider(
        self,
        prompt: str,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> int:
        """
        Count input tokens via the Bedrock CountTokens operation.

        Builds the same Converse-shaped request send_prompt would submit and
        asks Bedrock to count its input tokens without running inference.

        Args:
            prompt: Validated text prompt to measure.
            other_params: Optional provider-specific parameters.

        Returns:
            Provider-reported input token count.
        """
        user_content: list[dict[str, Any]] = self._build_user_content(
            prompt, other_params
        )
        system_prompt: str = (
            other_params.system_prompt
            if other_params is not None and other_params.system_prompt is not None
            else AICompletionsPromptParamsBase.DEFAULT_SYSTEM_PROMPT
        )
        dict_converse_input: dict[str, Any] = {
            "messages": [{"role": "user", "content": user_content}],
            "system": [{"text": system_prompt}],
        }
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = (
            self._build_completions_observability_input_metadata(
                prompt=prompt,
                system_prompt=system_prompt,
                other_params=other_params,
                response_mode=self.RESPONSE_MODE_TEXT,
            )
        )

        def _execute_count_tokens() -> AiApiObservedCompletionsResultModel[int]:
            response = self.client.count_tokens(
                modelId=self.model,
                input={"converse": dict_converse_input},
            )
            int_input_tokens: int = int(response.get("inputTokens", 0))
            observed_result: AiApiObservedCompletionsResultModel[int] = (
                AiApiObservedCompletionsResultModel(
                    return_value=int_input_tokens,
                    raw_output_text="",
                    provider_prompt_tokens=int_input_tokens,
                    provider_total_tokens=int_input_tokens,
                    dict_metadata={"operation": "count_tokens"},
                )
            )
            # Normal return with the provider-counted input token total.
            return observed_result

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
            )
        )
        # Normal return with the caller-facing input token count.
        return observed_result.return_value

    @staticmethod
    def _extract_bedrock_prompt_tokens(response: dict[str, Any]) -> int | None:
        """
        Returns provider-reported prompt token counts from one Bedrock converse response.

        Converse reports `inputTokens` exclusive of prompt-cache reads
        (`cacheReadInputTokens`); folds cache reads into the prompt count so the
        library-wide convention holds (cached is a subset of prompt tokens).

        Args:
            response: Bedrock converse response dictionary.

        Returns:
            Provider-reported prompt token count when available, otherwise None.
        """
        usage: dict[str, Any] = response.get("usage", {})
        input_tokens: int | None = usage.get("inputTokens")
        cached_tokens: int | None = usage.get("cacheReadInputTokens")
        if input_tokens is None and cached_tokens is None:
            # Early return because the response carried no input token usage.
            return None
        # Normal return with prompt tokens including any cached-read subset.
        return (input_tokens or 0) + (cached_tokens or 0)

    @staticmethod
    def _extract_bedrock_cached_tokens(response: dict[str, Any]) -> int | None:
        """
        Returns provider-reported cached prompt token counts from a Bedrock response.

        Args:
            response: Bedrock converse response dictionary.

        Returns:
            Provider-reported cache-read token count when available, otherwise None.
        """
        usage: dict[str, Any] = response.get("usage", {})
        # Normal return with provider cache-read token usage when present.
        return usage.get("cacheReadInputTokens")

    @staticmethod
    def _extract_bedrock_completion_tokens(response: dict[str, Any]) -> int | None:
        """
        Returns provider-reported completion token counts from one Bedrock converse response.

        Args:
            response: Bedrock converse response dictionary.

        Returns:
            Provider-reported completion token count when available, otherwise None.
        """
        usage: dict[str, Any] = response.get("usage", {})
        completion_tokens: int | None = usage.get("outputTokens")
        # Normal return with provider completion token usage when present.
        return completion_tokens

    @staticmethod
    def _extract_bedrock_total_tokens(response: dict[str, Any]) -> int | None:
        """
        Returns the total token count for one Bedrock converse response.

        AWS reports `totalTokens` as `inputTokens + outputTokens`, both exclusive
        of cache reads. Because `_extract_bedrock_prompt_tokens` folds cache reads
        into the prompt count, the total is recomputed from the folded prompt plus
        output so the emitted triple stays consistent (prompt + completion =
        total). With no cache reads this equals the provider `totalTokens`.

        Args:
            response: Bedrock converse response dictionary.

        Returns:
            Total token count when available, otherwise None.
        """
        prompt_tokens: int | None = AiBedrockCompletions._extract_bedrock_prompt_tokens(
            response
        )
        completion_tokens: int | None = (
            AiBedrockCompletions._extract_bedrock_completion_tokens(response)
        )
        if prompt_tokens is None and completion_tokens is None:
            # Early return because the response carried no token usage to total.
            return None
        # Normal return with the folded prompt plus output token total.
        return (prompt_tokens or 0) + (completion_tokens or 0)

    def _build_user_content(
        self,
        prompt: str,
        other_params: AICompletionsPromptParamsBase | None,
    ) -> list[dict[str, Any]]:
        """
        Build AWS Bedrock message content supporting optional image attachments.
        """

        content_parts: list[dict[str, Any]] = [{"text": prompt}]
        if other_params is None or not other_params.has_included_media:
            return content_parts

        for (
            _,
            media_type,
            media_bytes,
            mime_type,
        ) in other_params.iter_included_media():
            if media_type is not SupportedDataType.IMAGE:
                continue

            if len(media_bytes) > AICompletionsPromptParamsBase.MAX_IMAGE_BYTES:
                raise ValueError(
                    "Image attachment exceeds the maximum allowed size of "
                    f"{AICompletionsPromptParamsBase.MAX_IMAGE_BYTES} bytes."
                )

            content_parts.append(
                {
                    "image": {
                        "format": self._derive_image_format(mime_type),
                        "source": {"bytes": media_bytes},
                    }
                }
            )

        return content_parts

    @staticmethod
    def _derive_image_format(mime_type: str) -> str:
        """
        Extract the provider-specific image format token from a MIME type.
        """

        return mime_type.split("/", maxsplit=1)[-1]

    @staticmethod
    def _repair_json(raw_json: str) -> str:
        """
        Attempt a minimal repair of malformed JSON returned by the model.

        The current implementation simply returns the original string, acting as a
        placeholder hook for more advanced repair strategies if needed later.
        """

        return raw_json

    # ── Conversation and structured output (Converse API, 2.15.0) ───────────
    # Async variants and per-call timeouts stay unimplemented on this engine:
    # boto3 has no official async client and no per-call timeout override.

    DICT_FINISH_REASON_MAP: ClassVar[dict[str, AIFinishReason]] = {
        "end_turn": AIFinishReason.COMPLETE,
        "stop_sequence": AIFinishReason.COMPLETE,
        "max_tokens": AIFinishReason.LENGTH,
        "tool_use": AIFinishReason.TOOL_USE,
        "guardrail_intervened": AIFinishReason.REFUSAL,
        "content_filtered": AIFinishReason.REFUSAL,
    }

    PROVIDER_ENGINE_TOKEN: ClassVar[str] = "bedrock"

    def _reject_bedrock_timeout(self, request_timeout_seconds: float | None) -> None:
        """
        Rejects per-call timeouts, which boto3 does not support.
        """
        if request_timeout_seconds is None:
            # Normal return because no unsupported parameter was supplied.
            return None
        raise AiProviderCapabilityUnsupportedError(
            f"{type(self).__name__} model '{self.model_name}' does not support "
            "request_timeout_seconds: boto3 has no per-call timeout. Configure "
            "botocore read_timeout at client construction instead."
        )

    def _raise_bedrock_request_error(self, exception: Exception) -> None:
        """
        Re-raises one botocore transport error as the typed request error.

        Carries the HTTP status code from ClientError response metadata so
        caller-owned backoff can classify 429/5xx uniformly across engines.
        Non-transport exceptions propagate unchanged.
        """
        if isinstance(exception, ClientError):
            dict_response: dict[str, Any] = getattr(exception, "response", None) or {}
            raw_status = dict_response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            raise AiProviderRequestError(
                f"Bedrock request failed: {exception}",
                status_code=raw_status if isinstance(raw_status, int) else None,
                provider_engine=self.PROVIDER_ENGINE_TOKEN,
            ) from exception
        if isinstance(exception, BotoCoreError):
            raise AiProviderRequestError(
                f"Bedrock request failed before a status was available: "
                f"{exception}",
                status_code=None,
                provider_engine=self.PROVIDER_ENGINE_TOKEN,
            ) from exception
        # Normal return so non-transport exceptions propagate unchanged.
        return None

    def _is_retryable_client_error(self, exception: Exception) -> bool:
        """
        Classifies one exception as retryable per the base error-code policy.
        """
        if isinstance(exception, ClientError):
            str_error_code: str = str(
                (getattr(exception, "response", None) or {})
                .get("Error", {})
                .get("Code", "")
            )
            # Normal return based on the shared non-retryable code set.
            return str_error_code not in self.NON_RETRYABLE_ERROR_CODES
        # Normal return treating other transport errors as retryable.
        return isinstance(exception, BotoCoreError)

    def _execute_converse_with_retries(
        self,
        callable_converse: Any,
        *,
        retry_override: str | None = None,
    ) -> dict[str, Any]:
        """
        Runs one Converse call through the engine retry schedule.

        retry_override "none" (from provider_options) collapses the schedule
        to a single attempt for that call. Exhausted or non-retryable
        transport failures surface as typed AiProviderRequestError.
        """
        list_delays: list[float] = list(self.backoff_delays)
        if retry_override is not None:
            if normalize_retry_policy(retry_override) == RETRY_POLICY_NONE:
                list_delays = [0.0]
        # Loop through the retry schedule while preserving backoff behavior.
        for attempt, delay in enumerate(list_delays, start=1):
            try:
                # Normal return with the raw Converse response dictionary.
                return callable_converse()
            except Exception as exception:
                bool_last_attempt: bool = attempt == len(list_delays)
                if bool_last_attempt or not self._is_retryable_client_error(exception):
                    self._raise_bedrock_request_error(exception)
                    raise
                self._sleep_with_backoff(delay)
        raise RuntimeError("Unreachable: converse retry schedule was empty.")

    @staticmethod
    def _build_converse_tool_config(
        tools: list[AITool],
        tool_choice: str | None,
    ) -> dict[str, Any]:
        """
        Maps provider-neutral tools onto the Converse toolConfig shape.
        """
        list_provider_tools: list[dict[str, Any]] = []
        # Loop over tool definitions so each maps to a Converse toolSpec.
        for ai_tool in tools:
            dict_tool_spec: dict[str, Any] = {
                "name": ai_tool.name,
                "description": ai_tool.description,
                "inputSchema": {"json": ai_tool.input_schema},
            }
            if ai_tool.strict:
                dict_tool_spec["strict"] = True
            list_provider_tools.append({"toolSpec": dict_tool_spec})
        dict_tool_config: dict[str, Any] = {"tools": list_provider_tools}
        if tool_choice is not None:
            dict_tool_config["toolChoice"] = {"tool": {"name": tool_choice}}
        # Normal return with the Converse-shaped tool configuration.
        return dict_tool_config

    def _usage_from_converse(self, dict_response: dict[str, Any]) -> AITokenUsage:
        """
        Builds the provider-neutral usage model from one Converse response.
        """
        # Normal return with the provider-neutral token usage model.
        return AITokenUsage(
            input_tokens=self._extract_bedrock_prompt_tokens(dict_response),
            output_tokens=self._extract_bedrock_completion_tokens(dict_response),
            cached_input_tokens=self._extract_bedrock_cached_tokens(dict_response),
            total_tokens=self._extract_bedrock_total_tokens(dict_response),
        )

    def _build_turn_result_from_converse(
        self, dict_response: dict[str, Any]
    ) -> AITurnResult:
        """
        Maps one Converse response onto the provider-neutral turn.

        Converse content blocks are already plain dictionaries, so raw_content
        is the output message content list, replayable verbatim inside the
        next assistant message.
        """
        list_content: list[dict[str, Any]] = (
            dict_response.get("output", {}).get("message", {}).get("content", []) or []
        )
        list_text_parts: list[str] = []
        list_tool_calls: list[AIToolCall] = []
        # Loop over content blocks so text and tool calls align with replay.
        for dict_block in list_content:
            if "text" in dict_block:
                list_text_parts.append(str(dict_block.get("text") or ""))
            elif "toolUse" in dict_block:
                dict_tool_use: dict[str, Any] = dict_block.get("toolUse") or {}
                list_tool_calls.append(
                    AIToolCall(
                        id=str(dict_tool_use.get("toolUseId") or ""),
                        name=str(dict_tool_use.get("name") or ""),
                        input=dict(dict_tool_use.get("input") or {}),
                    )
                )
        str_stop_reason: str = str(dict_response.get("stopReason", "") or "").lower()
        finish_reason: AIFinishReason = self._normalize_finish_reason(
            str_stop_reason if str_stop_reason else None,
            self.DICT_FINISH_REASON_MAP,
        )
        str_text: str = "".join(list_text_parts)
        # Normal return with the provider-neutral turn result.
        return AITurnResult(
            text=str_text if str_text else None,
            tool_calls=list_tool_calls,
            finish_reason=finish_reason,
            raw_content=list_content,
            usage=self._usage_from_converse(dict_response),
        )

    def _observed_converse_turn_result(
        self, dict_response: dict[str, Any]
    ) -> AiApiObservedCompletionsResultModel[AITurnResult]:
        """
        Wraps one Converse response as an observed conversation-turn result.
        """
        turn_result: AITurnResult = self._build_turn_result_from_converse(dict_response)
        # Normal return with the observed turn result and usage metadata.
        return AiApiObservedCompletionsResultModel(
            return_value=turn_result,
            raw_output_text=turn_result.text or "",
            finish_reason=str(dict_response.get("stopReason", "") or "") or None,
            provider_prompt_tokens=turn_result.usage.input_tokens,
            provider_completion_tokens=turn_result.usage.output_tokens,
            provider_cached_input_tokens=turn_result.usage.cached_input_tokens,
            provider_total_tokens=turn_result.usage.total_tokens,
            dict_metadata={"tool_call_count": len(turn_result.tool_calls)},
        )

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
        Sends one conversation turn via the Converse API with toolConfig.
        """
        self._reject_bedrock_timeout(request_timeout_seconds)
        dict_merge_options, str_retry_override = self._split_provider_options(
            provider_options
        )
        dict_request_kwargs: dict[str, Any] = {
            "modelId": self.model,
            "messages": messages,
            "system": [{"text": system_prompt}],
            "inferenceConfig": {"maxTokens": max_response_tokens or 1024},
        }
        if tools:
            dict_request_kwargs["toolConfig"] = self._build_converse_tool_config(
                tools, tool_choice
            )
        dict_request_kwargs.update(dict_merge_options)
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
            dict_response: dict[str, Any] = self._execute_converse_with_retries(
                lambda: self.client.converse(**dict_request_kwargs),
                retry_override=str_retry_override,
            )
            # Normal return with the observed conversation turn.
            return self._observed_converse_turn_result(dict_response)

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
        Builds one Converse toolResult user message.
        """
        dict_tool_result: dict[str, Any] = {
            "toolUseId": tool_call_id,
            "content": [{"json": result}],
        }
        if is_error:
            dict_tool_result["status"] = "error"
        # Normal return with the Converse-shaped tool-result entry.
        return {"role": "user", "content": [{"toolResult": dict_tool_result}]}

    def _extend_messages_with_turn_provider(
        self,
        *,
        messages: list[dict[str, Any]],
        turn: AITurnResult,
    ) -> None:
        """
        Appends one Converse assistant message wrapping the raw content blocks.
        """
        messages.append({"role": "assistant", "content": turn.raw_content})
        # Normal return after appending the assistant turn.
        return None

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
        Generates structured output via Converse outputConfig.textFormat.

        Reachable only on models whose capabilities flag structured-output
        support (AWS's supported-model list; see the capabilities class).
        """
        self._reject_bedrock_timeout(request_timeout_seconds)
        dict_merge_options, str_retry_override = self._split_provider_options(
            provider_options
        )
        str_system_prompt: str = self._resolve_system_prompt(
            system_prompt,
            None,
            AICompletionsPromptParamsBase.DEFAULT_STRICT_SCHEMA_SYSTEM_PROMPT,
        )
        list_messages: list[dict[str, Any]] = list(messages or [])
        if prompt is not None and prompt.strip():
            str_redacted_prompt: str = self.pii_middleware.process_input(prompt)
            list_messages.append(
                {"role": "user", "content": [{"text": str_redacted_prompt}]}
            )
        dict_request_kwargs: dict[str, Any] = {
            "modelId": self.model,
            "messages": list_messages,
            "system": [{"text": str_system_prompt}],
            "inferenceConfig": {"maxTokens": max_response_tokens},
            "outputConfig": {
                "textFormat": {
                    "type": "json_schema",
                    "structure": {
                        "jsonSchema": {
                            "schema": response_schema,
                            "name": "structured_output",
                        }
                    },
                }
            },
        }
        dict_request_kwargs.update(dict_merge_options)
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = {
            "response_mode": self.RESPONSE_MODE_STRUCTURED,
            "prompt_char_count": len(prompt) if prompt else 0,
            "message_count": len(messages) if messages else 0,
            "max_response_tokens": max_response_tokens,
        }

        def _execute_structured() -> (
            AiApiObservedCompletionsResultModel[AIStructuredOutputResult]
        ):
            dict_response: dict[str, Any] = self._execute_converse_with_retries(
                lambda: self.client.converse(**dict_request_kwargs),
                retry_override=str_retry_override,
            )
            str_stop_reason: str = str(
                dict_response.get("stopReason", "") or ""
            ).lower()
            finish_reason: AIFinishReason = self._normalize_finish_reason(
                str_stop_reason if str_stop_reason else None,
                self.DICT_FINISH_REASON_MAP,
            )
            usage: AITokenUsage = self._usage_from_converse(dict_response)
            list_content: list[dict[str, Any]] = (
                dict_response.get("output", {}).get("message", {}).get("content", [])
                or []
            )
            str_raw_text: str = "".join(
                str(dict_block.get("text") or "")
                for dict_block in list_content
                if "text" in dict_block
            )
            if finish_reason in (AIFinishReason.LENGTH, AIFinishReason.REFUSAL):
                structured_result = AIStructuredOutputResult(
                    data=None,
                    finish_reason=finish_reason,
                    usage=usage,
                    raw_text=self.pii_middleware.process_output(str_raw_text),
                )
            else:
                str_content: str = self.pii_middleware.process_output(str_raw_text)
                if not str_content:
                    raise ValueError("Empty response from Bedrock Converse API")
                try:
                    parsed_data: Any = json.loads(str_content)
                except json.JSONDecodeError as json_error:
                    raise ValueError(
                        f"Invalid JSON response: {json_error}"
                    ) from json_error
                if not isinstance(parsed_data, dict):
                    raise ValueError(
                        "Structured response was valid JSON but not a JSON object."
                    )
                structured_result = AIStructuredOutputResult(
                    data=parsed_data,
                    finish_reason=finish_reason,
                    usage=usage,
                    raw_text=str_content,
                )
            # Normal return with the observed structured output result.
            return AiApiObservedCompletionsResultModel(
                return_value=structured_result,
                raw_output_text=structured_result.raw_text,
                finish_reason=str(dict_response.get("stopReason", "") or "") or None,
                provider_prompt_tokens=usage.input_tokens,
                provider_completion_tokens=usage.output_tokens,
                provider_cached_input_tokens=usage.cached_input_tokens,
                provider_total_tokens=usage.total_tokens,
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
        )
        # Normal return with the caller-facing structured output result.
        return observed_result.return_value
