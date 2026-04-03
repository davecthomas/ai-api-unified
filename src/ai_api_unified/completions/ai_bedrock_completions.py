# ai_bedrock_completions.py

import json
import logging
from typing import Any, ClassVar, Type

from pydantic import ValidationError

from ..ai_completions_exceptions import StructuredResponseTokenLimitError
from ..ai_base import (
    AIBaseCompletions,
    AiApiObservedCompletionsResultModel,
    AIStructuredPrompt,
    AICompletionsPromptParamsBase,
    SupportedDataType,
)
from ..ai_bedrock_base import AIBedrockBase, ClientError
from ..middleware.observability_runtime import ObservabilityMetadataValue
from ..util.env_settings import EnvSettings

_LOGGER: logging.Logger = logging.getLogger(__name__)


class AiBedrockCompletions(AIBedrockBase, AIBaseCompletions):
    """
    Completion client for Amazon Bedrock via the Converse API, with
    structured-output prompts.
    """

    def __init__(self, model: str = "", **kwargs: Any):
        settings = EnvSettings()
        resolved_model: str = (
            model
            if model
            else settings.get("COMPLETIONS_MODEL_NAME", "amazon.nova-lite-v1:0")
        )
        self.completions_model: str = resolved_model
        AIBedrockBase.__init__(self, model=resolved_model, **kwargs)
        AIBaseCompletions.__init__(self, model=resolved_model, **kwargs)
        # Reuse the existing retry schedule from the base but allow overrides via kwargs
        custom_backoff: list[float] | None = kwargs.get("backoff_delays")
        if custom_backoff is not None:
            self.backoff_delays = custom_backoff

    @property
    def max_context_tokens(self) -> int:
        """
        Look up the Bedrock model context window for the current model_name.
        Falls back to 0 if unknown (i.e. no guard will occur).
        """
        return self.DICT_CONTEXT_WINDOWS.get(self.completions_model, 0)

    @property
    def price_per_1k_tokens(self) -> float:
        """
        Look up the cost-per-1 k tokens for this model.
        Returns 0.0 if unknown (no guard).
        """
        return self.DICT_PRICES.get(self.completions_model, 0.0)

    DICT_CONTEXT_WINDOWS: dict[str, int] = {
        "amazon.nova-micro-v1:0": 4_096_000,
        "amazon.nova-lite-v1:0": 8_192_000,
        "amazon.nova-pro-v1:0": 16_384_000,
        "amazon.nova-premier-v1:0": 32_768_000,
        "us.anthropic.claude-3-5-haiku-20241022-v1:0": 8_192_000,
    }

    # dollars per 1 k tokens for each supported model
    DICT_PRICES: ClassVar[dict[str, float]] = {
        "amazon.nova-micro-v1:0": 0.0004,
        "amazon.nova-lite-v1:0": 0.0008,
        "amazon.nova-pro-v1:0": 0.0016,
        "amazon.nova-premier-v1:0": 0.0032,
        "us.anthropic.claude-3-5-haiku-20241022-v1:0": 0.0015,
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
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> str:
        prompt = self.pii_middleware.process_input(prompt)
        # AWS Bedrock expects messages in this format for the Converse API
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

    @staticmethod
    def _extract_bedrock_prompt_tokens(response: dict[str, Any]) -> int | None:
        """
        Returns provider-reported prompt token counts from one Bedrock converse response.

        Args:
            response: Bedrock converse response dictionary.

        Returns:
            Provider-reported prompt token count when available, otherwise None.
        """
        usage: dict[str, Any] = response.get("usage", {})
        prompt_tokens: int | None = usage.get("inputTokens")
        # Normal return with provider prompt token usage when present.
        return prompt_tokens

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
        Returns provider-reported total token counts from one Bedrock converse response.

        Args:
            response: Bedrock converse response dictionary.

        Returns:
            Provider-reported total token count when available, otherwise None.
        """
        usage: dict[str, Any] = response.get("usage", {})
        total_tokens: int | None = usage.get("totalTokens")
        # Normal return with provider total token usage when present.
        return total_tokens

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
