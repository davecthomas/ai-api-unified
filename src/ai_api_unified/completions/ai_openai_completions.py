# ai_openai_completions.py

import base64
import json
import logging
import time
from datetime import date
from typing import Any, ClassVar, Type

from pydantic import ValidationError, model_validator

from ai_api_unified.ai_completions_exceptions import (
    StructuredResponseTokenLimitError,
)
from ai_api_unified.ai_openai_base import AIOpenAIBase

from ..ai_base import (
    AIBaseCompletions,
    AiApiObservedCompletionsResultModel,
    AIStructuredPrompt,
    AICompletionsCapabilitiesBase,
    AICompletionsPromptParamsBase,
    SupportedDataType,
)
from ..middleware.observability_runtime import ObservabilityMetadataValue


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
    }

    # Context window sizes (max tokens each model can handle)
    DICT_OPENAI_CONTEXT_WINDOWS: ClassVar[dict[str, int]] = {
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

    # dollars per 1 k tokens for each supported model
    DICT_OPENAI_PRICES: ClassVar[dict[str, float]] = {
        # --- GPT-5 Family (Aug 2025 release, latest generation) ---
        "gpt-5": 0.0100,  # $0.010 / 1K tokens  (input+output, flagship tier)
        "gpt-5-mini": 0.0040,  # $0.004 / 1K tokens
        "gpt-5-nano": 0.0015,  # $0.0015 / 1K tokens  (lowest cost, speed optimized)
        # --- GPT-4.1 Family (Apr 2025, still supported) ---
        "gpt-4.1": 0.0060,  # $0.006 / 1K tokens
        "gpt-4.1-mini": 0.0020,  # $0.002 / 1K tokens
        "gpt-4.1-nano": 0.0001,  # $0.0001 / 1K tokens (experimental low-cost tier)
        # --- o4 Reasoning Series (active) ---
        "o4-mini": 0.0010,  # $0.001 / 1K tokens
        "o4-mini-high": 0.0025,  # $0.0025 / 1K tokens (reasoning enhanced variant)
        # --- GPT-4o Omni Series (superseded by GPT-5, not formally deprecated) ---
        "gpt-4o": 0.0050,  # $0.005 / 1K tokens
        "gpt-4o-mini": 0.0005,  # $0.0005 / 1K tokens
    }

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

        # 2) Family properties: map by prefix to avoid duplicating per-model conditionals
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
            # --- GPT-5 Family (current, released Aug 2025) ---
            "gpt-5",  # latest flagship
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
    def price_per_1k_tokens(self) -> float:
        """
        Look up the cost-per-1 k tokens for this model.
        Returns 0.0 if unknown (no guard).
        """
        return AICompletionsCapabilitiesOpenAI.DICT_OPENAI_PRICES.get(
            self.completions_model, 0.0
        )

    @property
    def max_context_tokens(self) -> int:
        """
        Look up the OpenAI context window for the current model_name.
        Falls back to 0 if unknown (i.e. no guard will occur).
        """
        return AICompletionsCapabilitiesOpenAI.DICT_OPENAI_CONTEXT_WINDOWS.get(
            self.completions_model, 0
        )

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
        self, prompt: str, *, other_params: AICompletionsPromptParamsBase | None = None
    ) -> str:
        """
        Sends a prompt to the latest version of the OpenAI API for chat and returns the completion result.

        Args:
            prompt (str): The prompt string to send.
            other_params: Optional provider-specific parameters (not used for OpenAI currently)

        Returns:
            str: The completion result as a string.
        """
        try:
            prompt = self.pii_middleware.process_input(prompt)

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

            def _execute_text_prompt() -> AiApiObservedCompletionsResultModel[str]:
                response = self.client.chat.completions.create(
                    model=self.completions_model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {"role": "user", "content": user_content},
                    ],
                )

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
                bool_continued_response: bool = False

                while response.choices[0].finish_reason == "length":
                    bool_continued_response = True
                    response = self.client.chat.completions.create(
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

                observed_result: AiApiObservedCompletionsResultModel[str] = (
                    AiApiObservedCompletionsResultModel(
                        return_value=raw_output_text,
                        raw_output_text=raw_output_text,
                        finish_reason=str(response.choices[0].finish_reason),
                        provider_prompt_tokens=int_provider_prompt_tokens,
                        provider_completion_tokens=int_provider_completion_tokens,
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
