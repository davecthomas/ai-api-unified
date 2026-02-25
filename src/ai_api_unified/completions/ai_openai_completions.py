# ai_openai_completions.py

import base64
import json
import logging
import time
from datetime import date
from typing import Any, ClassVar, Type

from pydantic import ValidationError, model_validator

from ai_api_unified.ai_openai_base import AIOpenAIBase

from ..ai_base import (
    AIBaseCompletions,
    AIStructuredPrompt,
    AICompletionsCapabilitiesBase,
    AICompletionsPromptParamsBase,
    SupportedDataType,
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
        AIBaseCompletions.__init__(self, model=model, **kwargs)
        self.model = model

        self.completions_model = self.env.get_setting(
            "COMPLETIONS_MODEL_NAME", "gpt-4o-mini"
        )
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
        max_response_tokens: int = 512,
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
        # Include a brief system instruction to nudge the model toward JSON-only output
        system_prompt: str = (
            AICompletionsPromptParamsBase.DEFAULT_STRICT_SCHEMA_SYSTEM_PROMPT
        )
        if other_params is not None and other_params.system_prompt is not None:
            system_prompt = other_params.system_prompt
        user_content = self._build_user_message_content(prompt, other_params)
        messages: Any = [
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

        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.completions_model,
                    messages=messages,
                    functions=functions,  # type: ignore
                    function_call={"name": "strict_schema_response"},
                )

                choice_msg = completion.choices[0].message

                # If the model invoked our dummy function, grab its arguments (the JSON)
                if choice_msg.function_call and choice_msg.function_call.arguments:
                    content_str = choice_msg.function_call.arguments
                else:
                    content_str = choice_msg.content or ""

                parsed_json = json.loads(content_str)
                return response_model.model_validate(parsed_json)

            except ValidationError as e:
                # Handle validation errors
                print(f"Validation error: {e}")
                raise

            except Exception as e:
                # exponential backoff on failure
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                # on final failure, raise so caller can handle it
                raise RuntimeError(
                    f"strict_schema_prompt failed after {max_retries} attempts: {e}"
                )

        raise RuntimeError(
            f"strict_schema_prompt failed to return a valid response after {max_retries} attempts."
        )

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
            system_prompt: str = (
                other_params.system_prompt
                if other_params is not None and other_params.system_prompt is not None
                else AICompletionsPromptParamsBase.DEFAULT_SYSTEM_PROMPT
            )

            user_content = self._build_user_message_content(prompt, other_params)

            messages: Any = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": user_content},
            ]

            response = self.client.chat.completions.create(
                model=self.completions_model,
                messages=messages,
            )

            # Extract the response from the completion
            completion = response.choices[0].message.content or ""

            # If the content seems truncated, send a follow-up request or handle continuation
            while response.choices[0].finish_reason == "length":
                response = self.client.chat.completions.create(
                    model=self.completions_model,
                    messages=[
                        {"role": "system", "content": "Continue."},
                    ],
                )
                completion += response.choices[0].message.content or ""
            return completion

        except Exception as e:
            print(f"An error occurred while sending the prompt: {e}")
            raise
