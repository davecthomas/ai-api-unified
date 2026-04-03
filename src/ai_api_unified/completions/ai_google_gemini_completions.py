# ai_google_gemini_completions.py
"""
Google Gemini completions implementation with structured prompting support.

Environment Variables Required:
    GOOGLE_GEMINI_API_KEY: API key used by default when GOOGLE_AUTH_METHOD is unset.
    GOOGLE_AUTH_METHOD: (optional) api_key | service_account
    GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON file for service-account auth
    COMPLETIONS_MODEL_NAME: (optional) Override default model, defaults to 'gemini-2.5-flash'

Default Endpoints:
    Uses Google Gemini APIs for text generation, defaulting to API-key auth.

Features:
    - Text completion generation
    - Structured prompting with function calling support
    - Schema-based response generation
    - Exponential backoff retry for rate limits and transient errors
    - Comprehensive error handling for authentication and API failures
    - Consistent with other provider patterns in this library

Error Handling:
    - HTTP 401: Clear authentication error with retry suggestion
    - HTTP 429/5xx: Exponential backoff retry with max attempts
    - Network errors: Retry with backoff
    - JSON parse errors: Clear error messages
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Type

from google import genai
from google.genai import errors as gerr
from google.genai.types import GenerateContentResponse
from pydantic import ValidationError

import ai_api_unified.ai_google_base as ai_google_base_module
from ai_api_unified.ai_completions_exceptions import (
    StructuredResponseTokenLimitError,
)
from ai_api_unified.ai_google_base import AIGoogleBase

from ..ai_base import (
    AIBaseCompletions,
    AiApiObservedCompletionsResultModel,
    AIStructuredPrompt,
    AICompletionsPromptParamsBase,
    SupportedDataType,
)
from ..middleware.observability_runtime import ObservabilityMetadataValue
from ..util.env_settings import EnvSettings
from .ai_google_gemini_capabilities import (
    AICompletionsCapabilitiesGoogle,
    AICompletionsPromptParamsGoogle,
)

GOOGLE_GENAI_ERRORS: object = gerr

_LOGGER: logging.Logger = logging.getLogger(__name__)

# Constants
DEFAULT_COMPLETIONS_MODEL: str = "gemini-2.5-flash"
DEFAULT_FALLBACK_MODEL: str = "gemini-2.5-flash"
MAX_RETRIES: int = 5
INITIAL_BACKOFF_DELAY: float = 1.0
BACKOFF_MULTIPLIER: float = 2.0
MAX_JITTER: float = 1.0
RETRY_STATUS_CODES: set[int] = {429, 500, 502, 503, 504}
STRUCTURED_DEFAULT_TEMPERATURE: float = 0.1
STRUCTURED_DEFAULT_TOP_P: float = 0.8

# Model specifications based on https://ai.google.dev/gemini-api/docs/models#model-variations
# Pricing source: Vertex AI Generative AI pricing tables (online, <=200K input tokens tier).
#  - Gemini 2.5 Pro: $1.25 / 1M input tokens  → 0.00125 / 1k
#  - Gemini 2.5 Flash: $0.30 / 1M input tokens → 0.00030 / 1k
#  - Gemini 2.5 Flash-Lite: $0.10 / 1M input tokens → 0.00010 / 1k
#  - Gemini 2.0 Flash: $0.15 / 1M input tokens → 0.00015 / 1k
#  - Gemini 2.0 Flash-Lite: $0.075 / 1M input tokens → 0.000075 / 1k
GEMINI_MODEL_SPECS: dict[str, dict[str, Any]] = {
    # Latest stable models
    "gemini-2.5-pro": {
        "max_context_tokens": 1_048_576,
        "price_per_1k_tokens": 0.00125,
        "status": "Latest Stable",
    },
    "gemini-2.5-flash": {
        "max_context_tokens": 1_048_576,
        "price_per_1k_tokens": 0.00030,
        "status": "Latest Stable",
    },
    "gemini-2.5-flash-lite": {
        "max_context_tokens": 1_048_576,
        "price_per_1k_tokens": 0.00010,
        "status": "Latest Stable",
    },
    "gemini-2.0-flash-lite": {
        "max_context_tokens": 1_048_576,
        "price_per_1k_tokens": 0.000075,
        "status": "Latest Stable",
    },
    "gemini-2.0-flash": {
        "max_context_tokens": 1_048_576,
        "price_per_1k_tokens": 0.00015,
        "status": "Latest Stable",
    },
    "gemini-2.0-flash-001": {
        "max_context_tokens": 1_048_576,
        "price_per_1k_tokens": 0.00015,
        "status": "Latest Stable",
    },
    "gemini-2.0-flash-lite-001": {
        "max_context_tokens": 1_048_576,
        "price_per_1k_tokens": 0.000075,
        "status": "Latest Stable",
    },
    # Legacy stable models (restricted for new projects)
    "gemini-1.5-pro-002": {
        "max_context_tokens": 1_048_576,
        "price_per_1k_tokens": 0.00125,  # converted from $0.0003125 per 1k characters
        "status": "Legacy Stable (restricted for new projects)",
    },
    "gemini-1.5-flash-002": {
        "max_context_tokens": 1_048_576,
        "price_per_1k_tokens": 0.000075,  # converted from $0.00001875 per 1k characters
        "status": "Legacy Stable (restricted for new projects)",
    },
}


class GoogleGeminiCompletions(AIBaseCompletions, AIGoogleBase):
    """
    Google Gemini completions client with structured prompting support.

    Supports text generation and structured responses using function calling
    and schema-based generation.
    """

    def __init__(self, model: str = "", **kwargs: Any) -> None:
        """
        Initialize Google Gemini completions client.

        Args:
            model: Completions model name, defaults to DEFAULT_COMPLETIONS_MODEL
            dimensions: Not used for completions but kept for interface compatibility
        """
        AIGoogleBase.__init__(self, **kwargs)
        self.env: EnvSettings = EnvSettings()
        self.geo_residency: str | None = self.env.get_geo_residency()
        if self.geo_residency:
            _LOGGER.warning(
                "AI_API_GEO_RESIDENCY=%s requested, but Google Gemini does not support data residency constraints. Using default location configuration.",
                self.geo_residency,
            )

        # Set model with fallbacks
        self.completions_model: str = model or self.env.get_setting(
            "COMPLETIONS_MODEL_NAME", DEFAULT_COMPLETIONS_MODEL
        )

        # Fallback to a known working model if the specified one isn't available
        if self.completions_model not in GEMINI_MODEL_SPECS:
            _LOGGER.warning(
                "Model %s not in known specs, falling back to %s",
                self.completions_model,
                DEFAULT_FALLBACK_MODEL,
            )
            self.completions_model = DEFAULT_FALLBACK_MODEL

        AIBaseCompletions.__init__(self, model=self.completions_model, **kwargs)

        # Initialize the client
        self._initialize_client()

        # Set up retry configuration
        self.max_retries: int = MAX_RETRIES
        self.initial_delay: float = INITIAL_BACKOFF_DELAY
        self.backoff_multiplier: float = BACKOFF_MULTIPLIER
        self.max_jitter: float = MAX_JITTER

    def _initialize_client(self) -> None:
        """Initialize the Google Gemini client with proper authentication."""
        ai_google_base_module.genai = genai
        ai_google_base_module.gerr = gerr
        self.client = self.get_client(model=self.completions_model)

    @property
    def model_name(self) -> str:
        """Return the current completions model name."""
        return self.completions_model

    @property
    def list_model_names(self) -> list[str]:
        """Return list of supported completion model names."""
        return list(GEMINI_MODEL_SPECS.keys())

    @property
    def capabilities(self) -> AICompletionsCapabilitiesGoogle:
        """Return model capabilities for the current model."""
        return AICompletionsCapabilitiesGoogle.for_model(self.completions_model)

    @property
    def max_context_tokens(self) -> int:
        """Return the maximum context window size for the current model."""
        return GEMINI_MODEL_SPECS.get(
            self.completions_model, GEMINI_MODEL_SPECS[DEFAULT_FALLBACK_MODEL]
        )["max_context_tokens"]

    @property
    def price_per_1k_tokens(self) -> float:
        """Return the price per 1000 tokens for the current model."""
        return GEMINI_MODEL_SPECS.get(
            self.completions_model, GEMINI_MODEL_SPECS[DEFAULT_FALLBACK_MODEL]
        )["price_per_1k_tokens"]

    def send_prompt(
        self,
        prompt: str,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> str:
        """
        Send a text prompt to Google Gemini and return the response.

        Args:
            prompt: Text prompt to send
            other_params: Optional Google-specific parameters (AICompletionsPromptParamsGoogle)

        Returns:
            Generated text response
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty or None")

        prompt = self.pii_middleware.process_input(prompt)

        # Use provided params or create default ones
        params = self._coerce_params(other_params)

        system_prompt: str | None = params.system_prompt
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = (
            self._build_completions_observability_input_metadata(
                prompt=prompt,
                system_prompt=system_prompt,
                other_params=params,
                response_mode=self.RESPONSE_MODE_TEXT,
            )
        )

        def _generate_text() -> AiApiObservedCompletionsResultModel[str]:
            try:
                contents: list[dict[str, Any]] = self._build_contents(
                    prompt=prompt, params=params
                )
                config: genai.types.GenerateContentConfig = self._build_config(
                    params=params,
                    system_prompt=system_prompt,
                    max_output_tokens=params.max_output_tokens,
                )
                response: GenerateContentResponse = self.client.models.generate_content(
                    model=self.completions_model,
                    contents=contents,
                    config=config,
                )

                if not response.text:
                    _LOGGER.warning("Empty response from Gemini API")
                    return AiApiObservedCompletionsResultModel(
                        return_value="",
                        raw_output_text="",
                        finish_reason=self._extract_gemini_finish_reason(response),
                        provider_prompt_tokens=self._extract_gemini_prompt_tokens(
                            response
                        ),
                        provider_completion_tokens=self._extract_gemini_completion_tokens(
                            response
                        ),
                        provider_total_tokens=self._extract_gemini_total_tokens(
                            response
                        ),
                    )

                raw_output_text: str = response.text.strip()
                return AiApiObservedCompletionsResultModel(
                    return_value=raw_output_text,
                    raw_output_text=raw_output_text,
                    finish_reason=self._extract_gemini_finish_reason(response),
                    provider_prompt_tokens=self._extract_gemini_prompt_tokens(response),
                    provider_completion_tokens=self._extract_gemini_completion_tokens(
                        response
                    ),
                    provider_total_tokens=self._extract_gemini_total_tokens(response),
                )

            except Exception as generate_error:
                _LOGGER.error("Failed to generate text: %s", generate_error)
                raise

        observed_result: AiApiObservedCompletionsResultModel[str] = (
            self._execute_provider_call_with_observability(
                capability=self.CLIENT_TYPE_COMPLETIONS,
                operation="send_prompt",
                dict_input_metadata=dict_input_metadata,
                callable_execute=lambda: self._retry_with_exponential_backoff(
                    _generate_text
                ),
                callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_completions_observability_result_summary(
                    observed_result=result,
                    provider_elapsed_ms=provider_elapsed_ms,
                ),
            )
        )
        sanitized_output: str = self.pii_middleware.process_output(
            observed_result.return_value
        )
        # Normal return with sanitized Gemini text after observability wrapping.
        return sanitized_output

    def strict_schema_prompt(
        self,
        prompt: str,
        response_model: Type[AIStructuredPrompt],
        max_response_tokens: int = AIBaseCompletions.STRUCTURED_DEFAULT_MAX_RESPONSE_TOKENS,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> AIStructuredPrompt:
        """
        Generate a structured response using schema-based prompting.

        Args:
            prompt: Text prompt to send
            response_model: Pydantic model class for structured response
            max_response_tokens: Maximum tokens for response
            other_params: Optional provider-specific parameters, including custom system prompt.

        Returns:
            Structured response as an instance of response_model
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty or None")

        prompt = self.pii_middleware.process_input(prompt)

        if not issubclass(response_model, AIStructuredPrompt):
            raise ValueError("response_model must be a subclass of AIStructuredPrompt")

        self._validate_structured_max_response_tokens(
            provider_name=self.PROVIDER_ENGINE_GOOGLE_GEMINI,
            model_name=self.completions_model,
            max_response_tokens=max_response_tokens,
        )

        params = self._coerce_params(other_params)

        system_prompt: str | None = (
            params.system_prompt
            if params.system_prompt is not None
            else AICompletionsPromptParamsBase.DEFAULT_STRICT_SCHEMA_SYSTEM_PROMPT
        )
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = (
            self._build_completions_observability_input_metadata(
                prompt=prompt,
                system_prompt=system_prompt,
                other_params=params,
                response_mode=self.RESPONSE_MODE_STRUCTURED,
                max_response_tokens=max_response_tokens,
            )
        )

        def _generate_structured() -> (
            AiApiObservedCompletionsResultModel[AIStructuredPrompt]
        ):
            try:
                contents = self._build_contents(prompt=prompt, params=params)
                config = self._build_config(
                    params=params,
                    system_prompt=system_prompt,
                    max_output_tokens=max_response_tokens,
                    response_schema=response_model,
                )
                response: GenerateContentResponse = self.client.models.generate_content(
                    model=self.completions_model,
                    contents=contents,
                    config=config,
                )
                raw_output_text: str = response.text or ""
                if self._did_stop_on_max_tokens(response):
                    self._raise_structured_token_limit_error(
                        provider_name=self.PROVIDER_ENGINE_GOOGLE_GEMINI,
                        model_name=self.completions_model,
                        max_response_tokens=max_response_tokens,
                        finish_reason=self._extract_gemini_finish_reason(response),
                        raw_output_text=raw_output_text,
                    )

                if not response.text:
                    _LOGGER.warning(
                        "Empty response from Gemini API for structured prompt"
                    )
                    raise ValueError("Empty response from Gemini API")

                try:
                    content_str = self.pii_middleware.process_output(raw_output_text)
                    response_data = json.loads(content_str)
                    validated_response: AIStructuredPrompt = response_model(
                        **response_data
                    )
                    return AiApiObservedCompletionsResultModel(
                        return_value=validated_response,
                        raw_output_text=raw_output_text,
                        finish_reason=self._extract_gemini_finish_reason(response),
                        provider_prompt_tokens=self._extract_gemini_prompt_tokens(
                            response
                        ),
                        provider_completion_tokens=self._extract_gemini_completion_tokens(
                            response
                        ),
                        provider_total_tokens=self._extract_gemini_total_tokens(
                            response
                        ),
                    )
                except json.JSONDecodeError as json_error:
                    _LOGGER.warning(
                        "Failed to parse JSON response: model=%s error=%s response_char_count=%s",
                        self.completions_model,
                        json_error,
                        len(raw_output_text),
                    )
                    raise ValueError(
                        f"Invalid JSON response: {json_error}"
                    ) from json_error
                except ValidationError as validation_error:
                    list_validation_errors: list[dict[str, Any]] = (
                        validation_error.errors()
                    )
                    _LOGGER.warning(
                        "Structured response validation failed: model=%s error_count=%s response_char_count=%s",
                        self.completions_model,
                        len(list_validation_errors),
                        len(raw_output_text),
                    )
                    raise ValueError(
                        "Response validation failed with "
                        f"{len(list_validation_errors)} validation errors."
                    ) from None
            except StructuredResponseTokenLimitError:
                raise
            except Exception as structured_error:
                if "MAX_TOKENS" in str(structured_error).upper():
                    _LOGGER.error(
                        "Gemini structured prompt failed due to MAX_TOKENS for model=%s with max_response_tokens=%s.",
                        self.completions_model,
                        max_response_tokens,
                    )
                _LOGGER.error(
                    "Failed to generate structured response: model=%s error_type=%s",
                    self.completions_model,
                    type(structured_error).__name__,
                )
                raise

        observed_result: AiApiObservedCompletionsResultModel[AIStructuredPrompt] = (
            self._execute_provider_call_with_observability(
                capability=self.CLIENT_TYPE_COMPLETIONS,
                operation="strict_schema_prompt",
                dict_input_metadata=dict_input_metadata,
                callable_execute=lambda: self._retry_with_exponential_backoff(
                    _generate_structured
                ),
                callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_completions_observability_result_summary(
                    observed_result=result,
                    provider_elapsed_ms=provider_elapsed_ms,
                ),
            )
        )
        # Normal return with validated Gemini structured output after observability wrapping.
        return observed_result.return_value

    @staticmethod
    def _did_stop_on_max_tokens(response: GenerateContentResponse) -> bool:
        """
        Detect whether Gemini stopped generation because token output limit was reached.

        Args:
            response: Google GenerateContentResponse returned by the SDK.

        Returns:
            True when any candidate reports MAX_TOKENS as finish reason, otherwise False.
        """
        try:
            list_candidates: list[Any] = list(response.candidates or [])
        except TypeError:
            # Normal return because candidate collection could not be iterated safely.
            return False

        # Loop over response candidates to find a MAX_TOKENS finish reason.
        for object_candidate in list_candidates:
            try:
                str_finish_reason: str = str(object_candidate.finish_reason)
            except AttributeError:
                continue
            if "MAX_TOKENS" in str_finish_reason.upper():
                # Early return because at least one candidate hit token limit.
                return True

        # Normal return because no candidate reported token-limit termination.
        return False

    def _coerce_params(
        self, other_params: AICompletionsPromptParamsBase | None
    ) -> AICompletionsPromptParamsGoogle:
        """
        Convert base prompt params to Google-specific params while preserving
        media attachments and other shared fields.
        """

        if other_params is None:
            return AICompletionsPromptParamsGoogle()

        if isinstance(other_params, AICompletionsPromptParamsGoogle):
            return other_params

        return AICompletionsPromptParamsGoogle(**other_params.model_dump())

    def _build_contents(
        self,
        prompt: str,
        params: AICompletionsPromptParamsBase,
    ) -> list[dict[str, Any]]:
        """
        Assemble Google Gemini content parts for text and optional media.
        """

        user_parts: list[dict[str, Any]] = [{"text": prompt}]

        if params.has_included_media:
            for (
                _,
                media_type,
                media_bytes,
                mime_type,
            ) in params.iter_included_media():
                if media_type is not SupportedDataType.IMAGE:
                    continue
                if len(media_bytes) > AICompletionsPromptParamsBase.MAX_IMAGE_BYTES:
                    raise ValueError(
                        "Image attachment exceeds the maximum allowed size of "
                        f"{AICompletionsPromptParamsBase.MAX_IMAGE_BYTES} bytes."
                    )
                encoded_bytes: str = base64.b64encode(media_bytes).decode("ascii")
                user_parts.append(
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": encoded_bytes,
                        }
                    }
                )
        return [{"role": "user", "parts": user_parts}]

    def _build_config(
        self,
        params: AICompletionsPromptParamsGoogle,
        system_prompt: str | None,
        max_output_tokens: int,
        response_schema: Type[AIStructuredPrompt] | None = None,
    ) -> genai.types.GenerateContentConfig:
        """
        Construct GenerateContentConfig, including optional system instruction.
        """

        config_kwargs: dict[str, Any] = {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "max_output_tokens": max_output_tokens,
        }

        if system_prompt is not None:
            config_kwargs["system_instruction"] = system_prompt

        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema
            default_temperature: float = AICompletionsPromptParamsGoogle.model_fields[
                "temperature"
            ].default
            if params.temperature == default_temperature:
                config_kwargs["temperature"] = STRUCTURED_DEFAULT_TEMPERATURE
            default_top_p: float = AICompletionsPromptParamsGoogle.model_fields[
                "top_p"
            ].default
            if params.top_p == default_top_p:
                config_kwargs["top_p"] = STRUCTURED_DEFAULT_TOP_P

        return genai.types.GenerateContentConfig(**config_kwargs)

    @staticmethod
    def _extract_gemini_finish_reason(response: GenerateContentResponse) -> str | None:
        """
        Returns the primary Gemini candidate finish reason when available.

        Args:
            response: Gemini SDK response object returned by `generate_content`.

        Returns:
            Finish reason string when a candidate exists, otherwise None.
        """
        if not response.candidates:
            # Early return because the Gemini response did not include candidates.
            return None
        # Normal return with the primary candidate finish reason.
        return str(response.candidates[0].finish_reason)

    @staticmethod
    def _extract_gemini_prompt_tokens(response: GenerateContentResponse) -> int | None:
        """
        Returns provider-reported prompt token counts from one Gemini response.

        Args:
            response: Gemini SDK response object returned by `generate_content`.

        Returns:
            Provider-reported prompt token count when available, otherwise None.
        """
        try:
            if response.usage_metadata is None:
                # Early return because the Gemini response did not include usage metadata.
                return None
            # Normal return with Gemini prompt token usage.
            return response.usage_metadata.prompt_token_count
        except AttributeError:
            # Early return because the SDK response did not expose usage metadata as expected.
            return None

    @staticmethod
    def _extract_gemini_completion_tokens(
        response: GenerateContentResponse,
    ) -> int | None:
        """
        Returns provider-reported completion token counts from one Gemini response.

        Args:
            response: Gemini SDK response object returned by `generate_content`.

        Returns:
            Provider-reported completion token count when available, otherwise None.
        """
        try:
            if response.usage_metadata is None:
                # Early return because the Gemini response did not include usage metadata.
                return None
            # Normal return with Gemini completion token usage.
            return response.usage_metadata.candidates_token_count
        except AttributeError:
            # Early return because the SDK response did not expose usage metadata as expected.
            return None

    @staticmethod
    def _extract_gemini_total_tokens(response: GenerateContentResponse) -> int | None:
        """
        Returns provider-reported total token counts from one Gemini response.

        Args:
            response: Gemini SDK response object returned by `generate_content`.

        Returns:
            Provider-reported total token count when available, otherwise None.
        """
        try:
            if response.usage_metadata is None:
                # Early return because the Gemini response did not include usage metadata.
                return None
            # Normal return with Gemini total token usage.
            return response.usage_metadata.total_token_count
        except AttributeError:
            # Early return because the SDK response did not expose usage metadata as expected.
            return None
