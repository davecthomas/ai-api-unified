# ai_google_gemini_completions.py
"""
Google Gemini completions implementation with structured prompting support.

Environment Variables Required:
    GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON file for authentication
    COMPLETIONS_MODEL_NAME: (optional) Override default model, defaults to 'gemini-2.0-flash-lite'

Default Endpoints:
    Uses Google's Vertex AI Gemini API for text generation

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

GOOGLE_DEPENDENCIES_AVAILABLE: bool = False
try:
    from google import genai
    from google.genai import errors as gerr
    from google.genai.types import GenerateContentResponse
    from ai_api_unified.ai_google_base import AIGoogleBase

    GOOGLE_DEPENDENCIES_AVAILABLE = True


except ImportError as import_error:
    GOOGLE_DEPENDENCIES_AVAILABLE = False

if GOOGLE_DEPENDENCIES_AVAILABLE:
    from pydantic import ValidationError

    from ..ai_base import (
        AIBaseCompletions,
        AIStructuredPrompt,
        AICompletionsPromptParamsBase,
        SupportedDataType,
    )
    from ..util.env_settings import EnvSettings
    from .ai_google_gemini_capabilities import (
        AICompletionsCapabilitiesGoogle,
        AICompletionsPromptParamsGoogle,
    )

    _LOGGER: logging.Logger = logging.getLogger(__name__)

    # Constants
    DEFAULT_COMPLETIONS_MODEL: str = "gemini-2.0-flash-lite"
    DEFAULT_FALLBACK_MODEL: str = "gemini-2.0-flash"
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

        def __init__(self, model: str = "") -> None:
            """
            Initialize Google Gemini completions client.

            Args:
                model: Completions model name, defaults to DEFAULT_COMPLETIONS_MODEL
                dimensions: Not used for completions but kept for interface compatibility
            """
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

            # Initialize the client
            self._initialize_client()

            # Set up retry configuration
            self.max_retries: int = MAX_RETRIES
            self.initial_delay: float = INITIAL_BACKOFF_DELAY
            self.backoff_multiplier: float = BACKOFF_MULTIPLIER
            self.max_jitter: float = MAX_JITTER

        def _initialize_client(self) -> None:
            """Initialize the Google Gemini client with proper authentication."""

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

            # Use provided params or create default ones
            params = self._coerce_params(other_params)

            system_prompt: str | None = params.system_prompt

            def _generate_text() -> str:
                try:
                    contents: list[dict[str, Any]] = self._build_contents(
                        prompt=prompt, params=params
                    )
                    config: genai.types.GenerateContentConfig = self._build_config(
                        params=params,
                        system_prompt=system_prompt,
                        max_output_tokens=params.max_output_tokens,
                    )
                    response: GenerateContentResponse = (
                        self.client.models.generate_content(
                            model=self.completions_model,
                            contents=contents,
                            config=config,
                        )
                    )

                    if not response.text:
                        _LOGGER.warning("Empty response from Gemini API")
                        return ""

                    return response.text.strip()

                except Exception as generate_error:
                    _LOGGER.error("Failed to generate text: %s", generate_error)
                    raise

            return self._retry_with_exponential_backoff(_generate_text)

        def strict_schema_prompt(
            self,
            prompt: str,
            response_model: Type[AIStructuredPrompt],
            max_response_tokens: int = 512,
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

            if not issubclass(response_model, AIStructuredPrompt):
                raise ValueError(
                    "response_model must be a subclass of AIStructuredPrompt"
                )

            params = self._coerce_params(other_params)

            system_prompt: str | None = (
                params.system_prompt
                if params.system_prompt is not None
                else AICompletionsPromptParamsBase.DEFAULT_STRICT_SCHEMA_SYSTEM_PROMPT
            )

            def _generate_structured() -> AIStructuredPrompt:
                try:
                    contents = self._build_contents(prompt=prompt, params=params)
                    config = self._build_config(
                        params=params,
                        system_prompt=system_prompt,
                        max_output_tokens=max_response_tokens,
                        response_schema=response_model,
                    )
                    response: GenerateContentResponse = (
                        self.client.models.generate_content(
                            model=self.completions_model,
                            contents=contents,
                            config=config,
                        )
                    )

                    if not response.text:
                        _LOGGER.warning(
                            "Empty response from Gemini API for structured prompt"
                        )
                        raise ValueError("Empty response from Gemini API")

                    try:
                        response_data = json.loads(response.text)
                        return response_model(**response_data)
                    except json.JSONDecodeError as json_error:
                        _LOGGER.warning(
                            "Failed to parse JSON response: %s. Response: %s",
                            json_error,
                            response.text,
                        )
                        raise ValueError(
                            f"Invalid JSON response: {json_error}"
                        ) from json_error
                    except ValidationError as validation_error:
                        _LOGGER.warning(
                            "Structured response validation failed: %s. Response: %s",
                            validation_error,
                            response.text,
                        )
                        raise ValueError(
                            f"Response validation failed: {validation_error}"
                        ) from validation_error
                except Exception as structured_error:
                    _LOGGER.error(
                        "Failed to generate structured response: %s", structured_error
                    )
                    raise

            return self._retry_with_exponential_backoff(_generate_structured)

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
                default_temperature: float = (
                    AICompletionsPromptParamsGoogle.model_fields["temperature"].default
                )
                if params.temperature == default_temperature:
                    config_kwargs["temperature"] = STRUCTURED_DEFAULT_TEMPERATURE
                default_top_p: float = AICompletionsPromptParamsGoogle.model_fields[
                    "top_p"
                ].default
                if params.top_p == default_top_p:
                    config_kwargs["top_p"] = STRUCTURED_DEFAULT_TOP_P

            return genai.types.GenerateContentConfig(**config_kwargs)
