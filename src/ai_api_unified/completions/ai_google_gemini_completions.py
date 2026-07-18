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
from collections.abc import Iterator
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
    AIFinishReason,
    AIStructuredOutputResult,
    AIStructuredPrompt,
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
from ..ai_provider_exceptions import AiProviderRequestError
from ..middleware.observability_runtime import (
    AiApiCallResultSummaryModel,
    ObservabilityMetadataValue,
)
from ..pricing.pricing_registry import PROVIDER_GOOGLE, enforce_model_lifecycle
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
# Context window and status only. Pricing now lives in the pricing registry
# (single source of truth); lifecycle (deprecated/retired) is enforced there.
GEMINI_MODEL_SPECS: dict[str, dict[str, Any]] = {
    # Latest stable models
    "gemini-2.5-pro": {"max_context_tokens": 1_048_576, "status": "Latest Stable"},
    "gemini-2.5-flash": {"max_context_tokens": 1_048_576, "status": "Latest Stable"},
    "gemini-2.5-flash-lite": {
        "max_context_tokens": 1_048_576,
        "status": "Latest Stable",
    },
    # Deprecated (still functional; the registry warns and names a replacement).
    "gemini-2.0-flash-lite": {"max_context_tokens": 1_048_576, "status": "Deprecated"},
    "gemini-2.0-flash": {"max_context_tokens": 1_048_576, "status": "Deprecated"},
    "gemini-2.0-flash-001": {"max_context_tokens": 1_048_576, "status": "Deprecated"},
    "gemini-2.0-flash-lite-001": {
        "max_context_tokens": 1_048_576,
        "status": "Deprecated",
    },
    # gemini-1.5-pro-002 and gemini-1.5-flash-002 were removed: both are retired
    # by Google (requests return 404). The pricing registry keeps RETIRED
    # lifecycle entries so a request for them fails fast with a clear error.
}


class GoogleGeminiCompletions(AIBaseCompletions, AIGoogleBase):
    """
    Google Gemini completions client with structured prompting support.

    Supports text generation and structured responses using function calling
    and schema-based generation.
    """

    def __init__(
        self, model: str = "", *, retry_policy: str | None = None, **kwargs: Any
    ) -> None:
        """
        Initialize Google Gemini completions client.

        Args:
            model: Completions model name, defaults to DEFAULT_COMPLETIONS_MODEL
            retry_policy: "default" keeps the engine's exponential-backoff
                retries; "none" disables them (single attempt). Falls back to
                the COMPLETIONS_RETRY_POLICY environment setting, then
                "default".
            dimensions: Not used for completions but kept for interface compatibility
        """
        AIGoogleBase.__init__(self, **kwargs)
        self.env: EnvSettings = EnvSettings()
        str_retry_candidate: str = (
            retry_policy
            if retry_policy is not None
            else str(
                self.env.get_setting("COMPLETIONS_RETRY_POLICY", RETRY_POLICY_DEFAULT)
            )
        )
        self.retry_policy: str = normalize_retry_policy(str_retry_candidate)
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

        # Apply lifecycle policy on the requested model before any fallback so a
        # retired model fails fast instead of being silently swapped.
        enforce_model_lifecycle(PROVIDER_GOOGLE, self.completions_model)

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
        Send a text prompt to Google Gemini and return the response.

        Args:
            prompt: Text prompt to send
            system_prompt: Optional persistent instructions; overrides
                other_params.system_prompt when both are supplied.
            max_response_tokens: Optional response token budget; maps to the
                GenerateContentConfig max_output_tokens field and overrides
                params.max_output_tokens when both are supplied.
            request_timeout_seconds: Optional per-call timeout; maps to
                per-request http_options (SDK milliseconds).
            other_params: Optional Google-specific parameters (AICompletionsPromptParamsGoogle)

        Returns:
            Generated text response
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty or None")

        prompt = self.pii_middleware.process_input(prompt)

        # Use provided params or create default ones
        params = self._coerce_params(other_params)
        if system_prompt is not None and system_prompt.strip():
            params.system_prompt = system_prompt

        system_prompt = params.system_prompt
        int_max_output_tokens: int = max_response_tokens or params.max_output_tokens
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = (
            self._build_completions_observability_input_metadata(
                prompt=prompt,
                system_prompt=system_prompt,
                other_params=params,
                response_mode=self.RESPONSE_MODE_TEXT,
                max_response_tokens=max_response_tokens,
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
                    max_output_tokens=int_max_output_tokens,
                    request_timeout_seconds=request_timeout_seconds,
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
                        provider_cached_input_tokens=self._extract_gemini_cached_tokens(
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
                    provider_cached_input_tokens=self._extract_gemini_cached_tokens(
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
                    _generate_text,
                    max_retries=self._effective_max_retries(),
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

    def _send_prompt_streaming_provider(
        self,
        prompt: str,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> Iterator[str]:
        """
        Stream a text prompt response from Google Gemini chunk by chunk.

        Capability and PII gating already ran in the base template method, so
        the streamed chunks are yielded to the caller unmodified. There is no
        retry wrapper: retrying a partially consumed stream would duplicate
        output.

        Args:
            prompt: Validated text prompt to send.
            other_params: Optional Google-specific parameters (AICompletionsPromptParamsGoogle).

        Returns:
            Iterator of response text chunks in provider order.
        """
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
        dict_input_metadata["response_streaming"] = True

        dict_stream_state: dict[str, Any] = {
            "list_text_parts": [],
            "int_chunk_count": 0,
            "last_chunk": None,
            "bool_completed": False,
        }

        def _open_stream() -> Iterator[str]:
            contents: list[dict[str, Any]] = self._build_contents(
                prompt=prompt, params=params
            )
            config: genai.types.GenerateContentConfig = self._build_config(
                params=params,
                system_prompt=system_prompt,
                max_output_tokens=params.max_output_tokens,
            )
            stream = self.client.models.generate_content_stream(
                model=self.completions_model,
                contents=contents,
                config=config,
            )
            # Loop through provider chunks so callers see text as it arrives.
            for chunk in stream:
                dict_stream_state["last_chunk"] = chunk
                str_chunk_text: str | None = chunk.text
                if str_chunk_text:
                    dict_stream_state["int_chunk_count"] += 1
                    dict_stream_state["list_text_parts"].append(str_chunk_text)
                    yield str_chunk_text
            dict_stream_state["bool_completed"] = True

        def _build_summary(provider_elapsed_ms: float) -> AiApiCallResultSummaryModel:
            last_chunk = dict_stream_state["last_chunk"]
            str_full_text: str = "".join(dict_stream_state["list_text_parts"])
            observed_result: AiApiObservedCompletionsResultModel[str] = (
                AiApiObservedCompletionsResultModel(
                    return_value=str_full_text,
                    raw_output_text=str_full_text,
                    finish_reason=(
                        self._extract_gemini_finish_reason(last_chunk)
                        if last_chunk is not None
                        else None
                    ),
                    provider_prompt_tokens=(
                        self._extract_gemini_prompt_tokens(last_chunk)
                        if last_chunk is not None
                        else None
                    ),
                    provider_completion_tokens=(
                        self._extract_gemini_completion_tokens(last_chunk)
                        if last_chunk is not None
                        else None
                    ),
                    provider_cached_input_tokens=(
                        self._extract_gemini_cached_tokens(last_chunk)
                        if last_chunk is not None
                        else None
                    ),
                    provider_total_tokens=(
                        self._extract_gemini_total_tokens(last_chunk)
                        if last_chunk is not None
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

        # Normal return with the observability-wrapped Gemini stream.
        return self._execute_streaming_provider_call_with_observability(
            operation="send_prompt_streaming",
            dict_input_metadata=dict_input_metadata,
            callable_open_stream=_open_stream,
            callable_build_result_summary=_build_summary,
        )

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
                        provider_cached_input_tokens=self._extract_gemini_cached_tokens(
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
                    _generate_structured,
                    max_retries=self._effective_max_retries(),
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
        *,
        response_json_schema: dict[str, Any] | None = None,
        request_timeout_seconds: float | None = None,
    ) -> genai.types.GenerateContentConfig:
        """
        Construct GenerateContentConfig, including optional system instruction.

        response_schema carries a pydantic class (legacy strict_schema_prompt
        path); response_json_schema carries a raw JSON Schema for
        send_structured_output. request_timeout_seconds maps to per-request
        http_options (the SDK timeout unit is milliseconds).
        """

        config_kwargs: dict[str, Any] = {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "max_output_tokens": max_output_tokens,
        }

        if system_prompt is not None:
            config_kwargs["system_instruction"] = system_prompt

        if response_schema is not None or response_json_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            if response_schema is not None:
                config_kwargs["response_schema"] = response_schema
            else:
                config_kwargs["response_json_schema"] = response_json_schema
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

        if request_timeout_seconds is not None:
            config_kwargs["http_options"] = genai.types.HttpOptions(
                timeout=int(request_timeout_seconds * 1000)
            )

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

    @staticmethod
    def _extract_gemini_cached_tokens(
        response: GenerateContentResponse,
    ) -> int | None:
        """
        Returns provider-reported cached prompt token counts from a Gemini response.

        Gemini reports context-cache hits in
        `usage_metadata.cached_content_token_count`; they are a subset of
        `prompt_token_count`.

        Args:
            response: Gemini SDK response object returned by `generate_content`.

        Returns:
            Provider-reported cached prompt token count when available, otherwise None.
        """
        try:
            if response.usage_metadata is None:
                # Early return because the Gemini response did not include usage metadata.
                return None
            # Normal return with Gemini cached prompt token usage.
            return response.usage_metadata.cached_content_token_count
        except AttributeError:
            # Early return because the SDK response did not expose usage metadata as expected.
            return None

    # ── Conversation, structured output, async, retries (2.15.0) ────────────

    PROVIDER_ENGINE_TOKEN = "google-gemini"

    # Finish-reason markers matched as substrings of str(finish_reason), which
    # renders as e.g. "FinishReason.MAX_TOKENS" on the real SDK.
    TUPLE_LENGTH_FINISH_MARKERS = ("MAX_TOKENS",)
    TUPLE_REFUSAL_FINISH_MARKERS = (
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
    )

    def _effective_max_retries(self, retry_override: str | None = None) -> int | None:
        """
        Resolves the retry budget for one call from policy and override.

        Returns:
            0 when retries are disabled (single attempt), otherwise None so
            the backoff helper uses the engine default.
        """
        # getattr with a default keeps init-bypassing test doubles working.
        str_policy: str = getattr(self, "retry_policy", RETRY_POLICY_DEFAULT)
        if retry_override is not None:
            str_policy = normalize_retry_policy(retry_override)
        if str_policy == RETRY_POLICY_NONE:
            # Early return with a single-attempt budget.
            return 0
        # Normal return deferring to the engine default retry budget.
        return None

    def _raise_gemini_request_error(self, exception: Exception) -> None:
        """
        Re-raises one Google SDK transport error as the typed request error.

        Probes the exception and its cause chain (the retry helper wraps
        exhausted errors in RuntimeError "from" the original) for Google API
        errors and their status codes. Non-Google exceptions propagate
        unchanged.
        """

        def _probe_status_code(error: Any) -> int | None:
            for str_attr in ("code", "status"):
                raw_value = getattr(error, str_attr, None)
                raw_value = getattr(raw_value, "value", raw_value)
                if isinstance(raw_value, int):
                    return raw_value
                if isinstance(raw_value, str) and raw_value.isdigit():
                    return int(raw_value)
            response = getattr(error, "response", None)
            raw_status = getattr(response, "status_code", None)
            if isinstance(raw_status, int):
                return raw_status
            return None

        # Loop over the exception and its cause so wrapped errors still map.
        for candidate in (exception, getattr(exception, "__cause__", None)):
            if candidate is None:
                continue
            str_module: str = type(candidate).__module__ or ""
            if str_module.startswith("google"):
                raise AiProviderRequestError(
                    f"Google Gemini request failed: {candidate}",
                    status_code=_probe_status_code(candidate),
                    provider_engine=self.PROVIDER_ENGINE_TOKEN,
                ) from exception
        # Normal return so non-Google exceptions propagate unchanged.
        return None

    def _normalize_gemini_finish_reason(
        self,
        str_finish_reason: str | None,
        *,
        bool_has_tool_calls: bool,
    ) -> AIFinishReason:
        """
        Maps one Gemini finish-reason string onto the normalized enum.
        """
        if bool_has_tool_calls:
            # Early return because requested tool calls define the turn.
            return AIFinishReason.TOOL_USE
        str_upper: str = (str_finish_reason or "").upper()
        if any(marker in str_upper for marker in self.TUPLE_LENGTH_FINISH_MARKERS):
            # Early return because output was truncated at the budget.
            return AIFinishReason.LENGTH
        if any(marker in str_upper for marker in self.TUPLE_REFUSAL_FINISH_MARKERS):
            # Early return because the response was blocked or declined.
            return AIFinishReason.REFUSAL
        # Normal return because the turn completed normally.
        return AIFinishReason.COMPLETE

    def _usage_from_gemini(self, response: Any) -> AITokenUsage:
        """
        Builds the provider-neutral usage model from one Gemini response.
        """
        # Normal return with the provider-neutral token usage model.
        return AITokenUsage(
            input_tokens=self._extract_gemini_prompt_tokens(response),
            output_tokens=self._extract_gemini_completion_tokens(response),
            cached_input_tokens=self._extract_gemini_cached_tokens(response),
            total_tokens=self._extract_gemini_total_tokens(response),
        )

    @staticmethod
    def _serialize_gemini_part(part: Any) -> dict[str, Any]:
        """
        Serializes one candidate content part into a replayable dictionary.
        """
        model_dump = getattr(part, "model_dump", None)
        if callable(model_dump):
            try:
                dict_part: Any = model_dump(exclude_none=True)
                if isinstance(dict_part, dict):
                    # Early return with the SDK-serialized part.
                    return dict_part
            except TypeError:
                # Fall through to attribute-based reconstruction (test doubles).
                pass
        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            dict_function_call: dict[str, Any] = {
                "name": str(getattr(function_call, "name", "") or ""),
                "args": dict(getattr(function_call, "args", None) or {}),
            }
            return {"function_call": dict_function_call}
        str_text: Any = getattr(part, "text", None)
        if str_text:
            return {"text": str_text}
        # Normal return with an empty part placeholder for unknown shapes.
        return {}

    def _build_turn_result_from_gemini(self, response: Any) -> AITurnResult:
        """
        Maps one Gemini response onto the provider-neutral turn.

        Gemini function calls carry no stable call id, so AIToolCall.id is
        the function name; build_tool_result_message expects that name back.
        """
        list_text_parts: list[str] = []
        list_tool_calls: list[AIToolCall] = []
        list_raw_parts: list[dict[str, Any]] = []
        candidates = getattr(response, "candidates", None) or []
        content = getattr(candidates[0], "content", None) if candidates else None
        # Loop over content parts so text, tool calls, and replay parts align.
        for part in getattr(content, "parts", None) or []:
            list_raw_parts.append(self._serialize_gemini_part(part))
            function_call = getattr(part, "function_call", None)
            if function_call is not None:
                str_name: str = str(getattr(function_call, "name", "") or "")
                list_tool_calls.append(
                    AIToolCall(
                        id=str_name,
                        name=str_name,
                        input=dict(getattr(function_call, "args", None) or {}),
                    )
                )
                continue
            str_part_text: Any = getattr(part, "text", None)
            if str_part_text:
                list_text_parts.append(str_part_text)
        str_finish_reason: str | None = self._extract_gemini_finish_reason(response)
        finish_reason: AIFinishReason = self._normalize_gemini_finish_reason(
            str_finish_reason,
            bool_has_tool_calls=bool(list_tool_calls),
        )
        str_text: str = "".join(list_text_parts)
        # Normal return with the provider-neutral turn result.
        return AITurnResult(
            text=str_text if str_text else None,
            tool_calls=list_tool_calls,
            finish_reason=finish_reason,
            raw_content=list_raw_parts,
            usage=self._usage_from_gemini(response),
        )

    def _build_gemini_conversation_config_kwargs(
        self,
        *,
        system_prompt: str,
        tools: list[AITool],
        tool_choice: str | None,
        max_response_tokens: int | None,
        request_timeout_seconds: float | None,
        dict_merge_options: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Builds GenerateContentConfig kwargs for one conversation turn.

        provider_options keys merge into the config kwargs, so callers can set
        any GenerateContentConfig field the engine has not mapped.
        """
        dict_config_kwargs: dict[str, Any] = {"system_instruction": system_prompt}
        if max_response_tokens is not None:
            dict_config_kwargs["max_output_tokens"] = max_response_tokens
        if tools:
            list_declarations: list[Any] = []
            # Loop over tool definitions so each maps to a function declaration.
            for ai_tool in tools:
                list_declarations.append(
                    genai.types.FunctionDeclaration(
                        name=ai_tool.name,
                        description=ai_tool.description,
                        parameters_json_schema=ai_tool.input_schema,
                    )
                )
            dict_config_kwargs["tools"] = [
                genai.types.Tool(function_declarations=list_declarations)
            ]
        if tool_choice is not None:
            dict_config_kwargs["tool_config"] = genai.types.ToolConfig(
                function_calling_config=genai.types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=[tool_choice],
                )
            )
        if request_timeout_seconds is not None:
            dict_config_kwargs["http_options"] = genai.types.HttpOptions(
                timeout=int(request_timeout_seconds * 1000)
            )
        dict_config_kwargs.update(dict_merge_options)
        # Normal return with the conversation config kwargs.
        return dict_config_kwargs

    def _observed_gemini_turn_result(
        self, response: Any
    ) -> AiApiObservedCompletionsResultModel[AITurnResult]:
        """
        Wraps one Gemini response as an observed conversation-turn result.
        """
        turn_result: AITurnResult = self._build_turn_result_from_gemini(response)
        # Normal return with the observed turn result and usage metadata.
        return AiApiObservedCompletionsResultModel(
            return_value=turn_result,
            raw_output_text=turn_result.text or "",
            finish_reason=self._extract_gemini_finish_reason(response),
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
        Sends one conversation turn via generate_content with function tools.
        """
        dict_merge_options, str_retry_override = self._split_provider_options(
            provider_options
        )
        dict_config_kwargs: dict[str, Any] = (
            self._build_gemini_conversation_config_kwargs(
                system_prompt=system_prompt,
                tools=tools,
                tool_choice=tool_choice,
                max_response_tokens=max_response_tokens,
                request_timeout_seconds=request_timeout_seconds,
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

        def _generate_turn() -> AiApiObservedCompletionsResultModel[AITurnResult]:
            response = self.client.models.generate_content(
                model=self.completions_model,
                contents=messages,
                config=genai.types.GenerateContentConfig(**dict_config_kwargs),
            )
            # Normal return with the observed conversation turn.
            return self._observed_gemini_turn_result(response)

        def _execute_with_policy() -> AiApiObservedCompletionsResultModel[AITurnResult]:
            try:
                return self._retry_with_exponential_backoff(
                    _generate_turn,
                    max_retries=self._effective_max_retries(str_retry_override),
                )
            except Exception as exception:
                self._raise_gemini_request_error(exception)
                raise

        observed_result: AiApiObservedCompletionsResultModel[AITurnResult] = (
            self._execute_provider_call_with_observability(
                capability=self.CLIENT_TYPE_COMPLETIONS,
                operation="send_conversation",
                dict_input_metadata=dict_input_metadata,
                callable_execute=_execute_with_policy,
                callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_completions_observability_result_summary(
                    observed_result=result,
                    provider_elapsed_ms=provider_elapsed_ms,
                ),
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
        Async twin of _send_conversation_provider via client.aio.

        Async variants perform a single attempt (the engine backoff helper is
        synchronous); transport failures surface as typed request errors for
        caller-owned backoff.
        """
        dict_merge_options, _ = self._split_provider_options(provider_options)
        dict_config_kwargs: dict[str, Any] = (
            self._build_gemini_conversation_config_kwargs(
                system_prompt=system_prompt,
                tools=tools,
                tool_choice=tool_choice,
                max_response_tokens=max_response_tokens,
                request_timeout_seconds=request_timeout_seconds,
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

        async def _generate_turn() -> AiApiObservedCompletionsResultModel[AITurnResult]:
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.completions_model,
                    contents=messages,
                    config=genai.types.GenerateContentConfig(**dict_config_kwargs),
                )
            except Exception as exception:
                self._raise_gemini_request_error(exception)
                raise
            # Normal return with the observed conversation turn.
            return self._observed_gemini_turn_result(response)

        observed_result: AiApiObservedCompletionsResultModel[AITurnResult] = (
            await self._execute_provider_acall_with_observability(
                capability=self.CLIENT_TYPE_COMPLETIONS,
                operation="asend_conversation",
                dict_input_metadata=dict_input_metadata,
                callable_execute=_generate_turn,
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
        Builds one Gemini function-response user message.

        tool_call_id is the function name (Gemini function calls carry no
        stable id; AIToolCall.id is set to the name). Errors are encoded in
        the response payload under an "error" key.
        """
        dict_payload: dict[str, Any] = {"error": result} if is_error else result
        # Normal return with the Gemini-shaped function response entry.
        return {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": tool_call_id,
                        "response": dict_payload,
                    }
                }
            ],
        }

    def _extend_messages_with_turn_provider(
        self,
        *,
        messages: list[dict[str, Any]],
        turn: AITurnResult,
    ) -> None:
        """
        Appends one Gemini model turn wrapping the raw content parts.
        """
        messages.append({"role": "model", "parts": turn.raw_content})
        # Normal return after appending the model turn.
        return None

    def _build_gemini_structured_request(
        self,
        *,
        response_schema: dict[str, Any],
        system_prompt: str | None,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        max_response_tokens: int,
        request_timeout_seconds: float | None,
        dict_merge_options: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Builds (contents, config kwargs) for one structured-output call.
        """
        str_system_prompt: str = self._resolve_system_prompt(
            system_prompt,
            None,
            AICompletionsPromptParamsBase.DEFAULT_STRICT_SCHEMA_SYSTEM_PROMPT,
        )
        list_contents: list[dict[str, Any]] = list(messages or [])
        if prompt is not None and prompt.strip():
            str_redacted_prompt: str = self.pii_middleware.process_input(prompt)
            list_contents.append(
                {"role": "user", "parts": [{"text": str_redacted_prompt}]}
            )
        dict_config_kwargs: dict[str, Any] = {
            "system_instruction": str_system_prompt,
            "max_output_tokens": max_response_tokens,
            "temperature": STRUCTURED_DEFAULT_TEMPERATURE,
            "top_p": STRUCTURED_DEFAULT_TOP_P,
            "response_mime_type": "application/json",
            "response_json_schema": response_schema,
        }
        if request_timeout_seconds is not None:
            dict_config_kwargs["http_options"] = genai.types.HttpOptions(
                timeout=int(request_timeout_seconds * 1000)
            )
        dict_config_kwargs.update(dict_merge_options)
        # Normal return with contents and config kwargs.
        return list_contents, dict_config_kwargs

    def _build_structured_output_result_from_gemini(
        self, response: Any
    ) -> AIStructuredOutputResult:
        """
        Maps one Gemini structured response onto the provider-neutral result.
        """
        str_finish_reason: str | None = self._extract_gemini_finish_reason(response)
        finish_reason: AIFinishReason = self._normalize_gemini_finish_reason(
            str_finish_reason,
            bool_has_tool_calls=False,
        )
        usage: AITokenUsage = self._usage_from_gemini(response)
        raw_output_text: str = str(getattr(response, "text", "") or "")
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
            raise ValueError("Empty response from Gemini API")
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

    def _observed_gemini_structured_result(
        self, response: Any
    ) -> AiApiObservedCompletionsResultModel[AIStructuredOutputResult]:
        """
        Wraps one Gemini structured response as an observed result.
        """
        structured_result: AIStructuredOutputResult = (
            self._build_structured_output_result_from_gemini(response)
        )
        # Normal return with the observed structured output result.
        return AiApiObservedCompletionsResultModel(
            return_value=structured_result,
            raw_output_text=structured_result.raw_text,
            finish_reason=self._extract_gemini_finish_reason(response),
            provider_prompt_tokens=structured_result.usage.input_tokens,
            provider_completion_tokens=structured_result.usage.output_tokens,
            provider_cached_input_tokens=structured_result.usage.cached_input_tokens,
            provider_total_tokens=structured_result.usage.total_tokens,
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
        Generates structured output via response_json_schema.
        """
        dict_merge_options, str_retry_override = self._split_provider_options(
            provider_options
        )
        list_contents, dict_config_kwargs = self._build_gemini_structured_request(
            response_schema=response_schema,
            system_prompt=system_prompt,
            prompt=prompt,
            messages=messages,
            max_response_tokens=max_response_tokens,
            request_timeout_seconds=request_timeout_seconds,
            dict_merge_options=dict_merge_options,
        )
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = (
            self._build_structured_observability_metadata(
                prompt=prompt,
                messages=messages,
                max_response_tokens=max_response_tokens,
            )
        )

        def _generate_structured() -> (
            AiApiObservedCompletionsResultModel[AIStructuredOutputResult]
        ):
            response = self.client.models.generate_content(
                model=self.completions_model,
                contents=list_contents,
                config=genai.types.GenerateContentConfig(**dict_config_kwargs),
            )
            # Normal return with the observed structured output result.
            return self._observed_gemini_structured_result(response)

        def _execute_with_policy() -> (
            AiApiObservedCompletionsResultModel[AIStructuredOutputResult]
        ):
            try:
                return self._retry_with_exponential_backoff(
                    _generate_structured,
                    max_retries=self._effective_max_retries(str_retry_override),
                )
            except (ValueError, ValidationError):
                raise
            except Exception as exception:
                self._raise_gemini_request_error(exception)
                raise

        observed_result: AiApiObservedCompletionsResultModel[
            AIStructuredOutputResult
        ] = self._execute_provider_call_with_observability(
            capability=self.CLIENT_TYPE_COMPLETIONS,
            operation="send_structured_output",
            dict_input_metadata=dict_input_metadata,
            callable_execute=_execute_with_policy,
            callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_completions_observability_result_summary(
                observed_result=result,
                provider_elapsed_ms=provider_elapsed_ms,
            ),
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
        Async twin of _send_structured_output_provider via client.aio.

        Single attempt; see _asend_conversation_provider.
        """
        dict_merge_options, _ = self._split_provider_options(provider_options)
        list_contents, dict_config_kwargs = self._build_gemini_structured_request(
            response_schema=response_schema,
            system_prompt=system_prompt,
            prompt=prompt,
            messages=messages,
            max_response_tokens=max_response_tokens,
            request_timeout_seconds=request_timeout_seconds,
            dict_merge_options=dict_merge_options,
        )
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = (
            self._build_structured_observability_metadata(
                prompt=prompt,
                messages=messages,
                max_response_tokens=max_response_tokens,
            )
        )

        async def _generate_structured() -> (
            AiApiObservedCompletionsResultModel[AIStructuredOutputResult]
        ):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.completions_model,
                    contents=list_contents,
                    config=genai.types.GenerateContentConfig(**dict_config_kwargs),
                )
            except Exception as exception:
                self._raise_gemini_request_error(exception)
                raise
            # Normal return with the observed structured output result.
            return self._observed_gemini_structured_result(response)

        observed_result: AiApiObservedCompletionsResultModel[
            AIStructuredOutputResult
        ] = await self._execute_provider_acall_with_observability(
            capability=self.CLIENT_TYPE_COMPLETIONS,
            operation="asend_structured_output",
            dict_input_metadata=dict_input_metadata,
            callable_execute=_generate_structured,
            callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_completions_observability_result_summary(
                observed_result=result,
                provider_elapsed_ms=provider_elapsed_ms,
            ),
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
        Async twin of send_prompt via client.aio.

        Single attempt; see _asend_conversation_provider.
        """
        str_redacted_prompt: str = self.pii_middleware.process_input(prompt)
        params = self._coerce_params(other_params)
        if system_prompt is not None and system_prompt.strip():
            params.system_prompt = system_prompt
        str_system_prompt: str | None = params.system_prompt
        int_max_output_tokens: int = max_response_tokens or params.max_output_tokens
        list_contents: list[dict[str, Any]] = self._build_contents(
            prompt=str_redacted_prompt, params=params
        )
        config: genai.types.GenerateContentConfig = self._build_config(
            params=params,
            system_prompt=str_system_prompt,
            max_output_tokens=int_max_output_tokens,
            request_timeout_seconds=request_timeout_seconds,
        )
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = (
            self._build_completions_observability_input_metadata(
                prompt=str_redacted_prompt,
                system_prompt=str_system_prompt,
                other_params=params,
                response_mode=self.RESPONSE_MODE_TEXT,
                max_response_tokens=max_response_tokens,
            )
        )

        async def _generate_text() -> AiApiObservedCompletionsResultModel[str]:
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.completions_model,
                    contents=list_contents,
                    config=config,
                )
            except Exception as exception:
                self._raise_gemini_request_error(exception)
                raise
            raw_output_text: str = str(getattr(response, "text", "") or "")
            # Normal return with the Gemini text output and usage metadata.
            return AiApiObservedCompletionsResultModel(
                return_value=raw_output_text,
                raw_output_text=raw_output_text,
                finish_reason=self._extract_gemini_finish_reason(response),
                provider_prompt_tokens=self._extract_gemini_prompt_tokens(response),
                provider_completion_tokens=self._extract_gemini_completion_tokens(
                    response
                ),
                provider_cached_input_tokens=self._extract_gemini_cached_tokens(
                    response
                ),
                provider_total_tokens=self._extract_gemini_total_tokens(response),
            )

        observed_result: AiApiObservedCompletionsResultModel[str] = (
            await self._execute_provider_acall_with_observability(
                capability=self.CLIENT_TYPE_COMPLETIONS,
                operation="asend_prompt",
                dict_input_metadata=dict_input_metadata,
                callable_execute=_generate_text,
                callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_completions_observability_result_summary(
                    observed_result=result,
                    provider_elapsed_ms=provider_elapsed_ms,
                ),
            )
        )
        # Normal return with sanitized Gemini text after observability wrapping.
        return self.pii_middleware.process_output(observed_result.return_value)
