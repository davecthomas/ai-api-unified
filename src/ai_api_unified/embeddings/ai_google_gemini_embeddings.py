# ai_google_gemini_embeddings.py
"""
Google Gemini embeddings implementation.

API references:
    - Gemini API embeddings docs:
      https://ai.google.dev/gemini-api/docs/embeddings
    - Vertex AI text embeddings model reference:
      https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api

Environment Variables:
    Default runtime mode (API key):
        GOOGLE_GEMINI_API_KEY: API key used when GOOGLE_AUTH_METHOD is unset or api_key.

    Optional service-account mode:
        GOOGLE_AUTH_METHOD: api_key | service_account
        GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON credentials.
        GOOGLE_PROJECT_ID: Google Cloud project ID used for Vertex AI requests.
        GOOGLE_LOCATION: (optional) Vertex AI region, defaults to us-central1.

    Model/runtime tuning:
        EMBEDDING_MODEL_NAME: (optional) Override default model 'gemini-embedding-001'.
        EMBEDDING_DIMENSIONS: (optional) Override default output dimensions 3072.

Model naming and capability notes:
    - This implementation is pinned to Gemini embedding model family naming.
    - Default model is 'gemini-embedding-001' (text-only).
    - 'gemini-embedding-2' is natively multimodal: text, images, video, audio,
      and PDF inputs share one embedding space. Select it via
      EMBEDDING_MODEL_NAME=gemini-embedding-2 to enable
      generate_embeddings_multimodal; capabilities flip automatically.
    - The code supports passing output dimensionality via EmbedContentConfig.
    - Documented supported output dimensions are 768-3072 for
      gemini-embedding-001 and 128-3072 for gemini-embedding-2.
    - Google-recommended output dimensions are 768, 1536, and 3072.
      This class default is 3072 when no override is provided.

Features:
    - Batch embedding support
    - Multimodal (interleaved text + media) embedding support on
      gemini-embedding-2, capability-gated per model
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

import logging
from typing import Any

from google import genai
from google.api_core.exceptions import ClientError
from google.genai import errors as gerr
from google.genai.types import Content, EmbedContentConfig, Part
import ai_api_unified.ai_google_base as ai_google_base_module
from ai_api_unified.ai_google_base import AIGoogleBase

from ..ai_base import (
    AIBaseEmbeddings,
    AIEmbeddingsCapabilitiesBase,
    AIEmbeddingsMultimodalParams,
    AiApiObservedEmbeddingsResultModel,
    SupportedDataType,
)
from ..pricing.pricing_registry import (
    PROVIDER_GOOGLE,
    enforce_model_lifecycle,
    get_model_pricing,
)
from ..util.env_settings import EnvSettings

GOOGLE_GENAI_ERRORS: object = gerr
GOOGLE_GENAI_MODULE: object = genai

_LOGGER: logging.Logger = logging.getLogger(__name__)


class AIEmbeddingsCapabilitiesGoogle(AIEmbeddingsCapabilitiesBase):
    """
    Google Gemini-specific embeddings capabilities.

    Based on https://ai.google.dev/gemini-api/docs/embeddings
    """

    @classmethod
    def for_model(cls, model_name: str) -> "AIEmbeddingsCapabilitiesGoogle":
        """Create capabilities instance for a specific Gemini embedding model."""
        pricing = get_model_pricing(PROVIDER_GOOGLE, model_name)
        if "gemini-embedding-2" in model_name:
            # Natively multimodal: text, images, video, audio, and PDF inputs
            # share one embedding space (stable since April 2026).
            return cls(
                supported_data_types=[
                    SupportedDataType.TEXT,
                    SupportedDataType.IMAGE,
                    SupportedDataType.VIDEO,
                    SupportedDataType.AUDIO,
                    SupportedDataType.PDF,
                ],
                default_dimensions=3072,
                min_dimensions=128,
                max_dimensions=3072,
                recommended_dimensions=[768, 1536, 3072],
                max_input_tokens=8192,
                max_batch_size=100,
                max_images_per_request=6,
                max_video_seconds=120,
                max_audio_seconds=180,
                pricing=pricing,
            )
        # gemini-embedding-001 and unknown models: text-only defaults.
        return cls(
            default_dimensions=3072,
            min_dimensions=768,
            max_dimensions=3072,
            recommended_dimensions=[768, 1536, 3072],
            max_input_tokens=2048,
            max_batch_size=100,
            pricing=pricing,
        )


class GoogleGeminiEmbeddings(AIBaseEmbeddings, AIGoogleBase):
    """
    Google Gemini embeddings client for Gemini models.

    Supports both single and batch embedding generation with automatic retries
    for transient failures.

    Runtime mode clarification:
        This class uses the shared Google client helper and therefore defaults to
        API-key auth unless GOOGLE_AUTH_METHOD explicitly switches to service-account mode.
    """

    # Constants
    # Default Gemini embedding model identifier used by this provider.
    DEFAULT_EMBEDDING_MODEL: str = "gemini-embedding-001"
    # Default vector width returned when no output_dimensionality override is
    # provided. Per-model dimension limits live in AIEmbeddingsCapabilitiesGoogle.
    DEFAULT_EMBEDDING_DIMENSIONS: int = 3072
    # Provider-side operational limits and retry/backoff tuning.
    MAX_BATCH_SIZE: int = 100
    MAX_RETRIES: int = 5
    INITIAL_BACKOFF_DELAY: float = 1.0
    BACKOFF_MULTIPLIER: float = 2.0
    MAX_JITTER: float = 1.0
    RETRY_STATUS_CODES: set[int] = {429, 500, 502, 503, 504}

    def __init__(self, model: str = "", dimensions: int = 0) -> None:
        """
        Initialize Google Gemini embeddings client.

        Args:
            model: Embedding model name, defaults to DEFAULT_EMBEDDING_MODEL
            dimensions: Output embedding dimensions, defaults to DEFAULT_EMBEDDING_DIMENSIONS
                (3072) when omitted. Documented supported range is 768-3072, with
                768, 1536, and 3072 as recommended presets.
        """
        self.env: EnvSettings = EnvSettings()

        # Set model and dimensions with fallbacks
        self.embedding_model: str = model or self.env.get_setting(
            "EMBEDDING_MODEL_NAME", self.DEFAULT_EMBEDDING_MODEL
        )
        enforce_model_lifecycle(PROVIDER_GOOGLE, self.embedding_model)
        self.dimensions: int = dimensions or int(
            self.env.get_setting(
                "EMBEDDING_DIMENSIONS", str(self.DEFAULT_EMBEDDING_DIMENSIONS)
            )
        )
        embeddings_capabilities: AIEmbeddingsCapabilitiesBase = self.capabilities
        if self.dimensions not in embeddings_capabilities.recommended_dimensions:
            _LOGGER.info(
                "Requested non-recommended Gemini embedding dimensions: %s. Recommended values are %s.",
                self.dimensions,
                embeddings_capabilities.recommended_dimensions,
            )

        # Initialize the client
        self._initialize_client()

        # Set up retry configuration
        self.max_retries: int = self.MAX_RETRIES
        self.initial_delay: float = self.INITIAL_BACKOFF_DELAY
        self.backoff_multiplier: float = self.BACKOFF_MULTIPLIER
        self.max_jitter: float = self.MAX_JITTER

    def _initialize_client(self) -> None:
        """
        Initialize the Google Gemini client using shared Google auth defaults.
        """
        ai_google_base_module.genai = genai
        ai_google_base_module.gerr = gerr
        self.client = self.get_client(model=self.embedding_model)

    @property
    def model_name(self) -> str:
        """Return the current embedding model name."""
        return self.embedding_model

    @property
    def list_model_names(self) -> list[str]:
        """
        Return model names officially supported by this provider implementation.

        This list documents the models this class is designed to target, not the
        full dynamic catalog returned by Google model listing endpoints.
        """
        return [
            "gemini-embedding-001",
            "gemini-embedding-2",
        ]

    @property
    def capabilities(self) -> AIEmbeddingsCapabilitiesGoogle:
        """Return capabilities for the configured Gemini embedding model."""
        return AIEmbeddingsCapabilitiesGoogle.for_model(self.embedding_model)

    def generate_embeddings(self, text: str) -> dict[str, Any]:
        """
        Generate embeddings for a single text string.

        Args:
            text: Text to generate embeddings for

        Returns:
            Dictionary containing embedding vector and metadata
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty or None")
        dict_input_metadata: dict[str, str | int | float | bool | None] = (
            self._build_embeddings_observability_input_metadata(
                list_texts=[text],
                bool_is_batch=False,
                requested_dimensions=self.dimensions,
            )
        )

        observed_result: AiApiObservedEmbeddingsResultModel[dict[str, Any]] = (
            self._execute_provider_call_with_observability(
                capability=self.CLIENT_TYPE_EMBEDDING,
                operation="generate_embeddings",
                dict_input_metadata=dict_input_metadata,
                callable_execute=lambda: self._retry_with_exponential_backoff(
                    lambda: self._embed_one_content_observed(
                        contents=[text],
                        dict_payload_extra={"text": text},
                        str_operation_context="embedding",
                    )
                ),
                callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_embeddings_observability_result_summary(
                    observed_result=result,
                    provider_elapsed_ms=provider_elapsed_ms,
                ),
            )
        )
        # Normal return with the caller-facing single embedding payload.
        return observed_result.return_value

    def generate_embeddings_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """
        Generate embeddings for multiple text strings in batch.

        Args:
            texts: list of text strings to generate embeddings for

        Returns:
            list of dictionaries, each containing embedding vector and metadata
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        # Filter out empty texts
        valid_texts: list[str] = [text for text in texts if text and text.strip()]
        if not valid_texts:
            raise ValueError("All texts in the list are empty")

        dict_input_metadata: dict[str, str | int | float | bool | None] = (
            self._build_embeddings_observability_input_metadata(
                list_texts=valid_texts,
                bool_is_batch=True,
                requested_dimensions=self.dimensions,
            )
        )

        def _execute_batch_embeddings() -> (
            AiApiObservedEmbeddingsResultModel[list[dict[str, Any]]]
        ):
            list_results: list[dict[str, Any]] = []
            int_provider_prompt_tokens: int | None = 0
            int_provider_total_tokens: int | None = 0
            int_returned_dimensions: int | None = None
            int_provider_request_count: int = 0

            # Loop through deterministic batch slices so one public batch call can span provider request limits.
            for index_start in range(0, len(valid_texts), self.MAX_BATCH_SIZE):
                batch_texts: list[str] = valid_texts[
                    index_start : index_start + self.MAX_BATCH_SIZE
                ]

                def _embed_batch() -> (
                    tuple[list[dict[str, Any]], int | None, int | None]
                ):
                    try:
                        embed_kwargs: dict[str, Any] = {
                            "model": self.embedding_model,
                            "contents": batch_texts,
                        }
                        if self.dimensions != self.DEFAULT_EMBEDDING_DIMENSIONS:
                            embed_kwargs["config"] = EmbedContentConfig(
                                output_dimensionality=self.dimensions
                            )
                        response = self.client.models.embed_content(
                            **embed_kwargs,
                        )
                        if response.embeddings is None or len(
                            response.embeddings
                        ) != len(batch_texts):
                            raise RuntimeError(
                                "Gemini embeddings response count mismatch. "
                                f"Requested {len(batch_texts)} embeddings but received {len(response.embeddings or [])}."
                            )
                        list_batch_results: list[dict[str, Any]] = []
                        # Loop through provider vectors so the caller-facing batch payload preserves input order.
                        for index, embed_obj in enumerate(response.embeddings):
                            batch_text: str = batch_texts[index]
                            if embed_obj.values is None:
                                raise RuntimeError(
                                    "Gemini embeddings response entry did not contain embedding values."
                                )
                            embedding_values: list[float] = embed_obj.values
                            list_batch_results.append(
                                {
                                    "embedding": embedding_values,
                                    "model": self.embedding_model,
                                    "dimensions": len(embedding_values),
                                    "text": batch_text,
                                }
                            )
                        prompt_tokens: int | None = self._extract_gemini_prompt_tokens(
                            response
                        )
                        total_tokens: int | None = self._extract_gemini_total_tokens(
                            response
                        )
                        # Normal return with one provider-slice batch payload and its usage metadata.
                        return list_batch_results, prompt_tokens, total_tokens

                    except Exception as batch_error:
                        _LOGGER.error(
                            "Failed to generate batch embeddings: %s", batch_error
                        )
                        raise

                (
                    list_batch_results,
                    slice_prompt_tokens,
                    slice_total_tokens,
                ) = self._retry_with_exponential_backoff(_embed_batch)
                int_provider_request_count += 1
                list_results.extend(list_batch_results)
                if list_batch_results and int_returned_dimensions is None:
                    int_returned_dimensions = list_batch_results[0]["dimensions"]
                if int_provider_prompt_tokens is not None:
                    if slice_prompt_tokens is None:
                        int_provider_prompt_tokens = None
                    else:
                        int_provider_prompt_tokens += slice_prompt_tokens
                if int_provider_total_tokens is not None:
                    if slice_total_tokens is None:
                        int_provider_total_tokens = None
                    else:
                        int_provider_total_tokens += slice_total_tokens

            observed_result: AiApiObservedEmbeddingsResultModel[
                list[dict[str, Any]]
            ] = AiApiObservedEmbeddingsResultModel(
                return_value=list_results,
                embedding_count=len(list_results),
                returned_dimensions=int_returned_dimensions,
                provider_input_tokens=int_provider_prompt_tokens,
                provider_total_tokens=int_provider_total_tokens,
                dict_metadata={
                    "provider_request_count": int_provider_request_count,
                },
            )
            # Normal return with the caller-facing batch payload across all provider slices.
            return observed_result

        observed_result: AiApiObservedEmbeddingsResultModel[list[dict[str, Any]]] = (
            self._execute_provider_call_with_observability(
                capability=self.CLIENT_TYPE_EMBEDDING,
                operation="generate_embeddings_batch",
                dict_input_metadata=dict_input_metadata,
                callable_execute=_execute_batch_embeddings,
                callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_embeddings_observability_result_summary(
                    observed_result=result,
                    provider_elapsed_ms=provider_elapsed_ms,
                ),
            )
        )
        # Normal return with the caller-facing batch embeddings payload.
        return observed_result.return_value

    def _generate_embeddings_multimodal_provider(
        self,
        params: AIEmbeddingsMultimodalParams,
    ) -> dict[str, Any]:
        """
        Generate one embedding for interleaved multimodal input.

        Requires a model whose capabilities include the attached media types
        (set EMBEDDING_MODEL_NAME=gemini-embedding-2 for image/video/audio/PDF
        support). All supplied text and media are embedded jointly into a
        single vector. Capability gating already ran in the base template
        method.

        Args:
            params: Capability-validated multimodal embeddings input.

        Returns:
            Dictionary containing the embedding vector and metadata.
        """
        if params.has_included_media and getattr(self.client, "vertexai", False):
            # google-genai's embed transformer keeps only text parts when the
            # client runs in Vertex mode, which would silently drop the media.
            raise NotImplementedError(
                "Google Gemini multimodal embeddings with media attachments require "
                "API-key auth: the google-genai SDK sends only text parts to the "
                "Vertex embedContent endpoint. Set GOOGLE_AUTH_METHOD=api_key."
            )

        dict_input_metadata: dict[str, str | int | float | bool | None] = (
            self._build_multimodal_embeddings_observability_input_metadata(
                params=params,
                requested_dimensions=self.dimensions,
            )
        )

        def _build_multimodal_contents() -> Content:
            list_parts: list[Part] = []
            if params.text and params.text.strip():
                list_parts.append(Part.from_text(text=params.text))
            # Loop through attachments so every media item becomes one inline part.
            for _, _, media_bytes, mime_type in params.iter_included_media():
                list_parts.append(
                    Part.from_bytes(data=media_bytes, mime_type=mime_type)
                )
            # One Content wrapping all parts makes the interleaved-input intent
            # explicit: the provider returns one embedding for the whole request.
            return Content(parts=list_parts)

        observed_result: AiApiObservedEmbeddingsResultModel[dict[str, Any]] = (
            self._execute_provider_call_with_observability(
                capability=self.CLIENT_TYPE_EMBEDDING,
                operation="generate_embeddings_multimodal",
                dict_input_metadata=dict_input_metadata,
                callable_execute=lambda: self._retry_with_exponential_backoff(
                    lambda: self._embed_one_content_observed(
                        contents=_build_multimodal_contents(),
                        dict_payload_extra={
                            "text": params.text,
                            "included_media_count": len(params.included_types or []),
                        },
                        str_operation_context="multimodal embedding",
                    )
                ),
                callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_embeddings_observability_result_summary(
                    observed_result=result,
                    provider_elapsed_ms=provider_elapsed_ms,
                ),
            )
        )
        # Normal return with the caller-facing multimodal embedding payload.
        return observed_result.return_value

    def _embed_one_content_observed(
        self,
        *,
        contents: Any,
        dict_payload_extra: dict[str, Any],
        str_operation_context: str,
    ) -> AiApiObservedEmbeddingsResultModel[dict[str, Any]]:
        """
        Executes one embed_content call that returns a single embedding.

        Shared request/guard/error handling for the single-text and multimodal
        paths; the batch path keeps its own loop over provider slices.

        Args:
            contents: SDK contents payload for one embedding (text list or Content).
            dict_payload_extra: Caller-facing payload fields beyond the vector.
            str_operation_context: Lowercase operation label used in error text.

        Returns:
            Observed result container with the caller-facing embedding payload.
        """
        try:
            embed_kwargs: dict[str, Any] = {
                "model": self.embedding_model,
                "contents": contents,
            }
            # Only include output_dimensionality when the caller requested a
            # non-default vector width for the configured model, preserving
            # server-side default behavior otherwise.
            if self.dimensions != self.capabilities.default_dimensions:
                embed_kwargs["config"] = EmbedContentConfig(
                    output_dimensionality=self.dimensions
                )
            result = self.client.models.embed_content(
                **embed_kwargs,
            )
            if not result.embeddings:
                raise RuntimeError(
                    "Gemini embeddings response did not contain any embeddings."
                )
            if result.embeddings[0].values is None:
                raise RuntimeError(
                    "Gemini embeddings response entry did not contain embedding values."
                )
            embedding_values: list[float] = result.embeddings[0].values

            observed_result: AiApiObservedEmbeddingsResultModel[dict[str, Any]] = (
                AiApiObservedEmbeddingsResultModel(
                    return_value={
                        "embedding": embedding_values,
                        "model": self.embedding_model,
                        "dimensions": len(embedding_values),
                        **dict_payload_extra,
                    },
                    embedding_count=1,
                    returned_dimensions=len(embedding_values),
                    provider_input_tokens=self._extract_gemini_prompt_tokens(result),
                    provider_total_tokens=self._extract_gemini_total_tokens(result),
                )
            )
            # Normal return with the single embedding payload and metadata-only summary inputs.
            return observed_result

        except ClientError as client_error:
            _LOGGER.error(
                "Google API client error while generating %s: %s",
                str_operation_context,
                client_error,
            )
            raise RuntimeError(
                f"{str_operation_context.capitalize()} call failed with client error: "
                f"{client_error}"
            ) from client_error

        except Exception as embed_error:
            _LOGGER.error(
                "Failed to generate %s: %s", str_operation_context, embed_error
            )
            raise

    @staticmethod
    def _extract_gemini_prompt_tokens(response: Any) -> int | None:
        """
        Returns provider-reported prompt token counts from one Gemini embeddings response.

        Args:
            response: Gemini SDK response object returned by `embed_content`.

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
    def _extract_gemini_total_tokens(response: Any) -> int | None:
        """
        Returns provider-reported total token counts from one Gemini embeddings response.

        Args:
            response: Gemini SDK response object returned by `embed_content`.

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
