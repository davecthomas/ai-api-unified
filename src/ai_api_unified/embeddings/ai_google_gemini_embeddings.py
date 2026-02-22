# ai_google_gemini_embeddings.py
"""
Google Gemini embeddings implementation.
Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api

Environment Variables Required:
    GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON file for authentication
    EMBEDDING_MODEL_NAME: (optional) Override default model, defaults to 'gemini-embedding-001'
    EMBEDDING_DIMENSIONS: (optional) Override default dimensions, defaults to 3072

Default Endpoints:
    Uses Google's Vertex AI Gemini embedding API

Features:
    - Batch embedding support
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

GOOGLE_DEPENDENCIES_AVAILABLE: bool = False
try:
    from google.api_core.exceptions import ClientError
    from google.genai import pagers
    from google.genai.types import EmbedContentConfig, Model
    from ai_api_unified.ai_google_base import AIGoogleBase

    GOOGLE_DEPENDENCIES_AVAILABLE = True

except ImportError as import_error:
    GOOGLE_DEPENDENCIES_AVAILABLE = False


if GOOGLE_DEPENDENCIES_AVAILABLE:
    from ..ai_base import AIBaseEmbeddings
    from ..util.env_settings import EnvSettings

    _LOGGER: logging.Logger = logging.getLogger(__name__)

    class GoogleGeminiEmbeddings(AIBaseEmbeddings, AIGoogleBase):
        """
        Google Gemini embeddings client using Application Default Credentials.

        Supports both single and batch embedding generation with automatic retries
        for transient failures.
        """

        # Constants
        DEFAULT_EMBEDDING_MODEL: str = "gemini-embedding-001"
        DEFAULT_EMBEDDING_DIMENSIONS: int = (
            3072  # Default dimensions for gemini-embedding-001
        )
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
                dimensions: Embedding dimensions, defaults to DEFAULT_EMBEDDING_DIMENSIONS
            """
            self.env: EnvSettings = EnvSettings()

            # Set model and dimensions with fallbacks
            self.embedding_model: str = model or self.env.get_setting(
                "EMBEDDING_MODEL_NAME", self.DEFAULT_EMBEDDING_MODEL
            )
            self.dimensions: int = dimensions or int(
                self.env.get_setting(
                    "EMBEDDING_DIMENSIONS", str(self.DEFAULT_EMBEDDING_DIMENSIONS)
                )
            )

            self.models_list: pagers.Pager[Model] = []
            # Initialize the client
            self._initialize_client()

            # Set up retry configuration
            self.max_retries: int = self.MAX_RETRIES
            self.initial_delay: float = self.INITIAL_BACKOFF_DELAY
            self.backoff_multiplier: float = self.BACKOFF_MULTIPLIER
            self.max_jitter: float = self.MAX_JITTER

        def _initialize_client(self) -> None:
            """Initialize the Google Gemini client with proper authentication."""
            self.client = self.get_client(model=self.embedding_model)

        @property
        def model_name(self) -> str:
            """Return the current embedding model name."""
            return self.embedding_model

        @property
        def list_model_names(self) -> list[str]:
            """Return list of supported embedding model names."""
            return [
                "gemini-embedding-001",
            ]

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

            def _embed_single() -> dict[str, Any]:
                try:
                    embed_kwargs: dict[str, Any] = {
                        "model": self.embedding_model,
                        "contents": [text],
                    }
                    if self.dimensions != self.DEFAULT_EMBEDDING_DIMENSIONS:
                        embed_kwargs["config"] = EmbedContentConfig(
                            output_dimensionality=self.dimensions
                        )
                    result = self.client.models.embed_content(
                        **embed_kwargs,
                    )
                    embedding_values: list[float] = result.embeddings[0].values

                    return {
                        "embedding": embedding_values,
                        "model": self.embedding_model,
                        "dimensions": len(embedding_values),
                        "text": text,
                    }

                except ClientError as client_error:  # NEW
                    _LOGGER.error(
                        "Google API client error while generating embedding: %s",
                        client_error,
                    )
                    raise RuntimeError(
                        f"Embedding call failed with client error: {client_error}"
                    ) from client_error

                except Exception as embed_error:
                    _LOGGER.error("Failed to generate embedding: %s", embed_error)
                    raise

            return self._retry_with_exponential_backoff(_embed_single)

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

            # Process in batches if needed
            results: list[dict[str, Any]] = []

            for i in range(0, len(valid_texts), self.MAX_BATCH_SIZE):
                batch_texts: list[str] = valid_texts[i : i + self.MAX_BATCH_SIZE]

                def _embed_batch() -> list[dict[str, Any]]:
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
                        batch_results: list[dict[str, Any]] = []
                        for text, embed_obj in zip(batch_texts, response.embeddings):
                            embedding_values: list[float] = embed_obj.values
                            batch_results.append(
                                {
                                    "embedding": embedding_values,
                                    "model": self.embedding_model,
                                    "dimensions": len(embedding_values),
                                    "text": text,
                                }
                            )
                        return batch_results

                    except Exception as batch_error:
                        _LOGGER.error(
                            "Failed to generate batch embeddings: %s", batch_error
                        )
                        raise

                batch_results: list[dict[str, Any]] = (
                    self._retry_with_exponential_backoff(_embed_batch)
                )
                results.extend(batch_results)

            return results
