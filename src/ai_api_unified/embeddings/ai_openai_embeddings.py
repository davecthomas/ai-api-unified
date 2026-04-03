# ai_openai_embeddings.py

import logging
import random
import time
from typing import Any

import httpx
from openai import BadRequestError, OpenAI
from openai.types import CreateEmbeddingResponse, EmbeddingCreateParams

from ai_api_unified.ai_openai_base import AIOpenAIBase

from ..ai_base import AIBaseEmbeddings, AiApiObservedEmbeddingsResultModel
from ..util.env_settings import EnvSettings

_LOGGER: logging.Logger = logging.getLogger(__name__)


class AiOpenAIEmbeddings(AIBaseEmbeddings, AIOpenAIBase):
    """
    Handles OpenAI embedding operations.
    """

    # Static list of embedding models with their settings
    embedding_models: dict = {
        "text-embedding-ada-002": {"dimensions": 1536, "pricing_per_token": 0.0004},
        "text-embedding-3-small": {"dimensions": 1536, "pricing_per_token": 0.00025},
        "text-embedding-3-large": {"dimensions": 3072, "pricing_per_token": 0.0005},
    }

    embedding_model_default = "text-embedding-3-small"
    EMBEDDING_BATCH_MAX_SIZE = 2048
    MODELS_SUPPORTING_CUSTOM_DIMENSIONS: set[str] = {
        "text-embedding-3-small",
        "text-embedding-3-large",
    }

    @property
    def list_model_names(self) -> list[str]:
        # As of May 2025, aggregated from OpenAI docs and release notes:
        return [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ]

    def __init__(self, model: str = "text-embedding-3-small", dimensions: int = 512):
        env = EnvSettings()
        self.api_key = env.get_setting("OPENAI_API_KEY")
        self.embedding_model = model
        self.dimensions = dimensions
        self.base_url = self.get_api_base_url()
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.backoff_delays = [1, 2, 4, 8, 16]
        self.user = env.get_setting("OPENAI_USER", "default_user")

    @property
    def model_name(self) -> str:
        """
        Returns the OpenAI embeddings model identifier this client is using.
        """
        return self.embedding_model

    def calculate_cost(self, num_tokens: int) -> float:
        """
        Calculate the cost of generating embeddings based on the number of tokens.

        Args:
            num_tokens (int): The number of tokens used.

        Returns:
            float: The calculated cost.
        """
        pricing_per_token: float = self.embedding_models[self.embedding_model].get(
            "pricing_per_token", 0.0
        )
        cost = pricing_per_token * num_tokens
        return cost

    def generate_embeddings(self, text: str) -> dict[str, Any]:
        """
        Generates embeddings for a given text using OpenAI's embeddings API.

        Args:
            text (str): The text for which to generate embeddings.

        Returns:
            dict[str, Any]: A dictionary containing the embeddings and metadata.
        """
        if not text:
            raise ValueError("Text is required for generating embeddings.")
        # Construct the request parameters for one single-text embeddings call.
        params: EmbeddingCreateParams = {
            "input": [text],
            "model": self.embedding_model,
        }
        if self.embedding_model in self.MODELS_SUPPORTING_CUSTOM_DIMENSIONS:
            params["dimensions"] = self.dimensions
        explicit_caller_id: str | None = self._get_explicit_observability_caller_id()
        if explicit_caller_id is not None:
            params["user"] = explicit_caller_id

        dict_input_metadata: dict[str, str | int | float | bool | None] = (
            self._build_embeddings_observability_input_metadata(
                list_texts=[text],
                bool_is_batch=False,
                requested_dimensions=self.dimensions,
            )
        )

        def _execute_single_embedding() -> (
            AiApiObservedEmbeddingsResultModel[dict[str, Any]]
        ):
            max_retries: int = len(self.backoff_delays)

            # Loop through retry attempts while keeping one observability sequence per public call.
            for attempt in range(max_retries):
                try:
                    response: CreateEmbeddingResponse = self.client.embeddings.create(
                        **params
                    )
                    if not response.data:
                        raise RuntimeError(
                            "OpenAI embeddings response did not contain any embeddings."
                        )
                    embedding: list[float] = response.data[0].embedding
                    if (
                        self.embedding_model in self.MODELS_SUPPORTING_CUSTOM_DIMENSIONS
                        and len(embedding) != self.dimensions
                    ):
                        raise RuntimeError(
                            "OpenAI embeddings dimensions mismatch. "
                            f"Expected {self.dimensions} but received {len(embedding)}."
                        )
                    dict_embedding_result: dict[str, Any] = {
                        "embedding": embedding,
                        "text": text,
                        "dimensions": len(embedding),
                    }
                    observed_result: AiApiObservedEmbeddingsResultModel[
                        dict[str, Any]
                    ] = AiApiObservedEmbeddingsResultModel(
                        return_value=dict_embedding_result,
                        embedding_count=1,
                        returned_dimensions=len(embedding),
                        provider_input_tokens=self._extract_openai_prompt_tokens(
                            response
                        ),
                        provider_total_tokens=self._extract_openai_total_tokens(
                            response
                        ),
                    )
                    # Normal return with the single embedding result and metadata-only summary inputs.
                    return observed_result
                except httpx.TimeoutException as exception:
                    wait_time: int = self.backoff_delays[attempt]
                    _LOGGER.warning(
                        "OpenAI embeddings attempt %s failed: %s. Retrying in %s seconds.",
                        attempt + 1,
                        exception,
                        wait_time,
                    )
                    time.sleep(wait_time + random.uniform(0, 1))

                except httpx.HTTPStatusError as exception:
                    if exception.response.status_code == 429:
                        wait_time = self.backoff_delays[attempt]
                        _LOGGER.warning(
                            "OpenAI embeddings rate limit reached. Retrying in %s seconds.",
                            wait_time,
                        )
                        time.sleep(wait_time + random.uniform(0, 1))
                        continue
                    _LOGGER.exception(
                        "OpenAI embeddings HTTP error occurred: %s", exception
                    )
                    raise

                except Exception as exception:
                    _LOGGER.exception(
                        "OpenAI embeddings unexpected error occurred: %s",
                        exception,
                    )
                    raise

            raise TimeoutError(
                "Failed to generate embeddings after multiple retries due to repeated timeouts."
            )

        observed_result: AiApiObservedEmbeddingsResultModel[dict[str, Any]] = (
            self._execute_provider_call_with_observability(
                capability=self.CLIENT_TYPE_EMBEDDING,
                operation="generate_embeddings",
                dict_input_metadata=dict_input_metadata,
                callable_execute=_execute_single_embedding,
                callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_embeddings_observability_result_summary(
                    observed_result=result,
                    provider_elapsed_ms=provider_elapsed_ms,
                ),
                legacy_caller_id=self.user,
            )
        )
        # Normal return with the caller-facing single embedding payload.
        return observed_result.return_value

    def generate_embeddings_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """
        Generates embeddings for a batch of texts using OpenAI's embeddings API.

        Args:
            texts (list[str]): A list of text strings for which to generate embeddings.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing embeddings and metadata.
        """
        if not texts or not all(
            isinstance(text, str) and text.strip() for text in texts
        ):
            raise ValueError(
                "generate_embeddings_batch: All input texts must be non-empty strings."
            )

        # Construct the request parameters for one batch embeddings call.
        params: EmbeddingCreateParams = {
            "input": texts,
            "model": self.embedding_model,
        }
        if self.embedding_model in self.MODELS_SUPPORTING_CUSTOM_DIMENSIONS:
            params["dimensions"] = self.dimensions
        explicit_caller_id: str | None = self._get_explicit_observability_caller_id()
        if explicit_caller_id is not None:
            params["user"] = explicit_caller_id

        dict_input_metadata: dict[str, str | int | float | bool | None] = (
            self._build_embeddings_observability_input_metadata(
                list_texts=texts,
                bool_is_batch=True,
                requested_dimensions=self.dimensions,
            )
        )

        def _execute_batch_embeddings() -> (
            AiApiObservedEmbeddingsResultModel[list[dict[str, Any]]]
        ):
            max_retries: int = len(self.backoff_delays)
            attempt: int = 0

            while attempt < max_retries:
                try:
                    response: CreateEmbeddingResponse = self.client.embeddings.create(
                        **params
                    )
                    if len(response.data) != len(texts):
                        raise RuntimeError(
                            "OpenAI embeddings response count mismatch. "
                            f"Requested {len(texts)} embeddings but received {len(response.data)}."
                        )
                    list_embeddings: list[list[float]] = [
                        data.embedding for data in response.data
                    ]
                    list_dict_embeddings: list[dict[str, Any]] = []
                    # Loop through provider vectors so the caller-facing batch payload preserves input order.
                    for current_text, embedding in zip(texts, list_embeddings):
                        if (
                            self.embedding_model
                            in self.MODELS_SUPPORTING_CUSTOM_DIMENSIONS
                            and len(embedding) != self.dimensions
                        ):
                            raise RuntimeError(
                                "OpenAI embeddings dimensions mismatch. "
                                f"Expected {self.dimensions} but received {len(embedding)}."
                            )
                        list_dict_embeddings.append(
                            {
                                "embedding": embedding,
                                "text": current_text,
                                "dimensions": len(embedding),
                            }
                        )
                    int_returned_dimensions: int | None = None
                    if list_embeddings:
                        int_returned_dimensions = len(list_embeddings[0])
                    observed_result: AiApiObservedEmbeddingsResultModel[
                        list[dict[str, Any]]
                    ] = AiApiObservedEmbeddingsResultModel(
                        return_value=list_dict_embeddings,
                        embedding_count=len(list_embeddings),
                        returned_dimensions=int_returned_dimensions,
                        provider_input_tokens=self._extract_openai_prompt_tokens(
                            response
                        ),
                        provider_total_tokens=self._extract_openai_total_tokens(
                            response
                        ),
                    )
                    # Normal return with the caller-facing batch embeddings payload.
                    return observed_result

                except (httpx.TimeoutException, httpx.HTTPStatusError) as exception:
                    wait_time: int = self.backoff_delays[attempt]
                    if (
                        isinstance(exception, httpx.HTTPStatusError)
                        and exception.response.status_code != 429
                    ):
                        _LOGGER.exception(
                            "OpenAI embeddings HTTP error occurred: %s", exception
                        )
                        raise

                    _LOGGER.warning(
                        "OpenAI embeddings batch attempt %s failed: %s. Retrying in %s seconds.",
                        attempt + 1,
                        exception,
                        wait_time,
                    )
                    time.sleep(wait_time + random.uniform(0, 1))
                    attempt += 1

                except BadRequestError as exception:
                    _LOGGER.exception(
                        "OpenAI embeddings bad request error occurred: %s", exception
                    )
                    raise

                except Exception as exception:
                    _LOGGER.exception(
                        "OpenAI embeddings unexpected error occurred: %s",
                        exception,
                    )
                    raise

            raise TimeoutError(
                "Failed to generate embeddings after multiple retries due to repeated errors."
            )

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
                legacy_caller_id=self.user,
            )
        )
        # Normal return with the caller-facing batch embeddings payload.
        return observed_result.return_value

    def _extract_openai_prompt_tokens(
        self,
        response: CreateEmbeddingResponse,
    ) -> int | None:
        """
        Extracts the provider-reported input token count from an OpenAI embeddings response.

        Args:
            response: OpenAI embeddings response object returned by the provider SDK.

        Returns:
            Provider-reported prompt token count, or None when unavailable.
        """
        usage = response.usage
        if usage is None:
            # Early return because the provider response did not include usage data.
            return None
        # Normal return with the provider-reported prompt token count.
        return usage.prompt_tokens

    def _extract_openai_total_tokens(
        self,
        response: CreateEmbeddingResponse,
    ) -> int | None:
        """
        Extracts the provider-reported total token count from an OpenAI embeddings response.

        Args:
            response: OpenAI embeddings response object returned by the provider SDK.

        Returns:
            Provider-reported total token count, or None when unavailable.
        """
        usage = response.usage
        if usage is None:
            # Early return because the provider response did not include usage data.
            return None
        # Normal return with the provider-reported total token count.
        return usage.total_tokens
