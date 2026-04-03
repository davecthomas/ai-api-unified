# ai_titan_embeddings.py

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import boto3  # AWS SDK for Python to call Bedrock runtime service
import botocore
from botocore.config import Config

from ..ai_base import AIBaseEmbeddings, AiApiObservedEmbeddingsResultModel
from ..util.env_settings import EnvSettings

_LOGGER: logging.Logger = logging.getLogger(__name__)


class AiTitanEmbeddings(AIBaseEmbeddings):
    """
    Example class to interface with Amazon Titan Text Embeddings v2
    (model ID: amazon.titan-embed-text-v2:0).
    """

    MAX_EMBEDDING_WORKERS = 32  # Max number of parallel embedding requests

    def __init__(self, model: str = "", dimensions: int = 0):
        """
        Args:
            model (str): Titan v2 model ID, e.g. 'amazon.titan-embed-text-v2:0'.
            dimensions (int): The desired embedding size: 1024 (default), 512, or 256.
        """
        settings = EnvSettings()  # Load settings from the EnvSettings class

        self.embedding_model = model if model else "amazon.titan-embed-text-v2:0"
        self.dimensions = dimensions if dimensions else 1024
        self.region_name = settings.get("AWS_REGION", "us-east-1")

        try:
            # Config and create the Bedrock runtime client, ensuring things don't hang forever (the default behavior).
            bedrock_cfg = Config(
                connect_timeout=10,  # max 10s to open connection
                read_timeout=30,  # max 30s to receive any data
                retries={"max_attempts": 3},  # optional: auto-retry at HTTP level
            )
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.region_name,
                config=bedrock_cfg,
                # aws_access_key_id=aws_access_key,
                # aws_secret_access_key=aws_secret_key,
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to create Bedrock runtime client. "
                "Check that your AWS credentials and region are correctly configured."
            ) from e

        # Exponential backoff intervals
        self.backoff_delays = [1, 2, 4, 8]

    @property
    def model_name(self) -> str:
        """
        Returns the Titan embeddings model identifier this client is using.
        """
        return self.embedding_model

    @property
    def list_model_names(self) -> list[str]:
        # AWS Bedrock ‘Titan Text Embeddings’ models
        return [
            "amazon.titan-embed-text-v2:0",
            "amazon.titan-embed-text-v1",
        ]

    def calculate_cost(self, num_tokens: int) -> float:
        """
        Stub for Titan cost calculation. Adjust once AWS publishes official pricing.
        """
        cost_per_token = 0.00000002
        return cost_per_token * num_tokens

    def generate_embeddings(self, text: str) -> dict[str, Any]:
        """
        Generates an embedding for a single text string using Titan v2.
        """
        if not text.strip():
            raise ValueError("Text must be non-empty for Titan embeddings.")
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
                callable_execute=lambda: self._generate_single_embedding_observed_result(
                    text
                ),
                callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_embeddings_observability_result_summary(
                    observed_result=result,
                    provider_elapsed_ms=provider_elapsed_ms,
                ),
            )
        )
        # Normal return with the caller-facing Titan single embedding payload.
        return observed_result.return_value

    def _generate_single_embedding_observed_result(
        self,
        text: str,
    ) -> AiApiObservedEmbeddingsResultModel[dict[str, Any]]:
        """
        Generates one Titan embedding result container without applying observability wrappers.

        Args:
            text: Non-empty text string to embed through Titan.

        Returns:
            AiApiObservedEmbeddingsResultModel containing the caller-facing payload and
            metadata-only observability fields derived from the Titan response.
        """
        dict_embedding_result, int_input_tokens = self._generate_single_embedding_raw(
            text
        )
        observed_result: AiApiObservedEmbeddingsResultModel[dict[str, Any]] = (
            AiApiObservedEmbeddingsResultModel(
                return_value=dict_embedding_result,
                embedding_count=1,
                returned_dimensions=int(dict_embedding_result["dimensions"]),
                provider_input_tokens=int_input_tokens,
                provider_total_tokens=int_input_tokens,
            )
        )
        # Normal return with the single Titan embedding payload and metadata-only summary inputs.
        return observed_result

    def _generate_single_embedding_raw(
        self,
        text: str,
    ) -> tuple[dict[str, Any], int | None]:
        """
        Executes one Titan embeddings request with existing retry behavior and no observability.

        Args:
            text: Non-empty text string to embed through Titan.

        Returns:
            Tuple containing the caller-facing embedding payload and Titan input token count
            when the provider returns it.
        """
        if not text.strip():
            raise ValueError("Text must be non-empty for Titan embeddings.")
        payload: dict[str, str | int | bool] = {
            "inputText": text,
            "dimensions": self.dimensions,
            "normalize": True,
        }

        # Loop through retry attempts while preserving existing Titan backoff behavior.
        for attempt, delay in enumerate(self.backoff_delays, start=1):
            try:
                dict_response = self.client.invoke_model(
                    modelId=self.embedding_model,
                    body=json.dumps(payload),
                    contentType="application/json",
                )
                response_body: botocore.response.StreamingBody = dict_response.get(
                    "body", None
                )
                if not response_body:
                    raise RuntimeError(
                        "Empty response body from Titan for "
                        f"model {self.embedding_model} with dimensions {self.dimensions}."
                    )

                response_body_bytes: bytes = dict_response["body"].read()
                dict_response_body: dict[str, Any] = json.loads(response_body_bytes)
                titan_embedding: list[float] = dict_response_body.get("embedding", [])

                if not titan_embedding:
                    raise RuntimeError(
                        "Titan response did not contain an embedding vector."
                    )
                if len(titan_embedding) != self.dimensions:
                    raise RuntimeError(
                        "Titan embedding dimensions mismatch. "
                        f"Expected {self.dimensions} but received {len(titan_embedding)}."
                    )

                dict_embedding_result: dict[str, Any] = {
                    "embedding": titan_embedding,
                    "text": text,
                    "dimensions": self.dimensions,
                    "input_tokens": dict_response_body.get("inputTextTokenCount"),
                }
                int_input_tokens: int | None = dict_response_body.get(
                    "inputTextTokenCount"
                )
                # Normal return with the raw Titan embedding payload and provider token count.
                return dict_embedding_result, int_input_tokens

            except Exception as exception:
                _LOGGER.warning(
                    "Titan embedding attempt %s failed: %s", attempt, exception
                )
                if attempt == len(self.backoff_delays):
                    raise RuntimeError(f"Failed Titan embedding request: {exception}")
                time.sleep(delay)

        raise RuntimeError("Titan embedding retries exhausted unexpectedly.")

    def generate_embeddings_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """
        This isn't really a batch call; We'd need to use S2 to make that work and this POC isn't
        designed for that."""
        if not texts or not all(
            isinstance(text, str) and text.strip() for text in texts
        ):
            raise ValueError(
                "generate_embeddings_batch: All input texts must be non-empty strings."
            )
        dict_input_metadata: dict[str, str | int | float | bool | None] = (
            self._build_embeddings_observability_input_metadata(
                list_texts=texts,
                bool_is_batch=True,
                requested_dimensions=self.dimensions,
            )
        )
        observed_result: AiApiObservedEmbeddingsResultModel[list[dict[str, Any]]] = (
            self._execute_provider_call_with_observability(
                capability=self.CLIENT_TYPE_EMBEDDING,
                operation="generate_embeddings_batch",
                dict_input_metadata=dict_input_metadata,
                callable_execute=lambda: self._generate_batch_embeddings_observed_result(
                    texts
                ),
                callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_embeddings_observability_result_summary(
                    observed_result=result,
                    provider_elapsed_ms=provider_elapsed_ms,
                ),
            )
        )
        # Normal return with the caller-facing Titan batch embedding payload.
        return observed_result.return_value

    def _generate_batch_embeddings_observed_result(
        self,
        texts: list[str],
    ) -> AiApiObservedEmbeddingsResultModel[list[dict[str, Any]]]:
        """
        Generates one Titan batch result container without applying observability wrappers per worker.

        Args:
            texts: Non-empty list of input texts to embed.

        Returns:
            AiApiObservedEmbeddingsResultModel containing the caller-facing batch payload and
            aggregated token metadata for the public batch call.
        """
        list_dict_embedding_results: list[dict[str, Any]] = (
            self.generate_embeddings_in_parallel(texts)
        )
        int_total_input_tokens: int | None = 0
        # Loop through batch results so provider token counts are aggregated once for the public call.
        for dict_embedding_result in list_dict_embedding_results:
            int_input_tokens: int | None = dict_embedding_result.get("input_tokens")
            if int_total_input_tokens is not None:
                if int_input_tokens is None:
                    int_total_input_tokens = None
                else:
                    int_total_input_tokens += int_input_tokens
        observed_result: AiApiObservedEmbeddingsResultModel[list[dict[str, Any]]] = (
            AiApiObservedEmbeddingsResultModel(
                return_value=list_dict_embedding_results,
                embedding_count=len(list_dict_embedding_results),
                returned_dimensions=self.dimensions,
                provider_input_tokens=int_total_input_tokens,
                provider_total_tokens=int_total_input_tokens,
            )
        )
        # Normal return with the Titan batch payload and aggregated metadata-only summary inputs.
        return observed_result

    def generate_embeddings_in_parallel(self, texts: list[str]) -> list[dict[str, Any]]:
        """
        Batch embedding for multiple strings by calling generate_embeddings() in parallel.
        Titan v2 itself does not accept multiple texts in one request.
        """
        if not texts:
            raise ValueError("No texts provided for Titan batch embedding.")
        if not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError(
                "generate_embeddings_in_parallel: All input texts must be non-empty strings."
            )

        # Preallocate results list so we can keep outputs in order
        results: list[dict[str, Any] | None] = [None] * len(texts)

        # Parallelize embedding calls to improve throughput
        max_workers = min(
            AiTitanEmbeddings.MAX_EMBEDDING_WORKERS, len(texts)
        )  # cap the pool size
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all embedding tasks
            future_to_index = {
                executor.submit(self._generate_single_embedding_raw, text): idx
                for idx, text in enumerate(texts)
            }

            # As each finishes, store it in the correct slot
            for future in as_completed(future_to_index, timeout=60):
                idx = future_to_index[future]
                try:
                    dict_embedding_data, _ = future.result(timeout=20)
                    results[idx] = dict_embedding_data
                except Exception as exception:
                    # If one text fails, you can decide to skip/log or fail fast.
                    raise RuntimeError(
                        f"Embedding failed for text at index {idx}: {exception}"
                    )

        list_dict_embedding_results: list[dict[str, Any]] = []
        # Loop over preallocated results and fail fast if any task did not produce data.
        for idx, dict_embedding_result in enumerate(results):
            if dict_embedding_result is None:
                raise RuntimeError(f"Embedding result missing for text at index {idx}.")
            list_dict_embedding_results.append(dict_embedding_result)

        return list_dict_embedding_results

    def send_prompt(self, prompt: str) -> str:
        """
        Titan v2 is for embeddings only; not a chat/generation model.
        """
        raise NotImplementedError("send_prompt is not applicable to Titan embeddings.")
