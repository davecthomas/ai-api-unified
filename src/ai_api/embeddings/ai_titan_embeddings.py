# ai_titan_embeddings.py

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
from typing import List, Dict, Any
from botocore.config import Config
import boto3
import botocore
from ..ai_base import AIBaseEmbeddings
from ..util.env_settings import EnvSettings


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

        self.model = model if model else "amazon.titan-embed-text-v2:0"
        self.dimensions = dimensions if dimensions else 1024
        self.region_name = settings.get("AWS_REGION", "us-east-1")

        # These are set in the environment and AWS automatically uses them.
        # aws_access_key = settings.get("AWS_ACCESS_KEY_ID")
        # aws_secret_key = settings.get("AWS_SECRET_ACCESS_KEY")

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
    def list_model_names(self) -> List[str]:
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

    def generate_embeddings(self, text: str) -> Dict[str, Any]:
        """
        Generates an embedding for a single text string using Titan v2.
        """
        if not text.strip():
            raise ValueError("Text must be non-empty for Titan embeddings.")

        # Construct the request payload. Additional fields are optional:
        # "normalize": True/False, "embeddingTypes": ["float"], etc.
        payload = {"inputText": text, "dimensions": self.dimensions, "normalize": True}

        # Attempt the call with simple retry logic
        for attempt, delay in enumerate(self.backoff_delays, start=1):
            try:
                response = self.client.invoke_model(
                    modelId=self.model,
                    body=json.dumps(payload),
                    contentType="application/json",
                )
                # Titan v2 returns something like:
                # { "embedding": [...], "inputTextTokenCount": int, ... }
                response_body: botocore.response.StreamingBody = response.get(
                    "body", None
                )
                if not response_body:
                    raise RuntimeError(
                        f"Empty response body from Titan for payload: '{json.dumps(payload)}'."
                    )

                response_body_bytes = response["body"].read()
                dict_response_body = json.loads(response_body_bytes)
                titan_embedding = dict_response_body.get("embedding", [])

                if len(titan_embedding) != self.dimensions:
                    raise RuntimeError("No embedding found in Titan response.")

                return {
                    "embedding": titan_embedding,
                    "text": text,
                    "dimensions": self.dimensions,
                    "input_tokens": dict_response_body.get("inputTextTokenCount", 0),
                }

            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt == len(self.backoff_delays):
                    raise RuntimeError(f"Failed Titan embedding request: {e}")
                time.sleep(delay)

    def generate_embeddings_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        This isn't really a batch call; We'd need to use S2 to make that work and this POC isn't
        designed for that."""
        return self.generate_embeddings_in_parallel(texts)

    def generate_embeddings_in_parallel(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Batch embedding for multiple strings by calling generate_embeddings() in parallel.
        Titan v2 itself does not accept multiple texts in one request.
        """
        if not texts:
            raise ValueError("No texts provided for Titan batch embedding.")

        # Preallocate results list so we can keep outputs in order
        results: List[Dict[str, Any]] = [None] * len(texts)

        # Parallelize embedding calls to improve throughput
        max_workers = min(
            AiTitanEmbeddings.MAX_EMBEDDING_WORKERS, len(texts)
        )  # cap the pool size
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all embedding tasks
            future_to_index = {
                executor.submit(self.generate_embeddings, text): idx
                for idx, text in enumerate(texts)
            }

            # As each finishes, store it in the correct slot
            for future in as_completed(future_to_index, timeout=60):
                idx = future_to_index[future]
                try:
                    embedding_data = future.result(timeout=20)
                    results[idx] = embedding_data
                except Exception as e:
                    # If one text fails, you can decide to skip/log or fail fast.
                    raise RuntimeError(f"Embedding failed for text at index {idx}: {e}")

        return results

    def send_prompt(self, prompt: str) -> str:
        """
        Titan v2 is for embeddings only; not a chat/generation model.
        """
        raise NotImplementedError("send_prompt is not applicable to Titan embeddings.")
