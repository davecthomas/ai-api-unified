# ai_openai_embeddings.py

from ..ai_base import AIBaseEmbeddings
from openai import BadRequestError, OpenAI
from openai.types import EmbeddingCreateParams, CreateEmbeddingResponse
import httpx
import time
import random
from ..util.env_settings import EnvSettings
from typing import List, Dict, Any


class AiOpenAIEmbeddings(AIBaseEmbeddings):
    """
    Handles OpenAI embedding operations.
    """

    # Static list of embedding models with their settings
    embedding_models: Dict = {
        "text-embedding-ada-002": {"dimensions": 1536, "pricing_per_token": 0.0004},
        "text-embedding-3-small": {"dimensions": 1536, "pricing_per_token": 0.00025},
        "text-embedding-3-large": {"dimensions": 3072, "pricing_per_token": 0.0005},
    }

    embedding_model_default = "text-embedding-3-small"
    EMBEDDING_BATCH_MAX_SIZE = 2048

    @property
    def list_model_names(self) -> List[str]:
        # As of May 2025, aggregated from OpenAI docs and release notes:
        return [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ]

    def __init__(self, model: str = "text-embedding-3-small", dimensions: int = 512):
        self.env = EnvSettings()
        self.api_key = self.env.get_setting("OPENAI_API_KEY")
        self.embedding_model = model
        self.dimensions = dimensions
        self.client = OpenAI(api_key=self.api_key)
        self.backoff_delays = [1, 2, 4, 8, 16]
        # Is this needed?
        self.user = self.env.get_setting("OPENAI_USER", "default_user")

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

    def generate_embeddings(self, text: str) -> Dict[str, Any]:
        """
        Generates embeddings for a given text using OpenAI's embeddings API.

        Args:
            text (str): The text for which to generate embeddings.

        Returns:
            Dict[str, Any]: A dictionary containing the embeddings and metadata.
        """
        if not text:
            raise ValueError("Text is required for generating embeddings.")
        # Construct the parameters using EmbeddingCreateParams
        params: EmbeddingCreateParams = {
            "input": [text],  # Input text as a list of strings
            "model": self.embedding_model,  # Model to use for embedding
        }

        max_retries = len(self.backoff_delays)

        for attempt in range(max_retries):
            try:
                # Call the API to create the embedding
                response: CreateEmbeddingResponse = self.client.embeddings.create(
                    **params
                )

                # Extract the embedding from the response
                embedding = response.data[0].embedding
                return {
                    "embedding": embedding,
                    "text": text,
                    "dimensions": self.dimensions,  # Dimensions of the embedding
                    # "user": self.user,  # User identifier
                }
            except httpx.TimeoutException as e:
                wait_time = self.backoff_delays[attempt]
                print(
                    (
                        f"\tOpenAI embeddings attempt "
                        f"{attempt + 1} failed: {e}. Retrying in {wait_time} seconds..."
                    )
                )
                time.sleep(wait_time + random.uniform(0, 1))

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limit error
                    wait_time = self.backoff_delays[attempt]
                    print(
                        (
                            f"\tRate limit reached. Retrying in "
                            f"{wait_time} seconds..."
                        )
                    )
                    time.sleep(wait_time + random.uniform(0, 1))
                else:
                    print(f"\tHTTP error occurred: {e}")
                    raise

            except Exception as e:
                print(f"\tAn unexpected error occurred: {e}")
                raise

        raise TimeoutError(
            "Failed to generate embeddings after multiple retries due to repeated timeouts."
        )

    def generate_embeddings_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Generates embeddings for a batch of texts using OpenAI's embeddings API.

        Args:
            texts (List[str]): A list of text strings for which to generate embeddings.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing embeddings and metadata.
        """
        if not texts or not all(
            isinstance(text, str) and text.strip() for text in texts
        ):
            raise ValueError(
                "generate_embeddings_batch: All input texts must be non-empty strings."
            )

        # Construct the parameters using EmbeddingCreateParams
        params: EmbeddingCreateParams = {
            "input": texts,  # List of input texts
            "model": self.embedding_model,  # Model to use for embedding
        }

        max_retries = len(self.backoff_delays)
        attempt = 0

        while attempt < max_retries:
            try:
                # Call the API to create embeddings
                response: CreateEmbeddingResponse = self.client.embeddings.create(
                    **params
                )

                # Extract embeddings from the response
                embeddings = [data.embedding for data in response.data]

                # Prepare the list of embedding dictionaries
                embeddings_list = []
                for text, embedding in zip(texts, embeddings):
                    embeddings_list.append(
                        {
                            "embedding": embedding,
                            "text": text,
                            "dimensions": self.dimensions,
                            # "user": self.user,
                        }
                    )

                return embeddings_list

            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                wait_time = self.backoff_delays[attempt]
                if (
                    isinstance(e, httpx.HTTPStatusError)
                    and e.response.status_code != 429
                ):
                    print(f"HTTP error occurred: {e}")
                    raise  # Re-raise exception for non-rate limit errors

                print(
                    (
                        f"Attempt "
                        f"{attempt + 1} failed with error: {e}. Retrying in {wait_time} seconds..."
                    )
                )
                time.sleep(wait_time + random.uniform(0, 1))
                attempt += 1

            except BadRequestError as e:  # Specific exception for bad request
                print(f"BadRequestError occurred: {e}")
                raise  # Re-raise the exception or handle it as needed

            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                raise

        raise TimeoutError(
            "Failed to generate embeddings after multiple retries due to repeated errors."
        )
