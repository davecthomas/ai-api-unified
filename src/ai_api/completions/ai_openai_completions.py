# ai_openai_completions.py

import json
import time
import warnings
from openai import OpenAI
from typing import ClassVar, List, Dict, Any, Type
import pandas as pd
from pydantic import ValidationError, field_validator

from ..ai_base import AIBaseCompletions, AIStructuredPrompt

from ..util.env_settings import EnvSettings

# I return the full product name column in a dataframe from _create_unique_product_name_key so this warning is bogus
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


class AiOpenAICompletions(AIBaseCompletions):
    def __init__(self, model: str = "4o-mini", dimensions: int = 512):
        """
        Initializes the GhvOpenAI class, setting the model and dimensions.

        Args:
            model (str): The embedding model to use.
            dimensions (int): The number of dimensions for the embeddings.
        """
        env = EnvSettings()
        self.api_key = env.get_setting("OPENAI_API_KEY")
        self.embedding_model = model
        self.dimensions = dimensions
        self.user = env.get_setting("OPENAI_USER", "default_user")
        self.completions_model = env.get_setting("COMPLETIONS_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=self.api_key)
        self.backoff_delays = [1, 2, 4, 8, 16]

    # dollars per 1 k tokens for each supported model
    DICT_OPENAI_PRICES: ClassVar[Dict[str, float]] = {
        "gpt-4o-mini": 0.0005,
        "gpt-o4-mini": 0.0010,
        "gpt-4.1-mini": 0.0020,
        "gpt-4.1-nano": 0.0001,
    }

    @property
    def list_model_names(self) -> List[str]:
        # As of May 2025, aggregated from OpenAI docs and release notes:
        return [
            "gpt-4o",
            "gpt-4o-mini",
            "o4-mini",
            "o4-mini-high",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
        ]

    @property
    def price_per_1k_tokens(self) -> float:
        """
        Look up the cost-per-1 k tokens for this model.
        Returns 0.0 if unknown (no guard).
        """
        return self.DICT_OPENAI_PRICES.get(self.completions_model, 0.0)

    DICT_OPENAI_CONTEXT_WINDOWS: dict[str, int] = {
        "gpt-4o-mini": 128_000,
        "gpt-o4-mini": 128_000,
        "gpt-4.1-mini": 1_000_000,
        "gpt-4.1-nano": 100_000,
        # …add any others we support…
    }

    @property
    def max_context_tokens(self) -> int:
        """
        Look up the OpenAI context window for the current model_name.
        Falls back to 0 if unknown (i.e. no guard will occur).
        """
        return self.DICT_OPENAI_CONTEXT_WINDOWS.get(self.completions_model, 0)

    def strict_schema_prompt(
        self,
        prompt: str,
        response_model: Type[AIStructuredPrompt],
        max_response_tokens: int = 512,
    ) -> AIStructuredPrompt:
        """
        Sends a prompt to the OpenAI API using function calling to enforce a JSON schema
        and parses the response into the specified Pydantic model.

        Args:
            prompt (str): The prompt string to send.
            strict_schema (Dict[str, Any]): The JSON schema to enforce.
            response_model (Type[PromptResult]): The Pydantic model to parse the response into.

        Returns:
            PromptResult: An instance of the specified Pydantic model containing the parsed response.
        """
        # Include a brief system instruction to nudge the model toward JSON-only output
        messages = [
            {
                "role": "system",
                "content": "Respond only with JSON following the provided schema.",
            },
            {"role": "user", "content": prompt},
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
                    functions=functions,
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

    def send_prompt(self, prompt: str) -> str:
        """
        Sends a prompt to the latest version of the OpenAI API for chat and returns the completion result.

        Args:
            prompt (str): The prompt string to send.

        Returns:
            str: The completion result as a string.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.completions_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            # Extract the response from the completion
            completion = response.choices[0].message.content

            # If the content seems truncated, send a follow-up request or handle continuation
            while response.choices[0].finish_reason == "length":
                response = self.client.chat.completions.create(
                    model=self.completions_model,
                    messages=[
                        {"role": "system", "content": "Continue."},
                    ],
                )
                completion += response.choices[0].message.content
            return completion

        except Exception as e:
            print(f"An error occurred while sending the prompt: {e}")
            raise
