from abc import ABC, abstractmethod
import json
from typing import List, Dict, Any, Optional, Type
import uuid
import math
from pydantic import BaseModel, ValidationError, model_validator
import tiktoken


class AIBase(ABC):
    """
    Abstract base class that defines methods for interacting with OpenAI
    or any large language model service.
    """

    CLIENT_TYPE_EMBEDDING = "embedding"
    CLIENT_TYPE_COMPLETIONS = "completions"

    @abstractmethod
    def __init__(self, model: str = "", dimensions: int = 0):
        pass

    @property
    @abstractmethod
    def list_model_names(self) -> List[str]:
        """Supported model identifiers for this client."""
        ...

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text based on a generic tokenizer cl100k_base.

        Args:
            text (str): The text to be tokenized.

        Returns:
            int: The number of tokens in the text.
        """
        # Get the appropriate tokenizer for the model
        tokenizer = tiktoken.get_encoding("cl100k_base")

        # Tokenize the text and count the tokens
        tokens = tokenizer.encode(text)
        return len(tokens)


class AIBaseEmbeddings(AIBase):
    """
    Abstract base class for generating embeddings.
    """

    @property
    @abstractmethod
    def list_model_names(self) -> List[str]:
        """Embedding model IDs (e.g. amazon.titan-embed-text-v2:0)."""
        ...

    @abstractmethod
    def generate_embeddings(self, text: str) -> Dict[str, Any]:
        """
        Generates embeddings for a single piece of text.
        """

    @abstractmethod
    def generate_embeddings_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Generates embeddings for multiple pieces of text in a single API call.
        """


class AIStructuredPrompt(BaseModel):
    """
    Base class for all structured prompts.
    This class is used to define the structure of the results returned by the AI model.

    """

    prompt: str = ""  # This is automatically populated after validation

    @model_validator(mode="after")
    def _populate_prompt(self: "AIStructuredPrompt", __: Any) -> "AIStructuredPrompt":
        """
        After validation, build and store the prompt string
        """
        object.__setattr__(
            self,
            "prompt",
            self.get_prompt(),
        )
        return self

    @classmethod
    def model_json_schema(cls) -> dict:
        from copy import deepcopy

        schema = deepcopy(super().model_json_schema())
        schema.setdefault("required", [])
        return schema

    def __str__(self):
        # Dump only the fields you actually care about (skip None/defaults)
        return self.model_dump_json(
            exclude_none=True,
            exclude_defaults=True,
        )

    @staticmethod
    @abstractmethod
    def get_prompt() -> Optional[str]:
        """
        Optional method that subclasses can override to produce
        a “prompt string” given the model’s fields.

        """
        ...

    def send_structured_prompt(
        self,
        ai_client: "AIBaseCompletions",
        response_model: Type["AIStructuredPrompt"] = None,
    ) -> Optional["AIStructuredPrompt"]:
        """
        Execute the specific AIStructuredPrompt structured prompt and return the result as a structured object.
        """
        if self.prompt is None:
            raise ValueError(
                "You must provide a prompt string to send_structured_prompt(). "
                "This is done by calling the classmethod get_prompt() on the subclass."
            )
        if response_model is None:
            raise ValueError(
                "You must provide a response_model to send_structured_prompt(). "
                "This is done by passing the class itself, e.g. a non-abstract subclass of AIStructuredPrompt."
            )
        try:
            return ai_client.strict_schema_prompt(
                prompt=self.prompt,
                response_model=response_model,
            )
        except ValidationError as ve:
            print(f"Validation errors: {ve.errors()}")
            # either return None or raise a more descriptive error:
            return None

        except Exception as exc:
            # any other unexpected error
            print(f"Unexpected error sending structured prompt: {exc}")
            return None


class AIBaseCompletions(AIBase):
    """
    Base class for generating text completions.
    """

    @property
    @abstractmethod
    def list_model_names(self) -> List[str]:
        """Embedding model IDs (e.g. amazon.titan-embed-text-v2:0)."""
        ...

    @property
    @abstractmethod
    def max_context_tokens(self) -> int:
        """
        Return the maximum number of tokens supported by the model's
        context window.  Concrete subclasses must implement this.
        """
        ...

    @property
    @abstractmethod
    def price_per_1k_tokens(self) -> float:
        """
        USD cost for 1 000 tokens (in+out) on this model.
        Subclasses **must** override.
        """
        ...

    @staticmethod
    def generate_prompt_addendum_json_schema_instruction(
        response_model: Type[AIStructuredPrompt],
        *,
        code_fence: bool = True,
    ) -> str:
        """
        Generate a prompt addendum that tells the model to return ONLY JSON
        matching the model’s JSON Schema.

        Parameters
        ----------
        response_model
            A PromptResult subclass implementing `model_json_schema()`.
        code_fence
            If True, wraps the schema in a ```json …``` fence for clarity.

        Returns
        -------
        A string you can append to your user prompt.
        """
        # 1) grab the minimal JSON Schema dict
        schema = response_model.model_json_schema()

        # 2) pretty-print it
        schema_str = json.dumps(schema, indent=2)

        # 3) wrap in code fences if requested
        if code_fence:
            schema_str = f"```json\n{schema_str}\n```"

        # 4) build the instruction
        return (
            "Return *only* a JSON object (not Markdown) that matches the following JSON Schema "
            "and nothing else:\n"
            f"{schema_str}"
        )

    @staticmethod
    def generate_prompt_entropy_tag(prefix: str = "nonce") -> str:
        """
        Returns a short random tag such as 'nonce:5e3a7c2d'.

        * prefix  - leading label so you can grep for it in logs.
        * Uses uuid4 → 128-bit randomness → virtually zero chance of repeat.
        * Only first 8 hex chars are kept to keep prompts small.
        """
        random_hex = uuid.uuid4().hex[:8]  # e.g. '5e3a7c2d'
        return f"{prefix}:{random_hex}"

    @staticmethod
    def estimate_max_tokens(
        n: int,
        *,
        avg_words_per_phrase: float = 2.5,
        tokens_per_word: float = 1.3,
        json_overhead_tokens: int = 12,
        chain_of_thought_allowance: int = 120,
        safety_margin: float = 1.15,
    ) -> int:
        """
        Heuristic for Bedrock • maxTokens
        --------------------------------
        n                       – number of phrases you’ll ask the model to return
        avg_words_per_phrase    – average length of each phrase (default 2.5 words)
        tokens_per_word         – ~1.3 is OpenAI/BPE average
        json_overhead_tokens    – brackets, quotes, commas, field name
        chain_of_thought_allowance – room for the model’s <thinking> preamble
        safety_margin           – final head-room factor so we don’t truncate

        Returns an **int**, rounded up to the nearest multiple of 16 (just tidy).
        """
        tokens_for_phrases = (
            n * avg_words_per_phrase * tokens_per_word  # natural language
            + n  # one token/phrase for quotes & commas
        )

        raw_total = (
            tokens_for_phrases + json_overhead_tokens + chain_of_thought_allowance
        ) * safety_margin

        # Round up to nearest multiple of 16 (helpful for later batching)
        return int(math.ceil(raw_total / 16.0) * 16)

    @abstractmethod
    def strict_schema_prompt(
        self,
        prompt: str,
        response_model: Type[AIStructuredPrompt],
        max_response_tokens: int = 512,
    ) -> AIStructuredPrompt:
        """
        Generates a strict schema prompt and returns the result as a structured object.
        """

    @abstractmethod
    def send_prompt(self, prompt: str) -> str:
        """
        Sends a prompt to the completions engine and returns the result as a string.
        """
