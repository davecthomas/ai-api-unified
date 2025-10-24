from __future__ import (
    annotations,
)  # Postpone evaluation of type hints to avoid circular imports and allow forward references with | None

import json
import math
import uuid
from abc import ABC, abstractmethod
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Iterator, Type

from pydantic import BaseModel, Field, ValidationError, model_validator


class SupportedDataType(Enum):
    """Enumeration of data types supported by AI models."""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    PDF = "pdf"


class AICompletionsCapabilitiesBase(BaseModel):
    """
    Base class for capturing important attributes of completions models.
    """

    context_window_length: int
    knowledge_cutoff_date: date | None = None
    reasoning: bool = False
    supported_data_types: list[SupportedDataType] = [SupportedDataType.TEXT]
    supports_data_residency_constraint: bool = False


class AICompletionsPromptParamsBase(BaseModel, ABC):
    """
    Base class for completion prompt parameters.
    This allows passing media attachments (images, etc.) alongside text prompts as well as
    specifying a system prompt to guide the model's behavior.
    Typical prompting doesn't require any use of this class. All fields are optional.
    """

    DEFAULT_SYSTEM_PROMPT: ClassVar[str] = "You are a helpful assistant."
    DEFAULT_STRICT_SCHEMA_SYSTEM_PROMPT: ClassVar[str] = (
        "Respond only with JSON following the provided schema."
    )
    MAX_IMAGE_BYTES: ClassVar[int] = 20_000_000

    system_prompt: str | None = None
    included_types: list[SupportedDataType] | None = None
    included_data: list[bytes] | None = None
    included_mime_types: list[str] | None = None

    @model_validator(mode="after")
    def _validate_and_normalize_included_media(
        self,
    ) -> "AICompletionsPromptParamsBase":
        """Ensure media attachment lists are aligned and metadata validated."""

        list_types: list[SupportedDataType] = list(self.included_types or [])
        list_data: list[bytes] = list(self.included_data or [])
        list_mime_types: list[str] = list(self.included_mime_types or [])

        lengths: tuple[int, int, int] = (
            len(list_types),
            len(list_data),
            len(list_mime_types),
        )
        unique_lengths: set[int] = {length for length in lengths}
        if len(unique_lengths) > 1:
            raise ValueError(
                "included_types, included_data, and included_mime_types must be the same length."
            )

        if not list_types:
            self.included_data = None
            self.included_mime_types = None
            self.included_types = None
            return self

        for index, media_type in enumerate(list_types):
            mime_type: str | None = None
            if index < len(list_mime_types):
                mime_type = list_mime_types[index]

            if not mime_type:
                raise ValueError("Each included media item must specify a MIME type.")

            if not mime_type.lower().startswith("image/"):
                raise ValueError(
                    f"MIME type {mime_type!r} is not supported. Only image MIME types are allowed."
                )

            if media_type is not SupportedDataType.IMAGE:
                raise ValueError(
                    "Only SupportedDataType.IMAGE attachments are currently accepted."
                )

            media_bytes: bytes = list_data[index]
            if not media_bytes:
                raise ValueError("Image attachment bytes cannot be empty.")
            self.included_data = list_data
            self.included_mime_types = list_mime_types
            self.included_types = list_types
        return self

    def iter_included_media(
        self,
    ) -> Iterator[tuple[int, SupportedDataType, bytes, str]]:
        """
        Yield the index, type, raw bytes, and MIME type for every attached media item.
        """

        list_types: list[SupportedDataType] = list(self.included_types or [])
        list_data: list[bytes] = list(self.included_data or [])
        list_mime_types: list[str] = list(self.included_mime_types or [])

        for index, (media_type, media_bytes, mime_type) in enumerate(
            zip(list_types, list_data, list_mime_types)
        ):
            yield index, media_type, media_bytes, mime_type

    @property
    def has_included_media(self) -> bool:
        """Return True when any media attachments are present."""

        return bool(self.included_types)


class AIBase(ABC):
    """
    Abstract base class that defines methods for interacting with OpenAI
    or any large language model service.
    """

    CLIENT_TYPE_EMBEDDING = "embedding"
    CLIENT_TYPE_COMPLETIONS = "completions"
    CLIENT_TYPE_IMAGES = "images"

    def __init__(self, model: str | None = None):
        super().__init__()
        self.model: str | None = model

    @property
    def model_name(self) -> str | None:
        """
        Identifier of the model in use (e.g. 'gpt-4o-mini').
        """
        return self.model

    @property
    @abstractmethod
    def list_model_names(self) -> list[str]:
        """Supported model identifiers for this client."""
        ...


class AIBaseEmbeddings(AIBase):
    """
    Abstract base class for generating embeddings.
    """

    def __init__(self, model: str | None = None, dimensions: int = 0):
        super().__init__(model=model)
        self.dimensions = dimensions

    @abstractmethod
    def generate_embeddings(self, text: str) -> dict[str, Any]:
        """
        Generates embeddings for a single piece of text.
        """

    @abstractmethod
    def generate_embeddings_batch(self, texts: list[str]) -> list[dict[str, Any]]:
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
    def get_prompt() -> str | None:
        """
        Optional method that subclasses can override to produce
        a “prompt string” given the model’s fields.

        """
        ...

    def send_structured_prompt(
        self,
        ai_client: AIBaseCompletions,
        response_model: Type[AIStructuredPrompt] = None,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> AIStructuredPrompt | None:
        """
        Execute the specific AIStructuredPrompt structured prompt and return the result as a structured object.

        Args:
            ai_client: The completions client used to execute the structured prompt.
            response_model: The expected structured response model.
            other_params: Optional provider-specific parameters, including system prompt overrides.
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
                other_params=other_params,
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
        maxTokens
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
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> AIStructuredPrompt:
        """
        Generates a strict schema prompt and returns the result as a structured object.

        Args:
            prompt: The prompt string to send.
            response_model: The structured response model used for validation.
            max_response_tokens: Maximum number of tokens allowed in the response.
            other_params: Optional provider-specific parameters, including system prompt overrides.
        """

    @abstractmethod
    def send_prompt(
        self, prompt: str, *, other_params: AICompletionsPromptParamsBase | None = None
    ) -> str:
        """
        Sends a prompt to the completions engine and returns the result as a string.

        Args:
            prompt: The text prompt to send
            other_params: Optional provider-specific parameters
        """


class AIBaseImageProperties(BaseModel):
    """Carries width/height (pixels), output format, and image count for generation requests."""

    width: int | None = Field(default=1_536)
    height: int | None = Field(default=1_024)
    format: str = "png"
    quality: str = "medium"
    background: str = "auto"  # transparent, auto
    num_images: int = 1

    @model_validator(mode="after")
    def _validate_dimensions(self) -> "AIBaseImageProperties":
        """Ensure requested dimensions and count are strictly positive."""

        if self.width is not None and self.width <= 0:
            raise ValueError("AIBaseImageProperties.width must be a positive integer.")
        if self.height is not None and self.height <= 0:
            raise ValueError("AIBaseImageProperties.height must be a positive integer.")
        if self.num_images <= 0:
            raise ValueError(
                "AIBaseImageProperties.num_images must be greater than zero."
            )
        return self

    @model_validator(mode="after")
    def _validate_quality(self) -> "AIBaseImageProperties":
        """Ensure requested quality is one of the supported values."""
        valid_qualities = {"low", "medium", "high"}
        if self.quality not in valid_qualities:
            raise ValueError(
                f"AIBaseImageProperties.quality must be one of {valid_qualities}, got {self.quality!r}."
            )
        return self

    @model_validator(mode="after")
    def _validate_format(self) -> "AIBaseImageProperties":
        """Ensure requested format is one of the supported values."""
        valid_formats = {"png", "jpeg", "webp"}
        if self.format.lower() not in valid_formats:
            raise ValueError(
                f"AIBaseImageProperties.format must be one of {valid_formats}, got {self.format!r}."
            )
        return self

    @model_validator(mode="after")
    def _validate_background(self) -> "AIBaseImageProperties":
        """Ensure requested background is one of the supported values."""
        valid_backgrounds = {"transparent", "auto"}
        if self.background.lower() not in valid_backgrounds:
            raise ValueError(
                f"AIBaseImageProperties.background must be one of {valid_backgrounds}, got {self.background!r}."
            )
        return self


class AIBaseImages(AIBase):

    def __init__(self, model: str | None = None, **kwargs: Any):
        super().__init__(model=model, **kwargs)

    def generate_images(
        self, image_prompt: str, image_properties: AIBaseImageProperties
    ) -> list[bytes]:
        """Synchronously generate one or more images for the prompt with requested size and format.

        Implementations must honor the requested width, height, format, and quantity while returning
        raw image bytes so callers can persist or stream the result as needed.
        """
        raise NotImplementedError(
            "This feature has not been implemented for this provider."
        )

    def generate_image_files(
        self,
        image_prompt: str,
        image_properties: AIBaseImageProperties = AIBaseImageProperties(),
        root_file_name: str = "generate_image_files",
    ) -> list[Path]:
        """Generate images, save them, and return saved paths."""

        if root_file_name.strip() == "":
            raise ValueError("root_file_name must be a non-empty string.")

        image_bytes_list: list[bytes] = self.generate_images(
            image_prompt, image_properties
        )
        if not image_bytes_list:
            raise ValueError("generate_images returned no image data to save.")

        root_path: Path = Path(root_file_name)
        directory_path: Path = (
            root_path.parent if root_path.parent != Path("") else Path(".")
        )
        directory_path.mkdir(parents=True, exist_ok=True)

        suffix: str
        stem: str
        if root_path.suffix:
            suffix = root_path.suffix
            stem = root_path.stem
        else:
            suffix = f".{image_properties.format.lower()}"
            stem = root_path.name

        saved_paths: list[Path] = []
        for index, image_bytes in enumerate(image_bytes_list, start=1):
            file_name: str = f"{stem}_{index}{suffix}"
            file_path: Path = directory_path / file_name
            # WHY: Sequential suffixes prevent overwriting when multiple images are requested.
            file_path.write_bytes(image_bytes)
            saved_paths.append(file_path)

        return saved_paths
