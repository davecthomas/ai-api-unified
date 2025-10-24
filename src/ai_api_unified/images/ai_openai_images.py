# ai_openai_images.py

import base64
import binascii
import logging
import time

from typing import Any, ClassVar

from openai import OpenAIError

from ai_api_unified.ai_openai_base import AIOpenAIBase

from ..ai_base import AIBaseImageProperties, AIBaseImages
from pydantic import model_validator


_LOGGER: logging.Logger = logging.getLogger(__name__)


class AIOpenAIImageProperties(AIBaseImageProperties):
    """
    OpenAI-specific image properties constrained to supported dimensions.

    width and height can be left unset (None) to let the provider determine the default size.
    """

    width: int | None = None
    height: int | None = None

    _ALLOWED_DIMENSIONS: ClassVar[set[tuple[int, int]]] = {
        (1_024, 1_024),
        (1_024, 1_536),
        (1_536, 1_024),
    }

    @model_validator(mode="after")
    def _validate_openai_dimensions(self) -> "AIOpenAIImageProperties":
        if self.width is None and self.height is None:
            return self
        if self.width is None or self.height is None:
            raise ValueError(
                "Both width and height must be provided for OpenAI image requests when specifying dimensions."
            )
        if (self.width, self.height) not in self._ALLOWED_DIMENSIONS:
            raise ValueError(
                "OpenAI image dimensions must be one of 1024x1024, 1024x1536, or 1536x1024."
            )
        return self


class AIOpenAIImages(AIOpenAIBase, AIBaseImages):
    def __init__(self, model: str | None = None, **kwargs: Any):
        """
        Initializes the AIOpenAIImages class, setting the model and related configuration.

        Args:
            model (str): The image model to use.
        """
        super().__init__(model=model, **kwargs)
        image_model = model
        if image_model is None:
            image_model = self.env.get_setting("IMAGE_MODEL_NAME", "gpt-image-1")
        if image_model.strip() == "":
            raise ValueError(
                "IMAGE_MODEL_NAME environment variable must be set to a valid OpenAI image model name."
            )
        self.image_model_name: str = image_model.strip()

    def model_name(self) -> str:
        """Return the name of the image model in use."""
        return self.image_model_name

    def list_model_names(self) -> list[str]:
        """Return the list of image model names supported by this client."""
        return ["gpt-image-1", "dall-e-2", "dall-e-3"]

    def generate_images(
        self, image_prompt: str, image_properties: AIBaseImageProperties
    ) -> list[bytes]:
        """Generate images honoring requested dimensions, format, and quantity, returning raw bytes."""

        if image_prompt.strip() == "":
            raise ValueError("image_prompt must be a non-empty string.")

        openai_props: AIOpenAIImageProperties
        if isinstance(image_properties, AIOpenAIImageProperties):
            openai_props = image_properties
        else:
            openai_props = AIOpenAIImageProperties(
                width=image_properties.width,
                height=image_properties.height,
                format=image_properties.format,
                quality=image_properties.quality,
                background=image_properties.background,
                num_images=image_properties.num_images,
            )

        normalized_model: str = self.image_model_name.lower()
        normalized_format: str = openai_props.format.lower()
        if openai_props.width is None and openai_props.height is None:
            size_param = "auto"
        else:
            assert (
                openai_props.width is not None and openai_props.height is not None
            ), "OpenAI image dimensions must be provided together."
            size_param = f"{openai_props.width}x{openai_props.height}"

        request_payload: dict[str, Any] = {
            "model": self.image_model_name,
            "prompt": image_prompt,
            "size": size_param,
            "quality": openai_props.quality,
            "user": self.user,
            "n": openai_props.num_images,
        }

        # Reference: https://platform.openai.com/docs/guides/images for parameter details.
        if "gpt-image" in normalized_model:
            request_payload["output_format"] = normalized_format
        else:
            request_payload["response_format"] = "b64_json"

        max_retries: int = len(self.backoff_delays)
        for attempt_index in range(1, max_retries + 1):
            try:
                response = self.client.images.generate(**request_payload)
                if not response.data:
                    raise ValueError(
                        "OpenAI image generation returned no data entries."
                    )

                image_bytes_results: list[bytes] = []
                for data_item in response.data:
                    encoded_image: str | None = getattr(data_item, "b64_json", None)
                    if encoded_image is None:
                        encoded_image = getattr(data_item, "image_base64", None)
                    if encoded_image is None and hasattr(data_item, "content"):
                        content_list: Any = getattr(data_item, "content")
                        for content_entry in content_list or []:
                            entry_b64: str | None = getattr(
                                content_entry, "b64_json", None
                            )
                            if entry_b64 is None:
                                entry_b64 = getattr(content_entry, "image_base64", None)
                            if entry_b64 is not None:
                                encoded_image = entry_b64
                                break
                    if encoded_image is None:
                        raise ValueError(
                            "OpenAI image generation response did not include base64 content."
                        )

                    image_bytes: bytes = base64.b64decode(encoded_image, validate=True)
                    image_bytes_results.append(image_bytes)

                if not image_bytes_results:
                    raise ValueError(
                        "OpenAI image generation did not produce any image bytes."
                    )

                _LOGGER.info(
                    "openai_image_generated",
                    extra={
                        "model": self.image_model_name,
                        "width": openai_props.width,
                        "height": openai_props.height,
                        "format": normalized_format,
                        "attempt": attempt_index,
                        "count": len(image_bytes_results),
                    },
                )
                return image_bytes_results
            except ValueError as exc:
                _LOGGER.error(
                    "openai_image_generation_invalid_response",
                    extra={
                        "model": self.image_model_name,
                        "size": size_param,
                        "format": normalized_format,
                        "error_type": exc.__class__.__name__,
                    },
                )
                raise
            except (OpenAIError, binascii.Error) as exc:
                wait_seconds: int = self.backoff_delays[attempt_index - 1]
                if attempt_index == max_retries:
                    _LOGGER.error(
                        "openai_image_generation_failed",
                        extra={
                            "model": self.image_model_name,
                            "size": size_param,
                            "format": normalized_format,
                            "attempts": attempt_index,
                            "error_type": exc.__class__.__name__,
                        },
                    )
                    raise RuntimeError(
                        "OpenAI image generation failed after multiple retries."
                    ) from exc
                _LOGGER.warning(
                    "openai_image_generation_retry",
                    extra={
                        "model": self.image_model_name,
                        "size": size_param,
                        "format": normalized_format,
                        "attempt": attempt_index,
                        "retry_in_seconds": wait_seconds,
                        "error_type": exc.__class__.__name__,
                    },
                )
                time.sleep(wait_seconds)
