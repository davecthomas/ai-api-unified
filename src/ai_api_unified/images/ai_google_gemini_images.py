from __future__ import annotations

import logging
from typing import Any, ClassVar

from google import genai
from google.api_core import exceptions as gexc
from google.auth.exceptions import DefaultCredentialsError
from google.genai import errors as gerr
from google.genai.types import GenerateImagesResponse
from pydantic import Field, model_validator

from ai_api_unified.ai_base import (
    AIBaseImageProperties,
    AIBaseImages,
    AiApiObservedImagesResultModel,
)
from ai_api_unified.ai_google_base import AIGoogleBase
from ai_api_unified.util.env_settings import EnvSettings

_LOGGER: logging.Logger = logging.getLogger(__name__)


class AIGoogleGeminiImageProperties(AIBaseImageProperties):
    """
    Gemini image request properties constrained to supported aspect ratios.
    """

    aspect_ratio: str | None = Field(default=None)
    person_generation: str | None = Field(
        default="allow_adult",
        description="Must be 'dont_allow', 'allow_adult', or 'allow_all'.",
    )

    _ALLOWED_ASPECT_RATIOS: ClassVar[set[str]] = {
        "1:1",
        "3:4",
        "4:3",
        "9:16",
        "16:9",
    }
    _ALLOWED_PERSON_GENERATION: ClassVar[set[str]] = {
        "dont_allow",
        "allow_adult",
        "allow_all",
    }

    @model_validator(mode="after")
    def _validate_gemini_dimensions(self) -> "AIGoogleGeminiImageProperties":
        if self.aspect_ratio is None:
            if self.width and self.height:
                if self.width == self.height:
                    self.aspect_ratio = "1:1"
                elif self.width > self.height:
                    self.aspect_ratio = (
                        "16:9"
                        if abs((self.width / self.height) - (16 / 9)) < 0.1
                        else "4:3"
                    )
                else:
                    self.aspect_ratio = (
                        "9:16"
                        if abs((self.height / self.width) - (16 / 9)) < 0.1
                        else "3:4"
                    )
            else:
                self.aspect_ratio = "1:1"

        if self.aspect_ratio not in self._ALLOWED_ASPECT_RATIOS:
            raise ValueError(
                f"Google Gemini aspect ratio must be one of {self._ALLOWED_ASPECT_RATIOS}. "
                f"Derived/provided: {self.aspect_ratio}"
            )
        if self.person_generation not in self._ALLOWED_PERSON_GENERATION:
            raise ValueError(
                "Google Gemini person generation must be one of "
                f"{self._ALLOWED_PERSON_GENERATION}."
            )
        if self.format.lower() not in {"png", "jpeg"}:
            _LOGGER.warning(
                "Gemini images natively return PNG/JPEG; requested format %s will fall back to PNG.",
                self.format,
            )
        return self


class AIGoogleGeminiImages(AIGoogleBase, AIBaseImages):
    """
    Google Gemini image generation provider wired into the lazy-loader registry.
    """

    DEFAULT_IMAGE_MODEL: ClassVar[str] = "imagen-4.0-generate-001"
    SUPPORTED_IMAGE_MODELS: ClassVar[list[str]] = [
        "gemini-2.5-flash-image",
        "gemini-3-pro-image-preview",
        "gemini-3.1-flash-image-preview",
        "imagen-4.0-generate-001",
        "imagen-4.0-ultra-generate-001",
        "imagen-4.0-fast-generate-001",
    ]

    def __init__(self, model: str | None = None, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)

        env_settings: EnvSettings = EnvSettings()
        image_model: str | None = model
        if image_model is None:
            image_model = env_settings.get_setting(
                "IMAGE_MODEL_NAME",
                self.DEFAULT_IMAGE_MODEL,
            )
        if image_model is None or not image_model.strip():
            raise ValueError(
                "IMAGE_MODEL_NAME environment variable must be set to a valid Gemini image model name."
            )

        self.image_model_name: str = image_model.strip()
        self.client: genai.Client = self.get_client(model=self.image_model_name)

    def model_name(self) -> str:
        return self.image_model_name

    def list_model_names(self) -> list[str]:
        return list(self.SUPPORTED_IMAGE_MODELS)

    def generate_images(
        self, image_prompt: str, image_properties: AIBaseImageProperties
    ) -> list[bytes]:
        if not image_prompt.strip():
            raise ValueError("image_prompt must be a non-empty string.")

        gemini_props: AIGoogleGeminiImageProperties
        if isinstance(image_properties, AIGoogleGeminiImageProperties):
            gemini_props = image_properties
        else:
            gemini_props = AIGoogleGeminiImageProperties(
                width=image_properties.width,
                height=image_properties.height,
                format=image_properties.format,
                quality=image_properties.quality,
                background=image_properties.background,
                num_images=image_properties.num_images,
            )

        output_format: str = gemini_props.format.lower()
        output_mime_type: str = "image/jpeg" if output_format == "jpeg" else "image/png"
        dict_input_metadata: dict[str, str | int | float | bool | None] = (
            self._build_images_observability_input_metadata(
                image_prompt=image_prompt,
                image_properties=gemini_props,
            )
        )
        dict_input_metadata["aspect_ratio"] = gemini_props.aspect_ratio
        dict_input_metadata["person_generation"] = gemini_props.person_generation

        request_kwargs: dict[str, Any] = {
            "model": self.image_model_name,
            "prompt": image_prompt,
            "config": {
                "number_of_images": gemini_props.num_images,
                "aspect_ratio": gemini_props.aspect_ratio,
                "output_mime_type": output_mime_type,
                "person_generation": gemini_props.person_generation,
            },
        }

        def _execute_image_generation() -> AiApiObservedImagesResultModel[list[bytes]]:
            try:
                response: GenerateImagesResponse = self._retry_with_exponential_backoff(
                    lambda: self.client.models.generate_images(**request_kwargs)
                )
            except (
                gerr.APIError,
                gexc.GoogleAPICallError,
                DefaultCredentialsError,
            ) as exception:
                _LOGGER.error(
                    "google_gemini_image_generation_failed",
                    extra={
                        "model": self.image_model_name,
                        "aspect_ratio": gemini_props.aspect_ratio,
                        "error_type": exception.__class__.__name__,
                    },
                )
                raise RuntimeError(
                    "Google Gemini image generation failed."
                ) from exception

            generated_images: Any = getattr(response, "generated_images", None)
            if not generated_images:
                raise ValueError("Google Gemini image generation returned no images.")

            image_bytes_results: list[bytes] = []
            for generated_image in generated_images:
                image_payload: Any = getattr(generated_image, "image", None)
                image_bytes: bytes | None = getattr(image_payload, "image_bytes", None)
                if image_bytes:
                    image_bytes_results.append(image_bytes)
                else:
                    _LOGGER.warning(
                        "Gemini generated an image entry without image_bytes content."
                    )

            if not image_bytes_results:
                raise ValueError(
                    "Google Gemini image generation succeeded but returned no image byte payloads."
                )

            return AiApiObservedImagesResultModel(
                return_value=image_bytes_results,
                generated_image_count=len(image_bytes_results),
                total_output_bytes=sum(
                    len(image_bytes) for image_bytes in image_bytes_results
                ),
                provider_input_tokens=None,
                provider_total_tokens=None,
                dict_metadata={
                    "output_mime_type": output_mime_type,
                    "aspect_ratio": gemini_props.aspect_ratio,
                    "person_generation": gemini_props.person_generation,
                },
            )

        observed_result: AiApiObservedImagesResultModel[list[bytes]] = (
            self._execute_provider_call_with_observability(
                capability=self.CLIENT_TYPE_IMAGES,
                operation="generate_images",
                dict_input_metadata=dict_input_metadata,
                callable_execute=_execute_image_generation,
                callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_images_observability_result_summary(
                    observed_result=result,
                    provider_elapsed_ms=provider_elapsed_ms,
                ),
                legacy_caller_id=None,
            )
        )
        return observed_result.return_value
