from __future__ import annotations

import logging
from typing import Any, ClassVar

from google import genai
from google.api_core import exceptions as gexc
from google.auth.exceptions import DefaultCredentialsError
from google.genai import errors as gerr
from google.genai import types
from pydantic import Field, model_validator

from ai_api_unified.ai_base import (
    AIBaseImageProperties,
    AIBaseImages,
    AiApiObservedImagesResultModel,
)
from ai_api_unified.ai_google_base import AIGoogleBase
from ai_api_unified.pricing.pricing_registry import (
    PROVIDER_GOOGLE,
    enforce_model_lifecycle,
)
from ai_api_unified.util.env_settings import EnvSettings

_LOGGER: logging.Logger = logging.getLogger(__name__)


class AIGoogleGeminiImageProperties(AIBaseImageProperties):
    """
    Gemini image request properties constrained to supported aspect ratios.
    """

    aspect_ratio: str | None = Field(default=None)
    person_generation: str | None = Field(
        default="allow_adult",
        description="Must be 'dont_allow', 'allow_adult', or 'allow_all'. "
        "Sent only when the client runs in Vertex AI mode; the Gemini "
        "Developer API rejects the parameter.",
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
                "Gemini images natively return PNG/JPEG; requested format %s is ignored.",
                self.format,
            )
        return self


class AIGoogleGeminiImages(AIGoogleBase, AIBaseImages):
    """
    Google Gemini image generation provider wired into the lazy-loader registry.

    Generates with the native Gemini image models (Nano Banana) through
    generate_content with an IMAGE response modality. The Imagen 4 models
    this engine previously called through generate_images were withdrawn
    from the Gemini API on 2026-08-17.
    """

    DEFAULT_IMAGE_MODEL: ClassVar[str] = "gemini-3.1-flash-image"
    # Verified against the live models.list catalogue on 2026-09-25.
    SUPPORTED_IMAGE_MODELS: ClassVar[list[str]] = [
        "gemini-3.1-flash-image",  # Nano Banana 2
        "gemini-3.1-flash-lite-image",  # Nano Banana 2 Lite
        "gemini-3-pro-image",  # Nano Banana Pro
        "gemini-2.5-flash-image",  # shuts down 2026-10-02
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
        enforce_model_lifecycle(PROVIDER_GOOGLE, self.image_model_name)
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

        dict_input_metadata: dict[str, str | int | float | bool | None] = (
            self._build_images_observability_input_metadata(
                image_prompt=image_prompt,
                image_properties=gemini_props,
            )
        )
        dict_input_metadata["aspect_ratio"] = gemini_props.aspect_ratio
        dict_input_metadata["person_generation"] = gemini_props.person_generation

        dict_image_config: dict[str, Any] = {"aspect_ratio": gemini_props.aspect_ratio}
        # The Gemini Developer API rejects person_generation; only Vertex AI
        # accepts it.
        if getattr(self.client, "vertexai", False) and gemini_props.person_generation:
            dict_image_config["person_generation"] = (
                gemini_props.person_generation.upper()
            )
        generate_config: types.GenerateContentConfig = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(**dict_image_config),
        )

        def _generate_one_image() -> tuple[bytes, str, int | None, int | None]:
            """Runs one generate_content call and returns its first image."""
            try:
                response: types.GenerateContentResponse = (
                    self._retry_with_exponential_backoff(
                        lambda: self.client.models.generate_content(
                            model=self.image_model_name,
                            contents=image_prompt,
                            config=generate_config,
                        )
                    )
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

            usage_metadata: Any = getattr(response, "usage_metadata", None)
            input_token_count: int | None = getattr(
                usage_metadata, "prompt_token_count", None
            )
            total_token_count: int | None = getattr(
                usage_metadata, "total_token_count", None
            )
            # Loop through candidates and parts until the first inline image.
            for candidate in getattr(response, "candidates", None) or []:
                content: Any = getattr(candidate, "content", None)
                for part in getattr(content, "parts", None) or []:
                    inline_data: Any = getattr(part, "inline_data", None)
                    image_bytes: bytes | None = getattr(inline_data, "data", None)
                    if image_bytes:
                        str_mime_type: str = (
                            getattr(inline_data, "mime_type", None) or "image/png"
                        )
                        # Normal return with the first image in the response.
                        return (
                            image_bytes,
                            str_mime_type,
                            input_token_count,
                            total_token_count,
                        )
            raise ValueError("Google Gemini image generation returned no images.")

        def _execute_image_generation() -> AiApiObservedImagesResultModel[list[bytes]]:
            image_bytes_results: list[bytes] = []
            set_mime_types: set[str] = set()
            list_input_tokens: list[int] = []
            list_total_tokens: list[int] = []
            # Gemini image models return one image per call, so num_images
            # issues one request per requested image.
            for _ in range(gemini_props.num_images):
                image_bytes, str_mime_type, input_tokens, total_tokens = (
                    _generate_one_image()
                )
                image_bytes_results.append(image_bytes)
                set_mime_types.add(str_mime_type)
                if input_tokens is not None:
                    list_input_tokens.append(input_tokens)
                if total_tokens is not None:
                    list_total_tokens.append(total_tokens)

            return AiApiObservedImagesResultModel(
                return_value=image_bytes_results,
                generated_image_count=len(image_bytes_results),
                total_output_bytes=sum(
                    len(image_bytes) for image_bytes in image_bytes_results
                ),
                provider_input_tokens=(
                    sum(list_input_tokens) if list_input_tokens else None
                ),
                provider_total_tokens=(
                    sum(list_total_tokens) if list_total_tokens else None
                ),
                dict_metadata={
                    "output_mime_type": ",".join(sorted(set_mime_types)),
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
