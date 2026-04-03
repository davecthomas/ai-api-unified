# ai_bedrock_images.py

from __future__ import annotations

import base64
import binascii
import logging
import secrets
import time
from typing import Any, ClassVar

from pydantic import Field, model_validator

from ..ai_base import (
    AIBaseImageProperties,
    AIBaseImages,
    AiApiObservedImagesResultModel,
)
from ..util.env_settings import EnvSettings

from ..ai_bedrock_base import AIBedrockBase

_LOGGER: logging.Logger = logging.getLogger(__name__)


class AINovaCanvasImageProperties(AIBaseImageProperties):
    """
    Nova Canvas-specific image options supporting arbitrary width/height.

    Example
    -------
    >>> props = AINovaCanvasImageProperties(
    ...     width=2208,
    ...     height=1008,
    ...     num_images=2,
    ...     background="transparent",
    ... )
    """

    MAX_NOVA_RANDOMNESS_SEED: ClassVar[int] = 2_147_483_647
    DEFAULT_WIDTH: ClassVar[int] = 1_536
    DEFAULT_HEIGHT: ClassVar[int] = 1_024
    MIN_DIMENSION: ClassVar[int] = 320
    MAX_DIMENSION: ClassVar[int] = 4_096
    MAX_TOTAL_PIXELS: ClassVar[int] = 16_777_216  # 4096 * 4096
    DIMENSION_STEP: ClassVar[int] = 16
    MIN_ASPECT_RATIO: ClassVar[float] = 0.25  # 4:1 landscape equivalent
    MAX_ASPECT_RATIO: ClassVar[float] = 4.0
    MAX_IMAGES: ClassVar[int] = 4
    width: int = Field(default=DEFAULT_WIDTH)
    height: int = Field(default=DEFAULT_HEIGHT)
    negative_text: str | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_nova_constraints(self) -> "AINovaCanvasImageProperties":
        width = self.width
        height = self.height

        if width % self.DIMENSION_STEP != 0:
            raise ValueError(
                f"width {width} is not a multiple of {self.DIMENSION_STEP}."
            )
        if height % self.DIMENSION_STEP != 0:
            raise ValueError(
                f"height {height} is not a multiple of {self.DIMENSION_STEP}."
            )

        if not self.MIN_DIMENSION <= width <= self.MAX_DIMENSION:
            raise ValueError(
                f"width {width} must be between {self.MIN_DIMENSION} and {self.MAX_DIMENSION} pixels."
            )
        if not self.MIN_DIMENSION <= height <= self.MAX_DIMENSION:
            raise ValueError(
                f"height {height} must be between {self.MIN_DIMENSION} and {self.MAX_DIMENSION} pixels."
            )

        pixel_count = width * height
        if pixel_count > self.MAX_TOTAL_PIXELS:
            raise ValueError(
                f"Total pixel count {pixel_count} exceeds the Nova Canvas limit of {self.MAX_TOTAL_PIXELS}."
            )

        aspect_ratio = width / height
        if not self.MIN_ASPECT_RATIO <= aspect_ratio <= self.MAX_ASPECT_RATIO:
            raise ValueError(
                f"Aspect ratio {aspect_ratio:.2f} is outside the Nova Canvas supported range "
                f"{self.MIN_ASPECT_RATIO} – {self.MAX_ASPECT_RATIO}."
            )

        if self.num_images <= 0 or self.num_images > self.MAX_IMAGES:
            raise ValueError(
                f"num_images must be between 1 and {self.MAX_IMAGES}, got {self.num_images}."
            )

        normalized_background = self.background.lower()
        if normalized_background not in {"auto", "transparent"}:
            raise ValueError(
                f"background must be either 'auto' or 'transparent', got {self.background!r}."
            )
        self.background = normalized_background

        normalized_format = self.format.lower()
        if normalized_format != "png":
            raise ValueError("Nova Canvas currently supports PNG output only.")
        self.format = "png"

        normalized_quality = self.quality.lower()
        if normalized_quality not in {"low", "medium", "high"}:
            raise ValueError(
                f"quality must be 'low', 'medium', or 'high', got {self.quality!r}."
            )
        self.quality = normalized_quality

        return self

    def to_nova_dimension(self) -> tuple[int, int]:
        """
        Return the validated (width, height) tuple for Nova Canvas requests.
        """
        assert self.width is not None and self.height is not None
        return self.width, self.height


class AINovaCanvasImages(AIBedrockBase, AIBaseImages):
    """
    Amazon Bedrock Nova Canvas image generation client.

    Example
    -------
    >>> client = AINovaCanvasImages()
    >>> properties = AINovaCanvasImageProperties(width=1536, height=1024)
    >>> images = client.generate_images("Create a watercolor skyline.", properties)
    >>> len(images)
    1
    """

    DEFAULT_MODEL_ID: ClassVar[str] = "amazon.nova-canvas-v1:0"
    QUALITY_MAPPING: ClassVar[dict[str, str]] = {
        "low": "draft",
        "medium": "standard",
        "high": "premium",
    }

    BACKGROUND_MAPPING: ClassVar[dict[str, str]] = {
        "auto": "DEFAULT",
        "transparent": "TRANSPARENT",
    }

    def __init__(self, model: str | None = None, **kwargs: Any):
        env = EnvSettings()
        resolved_model: str = (
            model
            or env.get_setting("IMAGE_MODEL_NAME", self.DEFAULT_MODEL_ID)
            or self.DEFAULT_MODEL_ID
        )
        resolved_model = resolved_model.strip() or self.DEFAULT_MODEL_ID
        self.image_model_name: str = resolved_model
        AIBedrockBase.__init__(self, model=resolved_model, **kwargs)
        AIBaseImages.__init__(self, model=resolved_model)

    def model_name(self) -> str | None:
        """Return the Bedrock model identifier currently in use."""
        return self.image_model_name

    def list_model_names(self) -> list[str]:
        """Return the list of supported Nova Canvas models."""
        return [self.DEFAULT_MODEL_ID]

    def _coerce_properties(
        self, image_properties: AIBaseImageProperties
    ) -> AINovaCanvasImageProperties:
        if isinstance(image_properties, AINovaCanvasImageProperties):
            return image_properties
        width = (
            image_properties.width
            if image_properties.width is not None
            else AINovaCanvasImageProperties.DEFAULT_WIDTH
        )
        height = (
            image_properties.height
            if image_properties.height is not None
            else AINovaCanvasImageProperties.DEFAULT_HEIGHT
        )
        negative_text = getattr(image_properties, "negative_text", None)

        return AINovaCanvasImageProperties(
            width=width,
            height=height,
            negative_text=negative_text,
            num_images=image_properties.num_images,
            quality=image_properties.quality,
            background=image_properties.background,
        )

    def _map_quality(self, quality: str) -> str:
        mapped = self.QUALITY_MAPPING.get(quality.lower())
        if mapped is None:
            raise ValueError(
                f"Unsupported quality {quality!r} for Nova Canvas. "
                f"Valid values: {list(self.QUALITY_MAPPING)}."
            )
        return mapped

    def _build_payload(
        self, prompt: str, props: AINovaCanvasImageProperties
    ) -> dict[str, Any]:
        width, height = props.to_nova_dimension()
        image_config: dict[str, Any] = {
            "width": width,
            "height": height,
            "numberOfImages": props.num_images,
            "quality": self._map_quality(props.quality),
            "seed": secrets.randbelow(
                AINovaCanvasImageProperties.MAX_NOVA_RANDOMNESS_SEED
            ),
        }

        background_token = self.BACKGROUND_MAPPING.get(props.background)
        if background_token is None:
            raise ValueError(
                f"Background option {props.background!r} is not supported by Nova Canvas."
            )
        if background_token == "TRANSPARENT":
            image_config["background"] = background_token

        payload: dict[str, Any] = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {"text": prompt},
            "imageGenerationConfig": image_config,
        }
        if props.negative_text:
            payload["textToImageParams"]["negativeText"] = props.negative_text
        return payload

    def _decode_images(
        self, response: dict[str, Any], expected_count: int
    ) -> list[bytes]:
        images_field = response.get("images") or response.get("imageArtifacts")
        if not images_field:
            raise RuntimeError(
                "Nova Canvas response did not include any image artifacts."
            )

        results: list[bytes] = []
        for entry in images_field:
            if isinstance(entry, dict):
                encoded = (
                    entry.get("b64Image")
                    or entry.get("base64")
                    or entry.get("image")
                    or entry.get("base64Image")
                )
            else:
                encoded = entry
            if not encoded:
                continue
            try:
                image_bytes = base64.b64decode(encoded, validate=True)
            except binascii.Error as decode_error:
                raise ValueError(
                    "Nova Canvas returned invalid base64 image data."
                ) from decode_error
            results.append(image_bytes)

        if len(results) != expected_count:
            raise RuntimeError(
                f"Nova Canvas returned {len(results)} images, expected {expected_count}."
            )
        return results

    def generate_images(
        self, image_prompt: str, image_properties: AIBaseImageProperties
    ) -> list[bytes]:
        if not image_prompt or not image_prompt.strip():
            raise ValueError("image_prompt must be a non-empty string.")

        props = self._coerce_properties(image_properties)
        payload = self._build_payload(image_prompt, props)
        dict_input_metadata: dict[str, str | int | float | bool | None] = (
            self._build_images_observability_input_metadata(
                image_prompt=image_prompt,
                image_properties=props,
            )
        )
        if props.negative_text:
            dict_input_metadata["negative_prompt_char_count"] = len(props.negative_text)

        def _execute_image_generation() -> AiApiObservedImagesResultModel[list[bytes]]:
            start_time = time.perf_counter()
            response = self._invoke_bedrock_json(
                model_id=self.image_model_name,
                payload=payload,
                trace_name="nova_canvas_generate",
            )
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            request_id: str | None = response.get("requestId")

            if "error" in response:
                raise RuntimeError(
                    f"Nova Canvas generation failed: {response['error']}"
                )

            images = self._decode_images(response, props.num_images)

            _LOGGER.info(
                "bedrock_nova_canvas_generated",
                extra={
                    "model": self.image_model_name,
                    "region": self.region,
                    "width": props.width,
                    "height": props.height,
                    "num_images": props.num_images,
                    "duration_ms": round(duration_ms, 2),
                    "request_id": request_id,
                },
            )
            observed_result: AiApiObservedImagesResultModel[list[bytes]] = (
                AiApiObservedImagesResultModel(
                    return_value=images,
                    generated_image_count=len(images),
                    total_output_bytes=sum(len(image_bytes) for image_bytes in images),
                    dict_metadata={
                        "output_format": props.format.lower(),
                        "request_id": request_id,
                    },
                )
            )
            # Normal return with the caller-facing image bytes payload and metadata-only summary inputs.
            return observed_result

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
            )
        )
        # Normal return with the caller-facing image bytes payload.
        return observed_result.return_value
