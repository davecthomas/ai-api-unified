import logging
from typing import ClassVar, Any, TYPE_CHECKING
from pydantic import model_validator, Field

from ai_api_unified.ai_google_base import AIGoogleBase
from ai_api_unified.ai_base import AIBaseImageProperties, AIBaseImages

if TYPE_CHECKING:
    from google import genai  # type: ignore
    from google.genai.types import GenerateImagesResponse, DefaultCredentialsError  # type: ignore
    from google.api_core import exceptions as gexc  # type: ignore
    from google.genai import errors as gerr  # type: ignore

GOOGLE_DEPENDENCIES_AVAILABLE: bool = False
try:
    from google import genai  # type: ignore
    from google.genai.types import GenerateImagesResponse  # type: ignore
    from google.api_core import exceptions as gexc  # type: ignore
    from google.genai import errors as gerr  # type: ignore
    from google.auth.exceptions import DefaultCredentialsError  # type: ignore

    GOOGLE_DEPENDENCIES_AVAILABLE = True
except ImportError:
    GOOGLE_DEPENDENCIES_AVAILABLE = False


_LOGGER: logging.Logger = logging.getLogger(__name__)


class AIGoogleGeminiImageProperties(AIBaseImageProperties):
    """
    Google Gemini-specific image properties constrained to supported dimensions/aspect ratios.

    Width and height can be specified, but Gemini strictly enforces specific aspect ratios like:
    "1:1", "3:4", "4:3", "9:16", "16:9".
    """

    aspect_ratio: str | None = Field(default=None)
    person_generation: str | None = Field(
        default="allow_adult",
        description="Must be 'dont_allow', 'allow_adult', or 'allow_all'",
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
        # Determine aspect ratio based on width/height if aspect_ratio isn't explicitly requested
        if self.aspect_ratio is None:
            if self.width and self.height:
                if self.width == self.height:
                    self.aspect_ratio = "1:1"
                elif self.width > self.height:
                    # check if 16:9 or 4:3
                    if abs((self.width / self.height) - (16 / 9)) < 0.1:
                        self.aspect_ratio = "16:9"
                    else:
                        self.aspect_ratio = "4:3"
                else:  # height > width
                    if abs((self.height / self.width) - (16 / 9)) < 0.1:
                        self.aspect_ratio = "9:16"
                    else:
                        self.aspect_ratio = "3:4"
            else:
                self.aspect_ratio = "1:1"  # Default

        if self.aspect_ratio not in self._ALLOWED_ASPECT_RATIOS:
            raise ValueError(
                f"Google Gemini aspect ratio must be one of {self._ALLOWED_ASPECT_RATIOS}. "
                f"Derived/provided: {self.aspect_ratio}"
            )

        if self.person_generation not in self._ALLOWED_PERSON_GENERATION:
             raise ValueError(
                f"Google Gemini person generation must be one of {self._ALLOWED_PERSON_GENERATION}."
             )
             
        # Gemini does NOT support WebP or JPEG transparently from the Python SDK's default models.
        if self.format.lower() != "png":
            _LOGGER.warning(
                "Gemini natively exports uncompressed PNG/JPEG via the API. "
                "Defaulting to downstream SDK default format handling for %s output.",
                self.format,
            )

        return self


class AIGoogleGeminiImages(AIGoogleBase, AIBaseImages):
    def __init__(self, model: str | None = None, **kwargs: Any):
        super().__init__(model=model, **kwargs)
        
        if not GOOGLE_DEPENDENCIES_AVAILABLE:
            raise RuntimeError(
                "Google GenAI dependencies are missing. "
                "Please install them via 'poetry install --extras google_gemini'."
            )
            
        image_model = model
        if image_model is None:
            from ai_api_unified.util.env_settings import EnvSettings
            image_model = EnvSettings().get_setting("IMAGE_MODEL_NAME", "imagen-3.0-generate-001")
        if image_model.strip() == "":
            raise ValueError(
                "IMAGE_MODEL_NAME environment variable must be set to a valid Gemini image model name."
            )
        self.image_model_name: str = image_model.strip()

        # Initialize the generalized client from AIGoogleBase
        self.client: genai.Client = self.get_client(model=self.image_model_name)

    @property
    def model_name(self) -> str:
        return self.image_model_name

    @property
    def list_model_names(self) -> list[str]:
        return ["imagen-3.0-generate-001", "imagen-3.0-fast-generate-001"]

    def generate_images(
        self, image_prompt: str, image_properties: AIBaseImageProperties
    ) -> list[bytes]:
        """Generate images honoring requested parameters, returning raw bytes."""

        if image_prompt.strip() == "":
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

        output_mime_type: str = "image/png"
        if gemini_props.format.lower() == "jpeg":
             output_mime_type = "image/jpeg"

        kwargs: dict[str, Any] = {
            "model": self.image_model_name,
            "prompt": image_prompt,
            "config": {
                "number_of_images": gemini_props.num_images,
                "aspect_ratio": gemini_props.aspect_ratio,
                "output_mime_type": output_mime_type,
                "person_generation": gemini_props.person_generation,
            },
        }

        # Uses generalized exponential backoff from AIGoogleBase
        def _execute_api_call() -> GenerateImagesResponse:
            try:
                response = self.client.models.generate_images(**kwargs)
                return response
            except (
                gerr.APIError,
                gexc.GoogleAPICallError,
                DefaultCredentialsError,
            ) as api_err:
                 # Throw to _retry_with_exponential_backoff logic
                 raise api_err
                 
        try:
             response = self._retry_with_exponential_backoff(_execute_api_call)
        except Exception as exc:
             _LOGGER.error(
                "google_gemini_image_generation_failed",
                extra={
                    "model": self.image_model_name,
                    "aspect_ratio": gemini_props.aspect_ratio,
                    "error_type": exc.__class__.__name__,
                },
             )
             raise RuntimeError("Google Gemini image generation failed.") from exc

        if not getattr(response, "generated_images", None):
            raise ValueError("Google Gemini image generation returned no images.")

        image_bytes_results: list[bytes] = []
        for generated_image in response.generated_images:
             if generated_image.image.image_bytes:
                 image_bytes_results.append(generated_image.image.image_bytes)
             else:
                 _LOGGER.warning("Gemini generated an image but image_bytes was empty.")
                 
        if not image_bytes_results:
             raise ValueError("Google Gemini image generation successfully executed but no byte structures were found.")

        _LOGGER.info(
            "google_gemini_image_generated",
            extra={
                "model": self.image_model_name,
                "aspect_ratio": gemini_props.aspect_ratio,
                "count": len(image_bytes_results),
            },
        )
        return image_bytes_results
