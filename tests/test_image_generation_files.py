from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any

from ai_api_unified.ai_bedrock_base import EnvSettings
from ai_api_unified.ai_factory import AIFactory

BEDROCK_DEPENDENCIES_AVAILABLE: bool = False
try:
    from ai_api_unified.images.ai_bedrock_images import (
        AINovaCanvasImageProperties,
    )

    BEDROCK_DEPENDENCIES_AVAILABLE = True
except ImportError:  # pragma: no cover
    BEDROCK_DEPENDENCIES_AVAILABLE = False

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

from ai_api_unified.ai_base import (
    AIBaseImages,
    AIBaseImageProperties,
)


def test_generate_image_files(tmp_path: Path) -> None:
    """generate_image_files should persist sequentially numbered files for each image."""
    image_files_path = PROJECT_ROOT / "tests" / "output"
    env: EnvSettings = EnvSettings()
    images_client: AIBaseImages = AIFactory.get_ai_images_client()
    common_kwargs: dict[str, Any] = {
        "width": 1_536,
        "height": 1_024,
        "quality": "high",
        "format": "png",
        "background": "auto",
        "num_images": 2,
    }
    image_model: str = env.get_setting("IMAGE_MODEL_NAME", "").strip()

    if "nova-canvas" in image_model and BEDROCK_DEPENDENCIES_AVAILABLE:
        image_properties = AINovaCanvasImageProperties(
            negative_text="chicken wings",
            **common_kwargs,
        )
    else:
        image_properties = AIBaseImageProperties(**common_kwargs)
    # add datetime to root_file_path to avoid collisions
    date_str: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_file_path: Path = image_files_path / f"test_generate_image_files_{date_str}"

    saved_paths: list[Path] = images_client.generate_image_files(
        image_prompt="High quality professional photo of an excellent example of a dish that might be served at Domino's, a restaurant that is known for their Pizza or Chicken Wings.. Select one dish and place it in a setting appropriate for the restaurant.",
        image_properties=image_properties,
        root_file_name=str(root_file_path),
    )

    assert len(saved_paths) == image_properties.num_images


# Add main here
if __name__ == "__main__":
    test_generate_image_files(tmp_path=PROJECT_ROOT / "tests" / "output")
