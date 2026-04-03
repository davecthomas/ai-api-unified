from datetime import datetime
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.test_image_generation_files import PROJECT_ROOT
from ai_api_unified.ai_base import AIBaseImages
from ai_api_unified.ai_factory import AIFactory

module_bedrock_images: ModuleType = pytest.importorskip(
    "ai_api_unified.images.ai_bedrock_images",
    reason="Bedrock image extras are not installed.",
)
AINovaCanvasImageProperties: type[Any] = (
    module_bedrock_images.AINovaCanvasImageProperties
)


@pytest.mark.parametrize(
    "width,height",
    [(1536, 1024), (2048, 1024)],
)
def test_nova_canvas_image_properties_valid_dimensions(width: int, height: int) -> None:
    props: Any = AINovaCanvasImageProperties(width=width, height=height, num_images=2)
    assert props.to_nova_dimension() == (width, height)
    assert props.width == width
    assert props.height == height
    assert props.quality == "medium"


def test_nova_canvas_image_properties_invalid_multiple() -> None:
    with pytest.raises(ValueError, match="multiple of 16"):
        AINovaCanvasImageProperties(width=1500, height=1024)


def test_nova_canvas_image_properties_num_images_limit() -> None:
    with pytest.raises(ValueError, match="num_images must be between"):
        AINovaCanvasImageProperties(width=1536, height=1024, num_images=10)


@pytest.mark.nonmock
def test_nova_canvas_generate_images() -> None:
    """Integration-lite test that generates a single Nova Canvas image when AWS credentials are present."""

    required_env_vars: list[str] = ["AWS_REGION"]
    if not all(os.getenv(var) for var in required_env_vars):
        pytest.skip(
            "Bedrock credentials not configured; set AWS_REGION to run this test."
        )

    image_client: AIBaseImages = AIFactory.get_ai_images_client(
        image_model="amazon.nova-canvas-v1:0"
    )
    image_files_path: Path = PROJECT_ROOT / "tests" / "output"
    # add datetime to root_file_path to avoid collisions
    date_str: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_file_path: Path = image_files_path / f"test_generate_image_files_{date_str}"

    prompt: str = "Generate a trippy abstract pattern with soft colors."
    props: Any = AINovaCanvasImageProperties(width=704, height=320, num_images=1)

    images: list[Path] = image_client.generate_image_files(
        image_prompt=prompt,
        image_properties=props,
        root_file_name=str(root_file_path),
    )

    assert len(images) == 1
    assert isinstance(images[0], Path)
