from __future__ import annotations

from datetime import datetime
from pathlib import Path
import socket
from typing import Any

import pytest

from ai_api_unified.ai_provider_exceptions import (
    AiProviderDependencyUnavailableError,
)
from ai_api_unified.ai_factory import AIFactory
from ai_api_unified.util.env_settings import EnvSettings

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
GOOGLE_GEMINI_HOSTNAME: str = "generativelanguage.googleapis.com"

from ai_api_unified.ai_base import (  # noqa: E402
    AIBaseImages,
    AIBaseImageProperties,
)


def _skip_if_google_image_generation_unavailable(exception: Exception) -> None:
    """Skip live Google image tests when the current account lacks image-generation access."""
    message: str = str(exception)
    if (
        "paid plans" in message.lower()
        or "imagen 3 is only available" in message.lower()
    ):
        pytest.skip(
            "Skipping Google image generation test because the current Google account does not have paid image-generation access."
        )


@pytest.mark.nonmock
def test_generate_image_files(tmp_path: Path) -> None:
    """generate_image_files should persist sequentially numbered files for each image."""
    # This marker identifies tests that call live provider APIs.
    # Run only these with: pytest -m nonmock
    # Exclude these from regular runs with: pytest -m "not nonmock"
    image_files_path: Path = tmp_path
    image_engine_value: object = EnvSettings().get_setting("IMAGE_ENGINE", "")
    image_engine: str = (
        str(image_engine_value).strip().lower()
        if image_engine_value is not None
        else ""
    )
    if image_engine in {"google-gemini", "google"}:
        try:
            socket.getaddrinfo(GOOGLE_GEMINI_HOSTNAME, 443)
        except OSError as exception:
            pytest.skip(
                f"Skipping image generation test because DNS is unavailable for {GOOGLE_GEMINI_HOSTNAME}: {exception}"
            )
    try:
        images_client: AIBaseImages = AIFactory.get_ai_images_client()
    except AiProviderDependencyUnavailableError as exception:
        pytest.skip(
            f"Skipping image generation test due to missing dependency: {exception}"
        )
    common_kwargs: dict[str, Any] = {
        "width": 1_536,
        "height": 1_024,
        "quality": "high",
        "format": "png",
        "background": "auto",
        "num_images": 2,
    }
    image_properties: AIBaseImageProperties = AIBaseImageProperties(**common_kwargs)
    # add datetime to root_file_path to avoid collisions
    date_str: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_file_path: Path = image_files_path / f"test_generate_image_files_{date_str}"

    try:
        saved_paths: list[Path] = images_client.generate_image_files(
            image_prompt="High quality professional photo of an excellent example of a dish that might be served at Domino's, a restaurant that is known for their Pizza or Chicken Wings.. Select one dish and place it in a setting appropriate for the restaurant.",
            image_properties=image_properties,
            root_file_name=str(root_file_path),
        )
    except RuntimeError as exception:
        if image_engine in {"google-gemini", "google"}:
            _skip_if_google_image_generation_unavailable(exception)
        raise

    assert len(saved_paths) == image_properties.num_images


# Add main here
if __name__ == "__main__":
    test_generate_image_files(tmp_path=PROJECT_ROOT / "tests" / "output")
