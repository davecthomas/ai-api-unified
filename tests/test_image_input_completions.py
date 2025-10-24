from __future__ import annotations

import logging
from pathlib import Path

import pytest

from ai_api_unified.ai_base import (
    AIBaseCompletions,
    AICompletionsPromptParamsBase,
    SupportedDataType,
)
from ai_api_unified.ai_factory import AIFactory

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PROMPT_TEXT: str = "Describe this image."
MIME_TYPE_PNG: str = "image/png"
OPENAI_API_KEY_ENV: str = "OPENAI_API_KEY"
EXPECTED_KEYWORD_ONE: str = "image"
EXPECTED_KEYWORD_TWO: str = "photo"
INPUT_IMAGE_PATH: Path = Path("tests/input/test_input_image_file.png")


@pytest.mark.integration
def test_image_input_completions() -> None:
    """
    End-to-end validation that passing inline image bytes through the completions
    interface returns a textual response referencing the image.
    """
    if not INPUT_IMAGE_PATH.exists():
        pytest.skip(f"Input image not found at {INPUT_IMAGE_PATH}")

    client: AIBaseCompletions = AIFactory.get_ai_completions_client(
        completions_engine="openai"
    )

    image_bytes: bytes = INPUT_IMAGE_PATH.read_bytes()
    params: AICompletionsPromptParamsBase = AICompletionsPromptParamsBase(
        included_types=[SupportedDataType.IMAGE],
        included_data=[image_bytes],
        included_mime_types=[MIME_TYPE_PNG],
    )

    response_text: str = client.send_prompt(PROMPT_TEXT, other_params=params)
    logging.info("Completions response: %s", response_text)
    response_text_lower: str = response_text.lower()
    keywords: tuple[str, str] = (EXPECTED_KEYWORD_ONE, EXPECTED_KEYWORD_TWO)

    assert any(
        keyword in response_text_lower for keyword in keywords
    ), "Response did not reference the supplied image."
