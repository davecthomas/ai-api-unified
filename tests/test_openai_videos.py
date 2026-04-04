from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

pytest.importorskip("openai")

from ai_api_unified.ai_base import AIMediaReference
from ai_api_unified.videos.ai_openai_videos import (
    AIOpenAIVideoProperties,
    AIOpenAIVideos,
)


class _InspectableOpenAIVideos(AIOpenAIVideos):
    """OpenAI video test double that captures outbound request payloads."""

    def __init__(self) -> None:
        self.video_model_name: str = "sora-2"
        self.base_url: str = "https://api.openai.com/v1"
        self.user: str = "test-user"
        self.captured_request: dict[str, Any] | None = None

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        data_payload: dict[str, str] | None = None,
        files_payload: dict[str, tuple[str, bytes, str]] | None = None,
        accept: str = "application/json",
    ) -> httpx.Response:
        self.captured_request = {
            "method": method,
            "path": path,
            "json_payload": json_payload,
            "data_payload": data_payload,
            "files_payload": files_payload,
            "accept": accept,
        }
        return httpx.Response(
            200,
            json={
                "id": "video_123",
                "model": self.video_model_name,
                "status": "queued",
                "progress": 0,
                "created_at": 1_712_697_600,
            },
            request=httpx.Request(method, path),
        )

    def _execute_provider_call_with_observability(
        self,
        *,
        callable_execute: Any,
        **_: Any,
    ) -> Any:
        return callable_execute()

    def _resolve_observability_provider_engine(self) -> str:
        return "openai"


def test_openai_video_submit_uses_multipart_upload_for_reference_images() -> None:
    """Image-guided OpenAI video generation should upload input_reference as a file."""

    provider: _InspectableOpenAIVideos = _InspectableOpenAIVideos()
    properties: AIOpenAIVideoProperties = AIOpenAIVideoProperties(
        reference_image=AIMediaReference(
            bytes_data=b"png-bytes", mime_type="image/png"
        ),
        output_dir=Path("/tmp/openai-videos"),
    )

    provider.submit_video_generation("A bright city skyline at dusk.", properties)

    assert provider.captured_request is not None
    assert provider.captured_request["json_payload"] is None
    assert provider.captured_request["data_payload"] == {
        "prompt": "A bright city skyline at dusk.",
        "model": "sora-2",
        "seconds": "8",
        "size": "1280x720",
    }
    files_payload: dict[str, tuple[str, bytes, str]] = provider.captured_request[
        "files_payload"
    ]
    assert "input_reference" in files_payload
    file_name, file_bytes, mime_type = files_payload["input_reference"]
    assert file_name == "input_reference.png"
    assert file_bytes == b"png-bytes"
    assert mime_type == "image/png"


def test_openai_video_submit_rejects_remote_reference_images() -> None:
    """OpenAI input_reference currently requires uploadable local media."""

    provider: _InspectableOpenAIVideos = _InspectableOpenAIVideos()
    properties: AIOpenAIVideoProperties = AIOpenAIVideoProperties(
        reference_image=AIMediaReference(remote_uri="https://example.com/image.png"),
        output_dir=Path("/tmp/openai-videos"),
    )

    with pytest.raises(
        ValueError,
        match="input_reference must be provided as local bytes or a local file path",
    ):
        provider.submit_video_generation("A bright city skyline at dusk.", properties)
