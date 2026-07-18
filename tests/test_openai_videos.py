# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

pytest.importorskip("openai")

from ai_api_unified.ai_base import AIMediaReference, AIBaseVideoProperties
from ai_api_unified.videos.ai_openai_videos import (
    AIOpenAIVideoProperties,
    AIOpenAIVideos,
)


def _fake_video(**overrides: Any) -> SimpleNamespace:
    """Build a stand-in for the openai SDK Video object (model_dump-compatible)."""
    payload: dict[str, Any] = {
        "id": "video_123",
        "model": "sora-2",
        "status": "queued",
        "progress": 0,
        "created_at": 1_712_697_600,
        "completed_at": None,
        "error": None,
        "expires_at": None,
    }
    payload.update(overrides)
    return SimpleNamespace(model_dump=lambda: payload)


class _InspectableOpenAIVideos(AIOpenAIVideos):
    """OpenAI video test double with a mocked SDK client, no network or auth."""

    def __init__(self) -> None:
        self.video_model_name: str = "sora-2"
        self.base_url: str = "https://api.openai.com/v1"
        self.user: str = "test-user"
        self.client = Mock()

    def _execute_provider_call_with_observability(
        self,
        *,
        callable_execute: Any,
        **_: Any,
    ) -> Any:
        return callable_execute()

    def _resolve_observability_provider_engine(self) -> str:
        return "openai"


def test_openai_video_submit_passes_string_seconds_to_sdk() -> None:
    """The SDK requires seconds as a string enum; the migration fixes the prior 400."""

    provider: _InspectableOpenAIVideos = _InspectableOpenAIVideos()
    provider.client.videos.create.return_value = _fake_video()
    properties: AIOpenAIVideoProperties = AIOpenAIVideoProperties(
        output_dir=Path("/tmp/openai-videos"),
    )

    job = provider.submit_video_generation("A bright city skyline at dusk.", properties)

    create_kwargs = provider.client.videos.create.call_args.kwargs
    assert create_kwargs["prompt"] == "A bright city skyline at dusk."
    assert create_kwargs["model"] == "sora-2"
    assert create_kwargs["seconds"] == "8"
    assert isinstance(create_kwargs["seconds"], str)
    assert create_kwargs["size"] == "1280x720"
    assert "input_reference" not in create_kwargs
    assert job.provider_job_id == "video_123"


def test_openai_video_submit_passes_reference_image_as_input_reference() -> None:
    """Image-guided generation forwards the reference as an SDK input_reference tuple."""

    provider: _InspectableOpenAIVideos = _InspectableOpenAIVideos()
    provider.client.videos.create.return_value = _fake_video()
    properties: AIOpenAIVideoProperties = AIOpenAIVideoProperties(
        reference_image=AIMediaReference(
            bytes_data=b"png-bytes", mime_type="image/png"
        ),
        output_dir=Path("/tmp/openai-videos"),
    )

    provider.submit_video_generation("A bright city skyline at dusk.", properties)

    create_kwargs = provider.client.videos.create.call_args.kwargs
    file_name, file_bytes, mime_type = create_kwargs["input_reference"]
    assert file_name == "input_reference.png"
    assert file_bytes == b"png-bytes"
    assert mime_type == "image/png"


def test_openai_video_submit_rejects_remote_reference_images() -> None:
    """OpenAI input_reference currently requires uploadable local media."""

    provider: _InspectableOpenAIVideos = _InspectableOpenAIVideos()
    provider.client.videos.create.return_value = _fake_video()
    properties: AIOpenAIVideoProperties = AIOpenAIVideoProperties(
        reference_image=AIMediaReference(remote_uri="https://example.com/image.png"),
        output_dir=Path("/tmp/openai-videos"),
    )

    with pytest.raises(
        ValueError,
        match="input_reference must be provided as local bytes or a local file path",
    ):
        provider.submit_video_generation("A bright city skyline at dusk.", properties)


def test_openai_video_get_job_uses_sdk_retrieve() -> None:
    """Polling a job maps the SDK retrieve response into a normalized job."""

    provider: _InspectableOpenAIVideos = _InspectableOpenAIVideos()
    provider.client.videos.retrieve.return_value = _fake_video(
        status="completed", progress=100, completed_at=1_712_697_700
    )

    job = provider.get_video_generation_job("video_123")

    provider.client.videos.retrieve.assert_called_once_with("video_123")
    assert job.status.value == "completed"
    assert job.progress_percent == 100


def test_openai_video_coerce_properties_applies_provider_defaults_for_base_inputs() -> (
    None
):
    """Base video properties should not suppress OpenAI provider defaults."""

    provider: _InspectableOpenAIVideos = _InspectableOpenAIVideos()

    coerced_properties: AIOpenAIVideoProperties = provider._coerce_properties(
        AIBaseVideoProperties()
    )

    assert coerced_properties.duration_seconds == 8
    assert coerced_properties.resolution == "1280x720"
    assert coerced_properties.aspect_ratio == "16:9"
