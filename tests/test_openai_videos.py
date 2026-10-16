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


@pytest.mark.parametrize(
    "str_job_id",
    ["../../escaped", "/etc/cron.d/job", "..", "sub/../../escaped"],
)
def test_openai_video_download_keeps_hostile_job_id_inside_output_dir(
    tmp_path: Path, str_job_id: str
) -> None:
    """A job id from the provider response never writes outside the output dir."""

    output_dir: Path = tmp_path / "videos"
    provider: _InspectableOpenAIVideos = _InspectableOpenAIVideos()
    provider._resolve_video_output_dir = lambda _properties: output_dir
    provider.client.videos.retrieve.return_value = _fake_video(
        id=str_job_id, status="completed", progress=100
    )
    provider.client.videos.download_content.return_value = SimpleNamespace(
        read=lambda: b"mp4-bytes"
    )

    try:
        result = provider.download_video_result(str_job_id)
    except ValueError as exception:
        assert "Refusing" in str(exception)
    else:
        file_path: Path = result.artifacts[0].file_path
        assert file_path.parent == output_dir
        assert file_path.read_bytes() == b"mp4-bytes"
    for written in tmp_path.rglob("*"):
        if written.is_file():
            assert written.parent == output_dir


@pytest.mark.parametrize(
    "str_job_id, str_suffix, str_expected_name",
    [
        ("video_123", "", "video_123.mp4"),
        ("models/veo/operations/abc", "_0", "models_veo_operations_abc_0.mp4"),
        ("../../x", "", ".._.._x.mp4"),
        ("/etc/passwd", "", "_etc_passwd.mp4"),
    ],
)
def test_video_artifact_path_maps_job_id_to_a_file_inside_output_dir(
    tmp_path: Path, str_job_id: str, str_suffix: str, str_expected_name: str
) -> None:
    provider: _InspectableOpenAIVideos = _InspectableOpenAIVideos()

    file_path: Path = provider._video_artifact_path(tmp_path, str_job_id, str_suffix)

    assert file_path == tmp_path / str_expected_name


@pytest.mark.parametrize("str_job_id", ["", ".", "..", "..."])
def test_video_artifact_path_refuses_dots_only_job_id(
    tmp_path: Path, str_job_id: str
) -> None:
    provider: _InspectableOpenAIVideos = _InspectableOpenAIVideos()

    with pytest.raises(ValueError, match="Refusing"):
        provider._video_artifact_path(tmp_path, str_job_id)
