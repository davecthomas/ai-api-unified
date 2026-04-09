from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

pytest.importorskip("google.genai")
pytest.importorskip("openai")
pytest.importorskip("imageio")
pytest.importorskip("PIL")

from ai_api_unified import AIFactory, AIBaseVideoProperties, AIBaseVideos
from ai_api_unified.ai_base import AIVideoArtifact, AIVideoGenerationResult
from ai_api_unified.videos.ai_google_gemini_videos import AIGoogleGeminiVideoProperties

GOOGLE_GEMINI_HOSTNAME: str = "generativelanguage.googleapis.com"
OPENAI_HOSTNAME: str = "api.openai.com"
TEST_GOOGLE_VIDEO_MODEL: str = "veo-3.1-lite-generate-preview"
TEST_OPENAI_VIDEO_MODEL: str = "sora-2"


def _skip_if_dns_unavailable(hostname: str) -> None:
    """Skip live tests quickly when the current environment cannot resolve provider DNS."""

    try:
        socket.getaddrinfo(hostname, 443)
    except OSError as exception:
        pytest.skip(f"Skipping: DNS unavailable for {hostname}: {exception}")


def _skip_if_google_video_unavailable(exception: Exception) -> None:
    """Skip live Google video tests when the current key lacks access or quota."""

    message: str = str(exception).lower()
    if (
        "resource_exhausted" in message
        or "quota exceeded" in message
        or "paid plan" in message
        or "permission denied" in message
        or "not available" in message
    ):
        pytest.skip(
            "Skipping Google Gemini video test because the current account does not have Veo access or has exhausted quota."
        )


def _skip_if_openai_video_unavailable(exception: Exception) -> None:
    """Skip live OpenAI video tests when the current key lacks Sora access."""

    message: str = str(exception).lower()
    if (
        "rate limit" in message
        or "quota" in message
        or "billing" in message
        or "model" in message
        or "sora" in message
        or "not available" in message
        or "permission" in message
        or "deprecated" in message
    ):
        pytest.skip(
            "Skipping OpenAI video test because the current account does not have Sora access or the API is unavailable."
        )


def _assert_result_and_frame_helpers(
    *,
    result: AIVideoGenerationResult,
    frame_output_dir: Path,
) -> None:
    """Validate the normalized result shape and the shared frame helpers."""

    assert result.artifacts, "Expected at least one generated video artifact."
    artifact: AIVideoArtifact = result.artifacts[0]
    assert artifact.file_path is not None
    assert artifact.file_path.exists()
    assert artifact.file_path.stat().st_size > 0

    video_bytes: bytes = artifact.read_bytes()
    assert len(video_bytes) > 0

    image_buffers: list[bytes] = AIBaseVideos.extract_image_frames_from_video_buffer(
        video_bytes,
        time_offsets_seconds=[0.0],
    )
    assert len(image_buffers) == 1
    assert image_buffers[0]

    saved_frame_paths: list[Path] = AIBaseVideos.save_image_buffers_as_files(
        image_buffers,
        output_dir=frame_output_dir,
        root_file_name="frame",
        image_format="png",
    )
    assert len(saved_frame_paths) == 1
    assert saved_frame_paths[0].exists()
    assert saved_frame_paths[0].stat().st_size > 0


@pytest.mark.nonmock
def test_google_gemini_generate_video_and_extract_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate one Veo video for real, then extract and persist a frame."""

    api_key: str | None = os.environ.get("GOOGLE_GEMINI_API_KEY")
    if not api_key:
        pytest.skip("Skipping: GOOGLE_GEMINI_API_KEY not set")
    _skip_if_dns_unavailable(GOOGLE_GEMINI_HOSTNAME)

    monkeypatch.setenv("GOOGLE_GEMINI_API_KEY", api_key)
    monkeypatch.setenv("GOOGLE_API_KEY", api_key)
    monkeypatch.setenv("GOOGLE_AUTH_METHOD", "api_key")
    monkeypatch.setenv("VIDEO_ENGINE", "google-gemini")
    monkeypatch.setenv("VIDEO_MODEL_NAME", TEST_GOOGLE_VIDEO_MODEL)

    video_properties: AIGoogleGeminiVideoProperties = AIGoogleGeminiVideoProperties(
        duration_seconds=8,
        aspect_ratio="16:9",
        resolution="720p",
        timeout_seconds=1_200,
        poll_interval_seconds=10,
        output_dir=tmp_path / "google_videos",
        generate_audio=False,
    )
    videos_client: AIBaseVideos = AIFactory.get_ai_video_client(
        model_name=TEST_GOOGLE_VIDEO_MODEL,
        video_engine="google-gemini",
    )
    try:
        result: AIVideoGenerationResult = videos_client.generate_video(
            "A cinematic dolly shot of a red vintage train moving through a desert at sunset.",
            video_properties,
        )
    except RuntimeError as exception:
        _skip_if_google_video_unavailable(exception)
        raise

    _assert_result_and_frame_helpers(
        result=result,
        frame_output_dir=tmp_path / "google_frames",
    )


@pytest.mark.nonmock
def test_openai_generate_video_and_extract_frames(
    tmp_path: Path,
) -> None:
    """Generate one OpenAI Sora video for real, then extract and persist a frame."""

    api_key: str | None = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("Skipping: OPENAI_API_KEY not set")
    _skip_if_dns_unavailable(OPENAI_HOSTNAME)

    video_properties: AIBaseVideoProperties = AIBaseVideoProperties(
        duration_seconds=4,
        aspect_ratio="16:9",
        resolution="1280x720",
        timeout_seconds=1_200,
        poll_interval_seconds=10,
        output_dir=tmp_path / "openai_videos",
    )
    videos_client: AIBaseVideos = AIFactory.get_ai_video_client(
        model_name=TEST_OPENAI_VIDEO_MODEL,
        video_engine="openai",
    )
    try:
        result: AIVideoGenerationResult = videos_client.generate_video(
            "A wide cinematic shot of gentle ocean waves meeting a rocky coastline at golden hour.",
            video_properties,
        )
    except RuntimeError as exception:
        _skip_if_openai_video_unavailable(exception)
        raise

    _assert_result_and_frame_helpers(
        result=result,
        frame_output_dir=tmp_path / "openai_frames",
    )
