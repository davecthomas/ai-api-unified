from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

pytest.importorskip("google.genai")

from ai_api_unified.ai_base import (
    AIVideoGenerationJob,
    AIVideoGenerationStatus,
    AiApiObservedVideosResultModel,
)
from ai_api_unified.videos.ai_google_gemini_videos import AIGoogleGeminiVideos


class _FakeGoogleFilesClient:
    """Small files client used to materialize deterministic video bytes in tests."""

    def download(self, file: object) -> bytes:
        return b"video-bytes"


class _FakeGoogleClient:
    """Small Gemini client shape that exposes the files downloader used by the provider."""

    def __init__(self) -> None:
        self.files: _FakeGoogleFilesClient = _FakeGoogleFilesClient()


class _TestableGoogleGeminiVideos(AIGoogleGeminiVideos):
    """Provider test double that exercises the real download logic without live API calls."""

    DOWNLOAD_OPERATION_READY_TIMEOUT_SECONDS: ClassVar[int] = 1
    DOWNLOAD_OPERATION_READY_POLL_INTERVAL_SECONDS: ClassVar[int] = 0

    def __init__(self, operations: list[SimpleNamespace]) -> None:
        self.client: _FakeGoogleClient = _FakeGoogleClient()
        self.video_model_name: str = "veo-3.1-lite-generate-preview"
        self._operations: list[SimpleNamespace] = list(operations)

    def _get_operation(
        self,
        job: str | AIVideoGenerationJob,
    ) -> SimpleNamespace:
        if not self._operations:
            raise AssertionError("Expected another queued operation in the test.")
        return self._operations.pop(0)

    def _execute_provider_call_with_observability(
        self,
        *,
        callable_execute: Any,
        **_: Any,
    ) -> AiApiObservedVideosResultModel[Any]:
        return callable_execute()

    def _resolve_observability_provider_engine(self) -> str:
        return "google-gemini"


def test_google_video_download_retries_after_completed_job_race(
    tmp_path: Path,
) -> None:
    """Completed Gemini jobs should tolerate one stale operation read before download."""

    operation_name: str = "models/veo-3.1-lite-generate-preview/operations/test-op"
    pending_operation: SimpleNamespace = SimpleNamespace(
        done=False,
        error=None,
        name=operation_name,
        response=None,
    )
    completed_operation: SimpleNamespace = SimpleNamespace(
        done=True,
        error=None,
        name=operation_name,
        response=SimpleNamespace(
            generated_videos=[
                SimpleNamespace(
                    video=SimpleNamespace(
                        uri="gs://test/video.mp4",
                        mime_type="video/mp4",
                    )
                )
            ]
        ),
    )
    provider: _TestableGoogleGeminiVideos = _TestableGoogleGeminiVideos(
        [pending_operation, completed_operation]
    )
    completed_job: AIVideoGenerationJob = AIVideoGenerationJob(
        job_id=f"google-gemini:{operation_name}",
        provider_job_id=operation_name,
        status=AIVideoGenerationStatus.COMPLETED,
        provider_engine="google-gemini",
        provider_model_name="veo-3.1-lite-generate-preview",
        provider_metadata={
            "download_outputs": True,
            "resolved_output_dir": str(tmp_path),
            "resolved_duration_seconds": 8,
            "resolved_resolution": "720p",
            "resolved_aspect_ratio": "16:9",
        },
    )

    result = provider.download_video_result(completed_job)

    assert result.job.status == AIVideoGenerationStatus.COMPLETED
    assert len(result.artifacts) == 1
    assert result.artifacts[0].file_path is not None
    assert result.artifacts[0].file_path.exists()
    assert result.artifacts[0].read_bytes() == b"video-bytes"
