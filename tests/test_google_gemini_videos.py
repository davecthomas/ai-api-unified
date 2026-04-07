from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

pytest.importorskip("google.genai")

from ai_api_unified.ai_base import (
    AIMediaReference,
    AIVideoGenerationJob,
    AIVideoGenerationStatus,
    AiApiObservedVideosResultModel,
)
from ai_api_unified.videos.ai_google_gemini_videos import (
    AIGoogleGeminiVideoProperties,
    AIGoogleGeminiVideos,
)


class _FakeGoogleFilesClient:
    """Small files client used to materialize deterministic video bytes in tests."""

    def download(self, file: object) -> bytes:
        return b"video-bytes"


class _FakeGoogleClient:
    """Small Gemini client shape that exposes the files downloader used by the provider."""

    def __init__(self) -> None:
        self.files: _FakeGoogleFilesClient = _FakeGoogleFilesClient()


class _FakeGoogleModelsClient:
    """Small models client that records generate_videos() calls."""

    def __init__(self, operation: SimpleNamespace) -> None:
        self.operation: SimpleNamespace = operation
        self.calls: list[dict[str, Any]] = []

    def generate_videos(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.operation


class _FakeGoogleClientWithModels(_FakeGoogleClient):
    """Augmented fake client with a models surface for submit_video_generation tests."""

    def __init__(self, operation: SimpleNamespace) -> None:
        super().__init__()
        self.models: _FakeGoogleModelsClient = _FakeGoogleModelsClient(operation)


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


class _InspectableGoogleGeminiVideos(AIGoogleGeminiVideos):
    """Provider test double that captures submit_video_generation request config."""

    def __init__(self, operation: SimpleNamespace) -> None:
        self.client: _FakeGoogleClientWithModels = _FakeGoogleClientWithModels(
            operation
        )
        self.video_model_name: str = "veo-3.1-lite-generate-preview"

    def _retry_with_exponential_backoff(self, operation: Any, **_: Any) -> Any:
        return operation()

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


def test_google_video_submit_forwards_supported_generate_videos_config_fields() -> None:
    """Gemini submit should forward provider-supported config fields without mutating them."""

    operation: SimpleNamespace = SimpleNamespace(
        done=False,
        error=None,
        name="models/veo-3.1-lite-generate-preview/operations/test-op",
        response=None,
    )
    provider: _InspectableGoogleGeminiVideos = _InspectableGoogleGeminiVideos(operation)
    properties: AIGoogleGeminiVideoProperties = AIGoogleGeminiVideoProperties(
        seed=123,
        fps=24,
        generate_audio=True,
        person_generation="allow_adult",
        output_dir=Path("/tmp/google-videos"),
    )

    provider.submit_video_generation("A lighthouse in a storm.", properties)

    captured_call: dict[str, Any] = provider.client.models.calls[0]
    config: Any = captured_call["config"]
    assert config.seed == 123
    assert config.fps == 24
    assert config.generate_audio is True
    assert config.person_generation == "allow_adult"


def test_google_video_submit_rejects_source_video_when_api_key_auth_is_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Video-to-video generation should fail fast for API-key Gemini usage."""

    monkeypatch.setenv("GOOGLE_AUTH_METHOD", "api_key")
    operation: SimpleNamespace = SimpleNamespace(
        done=False,
        error=None,
        name="models/veo-3.1-lite-generate-preview/operations/test-op",
        response=None,
    )
    provider: _InspectableGoogleGeminiVideos = _InspectableGoogleGeminiVideos(operation)
    properties: AIGoogleGeminiVideoProperties = AIGoogleGeminiVideoProperties(
        source_video=AIMediaReference(remote_uri="gs://bucket/video.mp4"),
        output_dir=Path("/tmp/google-videos"),
    )

    with pytest.raises(
        NotImplementedError,
        match="requires Vertex AI",
    ):
        provider.submit_video_generation("Extend this video.", properties)


def test_google_video_submit_rejects_non_gcs_source_video_uris(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Video-to-video Gemini inputs must use GCS URIs."""

    monkeypatch.setenv("GOOGLE_AUTH_METHOD", "service_account")
    operation: SimpleNamespace = SimpleNamespace(
        done=False,
        error=None,
        name="models/veo-3.1-lite-generate-preview/operations/test-op",
        response=None,
    )
    provider: _InspectableGoogleGeminiVideos = _InspectableGoogleGeminiVideos(operation)
    properties: AIGoogleGeminiVideoProperties = AIGoogleGeminiVideoProperties(
        source_video=AIMediaReference(remote_uri="https://example.com/video.mp4"),
        output_dir=Path("/tmp/google-videos"),
    )

    with pytest.raises(
        ValueError,
        match="must use a gs:// URI",
    ):
        provider.submit_video_generation("Extend this video.", properties)


def test_google_video_submit_rejects_non_gcs_reference_image_uris() -> None:
    """Remote Gemini image inputs should fail fast unless they are backed by GCS."""

    operation: SimpleNamespace = SimpleNamespace(
        done=False,
        error=None,
        name="models/veo-3.1-lite-generate-preview/operations/test-op",
        response=None,
    )
    provider: _InspectableGoogleGeminiVideos = _InspectableGoogleGeminiVideos(operation)
    properties: AIGoogleGeminiVideoProperties = AIGoogleGeminiVideoProperties(
        image=AIMediaReference(remote_uri="https://example.com/image.png"),
        output_dir=Path("/tmp/google-videos"),
    )

    with pytest.raises(
        ValueError,
        match="remote_uri must use a gs:// URI",
    ):
        provider.submit_video_generation("Animate this image.", properties)


def test_google_video_job_timestamps_remain_stable_across_polls() -> None:
    """Gemini job timestamps should not be re-stamped on every status refresh."""

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

    running_job: AIVideoGenerationJob = provider.get_video_generation_job(
        operation_name
    )
    completed_job: AIVideoGenerationJob = provider.get_video_generation_job(running_job)

    assert running_job.submitted_at_utc is not None
    assert completed_job.submitted_at_utc == running_job.submitted_at_utc
    assert completed_job.completed_at_utc is not None


def test_google_video_job_timestamps_remain_stable_across_string_lookups() -> None:
    """Gemini job timestamps should remain stable even when callers only reuse the job id."""

    operation_name: str = "models/veo-3.1-lite-generate-preview/operations/test-op"
    first_pending_operation: SimpleNamespace = SimpleNamespace(
        done=False,
        error=None,
        name=operation_name,
        response=None,
    )
    second_pending_operation: SimpleNamespace = SimpleNamespace(
        done=False,
        error=None,
        name=operation_name,
        response=None,
    )
    first_completed_operation: SimpleNamespace = SimpleNamespace(
        done=True,
        error=None,
        name=operation_name,
        response=SimpleNamespace(generated_videos=[]),
    )
    second_completed_operation: SimpleNamespace = SimpleNamespace(
        done=True,
        error=None,
        name=operation_name,
        response=SimpleNamespace(generated_videos=[]),
    )
    provider: _TestableGoogleGeminiVideos = _TestableGoogleGeminiVideos(
        [
            first_pending_operation,
            second_pending_operation,
            first_completed_operation,
            second_completed_operation,
        ]
    )

    first_pending_job: AIVideoGenerationJob = provider.get_video_generation_job(
        operation_name
    )
    second_pending_job: AIVideoGenerationJob = provider.get_video_generation_job(
        operation_name
    )
    first_completed_job: AIVideoGenerationJob = provider.get_video_generation_job(
        operation_name
    )
    second_completed_job: AIVideoGenerationJob = provider.get_video_generation_job(
        operation_name
    )

    assert first_pending_job.submitted_at_utc is not None
    assert second_pending_job.submitted_at_utc == first_pending_job.submitted_at_utc
    assert first_completed_job.submitted_at_utc == first_pending_job.submitted_at_utc
    assert first_completed_job.completed_at_utc is not None
    assert second_completed_job.completed_at_utc == first_completed_job.completed_at_utc
