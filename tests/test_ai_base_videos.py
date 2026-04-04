from __future__ import annotations

from pathlib import Path

from ai_api_unified.ai_base import (
    AIBaseVideoProperties,
    AIBaseVideos,
    AIVideoArtifact,
    AIVideoGenerationJob,
    AIVideoGenerationResult,
    AIVideoGenerationStatus,
)


class _GenerateVideoDummyProvider(AIBaseVideos):
    """Small AIBaseVideos test double that captures generate_video() wait settings."""

    def __init__(self) -> None:
        super().__init__(model="dummy-video")
        self.wait_timeout_seconds: int | None = -1
        self.wait_poll_interval_seconds: int | None = -1
        self.submitted_job: AIVideoGenerationJob | None = None

    def model_name(self) -> str:
        return "dummy-video"

    def list_model_names(self) -> list[str]:
        return ["dummy-video"]

    def submit_video_generation(
        self,
        video_prompt: str,
        video_properties: AIBaseVideoProperties = AIBaseVideoProperties(),
    ) -> AIVideoGenerationJob:
        self.submitted_job = AIVideoGenerationJob(
            job_id="dummy:job-1",
            provider_job_id="job-1",
            status=AIVideoGenerationStatus.RUNNING,
            provider_engine="dummy",
            provider_model_name="dummy-video",
            provider_metadata={
                "resolved_timeout_seconds": 1_800,
                "resolved_poll_interval_seconds": 25,
            },
        )
        return self.submitted_job

    def wait_for_video_generation(
        self,
        job: str | AIVideoGenerationJob,
        *,
        timeout_seconds: int | None = None,
        poll_interval_seconds: int | None = None,
    ) -> AIVideoGenerationJob:
        self.wait_timeout_seconds = timeout_seconds
        self.wait_poll_interval_seconds = poll_interval_seconds
        assert isinstance(job, AIVideoGenerationJob)
        return job.model_copy(update={"status": AIVideoGenerationStatus.COMPLETED})

    def get_video_generation_job(
        self,
        job: str | AIVideoGenerationJob,
    ) -> AIVideoGenerationJob:
        raise AssertionError("generate_video should call the override above directly.")

    def download_video_result(
        self,
        job: str | AIVideoGenerationJob,
    ) -> AIVideoGenerationResult:
        assert isinstance(job, AIVideoGenerationJob)
        return AIVideoGenerationResult(
            job=job,
            artifacts=[
                AIVideoArtifact(
                    file_path=Path("/tmp/dummy-video.mp4"),
                    provider_metadata={"provider_job_id": job.provider_job_id},
                )
            ],
        )


def test_generate_video_defers_wait_settings_to_submitted_job_metadata() -> None:
    """generate_video() should not force the portable default wait settings onto providers."""

    provider: _GenerateVideoDummyProvider = _GenerateVideoDummyProvider()

    provider.generate_video("Render a lighthouse at sunset.")

    assert provider.wait_timeout_seconds is None
    assert provider.wait_poll_interval_seconds is None
