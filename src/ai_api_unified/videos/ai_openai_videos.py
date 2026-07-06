from __future__ import annotations

import logging
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from pydantic import model_validator

from ai_api_unified.ai_base import (
    AIMediaReference,
    AIBaseVideoProperties,
    AIBaseVideos,
    AIVideoArtifact,
    AIVideoGenerationJob,
    AIVideoGenerationResult,
    AIVideoGenerationStatus,
    AiApiObservedVideosResultModel,
)
from ai_api_unified.ai_openai_base import AIOpenAIBase
from ai_api_unified.util.env_settings import EnvSettings

_LOGGER: logging.Logger = logging.getLogger(__name__)


class AIOpenAIVideoProperties(AIBaseVideoProperties):
    """
    OpenAI video-generation request properties.

    Supports text-to-video plus one optional reference image.
    """

    reference_image: AIMediaReference | None = None
    source_video: AIMediaReference | None = None

    _ALLOWED_SECONDS: ClassVar[set[int]] = {4, 8, 12}
    _ALLOWED_RESOLUTIONS: ClassVar[set[str]] = {
        "720x1280",
        "1280x720",
        "1024x1792",
        "1792x1024",
    }
    _ALLOWED_ASPECT_RATIOS: ClassVar[set[str]] = {"16:9", "9:16"}

    @model_validator(mode="after")
    def _validate_openai_video_properties(self) -> "AIOpenAIVideoProperties":
        if (
            self.duration_seconds is not None
            and self.duration_seconds not in self._ALLOWED_SECONDS
        ):
            raise ValueError(
                "OpenAI video duration_seconds must be one of 4, 8, or 12 seconds."
            )
        if (
            self.resolution is not None
            and self.resolution not in self._ALLOWED_RESOLUTIONS
        ):
            raise ValueError(
                "OpenAI video resolution must be one of 720x1280, 1280x720, 1024x1792, or 1792x1024."
            )
        if (
            self.aspect_ratio is not None
            and self.aspect_ratio not in self._ALLOWED_ASPECT_RATIOS
        ):
            raise ValueError("OpenAI video aspect_ratio must be '16:9' or '9:16'.")
        if self.num_videos != 1:
            raise ValueError(
                "The current OpenAI video implementation supports exactly one generated video per request."
            )
        return self


class AIOpenAIVideos(AIOpenAIBase, AIBaseVideos):
    """
    OpenAI Sora video-generation provider.

    Uses the openai SDK's native ``client.videos`` resource for job submission,
    polling, and content download. The SDK sends the correct request shape for
    the current /v1/videos API (notably ``seconds`` as a string enum), which the
    prior hand-rolled HTTP client did not.
    """

    DEFAULT_VIDEO_MODEL: ClassVar[str] = "sora-2"
    DEFAULT_DURATION_SECONDS: ClassVar[int] = 8
    DEFAULT_RESOLUTION: ClassVar[str] = "1280x720"
    DEFAULT_ASPECT_RATIO: ClassVar[str] = "16:9"
    SUPPORTED_VIDEO_MODELS: ClassVar[list[str]] = ["sora-2", "sora-2-pro"]
    RESOLUTION_BY_ASPECT_RATIO: ClassVar[dict[str, str]] = {
        "16:9": "1280x720",
        "9:16": "720x1280",
    }
    ASPECT_RATIO_BY_RESOLUTION: ClassVar[dict[str, str]] = {
        "1280x720": "16:9",
        "1792x1024": "16:9",
        "720x1280": "9:16",
        "1024x1792": "9:16",
    }
    STATUS_MAPPING: ClassVar[dict[str, AIVideoGenerationStatus]] = {
        "queued": AIVideoGenerationStatus.QUEUED,
        "in_progress": AIVideoGenerationStatus.RUNNING,
        "completed": AIVideoGenerationStatus.COMPLETED,
        "failed": AIVideoGenerationStatus.FAILED,
    }

    def __init__(self, model: str | None = None, **kwargs: Any) -> None:
        env_settings: EnvSettings = EnvSettings()
        resolved_model: str = (
            model
            or env_settings.get_setting("VIDEO_MODEL_NAME", self.DEFAULT_VIDEO_MODEL)
            or self.DEFAULT_VIDEO_MODEL
        )
        resolved_model = resolved_model.strip() or self.DEFAULT_VIDEO_MODEL
        # AIOpenAIBase.__init__ builds self.client (the openai SDK client).
        AIOpenAIBase.__init__(self, **kwargs)
        AIBaseVideos.__init__(self, model=resolved_model)
        self.video_model_name: str = resolved_model

    def model_name(self) -> str:
        return self.video_model_name

    def list_model_names(self) -> list[str]:
        return list(self.SUPPORTED_VIDEO_MODELS)

    def _coerce_properties(
        self,
        video_properties: AIBaseVideoProperties,
    ) -> AIOpenAIVideoProperties:
        normalized_properties: AIBaseVideoProperties = (
            self._apply_environment_video_property_defaults(video_properties)
        )
        if isinstance(normalized_properties, AIOpenAIVideoProperties):
            openai_properties: AIOpenAIVideoProperties = (
                normalized_properties.model_copy(deep=True)
            )
        else:
            openai_property_payload: dict[str, Any] = normalized_properties.model_dump(
                exclude_unset=True
            )
            openai_properties = AIOpenAIVideoProperties(**openai_property_payload)
        explicit_fields: set[str] = set(openai_properties.model_fields_set)
        if (
            "duration_seconds" not in explicit_fields
            and openai_properties.duration_seconds is None
        ):
            openai_properties.duration_seconds = self.DEFAULT_DURATION_SECONDS
        if "resolution" not in explicit_fields and openai_properties.resolution is None:
            if openai_properties.aspect_ratio is not None:
                openai_properties.resolution = self.RESOLUTION_BY_ASPECT_RATIO[
                    openai_properties.aspect_ratio
                ]
            else:
                openai_properties.resolution = self.DEFAULT_RESOLUTION
        if (
            "aspect_ratio" not in explicit_fields
            and openai_properties.aspect_ratio is None
        ):
            assert openai_properties.resolution is not None
            openai_properties.aspect_ratio = self.ASPECT_RATIO_BY_RESOLUTION[
                openai_properties.resolution
            ]
        if (
            openai_properties.aspect_ratio is not None
            and openai_properties.resolution is not None
            and self.ASPECT_RATIO_BY_RESOLUTION[openai_properties.resolution]
            != openai_properties.aspect_ratio
        ):
            raise ValueError(
                "OpenAI video aspect_ratio and resolution must describe the same orientation."
            )
        return openai_properties

    def _build_reference_upload_file(
        self,
        media_reference: AIMediaReference,
    ) -> tuple[str, bytes, str]:
        if media_reference.remote_uri is not None:
            raise ValueError(
                "OpenAI video input_reference must be provided as local bytes or a local file path."
            )
        mime_type: str = media_reference.mime_type or "application/octet-stream"
        media_bytes: bytes = media_reference.read_bytes()
        if media_reference.file_path is not None:
            file_name: str = media_reference.file_path.name
        else:
            guessed_extension: str = mimetypes.guess_extension(mime_type) or ".bin"
            file_name = f"input_reference{guessed_extension}"
        return file_name, media_bytes, mime_type

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        return None

    def _extract_provider_job_id(self, job: str | AIVideoGenerationJob) -> str:
        if isinstance(job, AIVideoGenerationJob):
            return job.provider_job_id
        normalized_job_id: str = job.strip()
        if normalized_job_id == "":
            raise ValueError("job must be a non-empty string.")
        return normalized_job_id

    def _normalize_job(
        self,
        video: Any,
        *,
        previous_provider_metadata: (
            dict[str, str | int | float | bool | None] | None
        ) = None,
    ) -> AIVideoGenerationJob:
        # The SDK returns a pydantic Video; normalize to a plain payload so the
        # attribute reads below stay uniform across create/retrieve responses.
        payload: dict[str, Any] = (
            video.model_dump() if hasattr(video, "model_dump") else dict(video)
        )
        provider_job_id: str = str(payload["id"])
        raw_status: str = str(payload.get("status", "queued")).strip().lower()
        status: AIVideoGenerationStatus = self.STATUS_MAPPING.get(
            raw_status,
            AIVideoGenerationStatus.RUNNING,
        )
        dict_provider_metadata: dict[str, str | int | float | bool | None] = dict(
            previous_provider_metadata or {}
        )
        dict_provider_metadata.update(
            {
                "raw_status": raw_status,
                "expires_at": payload.get("expires_at"),
            }
        )
        error_payload: Any = payload.get("error")
        error_message: str | None = None
        if isinstance(error_payload, dict):
            error_message = error_payload.get("message")
        return AIVideoGenerationJob(
            job_id=f"openai:{provider_job_id}",
            provider_job_id=provider_job_id,
            status=status,
            progress_percent=payload.get("progress"),
            submitted_at_utc=self._parse_datetime(payload.get("created_at")),
            completed_at_utc=self._parse_datetime(payload.get("completed_at")),
            error_message=error_message,
            provider_engine=self._resolve_observability_provider_engine(),
            provider_model_name=str(payload.get("model") or self.video_model_name),
            provider_metadata=dict_provider_metadata,
        )

    def submit_video_generation(
        self,
        video_prompt: str,
        video_properties: AIBaseVideoProperties = AIBaseVideoProperties(),
    ) -> AIVideoGenerationJob:
        if video_prompt.strip() == "":
            raise ValueError("video_prompt must be a non-empty string.")
        openai_properties: AIOpenAIVideoProperties = self._coerce_properties(
            video_properties
        )
        if openai_properties.source_video is not None:
            raise NotImplementedError(
                "OpenAI source-video edits/extensions are not implemented in the initial video-generation rollout."
            )
        resolved_output_dir: Path | None = None
        if openai_properties.download_outputs:
            resolved_output_dir = self._resolve_video_output_dir(openai_properties)
        dict_input_metadata: dict[str, str | int | float | bool | None] = (
            self._build_videos_observability_input_metadata(
                video_prompt=video_prompt,
                video_properties=openai_properties,
            )
        )
        dict_input_metadata.update(
            {
                "request_mode": (
                    "image_guided"
                    if openai_properties.reference_image is not None
                    else "text_to_video"
                ),
                "provider_default_model": self.DEFAULT_VIDEO_MODEL,
            }
        )
        assert openai_properties.duration_seconds is not None
        assert openai_properties.resolution is not None
        # The SDK types seconds as a string enum ('4' | '8' | '12'); passing an
        # int is what the prior hand-rolled client got wrong.
        create_kwargs: dict[str, Any] = {
            "prompt": video_prompt,
            "model": self.video_model_name,
            "seconds": str(openai_properties.duration_seconds),
            "size": openai_properties.resolution,
        }
        if openai_properties.reference_image is not None:
            create_kwargs["input_reference"] = self._build_reference_upload_file(
                openai_properties.reference_image
            )
        provider_metadata: dict[str, str | int | float | bool | None] = {
            "download_outputs": openai_properties.download_outputs,
            "resolved_output_dir": (
                str(resolved_output_dir) if resolved_output_dir is not None else None
            ),
            "resolved_duration_seconds": openai_properties.duration_seconds,
            "resolved_resolution": openai_properties.resolution,
            "resolved_aspect_ratio": openai_properties.aspect_ratio,
            "resolved_timeout_seconds": openai_properties.timeout_seconds,
            "resolved_poll_interval_seconds": openai_properties.poll_interval_seconds,
        }

        def _execute_submit() -> AiApiObservedVideosResultModel[AIVideoGenerationJob]:
            video = self.client.videos.create(**create_kwargs)
            job: AIVideoGenerationJob = self._normalize_job(
                video,
                previous_provider_metadata=provider_metadata,
            )
            return AiApiObservedVideosResultModel(
                return_value=job,
                generated_video_count=1,
                total_output_bytes=0,
                dict_metadata={"job_status": job.status.value},
            )

        observed_result: AiApiObservedVideosResultModel[AIVideoGenerationJob] = (
            self._execute_provider_call_with_observability(
                capability=self.CLIENT_TYPE_VIDEOS,
                operation="submit_video_generation",
                dict_input_metadata=dict_input_metadata,
                callable_execute=_execute_submit,
                callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_videos_observability_result_summary(
                    observed_result=result,
                    provider_elapsed_ms=provider_elapsed_ms,
                ),
                legacy_caller_id=self.user,
            )
        )
        return observed_result.return_value

    def get_video_generation_job(
        self,
        job: str | AIVideoGenerationJob,
    ) -> AIVideoGenerationJob:
        provider_job_id: str = self._extract_provider_job_id(job)
        previous_provider_metadata: dict[str, str | int | float | bool | None] = {}
        if isinstance(job, AIVideoGenerationJob):
            previous_provider_metadata = dict(job.provider_metadata)
        video = self.client.videos.retrieve(provider_job_id)
        return self._normalize_job(
            video,
            previous_provider_metadata=previous_provider_metadata,
        )

    def download_video_result(
        self,
        job: str | AIVideoGenerationJob,
    ) -> AIVideoGenerationResult:
        current_job: AIVideoGenerationJob = self.get_video_generation_job(job)
        if current_job.status != AIVideoGenerationStatus.COMPLETED:
            raise ValueError(
                "download_video_result requires a completed job. "
                f"provider_job_id={current_job.provider_job_id} status={current_job.status.value}"
            )
        download_outputs: bool = bool(
            current_job.provider_metadata.get("download_outputs", True)
        )

        def _resolved_duration() -> int | None:
            raw_duration: Any = current_job.provider_metadata.get(
                "resolved_duration_seconds"
            )
            return int(raw_duration) if raw_duration is not None else None

        def _execute_download() -> (
            AiApiObservedVideosResultModel[AIVideoGenerationResult]
        ):
            artifacts: list[AIVideoArtifact] = []
            total_output_bytes: int = 0
            if download_outputs:
                output_dir_value: str | int | float | bool | None = (
                    current_job.provider_metadata.get("resolved_output_dir")
                )
                output_dir: Path = (
                    Path(str(output_dir_value))
                    if output_dir_value not in (None, "")
                    else self._resolve_video_output_dir(AIBaseVideoProperties())
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                content = self.client.videos.download_content(
                    current_job.provider_job_id,
                    variant="video",
                )
                video_bytes: bytes = content.read()
                total_output_bytes = len(video_bytes)
                file_path: Path = output_dir / f"{current_job.provider_job_id}.mp4"
                file_path.write_bytes(video_bytes)
                artifacts.append(
                    AIVideoArtifact(
                        mime_type="video/mp4",
                        file_path=file_path,
                        duration_seconds=_resolved_duration(),
                        provider_metadata={
                            "provider_job_id": current_job.provider_job_id
                        },
                    )
                )
            else:
                artifacts.append(
                    AIVideoArtifact(
                        mime_type="video/mp4",
                        remote_uri=f"{self.base_url}/videos/{current_job.provider_job_id}/content",
                        duration_seconds=_resolved_duration(),
                        provider_metadata={
                            "provider_job_id": current_job.provider_job_id
                        },
                    )
                )
            result: AIVideoGenerationResult = AIVideoGenerationResult(
                job=current_job,
                artifacts=artifacts,
                provider_metadata={
                    "download_outputs": download_outputs,
                    "provider_job_id": current_job.provider_job_id,
                },
            )
            return AiApiObservedVideosResultModel(
                return_value=result,
                generated_video_count=len(artifacts),
                total_output_bytes=total_output_bytes,
                dict_metadata={"download_outputs": download_outputs},
            )

        observed_result: AiApiObservedVideosResultModel[AIVideoGenerationResult] = (
            self._execute_provider_call_with_observability(
                capability=self.CLIENT_TYPE_VIDEOS,
                operation="download_video_result",
                dict_input_metadata={"provider_job_id": current_job.provider_job_id},
                callable_execute=_execute_download,
                callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_videos_observability_result_summary(
                    observed_result=result,
                    provider_elapsed_ms=provider_elapsed_ms,
                ),
                legacy_caller_id=self.user,
            )
        )
        return observed_result.return_value
