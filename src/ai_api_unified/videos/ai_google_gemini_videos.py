from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from google import genai
from google.api_core import exceptions as gexc
from google.auth.exceptions import DefaultCredentialsError
from google.genai import errors as gerr
from google.genai import types
from pydantic import Field, model_validator

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
from ai_api_unified.ai_google_base import AIGoogleBase
from ai_api_unified.util.env_settings import EnvSettings

_LOGGER: logging.Logger = logging.getLogger(__name__)


class AIGoogleGeminiVideoProperties(AIBaseVideoProperties):
    """
    Google Gemini Veo request properties.
    """

    reference_images: list[AIMediaReference] = Field(default_factory=list)
    image: AIMediaReference | None = None
    source_video: AIMediaReference | None = None
    last_frame_image: AIMediaReference | None = None
    person_generation: str | None = None
    generate_audio: bool | None = None

    _ALLOWED_ASPECT_RATIOS: ClassVar[set[str]] = {"16:9", "9:16"}
    _ALLOWED_RESOLUTIONS: ClassVar[set[str]] = {"720p", "1080p", "4k"}
    _ALLOWED_PERSON_GENERATION: ClassVar[set[str]] = {
        "dont_allow",
        "allow_adult",
        "allow_all",
    }

    @model_validator(mode="after")
    def _validate_google_video_properties(self) -> "AIGoogleGeminiVideoProperties":
        if (
            self.aspect_ratio is not None
            and self.aspect_ratio not in self._ALLOWED_ASPECT_RATIOS
        ):
            raise ValueError(
                "Google Gemini video aspect_ratio must be '16:9' or '9:16'."
            )
        if (
            self.resolution is not None
            and self.resolution not in self._ALLOWED_RESOLUTIONS
        ):
            raise ValueError(
                "Google Gemini video resolution must be one of 720p, 1080p, or 4k."
            )
        if (
            self.person_generation is not None
            and self.person_generation not in self._ALLOWED_PERSON_GENERATION
        ):
            raise ValueError(
                "Google Gemini person_generation must be one of dont_allow, allow_adult, or allow_all."
            )
        return self


class AIGoogleGeminiVideos(AIGoogleBase, AIBaseVideos):
    """
    Google Gemini Veo video-generation provider.
    """

    DEFAULT_VIDEO_MODEL: ClassVar[str] = "veo-3.1-lite-generate-preview"
    DEFAULT_ASPECT_RATIO: ClassVar[str] = "16:9"
    DEFAULT_RESOLUTION: ClassVar[str] = "720p"
    DEFAULT_TIMEOUT_SECONDS: ClassVar[int] = 900
    DEFAULT_POLL_INTERVAL_SECONDS: ClassVar[int] = 10
    DOWNLOAD_OPERATION_READY_TIMEOUT_SECONDS: ClassVar[int] = 30
    DOWNLOAD_OPERATION_READY_POLL_INTERVAL_SECONDS: ClassVar[int] = 2
    SUPPORTED_VIDEO_MODELS: ClassVar[list[str]] = [
        "veo-3.1-generate-preview",
        "veo-3.1-fast-generate-preview",
        "veo-3.1-lite-generate-preview",
        "veo-3.0-generate-001",
        "veo-3.0-fast-generate-001",
        "veo-2.0-generate-001",
    ]
    STATUS_RUNNING: ClassVar[AIVideoGenerationStatus] = AIVideoGenerationStatus.RUNNING
    PROVIDER_METADATA_SUBMITTED_AT_UTC_KEY: ClassVar[str] = "job_submitted_at_utc"
    PROVIDER_METADATA_COMPLETED_AT_UTC_KEY: ClassVar[str] = "job_completed_at_utc"

    def __init__(self, model: str | None = None, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        env_settings: EnvSettings = EnvSettings()
        resolved_model: str = (
            model
            or env_settings.get_setting("VIDEO_MODEL_NAME", self.DEFAULT_VIDEO_MODEL)
            or self.DEFAULT_VIDEO_MODEL
        )
        resolved_model = resolved_model.strip() or self.DEFAULT_VIDEO_MODEL
        self.video_model_name: str = resolved_model
        self.client: genai.Client = self.get_client(model=self.video_model_name)

    def model_name(self) -> str:
        return self.video_model_name

    def list_model_names(self) -> list[str]:
        return list(self.SUPPORTED_VIDEO_MODELS)

    def _get_job_timestamp_cache(self) -> dict[str, dict[str, str]]:
        cache: dict[str, dict[str, str]] | None = getattr(
            self,
            "_job_timestamp_cache",
            None,
        )
        if cache is None:
            cache = {}
            self._job_timestamp_cache = cache
        return cache

    def _parse_provider_timestamp(
        self,
        value: str | int | float | bool | None,
    ) -> datetime | None:
        if value in (None, "") or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None

    def _cache_job_timestamps(
        self,
        operation_name: str,
        submitted_at_utc: datetime,
        completed_at_utc: datetime | None,
    ) -> None:
        timestamp_cache: dict[str, dict[str, str]] = self._get_job_timestamp_cache()
        cached_values: dict[str, str] = {
            self.PROVIDER_METADATA_SUBMITTED_AT_UTC_KEY: submitted_at_utc.isoformat()
        }
        if completed_at_utc is not None:
            cached_values[self.PROVIDER_METADATA_COMPLETED_AT_UTC_KEY] = (
                completed_at_utc.isoformat()
            )
        timestamp_cache[operation_name] = cached_values

    def _resolve_stable_job_timestamps(
        self,
        operation_name: str,
        *,
        operation_done: bool,
        previous_provider_metadata: (
            dict[str, str | int | float | bool | None] | None
        ) = None,
        previous_job: AIVideoGenerationJob | None = None,
    ) -> tuple[datetime, datetime | None]:
        timestamp_cache: dict[str, dict[str, str]] = self._get_job_timestamp_cache()
        cached_values: dict[str, str] = timestamp_cache.get(operation_name, {})
        submitted_at_utc: datetime = (
            previous_job.submitted_at_utc
            if previous_job is not None and previous_job.submitted_at_utc is not None
            else self._parse_provider_timestamp(
                (previous_provider_metadata or {}).get(
                    self.PROVIDER_METADATA_SUBMITTED_AT_UTC_KEY
                )
            )
            or self._parse_provider_timestamp(
                cached_values.get(self.PROVIDER_METADATA_SUBMITTED_AT_UTC_KEY)
            )
            or datetime.now(timezone.utc)
        )
        completed_at_utc: datetime | None = (
            previous_job.completed_at_utc
            if previous_job is not None and previous_job.completed_at_utc is not None
            else self._parse_provider_timestamp(
                (previous_provider_metadata or {}).get(
                    self.PROVIDER_METADATA_COMPLETED_AT_UTC_KEY
                )
            )
            or self._parse_provider_timestamp(
                cached_values.get(self.PROVIDER_METADATA_COMPLETED_AT_UTC_KEY)
            )
        )
        if operation_done and completed_at_utc is None:
            completed_at_utc = datetime.now(timezone.utc)
        self._cache_job_timestamps(
            operation_name=operation_name,
            submitted_at_utc=submitted_at_utc,
            completed_at_utc=completed_at_utc,
        )
        return submitted_at_utc, completed_at_utc

    def _coerce_properties(
        self,
        video_properties: AIBaseVideoProperties,
    ) -> AIGoogleGeminiVideoProperties:
        normalized_properties: AIBaseVideoProperties = (
            self._apply_environment_video_property_defaults(video_properties)
        )
        if isinstance(normalized_properties, AIGoogleGeminiVideoProperties):
            google_properties: AIGoogleGeminiVideoProperties = (
                normalized_properties.model_copy(deep=True)
            )
        else:
            google_property_payload: dict[str, Any] = normalized_properties.model_dump(
                exclude_unset=True
            )
            google_properties = AIGoogleGeminiVideoProperties(**google_property_payload)
        explicit_fields: set[str] = set(google_properties.model_fields_set)
        if (
            "aspect_ratio" not in explicit_fields
            and google_properties.aspect_ratio is None
        ):
            google_properties.aspect_ratio = self.DEFAULT_ASPECT_RATIO
        if "resolution" not in explicit_fields and google_properties.resolution is None:
            google_properties.resolution = self.DEFAULT_RESOLUTION
        return google_properties

    def _to_google_image(self, media_reference: AIMediaReference) -> types.Image:
        if media_reference.remote_uri is not None:
            normalized_remote_uri: str = media_reference.remote_uri.strip()
            if not normalized_remote_uri.startswith("gs://"):
                raise ValueError(
                    "Google Gemini image inputs with remote_uri must use a gs:// URI."
                )
            return types.Image(gcs_uri=media_reference.remote_uri)
        mime_type: str = media_reference.mime_type or "image/png"
        return types.Image(
            image_bytes=media_reference.read_bytes(),
            mime_type=mime_type,
        )

    def _to_google_video(self, media_reference: AIMediaReference) -> types.Video:
        if media_reference.remote_uri is None:
            raise ValueError(
                "Google Gemini source_video must be provided as a GCS URI."
            )
        normalized_remote_uri: str = media_reference.remote_uri.strip()
        if not normalized_remote_uri.startswith("gs://"):
            raise ValueError("Google Gemini source_video must use a gs:// URI.")
        return types.Video(uri=normalized_remote_uri)

    def _get_operation(
        self, job: str | AIVideoGenerationJob
    ) -> types.GenerateVideosOperation:
        operation_reference: str
        if isinstance(job, AIVideoGenerationJob):
            operation_reference = job.provider_job_id
        else:
            operation_reference = job.strip()
        if operation_reference == "":
            raise ValueError("job must be a non-empty string.")
        operation_handle: types.GenerateVideosOperation = types.GenerateVideosOperation(
            name=operation_reference
        )
        return self._retry_with_exponential_backoff(
            lambda: self.client.operations.get(operation=operation_handle)
        )

    def _normalize_status(
        self,
        operation: types.GenerateVideosOperation,
    ) -> AIVideoGenerationStatus:
        if operation.done:
            if operation.error is not None:
                return AIVideoGenerationStatus.FAILED
            return AIVideoGenerationStatus.COMPLETED
        return self.STATUS_RUNNING

    def _normalize_job(
        self,
        operation: types.GenerateVideosOperation,
        *,
        previous_provider_metadata: (
            dict[str, str | int | float | bool | None] | None
        ) = None,
        previous_job: AIVideoGenerationJob | None = None,
    ) -> AIVideoGenerationJob:
        provider_metadata: dict[str, str | int | float | bool | None] = dict(
            previous_provider_metadata or {}
        )
        provider_metadata["done"] = operation.done
        error_message: str | None = None
        if operation.error is not None:
            error_message = str(operation.error)
        submitted_at_utc: datetime
        completed_at_utc: datetime | None
        submitted_at_utc, completed_at_utc = self._resolve_stable_job_timestamps(
            operation_name=operation.name,
            operation_done=operation.done,
            previous_provider_metadata=previous_provider_metadata,
            previous_job=previous_job,
        )
        provider_metadata[self.PROVIDER_METADATA_SUBMITTED_AT_UTC_KEY] = (
            submitted_at_utc.isoformat()
        )
        provider_metadata[self.PROVIDER_METADATA_COMPLETED_AT_UTC_KEY] = (
            completed_at_utc.isoformat() if completed_at_utc is not None else None
        )
        return AIVideoGenerationJob(
            job_id=f"google-gemini:{operation.name}",
            provider_job_id=operation.name,
            status=self._normalize_status(operation),
            progress_percent=None,
            submitted_at_utc=submitted_at_utc,
            completed_at_utc=completed_at_utc,
            error_message=error_message,
            provider_engine=self._resolve_observability_provider_engine(),
            provider_model_name=self.video_model_name,
            provider_metadata=provider_metadata,
        )

    def submit_video_generation(
        self,
        video_prompt: str,
        video_properties: AIBaseVideoProperties = AIBaseVideoProperties(),
    ) -> AIVideoGenerationJob:
        if video_prompt.strip() == "":
            raise ValueError("video_prompt must be a non-empty string.")
        google_properties: AIGoogleGeminiVideoProperties = self._coerce_properties(
            video_properties
        )
        resolved_output_dir: Path | None = None
        if google_properties.download_outputs:
            resolved_output_dir = self._resolve_video_output_dir(google_properties)

        dict_input_metadata: dict[str, str | int | float | bool | None] = (
            self._build_videos_observability_input_metadata(
                video_prompt=video_prompt,
                video_properties=google_properties,
            )
        )
        dict_input_metadata.update(
            {
                "has_image": google_properties.image is not None,
                "has_source_video": google_properties.source_video is not None,
                "reference_image_count": len(google_properties.reference_images),
                "has_last_frame_image": google_properties.last_frame_image is not None,
            }
        )

        config_kwargs: dict[str, Any] = {
            "number_of_videos": google_properties.num_videos,
            "aspect_ratio": google_properties.aspect_ratio,
            "resolution": google_properties.resolution,
        }
        if google_properties.duration_seconds is not None:
            config_kwargs["duration_seconds"] = google_properties.duration_seconds
        if google_properties.seed is not None:
            config_kwargs["seed"] = google_properties.seed
        if google_properties.fps is not None:
            config_kwargs["fps"] = google_properties.fps
        if google_properties.generate_audio is not None:
            config_kwargs["generate_audio"] = google_properties.generate_audio
        if google_properties.person_generation is not None:
            config_kwargs["person_generation"] = google_properties.person_generation
        if google_properties.last_frame_image is not None:
            config_kwargs["last_frame"] = self._to_google_image(
                google_properties.last_frame_image
            )
        if google_properties.reference_images:
            config_kwargs["reference_images"] = [
                types.VideoGenerationReferenceImage(
                    image=self._to_google_image(reference_image),
                    reference_type=types.VideoGenerationReferenceType.ASSET,
                )
                for reference_image in google_properties.reference_images
            ]
        config: types.GenerateVideosConfig = types.GenerateVideosConfig(**config_kwargs)
        request_kwargs: dict[str, Any] = {
            "model": self.video_model_name,
            "prompt": video_prompt,
            "config": config,
        }
        if google_properties.image is not None:
            request_kwargs["image"] = self._to_google_image(google_properties.image)
        if google_properties.source_video is not None:
            auth_method: str = self._resolve_google_auth_method(EnvSettings())
            if auth_method == self.GOOGLE_AUTH_METHOD_API_KEY:
                raise NotImplementedError(
                    "Google Gemini source_video generation requires Vertex AI and is not available when GOOGLE_AUTH_METHOD=api_key."
                )
            request_kwargs["video"] = self._to_google_video(
                google_properties.source_video
            )

        provider_metadata: dict[str, str | int | float | bool | None] = {
            "download_outputs": google_properties.download_outputs,
            "resolved_output_dir": (
                str(resolved_output_dir) if resolved_output_dir is not None else None
            ),
            "resolved_duration_seconds": google_properties.duration_seconds,
            "resolved_resolution": google_properties.resolution,
            "resolved_aspect_ratio": google_properties.aspect_ratio,
            "resolved_timeout_seconds": google_properties.timeout_seconds,
            "resolved_poll_interval_seconds": google_properties.poll_interval_seconds,
        }

        def _execute_submit() -> AiApiObservedVideosResultModel[AIVideoGenerationJob]:
            try:
                operation: types.GenerateVideosOperation = (
                    self._retry_with_exponential_backoff(
                        lambda: self.client.models.generate_videos(**request_kwargs)
                    )
                )
            except (
                gerr.APIError,
                gexc.GoogleAPICallError,
                DefaultCredentialsError,
            ) as exception:
                _LOGGER.error(
                    "google_gemini_video_generation_failed",
                    extra={
                        "model": self.video_model_name,
                        "error_type": exception.__class__.__name__,
                    },
                )
                raise RuntimeError(
                    "Google Gemini video generation failed."
                ) from exception
            job: AIVideoGenerationJob = self._normalize_job(
                operation,
                previous_provider_metadata=provider_metadata,
            )
            return AiApiObservedVideosResultModel(
                return_value=job,
                generated_video_count=google_properties.num_videos,
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
                legacy_caller_id=None,
            )
        )
        return observed_result.return_value

    def get_video_generation_job(
        self,
        job: str | AIVideoGenerationJob,
    ) -> AIVideoGenerationJob:
        previous_provider_metadata: dict[str, str | int | float | bool | None] = {}
        previous_job: AIVideoGenerationJob | None = None
        if isinstance(job, AIVideoGenerationJob):
            previous_job = job
            previous_provider_metadata = dict(job.provider_metadata)
        operation: types.GenerateVideosOperation = self._get_operation(job)
        return self._normalize_job(
            operation,
            previous_provider_metadata=previous_provider_metadata,
            previous_job=previous_job,
        )

    def _get_downloadable_operation(
        self,
        job: str | AIVideoGenerationJob,
        *,
        current_job: AIVideoGenerationJob,
    ) -> types.GenerateVideosOperation:
        operation: types.GenerateVideosOperation = self._get_operation(job)
        if operation.done or current_job.status != AIVideoGenerationStatus.COMPLETED:
            return operation

        deadline_monotonic: float = (
            time.monotonic() + self.DOWNLOAD_OPERATION_READY_TIMEOUT_SECONDS
        )
        while not operation.done and time.monotonic() < deadline_monotonic:
            time.sleep(self.DOWNLOAD_OPERATION_READY_POLL_INTERVAL_SECONDS)
            operation = self._get_operation(job)
        return operation

    def download_video_result(
        self,
        job: str | AIVideoGenerationJob,
    ) -> AIVideoGenerationResult:
        previous_provider_metadata: dict[str, str | int | float | bool | None]
        if isinstance(job, AIVideoGenerationJob):
            current_job = job.model_copy(deep=True)
            previous_provider_metadata = dict(current_job.provider_metadata)
        else:
            current_job = self.get_video_generation_job(job)
            previous_provider_metadata = dict(current_job.provider_metadata)
        if current_job.status != AIVideoGenerationStatus.COMPLETED:
            raise ValueError(
                "download_video_result requires a completed Google video-generation job."
            )
        operation = self._get_downloadable_operation(job, current_job=current_job)
        current_job = self._normalize_job(
            operation,
            previous_provider_metadata=previous_provider_metadata,
            previous_job=current_job,
        )
        if current_job.status != AIVideoGenerationStatus.COMPLETED:
            raise ValueError(
                "download_video_result requires a completed Google video-generation job."
            )
        generated_videos: Any = getattr(operation.response, "generated_videos", None)
        if not generated_videos:
            raise ValueError(
                "Google Gemini video generation completed but returned no generated videos."
            )
        download_outputs: bool = bool(
            current_job.provider_metadata.get("download_outputs", True)
        )

        def _execute_download() -> (
            AiApiObservedVideosResultModel[AIVideoGenerationResult]
        ):
            artifacts: list[AIVideoArtifact] = []
            total_output_bytes: int = 0
            resolved_output_dir: Path | None = None
            if download_outputs:
                output_dir_value: str | int | float | bool | None = (
                    current_job.provider_metadata.get("resolved_output_dir")
                )
                resolved_output_dir = (
                    Path(str(output_dir_value))
                    if output_dir_value not in (None, "")
                    else self._resolve_video_output_dir(AIBaseVideoProperties())
                )
                resolved_output_dir.mkdir(parents=True, exist_ok=True)

            for index, generated_video in enumerate(generated_videos, start=1):
                video_object: types.Video = generated_video.video
                remote_uri: str | None = getattr(video_object, "uri", None)
                video_bytes: bytes | None = None
                if download_outputs:
                    downloaded_bytes: Any = self.client.files.download(
                        file=video_object
                    )
                    if isinstance(downloaded_bytes, (bytes, bytearray)):
                        video_bytes = bytes(downloaded_bytes)
                    else:
                        candidate_bytes: Any = getattr(
                            video_object, "video_bytes", None
                        )
                        if isinstance(candidate_bytes, (bytes, bytearray)):
                            video_bytes = bytes(candidate_bytes)
                    if video_bytes is None:
                        raise ValueError(
                            "Google Gemini files.download did not return video bytes."
                        )
                    total_output_bytes += len(video_bytes)
                    assert resolved_output_dir is not None
                    file_path: Path = (
                        resolved_output_dir
                        / f"{current_job.provider_job_id.replace('/', '_')}_{index}.mp4"
                    )
                    file_path.write_bytes(video_bytes)
                else:
                    file_path = None
                artifacts.append(
                    AIVideoArtifact(
                        mime_type=getattr(video_object, "mime_type", "video/mp4")
                        or "video/mp4",
                        file_path=file_path,
                        remote_uri=remote_uri,
                        duration_seconds=(
                            int(
                                current_job.provider_metadata[
                                    "resolved_duration_seconds"
                                ]
                            )
                            if current_job.provider_metadata.get(
                                "resolved_duration_seconds"
                            )
                            is not None
                            else None
                        ),
                        has_audio=None,
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
                legacy_caller_id=None,
            )
        )
        return observed_result.return_value
