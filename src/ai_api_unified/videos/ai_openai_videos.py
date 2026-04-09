from __future__ import annotations

import logging
import mimetypes
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import httpx
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

    The initial rollout supports text-to-video plus one optional reference image.
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

    OpenAI documents this API as deprecated as of April 4, 2026, with shutdown scheduled
    for September 24, 2026. The implementation remains supported while the upstream API exists.
    """

    DEFAULT_VIDEO_MODEL: ClassVar[str] = "sora-2"
    DEFAULT_DURATION_SECONDS: ClassVar[int] = 8
    DEFAULT_RESOLUTION: ClassVar[str] = "1280x720"
    DEFAULT_ASPECT_RATIO: ClassVar[str] = "16:9"
    SUPPORTED_VIDEO_MODELS: ClassVar[list[str]] = ["sora-2", "sora-2-pro"]
    RETRYABLE_STATUS_CODES: ClassVar[set[int]] = {408, 429, 500, 502, 503, 504}
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
        AIOpenAIBase.__init__(self, **kwargs)
        AIBaseVideos.__init__(self, model=resolved_model)
        self.video_model_name: str = resolved_model
        self.http_client: httpx.Client | None = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
            },
            timeout=httpx.Timeout(timeout=120.0, connect=30.0),
            follow_redirects=True,
        )

    def model_name(self) -> str:
        return self.video_model_name

    def list_model_names(self) -> list[str]:
        return list(self.SUPPORTED_VIDEO_MODELS)

    def close(self) -> None:
        """Close the owned HTTP client."""

        http_client: httpx.Client | None = getattr(self, "http_client", None)
        if http_client is None:
            return
        http_client.close()
        self.http_client = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

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
        if json_payload is not None and (
            data_payload is not None or files_payload is not None
        ):
            raise ValueError(
                "OpenAI video requests must use either JSON or multipart form data, not both."
            )
        if self.http_client is None:
            raise RuntimeError(
                "AIOpenAIVideos http_client is closed and can no longer make requests."
            )
        max_attempts: int = len(self.backoff_delays)
        last_exception: Exception | None = None
        for attempt_index, delay_seconds in enumerate(self.backoff_delays, start=1):
            try:
                request_headers: dict[str, str] = {"Accept": accept}
                if json_payload is not None:
                    request_headers["Content-Type"] = "application/json"
                response: httpx.Response = self.http_client.request(
                    method=method,
                    url=path,
                    json=json_payload,
                    data=data_payload,
                    files=files_payload,
                    headers=request_headers,
                )
                if (
                    response.status_code in self.RETRYABLE_STATUS_CODES
                    and attempt_index < max_attempts
                ):
                    _LOGGER.warning(
                        "openai_video_request_retry",
                        extra={
                            "path": path,
                            "method": method,
                            "status_code": response.status_code,
                            "attempt": attempt_index,
                            "retry_in_seconds": delay_seconds,
                        },
                    )
                    time.sleep(delay_seconds)
                    continue
                response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.HTTPStatusError) as exception:
                last_exception = exception
                status_code: int | None = None
                if isinstance(exception, httpx.HTTPStatusError):
                    status_code = exception.response.status_code
                    if (
                        status_code not in self.RETRYABLE_STATUS_CODES
                        or attempt_index == max_attempts
                    ):
                        break
                elif attempt_index == max_attempts:
                    break
                _LOGGER.warning(
                    "openai_video_request_retry_exception",
                    extra={
                        "path": path,
                        "method": method,
                        "status_code": status_code,
                        "attempt": attempt_index,
                        "retry_in_seconds": delay_seconds,
                        "error_type": exception.__class__.__name__,
                    },
                )
                time.sleep(delay_seconds)
        assert last_exception is not None
        raise RuntimeError(
            f"OpenAI video request failed for {method} {path}."
        ) from last_exception

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
        payload: dict[str, Any],
        *,
        previous_provider_metadata: (
            dict[str, str | int | float | bool | None] | None
        ) = None,
    ) -> AIVideoGenerationJob:
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
        request_payload: dict[str, Any] = {
            "prompt": video_prompt,
            "model": self.video_model_name,
            "seconds": openai_properties.duration_seconds,
            "size": openai_properties.resolution,
        }
        if openai_properties.reference_image is not None:
            request_payload["input_reference"] = {
                "image_url": openai_properties.reference_image.to_data_url()
            }
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
            json_payload: dict[str, Any] | None = request_payload
            data_payload: dict[str, str] | None = None
            files_payload: dict[str, tuple[str, bytes, str]] | None = None
            if openai_properties.reference_image is not None:
                json_payload = None
                data_payload = {
                    "prompt": video_prompt,
                    "model": self.video_model_name,
                    "seconds": str(openai_properties.duration_seconds),
                    "size": str(openai_properties.resolution),
                }
                file_name, file_bytes, mime_type = self._build_reference_upload_file(
                    openai_properties.reference_image
                )
                files_payload = {"input_reference": (file_name, file_bytes, mime_type)}
            response: httpx.Response = self._request(
                "POST",
                "/videos",
                json_payload=json_payload,
                data_payload=data_payload,
                files_payload=files_payload,
            )
            payload: dict[str, Any] = response.json()
            job: AIVideoGenerationJob = self._normalize_job(
                payload,
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
        response: httpx.Response = self._request("GET", f"/videos/{provider_job_id}")
        payload: dict[str, Any] = response.json()
        return self._normalize_job(
            payload,
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
                response: httpx.Response = self._request(
                    "GET",
                    f"/videos/{current_job.provider_job_id}/content",
                    accept="video/mp4",
                )
                video_bytes: bytes = response.content
                total_output_bytes = len(video_bytes)
                file_path: Path = output_dir / f"{current_job.provider_job_id}.mp4"
                file_path.write_bytes(video_bytes)
                artifacts.append(
                    AIVideoArtifact(
                        mime_type=response.headers.get("content-type", "video/mp4"),
                        file_path=file_path,
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
