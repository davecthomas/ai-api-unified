from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, ClassVar

import boto3
from pydantic import BaseModel, Field, model_validator

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
from ai_api_unified.ai_bedrock_base import AIBedrockBase
from ai_api_unified.util.env_settings import EnvSettings

_LOGGER: logging.Logger = logging.getLogger(__name__)


class AIVideoShot(BaseModel):
    """One Nova Reel manual multi-shot segment."""

    text: str
    image: AIMediaReference | None = None

    @model_validator(mode="after")
    def _validate_text(self) -> "AIVideoShot":
        if self.text.strip() == "":
            raise ValueError("AIVideoShot.text must be a non-empty string.")
        return self


class AINovaReelVideoProperties(AIBaseVideoProperties):
    """Amazon Nova Reel request properties."""

    shots: list[AIVideoShot] = Field(default_factory=list)
    negative_prompt: str | None = None
    s3_output_uri: str | None = None
    reference_image: AIMediaReference | None = None

    _ALLOWED_RESOLUTION: ClassVar[str] = "1280x720"
    _ALLOWED_FPS: ClassVar[int] = 24

    @model_validator(mode="after")
    def _validate_nova_reel_properties(self) -> "AINovaReelVideoProperties":
        if self.resolution is not None and self.resolution != self._ALLOWED_RESOLUTION:
            raise ValueError("Nova Reel resolution must be 1280x720.")
        if self.fps is not None and self.fps != self._ALLOWED_FPS:
            raise ValueError("Nova Reel fps must be 24.")
        if self.num_videos != 1:
            raise ValueError(
                "The current Nova Reel implementation supports exactly one generated video per request."
            )
        if self.duration_seconds is not None:
            if self.duration_seconds < 6 or self.duration_seconds > 120:
                raise ValueError(
                    "Nova Reel duration_seconds must be between 6 and 120 seconds."
                )
            if self.duration_seconds % 6 != 0:
                raise ValueError(
                    "Nova Reel duration_seconds must be expressed in 6-second increments."
                )
        if self.s3_output_uri is not None and not self.s3_output_uri.startswith(
            "s3://"
        ):
            raise ValueError("Nova Reel s3_output_uri must start with 's3://'.")
        return self


class AINovaReelVideos(AIBedrockBase, AIBaseVideos):
    """Amazon Bedrock Nova Reel video-generation provider."""

    DEFAULT_VIDEO_MODEL: ClassVar[str] = "amazon.nova-reel-v1:1"
    DEFAULT_DURATION_SECONDS: ClassVar[int] = 6
    DEFAULT_RESOLUTION: ClassVar[str] = "1280x720"
    DEFAULT_FPS: ClassVar[int] = 24
    DEFAULT_TIMEOUT_SECONDS: ClassVar[int] = 1_800
    DEFAULT_POLL_INTERVAL_SECONDS: ClassVar[int] = 10
    SUPPORTED_VIDEO_MODELS: ClassVar[list[str]] = [DEFAULT_VIDEO_MODEL]
    STATUS_MAPPING: ClassVar[dict[str, AIVideoGenerationStatus]] = {
        "InProgress": AIVideoGenerationStatus.RUNNING,
        "Completed": AIVideoGenerationStatus.COMPLETED,
        "Failed": AIVideoGenerationStatus.FAILED,
    }

    def __init__(self, model: str | None = None, **kwargs: Any) -> None:
        env_settings: EnvSettings = EnvSettings()
        resolved_model: str = (
            model
            or env_settings.get_setting("VIDEO_MODEL_NAME", self.DEFAULT_VIDEO_MODEL)
            or self.DEFAULT_VIDEO_MODEL
        )
        resolved_model = resolved_model.strip() or self.DEFAULT_VIDEO_MODEL
        AIBedrockBase.__init__(self, model=resolved_model, **kwargs)
        AIBaseVideos.__init__(self, model=resolved_model)
        self.video_model_name: str = resolved_model
        self.s3_client = boto3.client("s3", region_name=self.region)

    def model_name(self) -> str:
        return self.video_model_name

    def list_model_names(self) -> list[str]:
        return list(self.SUPPORTED_VIDEO_MODELS)

    def _coerce_properties(
        self,
        video_properties: AIBaseVideoProperties,
    ) -> AINovaReelVideoProperties:
        normalized_properties: AIBaseVideoProperties = (
            self._apply_environment_video_property_defaults(video_properties)
        )
        if isinstance(normalized_properties, AINovaReelVideoProperties):
            nova_properties: AINovaReelVideoProperties = (
                normalized_properties.model_copy(deep=True)
            )
        else:
            nova_properties = AINovaReelVideoProperties(
                duration_seconds=normalized_properties.duration_seconds,
                aspect_ratio=normalized_properties.aspect_ratio,
                resolution=normalized_properties.resolution,
                fps=normalized_properties.fps,
                num_videos=normalized_properties.num_videos,
                seed=normalized_properties.seed,
                output_format=normalized_properties.output_format,
                poll_interval_seconds=normalized_properties.poll_interval_seconds,
                timeout_seconds=normalized_properties.timeout_seconds,
                output_dir=normalized_properties.output_dir,
                download_outputs=normalized_properties.download_outputs,
            )
        explicit_fields: set[str] = set(nova_properties.model_fields_set)
        if (
            "duration_seconds" not in explicit_fields
            and nova_properties.duration_seconds is None
        ):
            nova_properties.duration_seconds = self.DEFAULT_DURATION_SECONDS
        if "resolution" not in explicit_fields and nova_properties.resolution is None:
            nova_properties.resolution = self.DEFAULT_RESOLUTION
        if "fps" not in explicit_fields and nova_properties.fps is None:
            nova_properties.fps = self.DEFAULT_FPS
        if "timeout_seconds" not in explicit_fields:
            nova_properties.timeout_seconds = self.DEFAULT_TIMEOUT_SECONDS
        if "poll_interval_seconds" not in explicit_fields:
            nova_properties.poll_interval_seconds = self.DEFAULT_POLL_INTERVAL_SECONDS
        return nova_properties

    def _resolve_output_s3_uri(self, properties: AINovaReelVideoProperties) -> str:
        if properties.s3_output_uri is not None and properties.s3_output_uri.strip():
            return properties.s3_output_uri.strip()
        env_settings: EnvSettings = EnvSettings()
        configured_output_uri: str | None = env_settings.get_setting(
            "BEDROCK_VIDEO_OUTPUT_S3_URI",
            None,
        )
        if configured_output_uri is None or configured_output_uri.strip() == "":
            raise ValueError(
                "Nova Reel video generation requires an S3 output destination. "
                "Set AINovaReelVideoProperties.s3_output_uri or BEDROCK_VIDEO_OUTPUT_S3_URI."
            )
        return configured_output_uri.strip()

    def _to_bedrock_image(self, media_reference: AIMediaReference) -> dict[str, Any]:
        mime_type: str = (media_reference.mime_type or "image/png").lower()
        image_format: str = (
            "jpeg" if "jpeg" in mime_type or "jpg" in mime_type else "png"
        )
        if media_reference.remote_uri is not None:
            return {
                "format": image_format,
                "source": {"s3Location": {"uri": media_reference.remote_uri}},
            }
        image_bytes: bytes = media_reference.read_bytes()
        return {
            "format": image_format,
            "source": {"bytes": base64.b64encode(image_bytes).decode("ascii")},
        }

    def _build_model_input(
        self,
        *,
        video_prompt: str,
        properties: AINovaReelVideoProperties,
    ) -> dict[str, Any]:
        video_generation_config: dict[str, Any] = {
            "fps": properties.fps,
            "dimension": properties.resolution,
        }
        if properties.seed is not None:
            video_generation_config["seed"] = properties.seed

        if properties.shots:
            shots: list[dict[str, Any]] = []
            for shot in properties.shots:
                shot_payload: dict[str, Any] = {"text": shot.text}
                if shot.image is not None:
                    shot_payload["image"] = self._to_bedrock_image(shot.image)
                shots.append(shot_payload)
            return {
                "taskType": "MULTI_SHOT_MANUAL",
                "multiShotManualParams": {"shots": shots},
                "videoGenerationConfig": video_generation_config,
            }

        if properties.duration_seconds is not None and properties.duration_seconds > 6:
            automated_params: dict[str, Any] = {"text": video_prompt}
            if properties.negative_prompt:
                automated_params["negativeText"] = properties.negative_prompt
            video_generation_config["durationSeconds"] = properties.duration_seconds
            return {
                "taskType": "MULTI_SHOT_AUTOMATED",
                "multiShotAutomatedParams": automated_params,
                "videoGenerationConfig": video_generation_config,
            }

        text_to_video_params: dict[str, Any] = {"text": video_prompt}
        if properties.negative_prompt:
            text_to_video_params["negativeText"] = properties.negative_prompt
        if properties.reference_image is not None:
            text_to_video_params["images"] = [
                self._to_bedrock_image(properties.reference_image)
            ]
        video_generation_config["durationSeconds"] = properties.duration_seconds
        return {
            "taskType": "TEXT_VIDEO",
            "textToVideoParams": text_to_video_params,
            "videoGenerationConfig": video_generation_config,
        }

    def _extract_provider_job_id(self, job: str | AIVideoGenerationJob) -> str:
        if isinstance(job, AIVideoGenerationJob):
            return job.provider_job_id
        normalized_job_id: str = job.strip()
        if normalized_job_id == "":
            raise ValueError("job must be a non-empty string.")
        return normalized_job_id

    def _get_async_invoke_response(
        self,
        provider_job_id: str,
    ) -> dict[str, Any]:
        return self._execute_with_retries(
            operation=lambda: self.client.get_async_invoke(
                invocationArn=provider_job_id
            ),
            trace_name="get_async_invoke",
        )

    def _normalize_job(
        self,
        payload: dict[str, Any],
        *,
        previous_provider_metadata: (
            dict[str, str | int | float | bool | None] | None
        ) = None,
    ) -> AIVideoGenerationJob:
        provider_job_id: str = str(payload["invocationArn"])
        raw_status: str = str(payload.get("status", "InProgress"))
        status: AIVideoGenerationStatus = self.STATUS_MAPPING.get(
            raw_status,
            AIVideoGenerationStatus.RUNNING,
        )
        output_s3_uri: str | None = None
        output_data_config: Any = payload.get("outputDataConfig")
        if isinstance(output_data_config, dict):
            s3_output_data_config: Any = output_data_config.get("s3OutputDataConfig")
            if isinstance(s3_output_data_config, dict):
                output_s3_uri = s3_output_data_config.get("s3Uri")
        provider_metadata: dict[str, str | int | float | bool | None] = dict(
            previous_provider_metadata or {}
        )
        provider_metadata.update(
            {
                "raw_status": raw_status,
                "output_s3_uri": output_s3_uri,
            }
        )
        return AIVideoGenerationJob(
            job_id=f"bedrock:{provider_job_id}",
            provider_job_id=provider_job_id,
            status=status,
            progress_percent=None,
            submitted_at_utc=payload.get("submitTime"),
            completed_at_utc=payload.get("endTime"),
            error_message=payload.get("failureMessage"),
            provider_engine=self._resolve_observability_provider_engine(),
            provider_model_name=str(payload.get("modelArn") or self.video_model_name),
            provider_metadata=provider_metadata,
        )

    def _parse_s3_uri(self, s3_uri: str) -> tuple[str, str]:
        normalized_s3_uri: str = s3_uri.replace("s3://", "", 1)
        if "/" not in normalized_s3_uri:
            return normalized_s3_uri, ""
        bucket_name, key_prefix = normalized_s3_uri.split("/", 1)
        return bucket_name, key_prefix

    def _resolve_invocation_prefix(self, base_prefix: str, provider_job_id: str) -> str:
        invocation_id: str = provider_job_id.rsplit("/", 1)[-1]
        normalized_base_prefix: str = base_prefix.rstrip("/")
        if normalized_base_prefix == "":
            return f"{invocation_id}/"
        return f"{normalized_base_prefix}/{invocation_id}/"

    def _find_output_video_s3_uri(
        self,
        *,
        output_s3_uri: str,
        provider_job_id: str,
    ) -> str:
        bucket_name, base_prefix = self._parse_s3_uri(output_s3_uri)
        prefixes_to_check: list[str] = [
            self._resolve_invocation_prefix(base_prefix, provider_job_id),
            base_prefix,
        ]
        for prefix in prefixes_to_check:
            response: dict[str, Any] = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix,
            )
            contents: list[dict[str, Any]] = list(response.get("Contents", []))
            mp4_keys: list[str] = sorted(
                item["Key"]
                for item in contents
                if str(item.get("Key", "")).lower().endswith(".mp4")
            )
            if mp4_keys:
                return f"s3://{bucket_name}/{mp4_keys[0]}"
        raise RuntimeError(
            "Nova Reel output completed but no MP4 artifact was found in the configured S3 output location."
        )

    def submit_video_generation(
        self,
        video_prompt: str,
        video_properties: AIBaseVideoProperties = AIBaseVideoProperties(),
    ) -> AIVideoGenerationJob:
        if video_prompt.strip() == "":
            raise ValueError("video_prompt must be a non-empty string.")
        nova_properties: AINovaReelVideoProperties = self._coerce_properties(
            video_properties
        )
        output_s3_uri: str = self._resolve_output_s3_uri(nova_properties)
        model_input: dict[str, Any] = self._build_model_input(
            video_prompt=video_prompt,
            properties=nova_properties,
        )
        resolved_output_dir: Path | None = None
        if nova_properties.download_outputs:
            resolved_output_dir = self._resolve_video_output_dir(nova_properties)
        dict_input_metadata: dict[str, str | int | float | bool | None] = (
            self._build_videos_observability_input_metadata(
                video_prompt=video_prompt,
                video_properties=nova_properties,
            )
        )
        dict_input_metadata.update(
            {
                "shot_count": len(nova_properties.shots),
                "has_reference_image": nova_properties.reference_image is not None,
                "has_negative_prompt": bool(nova_properties.negative_prompt),
                "output_s3_uri": output_s3_uri,
            }
        )
        provider_metadata: dict[str, str | int | float | bool | None] = {
            "download_outputs": nova_properties.download_outputs,
            "resolved_output_dir": (
                str(resolved_output_dir) if resolved_output_dir is not None else None
            ),
            "resolved_duration_seconds": nova_properties.duration_seconds,
            "resolved_resolution": nova_properties.resolution,
            "resolved_fps": nova_properties.fps,
            "resolved_timeout_seconds": nova_properties.timeout_seconds,
            "resolved_poll_interval_seconds": nova_properties.poll_interval_seconds,
            "output_s3_uri": output_s3_uri,
        }

        def _execute_submit() -> AiApiObservedVideosResultModel[AIVideoGenerationJob]:
            payload: dict[str, Any] = self._execute_with_retries(
                operation=lambda: self.client.start_async_invoke(
                    modelId=self.video_model_name,
                    modelInput=model_input,
                    outputDataConfig={"s3OutputDataConfig": {"s3Uri": output_s3_uri}},
                ),
                trace_name="start_async_invoke",
            )
            job: AIVideoGenerationJob = self._normalize_job(
                payload,
                previous_provider_metadata=provider_metadata,
            )
            return AiApiObservedVideosResultModel(
                return_value=job,
                generated_video_count=nova_properties.num_videos,
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
        provider_job_id: str = self._extract_provider_job_id(job)
        previous_provider_metadata: dict[str, str | int | float | bool | None] = {}
        if isinstance(job, AIVideoGenerationJob):
            previous_provider_metadata = dict(job.provider_metadata)
        payload: dict[str, Any] = self._get_async_invoke_response(provider_job_id)
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
                "download_video_result requires a completed Nova Reel job."
            )
        output_s3_uri_value: str | int | float | bool | None = (
            current_job.provider_metadata.get("output_s3_uri")
        )
        if output_s3_uri_value in (None, ""):
            raise ValueError(
                "Nova Reel job metadata is missing the configured output S3 URI."
            )
        actual_output_s3_uri: str = self._find_output_video_s3_uri(
            output_s3_uri=str(output_s3_uri_value),
            provider_job_id=current_job.provider_job_id,
        )
        download_outputs: bool = bool(
            current_job.provider_metadata.get("download_outputs", True)
        )

        def _execute_download() -> (
            AiApiObservedVideosResultModel[AIVideoGenerationResult]
        ):
            artifacts: list[AIVideoArtifact] = []
            total_output_bytes: int = 0
            file_path: Path | None = None
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
                bucket_name, object_key = self._parse_s3_uri(actual_output_s3_uri)
                response: dict[str, Any] = self.s3_client.get_object(
                    Bucket=bucket_name,
                    Key=object_key,
                )
                video_bytes: bytes = response["Body"].read()
                total_output_bytes = len(video_bytes)
                sanitized_job_id: str = current_job.provider_job_id.rsplit("/", 1)[-1]
                file_path = output_dir / f"{sanitized_job_id}.mp4"
                file_path.write_bytes(video_bytes)
            artifacts.append(
                AIVideoArtifact(
                    mime_type="video/mp4",
                    file_path=file_path,
                    remote_uri=actual_output_s3_uri,
                    duration_seconds=(
                        int(current_job.provider_metadata["resolved_duration_seconds"])
                        if current_job.provider_metadata.get(
                            "resolved_duration_seconds"
                        )
                        is not None
                        else None
                    ),
                    fps=(
                        int(current_job.provider_metadata["resolved_fps"])
                        if current_job.provider_metadata.get("resolved_fps") is not None
                        else None
                    ),
                    provider_metadata={"provider_job_id": current_job.provider_job_id},
                )
            )
            result: AIVideoGenerationResult = AIVideoGenerationResult(
                job=current_job,
                artifacts=artifacts,
                provider_metadata={
                    "download_outputs": download_outputs,
                    "provider_job_id": current_job.provider_job_id,
                    "output_s3_uri": actual_output_s3_uri,
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
