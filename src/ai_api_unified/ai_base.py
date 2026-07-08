from __future__ import (
    annotations,
)  # Postpone evaluation of type hints to avoid circular imports and allow forward references with | None

import base64
import json
import logging
import math
import mimetypes
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping
from typing import Any, ClassVar, Generic, Iterator, Literal, NoReturn, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError, model_validator

from ai_api_unified.ai_completions_exceptions import (
    StructuredResponseTokenLimitError,
)
from ai_api_unified.ai_provider_exceptions import (
    AiProviderCapabilityUnsupportedError,
    AiProviderConfigurationError,
)
from ai_api_unified.middleware.observability import (
    AiApiObservabilityMiddleware,
    get_observability_middleware,
)
from ai_api_unified.middleware.observability_runtime import (
    AiApiCallContextModel,
    AiApiCallResultSummaryModel,
    OBSERVABILITY_DIRECTION_INPUT,
    ObservabilityMetadataValue,
    TOKEN_COUNT_SOURCE_NONE,
    TOKEN_COUNT_SOURCE_PROVIDER,
    execute_observed_call,
    execute_observed_streaming_call,
    get_observability_context,
    resolve_originating_caller,
)
from ai_api_unified.pricing.model_pricing import AIModelPricing
from ai_api_unified.middleware.pii_redactor import AiApiPiiMiddleware

_LOGGER: logging.Logger = logging.getLogger(__name__)
ProviderCallReturnType = TypeVar("ProviderCallReturnType")
CompletionsReturnType = TypeVar("CompletionsReturnType")
EmbeddingsReturnType = TypeVar("EmbeddingsReturnType")
ImagesReturnType = TypeVar("ImagesReturnType")
VideosReturnType = TypeVar("VideosReturnType")


class SupportedDataType(Enum):
    """Enumeration of data types supported by AI models."""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    PDF = "pdf"


class AICompletionsCapabilitiesBase(BaseModel):
    """
    Base class for capturing important attributes of completions models.
    """

    context_window_length: int
    knowledge_cutoff_date: date | None = None
    reasoning: bool = False
    supported_data_types: list[SupportedDataType] = [SupportedDataType.TEXT]
    supports_data_residency_constraint: bool = False
    supports_streaming: bool = False
    supports_token_counting: bool = False
    supports_batch: bool = False
    pricing: AIModelPricing | None = None


class AIIncludedMediaParamsBase(BaseModel):
    """
    Shared aligned-list media attachment fields and validation.

    Subclasses declare which media types they accept via
    DICT_ALLOWED_MIME_PREFIXES and an optional byte cap via MAX_MEDIA_BYTES
    (applied per attachment and to the combined payload; None disables it).
    """

    DICT_ALLOWED_MIME_PREFIXES: ClassVar[dict[SupportedDataType, tuple[str, ...]]] = {}
    MAX_MEDIA_BYTES: ClassVar[int | None] = None

    included_types: list[SupportedDataType] | None = None
    included_data: list[bytes] | None = None
    included_mime_types: list[str] | None = None

    @model_validator(mode="after")
    def _validate_and_normalize_included_media(
        self,
    ) -> "AIIncludedMediaParamsBase":
        """Ensure media attachment lists are aligned and metadata validated."""

        list_types: list[SupportedDataType] = list(self.included_types or [])
        list_data: list[bytes] = list(self.included_data or [])
        list_mime_types: list[str] = list(self.included_mime_types or [])

        if len({len(list_types), len(list_data), len(list_mime_types)}) > 1:
            raise ValueError(
                "included_types, included_data, and included_mime_types must be the same length."
            )

        if not list_types:
            self.included_data = None
            self.included_mime_types = None
            self.included_types = None
            return self

        # Loop through attachments so every media item is validated against its declared type.
        for index, media_type in enumerate(list_types):
            if media_type is SupportedDataType.TEXT:
                raise ValueError(
                    "Text belongs in the 'text' field, not in media attachments."
                )
            tuple_mime_prefixes: tuple[str, ...] | None = (
                self.DICT_ALLOWED_MIME_PREFIXES.get(media_type)
            )
            if tuple_mime_prefixes is None:
                raise ValueError(
                    f"SupportedDataType.{media_type.name} attachments are not accepted."
                )
            mime_type: str = (list_mime_types[index] or "").lower()
            if not mime_type:
                raise ValueError("Each included media item must specify a MIME type.")
            if not mime_type.startswith(tuple_mime_prefixes):
                raise ValueError(
                    f"MIME type {mime_type!r} does not match declared type "
                    f"SupportedDataType.{media_type.name}."
                )
            media_bytes: bytes = list_data[index]
            if not media_bytes:
                raise ValueError("Media attachment bytes cannot be empty.")
            if (
                self.MAX_MEDIA_BYTES is not None
                and len(media_bytes) > self.MAX_MEDIA_BYTES
            ):
                raise ValueError(
                    f"Media attachment at index {index} exceeds {self.MAX_MEDIA_BYTES} bytes."
                )

        if self.MAX_MEDIA_BYTES is not None:
            # Providers accept inline media per request, not per attachment, so
            # the combined payload is bounded by the same limit.
            int_total_media_bytes: int = sum(len(item) for item in list_data)
            if int_total_media_bytes > self.MAX_MEDIA_BYTES:
                raise ValueError(
                    f"Combined media attachments total {int_total_media_bytes} bytes, "
                    f"exceeding the {self.MAX_MEDIA_BYTES}-byte request limit."
                )

        self.included_data = list_data
        self.included_mime_types = list_mime_types
        self.included_types = list_types
        return self

    def iter_included_media(
        self,
    ) -> Iterator[tuple[int, SupportedDataType, bytes, str]]:
        """
        Yield the index, type, raw bytes, and MIME type for every attached media item.
        """

        list_types: list[SupportedDataType] = list(self.included_types or [])
        list_data: list[bytes] = list(self.included_data or [])
        list_mime_types: list[str] = list(self.included_mime_types or [])

        for index, (media_type, media_bytes, mime_type) in enumerate(
            zip(list_types, list_data, list_mime_types)
        ):
            yield index, media_type, media_bytes, mime_type

    @property
    def has_included_media(self) -> bool:
        """Return True when any media attachments are present."""

        return bool(self.included_types)


class AICompletionsPromptParamsBase(AIIncludedMediaParamsBase, ABC):
    """
    Base class for completion prompt parameters.
    This allows passing media attachments (images, etc.) alongside text prompts as well as
    specifying a system prompt to guide the model's behavior.
    Typical prompting doesn't require any use of this class. All fields are optional.

    Media validation (alignment, MIME/type agreement, size caps) comes from
    AIIncludedMediaParamsBase; completions currently accept image attachments only.
    """

    DEFAULT_SYSTEM_PROMPT: ClassVar[str] = "You are a helpful assistant."
    DEFAULT_STRICT_SCHEMA_SYSTEM_PROMPT: ClassVar[str] = (
        "Respond only with JSON following the provided schema."
    )
    MAX_IMAGE_BYTES: ClassVar[int] = 20_000_000

    DICT_ALLOWED_MIME_PREFIXES: ClassVar[dict[SupportedDataType, tuple[str, ...]]] = {
        SupportedDataType.IMAGE: ("image/",),
    }
    MAX_MEDIA_BYTES: ClassVar[int | None] = MAX_IMAGE_BYTES

    system_prompt: str | None = None


class AIBatchStatus(str, Enum):
    """Lifecycle status of a completions batch job.

    Normalized across providers. IN_PROGRESS and CANCELING are non-terminal;
    ENDED (results available), FAILED, EXPIRED, and CANCELED are terminal.
    """

    IN_PROGRESS = "in_progress"
    CANCELING = "canceling"
    ENDED = "ended"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELED = "canceled"


class AIBatchItemStatus(str, Enum):
    """Per-request outcome within an ended batch."""

    SUCCEEDED = "succeeded"
    ERRORED = "errored"
    CANCELED = "canceled"
    EXPIRED = "expired"


class AIBatchRequestItem(BaseModel):
    """One request in a completions batch.

    Text-prompt batch requests, mirroring send_prompt. custom_id must be unique
    within a batch and is echoed on the matching result so callers can correlate
    results back to requests (results arrive in arbitrary order).
    """

    custom_id: str
    prompt: str
    system_prompt: str | None = None
    max_response_tokens: int | None = None


class AIBatchJob(BaseModel):
    """Provider-agnostic handle for one submitted completions batch.

    provider_batch_id is the raw provider identifier; batch_id namespaces it by
    engine for stable cross-provider correlation.
    """

    batch_id: str
    provider_batch_id: str
    status: AIBatchStatus
    request_count: int | None = None
    succeeded_count: int | None = None
    errored_count: int | None = None
    canceled_count: int | None = None
    expired_count: int | None = None
    processing_count: int | None = None
    submitted_at_utc: datetime | None = None
    ended_at_utc: datetime | None = None
    provider_engine: str | None = None
    provider_model_name: str | None = None
    provider_metadata: dict[str, Any] = {}

    @property
    def is_terminal(self) -> bool:
        """Return True when the batch has stopped processing.

        IN_PROGRESS and CANCELING are non-terminal: a canceling batch is still
        winding down and its results are not yet retrievable.
        """
        return self.status not in (
            AIBatchStatus.IN_PROGRESS,
            AIBatchStatus.CANCELING,
        )


class AIBatchResultItem(BaseModel):
    """One request's outcome in an ended batch, keyed by custom_id."""

    custom_id: str
    status: AIBatchItemStatus
    text: str | None = None
    error_message: str | None = None
    provider_prompt_tokens: int | None = None
    provider_completion_tokens: int | None = None
    provider_metadata: dict[str, Any] = {}


class AIEmbeddingsCapabilitiesBase(BaseModel):
    """
    Base class for capturing important attributes of embeddings models.

    supported_data_types and max_images_per_request are enforced client-side
    before provider calls. The remaining limit fields (max_input_tokens,
    max_batch_size, max_video_seconds, max_audio_seconds) are advisory
    descriptors of provider-documented limits; exceeding them surfaces as a
    provider-side error.
    """

    supported_data_types: list[SupportedDataType] = [SupportedDataType.TEXT]
    default_dimensions: int
    min_dimensions: int | None = None
    max_dimensions: int | None = None
    recommended_dimensions: list[int] = []
    max_input_tokens: int | None = None
    max_batch_size: int | None = None
    max_images_per_request: int | None = None
    max_video_seconds: int | None = None
    max_audio_seconds: int | None = None
    pricing: AIModelPricing | None = None


# Maps each embeddable media type to the MIME prefixes accepted for it.
DICT_EMBEDDINGS_MEDIA_MIME_PREFIXES: dict[SupportedDataType, tuple[str, ...]] = {
    SupportedDataType.IMAGE: ("image/",),
    SupportedDataType.VIDEO: ("video/",),
    SupportedDataType.AUDIO: ("audio/",),
    SupportedDataType.PDF: ("application/pdf",),
}


class AIEmbeddingsMultimodalParams(AIIncludedMediaParamsBase):
    """
    Input parameters for one multimodal embedding request.

    Media validation (alignment, MIME/type agreement, size caps) comes from
    AIIncludedMediaParamsBase. All supplied inputs are embedded jointly as one
    interleaved input producing one embedding vector.
    """

    DICT_ALLOWED_MIME_PREFIXES: ClassVar[dict[SupportedDataType, tuple[str, ...]]] = (
        DICT_EMBEDDINGS_MEDIA_MIME_PREFIXES
    )
    MAX_MEDIA_BYTES: ClassVar[int | None] = 20_000_000

    text: str | None = None

    @model_validator(mode="after")
    def _require_text_or_media(self) -> "AIEmbeddingsMultimodalParams":
        """Reject requests that carry neither text nor media attachments."""

        if not self.has_included_media and (self.text is None or not self.text.strip()):
            raise ValueError(
                "Multimodal embeddings input requires text, media attachments, or both."
            )
        return self


class AIBase(ABC):
    """
    Abstract base class that defines methods for interacting with OpenAI
    or any large language model service.
    """

    CLIENT_TYPE_EMBEDDING = "embedding"
    CLIENT_TYPE_COMPLETIONS = "completions"
    CLIENT_TYPE_IMAGES = "images"
    CLIENT_TYPE_VIDEOS = "videos"
    PROVIDER_VENDOR_OPENAI = "openai"
    PROVIDER_VENDOR_BEDROCK = "bedrock"
    PROVIDER_VENDOR_GOOGLE = "google"
    PROVIDER_VENDOR_AZURE = "azure"
    PROVIDER_VENDOR_ELEVENLABS = "elevenlabs"
    PROVIDER_VENDOR_ANTHROPIC = "anthropic"
    PROVIDER_ENGINE_GOOGLE_GEMINI = "google-gemini"
    PROVIDER_ENGINE_CLAUDE = "claude"
    _observability_middleware: AiApiObservabilityMiddleware | None = None

    def __init__(self, model: str | None = None, **kwargs):
        super().__init__()
        self.model: str | None = model

    @property
    def model_name(self) -> str | None:
        """
        Identifier of the model in use (e.g. 'gpt-4o-mini').
        """
        return self.model

    @property
    @abstractmethod
    def list_model_names(self) -> list[str]:
        """Supported model identifiers for this client."""
        ...

    def _get_observability_middleware(self) -> AiApiObservabilityMiddleware:
        """
        Returns the effective observability middleware instance for this client.

        Args:
            None

        Returns:
            Effective observability middleware instance, lazily initialized when first needed.
        """
        if self._observability_middleware is None:
            self._observability_middleware = get_observability_middleware()
        # Normal return with the lazily initialized observability middleware instance.
        return self._observability_middleware

    def _get_explicit_observability_caller_id(self) -> str | None:
        """
        Returns the request-scoped caller identifier only when application code set it explicitly.

        Args:
            None

        Returns:
            Explicit application caller identifier, or None when the current request context
            did not set caller correlation data.
        """
        observability_context = get_observability_context()
        # Normal return with the explicit request-scoped caller identifier when present.
        return observability_context.caller_id

    def _build_observability_call_context(
        self,
        *,
        capability: str,
        operation: str,
        dict_metadata: dict[str, ObservabilityMetadataValue] | None = None,
        legacy_caller_id: str | None = None,
    ) -> AiApiCallContextModel:
        """
        Builds the immutable shared call-context object for one provider-boundary event sequence.

        Args:
            capability: Capability label for the public API surface being invoked.
            operation: Public operation name such as `send_prompt` or `generate_embeddings`.
            dict_metadata: Optional scalar metadata describing the request side of the call.
            legacy_caller_id: Optional explicit legacy caller hint supplied by existing config.

        Returns:
            AiApiCallContextModel containing shared metadata for input, output, and error events.
        """
        observability_context = get_observability_context()
        resolved_caller_id, caller_id_source = resolve_originating_caller(
            legacy_caller_id=legacy_caller_id
        )
        dict_context_metadata: dict[str, ObservabilityMetadataValue] = dict(
            dict_metadata or {}
        )
        if observability_context.session_id is not None:
            dict_context_metadata["session_id"] = observability_context.session_id
        if observability_context.workflow_id is not None:
            dict_context_metadata["workflow_id"] = observability_context.workflow_id
        model_name: str | None = self._resolve_observability_model_name()
        model_version: str | None = self._resolve_observability_model_version(
            model_name=model_name
        )
        ai_api_call_context: AiApiCallContextModel = AiApiCallContextModel(
            call_id=str(uuid.uuid4()),
            event_time_utc=self._get_observability_event_time_utc(),
            capability=capability,
            operation=operation,
            provider_vendor=self._resolve_observability_provider_vendor(),
            provider_engine=self._resolve_observability_provider_engine(),
            model_name=model_name,
            model_version=model_version,
            direction=OBSERVABILITY_DIRECTION_INPUT,
            originating_caller_id=resolved_caller_id,
            originating_caller_id_source=caller_id_source,
            dict_metadata=dict_context_metadata,
        )
        # Normal return with immutable provider-boundary call metadata.
        return ai_api_call_context

    def _execute_provider_call_with_observability(
        self,
        *,
        capability: str,
        operation: str,
        dict_input_metadata: dict[str, ObservabilityMetadataValue] | None,
        callable_execute: Callable[[], ProviderCallReturnType],
        callable_build_result_summary: Callable[
            [ProviderCallReturnType, float], AiApiCallResultSummaryModel
        ],
        legacy_caller_id: str | None = None,
    ) -> ProviderCallReturnType:
        """
        Wraps one provider call with shared observability lifecycle helpers when enabled.

        Args:
            capability: Capability label for the public API surface being invoked.
            operation: Public operation name such as `send_prompt` or `generate_embeddings`.
            dict_input_metadata: Optional scalar request metadata safe for input-event logs.
            callable_execute: Zero-argument callable that performs the provider call.
            callable_build_result_summary: Callable that summarizes provider output and elapsed time.
            legacy_caller_id: Optional explicit legacy caller hint supplied by existing config.

        Returns:
            Original provider return value from `callable_execute`.
        """
        observability_middleware: AiApiObservabilityMiddleware = (
            self._get_observability_middleware()
        )
        provider_result: ProviderCallReturnType = execute_observed_call(
            observability_middleware=observability_middleware,
            callable_build_call_context=lambda: self._build_observability_call_context(
                capability=capability,
                operation=operation,
                dict_metadata=dict_input_metadata,
                legacy_caller_id=legacy_caller_id,
            ),
            callable_execute=callable_execute,
            callable_build_result_summary=callable_build_result_summary,
        )
        # Normal return with the original provider result after optional observability wrapping.
        return provider_result

    def _resolve_observability_provider_vendor(self) -> str:
        """
        Resolves a best-effort provider vendor label for shared observability metadata.

        Args:
            None

        Returns:
            Best-effort provider vendor label derived from the concrete client module or class.
        """
        lower_module_name: str = self.__class__.__module__.lower()
        if "anthropic" in lower_module_name:
            # Early return for native Anthropic API clients.
            return self.PROVIDER_VENDOR_ANTHROPIC
        if "openai" in lower_module_name:
            # Early return for OpenAI-backed clients.
            return self.PROVIDER_VENDOR_OPENAI
        if "bedrock" in lower_module_name:
            # Early return for Bedrock-backed clients.
            return self.PROVIDER_VENDOR_BEDROCK
        if "google" in lower_module_name or "gemini" in lower_module_name:
            # Early return for Google-backed clients.
            return self.PROVIDER_VENDOR_GOOGLE
        if "azure" in lower_module_name:
            # Early return for Azure-backed clients.
            return self.PROVIDER_VENDOR_AZURE
        if "elevenlabs" in lower_module_name:
            # Early return for ElevenLabs-backed clients.
            return self.PROVIDER_VENDOR_ELEVENLABS
        # Normal return with a stable lower-case class-name fallback.
        return self.__class__.__name__.lower()

    def _resolve_observability_provider_engine(self) -> str:
        """
        Resolves a best-effort provider engine label for shared observability metadata.

        Args:
            None

        Returns:
            Best-effort provider engine label derived from the concrete client module or class.
        """
        lower_module_name: str = self.__class__.__module__.lower()
        if "anthropic" in lower_module_name:
            # Early return for native Anthropic API clients (the claude engine).
            return self.PROVIDER_ENGINE_CLAUDE
        if "google_gemini" in lower_module_name or "gemini" in lower_module_name:
            # Early return for Gemini-backed clients.
            return self.PROVIDER_ENGINE_GOOGLE_GEMINI
        if "openai" in lower_module_name:
            # Early return for OpenAI-backed clients.
            return self.PROVIDER_VENDOR_OPENAI
        if "bedrock" in lower_module_name:
            # Early return for Bedrock-backed clients.
            return self.PROVIDER_VENDOR_BEDROCK
        if "azure" in lower_module_name:
            # Early return for Azure-backed clients.
            return self.PROVIDER_VENDOR_AZURE
        if "elevenlabs" in lower_module_name:
            # Early return for ElevenLabs-backed clients.
            return self.PROVIDER_VENDOR_ELEVENLABS
        # Normal return with a stable lower-case class-name fallback.
        return self.__class__.__name__.lower()

    def _resolve_observability_model_name(self) -> str | None:
        """
        Resolves a best-effort model identifier from the concrete provider client.

        Args:
            None

        Returns:
            Best-effort model identifier string, or None when the client does not expose one.
        """
        model_name_candidate = self.model_name
        if callable(model_name_candidate):
            resolved_model_name: str | None = model_name_candidate()
            # Normal return with a resolved callable model identifier.
            return resolved_model_name
        # Normal return with the property-based model identifier.
        return model_name_candidate

    def _resolve_observability_model_version(
        self,
        *,
        model_name: str | None,
    ) -> str | None:
        """
        Resolves a best-effort model-version identifier for shared observability metadata.

        Args:
            model_name: Best-effort resolved model identifier for the current client.

        Returns:
            Best-effort model-version identifier, defaulting to the resolved model name.
        """
        # Normal return because model version currently defaults to the resolved model name.
        return model_name

    def _get_observability_event_time_utc(self) -> datetime:
        """
        Returns the UTC timestamp used when building shared observability call-context objects.

        Args:
            None

        Returns:
            Current UTC datetime object used for the input-side call-context event time.
        """
        event_time_utc: datetime = datetime.now(timezone.utc)
        # Normal return with the current UTC event timestamp.
        return event_time_utc


@dataclass(frozen=True)
class AiApiObservedCompletionsResultModel(Generic[CompletionsReturnType]):
    """
    Stores one completions-provider result alongside metadata needed for observability emission.

    Args:
        return_value: Caller-facing return value produced by the provider path.
        raw_output_text: Raw provider output text before output-side middleware transformation.
        finish_reason: Optional provider finish reason for the final observed response.
        provider_prompt_tokens: Optional provider-reported prompt/input token count.
            Includes any cached-input tokens (cache reads are a subset of this
            count); providers whose SDK reports cache reads separately are
            normalized to this convention in their result-summary builders.
        provider_completion_tokens: Optional provider-reported completion token count.
        provider_cached_input_tokens: Optional provider-reported cached-input
            token count (cache reads billed at the cached rate), a subset of
            provider_prompt_tokens. None when the provider does not report it.
        provider_total_tokens: Optional provider-reported total token count.
        dict_metadata: Additional metadata derived from the provider path.

    Returns:
        Immutable container used to build metadata-only observability output summaries.
    """

    return_value: CompletionsReturnType
    raw_output_text: str
    finish_reason: str | None = None
    provider_prompt_tokens: int | None = None
    provider_completion_tokens: int | None = None
    provider_cached_input_tokens: int | None = None
    provider_total_tokens: int | None = None
    dict_metadata: dict[str, ObservabilityMetadataValue] = field(default_factory=dict)


@dataclass(frozen=True)
class AiApiObservedEmbeddingsResultModel(Generic[EmbeddingsReturnType]):
    """
    Stores one embeddings-provider result alongside metadata needed for observability emission.

    Args:
        return_value: Caller-facing single or batch embedding payload produced by the provider path.
        embedding_count: Number of vectors returned by the provider call.
        returned_dimensions: Embedding width returned by the provider response when known.
        provider_input_tokens: Optional provider-reported input token count.
        provider_total_tokens: Optional provider-reported total token count.
        dict_metadata: Additional metadata derived from the provider path.

    Returns:
        Immutable container used to build metadata-only observability output summaries.
    """

    return_value: EmbeddingsReturnType
    embedding_count: int
    returned_dimensions: int | None = None
    provider_input_tokens: int | None = None
    provider_total_tokens: int | None = None
    dict_metadata: Mapping[str, ObservabilityMetadataValue] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """
        Freezes caller-supplied metadata so the observed embeddings result remains effectively immutable.

        Args:
            None

        Returns:
            None after metadata has been copied and wrapped in an immutable mapping.
        """
        frozen_metadata: Mapping[str, ObservabilityMetadataValue] = MappingProxyType(
            dict(self.dict_metadata)
        )
        object.__setattr__(self, "dict_metadata", frozen_metadata)
        # Normal return after replacing metadata with an immutable mapping copy.
        return None


@dataclass(frozen=True)
class AiApiObservedImagesResultModel(Generic[ImagesReturnType]):
    """
    Stores one image-provider result alongside metadata needed for observability emission.

    Args:
        return_value: Caller-facing image payload produced by the provider path.
        generated_image_count: Number of images returned by the provider call.
        total_output_bytes: Total number of bytes across all generated image payloads.
        provider_input_tokens: Optional provider-reported input token count.
        provider_total_tokens: Optional provider-reported total token count.
        dict_metadata: Additional metadata derived from the provider path.

    Returns:
        Immutable container used to build metadata-only observability output summaries.
    """

    return_value: ImagesReturnType
    generated_image_count: int
    total_output_bytes: int
    provider_input_tokens: int | None = None
    provider_total_tokens: int | None = None
    dict_metadata: Mapping[str, ObservabilityMetadataValue] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """
        Freezes caller-supplied metadata so the observed image result remains effectively immutable.

        Args:
            None

        Returns:
            None after metadata has been copied and wrapped in an immutable mapping.
        """
        frozen_metadata: Mapping[str, ObservabilityMetadataValue] = MappingProxyType(
            dict(self.dict_metadata)
        )
        object.__setattr__(self, "dict_metadata", frozen_metadata)
        # Normal return after replacing metadata with an immutable mapping copy.
        return None


class AIMediaReference(BaseModel):
    """
    Provider-agnostic reference to one media input used by multimodal APIs.
    """

    bytes_data: bytes | None = None
    file_path: Path | None = None
    remote_uri: str | None = None
    mime_type: str | None = None

    @model_validator(mode="after")
    def _validate_reference_source(self) -> "AIMediaReference":
        """Ensure exactly one source is configured and infer the MIME type when possible."""

        list_configured_sources: list[str] = [
            source_name
            for source_name, source_value in (
                ("bytes_data", self.bytes_data),
                ("file_path", self.file_path),
                ("remote_uri", self.remote_uri),
            )
            if source_value is not None
        ]
        if len(list_configured_sources) != 1:
            raise ValueError(
                "AIMediaReference requires exactly one of bytes_data, file_path, or remote_uri."
            )
        if self.file_path is not None and not self.file_path.exists():
            raise ValueError(
                f"AIMediaReference.file_path does not exist: {self.file_path}"
            )
        if self.mime_type is None and self.file_path is not None:
            inferred_mime_type, _ = mimetypes.guess_type(str(self.file_path))
            if inferred_mime_type is not None:
                self.mime_type = inferred_mime_type
        return self

    def read_bytes(self) -> bytes:
        """
        Materialize the reference into raw bytes when it is locally available.
        """

        if self.bytes_data is not None:
            return self.bytes_data
        if self.file_path is not None:
            return self.file_path.read_bytes()
        raise ValueError(
            "AIMediaReference cannot read bytes from a remote_uri without a provider-specific fetch path."
        )

    def to_data_url(self) -> str:
        """
        Convert the reference into a data URL or remote URL suitable for provider APIs.
        """

        if self.remote_uri is not None:
            return self.remote_uri
        mime_type: str | None = self.mime_type
        if mime_type is None or mime_type.strip() == "":
            raise ValueError(
                "AIMediaReference.mime_type is required when converting local media into a data URL."
            )
        media_bytes: bytes = self.read_bytes()
        encoded_media: str = base64.b64encode(media_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded_media}"


class AIVideoGenerationStatus(str, Enum):
    """Normalized lifecycle states for video-generation jobs."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AIVideoArtifact(BaseModel):
    """
    Normalized generated-video artifact.
    """

    mime_type: str = "video/mp4"
    file_path: Path | None = None
    remote_uri: str | None = None
    width: int | None = None
    height: int | None = None
    duration_seconds: int | None = None
    fps: int | None = None
    has_audio: bool | None = None
    provider_metadata: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict
    )

    def read_bytes(self) -> bytes:
        """
        Read the artifact bytes from the materialized local file.
        """

        if self.file_path is None:
            raise ValueError(
                "AIVideoArtifact.read_bytes() requires a local file_path. Generate with download_outputs=True to materialize the artifact."
            )
        return self.file_path.read_bytes()


class AIVideoGenerationJob(BaseModel):
    """
    Normalized video-generation job state.
    """

    job_id: str
    provider_job_id: str
    status: AIVideoGenerationStatus
    progress_percent: float | None = None
    submitted_at_utc: datetime | None = None
    completed_at_utc: datetime | None = None
    error_message: str | None = None
    provider_engine: str
    provider_model_name: str | None = None
    provider_metadata: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict
    )


class AIVideoGenerationResult(BaseModel):
    """
    Normalized completed video-generation result.
    """

    job: AIVideoGenerationJob
    artifacts: list[AIVideoArtifact]
    provider_metadata: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict
    )


class AIBaseVideoProperties(BaseModel):
    """
    Portable video-generation request properties shared across providers.
    """

    duration_seconds: int | None = None
    aspect_ratio: str | None = None
    resolution: str | None = None
    fps: int | None = None
    num_videos: int = 1
    seed: int | None = None
    output_format: str = "mp4"
    poll_interval_seconds: int = 10
    timeout_seconds: int = 900
    output_dir: Path | None = None
    download_outputs: bool = True

    @model_validator(mode="after")
    def _validate_video_properties(self) -> "AIBaseVideoProperties":
        """Validate shared video-generation property constraints."""

        if self.duration_seconds is not None and self.duration_seconds <= 0:
            raise ValueError(
                "AIBaseVideoProperties.duration_seconds must be a positive integer when provided."
            )
        if self.fps is not None and self.fps <= 0:
            raise ValueError(
                "AIBaseVideoProperties.fps must be a positive integer when provided."
            )
        if self.num_videos <= 0:
            raise ValueError(
                "AIBaseVideoProperties.num_videos must be greater than zero."
            )
        if self.poll_interval_seconds <= 0:
            raise ValueError(
                "AIBaseVideoProperties.poll_interval_seconds must be greater than zero."
            )
        if self.timeout_seconds <= 0:
            raise ValueError(
                "AIBaseVideoProperties.timeout_seconds must be greater than zero."
            )
        if self.output_format.strip().lower() != "mp4":
            raise ValueError(
                "AIBaseVideoProperties.output_format must currently be 'mp4'."
            )
        self.output_format = self.output_format.strip().lower()
        if self.aspect_ratio is not None:
            self.aspect_ratio = self.aspect_ratio.strip()
        if self.resolution is not None:
            self.resolution = self.resolution.strip().lower()
        return self


@dataclass(frozen=True)
class AiApiObservedVideosResultModel(Generic[VideosReturnType]):
    """
    Stores one video-provider result alongside metadata needed for observability emission.
    """

    return_value: VideosReturnType
    generated_video_count: int
    total_output_bytes: int = 0
    provider_input_tokens: int | None = None
    provider_total_tokens: int | None = None
    dict_metadata: Mapping[str, ObservabilityMetadataValue] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """
        Freezes caller-supplied metadata so the observed video result remains effectively immutable.
        """

        frozen_metadata: Mapping[str, ObservabilityMetadataValue] = MappingProxyType(
            dict(self.dict_metadata)
        )
        object.__setattr__(self, "dict_metadata", frozen_metadata)
        return None


class AIBaseEmbeddings(AIBase):
    """
    Abstract base class for generating embeddings.
    """

    def __init__(self, model: str | None = None, dimensions: int = 0):
        super().__init__(model=model)
        self.dimensions = dimensions

    def _build_embeddings_observability_input_metadata(
        self,
        *,
        list_texts: list[str],
        bool_is_batch: bool,
        requested_dimensions: int | None,
    ) -> dict[str, ObservabilityMetadataValue]:
        """
        Builds metadata-only input fields for one embeddings provider call.

        Args:
            list_texts: Sanitized input texts that will be embedded by the provider.
            bool_is_batch: True when the public API call embeds multiple texts.
            requested_dimensions: Requested output dimensions when the provider exposes that input.

        Returns:
            Dictionary of metadata-only input fields safe for observability logging.
        """
        int_total_input_chars: int = sum(len(text) for text in list_texts)
        int_max_input_chars: int = max((len(text) for text in list_texts), default=0)
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = {
            "input_text_count": len(list_texts),
            "input_text_total_chars": int_total_input_chars,
            "input_text_max_chars": int_max_input_chars,
            "batch_mode": bool_is_batch,
            "requested_dimensions": requested_dimensions,
        }
        # Normal return with embeddings input metadata only.
        return dict_input_metadata

    def _build_embeddings_observability_result_summary(
        self,
        *,
        observed_result: AiApiObservedEmbeddingsResultModel[Any],
        provider_elapsed_ms: float,
    ) -> AiApiCallResultSummaryModel:
        """
        Builds metadata-only output fields for one observed embeddings provider result.

        Args:
            observed_result: Raw provider result container returned by the wrapped call.
            provider_elapsed_ms: Measured elapsed milliseconds for the wrapped provider path.

        Returns:
            AiApiCallResultSummaryModel containing output metadata safe for observability logging.
        """
        input_token_count: int | None = observed_result.provider_input_tokens
        input_token_count_source: str = (
            TOKEN_COUNT_SOURCE_PROVIDER
            if input_token_count is not None
            else TOKEN_COUNT_SOURCE_NONE
        )
        call_result_summary: AiApiCallResultSummaryModel = AiApiCallResultSummaryModel(
            provider_elapsed_ms=provider_elapsed_ms,
            input_token_count=input_token_count,
            input_token_count_source=input_token_count_source,
            output_token_count=None,
            output_token_count_source=TOKEN_COUNT_SOURCE_NONE,
            provider_prompt_tokens=observed_result.provider_input_tokens,
            provider_completion_tokens=None,
            provider_total_tokens=observed_result.provider_total_tokens,
            finish_reason=None,
            dict_metadata={
                "embedding_count": observed_result.embedding_count,
                "returned_dimensions": observed_result.returned_dimensions,
                **observed_result.dict_metadata,
            },
        )
        # Normal return with embeddings output summary metadata derived from the provider result.
        return call_result_summary

    @property
    def capabilities(self) -> AIEmbeddingsCapabilitiesBase:
        """
        Returns the capabilities descriptor for the configured embeddings model.

        Providers override this with model-specific capabilities. The base
        default describes a text-only embeddings model at the configured
        dimensions.
        """
        # Normal return with text-only default capabilities.
        return AIEmbeddingsCapabilitiesBase(default_dimensions=self.dimensions)

    def compute_embedding_cost(self, *, input_tokens: int) -> float:
        """
        Return the USD cost for measured embedding input tokens.

        Uses `capabilities.pricing` (per-1M input rate). Embeddings are input
        only.

        Args:
            input_tokens: Provider-reported input tokens embedded.

        Returns:
            USD cost as a float; 0.0 when the model has no token pricing.
        """
        pricing: AIModelPricing | None = self.capabilities.pricing
        if pricing is None or pricing.token_rates is None:
            # Early return because this model has no token pricing on record.
            return 0.0
        # Normal return with the computed input-token cost.
        return float(pricing.compute_token_cost(input_tokens=input_tokens))

    def _raise_capability_unsupported(self, str_requested_input: str) -> NoReturn:
        """
        Raises the canonical capability error for one unsupported embeddings input.

        Args:
            str_requested_input: Human-readable description of the unsupported input.

        Raises:
            AiProviderCapabilityUnsupportedError: Always.
        """
        raise AiProviderCapabilityUnsupportedError(
            f"{type(self).__name__} model '{self.model_name}' does not support "
            f"{str_requested_input}. Supported input types: "
            f"{[t.value for t in self.capabilities.supported_data_types]}. "
            "Configure an embedding model that supports the requested input types."
        )

    def _ensure_multimodal_params_supported(
        self,
        params: AIEmbeddingsMultimodalParams,
    ) -> None:
        """
        Validates one multimodal input against the model's capabilities descriptor.

        Args:
            params: Multimodal embeddings input to validate.

        Raises:
            AiProviderCapabilityUnsupportedError: When the model supports no
                non-text input at all, an attached media type is unsupported,
                or a per-modality limit is exceeded.
        """
        embeddings_capabilities: AIEmbeddingsCapabilitiesBase = self.capabilities
        bool_supports_any_media: bool = any(
            data_type is not SupportedDataType.TEXT
            for data_type in embeddings_capabilities.supported_data_types
        )
        if not bool_supports_any_media:
            # Text-only models reject multimodal calls outright, even for
            # text-only params — generate_embeddings is the text path.
            self._raise_capability_unsupported("multimodal embeddings")
        int_image_count: int = 0
        # Loop through attachments so every declared media type is capability-checked.
        for _, media_type, _, _ in params.iter_included_media():
            if media_type not in embeddings_capabilities.supported_data_types:
                self._raise_capability_unsupported(
                    f"{media_type.value} input for embeddings"
                )
            if media_type is SupportedDataType.IMAGE:
                int_image_count += 1
        if (
            embeddings_capabilities.max_images_per_request is not None
            and int_image_count > embeddings_capabilities.max_images_per_request
        ):
            raise AiProviderCapabilityUnsupportedError(
                f"{type(self).__name__} model '{self.model_name}' accepts at most "
                f"{embeddings_capabilities.max_images_per_request} images per request "
                f"but received {int_image_count}."
            )
        # Normal return when every attachment is supported by the model capabilities.
        return None

    def _build_multimodal_embeddings_observability_input_metadata(
        self,
        *,
        params: AIEmbeddingsMultimodalParams,
        requested_dimensions: int | None,
    ) -> dict[str, ObservabilityMetadataValue]:
        """
        Builds metadata-only input fields for one multimodal embeddings provider call.

        Args:
            params: Multimodal embeddings input being embedded by the provider.
            requested_dimensions: Requested output dimensions when the provider exposes that input.

        Returns:
            Dictionary of metadata-only input fields safe for observability logging.
        """
        dict_modality_counts: dict[str, int] = {}
        int_total_media_bytes: int = 0
        # Loop through attachments so metadata reflects per-modality counts without content.
        for _, media_type, media_bytes, _ in params.iter_included_media():
            str_count_key: str = f"input_{media_type.value}_count"
            dict_modality_counts[str_count_key] = (
                dict_modality_counts.get(str_count_key, 0) + 1
            )
            int_total_media_bytes += len(media_bytes)
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = {
            "input_text_chars": len(params.text) if params.text else 0,
            "input_media_count": len(params.included_types or []),
            "input_media_total_bytes": int_total_media_bytes,
            "requested_dimensions": requested_dimensions,
            **dict_modality_counts,
        }
        # Normal return with multimodal embeddings input metadata only.
        return dict_input_metadata

    @abstractmethod
    def generate_embeddings(self, text: str) -> dict[str, Any]:
        """
        Generates embeddings for a single piece of text.
        """

    @abstractmethod
    def generate_embeddings_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """
        Generates embeddings for multiple pieces of text in a single API call.
        """

    def generate_embeddings_multimodal(
        self,
        params: AIEmbeddingsMultimodalParams,
    ) -> dict[str, Any]:
        """
        Generates one embedding for interleaved multimodal input.

        Template method: capability gating happens here so providers cannot
        skip it. Providers whose capabilities include non-text data types
        implement _generate_embeddings_multimodal_provider.

        Args:
            params: Multimodal embeddings input (text and/or media attachments).

        Raises:
            AiProviderCapabilityUnsupportedError: When the configured model does
                not support multimodal input or an attachment is unsupported.
        """
        self._ensure_multimodal_params_supported(params)
        # Normal return with the provider-implemented multimodal embedding payload.
        return self._generate_embeddings_multimodal_provider(params)

    def _generate_embeddings_multimodal_provider(
        self,
        params: AIEmbeddingsMultimodalParams,
    ) -> dict[str, Any]:
        """
        Provider hook for multimodal embedding generation.

        The base implementation raises: a provider whose capabilities declare
        non-text support must implement this hook.

        Args:
            params: Capability-validated multimodal embeddings input.

        Raises:
            AiProviderCapabilityUnsupportedError: Always, in the base class.
        """
        self._raise_capability_unsupported("multimodal embeddings")


class AIStructuredPrompt(BaseModel):
    """
    Base class for all structured prompts.
    This class is used to define the structure of the results returned by the AI model.

    """

    prompt: str = ""  # This is automatically populated after validation

    @model_validator(mode="after")
    def _populate_prompt(self: "AIStructuredPrompt", __: Any) -> "AIStructuredPrompt":
        """
        After validation, build and store the prompt string
        """
        object.__setattr__(
            self,
            "prompt",
            self.get_prompt(),
        )
        return self

    @classmethod
    def model_json_schema(cls) -> dict:
        from copy import deepcopy

        schema = deepcopy(super().model_json_schema())
        schema.setdefault("required", [])
        return schema

    def __str__(self):
        # Dump only the fields you actually care about (skip None/defaults)
        return self.model_dump_json(
            exclude_none=True,
            exclude_defaults=True,
        )

    @staticmethod
    @abstractmethod
    def get_prompt() -> str | None:
        """
        Optional method that subclasses can override to produce
        a “prompt string” given the model’s fields.

        """
        ...

    def send_structured_prompt(
        self,
        ai_client: AIBaseCompletions,
        response_model: Type[AIStructuredPrompt] | None = None,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> AIStructuredPrompt | None:
        """
        Execute the specific AIStructuredPrompt structured prompt and return the result as a structured object.

        Args:
            ai_client: The completions client used to execute the structured prompt.
            response_model: The expected structured response model.
            other_params: Optional provider-specific parameters, including system prompt overrides.
        """
        if self.prompt is None:
            raise ValueError(
                "You must provide a prompt string to send_structured_prompt(). "
                "This is done by calling the classmethod get_prompt() on the subclass."
            )
        if response_model is None:
            raise ValueError(
                "You must provide a response_model to send_structured_prompt(). "
                "This is done by passing the class itself, e.g. a non-abstract subclass of AIStructuredPrompt."
            )
        try:
            return ai_client.strict_schema_prompt(
                prompt=self.prompt,
                response_model=response_model,
                other_params=other_params,
            )
        except ValidationError as ve:
            _LOGGER.warning(
                "Validation errors sending structured prompt: %s",
                ve.errors(),
            )
            # either return None or raise a more descriptive error:
            return None

        except Exception as exc:
            # any other unexpected error
            _LOGGER.exception(
                "Unexpected error sending structured prompt: %s",
                exc,
            )
            return None


class AIBaseCompletions(AIBase):
    """
    Base class for generating text completions.
    """

    STRUCTURED_DEFAULT_MAX_RESPONSE_TOKENS: ClassVar[int] = 2048
    STRUCTURED_ENFORCED_MIN_MAX_RESPONSE_TOKENS: ClassVar[int] = 2048
    STRUCTURED_RECOMMENDED_MIN_MAX_RESPONSE_TOKENS: ClassVar[int] = 2048
    STRUCTURED_TOKEN_LIMIT_FAILURE_MODE_PREFLIGHT: ClassVar[str] = "preflight"
    STRUCTURED_TOKEN_LIMIT_FAILURE_MODE_PROVIDER_TRUNCATION: ClassVar[str] = (
        "provider_truncation"
    )
    RESPONSE_MODE_TEXT: ClassVar[str] = "text"
    RESPONSE_MODE_STRUCTURED: ClassVar[str] = "structured"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pii_middleware = AiApiPiiMiddleware()

    def _build_structured_token_limit_error_message(
        self,
        *,
        provider_name: str,
        model_name: str,
        max_response_tokens: int,
        failure_mode: Literal[
            "preflight",
            "provider_truncation",
        ],
        finish_reason: str | None = None,
        raw_output_text: str | None = None,
    ) -> str:
        """
        Build one high-signal error message for structured token-limit failures.

        Args:
            provider_name: Provider or engine name surfaced to the caller.
            model_name: Concrete provider model identifier used for the call.
            max_response_tokens: Structured response token limit requested by the caller.
            failure_mode: Failure category label describing validation or provider truncation.
            finish_reason: Optional provider finish reason associated with truncation.
            raw_output_text: Optional raw output text captured before the failure was raised.

        Returns:
            Human-readable message explaining the failure and next-step remediation.
        """
        if failure_mode == self.STRUCTURED_TOKEN_LIMIT_FAILURE_MODE_PREFLIGHT:
            # Normal return with the fail-fast validation message for undersized limits.
            return (
                "Structured response token limit is too small for "
                f"provider={provider_name} model={model_name}. "
                f"`max_response_tokens={max_response_tokens}` must be at least "
                f"{self.STRUCTURED_ENFORCED_MIN_MAX_RESPONSE_TOKENS} for "
                "`strict_schema_prompt()`. Increase `max_response_tokens` or omit it "
                "to use the library default. For best reliability, use at least "
                f"{self.STRUCTURED_RECOMMENDED_MIN_MAX_RESPONSE_TOKENS}."
            )

        int_raw_output_char_count: int = len(raw_output_text or "")
        str_finish_reason_fragment: str = ""
        if finish_reason is not None:
            str_finish_reason_fragment = f" finish_reason={finish_reason}."
        # Normal return with the provider truncation message for structured output.
        return (
            "Structured response generation was truncated for "
            f"provider={provider_name} model={model_name} because "
            f"`max_response_tokens={max_response_tokens}` was too small."
            f"{str_finish_reason_fragment} raw_output_char_count="
            f"{int_raw_output_char_count}. Increase `max_response_tokens` and retry "
            "in client code."
        )

    def _validate_structured_max_response_tokens(
        self,
        *,
        provider_name: str,
        model_name: str,
        max_response_tokens: int,
    ) -> None:
        """
        Fail fast when a structured token cap is below the library-supported floor.

        Args:
            provider_name: Provider or engine name surfaced to the caller.
            model_name: Concrete provider model identifier used for the call.
            max_response_tokens: Structured response token limit requested by the caller.

        Returns:
            None when the supplied token limit passes validation.
        """
        if max_response_tokens < self.STRUCTURED_ENFORCED_MIN_MAX_RESPONSE_TOKENS:
            raise StructuredResponseTokenLimitError(
                message=self._build_structured_token_limit_error_message(
                    provider_name=provider_name,
                    model_name=model_name,
                    max_response_tokens=max_response_tokens,
                    failure_mode=self.STRUCTURED_TOKEN_LIMIT_FAILURE_MODE_PREFLIGHT,
                ),
                provider_name=provider_name,
                model_name=model_name,
                max_response_tokens=max_response_tokens,
                minimum_supported_tokens=self.STRUCTURED_ENFORCED_MIN_MAX_RESPONSE_TOKENS,
            )
        # Normal return because the structured token cap passed library validation.
        return None

    def _raise_structured_token_limit_error(
        self,
        *,
        provider_name: str,
        model_name: str,
        max_response_tokens: int,
        finish_reason: str | None = None,
        raw_output_text: str | None = None,
    ) -> NoReturn:
        """
        Raise the shared structured token-limit exception for provider truncation.

        Args:
            provider_name: Provider or engine name surfaced to the caller.
            model_name: Concrete provider model identifier used for the call.
            max_response_tokens: Structured response token limit requested by the caller.
            finish_reason: Optional provider finish reason associated with truncation.
            raw_output_text: Optional raw output text captured before the failure was raised.

        Returns:
            This method does not return because it always raises a typed exception.
        """
        raise StructuredResponseTokenLimitError(
            message=self._build_structured_token_limit_error_message(
                provider_name=provider_name,
                model_name=model_name,
                max_response_tokens=max_response_tokens,
                failure_mode=self.STRUCTURED_TOKEN_LIMIT_FAILURE_MODE_PROVIDER_TRUNCATION,
                finish_reason=finish_reason,
                raw_output_text=raw_output_text,
            ),
            provider_name=provider_name,
            model_name=model_name,
            max_response_tokens=max_response_tokens,
            minimum_supported_tokens=self.STRUCTURED_ENFORCED_MIN_MAX_RESPONSE_TOKENS,
            finish_reason=finish_reason,
            raw_output_char_count=len(raw_output_text or ""),
        )

    def _build_completions_observability_input_metadata(
        self,
        *,
        prompt: str,
        system_prompt: str | None,
        other_params: AICompletionsPromptParamsBase | None,
        response_mode: str,
        max_response_tokens: int | None = None,
    ) -> dict[str, ObservabilityMetadataValue]:
        """
        Builds metadata-only input fields for one completions provider call.

        Args:
            prompt: Sanitized prompt text that will be sent to the provider.
            system_prompt: System prompt value that will be sent to the provider.
            other_params: Optional provider-specific prompt parameters.
            response_mode: Response mode label (`text` or `structured`).
            max_response_tokens: Optional maximum response token request for structured calls.

        Returns:
            Dictionary of metadata-only input fields safe for observability logging.
        """
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = {
            "prompt_char_count": len(prompt),
            "prompt_token_count_source": TOKEN_COUNT_SOURCE_NONE,
            "system_prompt_char_count": len(system_prompt or ""),
            "system_prompt_token_count_source": TOKEN_COUNT_SOURCE_NONE,
            "has_media_attachments": bool(
                other_params is not None and other_params.has_included_media
            ),
            "response_mode": response_mode,
        }
        if max_response_tokens is not None:
            dict_input_metadata["max_response_tokens"] = max_response_tokens
        if other_params is None or not other_params.has_included_media:
            # Normal return because there are no media attachments to summarize.
            return dict_input_metadata

        list_mime_types: list[str] = []
        int_media_total_bytes: int = 0
        int_media_attachment_count: int = 0
        # Loop through media attachments to summarize counts, bytes, and MIME types.
        for _, _, media_bytes, mime_type in other_params.iter_included_media():
            int_media_attachment_count += 1
            int_media_total_bytes += len(media_bytes)
            list_mime_types.append(mime_type)

        dict_input_metadata["media_attachment_count"] = int_media_attachment_count
        dict_input_metadata["media_total_bytes"] = int_media_total_bytes
        dict_input_metadata["media_mime_types"] = tuple(list_mime_types)
        # Normal return with completions input metadata including media summary fields.
        return dict_input_metadata

    def _build_completions_observability_result_summary(
        self,
        *,
        observed_result: AiApiObservedCompletionsResultModel[Any],
        provider_elapsed_ms: float,
    ) -> AiApiCallResultSummaryModel:
        """
        Builds metadata-only output fields for one observed completions provider result.

        Args:
            observed_result: Raw provider result container returned by the wrapped call.
            provider_elapsed_ms: Measured elapsed milliseconds for the wrapped provider path.

        Returns:
            AiApiCallResultSummaryModel containing output metadata safe for observability logging.
        """
        output_token_count: int | None = observed_result.provider_completion_tokens
        output_token_count_source: str = (
            TOKEN_COUNT_SOURCE_PROVIDER
            if output_token_count is not None
            else TOKEN_COUNT_SOURCE_NONE
        )
        input_token_count: int | None = observed_result.provider_prompt_tokens
        input_token_count_source: str = (
            TOKEN_COUNT_SOURCE_PROVIDER
            if input_token_count is not None
            else TOKEN_COUNT_SOURCE_NONE
        )
        call_result_summary: AiApiCallResultSummaryModel = AiApiCallResultSummaryModel(
            provider_elapsed_ms=provider_elapsed_ms,
            input_token_count=input_token_count,
            input_token_count_source=input_token_count_source,
            output_token_count=output_token_count,
            output_token_count_source=output_token_count_source,
            provider_prompt_tokens=observed_result.provider_prompt_tokens,
            provider_completion_tokens=observed_result.provider_completion_tokens,
            provider_cached_input_tokens=observed_result.provider_cached_input_tokens,
            provider_total_tokens=observed_result.provider_total_tokens,
            finish_reason=observed_result.finish_reason,
            dict_metadata={
                "output_char_count": len(observed_result.raw_output_text),
                **observed_result.dict_metadata,
            },
        )
        # Normal return with completions output summary metadata derived from the raw provider result.
        return call_result_summary

    @property
    @abstractmethod
    def max_context_tokens(self) -> int:
        """
        Return the maximum number of tokens supported by the model's
        context window.  Concrete subclasses must implement this.
        """
        ...

    @property
    def price_per_1k_tokens(self) -> float:
        """
        Deprecated blended USD cost for 1,000 tokens (input+output mean).

        Retained for back-compat. Prefer `capabilities.pricing` for the split
        input/output/cached rates and `compute_completion_cost(...)` for a real
        cost from measured usage. Returns 0.0 when the model is not priced.
        """
        pricing: AIModelPricing | None = self.capabilities.pricing
        # Normal return with a registry-backed blended rate, or 0.0 when unpriced.
        return pricing.blended_per_1k_tokens() if pricing is not None else 0.0

    def compute_completion_cost(
        self,
        *,
        input_tokens: int,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
    ) -> float:
        """
        Return the USD cost for measured token usage using split model rates.

        Uses `capabilities.pricing` (per-1M input/output/cached rates). This is
        the accurate replacement for the blended `price_per_1k_tokens`.

        Args:
            input_tokens: Non-cached input tokens billed at the input rate.
            output_tokens: Output tokens billed at the output rate.
            cached_input_tokens: Input tokens billed at the cached-input rate.

        Returns:
            USD cost as a float; 0.0 when the model has no token pricing.
        """
        pricing: AIModelPricing | None = self.capabilities.pricing
        if pricing is None or pricing.token_rates is None:
            # Early return because this model has no token pricing on record.
            return 0.0
        # Normal return with the computed cost from the split rates.
        return float(
            pricing.compute_token_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
            )
        )

    @staticmethod
    def generate_prompt_addendum_json_schema_instruction(
        response_model: Type[AIStructuredPrompt],
        *,
        code_fence: bool = True,
    ) -> str:
        """
        Generate a prompt addendum that tells the model to return ONLY JSON
        matching the model’s JSON Schema.

        Parameters
        ----------
        response_model
            A PromptResult subclass implementing `model_json_schema()`.
        code_fence
            If True, wraps the schema in a ```json …``` fence for clarity.

        Returns
        -------
        A string you can append to your user prompt.
        """
        # 1) grab the minimal JSON Schema dict
        schema = response_model.model_json_schema()

        # 2) pretty-print it
        schema_str = json.dumps(schema, indent=2)

        # 3) wrap in code fences if requested
        if code_fence:
            schema_str = f"```json\n{schema_str}\n```"

        # 4) build the instruction
        return (
            "Return *only* a JSON object (not Markdown) that matches the following JSON Schema "
            "and nothing else:\n"
            f"{schema_str}"
        )

    @staticmethod
    def generate_prompt_entropy_tag(prefix: str = "nonce") -> str:
        """
        Returns a short random tag such as 'nonce:5e3a7c2d'.

        * prefix  - leading label so you can grep for it in logs.
        * Uses uuid4 → 128-bit randomness → virtually zero chance of repeat.
        * Only first 8 hex chars are kept to keep prompts small.
        """
        random_hex = uuid.uuid4().hex[:8]  # e.g. '5e3a7c2d'
        return f"{prefix}:{random_hex}"

    @staticmethod
    def estimate_max_tokens(
        n: int,
        *,
        avg_words_per_phrase: float = 2.5,
        tokens_per_word: float = 1.3,
        json_overhead_tokens: int = 12,
        chain_of_thought_allowance: int = 120,
        safety_margin: float = 1.15,
    ) -> int:
        """
        maxTokens
        --------------------------------
        n                       – number of phrases you’ll ask the model to return
        avg_words_per_phrase    – average length of each phrase (default 2.5 words)
        tokens_per_word         – ~1.3 is OpenAI/BPE average
        json_overhead_tokens    – brackets, quotes, commas, field name
        chain_of_thought_allowance – room for the model’s <thinking> preamble
        safety_margin           – final head-room factor so we don’t truncate

        Returns an **int**, rounded up to the nearest multiple of 16 (just tidy).
        """
        tokens_for_phrases = (
            n * avg_words_per_phrase * tokens_per_word  # natural language
            + n  # one token/phrase for quotes & commas
        )

        raw_total = (
            tokens_for_phrases + json_overhead_tokens + chain_of_thought_allowance
        ) * safety_margin

        # Round up to nearest multiple of 16 (helpful for later batching)
        return int(math.ceil(raw_total / 16.0) * 16)

    @abstractmethod
    def strict_schema_prompt(
        self,
        prompt: str,
        response_model: Type[AIStructuredPrompt],
        max_response_tokens: int = STRUCTURED_DEFAULT_MAX_RESPONSE_TOKENS,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> AIStructuredPrompt:
        """
        Generates a strict schema prompt and returns the result as a structured object.

        Args:
            prompt: The prompt string to send.
            response_model: The structured response model used for validation.
            max_response_tokens: Maximum number of tokens allowed in the response.
            other_params: Optional provider-specific parameters, including system prompt overrides.
        """

    @abstractmethod
    def send_prompt(
        self, prompt: str, *, other_params: AICompletionsPromptParamsBase | None = None
    ) -> str:
        """
        Sends a prompt to the completions engine and returns the result as a string.

        Args:
            prompt: The text prompt to send
            other_params: Optional provider-specific parameters
        """

    @property
    def capabilities(self) -> AICompletionsCapabilitiesBase:
        """
        Returns the capabilities descriptor for the configured completions model.

        Providers override this with model-specific capabilities. The base
        default describes a text-only, non-streaming model at the configured
        context window.
        """
        # Normal return with conservative default completions capabilities.
        return AICompletionsCapabilitiesBase(
            context_window_length=self.max_context_tokens
        )

    def send_prompt_streaming(
        self,
        prompt: str,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> Iterator[str]:
        """
        Sends a prompt and yields response text chunks as the provider streams them.

        Template method: capability and configuration gating happen here so
        providers cannot skip them. Providers whose capabilities declare
        streaming support implement _send_prompt_streaming_provider.

        Args:
            prompt: The text prompt to send.
            other_params: Optional provider-specific parameters.

        Returns:
            Iterator of response text chunks in provider order.

        Raises:
            AiProviderCapabilityUnsupportedError: When the configured model does
                not support streaming completions.
            AiProviderConfigurationError: When PII redaction middleware is
                enabled; chunk-boundary redaction cannot be guaranteed for
                streamed output.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty or None")
        if not self.capabilities.supports_streaming:
            raise AiProviderCapabilityUnsupportedError(
                f"{type(self).__name__} model '{self.model_name}' does not support "
                "streaming completions. Use send_prompt, or configure a model whose "
                "capabilities include streaming."
            )
        if self.pii_middleware.bool_enabled:
            raise AiProviderConfigurationError(
                "Streaming completions are unavailable while PII redaction "
                "middleware is enabled: redaction cannot be guaranteed across "
                "stream chunk boundaries. Use send_prompt, or disable the PII "
                "redaction middleware profile."
            )
        # Normal return with the provider-implemented streaming iterator.
        return self._send_prompt_streaming_provider(prompt, other_params=other_params)

    def _send_prompt_streaming_provider(
        self,
        prompt: str,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> Iterator[str]:
        """
        Provider hook for streaming completion generation.

        The base implementation raises: a provider whose capabilities declare
        streaming support must implement this hook.

        Args:
            prompt: Validated text prompt to send.
            other_params: Optional provider-specific parameters.

        Raises:
            AiProviderCapabilityUnsupportedError: Always, in the base class.
        """
        raise AiProviderCapabilityUnsupportedError(
            f"{type(self).__name__} does not implement streaming completions."
        )

    def count_tokens(
        self,
        prompt: str,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> int:
        """
        Returns the provider-counted input token count for a prompt.

        Template method: capability gating and PII redaction happen here so
        providers cannot skip them. Providers whose capabilities declare token
        counting implement _count_tokens_provider. This measures input tokens
        only, using the same redacted request shape send_prompt would build:
        the prompt is redacted here so the count matches the request that would
        actually be sent and no unredacted PII leaves the trust boundary.

        Args:
            prompt: The text prompt to measure.
            other_params: Optional provider-specific parameters.

        Returns:
            Provider-reported input token count.

        Raises:
            AiProviderCapabilityUnsupportedError: When the configured model does
                not support provider-side token counting.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty or None")
        if not self.capabilities.supports_token_counting:
            raise AiProviderCapabilityUnsupportedError(
                f"{type(self).__name__} model '{self.model_name}' does not support "
                "provider-side token counting. Configure a model whose capabilities "
                "include token counting."
            )
        prompt = self.pii_middleware.process_input(prompt)
        # Normal return with the provider-counted input token total.
        return self._count_tokens_provider(prompt, other_params=other_params)

    def _count_tokens_provider(
        self,
        prompt: str,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> int:
        """
        Provider hook for input token counting.

        The base implementation raises: a provider whose capabilities declare
        token counting must implement this hook.

        Args:
            prompt: Validated text prompt to measure.
            other_params: Optional provider-specific parameters.

        Raises:
            AiProviderCapabilityUnsupportedError: Always, in the base class.
        """
        raise AiProviderCapabilityUnsupportedError(
            f"{type(self).__name__} does not implement token counting."
        )

    # ── Batch completions ───────────────────────────────────────────────────
    # Asynchronous batch processing of many prompts at reduced cost. Template
    # methods gate on capabilities.supports_batch and delegate to provider
    # hooks, mirroring the streaming and token-counting surfaces.

    BATCH_DEFAULT_POLL_INTERVAL_SECONDS: ClassVar[float] = 30.0
    BATCH_DEFAULT_TIMEOUT_SECONDS: ClassVar[float] = 86_400.0

    def _require_batch_capability(self) -> None:
        """Raise when the configured model does not support batch processing."""
        if not self.capabilities.supports_batch:
            raise AiProviderCapabilityUnsupportedError(
                f"{type(self).__name__} model '{self.model_name}' does not support "
                "batch completions. Configure a model whose capabilities include "
                "batch processing."
            )

    @staticmethod
    def _resolve_batch_id(batch: str | AIBatchJob) -> str:
        """Return the namespaced batch_id from a batch id string or job object."""
        if isinstance(batch, AIBatchJob):
            return batch.batch_id
        return batch

    def submit_batch(self, requests: list[AIBatchRequestItem]) -> AIBatchJob:
        """
        Submit many prompts as one asynchronous batch and return a job handle.

        Template method: capability and input validation happen here so
        providers cannot skip them. Providers whose capabilities declare batch
        support implement _submit_batch_provider.

        Args:
            requests: Batch request items, each with a unique custom_id.

        Returns:
            AIBatchJob handle for polling and result retrieval.

        Raises:
            AiProviderCapabilityUnsupportedError: When the model has no batch support.
            ValueError: When requests is empty or custom_ids are not unique.
        """
        self._require_batch_capability()
        if not requests:
            raise ValueError("Batch requires at least one request item.")
        # Validate each item up front so a bad request fails locally with its
        # custom_id, not deep in a provider 400 after the whole batch is built.
        for item in requests:
            if not item.custom_id or not item.custom_id.strip():
                raise ValueError("Batch request custom_id cannot be empty.")
            if not item.prompt or not item.prompt.strip():
                raise ValueError(
                    f"Batch request '{item.custom_id}' has an empty prompt."
                )
            if item.max_response_tokens is not None and item.max_response_tokens <= 0:
                raise ValueError(
                    f"Batch request '{item.custom_id}' has a non-positive "
                    "max_response_tokens."
                )
        list_custom_ids: list[str] = [item.custom_id for item in requests]
        if len(set(list_custom_ids)) != len(list_custom_ids):
            raise ValueError("Batch request custom_id values must be unique.")
        # Normal return with the provider-created batch job handle.
        return self._submit_batch_provider(requests)

    def get_batch(self, batch: str | AIBatchJob) -> AIBatchJob:
        """
        Return the current status of a submitted batch.

        Args:
            batch: Batch id string or a previously returned AIBatchJob.

        Returns:
            AIBatchJob with refreshed status and request counts.

        Raises:
            AiProviderCapabilityUnsupportedError: When the model has no batch support.
        """
        self._require_batch_capability()
        # Normal return with the refreshed batch job handle.
        return self._get_batch_provider(self._resolve_batch_id(batch))

    def get_batch_results(self, batch: str | AIBatchJob) -> list[AIBatchResultItem]:
        """
        Return per-request results for an ended batch, keyed by custom_id.

        Args:
            batch: Batch id string or a previously returned AIBatchJob.

        Returns:
            Result items in provider order; correlate to requests via custom_id.

        Raises:
            AiProviderCapabilityUnsupportedError: When the model has no batch support.
        """
        self._require_batch_capability()
        # Normal return with the provider-reported batch results.
        return self._get_batch_results_provider(self._resolve_batch_id(batch))

    def cancel_batch(self, batch: str | AIBatchJob) -> AIBatchJob:
        """
        Request cancellation of an in-progress batch.

        Args:
            batch: Batch id string or a previously returned AIBatchJob.

        Returns:
            AIBatchJob reflecting the canceling/canceled status.

        Raises:
            AiProviderCapabilityUnsupportedError: When the model has no batch support.
        """
        self._require_batch_capability()
        # Normal return with the batch job handle after requesting cancellation.
        return self._cancel_batch_provider(self._resolve_batch_id(batch))

    def run_batch(
        self,
        requests: list[AIBatchRequestItem],
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float | None = None,
    ) -> list[AIBatchResultItem]:
        """
        Submit a batch, poll until it ends, and return results.

        Blocking convenience wrapper over submit_batch, get_batch, and
        get_batch_results. Raises TimeoutError if the batch does not end within
        timeout_seconds.

        Args:
            requests: Batch request items, each with a unique custom_id.
            timeout_seconds: Max seconds to wait for the batch to end.
            poll_interval_seconds: Seconds between status polls.

        Returns:
            Result items for the ended batch.

        Raises:
            TimeoutError: When the batch does not end before the timeout.
        """
        float_timeout: float = (
            timeout_seconds
            if timeout_seconds is not None
            else self.BATCH_DEFAULT_TIMEOUT_SECONDS
        )
        float_poll_interval: float = (
            poll_interval_seconds
            if poll_interval_seconds is not None
            else self.BATCH_DEFAULT_POLL_INTERVAL_SECONDS
        )
        batch_job: AIBatchJob = self.submit_batch(requests)
        float_deadline: float = time.monotonic() + float_timeout
        # Loop until the batch reaches a terminal state or the deadline passes.
        while not batch_job.is_terminal:
            if time.monotonic() >= float_deadline:
                raise TimeoutError(
                    f"Batch '{batch_job.batch_id}' did not end within "
                    f"{float_timeout} seconds (status={batch_job.status.value})."
                )
            time.sleep(float_poll_interval)
            batch_job = self.get_batch(batch_job)
        # Normal return with the ended batch's per-request results.
        return self.get_batch_results(batch_job)

    def _submit_batch_provider(self, requests: list[AIBatchRequestItem]) -> AIBatchJob:
        """Provider hook for batch submission. Base implementation raises."""
        raise AiProviderCapabilityUnsupportedError(
            f"{type(self).__name__} does not implement batch completions."
        )

    def _get_batch_provider(self, batch_id: str) -> AIBatchJob:
        """Provider hook for batch status. Base implementation raises."""
        raise AiProviderCapabilityUnsupportedError(
            f"{type(self).__name__} does not implement batch completions."
        )

    def _get_batch_results_provider(self, batch_id: str) -> list[AIBatchResultItem]:
        """Provider hook for batch results. Base implementation raises."""
        raise AiProviderCapabilityUnsupportedError(
            f"{type(self).__name__} does not implement batch completions."
        )

    def _cancel_batch_provider(self, batch_id: str) -> AIBatchJob:
        """Provider hook for batch cancellation. Base implementation raises."""
        raise AiProviderCapabilityUnsupportedError(
            f"{type(self).__name__} does not implement batch completions."
        )

    def _execute_streaming_provider_call_with_observability(
        self,
        *,
        operation: str,
        dict_input_metadata: dict[str, ObservabilityMetadataValue] | None,
        callable_open_stream: Callable[[], Iterator[str]],
        callable_build_result_summary: Callable[[float], AiApiCallResultSummaryModel],
        legacy_caller_id: str | None = None,
    ) -> Iterator[str]:
        """
        Wraps one streaming provider call with shared observability lifecycle helpers.

        Args:
            operation: Public operation name such as `send_prompt_streaming`.
            dict_input_metadata: Optional scalar request metadata safe for input-event logs.
            callable_open_stream: Zero-argument callable that opens the provider stream.
            callable_build_result_summary: Callable that summarizes accumulated output
                using elapsed milliseconds; caller-owned accumulators supply output state.
            legacy_caller_id: Optional explicit legacy caller hint supplied by existing config.

        Returns:
            Iterator of caller-facing stream chunks with observability emission attached.
        """
        # Normal return with the observability-wrapped streaming iterator.
        return execute_observed_streaming_call(
            observability_middleware=self._get_observability_middleware(),
            callable_build_call_context=lambda: self._build_observability_call_context(
                capability=self.CLIENT_TYPE_COMPLETIONS,
                operation=operation,
                dict_metadata=dict_input_metadata,
                legacy_caller_id=legacy_caller_id,
            ),
            callable_open_stream=callable_open_stream,
            callable_build_result_summary=callable_build_result_summary,
        )

    def _build_streaming_completions_observability_result_summary(
        self,
        *,
        observed_result: AiApiObservedCompletionsResultModel[str],
        provider_elapsed_ms: float,
        int_chunk_count: int,
        bool_stream_completed: bool,
    ) -> AiApiCallResultSummaryModel:
        """
        Builds metadata-only output fields for one observed streaming completion.

        Args:
            observed_result: Accumulated stream state shaped as an observed completions result.
            provider_elapsed_ms: Elapsed milliseconds spanning the full stream consumption.
            int_chunk_count: Number of text chunks yielded to the caller.
            bool_stream_completed: False when the caller abandoned the stream early.

        Returns:
            AiApiCallResultSummaryModel containing output metadata safe for observability logging.
        """
        call_result_summary: AiApiCallResultSummaryModel = (
            self._build_completions_observability_result_summary(
                observed_result=AiApiObservedCompletionsResultModel(
                    return_value=observed_result.return_value,
                    raw_output_text=observed_result.raw_output_text,
                    finish_reason=observed_result.finish_reason,
                    provider_prompt_tokens=observed_result.provider_prompt_tokens,
                    provider_completion_tokens=observed_result.provider_completion_tokens,
                    provider_cached_input_tokens=observed_result.provider_cached_input_tokens,
                    provider_total_tokens=observed_result.provider_total_tokens,
                    dict_metadata={
                        "stream_chunk_count": int_chunk_count,
                        "stream_completed": bool_stream_completed,
                        **observed_result.dict_metadata,
                    },
                ),
                provider_elapsed_ms=provider_elapsed_ms,
            )
        )
        # Normal return with the streaming completions output summary.
        return call_result_summary


class AIBaseImageProperties(BaseModel):
    """Carries width/height (pixels), output format, and image count for generation requests."""

    width: int | None = Field(default=1_536)
    height: int | None = Field(default=1_024)
    format: str = "png"
    quality: str = "medium"
    background: str = "auto"  # transparent, auto
    num_images: int = 1

    @model_validator(mode="after")
    def _validate_dimensions(self) -> "AIBaseImageProperties":
        """Ensure requested dimensions and count are strictly positive."""

        if self.width is not None and self.width <= 0:
            raise ValueError("AIBaseImageProperties.width must be a positive integer.")
        if self.height is not None and self.height <= 0:
            raise ValueError("AIBaseImageProperties.height must be a positive integer.")
        if self.num_images <= 0:
            raise ValueError(
                "AIBaseImageProperties.num_images must be greater than zero."
            )
        return self

    @model_validator(mode="after")
    def _validate_quality(self) -> "AIBaseImageProperties":
        """Ensure requested quality is one of the supported values."""
        valid_qualities = {"low", "medium", "high"}
        if self.quality not in valid_qualities:
            raise ValueError(
                f"AIBaseImageProperties.quality must be one of {valid_qualities}, got {self.quality!r}."
            )
        return self

    @model_validator(mode="after")
    def _validate_format(self) -> "AIBaseImageProperties":
        """Ensure requested format is one of the supported values."""
        valid_formats = {"png", "jpeg", "webp"}
        if self.format.lower() not in valid_formats:
            raise ValueError(
                f"AIBaseImageProperties.format must be one of {valid_formats}, got {self.format!r}."
            )
        return self

    @model_validator(mode="after")
    def _validate_background(self) -> "AIBaseImageProperties":
        """Ensure requested background is one of the supported values."""
        valid_backgrounds = {"transparent", "auto"}
        if self.background.lower() not in valid_backgrounds:
            raise ValueError(
                f"AIBaseImageProperties.background must be one of {valid_backgrounds}, got {self.background!r}."
            )
        return self


class AIBaseImages(AIBase):
    """
    Abstract base class for image generation clients.
    """

    def __init__(self, model: str | None = None, **kwargs: Any):
        """
        Initializes the shared image-generation base class.

        Args:
            model: Optional provider model identifier supplied by the concrete image client.
            **kwargs: Additional provider-specific initialization arguments.

        Returns:
            None after the shared AI base initialization has completed.
        """
        super().__init__(model=model, **kwargs)

    def _build_images_observability_input_metadata(
        self,
        *,
        image_prompt: str,
        image_properties: AIBaseImageProperties,
    ) -> dict[str, ObservabilityMetadataValue]:
        """
        Builds metadata-only input fields for one image-generation provider call.

        Args:
            image_prompt: Prompt text sent to the image-generation provider.
            image_properties: Requested image generation properties for the provider call.

        Returns:
            Dictionary of metadata-only input fields safe for observability logging.
        """
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = {
            "prompt_char_count": len(image_prompt),
            "requested_width": image_properties.width,
            "requested_height": image_properties.height,
            "requested_num_images": image_properties.num_images,
            "requested_format": image_properties.format.lower(),
            "requested_quality": image_properties.quality.lower(),
            "requested_background": image_properties.background.lower(),
        }
        # Normal return with image-generation input metadata only.
        return dict_input_metadata

    def _build_images_observability_result_summary(
        self,
        *,
        observed_result: AiApiObservedImagesResultModel[Any],
        provider_elapsed_ms: float,
    ) -> AiApiCallResultSummaryModel:
        """
        Builds metadata-only output fields for one observed image-generation provider result.

        Args:
            observed_result: Raw provider result container returned by the wrapped call.
            provider_elapsed_ms: Measured elapsed milliseconds for the wrapped provider path.

        Returns:
            AiApiCallResultSummaryModel containing output metadata safe for observability logging.
        """
        input_token_count: int | None = observed_result.provider_input_tokens
        input_token_count_source: str = (
            TOKEN_COUNT_SOURCE_PROVIDER
            if input_token_count is not None
            else TOKEN_COUNT_SOURCE_NONE
        )
        call_result_summary: AiApiCallResultSummaryModel = AiApiCallResultSummaryModel(
            provider_elapsed_ms=provider_elapsed_ms,
            input_token_count=input_token_count,
            input_token_count_source=input_token_count_source,
            output_token_count=None,
            output_token_count_source=TOKEN_COUNT_SOURCE_NONE,
            provider_prompt_tokens=observed_result.provider_input_tokens,
            provider_completion_tokens=None,
            provider_total_tokens=observed_result.provider_total_tokens,
            finish_reason=None,
            dict_metadata={
                "generated_image_count": observed_result.generated_image_count,
                "total_output_bytes": observed_result.total_output_bytes,
                **observed_result.dict_metadata,
            },
        )
        # Normal return with image-generation output summary metadata derived from the provider result.
        return call_result_summary

    def generate_images(
        self, image_prompt: str, image_properties: AIBaseImageProperties
    ) -> list[bytes]:
        """Synchronously generate one or more images for the prompt with requested size and format.

        Implementations must honor the requested width, height, format, and quantity while returning
        raw image bytes so callers can persist or stream the result as needed.
        """
        raise NotImplementedError(
            "This feature has not been implemented for this provider."
        )

    def generate_image_files(
        self,
        image_prompt: str,
        image_properties: AIBaseImageProperties = AIBaseImageProperties(),
        root_file_name: str = "generate_image_files",
    ) -> list[Path]:
        """Generate images, save them, and return saved paths."""

        if root_file_name.strip() == "":
            raise ValueError("root_file_name must be a non-empty string.")

        image_bytes_list: list[bytes] = self.generate_images(
            image_prompt, image_properties
        )
        if not image_bytes_list:
            raise ValueError("generate_images returned no image data to save.")

        root_path: Path = Path(root_file_name)
        directory_path: Path = (
            root_path.parent if root_path.parent != Path("") else Path(".")
        )
        directory_path.mkdir(parents=True, exist_ok=True)

        suffix: str
        stem: str
        if root_path.suffix:
            suffix = root_path.suffix
            stem = root_path.stem
        else:
            suffix = f".{image_properties.format.lower()}"
            stem = root_path.name

        saved_paths: list[Path] = []
        for index, image_bytes in enumerate(image_bytes_list, start=1):
            file_name: str = f"{stem}_{index}{suffix}"
            file_path: Path = directory_path / file_name
            # WHY: Sequential suffixes prevent overwriting when multiple images are requested.
            file_path.write_bytes(image_bytes)
            saved_paths.append(file_path)

        return saved_paths


class AIBaseVideos(AIBase):
    """
    Abstract base class for video-generation clients.
    """

    TERMINAL_STATUSES: ClassVar[set[AIVideoGenerationStatus]] = {
        AIVideoGenerationStatus.COMPLETED,
        AIVideoGenerationStatus.FAILED,
        AIVideoGenerationStatus.CANCELLED,
    }

    def __init__(self, model: str | None = None, **kwargs: Any):
        super().__init__(model=model, **kwargs)

    def _build_videos_observability_input_metadata(
        self,
        *,
        video_prompt: str,
        video_properties: AIBaseVideoProperties,
    ) -> dict[str, ObservabilityMetadataValue]:
        """
        Builds metadata-only input fields for one video-generation provider call.
        """

        return {
            "prompt_char_count": len(video_prompt),
            "requested_duration_seconds": video_properties.duration_seconds,
            "requested_aspect_ratio": video_properties.aspect_ratio,
            "requested_resolution": video_properties.resolution,
            "requested_fps": video_properties.fps,
            "requested_num_videos": video_properties.num_videos,
            "requested_output_format": video_properties.output_format,
            "poll_interval_seconds": video_properties.poll_interval_seconds,
            "timeout_seconds": video_properties.timeout_seconds,
            "download_outputs": video_properties.download_outputs,
            "has_output_dir": video_properties.output_dir is not None,
        }

    def _build_videos_observability_result_summary(
        self,
        *,
        observed_result: AiApiObservedVideosResultModel[Any],
        provider_elapsed_ms: float,
    ) -> AiApiCallResultSummaryModel:
        """
        Builds metadata-only output fields for one observed video-generation provider result.
        """

        input_token_count: int | None = observed_result.provider_input_tokens
        input_token_count_source: str = (
            TOKEN_COUNT_SOURCE_PROVIDER
            if input_token_count is not None
            else TOKEN_COUNT_SOURCE_NONE
        )
        return AiApiCallResultSummaryModel(
            provider_elapsed_ms=provider_elapsed_ms,
            input_token_count=input_token_count,
            input_token_count_source=input_token_count_source,
            output_token_count=None,
            output_token_count_source=TOKEN_COUNT_SOURCE_NONE,
            provider_prompt_tokens=observed_result.provider_input_tokens,
            provider_completion_tokens=None,
            provider_total_tokens=observed_result.provider_total_tokens,
            finish_reason=None,
            dict_metadata={
                "generated_video_count": observed_result.generated_video_count,
                "total_output_bytes": observed_result.total_output_bytes,
                **observed_result.dict_metadata,
            },
        )

    def _resolve_video_output_dir(
        self, video_properties: AIBaseVideoProperties
    ) -> Path:
        """
        Resolve the local output directory for materialized video artifacts.
        """

        if video_properties.output_dir is not None:
            output_dir: Path = video_properties.output_dir
        else:
            from ai_api_unified.util.env_settings import EnvSettings

            env_settings: EnvSettings = EnvSettings()
            configured_output_dir: str | None = env_settings.get_setting(
                "VIDEO_OUTPUT_DIR",
                None,
            )
            if configured_output_dir is not None and configured_output_dir.strip():
                output_dir = Path(configured_output_dir)
            else:
                output_dir = (
                    Path(tempfile.gettempdir())
                    / "ai_api_unified"
                    / "generated_videos"
                    / self._resolve_observability_provider_engine()
                )
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _apply_environment_video_property_defaults(
        self,
        video_properties: AIBaseVideoProperties,
    ) -> AIBaseVideoProperties:
        """
        Apply environment-backed video defaults only for fields the caller did not set explicitly.
        """

        from ai_api_unified.util.env_settings import EnvSettings

        normalized_properties: AIBaseVideoProperties = video_properties.model_copy(
            deep=True
        )
        set_explicit_fields: set[str] = set(normalized_properties.model_fields_set)
        env_settings: EnvSettings = EnvSettings()

        if "output_dir" not in set_explicit_fields:
            configured_output_dir: str | None = env_settings.get_setting(
                "VIDEO_OUTPUT_DIR",
                None,
            )
            if configured_output_dir is not None and configured_output_dir.strip():
                normalized_properties.output_dir = Path(configured_output_dir)

        if "poll_interval_seconds" not in set_explicit_fields:
            raw_poll_interval_seconds: int | str | None = env_settings.get_setting(
                "VIDEO_POLL_INTERVAL_SECONDS",
                None,
            )
            if raw_poll_interval_seconds not in (None, ""):
                normalized_properties.poll_interval_seconds = int(
                    raw_poll_interval_seconds
                )

        if "timeout_seconds" not in set_explicit_fields:
            raw_timeout_seconds: int | str | None = env_settings.get_setting(
                "VIDEO_TIMEOUT_SECONDS",
                None,
            )
            if raw_timeout_seconds not in (None, ""):
                normalized_properties.timeout_seconds = int(raw_timeout_seconds)

        return normalized_properties

    def generate_video(
        self,
        video_prompt: str,
        video_properties: AIBaseVideoProperties = AIBaseVideoProperties(),
    ) -> AIVideoGenerationResult:
        """
        Submit, wait for, and materialize one normalized video-generation result.
        """

        if video_prompt.strip() == "":
            raise ValueError("video_prompt must be a non-empty string.")
        job: AIVideoGenerationJob = self.submit_video_generation(
            video_prompt,
            video_properties=video_properties,
        )
        completed_job: AIVideoGenerationJob = self.wait_for_video_generation(job)
        if completed_job.status != AIVideoGenerationStatus.COMPLETED:
            error_message_fragment: str = (
                f" error_message={completed_job.error_message}"
                if completed_job.error_message is not None
                and completed_job.error_message.strip() != ""
                else ""
            )
            raise RuntimeError(
                "Video generation did not complete successfully. "
                f"provider_job_id={completed_job.provider_job_id} "
                f"status={completed_job.status.value}"
                f"{error_message_fragment}"
            )
        return self.download_video_result(completed_job)

    def wait_for_video_generation(
        self,
        job: str | AIVideoGenerationJob,
        *,
        timeout_seconds: int | None = None,
        poll_interval_seconds: int | None = None,
    ) -> AIVideoGenerationJob:
        """
        Poll the provider until the target job reaches a terminal state.
        """

        started_at_monotonic: float = time.monotonic()
        current_job: AIVideoGenerationJob = self.get_video_generation_job(job)
        provider_metadata: dict[str, str | int | float | bool | None] = dict(
            current_job.provider_metadata
        )

        def _coerce_wait_setting(setting_key: str) -> int | None:
            raw_value: str | int | float | bool | None = provider_metadata.get(
                setting_key
            )
            if raw_value in (None, ""):
                return None
            if isinstance(raw_value, bool):
                return int(raw_value)
            if isinstance(raw_value, (int, float)):
                return int(raw_value)
            if isinstance(raw_value, str):
                return int(raw_value)
            return None

        effective_timeout_seconds: int = (
            timeout_seconds
            if timeout_seconds is not None
            else _coerce_wait_setting("resolved_timeout_seconds") or 900
        )
        effective_poll_interval_seconds: int = (
            poll_interval_seconds
            if poll_interval_seconds is not None
            else _coerce_wait_setting("resolved_poll_interval_seconds") or 10
        )

        while current_job.status not in self.TERMINAL_STATUSES:
            elapsed_seconds: float = time.monotonic() - started_at_monotonic
            if elapsed_seconds > effective_timeout_seconds:
                raise TimeoutError(
                    "Video generation did not reach a terminal state before timeout. "
                    f"provider_job_id={current_job.provider_job_id} timeout_seconds={effective_timeout_seconds}"
                )
            time.sleep(effective_poll_interval_seconds)
            current_job = self.get_video_generation_job(current_job)

        return current_job

    @staticmethod
    def extract_image_frames_from_video_buffer(
        video_buffer: bytes,
        *,
        time_offsets_seconds: list[float] | None = None,
        frame_indices: list[int] | None = None,
        image_format: str = "png",
    ) -> list[bytes]:
        """
        Extract one or more image buffers from a video buffer using the optional frame-decoding helpers.
        """

        from ai_api_unified.videos.frame_helpers import (
            extract_image_frames_from_video_buffer,
        )

        return extract_image_frames_from_video_buffer(
            video_buffer,
            time_offsets_seconds=time_offsets_seconds,
            frame_indices=frame_indices,
            image_format=image_format,
        )

    @staticmethod
    def save_image_buffers_as_files(
        image_buffers: list[bytes],
        *,
        output_dir: Path,
        root_file_name: str = "video_frame",
        image_format: str = "png",
    ) -> list[Path]:
        """
        Persist one or more image buffers as sequential files.
        """

        if not image_buffers:
            raise ValueError("image_buffers must contain at least one image buffer.")
        normalized_root_file_name: str = root_file_name.strip()
        if normalized_root_file_name == "":
            raise ValueError("root_file_name must be a non-empty string.")
        normalized_image_format: str = image_format.strip().lower()
        if normalized_image_format == "":
            raise ValueError("image_format must be a non-empty string.")
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[Path] = []
        for index, image_buffer in enumerate(image_buffers, start=1):
            file_path: Path = (
                output_dir
                / f"{normalized_root_file_name}_{index}.{normalized_image_format}"
            )
            file_path.write_bytes(image_buffer)
            saved_paths.append(file_path)
        return saved_paths

    @abstractmethod
    def submit_video_generation(
        self,
        video_prompt: str,
        video_properties: AIBaseVideoProperties = AIBaseVideoProperties(),
    ) -> AIVideoGenerationJob:
        """
        Submit a provider-side video-generation job and return the normalized job handle.
        """

    @abstractmethod
    def get_video_generation_job(
        self,
        job: str | AIVideoGenerationJob,
    ) -> AIVideoGenerationJob:
        """
        Retrieve one normalized job snapshot from the provider.
        """

    @abstractmethod
    def download_video_result(
        self,
        job: str | AIVideoGenerationJob,
    ) -> AIVideoGenerationResult:
        """
        Materialize one completed provider video result into normalized file-backed artifacts.
        """
