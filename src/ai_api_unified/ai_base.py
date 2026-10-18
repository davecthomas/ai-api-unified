from __future__ import (
    annotations,
)  # Postpone evaluation of type hints to avoid circular imports and allow forward references with | None

import asyncio
import base64
import contextvars
import functools
import json
import inspect
import logging
import math
import mimetypes
import re
import tempfile
import time
import urllib.parse
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping
from typing import (
    Any,
    AsyncIterator,
    ClassVar,
    Generic,
    Iterator,
    Literal,
    NoReturn,
    Type,
    TypeVar,
)

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ai_api_unified.ai_completions_exceptions import (
    StructuredOutputValidationError,
    StructuredResponseTokenLimitError,
)
from ai_api_unified.ai_provider_exceptions import (
    AiProviderCapabilityUnsupportedError,
    AiProviderConfigurationError,
    AiProviderRequestError,
)
from ai_api_unified.middleware.observability import (
    AiApiObservabilityMiddleware,
    get_observability_middleware,
)
from ai_api_unified.middleware.observability_runtime import (
    TOKEN_COUNT_SOURCE_NONE as _TOKEN_COUNT_SOURCE_NONE_RUNTIME,
    AiApiCallContextModel,
    AiApiCallResultSummaryModel,
    OBSERVABILITY_DIRECTION_INPUT,
    ObservabilityMetadataValue,
    TOKEN_COUNT_SOURCE_NONE,
    TOKEN_COUNT_SOURCE_PROVIDER,
    execute_observed_async_call,
    execute_observed_call,
    execute_observed_streaming_call,
    get_observability_context,
    push_observability_tags,
    reset_observability_context,
    resolve_originating_caller,
)
from ai_api_unified.middleware.request_cost import (
    REQUEST_STATUS_OK,
    AIRequestCostLineItem,
    AIRequestCostReport,
    JoinedRequestCostScope,
    RequestCallCapture,
    RequestCostScope,
    capture_request_calls,
    current_request_cost_scope,
    note_call_thinking,
    request_cost_scope,
)
from ai_api_unified.pricing.model_pricing import AIModelPricing
from ai_api_unified.middleware.pii_redactor import AiApiPiiMiddleware
from ai_api_unified.thinking import (
    COMPLETIONS_THINKING_KEY,
    DEFAULT_THINKING_LEVEL,
    THINKING_CALL_OVERRIDE,
    THINKING_NATIVE_OPTION,
    THINKING_REASON_PROVIDER_OPTION,
    AIThinkingCapabilities,
    AIThinkingLevel,
    AIThinkingResolution,
    THINKING_REASON_NO_CONTROL,
    parse_thinking_level,
    resolve_thinking_level,
    thinking_event_fields,
)

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


class AIPromptCacheRetention(str, Enum):
    """How long a provider should keep a cached prompt prefix.

    DEFAULT is the provider's standard window (about 5 minutes on Anthropic
    and Bedrock, in-memory on OpenAI). EXTENDED asks for the longer window
    where one exists (1 hour on Anthropic and Bedrock, 24 hours on OpenAI
    models that offer it). Anthropic and Bedrock bill cache writes at a
    premium over base input (1.25x for DEFAULT, 2x for EXTENDED), so EXTENDED
    pays off only when requests sharing the prefix arrive more than about 5
    minutes apart.
    """

    DEFAULT = "default"
    EXTENDED = "extended"


class AIPromptCacheHint(BaseModel):
    """
    Marks the system prompt and tool definitions as a stable prefix to cache.

    Caching only changes cost and latency, never the response, so an engine
    that cannot honor the hint ignores it. Engines map it to their native
    mechanism: Anthropic cache_control, Bedrock cachePoint blocks, OpenAI
    prompt_cache_key and prompt_cache_retention. Gemini caches repeated
    prefixes implicitly (durable, up to 24 hours on Gemini 3.x from
    2026-10-15), so it needs no request change. On conversation turns, Anthropic also caches the
    growing message history.

    A cache hit needs a byte-identical prefix: keep timestamps, request IDs,
    and other per-call values out of the system prompt.

    Attributes:
        retention: Requested cache lifetime; see AIPromptCacheRetention.
        key: Optional routing key that groups requests sharing a prefix
            (OpenAI prompt_cache_key). Ignored by other engines.
    """

    retention: AIPromptCacheRetention = AIPromptCacheRetention.DEFAULT
    key: str | None = None


class AICompletionsCapabilitiesBase(BaseModel):
    """
    Base class for capturing important attributes of completions models.

    implicit_prompt_caching is True when the provider caches repeated prompt
    prefixes without any request change. supports_prompt_cache_hint is True
    when the engine maps AIPromptCacheHint to a provider cache control.
    """

    context_window_length: int
    knowledge_cutoff_date: date | None = None
    reasoning: bool = False
    supported_data_types: list[SupportedDataType] = [SupportedDataType.TEXT]
    supports_data_residency_constraint: bool = False
    supports_streaming: bool = False
    supports_token_counting: bool = False
    supports_batch: bool = False
    supports_tool_use: bool = False
    supports_structured_output: bool = False
    supports_async: bool = False
    implicit_prompt_caching: bool = False
    supports_prompt_cache_hint: bool = False
    pricing: AIModelPricing | None = None
    # The standard thinking levels this model honors exactly; None when the
    # model exposes no thinking control. See AIThinkingLevel.
    thinking: AIThinkingCapabilities | None = None


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
    # Optional request to cache the system prompt as a stable prefix. Honored
    # by send_prompt, asend_prompt, send_prompt_streaming, and
    # strict_schema_prompt; see AIPromptCacheHint.
    prompt_cache: AIPromptCacheHint | None = None
    # Per-call thinking level; None uses the client's configured level. See
    # AIThinkingLevel and AIBaseCompletions.thinking_level.
    thinking: AIThinkingLevel | None = None


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
    # Optional request to cache the system prompt across the batch. Batch
    # requests run in any order over minutes to hours, so hits are best effort
    # and EXTENDED retention usually fits better; see AIPromptCacheHint.
    prompt_cache: AIPromptCacheHint | None = None
    # This request's thinking level; None means the client's level.
    thinking: AIThinkingLevel | None = None


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
    # Thinking resolution per custom_id, filled when the batch is submitted:
    # the level each request asked for and the one it was sent with.
    thinking: dict[str, AIThinkingResolution] = Field(default_factory=dict)
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
    # Cache reads are a subset of provider_prompt_tokens; cache writes are
    # counted separately and bill at their TTL's write rate.
    provider_cached_input_tokens: int | None = None
    provider_cache_write_5m_tokens: int | None = None
    provider_cache_write_1h_tokens: int | None = None
    provider_metadata: dict[str, Any] = {}


RETRY_POLICY_DEFAULT: str = "default"
RETRY_POLICY_NONE: str = "none"

ORG_INFO_SOURCE_ADMIN_API: str = "admin_api"
ORG_INFO_SOURCE_ACCOUNT_API: str = "account_api"
ORG_INFO_SOURCE_RESPONSE_HEADER: str = "response_header"
ORG_INFO_SOURCE_CONFIGURATION: str = "configuration"
ORG_INFO_SOURCE_NONE: str = "none"


class AIProviderOrgInfoBase(BaseModel):
    """
    Provider organization identity for billing attribution.

    Providers subclass this to add native fields (for example a Google
    project number); the normalized base fields cover cross-provider use.

    Attributes:
        org_id: Provider organization identifier, when resolvable.
        org_name: Organization display name; requires the provider admin key
            (v1: ANTHROPIC_ADMIN_KEY on the claude engine).
        source: How the identity was resolved: "admin_api" (id and name),
            "response_header" (id only), or "none" (engine cannot identify).
    """

    org_id: str | None = None
    org_name: str | None = None
    source: str = ORG_INFO_SOURCE_NONE


class AIProviderOrgInfoCapability(BaseModel):
    """
    Declares what organization identity an engine can resolve, and how.

    Lets consumers introspect before calling get_org_info: whether the org
    id and name are resolvable in the current configuration, and what would
    unlock more (for example an admin key or an IAM permission).

    Attributes:
        supports_org_id: True when the engine can resolve an organization
            (or account/project) identifier as configured.
        supports_org_name: True when the engine can resolve a display name
            as configured.
        requirement: What unlocks fuller identity when something is missing
            (for example "set ANTHROPIC_ADMIN_KEY for org_name"); None when
            nothing more is available.
    """

    supports_org_id: bool = False
    supports_org_name: bool = False
    requirement: str | None = None


BASE_URL_OVERRIDE_LOCAL_HOSTS: tuple[str, ...] = ("localhost", "127.0.0.1", "::1")
# Characters a provider job id may keep in a video artifact filename.
RE_VIDEO_ARTIFACT_UNSAFE_CHARS: re.Pattern[str] = re.compile(r"[^A-Za-z0-9._-]")


def _redact_url_for_error(str_base_url: str, parsed: Any) -> str:
    """
    Builds a display form of a URL safe to place in errors and logs.

    Gateway URLs legitimately carry secrets in userinfo or query strings, and
    configuration errors surface in logs and tracebacks, so only the scheme,
    host, and port are echoed back.
    """
    if not parsed.hostname:
        # Early return because a malformed value has no host to show.
        return "<malformed url>"
    str_port: str = f":{parsed.port}" if parsed.port else ""
    # Normal return with scheme, host, and port only.
    return f"{parsed.scheme}://{parsed.hostname}{str_port}"


def validate_base_url_override(str_base_url: str, *, str_env_key: str) -> str:
    """
    Validates one base-URL override before any credential is sent to it.

    A base-URL override redirects provider traffic — including API keys — to
    the supplied host, so it must be HTTPS. Plain http:// is permitted only
    for loopback hosts, which covers local gateways and recording proxies.

    Args:
        str_base_url: Raw override value from configuration or a caller.
        str_env_key: Setting name quoted in errors so the fix is obvious.

    Returns:
        The trimmed, validated base URL.

    Raises:
        AiProviderConfigurationError: When the URL is malformed or would send
            credentials over plaintext to a non-loopback host.
    """
    str_trimmed: str = str_base_url.strip()
    parsed = urllib.parse.urlparse(str_trimmed)
    if parsed.scheme not in ("https", "http") or not parsed.hostname:
        raise AiProviderConfigurationError(
            f"{str_env_key} must be an absolute http(s) URL; got "
            f"{_redact_url_for_error(str_trimmed, parsed)}."
        )
    if parsed.scheme == "http" and parsed.hostname not in BASE_URL_OVERRIDE_LOCAL_HOSTS:
        raise AiProviderConfigurationError(
            f"{str_env_key} must use https:// so provider credentials are not "
            f"sent in plaintext; got "
            f"{_redact_url_for_error(str_trimmed, parsed)}. Plain http:// is "
            f"allowed only for {', '.join(BASE_URL_OVERRIDE_LOCAL_HOSTS)}."
        )
    # Normal return with the validated override URL.
    return str_trimmed


def resolve_base_url_override(
    env_settings: Any,
    *,
    str_env_key: str,
    str_explicit: str | None = None,
) -> str | None:
    """
    Resolves a validated base-URL override from a caller argument or config.

    Precedence: explicit argument, then the `<ENGINE>_BASE_URL_OVERRIDE`
    setting. The override name is deliberately distinct from the provider
    SDKs' own base-URL environment variables (OPENAI_BASE_URL,
    ANTHROPIC_BASE_URL, GOOGLE_GEMINI_BASE_URL): engines pass the resolved
    value to the SDK explicitly, so a native variable can never take effect
    unvalidated and bypass the https check.

    Args:
        env_settings: EnvSettings-like accessor exposing get_setting.
        str_env_key: Override setting name for this engine.
        str_explicit: Optional caller-supplied override.

    Returns:
        Validated override URL, or None when no override is configured.
    """
    str_candidate: str = str_explicit or ""
    str_label: str = "base_url argument" if str_candidate.strip() else str_env_key
    if not str_candidate.strip():
        str_candidate = str(env_settings.get_setting(str_env_key, "") or "")
        from ai_api_unified.ai_models_config import DICT_SETTING_FIELD_NAMES

        str_field: str | None = DICT_SETTING_FIELD_NAMES.get(str_env_key)
        if str_field is not None:
            # The value may come from models.json, so name both places.
            str_label = f"{str_field} in models.json (or {str_env_key})"
    if not str_candidate.strip():
        # Early return because no override is configured.
        return None
    # Normal return with the validated override URL.
    return validate_base_url_override(str_candidate, str_env_key=str_label)


def normalize_retry_policy(retry_policy: str) -> str:
    """
    Normalizes and validates one retry-policy token shared by all engines.

    Args:
        retry_policy: Raw retry policy value from a constructor or environment.

    Returns:
        Normalized retry policy token ("default" or "none").

    Raises:
        ValueError: When an unrecognized retry policy value is supplied.
    """
    str_normalized: str = retry_policy.strip().lower()
    if str_normalized not in (RETRY_POLICY_DEFAULT, RETRY_POLICY_NONE):
        raise ValueError(
            f"Unsupported retry policy {retry_policy!r}; "
            f"expected '{RETRY_POLICY_DEFAULT}' or '{RETRY_POLICY_NONE}'."
        )
    # Normal return with the normalized retry policy token.
    return str_normalized


def resolve_retry_policy(
    *,
    str_explicit: str | None,
    object_configured: object,
) -> str:
    """
    Resolves one retry policy from an explicit argument or configuration.

    Every engine reads the same COMPLETIONS_RETRY_POLICY key, so they must agree
    on what a blank value means. Configuration that is present but blank is
    unconfigured: settings accessors return "" rather than the default for such a
    key, and a stray `COMPLETIONS_RETRY_POLICY=` line must not break construction.
    An argument a caller typed is validated as given, because passing "" is a
    mistake worth reporting rather than silently defaulting.

    Takes the already-read configuration value rather than an accessor, since
    engines source it from either EnvSettings or a settings mapping.

    Args:
        str_explicit: Retry policy supplied by the caller, or None when absent.
        object_configured: Raw configured value read from environment or settings.

    Returns:
        Normalized retry policy token ("default" or "none").

    Raises:
        ValueError: When an explicit or non-blank configured value is unrecognized.
    """
    if str_explicit is not None:
        # Early return validating the caller's argument exactly as supplied.
        return normalize_retry_policy(str_explicit)
    str_configured: str = (
        str(object_configured).strip() if object_configured is not None else ""
    )
    # Normal return treating a blank configured value as unconfigured.
    return normalize_retry_policy(str_configured or RETRY_POLICY_DEFAULT)


class AIFinishReason(str, Enum):
    """
    Normalized reason a completions turn stopped, uniform across engines.

    COMPLETE: The model finished its turn normally.
    LENGTH: Output was truncated by the max-response-token limit; callers
        typically retry with a larger budget.
    TOOL_USE: The model stopped to request one or more tool invocations.
    REFUSAL: The model declined to answer; callers typically abort.
    """

    COMPLETE = "complete"
    LENGTH = "length"
    TOOL_USE = "tool_use"
    REFUSAL = "refusal"


class AITool(BaseModel):
    """
    Provider-neutral tool definition offered to the model for one turn.

    Attributes:
        name: Tool name the model uses to request an invocation.
        description: Natural-language description of what the tool does.
        input_schema: JSON Schema describing the tool's input payload.
        strict: When True, engines that support it enforce schema-exact tool
            inputs; engines without strict support ignore the flag.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    strict: bool = False


class AIToolCall(BaseModel):
    """
    One tool invocation requested by the model during a conversation turn.

    Attributes:
        id: Provider-assigned identifier used to correlate the tool result.
        name: Name of the tool the model wants invoked.
        input: Parsed tool input arguments.
    """

    id: str
    name: str
    input: dict[str, Any]


class AITokenUsage(BaseModel):
    """
    Provider-reported token usage for one call or conversation turn.

    Attributes:
        input_tokens: Prompt-side tokens, including cached input as a subset.
        output_tokens: Completion-side tokens.
        cached_input_tokens: Cached prompt tokens billed at the cached rate.
        total_tokens: Provider-reported or derived total token count. Excludes
            cache-write tokens, which the provider counts separately.
        cache_write_5m_tokens: Tokens written to a 5-minute-TTL prompt cache,
            billed above base input. Not a subset of input_tokens.
        cache_write_1h_tokens: Tokens written to a 1-hour-TTL prompt cache, at
            a higher premium than the 5-minute tier.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    total_tokens: int | None = None
    # Thinking tokens, a subset of output_tokens, when reported separately.
    reasoning_tokens: int | None = None
    cache_write_5m_tokens: int | None = None
    cache_write_1h_tokens: int | None = None


class AITurnResult(BaseModel):
    """
    The model's turn returned by send_conversation.

    The caller executes any requested tools, appends the results to its own
    message history (replaying raw_content verbatim as the assistant turn),
    and calls send_conversation again; the library sends one turn at a time.

    Attributes:
        text: Concatenated text content of the turn, or None when the model
            produced no text.
        tool_calls: Tool invocations the model requested this turn.
        finish_reason: Normalized stop reason (see AIFinishReason).
        raw_content: Engine-specific content blocks for this turn, replayable
            verbatim as the next assistant message. Opaque to callers.
        usage: Provider-reported token usage for this turn.
        provider_engine: Engine token that served the turn, set by a fallback
            client; None when a single engine served it.
        model_name: Model that served the turn, set alongside provider_engine.
        thinking: The thinking level requested and the one applied; see
            AIThinkingResolution.
        provider: Vendor that served the turn (openai, anthropic, ...).
        route_tier: Router tier that served it; None off a router.
        route_method: How the route was chosen, or `model_pin` for a pinned
            call; None off a router unless pinned.
        model_pin: The engine:model the call was pinned to, or None.
        cost: Every provider call the turn made, with totals; see
            AIRequestCostReport.

    The client fills provider_engine through cost on every call it serves.
    """

    text: str | None = None
    tool_calls: list[AIToolCall] = Field(default_factory=list)
    finish_reason: AIFinishReason
    raw_content: Any = None
    usage: AITokenUsage = Field(default_factory=AITokenUsage)
    provider_engine: str | None = None
    model_name: str | None = None
    thinking: AIThinkingResolution | None = None
    provider: str | None = None
    route_tier: str | None = None
    route_method: str | None = None
    model_pin: str | None = None
    cost: AIRequestCostReport | None = None


class AIStructuredOutputResult(BaseModel):
    """
    Result of send_structured_output: parsed structured data plus turn metadata.

    Attributes:
        data: Parsed JSON object when finish_reason is COMPLETE; None when the
            turn ended in LENGTH truncation or REFUSAL, so callers distinguish
            retryable truncation from an abort in code.
        finish_reason: Normalized stop reason (see AIFinishReason).
        usage: Provider-reported token usage for the call.
        raw_text: Raw model output text before JSON parsing (empty on refusal).
        provider_engine: Engine token that served the call, set by a fallback
            client; None when a single engine served it.
        model_name: Model that served the call, set alongside provider_engine.
        thinking: The thinking level requested and the one applied; see
            AIThinkingResolution.
        provider: Vendor that served the call (openai, anthropic, ...).
        route_tier: Router tier that served it; None off a router.
        route_method: How the route was chosen, or `model_pin` for a pinned
            call; None off a router unless pinned.
        model_pin: The engine:model the call was pinned to, or None.
        cost: Every provider call this call made, with totals; see
            AIRequestCostReport.

    The client fills provider_engine through cost on every call it serves.
    """

    data: dict[str, Any] | None = None
    finish_reason: AIFinishReason
    usage: AITokenUsage = Field(default_factory=AITokenUsage)
    raw_text: str = ""
    provider_engine: str | None = None
    model_name: str | None = None
    thinking: AIThinkingResolution | None = None
    provider: str | None = None
    route_tier: str | None = None
    route_method: str | None = None
    model_pin: str | None = None
    cost: AIRequestCostReport | None = None


class AIPromptResult(BaseModel):
    """
    Result of send_prompt_with_result: the response text, and which model
    produced it, how, and at what cost.

    Every field describes this call alone, even when it ran inside a
    caller's larger `request_cost_scope()` or alongside other calls on the
    same client.

    Attributes:
        text: The model's response text, as send_prompt returns it.
        provider: Vendor that served the answer (openai, anthropic, google,
            bedrock, ...).
        provider_engine: Engine token that served it (openai,
            openai-responses, claude, google-gemini, nova, ...).
        model_name: Model that served it.
        route_tier: Router tier that served it; None off a router.
        route_method: How the route was chosen (classifier, default, floor,
            override, low_confidence, model_pin); None off a router unless
            the call was pinned.
        model_pin: The engine:model the call was pinned to, or None.
        thinking: The thinking level requested and the one applied on the
            served call; see AIThinkingResolution.
        usage: Token usage of the served call.
        cost: Every provider call this call made (the classifier, failed
            attempts, fallbacks) with its totals. Its request_id is the one
            on the call's `ai_api_request_cost` event.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    provider: str | None = None
    provider_engine: str | None = None
    model_name: str | None = None
    route_tier: str | None = None
    route_method: str | None = None
    model_pin: str | None = None
    thinking: AIThinkingResolution | None = None
    usage: AITokenUsage = Field(default_factory=AITokenUsage)
    cost: AIRequestCostReport

    @property
    def total_usd(self) -> Decimal:
        """What the call cost in USD, classifier and failed attempts included."""
        # Normal return with the call's priced total.
        return self.cost.total_usd

    @property
    def request_id(self) -> str:
        """Correlation id of the request the call ran in."""
        # Normal return with the report's id.
        return self.cost.request_id

    @property
    def config_id(self) -> str | None:
        """Hash of the configuration that routed the call, if any."""
        # Normal return with the tag, if set.
        return self.cost.tags.get("config_id")

    @property
    def config_version(self) -> str | None:
        """The models.json `config_version` label, if any."""
        # Normal return with the tag, if set.
        return self.cost.tags.get("config_version")


# Characters per token for the route preview's local input estimate; a
# conservative English average across the providers' tokenizers.
INT_PREVIEW_CHARS_PER_TOKEN: int = 4
ROUTE_PREVIEW_METHOD_DIRECT: str = "direct"


class AIRoutePreview(BaseModel):
    """
    Where one request would go, and what it would cost, without sending it.

    Attributes:
        tier: Router tier the request would use; None off a router or on
            the default route.
        method: How the tier was chosen (classifier, low_confidence, floor,
            override, pinned, default), or "direct" off a router.
        confidence: The classifier's confidence, when it ran and reported one.
        engine: Engine token of the model that would answer first.
        model: That model.
        thinking: The level that model would apply for the configured level.
        input_tokens: Estimated input tokens.
        input_token_source: "provider" (count_tokens) or "estimated"
            (characters / INT_PREVIEW_CHARS_PER_TOKEN).
        max_output_tokens: The output cap the estimate assumes.
        usd_input: Estimated input cost; None when the model is unpriced.
        usd_max: Input cost plus the cap's worth of output at the output
            rate: the most the call should cost before fallbacks; None when
            unpriced.
        pricing_source, pricing_effective_date: The rates used.
        classifier_usd: What the preview itself spent classifying.
        config_id: The routing configuration that would route it.
    """

    tier: str | None = None
    method: str = ROUTE_PREVIEW_METHOD_DIRECT
    confidence: float | None = None
    engine: str
    model: str | None = None
    thinking: AIThinkingResolution | None = None
    input_tokens: int
    input_token_source: str
    max_output_tokens: int
    usd_input: Decimal | None = None
    usd_max: Decimal | None = None
    pricing_source: str | None = None
    pricing_effective_date: date | None = None
    classifier_usd: Decimal = Decimal("0")
    config_id: str | None = None


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
    supports_async: bool = False
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
    CLIENT_TYPE_CLASSIFICATION = "classification"
    PROVIDER_VENDOR_OPENAI = "openai"
    PROVIDER_VENDOR_BEDROCK = "bedrock"
    PROVIDER_VENDOR_GOOGLE = "google"
    PROVIDER_VENDOR_AZURE = "azure"
    PROVIDER_VENDOR_ELEVENLABS = "elevenlabs"
    PROVIDER_VENDOR_ANTHROPIC = "anthropic"
    PROVIDER_VENDOR_VOYAGE = "voyage"
    PROVIDER_ENGINE_GOOGLE_GEMINI = "google-gemini"
    PROVIDER_ENGINE_CLAUDE = "claude"
    PROVIDER_ENGINE_VOYAGE = "voyage"
    _observability_middleware: AiApiObservabilityMiddleware | None = None
    _last_request_cost: AIRequestCostReport | None = None
    # Capabilities whose provider errors the base maps to
    # AiProviderRequestError; completions and classifier engines map their
    # own, with their providers' error codes.
    FROZENSET_MAPPED_ERROR_CAPABILITIES: ClassVar[frozenset[str]] = frozenset(
        {"embedding", "images", "videos", "voice"}
    )
    # config_id / config_version tags the factory stamps on every client it
    # builds; replaced, never mutated, so the class default stays empty.
    _dict_config_tags: Mapping[str, str] = MappingProxyType({})

    def __init__(self, model: str | None = None, **kwargs):
        super().__init__()
        self.model: str | None = model

    def set_config_identity(
        self,
        config_id: str | None,
        config_version: str | None = None,
        service: str | None = None,
        environment: str | None = None,
    ) -> None:
        """
        Records the configuration that built this client. Every request it
        opens carries them as the `config_id`, `config_version`, `service`,
        and `config_env` tags.

        Args:
            config_id: Hash of the routing configuration, or None.
            config_version: The models.json `config_version` label, or None.
            service: The models.json `service` name, or None.
            environment: The models.json environment merged in
                (AI_MODELS_ENV), or None.
        """
        dict_tags: dict[str, str] = {}
        if service:
            dict_tags["service"] = service
        if environment:
            dict_tags["config_env"] = environment
        if config_id:
            dict_tags["config_id"] = config_id
        if config_version:
            dict_tags["config_version"] = config_version
        self._dict_config_tags = MappingProxyType(dict_tags)

    @property
    def config_id(self) -> str | None:
        """Hash of the routing configuration that built this client, if any."""
        # Normal return with the stamped id.
        return self._dict_config_tags.get("config_id")

    @property
    def config_version(self) -> str | None:
        """The models.json `config_version` label, if any."""
        # Normal return with the stamped label.
        return self._dict_config_tags.get("config_version")

    @property
    def last_request_cost(self) -> AIRequestCostReport | None:
        """
        Cost report for the last request this client ran as the outermost
        call: every provider call it made (classification, failed attempts,
        fallbacks, escalations) with one total. None before the first
        request, and unchanged by calls made under another client's or a
        caller's `request_cost_scope()`, which own that report instead.
        """
        # Normal return with the stored report.
        return self._last_request_cost

    def _with_request_cost_scope(
        self,
        operation: str,
        callable_run: Callable[[], ProviderCallReturnType],
        *,
        request_id: str | None = None,
    ) -> ProviderCallReturnType:
        """
        Runs one public call inside a request-cost scope, opening it when
        none is active and storing the report when this client owns it.

        Args:
            operation: Public operation name for the report.
            callable_run: The call to charge to the request.
            request_id: Correlation id for a new scope; generated when omitted.

        Returns:
            Whatever `callable_run` returns.
        """
        scope: RequestCostScope | JoinedRequestCostScope | None = None
        try:
            with request_cost_scope(
                operation,
                request_id=request_id,
                dict_default_tags=self._dict_config_tags,
            ) as scope:
                result: ProviderCallReturnType = callable_run()
                scope.mark_served()
        finally:
            if scope is not None and scope.bool_owner:
                self._last_request_cost = scope.report
        # Normal return with the call's result.
        return result

    async def _awith_request_cost_scope(
        self,
        operation: str,
        callable_run: Callable[[], Any],
        *,
        request_id: str | None = None,
    ) -> Any:
        """Async twin of _with_request_cost_scope; `callable_run` returns an awaitable."""
        scope: RequestCostScope | JoinedRequestCostScope | None = None
        try:
            with request_cost_scope(
                operation,
                request_id=request_id,
                dict_default_tags=self._dict_config_tags,
            ) as scope:
                result: Any = await callable_run()
                scope.mark_served()
        finally:
            if scope is not None and scope.bool_owner:
                self._last_request_cost = scope.report
        # Normal return with the call's result.
        return result

    def _stream_with_request_cost_scope(
        self,
        operation: str,
        callable_open_stream: Callable[[], Iterator[str]],
        *,
        request_id: str | None = None,
    ) -> Iterator[str]:
        """
        Streaming twin of _with_request_cost_scope: the scope spans the
        stream's consumption and closes when it ends or is abandoned.
        """
        scope: RequestCostScope | JoinedRequestCostScope | None = None
        try:
            with request_cost_scope(
                operation,
                request_id=request_id,
                dict_default_tags=self._dict_config_tags,
            ) as scope:
                yield from callable_open_stream()
                scope.mark_served()
        finally:
            if scope is not None and scope.bool_owner:
                self._last_request_cost = scope.report

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
        last_thinking: Any = getattr(self, "_last_thinking", None)
        if capability == self.CLIENT_TYPE_COMPLETIONS and last_thinking is not None:
            # Both levels ride on every completions event so a consumer sees
            # when a model applied a level other than the one requested.
            dict_context_metadata.update(thinking_event_fields(last_thinking))
        model_name: str | None = self._resolve_observability_model_name()
        model_version: str | None = self._resolve_observability_model_version(
            model_name=model_name
        )
        str_org_id: str | None = None
        str_org_name: str | None = None
        try:
            str_org_id, str_org_name = self._resolve_provider_organization()
        except Exception as exception:
            # Fail open: organization identity is enrichment only and must
            # never affect the provider call.
            _LOGGER.debug(
                "provider organization resolution failed: %s",
                exception.__class__.__name__,
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
            dict_tags=observability_context.tags,
            provider_org_id=str_org_id,
            provider_org_name=str_org_name,
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
        # A plain engine call is its own one-call request unless a wrapper or
        # the caller already opened one.
        provider_result: ProviderCallReturnType = self._with_request_cost_scope(
            operation,
            lambda: execute_observed_call(
                observability_middleware=observability_middleware,
                callable_build_call_context=lambda: self._build_observability_call_context(
                    capability=capability,
                    operation=operation,
                    dict_metadata=dict_input_metadata,
                    legacy_caller_id=legacy_caller_id,
                ),
                callable_execute=self._typed_provider_call(
                    capability, callable_execute
                ),
                callable_build_result_summary=callable_build_result_summary,
            ),
        )
        # Normal return with the original provider result after optional observability wrapping.
        return provider_result

    async def _execute_provider_acall_with_observability(
        self,
        *,
        capability: str,
        operation: str,
        dict_input_metadata: dict[str, ObservabilityMetadataValue] | None,
        callable_execute: Callable[[], Any],
        callable_build_result_summary: Callable[
            [Any, float], AiApiCallResultSummaryModel
        ],
        legacy_caller_id: str | None = None,
    ) -> Any:
        """
        Async twin of _execute_provider_call_with_observability for awaitable calls.

        Args:
            capability: Capability label for the public API surface being invoked.
            operation: Public operation name such as `asend_prompt`.
            dict_input_metadata: Optional scalar request metadata safe for input-event logs.
            callable_execute: Zero-argument callable returning the awaitable provider call.
            callable_build_result_summary: Callable that summarizes provider output and elapsed time.
            legacy_caller_id: Optional explicit legacy caller hint supplied by existing config.

        Returns:
            Original provider return value from awaiting `callable_execute`.
        """
        observability_middleware: AiApiObservabilityMiddleware = (
            self._get_observability_middleware()
        )
        provider_result: Any = await self._awith_request_cost_scope(
            operation,
            lambda: execute_observed_async_call(
                observability_middleware=observability_middleware,
                callable_build_call_context=lambda: self._build_observability_call_context(
                    capability=capability,
                    operation=operation,
                    dict_metadata=dict_input_metadata,
                    legacy_caller_id=legacy_caller_id,
                ),
                callable_execute=self._atyped_provider_call(
                    capability, callable_execute
                ),
                callable_build_result_summary=callable_build_result_summary,
            ),
        )
        # Normal return with the original provider result after optional observability wrapping.
        return provider_result

    def _typed_provider_call(
        self, capability: str, callable_execute: Callable[[], Any]
    ) -> Callable[[], Any]:
        """
        Wraps a media provider call so an SDK failure raises
        AiProviderRequestError with its status and fallback reason; see
        ai_provider_error_mapping.py. Completions calls are returned as is.
        """
        if capability not in self.FROZENSET_MAPPED_ERROR_CAPABILITIES:
            # Early return: the engine maps its own errors.
            return callable_execute

        def _run() -> Any:
            try:
                # Normal return with the provider's result.
                return callable_execute()
            except AiProviderRequestError:
                raise
            except Exception as exception:
                self._raise_mapped_provider_error(exception)
                raise

        # Normal return with the mapping wrapper.
        return _run

    def _atyped_provider_call(
        self, capability: str, callable_execute: Callable[[], Any]
    ) -> Callable[[], Any]:
        """Async twin of _typed_provider_call."""
        if capability not in self.FROZENSET_MAPPED_ERROR_CAPABILITIES:
            # Early return: the engine maps its own errors.
            return callable_execute

        async def _run() -> Any:
            try:
                # Normal return with the provider's result.
                return await callable_execute()
            except AiProviderRequestError:
                raise
            except Exception as exception:
                self._raise_mapped_provider_error(exception)
                raise

        # Normal return with the mapping wrapper.
        return _run

    def _raise_mapped_provider_error(self, exception: Exception) -> None:
        """Raises the typed form of a provider failure, when one exists."""
        from ai_api_unified.ai_provider_error_mapping import map_provider_exception

        typed: AiProviderRequestError | None = map_provider_exception(
            exception, provider_engine=self._resolve_observability_provider_engine()
        )
        if typed is not None:
            raise typed from exception
        # Normal return: not a provider failure; the caller re-raises it.
        return None

    def get_org_info(self) -> AIProviderOrgInfoBase:
        """
        Returns the provider organization identity on demand.

        Unlike cost-event enrichment (which fails open), an explicit call
        surfaces resolution failures: engines raise AiProviderRequestError
        with status_code when their identity lookup fails (for example a
        rejected ANTHROPIC_ADMIN_KEY). Engines that cannot identify the
        paying organization return source="none". Successful resolution is
        cached per client; an explicit call after a failed background
        attempt retries.

        Returns:
            AIProviderOrgInfoBase (or a provider subclass) with org_id,
            org_name, and the resolution source.
        """
        cached: AIProviderOrgInfoBase | None = getattr(self, "_org_info_cache", None)
        if cached is not None:
            # Early return with the cached successful identity.
            return cached
        org_info: AIProviderOrgInfoBase = self._get_org_info_provider()
        self._org_info_cache = org_info
        self._bool_org_enrichment_failed = False
        # Normal return with the freshly resolved identity.
        return org_info

    def get_org_info_capability(self) -> AIProviderOrgInfoCapability:
        """
        Declares what organization identity this engine can resolve.

        Returns:
            AIProviderOrgInfoCapability describing id/name resolvability in
            the current configuration and what would unlock more.
        """
        # Normal return with the provider-declared capability.
        return self._get_org_info_capability_provider()

    def _get_org_info_capability_provider(self) -> AIProviderOrgInfoCapability:
        """
        Provider hook for the org-identity capability declaration.
        """
        # Normal return declaring no organization identity by default.
        return AIProviderOrgInfoCapability()

    def _get_org_info_provider(self) -> AIProviderOrgInfoBase:
        """
        Provider hook for organization identity resolution.

        The base default reports no organization identity; provider bases
        override it (anthropic: Admin API or response-header probe; openai:
        account API or response-header probe; bedrock: STS/IAM; google:
        configured project). Hooks fetch only — caching lives in
        get_org_info and the enrichment wrapper.
        """
        # Normal return with no organization identity by default.
        return AIProviderOrgInfoBase()

    def _resolve_provider_organization(self) -> tuple[str | None, str | None]:
        """
        Fail-open organization resolution used by cost-event enrichment.

        Uses the shared success cache and negative-caches failures so
        enrichment never re-probes on every call; get_org_info bypasses the
        negative cache and retries.

        Returns:
            Tuple of (org_id, org_name); either element may be None.
        """
        cached: AIProviderOrgInfoBase | None = getattr(self, "_org_info_cache", None)
        if cached is not None:
            # Early return with the cached successful identity.
            return cached.org_id, cached.org_name
        if getattr(self, "_bool_org_enrichment_failed", False):
            # Early return because a prior attempt failed; enrichment does
            # not retry (get_org_info does).
            return None, None
        try:
            org_info: AIProviderOrgInfoBase = self.get_org_info()
        except Exception as exception:
            self._bool_org_enrichment_failed = True
            _LOGGER.warning(
                "organization enrichment resolution failed (%s); cost events "
                "will omit org identity.",
                exception.__class__.__name__,
            )
            # Early return with no identity because enrichment fails open.
            return None, None
        # Normal return with the resolved identity fields.
        return org_info.org_id, org_info.org_name

    def _resolve_observability_provider_vendor(self) -> str:
        """
        Resolves a best-effort provider vendor label for shared observability metadata.

        Args:
            None

        Returns:
            Best-effort provider vendor label derived from the concrete client module or class.
        """
        lower_module_name: str = self.__class__.__module__.lower()
        if "voyage" in lower_module_name:
            # Early return for Voyage AI embeddings clients.
            return self.PROVIDER_VENDOR_VOYAGE
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
        if "voyage" in lower_module_name:
            # Early return for Voyage AI embeddings clients.
            return self.PROVIDER_ENGINE_VOYAGE
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
        provider_cache_write_5m_tokens: Optional provider-reported count of tokens
            written to a 5-minute-TTL prompt cache. Billed at a premium over base
            input and reported separately by the provider, so NOT a subset of
            provider_prompt_tokens.
        provider_cache_write_1h_tokens: Optional provider-reported count of tokens
            written to a 1-hour-TTL prompt cache, at a higher premium than the
            5-minute tier. Also not a subset of provider_prompt_tokens.
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
    # Appended after the pre-2.24 fields so positional construction keeps its
    # original argument mapping.
    provider_cache_write_5m_tokens: int | None = None
    provider_cache_write_1h_tokens: int | None = None
    # Thinking (reasoning) tokens, a subset of provider_completion_tokens,
    # when the provider reports them separately.
    provider_reasoning_tokens: int | None = None


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
    def generate_embeddings(
        self, text: str, *, input_type: str | None = None
    ) -> dict[str, Any]:
        """
        Generates embeddings for a single piece of text.

        Args:
            text: Text to embed.
            input_type: Optional retrieval hint ("query" or "document") that
                improves retrieval quality on engines that support it (the
                voyage engine); engines without the concept ignore it.
        """

    @abstractmethod
    def generate_embeddings_batch(
        self, texts: list[str], *, input_type: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Generates embeddings for multiple pieces of text in a single API call.

        Args:
            texts: Texts to embed.
            input_type: Optional retrieval hint ("query" or "document"); see
                generate_embeddings.
        """

    async def agenerate_embeddings(
        self, text: str, *, input_type: str | None = None
    ) -> dict[str, Any]:
        """
        Async variant of generate_embeddings for engines whose SDK has an
        async client.

        Raises:
            AiProviderCapabilityUnsupportedError: When the configured model
                does not support async embeddings.
        """
        if not self.capabilities.supports_async:
            self._raise_capability_unsupported("async embeddings")
        # Normal return with the provider-implemented async embedding.
        return await self._agenerate_embeddings_provider(text, input_type=input_type)

    async def _agenerate_embeddings_provider(
        self, text: str, *, input_type: str | None
    ) -> dict[str, Any]:
        """
        Provider hook for async single-text embedding generation.

        The base implementation raises: a provider whose capabilities declare
        async support must implement this hook.
        """
        self._raise_capability_unsupported("async embeddings")

    async def agenerate_embeddings_batch(
        self, texts: list[str], *, input_type: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Async variant of generate_embeddings_batch.

        Raises:
            AiProviderCapabilityUnsupportedError: When the configured model
                does not support async embeddings.
        """
        if not self.capabilities.supports_async:
            self._raise_capability_unsupported("async embeddings")
        # Normal return with the provider-implemented async batch embeddings.
        return await self._agenerate_embeddings_batch_provider(
            texts, input_type=input_type
        )

    async def _agenerate_embeddings_batch_provider(
        self, texts: list[str], *, input_type: str | None
    ) -> list[dict[str, Any]]:
        """
        Provider hook for async batch embedding generation.

        The base implementation raises: a provider whose capabilities declare
        async support must implement this hook.
        """
        self._raise_capability_unsupported("async embeddings")

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

    async def agenerate_embeddings_multimodal(
        self,
        params: AIEmbeddingsMultimodalParams,
    ) -> dict[str, Any]:
        """Async twin of generate_embeddings_multimodal; runs on a worker thread."""
        # Normal return with the embedding.
        return await asyncio.to_thread(self.generate_embeddings_multimodal, params)

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


# ── Classification ──────────────────────────────────────────────────────────

# Default cap on the text a classifier reads. Classification bills input
# tokens on every call and a label rarely needs more than the head and tail
# of a long prompt.
DEFAULT_CLASSIFIER_MAX_INPUT_CHARS: int = 6000

CLASSIFY_OPERATION_CHOICE: str = "classify_choice"
CLASSIFY_OPERATION_PREDICATE: str = "classify_predicate"
CLASSIFY_OPERATION_SCORE: str = "classify_score"

# Values of the `ai_classifier_event` field on this module's log records.
CLASSIFIER_EVENT_UNMETERED: str = "classifier_unmetered"


def truncate_classification_text(str_text: str, int_max_chars: int | None) -> str:
    """
    Shortens text for a classifier, keeping its head and tail.

    Args:
        str_text: The text to classify.
        int_max_chars: Cap on characters, or None for no cap.

    Returns:
        The text unchanged when it fits, else its first two thirds and last
        third of the cap joined by an ellipsis line.
    """
    if int_max_chars is None or len(str_text) <= int_max_chars:
        # Early return: nothing to cut.
        return str_text
    int_head: int = (int_max_chars * 2) // 3
    int_tail: int = int_max_chars - int_head
    # Slice the tail by start index: str_text[-0:] would return the whole text.
    str_tail: str = str_text[len(str_text) - int_tail :]
    # Normal return with the head and tail around a marker.
    return f"{str_text[:int_head]}\n...\n{str_tail}"


class AIClassifierCapabilitiesBase(BaseModel):
    """
    Capability descriptor for a classification engine.

    Attributes:
        supports_choice: Picks one of a caller-defined set of options.
        supports_predicate: Answers a yes/no question with a probability.
        supports_score: Places the input on an ordered scale of levels.
        supports_images: Accepts image input alongside text.
        supports_async: Exposes the aclassify_* coroutines.
        max_choices: Most options one choice question may offer, or None.
        max_input_chars: Characters kept from the input before the call; the
            template truncates longer text head-and-tail. None means no cap.
        pricing: Per-token rates used to price classification calls.
    """

    supports_choice: bool = True
    supports_predicate: bool = False
    supports_score: bool = False
    supports_images: bool = False
    supports_async: bool = False
    max_choices: int | None = None
    max_input_chars: int | None = DEFAULT_CLASSIFIER_MAX_INPUT_CHARS
    pricing: AIModelPricing | None = None


class AIClassificationChoice(BaseModel):
    """
    One option a choice question offers.

    Attributes:
        value: The token returned when this option is picked.
        description: What the option means, shown to the classifier.
    """

    model_config = ConfigDict(frozen=True)

    value: str
    description: str = ""


class AIClassificationLevel(BaseModel):
    """
    One step on a score question's ordered scale, lowest first.

    Attributes:
        label: The level's name.
        description: What the level means, shown to the classifier.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    description: str = ""


class AIClassificationResult(BaseModel):
    """
    What a classifier decided, plus how the call was metered.

    Attributes:
        question_type: "choice", "predicate", or "score".
        choice: The picked option value, or the picked level label for a
            score question; None when the classifier refused.
        probability: Predicate answers: probability that the condition holds.
        score: Score answers: position on the scale normalized to [0, 1].
        confidence: The classifier's confidence in the answer, when reported.
        probabilities: Distribution over options or levels, when reported.
        refused: True when the classifier declined to answer.
        raw_text: Provider text behind the answer, for debugging.
        usage: Token usage for the call.
        token_source: Where the token counts came from: "provider",
            "estimated", or "none". Never silently zero: "none" means the call
            could not be metered and was logged as such.
        provider_engine: Engine token that served the call.
        model_name: Model that served the call.
        latency_ms: Wall-clock duration of the provider call.
    """

    question_type: Literal["choice", "predicate", "score"]
    choice: str | None = None
    probability: float | None = None
    score: float | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    refused: bool = False
    raw_text: str = ""
    usage: AITokenUsage = Field(default_factory=AITokenUsage)
    token_source: str = _TOKEN_COUNT_SOURCE_NONE_RUNTIME
    provider_engine: str | None = None
    model_name: str | None = None
    latency_ms: float = 0.0


class AIBaseClassifier(AIBase):
    """
    Base class for classification engines.

    A classifier answers a narrow question about one input: pick one of a
    caller-defined set of options, say whether a condition holds, or place
    the input on an ordered scale. The public methods are template methods:
    they validate the question, run the input through the PII middleware,
    truncate it to the engine's cap, call the provider hook inside the shared
    observability lifecycle (so every call emits input, output, error, and
    cost events like any other capability), and stamp the result.

    An engine that delegates to another observed client (the structured
    output adapter wraps a completions engine) sets BOOL_PROVIDER_OBSERVED
    so the template does not wrap the call a second time.
    """

    BOOL_PROVIDER_OBSERVED: ClassVar[bool] = False

    def __init__(self, model: str | None = None, **kwargs: Any) -> None:
        super().__init__(model=model)
        self.pii_middleware = AiApiPiiMiddleware()

    @property
    def capabilities(self) -> AIClassifierCapabilitiesBase:
        """Conservative defaults; engines override with their own descriptor."""
        # Normal return with the conservative descriptor.
        return AIClassifierCapabilitiesBase()

    # ── Public template methods ─────────────────────────────────────────────

    def classify_choice(
        self,
        text: str,
        choices: list[AIClassificationChoice],
        *,
        instructions: str | None = None,
        request_timeout_seconds: float | None = None,
    ) -> AIClassificationResult:
        """
        Picks one of the given options for the text.

        Args:
            text: The input to classify.
            choices: Two or more options with distinct values.
            instructions: Optional guidance on how to choose.
            request_timeout_seconds: Optional per-call provider timeout.

        Returns:
            The result with `choice` set, or `refused` True.
        """
        self._require_question_support(
            self.capabilities.supports_choice, CLASSIFY_OPERATION_CHOICE
        )
        self._validate_choices(choices)
        str_text: str = self._prepare_text(text)
        # Normal return with the observed, stamped result.
        return self._run_classification(
            CLASSIFY_OPERATION_CHOICE,
            dict_input_metadata={
                "question_type": "choice",
                "option_count": len(choices),
                "input_chars": len(str_text),
            },
            callable_execute=lambda: self._classify_choice_provider(
                str_text,
                choices,
                instructions=instructions,
                request_timeout_seconds=request_timeout_seconds,
            ),
        )

    def classify_predicate(
        self,
        text: str,
        *,
        instructions: str,
        request_timeout_seconds: float | None = None,
    ) -> AIClassificationResult:
        """
        Reports the probability that a condition holds for the text.

        Args:
            text: The input to classify.
            instructions: The condition, as a yes/no question.
            request_timeout_seconds: Optional per-call provider timeout.

        Returns:
            The result with `probability` set, or `refused` True.
        """
        self._require_question_support(
            self.capabilities.supports_predicate, CLASSIFY_OPERATION_PREDICATE
        )
        if not instructions or not instructions.strip():
            raise ValueError(
                "classify_predicate needs instructions stating the condition."
            )
        str_text: str = self._prepare_text(text)
        # Normal return with the observed, stamped result.
        return self._run_classification(
            CLASSIFY_OPERATION_PREDICATE,
            dict_input_metadata={
                "question_type": "predicate",
                "input_chars": len(str_text),
            },
            callable_execute=lambda: self._classify_predicate_provider(
                str_text,
                instructions=instructions,
                request_timeout_seconds=request_timeout_seconds,
            ),
        )

    def classify_score(
        self,
        text: str,
        levels: list[AIClassificationLevel],
        *,
        instructions: str | None = None,
        request_timeout_seconds: float | None = None,
    ) -> AIClassificationResult:
        """
        Places the text on an ordered scale.

        Args:
            text: The input to classify.
            levels: Two or more levels, lowest first, with distinct labels.
            instructions: Optional guidance on how to rate.
            request_timeout_seconds: Optional per-call provider timeout.

        Returns:
            The result with `score` in [0, 1] and `choice` set to the level
            label, or `refused` True.
        """
        self._require_question_support(
            self.capabilities.supports_score, CLASSIFY_OPERATION_SCORE
        )
        self._validate_levels(levels)
        str_text: str = self._prepare_text(text)
        # Normal return with the observed, stamped result.
        return self._run_classification(
            CLASSIFY_OPERATION_SCORE,
            dict_input_metadata={
                "question_type": "score",
                "level_count": len(levels),
                "input_chars": len(str_text),
            },
            callable_execute=lambda: self._classify_score_provider(
                str_text,
                levels,
                instructions=instructions,
                request_timeout_seconds=request_timeout_seconds,
            ),
        )

    async def aclassify_choice(
        self,
        text: str,
        choices: list[AIClassificationChoice],
        *,
        instructions: str | None = None,
        request_timeout_seconds: float | None = None,
    ) -> AIClassificationResult:
        """Async twin of classify_choice."""
        self._require_async()
        self._require_question_support(
            self.capabilities.supports_choice, CLASSIFY_OPERATION_CHOICE
        )
        self._validate_choices(choices)
        str_text: str = self._prepare_text(text)
        # Normal return with the observed, stamped result.
        return await self._arun_classification(
            CLASSIFY_OPERATION_CHOICE,
            dict_input_metadata={
                "question_type": "choice",
                "option_count": len(choices),
                "input_chars": len(str_text),
            },
            callable_execute=lambda: self._aclassify_choice_provider(
                str_text,
                choices,
                instructions=instructions,
                request_timeout_seconds=request_timeout_seconds,
            ),
        )

    async def aclassify_predicate(
        self,
        text: str,
        *,
        instructions: str,
        request_timeout_seconds: float | None = None,
    ) -> AIClassificationResult:
        """Async twin of classify_predicate."""
        self._require_async()
        self._require_question_support(
            self.capabilities.supports_predicate, CLASSIFY_OPERATION_PREDICATE
        )
        if not instructions or not instructions.strip():
            raise ValueError(
                "classify_predicate needs instructions stating the condition."
            )
        str_text: str = self._prepare_text(text)
        # Normal return with the observed, stamped result.
        return await self._arun_classification(
            CLASSIFY_OPERATION_PREDICATE,
            dict_input_metadata={
                "question_type": "predicate",
                "input_chars": len(str_text),
            },
            callable_execute=lambda: self._aclassify_predicate_provider(
                str_text,
                instructions=instructions,
                request_timeout_seconds=request_timeout_seconds,
            ),
        )

    async def aclassify_score(
        self,
        text: str,
        levels: list[AIClassificationLevel],
        *,
        instructions: str | None = None,
        request_timeout_seconds: float | None = None,
    ) -> AIClassificationResult:
        """Async twin of classify_score."""
        self._require_async()
        self._require_question_support(
            self.capabilities.supports_score, CLASSIFY_OPERATION_SCORE
        )
        self._validate_levels(levels)
        str_text: str = self._prepare_text(text)
        # Normal return with the observed, stamped result.
        return await self._arun_classification(
            CLASSIFY_OPERATION_SCORE,
            dict_input_metadata={
                "question_type": "score",
                "level_count": len(levels),
                "input_chars": len(str_text),
            },
            callable_execute=lambda: self._aclassify_score_provider(
                str_text,
                levels,
                instructions=instructions,
                request_timeout_seconds=request_timeout_seconds,
            ),
        )

    # ── Provider hooks ──────────────────────────────────────────────────────

    def _classify_choice_provider(
        self,
        str_text: str,
        list_choices: list[AIClassificationChoice],
        *,
        instructions: str | None,
        request_timeout_seconds: float | None,
    ) -> AIClassificationResult:
        """Provider hook for classify_choice; engines override."""
        self._raise_capability_unsupported(CLASSIFY_OPERATION_CHOICE)

    def _classify_predicate_provider(
        self,
        str_text: str,
        *,
        instructions: str,
        request_timeout_seconds: float | None,
    ) -> AIClassificationResult:
        """Provider hook for classify_predicate; engines override."""
        self._raise_capability_unsupported(CLASSIFY_OPERATION_PREDICATE)

    def _classify_score_provider(
        self,
        str_text: str,
        list_levels: list[AIClassificationLevel],
        *,
        instructions: str | None,
        request_timeout_seconds: float | None,
    ) -> AIClassificationResult:
        """Provider hook for classify_score; engines override."""
        self._raise_capability_unsupported(CLASSIFY_OPERATION_SCORE)

    async def _aclassify_choice_provider(
        self,
        str_text: str,
        list_choices: list[AIClassificationChoice],
        *,
        instructions: str | None,
        request_timeout_seconds: float | None,
    ) -> AIClassificationResult:
        """Async provider hook for classify_choice; engines override."""
        self._raise_capability_unsupported("aclassify_choice")

    async def _aclassify_predicate_provider(
        self,
        str_text: str,
        *,
        instructions: str,
        request_timeout_seconds: float | None,
    ) -> AIClassificationResult:
        """Async provider hook for classify_predicate; engines override."""
        self._raise_capability_unsupported("aclassify_predicate")

    async def _aclassify_score_provider(
        self,
        str_text: str,
        list_levels: list[AIClassificationLevel],
        *,
        instructions: str | None,
        request_timeout_seconds: float | None,
    ) -> AIClassificationResult:
        """Async provider hook for classify_score; engines override."""
        self._raise_capability_unsupported("aclassify_score")

    # ── Template internals ──────────────────────────────────────────────────

    def _raise_capability_unsupported(self, str_operation: str) -> NoReturn:
        """Raises the typed capability error for one operation."""
        raise AiProviderCapabilityUnsupportedError(
            f"{type(self).__name__} model '{self.model_name}' does not support "
            f"{str_operation}."
        )

    def _require_question_support(
        self, bool_supported: bool, str_operation: str
    ) -> None:
        """Raises unless the capabilities descriptor allows the question type."""
        if not bool_supported:
            self._raise_capability_unsupported(str_operation)

    def _require_async(self) -> None:
        """Raises unless the engine declares async support."""
        if not self.capabilities.supports_async:
            self._raise_capability_unsupported("async classification")

    def _validate_choices(self, list_choices: list[AIClassificationChoice]) -> None:
        """Rejects fewer than two options, duplicate values, or too many."""
        if len(list_choices) < 2:
            raise ValueError("classify_choice needs at least two choices.")
        list_values: list[str] = [choice.value for choice in list_choices]
        if len(set(list_values)) != len(list_values):
            raise ValueError("classify_choice choices must have distinct values.")
        int_max: int | None = self.capabilities.max_choices
        if int_max is not None and len(list_choices) > int_max:
            raise ValueError(
                f"{type(self).__name__} accepts at most {int_max} choices; "
                f"{len(list_choices)} given."
            )

    def _validate_levels(self, list_levels: list[AIClassificationLevel]) -> None:
        """Rejects fewer than two levels or duplicate labels."""
        if len(list_levels) < 2:
            raise ValueError("classify_score needs at least two levels.")
        list_labels: list[str] = [level.label for level in list_levels]
        if len(set(list_labels)) != len(list_labels):
            raise ValueError("classify_score levels must have distinct labels.")

    def _prepare_text(self, text: str) -> str:
        """Validates, redacts, and truncates the input."""
        if not text or not text.strip():
            raise ValueError("Text to classify cannot be empty or None")
        str_text: str = self.pii_middleware.process_input(text)
        # Normal return with the capped text.
        return truncate_classification_text(str_text, self.capabilities.max_input_chars)

    def _run_classification(
        self,
        str_operation: str,
        *,
        dict_input_metadata: dict[str, ObservabilityMetadataValue],
        callable_execute: Callable[[], AIClassificationResult],
    ) -> AIClassificationResult:
        """
        Runs one provider hook inside the observability lifecycle and stamps
        the result.

        Args:
            str_operation: Public operation name, for events.
            dict_input_metadata: Scalar request metadata for the input event.
            callable_execute: The provider hook, already bound to its inputs.

        Returns:
            The stamped result.
        """
        float_started: float = time.perf_counter()
        result: AIClassificationResult
        if self.BOOL_PROVIDER_OBSERVED:
            # The delegate engine emits its own events; wrapping again would
            # double-count the call's cost.
            result = callable_execute()
        else:
            result = self._execute_provider_call_with_observability(
                capability=self.CLIENT_TYPE_CLASSIFICATION,
                operation=str_operation,
                dict_input_metadata=dict_input_metadata,
                callable_execute=callable_execute,
                callable_build_result_summary=self._build_classification_result_summary,
            )
        # Normal return with the stamped result.
        return self._stamp_result(result, str_operation, float_started)

    async def _arun_classification(
        self,
        str_operation: str,
        *,
        dict_input_metadata: dict[str, ObservabilityMetadataValue],
        callable_execute: Callable[[], Any],
    ) -> AIClassificationResult:
        """Async twin of _run_classification."""
        float_started: float = time.perf_counter()
        result: AIClassificationResult
        if self.BOOL_PROVIDER_OBSERVED:
            result = await callable_execute()
        else:
            result = await self._execute_provider_acall_with_observability(
                capability=self.CLIENT_TYPE_CLASSIFICATION,
                operation=str_operation,
                dict_input_metadata=dict_input_metadata,
                callable_execute=callable_execute,
                callable_build_result_summary=self._build_classification_result_summary,
            )
        # Normal return with the stamped result.
        return self._stamp_result(result, str_operation, float_started)

    def _stamp_result(
        self,
        result: AIClassificationResult,
        str_operation: str,
        float_started: float,
    ) -> AIClassificationResult:
        """Fills engine, model, and latency; warns when the call was unmetered."""
        if result.provider_engine is None:
            result.provider_engine = self._resolve_observability_provider_engine()
        if result.model_name is None:
            result.model_name = self.model_name
        if result.latency_ms == 0.0:
            result.latency_ms = (time.perf_counter() - float_started) * 1000.0
        if result.token_source == _TOKEN_COUNT_SOURCE_NONE_RUNTIME:
            _LOGGER.warning(
                "CLASSIFIER UNMETERED: %s on %s:%s returned no token counts.",
                str_operation,
                result.provider_engine,
                result.model_name,
                extra={
                    "ai_classifier_event": CLASSIFIER_EVENT_UNMETERED,
                    "operation": str_operation,
                    "provider_engine": result.provider_engine,
                    "model_name": result.model_name,
                },
            )
        # Normal return with the stamped result.
        return result

    def _build_classification_result_summary(
        self, result: AIClassificationResult, provider_elapsed_ms: float
    ) -> AiApiCallResultSummaryModel:
        """Summarizes one result for the output and cost events."""
        # Normal return with the summary the cost event prices from.
        return AiApiCallResultSummaryModel(
            provider_elapsed_ms=provider_elapsed_ms,
            input_token_count=result.usage.input_tokens,
            input_token_count_source=result.token_source,
            output_token_count=result.usage.output_tokens,
            output_token_count_source=(
                result.token_source
                if result.usage.output_tokens is not None
                else _TOKEN_COUNT_SOURCE_NONE_RUNTIME
            ),
            provider_prompt_tokens=result.usage.input_tokens,
            provider_completion_tokens=result.usage.output_tokens,
            provider_reasoning_tokens=result.usage.reasoning_tokens,
            provider_total_tokens=result.usage.total_tokens,
            finish_reason="refusal" if result.refused else "complete",
            dict_metadata={
                "question_type": result.question_type,
                "token_count_source": result.token_source,
                "refused": result.refused,
            },
        )


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


async def aiter_in_thread(iterator: Iterator[Any]) -> AsyncIterator[Any]:
    """
    Consumes a blocking iterator on one worker thread and yields its items
    on the event loop.

    The whole iteration runs in one copied context, so a request cost scope
    or observability tags the iterator opens stay valid from its first item
    to its last. Stopping early (break, cancel) asks the worker to close the
    iterator after its current item.

    Args:
        iterator: A blocking iterator, such as send_prompt_streaming's.

    Yields:
        The iterator's items, in order.

    Raises:
        Whatever the iterator raises.
    """
    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[bool, Any]] = asyncio.Queue()
    stop: list[bool] = [False]

    def _pump() -> None:
        try:
            # Loop over the blocking iterator on this worker thread.
            for item in iterator:
                loop.call_soon_threadsafe(queue.put_nowait, (False, item))
                if stop[0]:
                    close: Any = getattr(iterator, "close", None)
                    if callable(close):
                        close()
                    break
            loop.call_soon_threadsafe(queue.put_nowait, (True, None))
        except BaseException as exception:
            loop.call_soon_threadsafe(queue.put_nowait, (True, exception))

    future: asyncio.Future[None] = loop.run_in_executor(
        None, contextvars.copy_context().run, _pump
    )
    try:
        # Loop over the worker's items until it reports the end.
        while True:
            bool_done, value = await queue.get()
            if bool_done:
                if value is not None:
                    raise value
                break
            yield value
    finally:
        stop[0] = True
        await future


# True while an outer completions call owns the result envelope, so the
# clients it calls (a chain's candidates, a router's tiers) leave the
# envelope to it.
ENVELOPE_ACTIVE: ContextVar[bool] = ContextVar(
    "ai_api_unified_envelope_active", default=False
)
# Reserved provider_options key: pins one conversation or structured call
# to a model; see completions/ai_model_pin.py.
PROVIDER_OPTION_MODEL: str = "model"


def _split_model_option(
    provider_options: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Removes the reserved model key; returns (the rest, the pin or None)."""
    if not provider_options or PROVIDER_OPTION_MODEL not in provider_options:
        # Early return: no pin on this call.
        return provider_options, None
    dict_options: dict[str, Any] = dict(provider_options)
    raw_pin: Any = dict_options.pop(PROVIDER_OPTION_MODEL)
    str_pin: str | None = (
        str(raw_pin) if raw_pin is not None and str(raw_pin).strip() else None
    )
    # Normal return with the options left for the engine and the pin.
    return dict_options or None, str_pin


def enveloped_completions_call(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Makes a turn- or structured-result method fill its result envelope.

    The outermost call removes `provider_options["model"]`, pins the call
    when it was set, records its own ledger, and stamps provider, route,
    pin, and cost onto the result. Calls made inside it (a chain's
    candidates, a router's tiers) run the method unchanged.
    """

    @functools.wraps(func)
    def _wrapper(self: "AIBaseCompletions", *args: Any, **kwargs: Any) -> Any:
        dict_options, str_pin = _split_model_option(kwargs.get("provider_options"))
        if "provider_options" in kwargs:
            kwargs["provider_options"] = dict_options
        if ENVELOPE_ACTIVE.get():
            # Early return: an outer call owns the envelope.
            return func(self, *args, **kwargs)
        from ai_api_unified.completions.ai_model_pin import model_pin_scope

        pin, target = self._model_pin_for_call(str_pin)
        report_before: AIRequestCostReport | None = target.last_request_cost
        token: Any = ENVELOPE_ACTIVE.set(True)
        try:
            with (
                model_pin_scope(pin, func.__name__, config_id=self.config_id),
                capture_request_calls(func.__name__) as capture,
            ):
                result: Any = (
                    func(self, *args, **kwargs)
                    if target is self
                    else getattr(target, func.__name__)(*args, **kwargs)
                )
        finally:
            ENVELOPE_ACTIVE.reset(token)
        self._adopt_request_cost(target, report_before)
        # Normal return with the stamped result.
        return self._stamp_envelope(result, capture, pin)

    return _wrapper


def aenveloped_completions_call(func: Callable[..., Any]) -> Callable[..., Any]:
    """Async twin of enveloped_completions_call."""

    @functools.wraps(func)
    async def _wrapper(self: "AIBaseCompletions", *args: Any, **kwargs: Any) -> Any:
        dict_options, str_pin = _split_model_option(kwargs.get("provider_options"))
        if "provider_options" in kwargs:
            kwargs["provider_options"] = dict_options
        if ENVELOPE_ACTIVE.get():
            # Early return: an outer call owns the envelope.
            return await func(self, *args, **kwargs)
        from ai_api_unified.completions.ai_model_pin import model_pin_scope

        pin, target = self._model_pin_for_call(str_pin)
        report_before: AIRequestCostReport | None = target.last_request_cost
        token: Any = ENVELOPE_ACTIVE.set(True)
        try:
            with (
                model_pin_scope(pin, func.__name__, config_id=self.config_id),
                capture_request_calls(func.__name__) as capture,
            ):
                result: Any = await (
                    func(self, *args, **kwargs)
                    if target is self
                    else getattr(target, func.__name__)(*args, **kwargs)
                )
        finally:
            ENVELOPE_ACTIVE.reset(token)
        self._adopt_request_cost(target, report_before)
        # Normal return with the stamped result.
        return self._stamp_envelope(result, capture, pin)

    return _wrapper


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
    # Reserved provider_options key: a per-call thinking level on
    # send_conversation and send_structured_output; stripped before the
    # options reach the provider request.
    PROVIDER_OPTION_THINKING: ClassVar[str] = "thinking_level"
    # Reserved per-call purpose hint (for example "summarize"), carried as
    # the `purpose` tag on every event and the request cost report.
    PROVIDER_OPTION_PURPOSE: ClassVar[str] = "purpose"
    # provider_options paths that carry this engine's own thinking control.
    # A call that sets one keeps it, and the layer records the level as set
    # by the caller instead of applying its own.
    TUPLE_NATIVE_THINKING_OPTION_PATHS: ClassVar[tuple[tuple[str, ...], ...]] = ()

    _thinking_level: AIThinkingLevel | None = None
    _last_thinking: AIThinkingResolution | None = None
    # Resolutions this client already logged, so a steady state (an
    # always-on model asked for off, a model with no control) is one line per
    # client rather than one per call.
    _set_logged_thinking: set[tuple[str, str | None, str | None]] | None = None
    _set_warned_native_thinking: set[str] | None = None
    # What a single configured client needs to serve a per-call model pin;
    # set by the factory (see _configure_model_pins). Wrappers handle pins
    # themselves.
    _pin_candidate: Any = None
    _pin_builder: Callable[[Any], "AIBaseCompletions"] | None = None
    _pin_fallback_on: frozenset[Any] | None = None
    _pin_cooldown: Any = None
    _pin_chain: "AIBaseCompletions | None" = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pii_middleware = AiApiPiiMiddleware()

    # ── Thinking levels ──────────────────────────────────────────────────────

    @property
    def thinking_level(self) -> AIThinkingLevel:
        """
        The thinking level this client requests on every call unless a call
        overrides it: the `thinking=` factory argument, else
        `completions.thinking` in models.json, else medium. Set it to change
        the default for later calls.
        """
        if self._thinking_level is None:
            from ai_api_unified.util.env_settings import EnvSettings

            self._thinking_level = (
                parse_thinking_level(
                    EnvSettings().get_setting(COMPLETIONS_THINKING_KEY, None)
                )
                or DEFAULT_THINKING_LEVEL
            )
        # Normal return with the configured level.
        return self._thinking_level

    @thinking_level.setter
    def thinking_level(self, value: AIThinkingLevel | str | None) -> None:
        self._thinking_level = parse_thinking_level(value)
        # Normal return after storing the level (None re-reads configuration).
        return None

    @property
    def last_thinking(self) -> AIThinkingResolution | None:
        """
        The thinking level requested and applied on this client's last call,
        for send_prompt and the other text-returning methods whose result
        has nowhere to carry it. Turn and structured results carry their own
        `thinking` field.
        """
        # Normal return with the last resolution.
        return self._last_thinking

    def _thinking_for_call(
        self, other_params: AICompletionsPromptParamsBase | None = None
    ) -> AIThinkingResolution:
        """
        Resolves the thinking level for one provider call.

        Precedence: the per-call level (`other_params.thinking`, or the
        reserved `thinking_level` provider option the templates stash for
        the engine), then the client's level. The model's honored levels
        decide what is applied; a difference is recorded on the resolution
        and logged once per call.

        Args:
            other_params: The call's prompt parameters, if any.

        Returns:
            The resolution, also kept as `last_thinking`.
        """
        requested: AIThinkingLevel | None = (
            other_params.thinking if other_params is not None else None
        )
        if requested is None:
            requested = THINKING_CALL_OVERRIDE.get()
        if requested is None:
            requested = self.thinking_level
        # Capabilities may be a test double or unbuilt on a partially
        # constructed engine; treat anything but a real descriptor as none.
        try:
            thinking_caps: Any = getattr(self.capabilities, "thinking", None)
        except Exception:
            thinking_caps = None
        if not isinstance(thinking_caps, AIThinkingCapabilities):
            thinking_caps = None
        resolution: AIThinkingResolution = resolve_thinking_level(
            requested, thinking_caps
        )
        str_native: str | None = THINKING_NATIVE_OPTION.get()
        if str_native is not None:
            # The caller's provider field decides; apply nothing of our own.
            resolution = resolution.model_copy(
                update={
                    "applied": None,
                    "exact": False,
                    "reason": THINKING_REASON_PROVIDER_OPTION,
                }
            )
            self._last_thinking = resolution
            note_call_thinking(resolution)
            self._warn_native_thinking_once(str_native)
            # Early return with the caller-overridden resolution.
            return resolution
        self._last_thinking = resolution
        note_call_thinking(resolution)
        if not resolution.exact:
            self._log_thinking_once(resolution, "nearest_level")
        # Normal return with the resolution the engine applies.
        return resolution

    @classmethod
    @contextmanager
    def _purpose_scope(cls, provider_options: dict[str, Any] | None) -> Iterator[None]:
        """
        Tags everything inside with the call's purpose hint, for wrappers
        (fallback chains, the router) that open the request before the
        engine strips the key.
        """
        str_purpose: str = str(
            (provider_options or {}).get(cls.PROVIDER_OPTION_PURPOSE) or ""
        ).strip()
        token: Any = (
            push_observability_tags({"purpose": str_purpose}) if str_purpose else None
        )
        try:
            yield
        finally:
            if token is not None:
                reset_observability_context(token)

    @classmethod
    def _push_purpose_option(
        cls, provider_options: dict[str, Any] | None
    ) -> tuple[dict[str, Any] | None, Any]:
        """
        Removes the reserved purpose hint from provider_options and pushes it
        as the `purpose` observability tag.

        Returns:
            (the options without the key, the context token to reset, or
            None when the call names no purpose).
        """
        if not provider_options or cls.PROVIDER_OPTION_PURPOSE not in provider_options:
            # Early return: no purpose on this call.
            return provider_options, None
        dict_options: dict[str, Any] = dict(provider_options)
        str_purpose: str = str(
            dict_options.pop(cls.PROVIDER_OPTION_PURPOSE) or ""
        ).strip()
        if not str_purpose:
            # Early return: a blank purpose is no purpose.
            return dict_options or None, None
        # Normal return with the stripped options and the tag's reset token.
        return dict_options or None, push_observability_tags({"purpose": str_purpose})

    def _native_thinking_option(
        self, provider_options: dict[str, Any] | None
    ) -> str | None:
        """
        The provider_options path of a native thinking field this call sets,
        as a dotted name, or None.
        """
        if not provider_options:
            # Early return: no options.
            return None
        # Loop over the engine's native paths for the first one present.
        for tuple_path in self.TUPLE_NATIVE_THINKING_OPTION_PATHS:
            node: Any = provider_options
            for str_key in tuple_path:
                if not isinstance(node, dict) or str_key not in node:
                    node = None
                    break
                node = node[str_key]
            if node is not None:
                # Early return with the path found.
                return ".".join(tuple_path)
        # Normal return: no native thinking field.
        return None

    def _warn_native_thinking_once(self, str_native: str) -> None:
        """One WARNING per client per native field, naming the neutral option."""
        if self._set_warned_native_thinking is None:
            self._set_warned_native_thinking = set()
        if str_native in self._set_warned_native_thinking:
            # Early return: already reported for this client.
            return None
        self._set_warned_native_thinking.add(str_native)
        _LOGGER.warning(
            "THINKING: provider_options['%s'] sets %s's thinking directly, so the "
            "layer applies no level and records the call as caller-set. Use the "
            "provider-neutral provider_options['%s'] or other_params.thinking.",
            str_native,
            type(self).__name__,
            self.PROVIDER_OPTION_THINKING,
            extra={
                "ai_thinking_event": "provider_option",
                "model": self.model_name,
                "provider_option": str_native,
            },
        )
        # Normal return after the one-time warning.
        return None

    def _log_thinking_once(
        self, resolution: AIThinkingResolution, str_event: str
    ) -> None:
        """
        Logs one line per client per distinct resolution: INFO when a level
        was substituted or overridden, DEBUG when the model has no control.
        """
        if self._set_logged_thinking is None:
            self._set_logged_thinking = set()
        tuple_key: tuple[str, str | None, str | None] = (
            resolution.requested.value,
            resolution.applied.value if resolution.applied else None,
            resolution.reason,
        )
        if tuple_key in self._set_logged_thinking:
            # Early return: already reported for this client.
            return None
        self._set_logged_thinking.add(tuple_key)
        int_level: int = (
            logging.DEBUG
            if resolution.reason == THINKING_REASON_NO_CONTROL
            else logging.INFO
        )
        _LOGGER.log(
            int_level,
            "THINKING: %s on %s applies %s for requested level %s (%s).",
            type(self).__name__,
            self.model_name,
            resolution.applied.value if resolution.applied else "none",
            resolution.requested.value,
            resolution.reason,
            extra={
                "ai_thinking_event": str_event,
                "model": self.model_name,
                "reason": resolution.reason,
                **thinking_event_fields(resolution),
            },
        )
        # Normal return after the one-time line.
        return None

    def _thinking_applied_override(
        self, applied: AIThinkingLevel | None, reason: str
    ) -> None:
        """
        Records that the engine applied a level other than the resolved one,
        for an engine-specific reason (a forced tool choice the provider
        rejects with thinking on, a max_tokens too small for a budget).
        """
        if self._last_thinking is not None:
            self._last_thinking = self._last_thinking.model_copy(
                update={"applied": applied, "exact": False, "reason": reason}
            )
            note_call_thinking(self._last_thinking)
            self._log_thinking_once(self._last_thinking, "engine_override")
        # Normal return after recording the override.
        return None

    def _stamp_thinking(self, result: Any) -> Any:
        """Copies the last resolution onto a turn or structured result lacking one."""
        if (
            isinstance(result, (AITurnResult, AIStructuredOutputResult))
            and getattr(result, "thinking", None) is None
            and self._last_thinking is not None
        ):
            try:
                result.thinking = self._last_thinking
            except (AttributeError, TypeError):
                # A test double standing in for the result has no field to set.
                pass
        # Normal return with the stamped result.
        return result

    @staticmethod
    def _split_thinking_option(
        provider_options: dict[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, AIThinkingLevel | None]:
        """
        Removes the reserved per-call thinking level from provider_options.

        Args:
            provider_options: The call's provider_options, if any.

        Returns:
            (the options without the key, the parsed level or None).
        """
        if not provider_options or (
            AIBaseCompletions.PROVIDER_OPTION_THINKING not in provider_options
        ):
            # Early return: nothing to strip.
            return provider_options, None
        dict_options: dict[str, Any] = dict(provider_options)
        level: AIThinkingLevel | None = parse_thinking_level(
            dict_options.pop(AIBaseCompletions.PROVIDER_OPTION_THINKING)
        )
        # Normal return with the stripped options and the level.
        return dict_options or None, level

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
        # The call context snapshots `last_thinking` before the provider call
        # runs, so the text paths resolve the level here, ahead of the code
        # that builds the request; the engine's later resolution of the same
        # call is identical. An engine-side override (a forced tool choice,
        # a prefill, a max_tokens too small for a budget) lands after this
        # point: it is on `last_thinking` and the result, while the input
        # event carries the resolution before it.
        self._thinking_for_call(other_params)
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
            provider_cache_write_5m_tokens=observed_result.provider_cache_write_5m_tokens,
            provider_cache_write_1h_tokens=observed_result.provider_cache_write_1h_tokens,
            provider_reasoning_tokens=observed_result.provider_reasoning_tokens,
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
        cache_write_5m_tokens: int = 0,
        cache_write_1h_tokens: int = 0,
    ) -> float:
        """
        Return the USD cost for measured token usage using split model rates.

        Uses `capabilities.pricing` (per-1M input/output/cached rates). This is
        the accurate replacement for the blended `price_per_1k_tokens`.

        Args:
            input_tokens: Non-cached input tokens billed at the input rate.
            output_tokens: Output tokens billed at the output rate.
            cached_input_tokens: Input tokens billed at the cached-input rate.
            cache_write_5m_tokens: Tokens written to a 5-minute-TTL prompt
                cache, billed at the model's 5-minute write rate. Not a subset
                of input_tokens; pass them in addition.
            cache_write_1h_tokens: Tokens written to a 1-hour-TTL prompt cache,
                billed at the model's 1-hour write rate.

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
                cache_write_5m_tokens=cache_write_5m_tokens,
                cache_write_1h_tokens=cache_write_1h_tokens,
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
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_response_tokens: int | None = None,
        request_timeout_seconds: float | None = None,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> str:
        """
        Sends a prompt to the completions engine and returns the result as a string.

        Omitting the optional parameters leaves historical behavior unchanged.
        Engines map them to their provider's native fields; an engine that has
        not yet mapped max_response_tokens or request_timeout_seconds raises
        AiProviderCapabilityUnsupportedError when they are supplied.

        Args:
            prompt: The text prompt to send.
            system_prompt: Optional persistent instructions, separate from the
                user prompt. Overrides other_params.system_prompt when both are
                supplied.
            max_response_tokens: Optional response token budget override.
            request_timeout_seconds: Optional per-call provider timeout.
            other_params: Optional provider-specific parameters.
        """

    # ── Result envelope and model pins ───────────────────────────────────────

    def send_prompt_with_result(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_response_tokens: int | None = None,
        request_timeout_seconds: float | None = None,
        other_params: AICompletionsPromptParamsBase | None = None,
        model: str | None = None,
    ) -> AIPromptResult:
        """
        Sends a prompt like send_prompt, and returns the text with which
        provider and model answered, the router tier and method, the
        thinking level, the token usage, and the cost.

        Args:
            prompt: The text prompt to send.
            system_prompt: Optional persistent instructions.
            max_response_tokens: Optional response token budget override.
            request_timeout_seconds: Optional per-call provider timeout.
            other_params: Optional provider-specific parameters.
            model: Optional pin, `engine:model` or a configured model id:
                this call skips routing and starts on that model, keeping
                fallback, metering, logging, and redaction. See
                completions/ai_model_pin.py.

        Returns:
            The AIPromptResult for this call.

        Raises:
            ValueError: The pin is malformed, names an unknown bare model,
                or names a model the models.json policy denies.
            AiProviderConfigurationError: The client cannot build the
                pinned model (it was not built by AIFactory).
        """
        from ai_api_unified.completions.ai_model_pin import model_pin_scope

        pin, target = self._model_pin_for_call(model)
        report_before: AIRequestCostReport | None = target.last_request_cost
        with (
            model_pin_scope(pin, "send_prompt_with_result", config_id=self.config_id),
            capture_request_calls("send_prompt_with_result") as capture,
        ):
            str_text: str = target.send_prompt(
                prompt,
                system_prompt=system_prompt,
                max_response_tokens=max_response_tokens,
                request_timeout_seconds=request_timeout_seconds,
                other_params=other_params,
            )
        self._adopt_request_cost(target, report_before)
        # Normal return with the call's envelope.
        return self._prompt_result(str_text, capture, pin)

    async def asend_prompt_with_result(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_response_tokens: int | None = None,
        request_timeout_seconds: float | None = None,
        other_params: AICompletionsPromptParamsBase | None = None,
        model: str | None = None,
    ) -> AIPromptResult:
        """Async twin of send_prompt_with_result."""
        from ai_api_unified.completions.ai_model_pin import model_pin_scope

        pin, target = self._model_pin_for_call(model)
        report_before: AIRequestCostReport | None = target.last_request_cost
        with (
            model_pin_scope(pin, "asend_prompt_with_result", config_id=self.config_id),
            capture_request_calls("asend_prompt_with_result") as capture,
        ):
            str_text: str = await target.asend_prompt(
                prompt,
                system_prompt=system_prompt,
                max_response_tokens=max_response_tokens,
                request_timeout_seconds=request_timeout_seconds,
                other_params=other_params,
            )
        self._adopt_request_cost(target, report_before)
        # Normal return with the call's envelope.
        return self._prompt_result(str_text, capture, pin)

    def _configure_model_pins(
        self,
        *,
        candidate: Any,
        client_builder: Callable[[Any], "AIBaseCompletions"],
        fallback_on: frozenset[Any],
        cooldown: Any,
    ) -> None:
        """
        Lets a single configured client serve per-call model pins: a pinned
        call starts on the pinned model, built with `client_builder`, and
        falls back to this client.

        Args:
            candidate: This client's own AIFallbackCandidate.
            client_builder: Builds a client for the pinned candidate.
            fallback_on: Reasons that move a pinned call back to this client.
            cooldown: Rest policy for the pinned model after a trigger failure.
        """
        self._pin_candidate = candidate
        self._pin_builder = client_builder
        self._pin_fallback_on = frozenset(fallback_on)
        self._pin_cooldown = cooldown
        self._pin_chain = None
        # Normal return after recording the pin support.
        return None

    def _known_pin_candidates(self) -> list[Any]:
        """Configured candidates a bare-model pin may name."""
        # Normal return with this client's own candidate, when the factory set it.
        return [self._pin_candidate] if self._pin_candidate is not None else []

    def _model_pin_target(self, pin: Any) -> "AIBaseCompletions | None":
        """
        The client that serves a call pinned to `pin`, or None when this
        client cannot serve it. Wrappers return themselves; a single
        configured client returns a chain of the pinned model then itself.
        """
        if self._pin_candidate is not None and self._pin_builder is not None:
            if self._pin_chain is None:
                from ai_api_unified.completions.ai_fallback_completions import (
                    AiFallbackCompletions,
                )

                chain: AiFallbackCompletions = AiFallbackCompletions(
                    primary=self,
                    primary_candidate=self._pin_candidate,
                    fallback_candidates=(),
                    client_builder=self._pin_builder,
                    fallback_on=self._pin_fallback_on or frozenset(),
                    cooldown=self._pin_cooldown,
                )
                chain._dict_config_tags = self._dict_config_tags
                self._pin_chain = chain
            # Early return with the pinned-model-then-this-client chain.
            return self._pin_chain
        if pin.model and pin.model == (self.model_name or ""):
            # Early return: the pin names this client's own model.
            return self
        # Normal return: nothing here can build the pinned model.
        return None

    def _model_pin_for_call(
        self, str_model: str | None
    ) -> tuple[Any, "AIBaseCompletions"]:
        """
        Resolves a call's pin and the client that serves it.

        Returns:
            (the pin's AIFallbackCandidate or None, the client to call).

        Raises:
            ValueError: A malformed or unknown pin.
            AiProviderConfigurationError: No client here can serve the pin.
        """
        if str_model is None:
            # Early return: not pinned; this client serves the call.
            return None, self
        from ai_api_unified.completions.ai_model_pin import resolve_model_pin

        pin: Any = resolve_model_pin(str_model, self._known_pin_candidates())
        target: AIBaseCompletions | None = self._model_pin_target(pin)
        if target is None:
            raise AiProviderConfigurationError(
                f"{type(self).__name__} cannot pin a call to {pin.label}: it was "
                "not built by AIFactory, so it cannot build another model. Build "
                "it with AIFactory.get_ai_completions_client(), or construct a "
                f"client for {pin.label} directly."
            )
        # Normal return with the pin and its serving client.
        return pin, target

    def _adopt_request_cost(
        self,
        target: "AIBaseCompletions",
        report_before: AIRequestCostReport | None,
    ) -> None:
        """Keeps last_request_cost current when another client ran the request."""
        if target is not self and target.last_request_cost is not report_before:
            self._last_request_cost = target.last_request_cost
        # Normal return after adopting the report, if there was a new one.
        return None

    def _served_fields(self, capture: RequestCallCapture, pin: Any) -> dict[str, Any]:
        """
        What the call's own ledger says about the answer: provider, engine,
        model, route, pin, thinking, the served call's usage, and the cost.
        """
        capture.mark_served()
        report: AIRequestCostReport = capture.build_report(REQUEST_STATUS_OK)
        served: AIRequestCostLineItem | None = next(
            (item for item in reversed(report.calls) if item.served), None
        )
        # A wrapper's record names the chain's engine token and the router's
        # tier; the line item fills in what no wrapper recorded.
        str_engine: str | None = capture.served_engine
        if str_engine is None and self._pin_candidate is not None:
            str_engine = self._pin_candidate.engine
        if str_engine is None and served is not None:
            str_engine = served.provider_engine
        str_model: str | None = served.model if served is not None else None
        if not str_model:
            str_model = capture.served_model or self.model_name
        str_tier: str | None = capture.route_tier or (
            served.route_tier if served is not None else None
        )
        str_method: str | None = capture.route_method or (
            served.route_method if served is not None else None
        )
        if str_method is None and pin is not None:
            from ai_api_unified.completions.ai_model_pin import ROUTE_METHOD_MODEL_PIN

            str_method = ROUTE_METHOD_MODEL_PIN
        usage: AITokenUsage = AITokenUsage()
        if served is not None:
            usage = AITokenUsage(
                input_tokens=served.input_tokens,
                output_tokens=served.output_tokens,
                cached_input_tokens=served.cached_input_tokens,
                total_tokens=(
                    served.input_tokens + served.output_tokens
                    if served.input_tokens is not None
                    and served.output_tokens is not None
                    else None
                ),
                reasoning_tokens=served.reasoning_tokens,
                cache_write_5m_tokens=served.cache_write_5m_tokens,
                cache_write_1h_tokens=served.cache_write_1h_tokens,
            )
        # Normal return with the fields every result envelope carries.
        return {
            "provider": served.provider if served is not None else None,
            "provider_engine": str_engine,
            "model_name": str_model,
            "route_tier": str_tier,
            "route_method": str_method,
            "model_pin": pin.label if pin is not None else None,
            "thinking": capture.thinking,
            "usage": usage,
            "cost": report,
        }

    def _prompt_result(
        self, str_text: str, capture: RequestCallCapture, pin: Any
    ) -> AIPromptResult:
        """Builds the send_prompt_with_result envelope from the call's ledger."""
        # Normal return with the envelope.
        return AIPromptResult(text=str_text, **self._served_fields(capture, pin))

    def _stamp_envelope(
        self, result: Any, capture: RequestCallCapture, pin: Any
    ) -> Any:
        """
        Copies the call's provider, route, pin, and cost onto a turn or
        structured result. The provider's own usage stays; engine, model,
        and thinking are filled only where the result has none, except that
        the ledger's thinking wins because it belongs to this call alone.
        """
        if not isinstance(result, (AITurnResult, AIStructuredOutputResult)):
            # Early return: a test double or another shape carries no envelope.
            return result
        dict_fields: dict[str, Any] = self._served_fields(capture, pin)
        dict_update: dict[str, Any] = {
            "provider": dict_fields["provider"],
            "route_tier": dict_fields["route_tier"],
            "route_method": dict_fields["route_method"],
            "model_pin": dict_fields["model_pin"],
            "cost": dict_fields["cost"],
        }
        if dict_fields["thinking"] is not None:
            dict_update["thinking"] = dict_fields["thinking"]
        try:
            if result.provider_engine is None:
                dict_update["provider_engine"] = dict_fields["provider_engine"]
            if result.model_name is None:
                dict_update["model_name"] = dict_fields["model_name"]
            stamped: Any = result.model_copy(update=dict_update)
        except (AttributeError, TypeError):
            # Early return: a test double standing in for the result has no
            # fields to fill.
            return result
        # Normal return with the stamped copy.
        return stamped

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

    async def asend_prompt_streaming(
        self,
        prompt: str,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> AsyncIterator[str]:
        """
        Async twin of send_prompt_streaming: yields the same chunks on the
        event loop. The engine's stream runs on a worker thread, so every
        engine, fallback chain, and router that streams supports it, with
        the same failover, metering, and logging.
        """
        # Loop over the blocking stream's chunks as they arrive.
        async for str_chunk in aiter_in_thread(
            self.send_prompt_streaming(prompt, other_params=other_params)
        ):
            yield str_chunk

    async def acount_tokens(
        self,
        prompt: str,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> int:
        """Async twin of count_tokens; runs on a worker thread."""
        # Normal return with the token count.
        return await asyncio.to_thread(
            self.count_tokens, prompt, other_params=other_params
        )

    async def astrict_schema_prompt(
        self,
        prompt: str,
        response_model: Type[AIStructuredPrompt],
        max_response_tokens: int = STRUCTURED_DEFAULT_MAX_RESPONSE_TOKENS,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> AIStructuredPrompt:
        """Async twin of strict_schema_prompt; runs on a worker thread."""
        # Normal return with the validated model.
        return await asyncio.to_thread(
            self.strict_schema_prompt,
            prompt,
            response_model,
            max_response_tokens,
            other_params=other_params,
        )

    async def apreview_route(
        self,
        prompt: str | None = None,
        *,
        messages: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        max_response_tokens: int | None = None,
        provider_options: dict[str, Any] | None = None,
        bool_count_tokens: bool = False,
    ) -> AIRoutePreview:
        """Async twin of preview_route; runs on a worker thread."""
        # Normal return with the preview.
        return await asyncio.to_thread(
            self.preview_route,
            prompt,
            messages=messages,
            system_prompt=system_prompt,
            max_response_tokens=max_response_tokens,
            provider_options=provider_options,
            bool_count_tokens=bool_count_tokens,
        )

    async def asubmit_batch(self, requests: list[AIBatchRequestItem]) -> AIBatchJob:
        """Async twin of submit_batch; runs on a worker thread."""
        # Normal return with the submitted job.
        return await asyncio.to_thread(self.submit_batch, requests)

    async def aget_batch(self, batch: str | AIBatchJob) -> AIBatchJob:
        """Async twin of get_batch; runs on a worker thread."""
        # Normal return with the job's state.
        return await asyncio.to_thread(self.get_batch, batch)

    async def aget_batch_results(
        self, batch: str | AIBatchJob
    ) -> list[AIBatchResultItem]:
        """Async twin of get_batch_results; runs on a worker thread."""
        # Normal return with the results.
        return await asyncio.to_thread(self.get_batch_results, batch)

    async def acancel_batch(self, batch: str | AIBatchJob) -> AIBatchJob:
        """Async twin of cancel_batch; runs on a worker thread."""
        # Normal return with the canceled job.
        return await asyncio.to_thread(self.cancel_batch, batch)

    async def arun_batch(
        self,
        requests: list[AIBatchRequestItem],
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float | None = None,
    ) -> list[AIBatchResultItem]:
        """
        Async twin of run_batch: submits, polls, and collects on a worker
        thread, so the event loop stays free while the batch runs.
        """
        # Normal return with the results.
        return await asyncio.to_thread(
            self.run_batch,
            requests,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

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

    def preview_route(
        self,
        prompt: str | None = None,
        *,
        messages: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        max_response_tokens: int | None = None,
        provider_options: dict[str, Any] | None = None,
        bool_count_tokens: bool = False,
    ) -> AIRoutePreview:
        """
        Where this request would go and what it would cost, without sending
        it. A plain engine answers for itself; a fallback chain names the
        candidate it would try first, and a router runs its classifier (the
        only call a preview makes, and it is billed) to pick the tier.

        Args:
            prompt: The prompt text, for a send_prompt or structured call.
            messages: The conversation, for a conversation call.
            system_prompt: The system prompt, counted toward the estimate.
            max_response_tokens: The output cap the estimate assumes; the
                engine's own default when omitted.
            provider_options: The call's options; the router reads `route`
                and `route_floor` from them.
            bool_count_tokens: True asks the provider to count the prompt's
                tokens (one free API call where supported) instead of the
                local character estimate.

        Returns:
            The preview.
        """
        # Normal return with this engine's own preview.
        return self._preview_for_client(
            self,
            prompt=prompt,
            messages=messages,
            system_prompt=system_prompt,
            max_response_tokens=max_response_tokens,
            bool_count_tokens=bool_count_tokens,
        )

    @staticmethod
    def _preview_text(
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        system_prompt: str | None,
    ) -> str:
        """Every piece of text the request would send, for the estimate."""
        list_parts: list[str] = [system_prompt or "", prompt or ""]
        # Loop over messages so string and block content both count.
        for message in messages or []:
            content: Any = message.get("content")
            if isinstance(content, str):
                list_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        list_parts.append(block["text"])
        # Normal return with the joined text.
        return "\n".join(part for part in list_parts if part)

    @classmethod
    def _preview_for_client(
        cls,
        client: AIBaseCompletions,
        *,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        system_prompt: str | None,
        max_response_tokens: int | None,
        bool_count_tokens: bool,
    ) -> AIRoutePreview:
        """The preview of one engine client: its model, level, and cost."""
        from ai_api_unified.pricing.pricing_registry import get_model_pricing

        int_input: int | None = None
        str_source: str = "estimated"
        if (
            bool_count_tokens
            and prompt
            and not messages
            and getattr(client.capabilities, "supports_token_counting", False)
        ):
            int_input = client.count_tokens(prompt)
            str_source = "provider"
        if int_input is None:
            int_input = max(
                1,
                math.ceil(
                    len(cls._preview_text(prompt, messages, system_prompt))
                    / INT_PREVIEW_CHARS_PER_TOKEN
                ),
            )
        int_max_output: int = int(
            max_response_tokens
            or getattr(
                client,
                "SEND_PROMPT_MAX_TOKENS",
                cls.STRUCTURED_DEFAULT_MAX_RESPONSE_TOKENS,
            )
        )
        thinking_caps: Any = getattr(client.capabilities, "thinking", None)
        resolution: AIThinkingResolution = resolve_thinking_level(
            client.thinking_level,
            (
                thinking_caps
                if isinstance(thinking_caps, AIThinkingCapabilities)
                else None
            ),
        )
        pricing: Any = get_model_pricing(
            client._resolve_observability_provider_vendor(), client.model_name
        )
        rates: Any = getattr(pricing, "token_rates", None)
        usd_input: Decimal | None = None
        usd_max: Decimal | None = None
        if rates is not None:
            usd_input = Decimal(int_input) * rates.input_per_1m / Decimal(1_000_000)
            usd_max = usd_input + (
                Decimal(int_max_output) * (rates.output_per_1m or Decimal(0))
            ) / Decimal(1_000_000)
        # Normal return with the client's preview.
        return AIRoutePreview(
            engine=client._resolve_observability_provider_engine(),
            model=client.model_name,
            thinking=resolution,
            input_tokens=int_input,
            input_token_source=str_source,
            max_output_tokens=int_max_output,
            usd_input=usd_input,
            usd_max=usd_max,
            pricing_source=getattr(pricing, "source", None) if rates else None,
            pricing_effective_date=(
                getattr(pricing, "effective_date", None) if rates else None
            ),
            config_id=client.config_id,
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

    # ------------------------------------------------------------------
    # Conversation turns, structured output, and async variants
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_system_prompt(
        system_prompt: str | None,
        other_params: AICompletionsPromptParamsBase | None,
        default_system_prompt: str,
    ) -> str:
        """
        Resolves the effective system prompt for one completions call.

        Args:
            system_prompt: Optional explicit per-call system prompt override.
            other_params: Optional provider params that may carry a system prompt.
            default_system_prompt: Engine default when neither source is set.

        Returns:
            Effective system prompt string for the provider request.
        """
        if system_prompt is not None and system_prompt.strip():
            # Early return because the explicit per-call override wins.
            return system_prompt
        if other_params is not None and other_params.system_prompt:
            # Early return with the params-supplied system prompt.
            return other_params.system_prompt
        # Normal return with the engine default system prompt.
        return default_system_prompt

    @staticmethod
    def _prompt_cache_hook_kwargs(
        prompt_cache: AIPromptCacheHint | None,
    ) -> dict[str, AIPromptCacheHint]:
        """
        Returns the prompt_cache keyword for an overridable hook, or nothing.

        Provider hooks and request builders gained a prompt_cache keyword
        with the cache hint. Passing it only when a hint is set keeps
        subclasses that override a hook with the earlier signature working
        for every call that does not use caching.

        Args:
            prompt_cache: Optional caller cache hint.

        Returns:
            {"prompt_cache": hint} when a hint is set, otherwise {}.
        """
        if prompt_cache is None:
            # Early return: omit the keyword entirely.
            return {}
        # Normal return with the keyword to splat into the hook call.
        return {"prompt_cache": prompt_cache}

    def _raise_request_error(self, exception: Exception) -> None:
        """
        Re-raises one provider SDK transport error as AiProviderRequestError.

        Engines override this with their SDK's error mapping. The base returns
        without raising, so a caller that needs the typed form (for example a
        fallback client reading a streaming error) can call it on any engine
        and re-raise the original when nothing matched.

        Args:
            exception: Exception raised by the provider SDK.
        """
        # Normal return: the base engine maps nothing.
        return None

    @staticmethod
    def _resolve_prompt_cache_hint(
        other_params: AICompletionsPromptParamsBase | None,
    ) -> AIPromptCacheHint | None:
        """
        Returns the prompt cache hint carried by other_params, if any.

        Args:
            other_params: Optional provider params that may carry a hint.

        Returns:
            The caller's AIPromptCacheHint, or None when caching was not requested.
        """
        if other_params is None:
            # Early return because no params were supplied.
            return None
        # Normal return with the params-supplied hint (possibly None).
        return other_params.prompt_cache

    def _reject_unsupported_send_prompt_params(
        self,
        *,
        max_response_tokens: int | None,
        request_timeout_seconds: float | None,
    ) -> None:
        """
        Rejects extended send_prompt parameters on engines that have not mapped them.

        Args:
            max_response_tokens: Optional response token budget override.
            request_timeout_seconds: Optional per-call provider timeout.

        Raises:
            AiProviderCapabilityUnsupportedError: When either parameter is supplied.
        """
        if max_response_tokens is None and request_timeout_seconds is None:
            # Normal return because no unsupported parameter was supplied.
            return None
        raise AiProviderCapabilityUnsupportedError(
            f"{type(self).__name__} model '{self.model_name}' does not yet map "
            "max_response_tokens or request_timeout_seconds on send_prompt. "
            "Use the claude engine, or omit these parameters."
        )

    def _require_tool_use_capability(self) -> None:
        """
        Raises the typed capability error unless this model supports tool use.
        """
        if self.capabilities.supports_tool_use:
            # Normal return because tool use is supported.
            return None
        raise AiProviderCapabilityUnsupportedError(
            f"{type(self).__name__} model '{self.model_name}' does not support "
            "tool-use conversations. Configure an engine whose capabilities "
            "include supports_tool_use."
        )

    def _require_structured_output_capability(self) -> None:
        """
        Raises the typed capability error unless this model supports structured output.
        """
        if self.capabilities.supports_structured_output:
            # Normal return because structured output is supported.
            return None
        raise AiProviderCapabilityUnsupportedError(
            f"{type(self).__name__} model '{self.model_name}' does not support "
            "send_structured_output. Configure an engine whose capabilities "
            "include supports_structured_output, or use strict_schema_prompt."
        )

    def _require_async_capability(self) -> None:
        """
        Raises the typed capability error unless this engine offers async variants.
        """
        if self.capabilities.supports_async:
            # Normal return because async variants are supported.
            return None
        raise AiProviderCapabilityUnsupportedError(
            f"{type(self).__name__} model '{self.model_name}' does not support "
            "async calls. Configure an engine whose capabilities include "
            "supports_async, or use the synchronous methods."
        )

    @staticmethod
    def _normalize_finish_reason(
        provider_finish_reason: str | None,
        dict_finish_reason_map: Mapping[str, AIFinishReason],
    ) -> AIFinishReason:
        """
        Maps one provider-specific stop reason onto the normalized enum.

        Unknown or missing provider reasons normalize to COMPLETE so callers
        branch only on the defined enum values.

        Args:
            provider_finish_reason: Raw provider stop/finish reason token.
            dict_finish_reason_map: Engine-supplied provider-token-to-enum map.

        Returns:
            Normalized AIFinishReason value.
        """
        if provider_finish_reason is None:
            # Early return because a missing reason means a normally completed turn.
            return AIFinishReason.COMPLETE
        # Normal return with the mapped reason, defaulting unknown tokens to COMPLETE.
        return dict_finish_reason_map.get(
            provider_finish_reason, AIFinishReason.COMPLETE
        )

    def _validate_structured_output_request(
        self,
        *,
        prompt: str | None,
        response_model: Type[AIStructuredPrompt] | None,
        response_schema: dict[str, Any] | None,
        messages: list[dict[str, Any]] | None,
        max_response_tokens: int,
    ) -> dict[str, Any]:
        """
        Validates one send_structured_output request and resolves the JSON schema.

        Args:
            prompt: Optional single-shot user prompt.
            response_model: Optional AIStructuredPrompt subclass supplying the schema.
            response_schema: Optional raw JSON Schema supplied directly.
            messages: Optional caller-managed multi-turn input.
            max_response_tokens: Requested response token budget.

        Returns:
            Effective JSON schema dictionary for the provider request.

        Raises:
            ValueError: When the request shape is invalid.
            AiProviderConfigurationError: When PII redaction is enabled with
                caller-managed messages, whose opaque content cannot be redacted.
        """
        if (response_model is None) == (response_schema is None):
            raise ValueError(
                "Provide exactly one of response_model or response_schema."
            )
        if response_model is not None and not issubclass(
            response_model, AIStructuredPrompt
        ):
            raise ValueError("response_model must be a subclass of AIStructuredPrompt.")
        bool_has_prompt: bool = prompt is not None and bool(prompt.strip())
        bool_has_messages: bool = bool(messages)
        if not bool_has_prompt and not bool_has_messages:
            raise ValueError("Provide prompt, messages, or both.")
        if bool_has_messages and self.pii_middleware.bool_enabled:
            raise AiProviderConfigurationError(
                "send_structured_output with caller-managed messages is "
                "unavailable while PII redaction middleware is enabled: "
                "replayed message content is opaque and cannot be redacted. "
                "Use the prompt parameter, or disable the PII redaction profile."
            )
        int_context_window: int = self.capabilities.context_window_length
        if int_context_window > 0 and max_response_tokens > int_context_window:
            raise ValueError(
                f"max_response_tokens ({max_response_tokens}) exceeds the "
                f"model context window ({int_context_window})."
            )
        if response_schema is not None:
            # Normal return with the caller-supplied raw JSON schema.
            return response_schema
        # Normal return with the schema derived from the pydantic response model.
        return response_model.model_json_schema()

    def _normalize_messages(
        self,
        messages: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        """
        Translates interface-shaped messages into this provider's wire shape.

        send_conversation documents messages as {role, content} dictionaries
        and calls that shape provider-neutral. Engines whose wire format is
        already {role, content} need no translation, so the base returns the
        caller's list untouched; an engine with a different shape overrides
        this and translates. Routing every conversation and structured-output
        surface through one hook keeps the documented contract true on all of
        them rather than on whichever ones a given engine happened to cover.

        This deliberately does not live in _validate_conversation_request.
        That method is called only from send_conversation and
        asend_conversation and returns the tool list, so translating there
        would leave send_structured_output and asend_structured_output on the
        broken path and would need a signature change to carry messages back
        out. Do not move it there.

        Args:
            messages: Caller-managed history, or None where the surface
                accepts a bare prompt instead.

        Returns:
            Messages in the provider's wire shape. The base returns the
            argument unchanged.
        """
        # Normal return: the documented shape is already this engine's shape.
        return messages

    def _validate_conversation_request(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[AITool] | None,
        tool_choice: str | None,
    ) -> list[AITool]:
        """
        Validates one send_conversation request.

        Args:
            messages: Caller-managed conversation history for this turn.
            tools: Optional tool definitions offered to the model.
            tool_choice: Optional name of a tool the model must use.

        Returns:
            Normalized (possibly empty) tool definition list.

        Raises:
            ValueError: When the request shape is invalid.
            AiProviderConfigurationError: When PII redaction is enabled;
                conversation replay content is opaque and cannot be redacted.
        """
        if not messages:
            raise ValueError("messages cannot be empty.")
        if self.pii_middleware.bool_enabled:
            raise AiProviderConfigurationError(
                "send_conversation is unavailable while PII redaction "
                "middleware is enabled: replayed assistant turns and tool "
                "results are opaque and cannot be redacted. Disable the PII "
                "redaction middleware profile to use conversation turns."
            )
        list_tools: list[AITool] = list(tools or [])
        if tool_choice is not None:
            set_tool_names: set[str] = {ai_tool.name for ai_tool in list_tools}
            if tool_choice not in set_tool_names:
                raise ValueError(
                    f"tool_choice {tool_choice!r} does not name a supplied tool."
                )
        # Normal return with the normalized tool list.
        return list_tools

    @enveloped_completions_call
    def send_structured_output(
        self,
        prompt: str | None = None,
        *,
        response_model: Type[AIStructuredPrompt] | None = None,
        response_schema: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_response_tokens: int = STRUCTURED_DEFAULT_MAX_RESPONSE_TOKENS,
        request_timeout_seconds: float | None = None,
        provider_options: dict[str, Any] | None = None,
        prompt_cache: AIPromptCacheHint | None = None,
    ) -> AIStructuredOutputResult:
        """
        Generates schema-constrained structured output and returns the parsed result.

        Template method: capability and request-shape gating happen here so
        providers cannot skip them. Providers whose capabilities declare
        structured-output support implement _send_structured_output_provider.

        The result's finish_reason lets callers distinguish LENGTH truncation
        (retry with a larger budget) from REFUSAL (abort) in code; data is None
        in both of those cases.

        Args:
            prompt: Optional single-shot user prompt. May be omitted when
                messages is provided.
            response_model: AIStructuredPrompt subclass supplying the schema.
                Mutually exclusive with response_schema. When supplied, the
                parsed data is also validated against the model.
            response_schema: Raw JSON Schema supplied directly (for example a
                hand-written schema with anyOf variants). Mutually exclusive
                with response_model.
            system_prompt: Optional persistent instructions, separate from the
                user prompt.
            messages: Optional caller-managed multi-turn input of
                {role, content} dictionaries, replayed ahead of prompt.
            max_response_tokens: Response token budget, up to the engine's
                context limit. Engines handle provider non-streaming request
                limits internally; the caller sees one blocking call.
            request_timeout_seconds: Optional per-call provider timeout.
            provider_options: Engine-specific escape hatch merged into the
                underlying request; engines ignore keys they do not understand.
            prompt_cache: Optional request to cache the system prompt as a
                stable prefix. Engines that cannot honor it ignore it; see
                AIPromptCacheHint.

        Returns:
            AIStructuredOutputResult carrying parsed data, the normalized
            finish reason, and token usage.

        Raises:
            AiProviderCapabilityUnsupportedError: When the configured model
                does not support structured output.
            ValueError: When the request shape is invalid or the provider
                returned unparseable JSON on a COMPLETE turn.
        """
        self._require_structured_output_capability()
        dict_effective_schema: dict[str, Any] = (
            self._validate_structured_output_request(
                prompt=prompt,
                response_model=response_model,
                response_schema=response_schema,
                messages=messages,
                max_response_tokens=max_response_tokens,
            )
        )
        provider_options, thinking_override = self._split_thinking_option(
            provider_options
        )
        token_thinking = THINKING_CALL_OVERRIDE.set(thinking_override)
        token_native = THINKING_NATIVE_OPTION.set(
            self._native_thinking_option(provider_options)
        )
        provider_options, token_purpose = self._push_purpose_option(provider_options)
        try:
            structured_result: AIStructuredOutputResult = (
                self._send_structured_output_provider(
                    response_schema=dict_effective_schema,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    messages=self._normalize_messages(messages),
                    max_response_tokens=max_response_tokens,
                    request_timeout_seconds=request_timeout_seconds,
                    provider_options=provider_options,
                    **self._prompt_cache_hook_kwargs(prompt_cache),
                )
            )
        finally:
            THINKING_CALL_OVERRIDE.reset(token_thinking)
            THINKING_NATIVE_OPTION.reset(token_native)
            if token_purpose is not None:
                reset_observability_context(token_purpose)
        # Normal return after optional pydantic validation of the parsed data.
        return self._validate_structured_output_result(
            structured_result=self._stamp_thinking(structured_result),
            response_model=response_model,
        )

    @staticmethod
    def _validate_structured_output_result(
        *,
        structured_result: AIStructuredOutputResult,
        response_model: Type[AIStructuredPrompt] | None,
    ) -> AIStructuredOutputResult:
        """
        Validates parsed structured data against the optional response model.

        Args:
            structured_result: Provider-built structured output result.
            response_model: Optional AIStructuredPrompt subclass for validation.

        Returns:
            The structured result unchanged after successful validation.

        Raises:
            ValueError: When COMPLETE data fails response-model validation.
        """
        if (
            response_model is None
            or structured_result.data is None
            or structured_result.finish_reason is not AIFinishReason.COMPLETE
        ):
            # Early return because there is nothing to validate.
            return structured_result
        try:
            response_model.model_validate(structured_result.data)
        except ValidationError as exception:
            # The provider call already marked this answer served; it is
            # being discarded, so it counts as wasted on the request report.
            scope: RequestCostScope | None = current_request_cost_scope()
            if scope is not None:
                scope.discard_last_served()
            raise StructuredOutputValidationError(
                f"Response validation failed with {exception.error_count()} "
                "validation errors.",
                usage=structured_result.usage,
                raw_text=structured_result.raw_text,
                finish_reason=structured_result.finish_reason.value,
            ) from exception
        # Normal return with the validated structured result.
        return structured_result

    def _parse_structured_output_json(
        self,
        *,
        raw_output_text: str,
        finish_reason: AIFinishReason,
        usage: AITokenUsage,
        provider_label: str,
    ) -> AIStructuredOutputResult:
        """
        Maps one COMPLETE structured answer onto the result model, or raises
        a billed failure.

        Shared by every engine's structured-output builder so an unusable
        answer raises the same StructuredOutputValidationError everywhere,
        carrying the usage the provider reported for it.

        Args:
            raw_output_text: Provider output text before output redaction.
            finish_reason: Normalized finish reason; LENGTH and REFUSAL return
                data=None without parsing.
            usage: Provider-reported token usage for the answer.
            provider_label: Provider name for the empty-response message.

        Returns:
            AIStructuredOutputResult with parsed data on a COMPLETE turn.

        Raises:
            StructuredOutputValidationError: Empty, invalid JSON, or not an
                object. A ValueError and an INVALID_OUTPUT fallback trigger.
        """
        if finish_reason in (AIFinishReason.LENGTH, AIFinishReason.REFUSAL):
            # Early return so callers branch on finish_reason instead of parsing.
            # raw_text passes through output redaction like every other model
            # output surface, so truncated bodies follow the same PII policy.
            return AIStructuredOutputResult(
                data=None,
                finish_reason=finish_reason,
                usage=usage,
                raw_text=self.pii_middleware.process_output(raw_output_text),
            )
        str_content: str = self.pii_middleware.process_output(raw_output_text)
        if not str_content:
            raise StructuredOutputValidationError(
                f"Empty response from {provider_label}",
                usage=usage,
                raw_text=str_content,
                finish_reason=finish_reason.value,
            )
        try:
            parsed_data: Any = json.loads(str_content)
        except json.JSONDecodeError as json_error:
            raise StructuredOutputValidationError(
                f"Invalid JSON response: {json_error}",
                usage=usage,
                raw_text=str_content,
                finish_reason=finish_reason.value,
            ) from json_error
        if not isinstance(parsed_data, dict):
            raise StructuredOutputValidationError(
                "Structured response was valid JSON but not a JSON object.",
                usage=usage,
                raw_text=str_content,
                finish_reason=finish_reason.value,
            )
        # Normal return with the parsed structured output result.
        return AIStructuredOutputResult(
            data=parsed_data,
            finish_reason=finish_reason,
            usage=usage,
            raw_text=str_content,
        )

    def _send_structured_output_provider(
        self,
        *,
        response_schema: dict[str, Any],
        system_prompt: str | None,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        max_response_tokens: int,
        request_timeout_seconds: float | None,
        provider_options: dict[str, Any] | None,
        prompt_cache: AIPromptCacheHint | None = None,
    ) -> AIStructuredOutputResult:
        """
        Provider hook for structured output generation.

        The base implementation raises: a provider whose capabilities declare
        structured-output support must implement this hook.
        """
        self._raise_completions_capability_unsupported("structured output")

    @enveloped_completions_call
    def send_conversation(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[AITool] | None = None,
        tool_choice: str | None = None,
        max_response_tokens: int | None = None,
        request_timeout_seconds: float | None = None,
        provider_options: dict[str, Any] | None = None,
        prompt_cache: AIPromptCacheHint | None = None,
    ) -> AITurnResult:
        """
        Sends one conversation turn and returns the model's turn.

        The caller owns the loop: it executes any requested tools, appends the
        results (via build_tool_result_message) and the assistant turn's
        raw_content to its message history, and calls send_conversation again
        up to its own iteration bound.

        Template method: capability and request-shape gating happen here so
        providers cannot skip them. Providers whose capabilities declare
        tool-use support implement _send_conversation_provider.

        Args:
            system_prompt: Persistent instructions for the conversation.
            messages: Caller-managed history of {role, content} dictionaries,
                including prior raw_content assistant turns and tool-result
                entries.
            tools: Optional provider-neutral tool definitions for this turn.
            tool_choice: Optional name of a tool the model must use this turn.
            max_response_tokens: Optional response token budget for the turn.
            request_timeout_seconds: Optional per-call provider timeout.
            provider_options: Engine-specific escape hatch merged into the
                underlying request; engines ignore keys they do not understand.
            prompt_cache: Optional request to cache the system prompt, tools,
                and (on Anthropic) the growing message history across turns.
                Engines that cannot honor it ignore it; see AIPromptCacheHint.

        Returns:
            AITurnResult carrying text, requested tool calls, the normalized
            finish reason, replayable raw_content, and token usage.

        Raises:
            AiProviderCapabilityUnsupportedError: When the configured model
                does not support tool-use conversations.
            ValueError: When the request shape is invalid.
        """
        self._require_tool_use_capability()
        list_tools: list[AITool] = self._validate_conversation_request(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        list_messages: list[dict[str, Any]] = self._normalize_messages(messages) or []
        provider_options, thinking_override = self._split_thinking_option(
            provider_options
        )
        token_thinking = THINKING_CALL_OVERRIDE.set(thinking_override)
        token_native = THINKING_NATIVE_OPTION.set(
            self._native_thinking_option(provider_options)
        )
        provider_options, token_purpose = self._push_purpose_option(provider_options)
        try:
            turn_result: AITurnResult = self._send_conversation_provider(
                system_prompt=system_prompt,
                messages=list_messages,
                tools=list_tools,
                tool_choice=tool_choice,
                max_response_tokens=max_response_tokens,
                request_timeout_seconds=request_timeout_seconds,
                provider_options=provider_options,
                **self._prompt_cache_hook_kwargs(prompt_cache),
            )
        finally:
            THINKING_CALL_OVERRIDE.reset(token_thinking)
            THINKING_NATIVE_OPTION.reset(token_native)
            if token_purpose is not None:
                reset_observability_context(token_purpose)
        # Normal return with the provider-implemented conversation turn.
        return self._stamp_thinking(turn_result)

    def _send_conversation_provider(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[AITool],
        tool_choice: str | None,
        max_response_tokens: int | None,
        request_timeout_seconds: float | None,
        provider_options: dict[str, Any] | None,
        prompt_cache: AIPromptCacheHint | None = None,
    ) -> AITurnResult:
        """
        Provider hook for one conversation turn.

        The base implementation raises: a provider whose capabilities declare
        tool-use support must implement this hook.
        """
        self._raise_completions_capability_unsupported("tool-use conversations")

    def build_tool_result_message(
        self,
        *,
        tool_call_id: str,
        result: dict[str, Any],
        is_error: bool = False,
    ) -> dict[str, Any]:
        """
        Builds one tool-result message in the engine's expected wire shape.

        Callers append the returned dictionary to their message history after
        executing a requested tool, so they never touch provider block formats
        except as the opaque raw_content.

        Args:
            tool_call_id: The id from the AIToolCall being answered.
            result: JSON-serializable tool execution result.
            is_error: True when the tool execution failed.

        Returns:
            Engine-shaped {role, content} message dictionary.

        Raises:
            AiProviderCapabilityUnsupportedError: When the configured model
                does not support tool-use conversations.
        """
        self._require_tool_use_capability()
        # Normal return with the engine-shaped tool-result message.
        return self._build_tool_result_message_provider(
            tool_call_id=tool_call_id,
            result=result,
            is_error=is_error,
        )

    def _build_tool_result_message_provider(
        self,
        *,
        tool_call_id: str,
        result: dict[str, Any],
        is_error: bool,
    ) -> dict[str, Any]:
        """
        Provider hook for building one engine-shaped tool-result message.

        The base implementation raises: a provider whose capabilities declare
        tool-use support must implement this hook.
        """
        self._raise_completions_capability_unsupported("tool-use conversations")

    def extend_messages_with_turn(
        self,
        messages: list[dict[str, Any]],
        turn: AITurnResult,
    ) -> list[dict[str, Any]]:
        """
        Appends one model turn to a caller-managed history in engine wire shape.

        Engines shape assistant turns differently (a single message wrapping
        content blocks, a message carrying tool_calls, or a list of output
        items), so this helper keeps the caller's tool loop engine-agnostic:
        append the turn, append tool results via build_tool_result_message,
        and call send_conversation again.

        Args:
            messages: Caller-managed message history, mutated in place.
            turn: The AITurnResult returned by send_conversation.

        Returns:
            The same messages list, for chaining.

        Raises:
            AiProviderCapabilityUnsupportedError: When the configured model
                does not support tool-use conversations.
        """
        self._require_tool_use_capability()
        self._extend_messages_with_turn_provider(messages=messages, turn=turn)
        # Normal return with the extended caller-managed history.
        return messages

    def _extend_messages_with_turn_provider(
        self,
        *,
        messages: list[dict[str, Any]],
        turn: AITurnResult,
    ) -> None:
        """
        Provider hook for appending one model turn in engine wire shape.

        The base implementation raises: a provider whose capabilities declare
        tool-use support must implement this hook.
        """
        self._raise_completions_capability_unsupported("tool-use conversations")

    # Reserved provider_options key honored by engines that support per-call
    # retry overrides; every other key merges into the provider request.
    PROVIDER_OPTION_RETRY_POLICY: ClassVar[str] = "retry_policy"

    def _split_provider_options(
        self, provider_options: dict[str, Any] | None
    ) -> tuple[dict[str, Any], str | None]:
        """
        Splits provider_options into request-merge keys and reserved keys.

        Args:
            provider_options: Optional engine-specific escape hatch supplied
                per call.

        Returns:
            Tuple of (the subset of caller options this engine's SDK accepts,
            merged into the provider request, and the optional per-call retry
            policy override). Keys the SDK does not accept are dropped and
            logged; see _filter_known_provider_options for that rule.
        """
        if not provider_options:
            # Early return because there is nothing to split.
            return {}, None
        dict_merge_options: dict[str, Any] = dict(provider_options)
        str_retry_policy: str | None = dict_merge_options.pop(
            self.PROVIDER_OPTION_RETRY_POLICY, None
        )
        # The per-call thinking level is the templates' to apply; an engine
        # reached directly still must not forward it to the SDK.
        dict_merge_options.pop(AIBaseCompletions.PROVIDER_OPTION_THINKING, None)
        # Normal return with the filtered merge options and the retry override.
        return self._filter_known_provider_options(dict_merge_options), str_retry_policy

    def _filter_known_provider_options(
        self, merge_options: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Drops provider_options keys this engine's SDK does not accept.

        The public docstring promises engines ignore keys they do not
        understand, which is what makes provider_options usable in a
        cross-provider fallback: a caller tuned for one engine keeps its
        options when it fails over. Forwarding an unknown key instead reaches
        the SDK and fails inside the caller's process, so the escape hatch
        breaks exactly the callers it exists for.

        Args:
            merge_options: Caller options with reserved keys already removed.

        Returns:
            The subset the engine's SDK accepts. Dropped keys are logged at
            warning level so a silently ignored option is still discoverable.
        """
        set_known: frozenset[str] | None = self._known_provider_option_keys()
        if set_known is None:
            # Early return: introspection is unavailable (a mocked SDK, or a
            # version whose metadata moved), so forward unchanged rather than
            # guess and drop an option the caller needs.
            return merge_options
        dict_known: dict[str, Any] = {
            str_key: value
            for str_key, value in merge_options.items()
            if str_key in set_known
        }
        list_dropped: list[str] = sorted(set(merge_options) - set(dict_known))
        if list_dropped:
            _LOGGER.warning(
                "Ignoring provider_options key(s) %s: not accepted by the %s "
                "engine. See the engine's SDK for the options it supports.",
                ", ".join(repr(str_key) for str_key in list_dropped),
                type(self).__name__,
            )
        # Normal return with only the keys this engine understands.
        return dict_known

    def _sdk_option_method(self) -> tuple[Any, str] | None:
        """
        Names the SDK method whose keyword arguments are the accepted options.

        An engine whose provider options become keyword arguments implements
        only this: import the SDK class and hand back its UNBOUND method plus
        a label for logging. The base owns the signature walk and the cache,
        so a new engine cannot reimplement either incorrectly.

        Returning the unbound class attribute is load-bearing. A bound method
        off self.client is a Mock under test, whose signature is
        (*args, **kwargs); that reads as "accepts anything" and would silently
        disable filtering.

        Returns:
            Tuple of (unbound SDK method, label), or None when this engine
            resolves its keys some other way or cannot resolve them.
        """
        # Normal return: the base engine has no SDK method to introspect.
        return None

    def _known_provider_option_keys(self) -> frozenset[str] | None:
        """
        Reports the provider_options keys this engine's SDK accepts.

        Derived from the SDK rather than hardcoded, so a version that adds an
        option does not have it dropped here. Engines whose options are
        keyword arguments implement _sdk_option_method and inherit this;
        engines whose options are validated some other way (a pydantic config
        model, a service model) override this directly.

        Returns:
            Accepted key names, or None when they cannot be determined, in
            which case every key is forwarded unchanged.
        """
        type_self: type = type(self)
        frozenset_cached: frozenset[str] | None = type_self.__dict__.get(
            "_FROZENSET_SDK_OPTION_KEYS"
        )
        if frozenset_cached is not None:
            # Early return with the resolved set; the signature is stable.
            return frozenset_cached
        tuple_source: tuple[Any, str] | None = self._sdk_option_method()
        if tuple_source is None:
            # Early return: nothing to introspect, so forward every key.
            return None
        unbound, str_label = tuple_source
        frozenset_keys: frozenset[str] | None = self._keys_from_callable_signature(
            unbound, sdk_label=str_label
        )
        if frozenset_keys is None:
            # Early return: the helper already logged why.
            return None
        type_self._FROZENSET_SDK_OPTION_KEYS = frozenset_keys
        # Normal return with the accepted keyword arguments.
        return frozenset_keys

    @staticmethod
    def _keys_from_callable_signature(
        unbound: Any, *, sdk_label: str
    ) -> frozenset[str] | None:
        """
        Reports the keyword arguments an SDK method accepts.

        Shared by every engine whose provider options become keyword
        arguments. Pass the UNBOUND class attribute, never a bound method off
        self.client: a mocked client's signature is (*args, **kwargs), which
        reads as "accepts anything" and silently disables filtering.

        Args:
            unbound: The SDK's unbound method, for example Completions.create.
            sdk_label: Name used when logging that introspection failed.

        Returns:
            Accepted parameter names; None when the signature cannot be read
            or declares **kwargs, meaning the caller's keys must be forwarded
            unchanged rather than dropped on a guess.
        """
        try:
            signature = inspect.signature(unbound)
        except (TypeError, ValueError) as exception:
            _LOGGER.warning(
                "Could not read the %s signature (%s); forwarding "
                "provider_options unfiltered, so an unknown key will surface "
                "as the SDK's own error.",
                sdk_label,
                exception,
            )
            # Early return: forwarding beats dropping on a guess.
            return None
        set_keys: set[str] = set()
        # Loop over parameters so only real keyword arguments are kept.
        for str_name, parameter in signature.parameters.items():
            if str_name == "self":
                continue
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                # Early return: the SDK forwards anything, so drop nothing.
                return None
            if parameter.kind in (
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                set_keys.add(str_name)
        if not set_keys:
            _LOGGER.warning(
                "The %s signature exposed no keyword arguments; forwarding "
                "provider_options unfiltered.",
                sdk_label,
            )
            # Early return: an empty signature means introspection failed.
            return None
        # Normal return with the accepted keyword arguments.
        return frozenset(set_keys)

    async def asend_prompt(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_response_tokens: int | None = None,
        request_timeout_seconds: float | None = None,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> str:
        """
        Async variant of send_prompt for engines whose SDK has an async client.

        Args:
            prompt: The text prompt to send.
            system_prompt: Optional persistent instructions.
            max_response_tokens: Optional response token budget override.
            request_timeout_seconds: Optional per-call provider timeout.
            other_params: Optional provider-specific parameters.

        Returns:
            Model response text.

        Raises:
            AiProviderCapabilityUnsupportedError: When the engine has no async
                support.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty or None")
        self._require_async_capability()
        # Normal return with the provider-implemented async response.
        return await self._asend_prompt_provider(
            prompt,
            system_prompt=system_prompt,
            max_response_tokens=max_response_tokens,
            request_timeout_seconds=request_timeout_seconds,
            other_params=other_params,
        )

    async def _asend_prompt_provider(
        self,
        prompt: str,
        *,
        system_prompt: str | None,
        max_response_tokens: int | None,
        request_timeout_seconds: float | None,
        other_params: AICompletionsPromptParamsBase | None,
    ) -> str:
        """
        Provider hook for async prompt generation.

        The base implementation raises: a provider whose capabilities declare
        async support must implement this hook.
        """
        self._raise_completions_capability_unsupported("async completions")

    @aenveloped_completions_call
    async def asend_structured_output(
        self,
        prompt: str | None = None,
        *,
        response_model: Type[AIStructuredPrompt] | None = None,
        response_schema: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_response_tokens: int = STRUCTURED_DEFAULT_MAX_RESPONSE_TOKENS,
        request_timeout_seconds: float | None = None,
        provider_options: dict[str, Any] | None = None,
        prompt_cache: AIPromptCacheHint | None = None,
    ) -> AIStructuredOutputResult:
        """
        Async variant of send_structured_output. See send_structured_output.
        """
        self._require_async_capability()
        self._require_structured_output_capability()
        dict_effective_schema: dict[str, Any] = (
            self._validate_structured_output_request(
                prompt=prompt,
                response_model=response_model,
                response_schema=response_schema,
                messages=messages,
                max_response_tokens=max_response_tokens,
            )
        )
        provider_options, thinking_override = self._split_thinking_option(
            provider_options
        )
        token_thinking = THINKING_CALL_OVERRIDE.set(thinking_override)
        token_native = THINKING_NATIVE_OPTION.set(
            self._native_thinking_option(provider_options)
        )
        provider_options, token_purpose = self._push_purpose_option(provider_options)
        try:
            structured_result: AIStructuredOutputResult = (
                await self._asend_structured_output_provider(
                    response_schema=dict_effective_schema,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    messages=self._normalize_messages(messages),
                    max_response_tokens=max_response_tokens,
                    request_timeout_seconds=request_timeout_seconds,
                    provider_options=provider_options,
                    **self._prompt_cache_hook_kwargs(prompt_cache),
                )
            )
        finally:
            THINKING_CALL_OVERRIDE.reset(token_thinking)
            THINKING_NATIVE_OPTION.reset(token_native)
            if token_purpose is not None:
                reset_observability_context(token_purpose)
        # Normal return after optional pydantic validation of the parsed data.
        return self._validate_structured_output_result(
            structured_result=self._stamp_thinking(structured_result),
            response_model=response_model,
        )

    async def _asend_structured_output_provider(
        self,
        *,
        response_schema: dict[str, Any],
        system_prompt: str | None,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        max_response_tokens: int,
        request_timeout_seconds: float | None,
        provider_options: dict[str, Any] | None,
        prompt_cache: AIPromptCacheHint | None = None,
    ) -> AIStructuredOutputResult:
        """
        Provider hook for async structured output generation.

        The base implementation raises: a provider whose capabilities declare
        async support must implement this hook.
        """
        self._raise_completions_capability_unsupported("async structured output")

    @aenveloped_completions_call
    async def asend_conversation(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[AITool] | None = None,
        tool_choice: str | None = None,
        max_response_tokens: int | None = None,
        request_timeout_seconds: float | None = None,
        provider_options: dict[str, Any] | None = None,
        prompt_cache: AIPromptCacheHint | None = None,
    ) -> AITurnResult:
        """
        Async variant of send_conversation. See send_conversation.
        """
        self._require_async_capability()
        self._require_tool_use_capability()
        list_tools: list[AITool] = self._validate_conversation_request(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        list_messages: list[dict[str, Any]] = self._normalize_messages(messages) or []
        provider_options, thinking_override = self._split_thinking_option(
            provider_options
        )
        token_thinking = THINKING_CALL_OVERRIDE.set(thinking_override)
        token_native = THINKING_NATIVE_OPTION.set(
            self._native_thinking_option(provider_options)
        )
        provider_options, token_purpose = self._push_purpose_option(provider_options)
        try:
            turn_result: AITurnResult = await self._asend_conversation_provider(
                system_prompt=system_prompt,
                messages=list_messages,
                tools=list_tools,
                tool_choice=tool_choice,
                max_response_tokens=max_response_tokens,
                request_timeout_seconds=request_timeout_seconds,
                provider_options=provider_options,
                **self._prompt_cache_hook_kwargs(prompt_cache),
            )
        finally:
            THINKING_CALL_OVERRIDE.reset(token_thinking)
            THINKING_NATIVE_OPTION.reset(token_native)
            if token_purpose is not None:
                reset_observability_context(token_purpose)
        # Normal return with the provider-implemented async conversation turn.
        return self._stamp_thinking(turn_result)

    async def _asend_conversation_provider(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[AITool],
        tool_choice: str | None,
        max_response_tokens: int | None,
        request_timeout_seconds: float | None,
        provider_options: dict[str, Any] | None,
        prompt_cache: AIPromptCacheHint | None = None,
    ) -> AITurnResult:
        """
        Provider hook for one async conversation turn.

        The base implementation raises: a provider whose capabilities declare
        async support must implement this hook.
        """
        self._raise_completions_capability_unsupported("async tool-use conversations")

    def _raise_completions_capability_unsupported(
        self, str_requested_feature: str
    ) -> NoReturn:
        """
        Raises the shared typed capability error for one unsupported feature.

        Args:
            str_requested_feature: Human-readable feature label for the message.

        Raises:
            AiProviderCapabilityUnsupportedError: Always.
        """
        raise AiProviderCapabilityUnsupportedError(
            f"{type(self).__name__} model '{self.model_name}' does not support "
            f"{str_requested_feature}."
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
        return self._stream_with_request_cost_scope(
            operation,
            lambda: execute_observed_streaming_call(
                observability_middleware=self._get_observability_middleware(),
                callable_build_call_context=lambda: self._build_observability_call_context(
                    capability=self.CLIENT_TYPE_COMPLETIONS,
                    operation=operation,
                    dict_metadata=dict_input_metadata,
                    legacy_caller_id=legacy_caller_id,
                ),
                callable_open_stream=callable_open_stream,
                callable_build_result_summary=callable_build_result_summary,
            ),
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
                    provider_cache_write_5m_tokens=observed_result.provider_cache_write_5m_tokens,
                    provider_cache_write_1h_tokens=observed_result.provider_cache_write_1h_tokens,
                    provider_reasoning_tokens=observed_result.provider_reasoning_tokens,
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

    async def agenerate_images(
        self, image_prompt: str, image_properties: AIBaseImageProperties
    ) -> list[bytes]:
        """Async twin of generate_images; runs on a worker thread."""
        # Normal return with the image bytes.
        return await asyncio.to_thread(
            self.generate_images, image_prompt, image_properties
        )

    async def agenerate_image_files(
        self,
        image_prompt: str,
        image_properties: AIBaseImageProperties = AIBaseImageProperties(),
        root_file_name: str = "generate_image_files",
    ) -> list[Path]:
        """Async twin of generate_image_files; runs on a worker thread."""
        # Normal return with the saved paths.
        return await asyncio.to_thread(
            self.generate_image_files, image_prompt, image_properties, root_file_name
        )


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

    def _video_artifact_path(
        self, output_dir: Path, provider_job_id: str, str_suffix: str = ""
    ) -> Path:
        """
        Build the local path for a downloaded video from the provider's job id.

        The job id comes from the provider response, so it is untrusted: every
        character outside letters, digits, dot, dash, and underscore becomes an
        underscore, a dots-only name is refused, and the resolved path must stay
        inside output_dir.
        """

        str_stem: str = RE_VIDEO_ARTIFACT_UNSAFE_CHARS.sub("_", provider_job_id)
        if not str_stem.strip("."):
            raise ValueError(
                f"Refusing to write a video artifact for job id {provider_job_id!r}."
            )
        file_path: Path = output_dir / f"{str_stem}{str_suffix}.mp4"
        if not file_path.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError(
                f"Video artifact path for job id {provider_job_id!r} "
                f"escapes the output directory {output_dir}."
            )
        return file_path

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

    async def agenerate_video(
        self,
        video_prompt: str,
        video_properties: AIBaseVideoProperties = AIBaseVideoProperties(),
    ) -> AIVideoGenerationResult:
        """
        Async twin of generate_video: submits, polls, and downloads on a
        worker thread, so the event loop stays free while the video renders.
        """
        # Normal return with the finished video.
        return await asyncio.to_thread(
            self.generate_video, video_prompt, video_properties
        )

    async def asubmit_video_generation(
        self,
        video_prompt: str,
        video_properties: AIBaseVideoProperties = AIBaseVideoProperties(),
    ) -> AIVideoGenerationJob:
        """Async twin of submit_video_generation; runs on a worker thread."""
        # Normal return with the submitted job.
        return await asyncio.to_thread(
            self.submit_video_generation, video_prompt, video_properties
        )

    async def aget_video_generation_job(
        self, job: str | AIVideoGenerationJob
    ) -> AIVideoGenerationJob:
        """Async twin of get_video_generation_job; runs on a worker thread."""
        # Normal return with the job's state.
        return await asyncio.to_thread(self.get_video_generation_job, job)

    async def await_for_video_generation(
        self,
        job: str | AIVideoGenerationJob,
        *,
        timeout_seconds: int | None = None,
        poll_interval_seconds: int | None = None,
    ) -> AIVideoGenerationJob:
        """Async twin of wait_for_video_generation; polls on a worker thread."""
        # Normal return with the finished job.
        return await asyncio.to_thread(
            self.wait_for_video_generation,
            job,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    async def adownload_video_result(
        self, job: str | AIVideoGenerationJob
    ) -> AIVideoGenerationResult:
        """Async twin of download_video_result; runs on a worker thread."""
        # Normal return with the downloaded video.
        return await asyncio.to_thread(self.download_video_result, job)

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
