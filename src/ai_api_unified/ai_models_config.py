# ai_models_config.py
"""
The models configuration file: one place for every model selection.

`models.json` (or `.yaml`) holds what used to be scattered across `.env`
selection keys: the engine and model for each capability, the completions
fallback chain, the classifier, the router, the configured-model check,
and an allow/deny policy over models. Secrets and environment plumbing
(API keys, AWS credentials and region, base-URL overrides, the middleware
config path) stay in `.env`.

Precedence, highest first: an explicit factory argument, a process
environment variable, a `.env` value, this file, the library default. So
every `.env` key keeps working and can override the file per deployment;
the file is where the configuration is authored.

The file is found through `AI_MODELS_CONFIG` (a path, or inline JSON); when
that is unset, `./models.json` is used if it exists. A blank
`AI_MODELS_CONFIG` turns the file off.

Per-environment overrides live in the same file, under `environments`: a
map from an environment name to a partial config. `AI_MODELS_ENV` (the
process environment or `.env`) picks one, and it is deep-merged over the
rest of the file before validation: objects merge key by key, and lists
and values replace. An `AI_MODELS_ENV` that names no environment is a
load error that lists the defined ones.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .completions.ai_fallback_cooldown import AIFallbackCooldownPolicy
from .routing.config_checks import AIConfigRequirements
from .thinking import parse_thinking_level
from .routing.router_config import AIRouterClassifierConfig, AIRouterConfig

_LOGGER: logging.Logger = logging.getLogger(__name__)

AI_MODELS_CONFIG_KEY: str = "AI_MODELS_CONFIG"
# Names the `environments` entry merged over the file; see the module docstring.
AI_MODELS_ENV_KEY: str = "AI_MODELS_ENV"
DEFAULT_MODELS_CONFIG_FILENAME: str = "models.json"
TUPLE_YAML_SUFFIXES: tuple[str, ...] = (".yaml", ".yml")

# Values of the `ai_model_policy_event` field on this module's log records.
MODEL_POLICY_EVENT_DENIED: str = "model_denied"


def model_label(str_engine: str, str_model: str) -> str:
    """engine:model, the form the allow and deny lists match against."""
    # Normal return with the label.
    return f"{str_engine}:{str_model}"


class AIModelPolicy(BaseModel):
    """
    Which models the library may use.

    Entries are `engine:model` globs (`claude:*`, `*:gpt-4*`,
    `openai-decisions:gpt-6-luna`). Deny wins over allow; an empty allow
    list allows everything. A bare engine token (`bedrock`) matches every
    model of that engine.
    """

    model_config = ConfigDict(frozen=True)

    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)

    @field_validator("allow", "deny")
    @classmethod
    def _normalize(cls, value: list[str]) -> list[str]:
        list_out: list[str] = []
        # Loop over patterns so each is trimmed, lowercased on the engine,
        # and widened to engine:* when it names only an engine.
        for raw in value:
            str_pattern: str = str(raw).strip()
            if not str_pattern:
                continue
            if ":" not in str_pattern:
                str_pattern = f"{str_pattern}:*"
            str_engine, _, str_model = str_pattern.partition(":")
            list_out.append(f"{str_engine.strip().lower()}:{str_model.strip()}")
        # Normal return with the normalized patterns.
        return list_out

    @staticmethod
    def _matches(str_label: str, list_patterns: list[str]) -> str | None:
        # Loop over patterns for the first match.
        for str_pattern in list_patterns:
            if fnmatch.fnmatchcase(str_label, str_pattern):
                # Early return with the matching pattern.
                return str_pattern
        # Normal return: no pattern matched.
        return None

    def denial_reason(self, str_engine: str, str_model: str) -> str | None:
        """
        Why a model may not be used, or None when it may.

        Args:
            str_engine: Engine token.
            str_model: Model id; blank means the engine default, which is
                matched as `engine:` and so only by engine-wide patterns.

        Returns:
            A one-line reason naming the pattern, or None.
        """
        str_label: str = model_label(str_engine.strip().lower(), str_model.strip())
        str_denied: str | None = self._matches(str_label, self.deny)
        if str_denied is not None:
            # Early return: denied outright.
            return f"denied by policy pattern {str_denied!r}"
        if self.allow and self._matches(str_label, self.allow) is None:
            # Early return: not on the allow list.
            return "not on the allow list"
        # Normal return: permitted.
        return None

    def is_allowed(self, str_engine: str, str_model: str) -> bool:
        """True when the policy permits the model."""
        # Normal return with the policy answer.
        return self.denial_reason(str_engine, str_model) is None


class AICompletionsSelection(BaseModel):
    """The `completions` section."""

    model_config = ConfigDict(frozen=True)

    engine: str | None = None
    model: str | None = None
    fallbacks: list[str] = Field(default_factory=list)
    fallback_on: str | list[str] | None = None
    retry_policy: str | None = None
    # Provider-neutral thinking level: off, minimal, low, medium, high, max.
    thinking: str | None = None
    # How long a failed candidate rests: true/false, or
    # {"enabled": true, "seconds": {"rate_limited": 30, ...}}.
    cooldown: bool | AIFallbackCooldownPolicy | None = None
    # What every route must offer, checked at build:
    # {"capabilities": [...], "second_provider": true, "mode": "warn"}.
    require: AIConfigRequirements | None = None

    @field_validator("thinking", mode="before")
    @classmethod
    def _check_thinking(cls, value: Any) -> Any:
        """Fails a misspelled level at load, not at the first client build."""
        if value is None or not str(value).strip():
            # Early return: unset.
            return None
        # Normal return with the level's canonical value.
        return parse_thinking_level(str(value)).value


class AIClassifierSelection(BaseModel):
    """The `classifier` section; also the router's classifier."""

    model_config = ConfigDict(frozen=True)

    engine: str | None = None
    model: str | None = None
    base_url: str | None = None
    timeout_seconds: float | None = None
    instructions: str | None = None
    # Router only: below this confidence (0 to 1) a request goes one tier up.
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    # Thinking level for classifiers that run on a completions model;
    # default off, since a classification is one short label.
    thinking: str | None = None

    @field_validator("thinking", mode="before")
    @classmethod
    def _check_thinking(cls, value: Any) -> Any:
        """Fails a misspelled level at load."""
        if value is None or not str(value).strip():
            # Early return: unset.
            return None
        # Normal return with the level's canonical value.
        return parse_thinking_level(str(value)).value


class AIModelCheckSelection(BaseModel):
    """The `model_check` section."""

    model_config = ConfigDict(frozen=True)

    mode: Literal["prune", "strict", "off"] | None = None
    timeout_seconds: float | None = None


class AIEmbeddingsSelection(BaseModel):
    """The `embeddings` section."""

    model_config = ConfigDict(frozen=True)

    engine: str | None = None
    model: str | None = None
    dimensions: int | None = None


class AIImagesSelection(BaseModel):
    """The `images` section."""

    model_config = ConfigDict(frozen=True)

    engine: str | None = None
    model: str | None = None


class AIVideosSelection(BaseModel):
    """The `videos` section."""

    model_config = ConfigDict(frozen=True)

    engine: str | None = None
    model: str | None = None
    output_dir: str | None = None
    poll_interval_seconds: int | None = None
    timeout_seconds: int | None = None
    # S3 location Nova Reel writes to, unless a request names its own.
    bedrock_output_s3_uri: str | None = None


class AIOpenAIProviderSettings(BaseModel):
    """`providers.openai`: the OpenAI engines (chat, Responses, Decisions)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # API base URL; https required except for loopback hosts.
    base_url: str | None = None
    # End-user id sent with each request.
    user: str | None = None


class AIAnthropicProviderSettings(BaseModel):
    """`providers.anthropic`: the `claude` engine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_url: str | None = None
    # Admin API base URL for organization lookups; the admin key never
    # follows the inference base_url.
    admin_base_url: str | None = None
    user: str | None = None


class AIGoogleProviderSettings(BaseModel):
    """`providers.google`: Gemini, Google images, video, and voice."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    auth_method: Literal["api_key", "service_account"] | None = None
    # Vertex AI project and region, for service-account auth.
    project_id: str | None = None
    location: str | None = None
    base_url: str | None = None


class AIAwsProviderSettings(BaseModel):
    """`providers.aws`: Bedrock. Credentials stay in the AWS credential chain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    region: str | None = None


class AIOpenAICompatibleProviderSettings(BaseModel):
    """`providers.openai_compatible`: a self-hosted or third-party server."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_url: str | None = None
    structured_output: Literal["json_schema", "json_object"] | None = None
    # Input token limit for the context guard; unset disables the guard.
    context_window: int | None = Field(default=None, gt=0)
    # The server's thinking control: reasoning_effort (low|medium|high),
    # enable_thinking (vLLM/SGLang chat_template_kwargs on/off), or none.
    thinking_control: Literal["none", "reasoning_effort", "enable_thinking"] | None = (
        None
    )


class AIAzureSpeechProviderSettings(BaseModel):
    """`providers.azure_speech`: the Azure voice engine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    region: str | None = None


class AIProvidersSelection(BaseModel):
    """
    The `providers` section: per-provider settings that are not secrets.
    API keys and other credentials stay in the environment.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    openai: AIOpenAIProviderSettings = Field(default_factory=AIOpenAIProviderSettings)
    anthropic: AIAnthropicProviderSettings = Field(
        default_factory=AIAnthropicProviderSettings
    )
    google: AIGoogleProviderSettings = Field(default_factory=AIGoogleProviderSettings)
    aws: AIAwsProviderSettings = Field(default_factory=AIAwsProviderSettings)
    openai_compatible: AIOpenAICompatibleProviderSettings = Field(
        default_factory=AIOpenAICompatibleProviderSettings
    )
    azure_speech: AIAzureSpeechProviderSettings = Field(
        default_factory=AIAzureSpeechProviderSettings
    )


class AIVoiceSelection(BaseModel):
    """The `voice` section."""

    model_config = ConfigDict(frozen=True)

    engine: str | None = None
    tts_model: str | None = None
    language: str | None = None


class AIModelsConfig(BaseModel):
    """
    The whole file.

    Attributes:
        allow: Model globs the library may use; empty means all.
        deny: Model globs the library may never use; wins over allow.
        completions: Primary engine and model, chain, reasons, retry policy.
        classifier: The classification engine and model.
        router: The complexity router.
        model_check: The configured-model check mode and timeout.
        embeddings, images, videos, voice: Per-capability selections.
        geo_residency: Geo hint (`US`), as AI_API_GEO_RESIDENCY.
        providers: Per-provider settings that are not secrets (base URLs,
            regions, Vertex project, OpenAI-compatible server details).
        middleware: The PII redaction and observability middleware, the
            same list the middleware YAML holds under `middleware`.
        strict_deprecations: True turns a deprecated-model warning into a
            startup error.
        config_version: A label for this configuration (a date, a release,
            a commit); every call the completions client makes carries it
            as the `config_version` tag, next to the computed `config_id`.
        service: The name of the service this configuration belongs to,
            carried as the `service` tag for cost attribution.
        environments: Per-environment partial configs; AI_MODELS_ENV picks
            one to merge over the rest of the file.
        environment: The environment merged into this config, or None.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # A free-text note for humans reading the file; ignored by the library.
    comment: str | None = Field(default=None, alias="_comment")
    # The JSON Schema an editor validates the file against; ignored here.
    schema_ref: str | None = Field(default=None, alias="$schema")
    config_version: str | None = None
    # The service this configuration belongs to; every call and request
    # cost report carries it as the `service` tag for cost attribution.
    service: str | None = None
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)
    completions: AICompletionsSelection = Field(default_factory=AICompletionsSelection)
    classifier: AIClassifierSelection = Field(default_factory=AIClassifierSelection)
    router: AIRouterConfig | None = None
    model_check: AIModelCheckSelection = Field(default_factory=AIModelCheckSelection)
    embeddings: AIEmbeddingsSelection = Field(default_factory=AIEmbeddingsSelection)
    images: AIImagesSelection = Field(default_factory=AIImagesSelection)
    videos: AIVideosSelection = Field(default_factory=AIVideosSelection)
    voice: AIVoiceSelection = Field(default_factory=AIVoiceSelection)
    geo_residency: str | None = None
    providers: AIProvidersSelection = Field(default_factory=AIProvidersSelection)
    # Entries are validated one by one when the middleware loads, as the YAML
    # file's are, so one bad entry is skipped with a warning.
    middleware: list[dict[str, Any]] | None = None
    strict_deprecations: bool | None = None
    # Partial configs by environment name, merged by load_models_config; each
    # is validated as part of the merged config, not on its own.
    environments: dict[str, dict[str, Any]] | None = None
    # Set by the loader to the environment it merged; not written by hand.
    environment: str | None = None

    @field_validator("allow", "deny")
    @classmethod
    def _normalize_patterns(cls, value: list[str]) -> list[str]:
        # Normal return with the patterns in the policy's canonical form.
        return AIModelPolicy._normalize(value)

    @property
    def policy(self) -> AIModelPolicy:
        """The allow/deny policy."""
        # Normal return with the policy model.
        return AIModelPolicy(allow=self.allow, deny=self.deny)

    def setting(self, str_key: str) -> str | None:
        """
        The value this file supplies for one `.env`-style key, or None.

        Keeps every reader in the library unchanged: EnvSettings consults
        this after the process environment and `.env`.

        Args:
            str_key: A settings key such as COMPLETIONS_MODEL_NAME.

        Returns:
            The value as a string, or None when the file does not set it.
        """
        getter: Callable[[AIModelsConfig], Any] | None = DICT_SETTING_GETTERS.get(
            str_key
        )
        if getter is None:
            # Early return: not a key this file supplies.
            return None
        raw: Any = getter(self)
        if raw is None or raw == "" or raw == []:
            # Early return: unset in the file.
            return None
        if isinstance(raw, list):
            # Early return with the CSV the env readers parse.
            return ",".join(str(item) for item in raw)
        if isinstance(raw, BaseModel):
            # Early return with the JSON the router loader parses.
            return raw.model_dump_json(exclude_none=True)
        # Normal return with the scalar as text.
        return str(raw)


def _classifier_pair(config: AIModelsConfig) -> str | None:
    """engine:model for COMPLETIONS_ROUTER_CLASSIFIER, or None."""
    if not config.classifier.engine:
        # Early return: no classifier named.
        return None
    # Normal return with the pair.
    return (
        f"{config.classifier.engine}:{config.classifier.model}"
        if config.classifier.model
        else config.classifier.engine
    )


def _router_with_classifier(config: AIModelsConfig) -> AIRouterConfig | None:
    """
    The router for COMPLETIONS_ROUTER_CONFIG, carrying the `classifier`
    section's base_url, timeout, instructions, and min_confidence when the
    router names no
    classifier of its own (engine and model travel as
    COMPLETIONS_ROUTER_CLASSIFIER).
    """
    router: AIRouterConfig | None = config.router
    if router is None or router.classifier is not None:
        # Early return: no router, or it carries its own classifier config.
        return router
    selection: AIClassifierSelection = config.classifier
    if (
        selection.base_url is None
        and selection.timeout_seconds is None
        and selection.instructions is None
        and selection.min_confidence is None
    ):
        # Early return: nothing to carry over.
        return router
    dict_classifier: dict[str, Any] = {
        "engine": selection.engine,
        "model": selection.model,
        "base_url": selection.base_url,
        "instructions": selection.instructions,
    }
    if selection.timeout_seconds is not None:
        dict_classifier["timeout_seconds"] = selection.timeout_seconds
    if selection.min_confidence is not None:
        dict_classifier["min_confidence"] = selection.min_confidence
    # Normal return with the classifier section folded in.
    return router.model_copy(
        update={"classifier": AIRouterClassifierConfig(**dict_classifier)}
    )


# Settings key -> the models.json field that replaced it, for the
# deprecation warning when a key is still set in the environment or .env.
DICT_SETTING_FIELD_NAMES: dict[str, str] = {
    "COMPLETIONS_ENGINE": "completions.engine",
    "COMPLETIONS_MODEL_NAME": "completions.model",
    "COMPLETIONS_FALLBACKS": "completions.fallbacks",
    "COMPLETIONS_FALLBACK_ON": "completions.fallback_on",
    "COMPLETIONS_RETRY_POLICY": "completions.retry_policy",
    "COMPLETIONS_THINKING": "completions.thinking",
    "CLASSIFIER_ENGINE": "classifier.engine",
    "CLASSIFIER_MODEL_NAME": "classifier.model",
    "COMPLETIONS_ROUTER_CLASSIFIER": "classifier",
    "COMPLETIONS_ROUTER_CONFIG": "router",
    "AI_MODEL_CHECK": "model_check.mode",
    "AI_MODEL_CHECK_TIMEOUT_SECONDS": "model_check.timeout_seconds",
    "EMBEDDING_ENGINE": "embeddings.engine",
    "EMBEDDING_MODEL_NAME": "embeddings.model",
    "EMBEDDING_DIMENSIONS": "embeddings.dimensions",
    "IMAGE_ENGINE": "images.engine",
    "IMAGE_MODEL_NAME": "images.model",
    "VIDEO_ENGINE": "videos.engine",
    "VIDEO_MODEL_NAME": "videos.model",
    "VIDEO_OUTPUT_DIR": "videos.output_dir",
    "VIDEO_POLL_INTERVAL_SECONDS": "videos.poll_interval_seconds",
    "VIDEO_TIMEOUT_SECONDS": "videos.timeout_seconds",
    "AI_VOICE_ENGINE": "voice.engine",
    "AI_VOICE_LANGUAGE": "voice.language",
    "DEFAULT_GEMINI_TTS_MODEL": "voice.tts_model",
    "AI_API_GEO_RESIDENCY": "geo_residency",
    "OPENAI_BASE_URL_OVERRIDE": "providers.openai.base_url",
    "OPENAI_USER": "providers.openai.user",
    "ANTHROPIC_BASE_URL_OVERRIDE": "providers.anthropic.base_url",
    "ANTHROPIC_ADMIN_BASE_URL_OVERRIDE": "providers.anthropic.admin_base_url",
    "ANTHROPIC_USER": "providers.anthropic.user",
    "GOOGLE_AUTH_METHOD": "providers.google.auth_method",
    "GOOGLE_PROJECT_ID": "providers.google.project_id",
    "GOOGLE_LOCATION": "providers.google.location",
    "GOOGLE_GEMINI_BASE_URL_OVERRIDE": "providers.google.base_url",
    "OPENAI_COMPATIBLE_BASE_URL": "providers.openai_compatible.base_url",
    "OPENAI_COMPATIBLE_STRUCTURED_OUTPUT": (
        "providers.openai_compatible.structured_output"
    ),
    "OPENAI_COMPATIBLE_CONTEXT_WINDOW": "providers.openai_compatible.context_window",
    "MICROSOFT_COGNITIVE_SERVICES_REGION": "providers.azure_speech.region",
    "BEDROCK_VIDEO_OUTPUT_S3_URI": "videos.bedrock_output_s3_uri",
    "AI_STRICT_DEPRECATIONS": "strict_deprecations",
    "AI_MIDDLEWARE_CONFIG_PATH": "middleware",
}

# Settings key -> the file value that stands in for it.
DICT_SETTING_GETTERS: dict[str, Callable[[AIModelsConfig], Any]] = {
    "COMPLETIONS_ENGINE": lambda c: c.completions.engine,
    "COMPLETIONS_MODEL_NAME": lambda c: c.completions.model,
    "COMPLETIONS_FALLBACKS": lambda c: c.completions.fallbacks,
    "COMPLETIONS_FALLBACK_ON": lambda c: c.completions.fallback_on,
    "COMPLETIONS_RETRY_POLICY": lambda c: c.completions.retry_policy,
    # The thinking level has no .env key of its own; this is the settings
    # name the clients read, served from the file.
    "COMPLETIONS_THINKING": lambda c: c.completions.thinking,
    "CLASSIFIER_ENGINE": lambda c: c.classifier.engine,
    "CLASSIFIER_MODEL_NAME": lambda c: c.classifier.model,
    # No .env key ever held this; it is served from the file only.
    "CLASSIFIER_THINKING": lambda c: c.classifier.thinking,
    "COMPLETIONS_ROUTER_CLASSIFIER": _classifier_pair,
    "COMPLETIONS_ROUTER_CONFIG": _router_with_classifier,
    "AI_MODEL_CHECK": lambda c: c.model_check.mode,
    "AI_MODEL_CHECK_TIMEOUT_SECONDS": lambda c: c.model_check.timeout_seconds,
    "EMBEDDING_ENGINE": lambda c: c.embeddings.engine,
    "EMBEDDING_MODEL_NAME": lambda c: c.embeddings.model,
    "EMBEDDING_DIMENSIONS": lambda c: c.embeddings.dimensions,
    "IMAGE_ENGINE": lambda c: c.images.engine,
    "IMAGE_MODEL_NAME": lambda c: c.images.model,
    "VIDEO_ENGINE": lambda c: c.videos.engine,
    "VIDEO_MODEL_NAME": lambda c: c.videos.model,
    "VIDEO_OUTPUT_DIR": lambda c: c.videos.output_dir,
    "VIDEO_POLL_INTERVAL_SECONDS": lambda c: c.videos.poll_interval_seconds,
    "VIDEO_TIMEOUT_SECONDS": lambda c: c.videos.timeout_seconds,
    "AI_VOICE_ENGINE": lambda c: c.voice.engine,
    "AI_VOICE_LANGUAGE": lambda c: c.voice.language,
    "DEFAULT_GEMINI_TTS_MODEL": lambda c: c.voice.tts_model,
    "AI_API_GEO_RESIDENCY": lambda c: c.geo_residency,
    "OPENAI_BASE_URL_OVERRIDE": lambda c: c.providers.openai.base_url,
    "OPENAI_USER": lambda c: c.providers.openai.user,
    "ANTHROPIC_BASE_URL_OVERRIDE": lambda c: c.providers.anthropic.base_url,
    "ANTHROPIC_ADMIN_BASE_URL_OVERRIDE": lambda c: c.providers.anthropic.admin_base_url,
    "ANTHROPIC_USER": lambda c: c.providers.anthropic.user,
    "GOOGLE_AUTH_METHOD": lambda c: c.providers.google.auth_method,
    "GOOGLE_PROJECT_ID": lambda c: c.providers.google.project_id,
    "GOOGLE_LOCATION": lambda c: c.providers.google.location,
    "GOOGLE_GEMINI_BASE_URL_OVERRIDE": lambda c: c.providers.google.base_url,
    # AWS sets AWS_REGION itself on Lambda and other hosts, so the variable
    # is not deprecated; the file serves it when the environment does not.
    "AWS_REGION": lambda c: c.providers.aws.region,
    "OPENAI_COMPATIBLE_BASE_URL": lambda c: c.providers.openai_compatible.base_url,
    "OPENAI_COMPATIBLE_STRUCTURED_OUTPUT": (
        lambda c: c.providers.openai_compatible.structured_output
    ),
    "OPENAI_COMPATIBLE_CONTEXT_WINDOW": (
        lambda c: c.providers.openai_compatible.context_window
    ),
    # No .env key ever held this; it is served from the file only.
    "OPENAI_COMPATIBLE_THINKING_CONTROL": (
        lambda c: c.providers.openai_compatible.thinking_control
    ),
    "MICROSOFT_COGNITIVE_SERVICES_REGION": lambda c: c.providers.azure_speech.region,
    "BEDROCK_VIDEO_OUTPUT_S3_URI": lambda c: c.videos.bedrock_output_s3_uri,
    "AI_STRICT_DEPRECATIONS": lambda c: c.strict_deprecations,
}

FROZENSET_FILE_BACKED_SETTINGS: frozenset[str] = frozenset(DICT_SETTING_GETTERS)


def merge_config_overlay(
    dict_base: dict[str, Any], dict_overlay: dict[str, Any]
) -> dict[str, Any]:
    """
    Deep-merges an environment's partial config over the base: objects merge
    key by key; lists and values replace.
    """
    dict_merged: dict[str, Any] = dict(dict_base)
    # Loop over the overlay's keys, recursing into objects on both sides.
    for str_key, value in dict_overlay.items():
        base_value: Any = dict_merged.get(str_key)
        if isinstance(value, dict) and isinstance(base_value, dict):
            dict_merged[str_key] = merge_config_overlay(base_value, value)
        else:
            dict_merged[str_key] = value
    # Normal return with the merged payload.
    return dict_merged


def _apply_environment(
    dict_payload: dict[str, Any], str_environment: str | None
) -> dict[str, Any]:
    """
    Merges the named environment over the payload.

    Raises:
        ValueError: The environment is not defined, or an environment entry
            is not an object or nests `environments`.
    """
    raw_environments: Any = dict_payload.get("environments")
    if raw_environments is not None and not isinstance(raw_environments, dict):
        raise ValueError("Models config 'environments' must be an object.")
    dict_environments: dict[str, Any] = raw_environments or {}
    # Loop over the entries to reject shapes the merge cannot apply.
    for str_name, entry in dict_environments.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"Models config environments.{str_name} must be an object."
            )
        if "environments" in entry or "environment" in entry:
            raise ValueError(
                f"Models config environments.{str_name} cannot set "
                "'environments' or 'environment'."
            )
    if "environment" in dict_payload:
        raise ValueError(
            "Models config 'environment' is set by the loader from "
            f"{AI_MODELS_ENV_KEY}; remove it from the file."
        )
    str_name: str = (str_environment or "").strip()
    if not str_name:
        # Early return: no environment selected.
        return dict_payload
    if str_name not in dict_environments:
        raise ValueError(
            f"{AI_MODELS_ENV_KEY}={str_name!r} names no entry under "
            "'environments' in the models config; defined: "
            f"{sorted(dict_environments) or 'none'}."
        )
    dict_merged: dict[str, Any] = merge_config_overlay(
        dict_payload, dict_environments[str_name]
    )
    dict_merged["environment"] = str_name
    # Normal return with the environment merged in.
    return dict_merged


def load_models_config(
    str_value: str, str_environment: str | None = None
) -> AIModelsConfig:
    """
    Parses the models config from a file path or inline JSON, merging the
    named environment over it.

    Args:
        str_value: A path to a `.json`, `.yaml`, or `.yml` file, or inline
            JSON text.
        str_environment: The `environments` entry to merge (AI_MODELS_ENV),
            or None.

    Returns:
        The validated config.

    Raises:
        ValueError: The value is blank, the file is missing, the content is
            invalid or fails validation, or the environment is not defined.
    """
    str_stripped: str = str_value.strip()
    if not str_stripped:
        raise ValueError("Models config is blank.")
    dict_payload: Any
    if str_stripped.startswith(("{", "[")):
        try:
            dict_payload = json.loads(str_stripped)
        except json.JSONDecodeError as exception:
            raise ValueError(
                f"Models config is not valid JSON: {exception}"
            ) from exception
    else:
        path: Path = Path(str_stripped).expanduser()
        if not path.is_file():
            raise ValueError(f"Models config file not found: {path}")
        str_text: str = path.read_text(encoding="utf-8")
        if path.suffix.lower() in TUPLE_YAML_SUFFIXES:
            try:
                import yaml
            except ImportError as exception:
                raise ValueError(
                    f"Models config {path} is YAML, which needs PyYAML "
                    "(pip install pyyaml); or write the file as JSON."
                ) from exception
            try:
                dict_payload = yaml.safe_load(str_text)
            except yaml.YAMLError as exception:
                raise ValueError(
                    f"Models config {path} is not valid YAML: {exception}"
                ) from exception
        else:
            try:
                dict_payload = json.loads(str_text)
            except json.JSONDecodeError as exception:
                raise ValueError(
                    f"Models config {path} is not valid JSON: {exception}"
                ) from exception
    if dict_payload is None:
        dict_payload = {}
    if not isinstance(dict_payload, dict):
        raise ValueError("Models config must be an object at the top level.")
    dict_payload = _apply_environment(dict_payload, str_environment)
    try:
        # Normal return with the validated config.
        return AIModelsConfig.model_validate(dict_payload)
    except ValueError as exception:
        raise ValueError(f"Models config is invalid: {exception}") from exception


class _ModelsConfigCache:
    """Loads the file once per (path, mtime) for the process."""

    _lock: threading.Lock = threading.Lock()
    _dict_cache: dict[tuple[str, float, str], AIModelsConfig] = {}
    _dict_inline: dict[tuple[str, str], AIModelsConfig] = {}

    @classmethod
    def get(cls, str_value: str, str_environment: str | None = None) -> AIModelsConfig:
        str_stripped: str = str_value.strip()
        str_env: str = (str_environment or "").strip()
        if str_stripped.startswith(("{", "[")):
            key_inline: tuple[str, str] = (str_stripped, str_env)
            with cls._lock:
                if key_inline not in cls._dict_inline:
                    cls._dict_inline[key_inline] = load_models_config(
                        str_stripped, str_env
                    )
                # Early return with the inline config.
                return cls._dict_inline[key_inline]
        path: Path = Path(str_stripped).expanduser()
        float_mtime: float = path.stat().st_mtime if path.is_file() else -1.0
        key: tuple[str, float, str] = (str(path), float_mtime, str_env)
        with cls._lock:
            if key not in cls._dict_cache:
                cls._dict_cache[key] = load_models_config(str_stripped, str_env)
            # Normal return with the cached config.
            return cls._dict_cache[key]

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._dict_cache.clear()
            cls._dict_inline.clear()


def resolve_models_config(
    str_configured_value: str | None,
    *,
    bool_configured: bool,
    str_environment: str | None = None,
) -> AIModelsConfig | None:
    """
    Finds and loads the models config for this process.

    Args:
        str_configured_value: The AI_MODELS_CONFIG value, if any.
        bool_configured: True when AI_MODELS_CONFIG was set at all; a blank
            value then means "no file", while an unset key means "use
            ./models.json if it exists".
        str_environment: The AI_MODELS_ENV value, if any.

    Returns:
        The config, or None when there is none.
    """
    if bool_configured:
        str_value: str = (str_configured_value or "").strip()
        if not str_value:
            # Early return: turned off on purpose.
            return None
        # Early return with the configured file.
        return _ModelsConfigCache.get(str_value, str_environment)
    if os.path.isfile(DEFAULT_MODELS_CONFIG_FILENAME):
        # Early return with the default file.
        return _ModelsConfigCache.get(DEFAULT_MODELS_CONFIG_FILENAME, str_environment)
    # Normal return: no file.
    return None


def clear_models_config_cache() -> None:
    """Forgets loaded files (tests, or a deliberate reload)."""
    _ModelsConfigCache.clear()
