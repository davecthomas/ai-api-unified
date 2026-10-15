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

from .routing.router_config import AIRouterClassifierConfig, AIRouterConfig

_LOGGER: logging.Logger = logging.getLogger(__name__)

AI_MODELS_CONFIG_KEY: str = "AI_MODELS_CONFIG"
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


class AIClassifierSelection(BaseModel):
    """The `classifier` section; also the router's classifier."""

    model_config = ConfigDict(frozen=True)

    engine: str | None = None
    model: str | None = None
    base_url: str | None = None
    timeout_seconds: float | None = None
    instructions: str | None = None


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
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

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
    section's base_url, timeout, and instructions when the router names no
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
    # Normal return with the classifier section folded in.
    return router.model_copy(
        update={"classifier": AIRouterClassifierConfig(**dict_classifier)}
    )


# Settings key -> the file value that stands in for it.
DICT_SETTING_GETTERS: dict[str, Callable[[AIModelsConfig], Any]] = {
    "COMPLETIONS_ENGINE": lambda c: c.completions.engine,
    "COMPLETIONS_MODEL_NAME": lambda c: c.completions.model,
    "COMPLETIONS_FALLBACKS": lambda c: c.completions.fallbacks,
    "COMPLETIONS_FALLBACK_ON": lambda c: c.completions.fallback_on,
    "COMPLETIONS_RETRY_POLICY": lambda c: c.completions.retry_policy,
    "CLASSIFIER_ENGINE": lambda c: c.classifier.engine,
    "CLASSIFIER_MODEL_NAME": lambda c: c.classifier.model,
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
}

FROZENSET_FILE_BACKED_SETTINGS: frozenset[str] = frozenset(DICT_SETTING_GETTERS)


def load_models_config(str_value: str) -> AIModelsConfig:
    """
    Parses the models config from a file path or inline JSON.

    Args:
        str_value: A path to a `.json`, `.yaml`, or `.yml` file, or inline
            JSON text.

    Returns:
        The validated config.

    Raises:
        ValueError: The value is blank, the file is missing, or the content
            is invalid or fails validation.
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
    try:
        # Normal return with the validated config.
        return AIModelsConfig.model_validate(dict_payload)
    except ValueError as exception:
        raise ValueError(f"Models config is invalid: {exception}") from exception


class _ModelsConfigCache:
    """Loads the file once per (path, mtime) for the process."""

    _lock: threading.Lock = threading.Lock()
    _dict_cache: dict[tuple[str, float], AIModelsConfig] = {}
    _dict_inline: dict[str, AIModelsConfig] = {}

    @classmethod
    def get(cls, str_value: str) -> AIModelsConfig:
        str_stripped: str = str_value.strip()
        if str_stripped.startswith(("{", "[")):
            with cls._lock:
                if str_stripped not in cls._dict_inline:
                    cls._dict_inline[str_stripped] = load_models_config(str_stripped)
                # Early return with the inline config.
                return cls._dict_inline[str_stripped]
        path: Path = Path(str_stripped).expanduser()
        float_mtime: float = path.stat().st_mtime if path.is_file() else -1.0
        key: tuple[str, float] = (str(path), float_mtime)
        with cls._lock:
            if key not in cls._dict_cache:
                cls._dict_cache[key] = load_models_config(str_stripped)
            # Normal return with the cached config.
            return cls._dict_cache[key]

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._dict_cache.clear()
            cls._dict_inline.clear()


def resolve_models_config(
    str_configured_value: str | None, *, bool_configured: bool
) -> AIModelsConfig | None:
    """
    Finds and loads the models config for this process.

    Args:
        str_configured_value: The AI_MODELS_CONFIG value, if any.
        bool_configured: True when AI_MODELS_CONFIG was set at all; a blank
            value then means "no file", while an unset key means "use
            ./models.json if it exists".

    Returns:
        The config, or None when there is none.
    """
    if bool_configured:
        str_value: str = (str_configured_value or "").strip()
        if not str_value:
            # Early return: turned off on purpose.
            return None
        # Early return with the configured file.
        return _ModelsConfigCache.get(str_value)
    if os.path.isfile(DEFAULT_MODELS_CONFIG_FILENAME):
        # Early return with the default file.
        return _ModelsConfigCache.get(DEFAULT_MODELS_CONFIG_FILENAME)
    # Normal return: no file.
    return None


def clear_models_config_cache() -> None:
    """Forgets loaded files (tests, or a deliberate reload)."""
    _ModelsConfigCache.clear()
