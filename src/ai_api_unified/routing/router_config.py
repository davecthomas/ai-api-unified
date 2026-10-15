# router_config.py
"""
Configuration for the complexity-based model router.

A router config is an ordered list of tiers, ascending capability. Each tier
names itself, describes itself to the classifier, and lists an ordered
engine:model chain (the first is the tier's primary, the rest its fallback
chain). The config comes from code (AIRouterConfig) or from JSON, as a file
path or inline text in COMPLETIONS_ROUTER_CONFIG.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..ai_provider_exceptions import AiFallbackReason
from ..completions.ai_fallback_completions import (
    AIFallbackCandidate,
    parse_fallback_candidates,
    parse_fallback_reasons,
)

# Tier names double as classifier choice values and log fields.
PATTERN_TIER_NAME: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9_-]*$")

# Built-in descriptions, so a config that uses these names needs only models.
DICT_DEFAULT_TIER_DESCRIPTIONS: dict[str, str] = {
    "easy": (
        "Short, factual, or formulaic requests: a lookup, a rewrite, a label, "
        "a one-line answer, simple formatting. A small fast model answers well."
    ),
    "medium": (
        "Requests that need some reasoning or synthesis: summarizing a "
        "document, drafting a message, explaining a concept, light analysis, "
        "a few steps of logic."
    ),
    "hard": (
        "Multi-step reasoning, long or dense context, expert judgment, "
        "cross-file code changes, math, or anything where a wrong answer is "
        "costly. Needs the strongest model."
    ),
}

STR_ROUTER_CLASSIFIER_NONE: str = "none"

DEFAULT_ROUTER_CLASSIFIER_TIMEOUT_SECONDS: float = 5.0


class AIRouterTier(BaseModel):
    """
    One complexity tier.

    Attributes:
        name: Lowercase token, unique in the config; the classifier returns it.
        description: Prose the classifier reads to recognize the tier. Built
            in for easy, medium, and hard; required otherwise.
        candidates: The tier's models in order: the primary, then its chain.
        fallback_on: Reasons that move a request along the tier's chain;
            None means the router-level default.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    candidates: list[AIFallbackCandidate] = Field(min_length=1)
    fallback_on: frozenset[AiFallbackReason] | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_models_key(cls, data: Any) -> Any:
        """Lets JSON spell the chain as `models` (strings or dicts)."""
        if not isinstance(data, dict):
            # Early return: nothing to translate.
            return data
        dict_data: dict[str, Any] = dict(data)
        if dict_data.get("fallback_on") is not None:
            dict_data["fallback_on"] = parse_fallback_reasons(dict_data["fallback_on"])
        if "models" not in dict_data:
            # Early return: the chain is spelled as candidates.
            return dict_data
        raw_models: Any = dict_data.pop("models")
        if not isinstance(raw_models, list):
            raise ValueError("A tier's `models` must be a list.")
        list_candidates: list[AIFallbackCandidate] = []
        # Loop over entries so each spelling maps to a candidate.
        for item in raw_models:
            if isinstance(item, str):
                list_candidates.extend(parse_fallback_candidates(item))
            elif isinstance(item, dict):
                list_candidates.append(AIFallbackCandidate(**item))
            elif isinstance(item, AIFallbackCandidate):
                list_candidates.append(item)
            else:
                raise ValueError(f"Unrecognized model entry {item!r}.")
        dict_data["candidates"] = list_candidates
        # Normal return with the translated payload.
        return dict_data

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        str_name: str = value.strip().lower()
        if not PATTERN_TIER_NAME.match(str_name):
            raise ValueError(
                f"Tier name {value!r} must match {PATTERN_TIER_NAME.pattern}."
            )
        # Normal return with the normalized name.
        return str_name

    @model_validator(mode="after")
    def _fill_description(self) -> "AIRouterTier":
        if not self.description.strip():
            str_default: str | None = DICT_DEFAULT_TIER_DESCRIPTIONS.get(self.name)
            if str_default is None:
                raise ValueError(
                    f"Tier {self.name!r} needs a description; only "
                    f"{', '.join(DICT_DEFAULT_TIER_DESCRIPTIONS)} have built-in ones."
                )
            object.__setattr__(self, "description", str_default)
        # Normal return with the described tier.
        return self

    @property
    def label(self) -> str:
        """Name and primary, for logs."""
        # Normal return with the log label.
        return f"{self.name}({self.candidates[0].label})"


class AIRouterClassifierConfig(BaseModel):
    """
    Which classifier the router uses.

    Attributes:
        engine: Classification engine token, or "none" to route without a
            classifier. None means the factory's own resolution
            (CLASSIFIER_ENGINE, then openai-decisions when OPENAI_API_KEY is set).
        model: Classification model; None means the engine's default.
        base_url: Optional API base-URL override.
        timeout_seconds: Per-call classifier timeout.
        instructions: Prose prepended to the built-in routing instructions.
    """

    model_config = ConfigDict(frozen=True)

    engine: str | None = None
    model: str | None = None
    base_url: str | None = None
    timeout_seconds: float = DEFAULT_ROUTER_CLASSIFIER_TIMEOUT_SECONDS
    instructions: str | None = None

    @property
    def disabled(self) -> bool:
        """True when the config turns the classifier off."""
        # Normal return with the "none" check.
        return (self.engine or "").strip().lower() == STR_ROUTER_CLASSIFIER_NONE


class AIRouterConfig(BaseModel):
    """
    The router's tiers and policy.

    Attributes:
        tiers: Ascending capability; at least one.
        classifier: Classifier selection; None means the factory's resolution.
        default_tier: Tier used when classification is unavailable; None
            means the default chain (the configured completions LLM).
        on_exhausted: When a tier's chain is exhausted on a trigger reason,
            "escalate" tries the next tier up; "raise" re-raises.
        fallback_on: Default trigger reasons for every tier's chain; None
            means the factory's resolution (COMPLETIONS_FALLBACK_ON, then the
            library default).
    """

    model_config = ConfigDict(frozen=True)

    tiers: list[AIRouterTier] = Field(min_length=1)
    classifier: AIRouterClassifierConfig | None = None
    default_tier: str | None = None
    on_exhausted: Literal["escalate", "raise"] = "escalate"
    fallback_on: frozenset[AiFallbackReason] | None = None
    # Tiers the model check removed, each mapped to the next higher tier that
    # survived (None means the default route). A call that still names one
    # goes there. Filled by prune_router_config; not part of the JSON schema.
    pruned_tiers: dict[str, str | None] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _parse_reasons(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("fallback_on") is not None:
            dict_data: dict[str, Any] = dict(data)
            dict_data["fallback_on"] = parse_fallback_reasons(dict_data["fallback_on"])
            # Early return with parsed reasons.
            return dict_data
        # Normal return unchanged.
        return data

    @model_validator(mode="after")
    def _validate_tiers(self) -> "AIRouterConfig":
        list_names: list[str] = [tier.name for tier in self.tiers]
        if len(set(list_names)) != len(list_names):
            raise ValueError(f"Tier names must be unique; got {list_names}.")
        if self.default_tier is not None:
            str_default: str = self.default_tier.strip().lower()
            if str_default not in list_names:
                raise ValueError(
                    f"default_tier {self.default_tier!r} is not one of {list_names}."
                )
            object.__setattr__(self, "default_tier", str_default)
        # Normal return with the validated config.
        return self

    def resolve_tier_name(self, str_name: str) -> str | None:
        """
        A tier name as the router should use it: itself when it survived the
        model check, else the next higher surviving tier, else None for the
        default route.

        Args:
            str_name: A configured tier name, possibly pruned.

        Returns:
            The effective tier name, or None.
        """
        if str_name in self.pruned_tiers:
            # Early return with the remap.
            return self.pruned_tiers[str_name]
        # Normal return: the name as given.
        return str_name

    def tier_index(self, str_name: str) -> int | None:
        """Position of a tier by name, or None."""
        # Loop over tiers for the name.
        for int_index, tier in enumerate(self.tiers):
            if tier.name == str_name:
                # Early return with the position.
                return int_index
        # Normal return: no such tier.
        return None


def prune_router_config(
    config: AIRouterConfig,
    callable_keep: "Callable[[AIFallbackCandidate], bool]",
) -> AIRouterConfig | None:
    """
    Drops candidates the model check rejected, and tiers left with none.

    A dropped tier maps to the next higher surviving tier (routing goes up,
    never down); a default_tier that was dropped moves the same way.

    Args:
        config: The configured router.
        callable_keep: True for a candidate that survives.

    Returns:
        The pruned config, or None when no tier survived.
    """
    list_kept: list[AIRouterTier] = []
    list_dropped_names: list[str] = []
    # Loop over tiers so each keeps only its surviving candidates.
    for tier in config.tiers:
        list_candidates: list[AIFallbackCandidate] = [
            candidate for candidate in tier.candidates if callable_keep(candidate)
        ]
        if list_candidates:
            list_kept.append(tier.model_copy(update={"candidates": list_candidates}))
        else:
            list_dropped_names.append(tier.name)
    if not list_kept:
        # Early return: nothing left to route to.
        return None
    dict_pruned: dict[str, str | None] = dict(config.pruned_tiers)
    list_names: list[str] = [tier.name for tier in config.tiers]
    set_kept_names: set[str] = {tier.name for tier in list_kept}
    # Loop over dropped tiers to find the next higher survivor for each.
    for str_dropped in list_dropped_names:
        str_next: str | None = next(
            (
                str_name
                for str_name in list_names[list_names.index(str_dropped) + 1 :]
                if str_name in set_kept_names
            ),
            None,
        )
        dict_pruned[str_dropped] = str_next
    str_default: str | None = config.default_tier
    if str_default is not None and str_default in dict_pruned:
        str_default = dict_pruned[str_default]
    # Normal return with the pruned config.
    return config.model_copy(
        update={
            "tiers": list_kept,
            "default_tier": str_default,
            "pruned_tiers": dict_pruned,
        }
    )


def load_router_config(str_value: str) -> AIRouterConfig:
    """
    Parses a router config from a file path or inline JSON.

    Args:
        str_value: A path to a JSON file, or inline JSON text.

    Returns:
        The validated config.

    Raises:
        ValueError: The value is blank, the file is missing, or the JSON is
            invalid or fails validation.
    """
    str_stripped: str = str_value.strip()
    if not str_stripped:
        raise ValueError("Router config is blank.")
    str_json: str
    if str_stripped.startswith(("{", "[")):
        str_json = str_stripped
    else:
        path: Path = Path(str_stripped).expanduser()
        if not path.is_file():
            raise ValueError(f"Router config file not found: {path}")
        str_json = path.read_text(encoding="utf-8")
    try:
        dict_payload: Any = json.loads(str_json)
    except json.JSONDecodeError as exception:
        raise ValueError(f"Router config is not valid JSON: {exception}") from exception
    if not isinstance(dict_payload, dict):
        raise ValueError("Router config must be a JSON object.")
    # Normal return with the validated config.
    return AIRouterConfig.model_validate(dict_payload)


def classification_text_for(
    prompt: str | None, messages: list[dict[str, Any]] | None
) -> str:
    """
    Picks the text the router classifies: the latest user message, else the
    prompt.

    The system prompt is left out: it is long, constant across calls, and
    carries no complexity signal; the tier descriptions carry the policy.

    Args:
        prompt: The call's prompt, if any.
        messages: The call's history, if any.

    Returns:
        The text to classify; empty when the call carries none.
    """
    # Loop backward over the history for the latest user turn.
    for message in reversed(messages or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content: Any = message.get("content")
        if isinstance(content, str) and content.strip():
            # Early return with plain text content.
            return content
        if isinstance(content, list):
            list_parts: list[str] = [
                str(block.get("text"))
                for block in content
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            ]
            if list_parts:
                # Early return with the text blocks joined.
                return "\n".join(list_parts)
        break
    # Normal return with the prompt, or nothing.
    return prompt or ""
