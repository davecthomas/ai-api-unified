# config_checks.py
"""
Startup checks on the completions configuration as a whole, and the
identity every routed call records.

Config identity: `config_id` is a short one-way hash of the configured
routing (primary, fallbacks, trigger reasons, router, cooldown, thinking).
The same configuration gives the same id in every process, so a log or cost
event names the configuration that routed it. `config_version` is an
optional label set in models.json. Both ride on the observability tags of
every call the client makes, and on its request cost report.

Requirements: `completions.require` in models.json declares what callers
need from every route, and the factory checks it once at build, without
building any client:

- capabilities: each route's primary must support every listed capability
  (structured_output, tool_use, streaming, async, ...). A call starts at the
  primary, and a primary without the capability raises, so a fallback that
  has it does not help.
- second_provider: each route must hold a model on a provider other than
  its primary's, so one provider's outage cannot take the route down.

A route is the default chain (primary plus fallbacks) and each router tier.
An unmet requirement logs a WARNING, or raises in strict mode.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..ai_base import AICompletionsCapabilitiesBase
from ..completions.ai_fallback_completions import AIFallbackCandidate

_LOGGER: logging.Logger = logging.getLogger(__name__)

# Value of the `ai_config_event` field on an unmet-requirement record.
CONFIG_EVENT_REQUIREMENT_UNMET: str = "requirement_unmet"

# Observability tag names.
TAG_CONFIG_ID: str = "config_id"
TAG_CONFIG_VERSION: str = "config_version"

INT_CONFIG_ID_HEX_CHARS: int = 12

# Provider behind each engine token, for second_provider. Engines on one
# provider share its outages, whatever the model.
DICT_PROVIDER_BY_TOKEN: dict[str, str] = {
    "openai": "openai",
    "openai-responses": "openai",
    "openai-decisions": "openai",
    "claude": "anthropic",
    "google-gemini": "google",
    "anthropic": "aws-bedrock",
    "bedrock": "aws-bedrock",
    "nova": "aws-bedrock",
    "llama": "aws-bedrock",
    "mistral": "aws-bedrock",
    "cohere": "aws-bedrock",
    "ai21": "aws-bedrock",
}

FROZENSET_CAPABILITY_FIELDS: frozenset[str] = frozenset(
    name.removeprefix("supports_")
    for name in AICompletionsCapabilitiesBase.model_fields
    if name.startswith("supports_")
    and AICompletionsCapabilitiesBase.model_fields[name].annotation is bool
)


class AIConfigRequirements(BaseModel):
    """
    What every completions route must offer, checked at build.

    Attributes:
        capabilities: Capability names each route's primary must support,
            without the `supports_` prefix (structured_output, tool_use,
            streaming, async, batch, token_counting, ...).
        second_provider: True requires each route to hold a model on a
            provider other than its primary's.
        mode: "warn" logs an unmet requirement; "strict" raises.
    """

    model_config = ConfigDict(frozen=True)

    capabilities: list[str] = Field(default_factory=list)
    second_provider: bool = False
    mode: Literal["warn", "strict"] = "warn"

    @field_validator("capabilities", mode="before")
    @classmethod
    def _parse_capabilities(cls, value: Any) -> Any:
        if not isinstance(value, list):
            # Early return: pydantic reports the type error.
            return value
        list_names: list[str] = [
            str(item).strip().lower().removeprefix("supports_") for item in value
        ]
        list_unknown: list[str] = [
            name for name in list_names if name not in FROZENSET_CAPABILITY_FIELDS
        ]
        if list_unknown:
            raise ValueError(
                f"Unknown capabilities {list_unknown}; expected some of "
                f"{sorted(FROZENSET_CAPABILITY_FIELDS)}."
            )
        # Normal return with the normalized names.
        return list_names

    @property
    def empty(self) -> bool:
        """True when nothing is required."""
        # Normal return with the emptiness check.
        return not self.capabilities and not self.second_provider


def parse_config_requirements(value: Any) -> AIConfigRequirements:
    """Normalizes `completions.require` or the factory's `require=`."""
    if isinstance(value, AIConfigRequirements):
        # Early return: already parsed.
        return value
    if value is None:
        # Early return: nothing required.
        return AIConfigRequirements()
    # Normal return with the parsed mapping.
    return AIConfigRequirements.model_validate(value)


def compute_config_id(dict_config: dict[str, Any]) -> str:
    """
    A short, stable, one-way id for a routing configuration.

    Args:
        dict_config: JSON-serializable description of the configuration.

    Returns:
        The first INT_CONFIG_ID_HEX_CHARS hex characters of its SHA-256.
    """
    str_canonical: str = json.dumps(
        dict_config, sort_keys=True, separators=(",", ":"), default=str
    )
    # Normal return with the truncated digest.
    return hashlib.sha256(str_canonical.encode()).hexdigest()[:INT_CONFIG_ID_HEX_CHARS]


def provider_of(candidate: AIFallbackCandidate) -> str:
    """The provider a candidate runs on; an OpenAI-compatible server is its URL."""
    if candidate.engine == "openai-compatible":
        # Early return: each compatible server is its own provider.
        return f"openai-compatible:{candidate.base_url or 'default'}"
    # Normal return with the mapped provider, or the token itself.
    return DICT_PROVIDER_BY_TOKEN.get(candidate.engine, candidate.engine)


def static_capabilities(
    candidate: AIFallbackCandidate,
) -> AICompletionsCapabilitiesBase | None:
    """
    A model's capabilities from the engine's own table, without building a
    client; None when the engine has no table or its extra is missing.
    """
    str_model: str = (candidate.model or "").strip().lower()
    if not str_model:
        # Early return: the engine default model is not named here.
        return None
    try:
        if candidate.engine in ("openai", "openai-responses"):
            from ..completions.ai_openai_completions import (
                AICompletionsCapabilitiesOpenAI,
            )

            # Early return with the OpenAI table.
            return AICompletionsCapabilitiesOpenAI.for_model(str_model)
        if candidate.engine == "claude":
            from ..completions.ai_anthropic_completions import (
                AICompletionsCapabilitiesAnthropic,
            )

            # Early return with the Anthropic table.
            return AICompletionsCapabilitiesAnthropic.for_model(str_model)
        if candidate.engine == "google-gemini":
            from ..completions.ai_google_gemini_capabilities import (
                AICompletionsCapabilitiesGoogle,
            )

            # Early return with the Gemini table.
            return AICompletionsCapabilitiesGoogle.for_model(str_model)
        if DICT_PROVIDER_BY_TOKEN.get(candidate.engine) == "aws-bedrock":
            from ..completions.ai_bedrock_completions import (
                AICompletionsCapabilitiesBedrock,
            )

            # Early return with the Bedrock table; only the flags matter here.
            return AICompletionsCapabilitiesBedrock.for_model(
                str_model, context_window_length=1
            )
    except ImportError:
        # Early return: the engine's extra is not installed.
        return None
    # Normal return: no table for this engine.
    return None


def unmet_requirements(
    requirements: AIConfigRequirements,
    list_routes: Sequence[tuple[str, Sequence[AIFallbackCandidate]]],
) -> list[str]:
    """
    Every unmet requirement, one line each.

    Args:
        requirements: What every route must offer.
        list_routes: (route label, its candidates with the primary first).

    Returns:
        Problems; empty when every route meets every requirement.
    """
    list_problems: list[str] = []
    # Loop over routes so each is checked on its own.
    for str_route, candidates in list_routes:
        if not candidates:
            continue
        primary: AIFallbackCandidate = candidates[0]
        if requirements.capabilities:
            capabilities: AICompletionsCapabilitiesBase | None = static_capabilities(
                primary
            )
            if capabilities is None:
                _LOGGER.info(
                    "CONFIG: cannot check capabilities of %s on route %s "
                    "(no capability table for its engine or model).",
                    primary.label,
                    str_route,
                )
            else:
                list_missing: list[str] = [
                    name
                    for name in requirements.capabilities
                    if not getattr(capabilities, f"supports_{name}", False)
                ]
                if list_missing:
                    list_problems.append(
                        f"route {str_route}: primary {primary.label} lacks "
                        f"{', '.join(list_missing)}"
                    )
        if requirements.second_provider:
            str_provider: str = provider_of(primary)
            if all(provider_of(candidate) == str_provider for candidate in candidates):
                list_problems.append(
                    f"route {str_route}: every model is on {str_provider}; "
                    "add a fallback on another provider"
                )
    # Normal return with the problems found.
    return list_problems


def enforce_requirements(
    requirements: AIConfigRequirements,
    list_routes: Sequence[tuple[str, Sequence[AIFallbackCandidate]]],
) -> list[str]:
    """
    Checks the routes and logs, or raises in strict mode.

    Returns:
        The problems found (also logged).

    Raises:
        ValueError: Strict mode found a problem.
    """
    if requirements.empty:
        # Early return: nothing to check.
        return []
    list_problems: list[str] = unmet_requirements(requirements, list_routes)
    if list_problems and requirements.mode == "strict":
        raise ValueError(
            "completions.require (strict) is not met: " + "; ".join(list_problems)
        )
    # Loop over problems so each is its own record.
    for str_problem in list_problems:
        _LOGGER.warning(
            "CONFIG REQUIREMENT NOT MET: %s.",
            str_problem,
            extra={
                "ai_config_event": CONFIG_EVENT_REQUIREMENT_UNMET,
                "problem": str_problem,
            },
        )
    # Normal return with the problems.
    return list_problems
