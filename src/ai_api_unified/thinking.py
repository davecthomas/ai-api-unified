"""
Provider-neutral thinking levels.

Configuration and application code state how much a model should think as
one of six levels, from `off` to `max`. Each engine applies a level through
its provider's own control (Anthropic adaptive thinking and `effort`, OpenAI
`reasoning_effort`, Gemini `thinking_level` or `thinking_budget`, Bedrock
`additionalModelRequestFields`), and a model's capabilities list which
levels it can honor exactly. When a model cannot honor the requested level
the engine applies the nearest level it supports and records both.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_LOGGER: logging.Logger = logging.getLogger(__name__)

COMPLETIONS_THINKING_KEY: str = "COMPLETIONS_THINKING"


class AIThinkingLevel(str, Enum):
    """
    How much a model should think before answering, provider-neutral.

    Members, in order:
        OFF: No thinking. Models whose thinking is always on apply their
            lowest level instead and record it.
        MINIMAL: The least thinking a model offers above off.
        LOW: Light reasoning; quick answers.
        MEDIUM: The default. Balanced depth, mapped to each provider's own
            medium (Anthropic `effort: medium`, OpenAI `reasoning_effort:
            medium`, Gemini `thinking_level: medium`).
        HIGH: Deep reasoning.
        MAX: The most a model offers.
    """

    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"


LIST_THINKING_LEVELS: list[AIThinkingLevel] = list(AIThinkingLevel)
DEFAULT_THINKING_LEVEL: AIThinkingLevel = AIThinkingLevel.MEDIUM

# The structured-output classifier's own level: classification picks one
# label, so it runs with thinking off unless configured (classifier.thinking).
# An always-on model applies its nearest level, which is recorded.
CLASSIFIER_THINKING_KEY: str = "CLASSIFIER_THINKING"
DEFAULT_CLASSIFIER_THINKING_LEVEL: AIThinkingLevel = AIThinkingLevel.OFF


def parse_thinking_level(value: AIThinkingLevel | str | None) -> AIThinkingLevel | None:
    """
    Normalizes a thinking level from configuration or code.

    Args:
        value: A level, its string value (any case, surrounding blanks
            ignored), or None.

    Returns:
        The level, or None for None or a blank string.

    Raises:
        ValueError: When the string names no level.
    """
    if value is None:
        # Early return: nothing configured.
        return None
    if isinstance(value, AIThinkingLevel):
        # Early return: already typed.
        return value
    str_token: str = str(value).strip().lower()
    if not str_token:
        # Early return: a blank setting means unset.
        return None
    try:
        # Normal return with the parsed level.
        return AIThinkingLevel(str_token)
    except ValueError as exception:
        raise ValueError(
            f"Unknown thinking level {str_token!r}; expected one of "
            f"{', '.join(level.value for level in AIThinkingLevel)}."
        ) from exception


class AIThinkingCapabilities(BaseModel):
    """
    Which standard levels a model honors exactly.

    Attributes:
        levels: The levels the engine can apply as-is, in any order. A
            requested level outside this list resolves to the nearest one in
            it. OFF absent means the model's thinking is always on.
        notes: Optional provider wording for the mapping, for documentation
            and diagnostics.
    """

    model_config = ConfigDict(frozen=True)

    levels: list[AIThinkingLevel] = Field(default_factory=list)
    notes: str | None = None

    @property
    def always_on(self) -> bool:
        """True when the model cannot turn thinking off."""
        # Normal return: OFF missing from the honored levels.
        return AIThinkingLevel.OFF not in self.levels

    def nearest(self, requested: AIThinkingLevel) -> AIThinkingLevel | None:
        """
        The level this model honors that is closest to the requested one.

        Ties go to the lower level, since lower thinking is cheaper and
        faster and the caller asked for no more than the requested level.

        Args:
            requested: The level the caller or configuration asked for.

        Returns:
            The nearest honored level, or None when the model honors none.
        """
        if not self.levels:
            # Early return: nothing to apply.
            return None
        if requested in self.levels:
            # Early return: honored exactly.
            return requested
        int_wanted: int = LIST_THINKING_LEVELS.index(requested)
        # Normal return with the closest honored level, lower on a tie.
        return min(
            self.levels,
            key=lambda level: (
                abs(LIST_THINKING_LEVELS.index(level) - int_wanted),
                LIST_THINKING_LEVELS.index(level),
            ),
        )


class AIThinkingResolution(BaseModel):
    """
    What was asked for and what the engine applied on one call.

    Attributes:
        requested: The level from the call, the client, or configuration.
        applied: The level the engine sent, after nearest-level resolution;
            None when the model has no thinking control at all.
        exact: True when `applied` is `requested`.
        reason: Why they differ, when they do: `unsupported_level` (the
            model honors other levels), `no_thinking_control` (the model
            exposes none), or an engine-specific reason such as
            `forced_tool_choice`.
    """

    model_config = ConfigDict(frozen=True)

    requested: AIThinkingLevel
    applied: AIThinkingLevel | None
    exact: bool
    reason: str | None = None


THINKING_REASON_UNSUPPORTED_LEVEL: str = "unsupported_level"
THINKING_REASON_NO_CONTROL: str = "no_thinking_control"


def resolve_thinking_level(
    requested: AIThinkingLevel,
    capabilities: AIThinkingCapabilities | None,
) -> AIThinkingResolution:
    """
    Resolves the requested level against a model's honored levels.

    Args:
        requested: The level asked for.
        capabilities: The model's thinking capabilities, or None when the
            model exposes no thinking control.

    Returns:
        The resolution, with `applied` the nearest honored level.
    """
    if capabilities is None or not capabilities.levels:
        # Early return: the model has no control to apply the level with.
        return AIThinkingResolution(
            requested=requested,
            applied=None,
            exact=False,
            reason=THINKING_REASON_NO_CONTROL,
        )
    applied: AIThinkingLevel | None = capabilities.nearest(requested)
    # Normal return with the exact or nearest level.
    return AIThinkingResolution(
        requested=requested,
        applied=applied,
        exact=applied is requested,
        reason=None if applied is requested else THINKING_REASON_UNSUPPORTED_LEVEL,
    )


# Per-call override set by the public templates (send_conversation,
# send_structured_output, and their async twins) from the reserved
# provider_options key, read by the engine while it builds the request.
THINKING_CALL_OVERRIDE: ContextVar[AIThinkingLevel | None] = ContextVar(
    "ai_api_unified_thinking_call_override", default=None
)


def thinking_event_fields(resolution: AIThinkingResolution) -> dict[str, Any]:
    """Scalar fields describing one resolution, for logs and observability."""
    # Normal return with the two levels and whether they matched.
    return {
        "thinking_requested": resolution.requested.value,
        "thinking_applied": (
            resolution.applied.value if resolution.applied is not None else None
        ),
        "thinking_exact": resolution.exact,
    }


# ── Provider mappings ────────────────────────────────────────────────────────
#
# Each provider section declares which standard levels its models honor and
# how a level becomes request fields. The tables are the one place a model's
# thinking control is described; engines call these and merge the result
# before provider_options, so a caller's native override still wins.

THINKING_REASON_FORCED_TOOL_CHOICE: str = "forced_tool_choice"
THINKING_REASON_MAX_TOKENS_TOO_SMALL: str = "max_tokens_too_small"
# The caller set the provider's own thinking field in provider_options, so
# the layer applied no level of its own and the provider field decides.
THINKING_REASON_PROVIDER_OPTION: str = "provider_option"

# The provider_options path holding a native thinking field on this call
# (for example "reasoning_effort"), or None. Set by the completions templates
# so the engine's thinking resolution can step aside and record it.
THINKING_NATIVE_OPTION: ContextVar[str | None] = ContextVar(
    "ai_api_unified_thinking_native_option", default=None
)

LIST_EFFORT_LEVELS: list[AIThinkingLevel] = [
    AIThinkingLevel.LOW,
    AIThinkingLevel.MEDIUM,
    AIThinkingLevel.HIGH,
    AIThinkingLevel.MAX,
]
LIST_EFFORT_LEVELS_WITH_OFF: list[AIThinkingLevel] = [
    AIThinkingLevel.OFF,
    *LIST_EFFORT_LEVELS,
]

# --- Claude (native Anthropic API and Bedrock Converse) ---
#
# Model classes, by name prefix, from the Anthropic thinking table
# (platform.claude.com, cached 2026-10):
#   always_on: Fable 5 / 5.1, Mythos, Opus 5.5 — thinking cannot be turned
#       off; `output_config.effort` is the only control.
#   adaptive:  Opus 5, Haiku 5.5, Sonnet 5, Opus 4.6-4.8, Sonnet 4.6 —
#       `thinking: {type: adaptive}` plus effort; `{type: disabled}` for off.
#   between_tools: Sonnet 5.5 — `{type: disabled}` is rejected; off is
#       `{type: between_tools}`.
#   budget:    Haiku 4.5, Sonnet 4.5, Opus 4.5, Opus 4.1, Sonnet 4, Opus 4,
#       3.7 Sonnet — `{type: enabled, budget_tokens: N}`, N >= 1024 and
#       below max_tokens with room left for the answer; no effort parameter.
#   none:      Claude 3 Opus/Haiku and 3.5 Sonnet/Haiku have no thinking
#       control; the level is recorded as not applied.
CLAUDE_CLASS_ALWAYS_ON: str = "always_on"
CLAUDE_CLASS_ADAPTIVE: str = "adaptive"
CLAUDE_CLASS_BETWEEN_TOOLS: str = "between_tools"
CLAUDE_CLASS_BUDGET: str = "budget"
CLAUDE_CLASS_NONE: str = "none"

TUPLE_CLAUDE_ALWAYS_ON_MARKERS: tuple[str, ...] = (
    "claude-fable-",
    "claude-mythos-",
    "claude-opus-5-5",
)
TUPLE_CLAUDE_BETWEEN_TOOLS_MARKERS: tuple[str, ...] = ("claude-sonnet-5-5",)
TUPLE_CLAUDE_BUDGET_MARKERS: tuple[str, ...] = (
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
    "claude-opus-4-1",
    "claude-opus-4-2",
    "claude-sonnet-4-2",
    "claude-3-7-sonnet",
)
TUPLE_CLAUDE_NO_CONTROL_MARKERS: tuple[str, ...] = ("claude-3-",)
TUPLE_CLAUDE_BUDGET_EXACT: tuple[str, ...] = (
    "claude-sonnet-4",
    "claude-opus-4",
    "claude-sonnet-4-0",
    "claude-opus-4-0",
)
DICT_CLAUDE_EFFORT: dict[AIThinkingLevel, str] = {
    AIThinkingLevel.LOW: "low",
    AIThinkingLevel.MEDIUM: "medium",
    AIThinkingLevel.HIGH: "high",
    AIThinkingLevel.MAX: "max",
}
DICT_CLAUDE_BUDGET_TOKENS: dict[AIThinkingLevel, int] = {
    AIThinkingLevel.LOW: 1024,
    AIThinkingLevel.MEDIUM: 4096,
    AIThinkingLevel.HIGH: 16_384,
    AIThinkingLevel.MAX: 32_768,
}
INT_CLAUDE_MIN_BUDGET_TOKENS: int = 1024
# Output tokens kept free of the thinking budget so the answer itself fits;
# a budget that would leave less than this turns thinking off instead.
INT_CLAUDE_ANSWER_RESERVE_TOKENS: int = 1024


def claude_thinking_class(model_name: str) -> str:
    """
    Classifies a Claude model id (native or Bedrock) by its thinking control.

    Args:
        model_name: The model id, in any case; a Bedrock id with a region or
            vendor prefix is matched on its substring.

    Returns:
        One of the CLAUDE_CLASS_* labels. Unknown ids are `adaptive`, the
        current family's shape.
    """
    str_name: str = model_name.strip().lower()
    str_core: str = str_name.split("anthropic.")[-1]
    str_bare: str = str_core.split("@")[0]
    if any(marker in str_core for marker in TUPLE_CLAUDE_BUDGET_MARKERS) or any(
        str_bare == exact or str_bare.startswith(exact + "-2")
        for exact in TUPLE_CLAUDE_BUDGET_EXACT
    ):
        # Early return: the pre-4.6 budget_tokens control (3.7 Sonnet included).
        return CLAUDE_CLASS_BUDGET
    if any(marker in str_core for marker in TUPLE_CLAUDE_NO_CONTROL_MARKERS):
        # Early return: the rest of Claude 3.x has no thinking control.
        return CLAUDE_CLASS_NONE
    if any(marker in str_core for marker in TUPLE_CLAUDE_ALWAYS_ON_MARKERS):
        # Early return: thinking cannot be turned off.
        return CLAUDE_CLASS_ALWAYS_ON
    if any(marker in str_core for marker in TUPLE_CLAUDE_BETWEEN_TOOLS_MARKERS):
        # Early return: off is between_tools.
        return CLAUDE_CLASS_BETWEEN_TOOLS
    # Normal return: adaptive thinking with effort.
    return CLAUDE_CLASS_ADAPTIVE


def claude_thinking_capabilities(model_name: str) -> AIThinkingCapabilities | None:
    """The standard levels a Claude model honors, or None without a control."""
    str_class: str = claude_thinking_class(model_name)
    if str_class == CLAUDE_CLASS_NONE:
        # Early return: nothing to apply a level with.
        return None
    if str_class == CLAUDE_CLASS_ALWAYS_ON:
        # Early return: no off; effort low through max.
        return AIThinkingCapabilities(
            levels=list(LIST_EFFORT_LEVELS),
            notes="always-on thinking; output_config.effort low|medium|high|max",
        )
    if str_class == CLAUDE_CLASS_BUDGET:
        # Early return: budget_tokens tiers.
        return AIThinkingCapabilities(
            levels=list(LIST_EFFORT_LEVELS_WITH_OFF),
            notes="thinking.budget_tokens 1024|4096|16384|32768; off omits thinking",
        )
    # Normal return: adaptive with off (disabled or between_tools).
    return AIThinkingCapabilities(
        levels=list(LIST_EFFORT_LEVELS_WITH_OFF),
        notes=(
            "thinking adaptive + output_config.effort low|medium|high|max; off is "
            + (
                "thinking.type between_tools"
                if str_class == CLAUDE_CLASS_BETWEEN_TOOLS
                else "thinking.type disabled"
            )
        ),
    )


def claude_thinking_request_fields(
    model_name: str,
    level: AIThinkingLevel,
    *,
    max_tokens: int,
) -> tuple[dict[str, Any], str | None]:
    """
    Builds the Messages API fields that apply one level on a Claude model.

    Args:
        model_name: The model id.
        level: An honored level (see claude_thinking_capabilities).
        max_tokens: The request's max_tokens; a budget model's
            budget_tokens is capped at max_tokens less
            INT_CLAUDE_ANSWER_RESERVE_TOKENS so the answer still fits.

    Returns:
        (fields to merge into the request: `thinking` and/or
        `output_config`, as plain dicts; the reason the level could not be
        applied, or None). On a budget model whose max_tokens leaves less
        than the minimum budget after the answer reserve, thinking is left
        off and the reason is `max_tokens_too_small`.
    """
    str_class: str = claude_thinking_class(model_name)
    if str_class == CLAUDE_CLASS_NONE:
        # Early return: no control on this model.
        return {}, THINKING_REASON_NO_CONTROL
    if str_class == CLAUDE_CLASS_BUDGET:
        if level is AIThinkingLevel.OFF:
            # Early return: omitting thinking is off on these models.
            return {}, None
        int_budget: int = min(
            DICT_CLAUDE_BUDGET_TOKENS[level],
            max_tokens - INT_CLAUDE_ANSWER_RESERVE_TOKENS,
        )
        if int_budget < INT_CLAUDE_MIN_BUDGET_TOKENS:
            # Early return: the request cannot fit the minimum budget.
            return {}, THINKING_REASON_MAX_TOKENS_TOO_SMALL
        # Normal return with the enabled budget.
        return {"thinking": {"type": "enabled", "budget_tokens": int_budget}}, None
    if level is AIThinkingLevel.OFF:
        if str_class == CLAUDE_CLASS_BETWEEN_TOOLS:
            # Early return: Sonnet 5.5's off.
            return {"thinking": {"type": "between_tools"}}, None
        # Early return: adaptive-era off.
        return {"thinking": {"type": "disabled"}}, None
    dict_fields: dict[str, Any] = {
        "output_config": {"effort": DICT_CLAUDE_EFFORT[level]},
    }
    if str_class != CLAUDE_CLASS_ALWAYS_ON:
        dict_fields["thinking"] = {"type": "adaptive"}
    # Normal return with adaptive thinking at the effort.
    return dict_fields, None


# --- OpenAI (Chat Completions and Responses) ---
#
# reasoning_effort values from the OpenAI reasoning guide (cached 2026-10):
# none, minimal, low, medium, high, xhigh, max; the set each model accepts
# is model-dependent. `none` is accepted on GPT-5.1 and later except
# GPT-6 Astra and GPT-6.1 Sol; `minimal` is the GPT-5 family's floor; the
# o-series takes low|medium|high.
DICT_OPENAI_EFFORT: dict[AIThinkingLevel, str] = {
    AIThinkingLevel.OFF: "none",
    AIThinkingLevel.MINIMAL: "minimal",
    AIThinkingLevel.LOW: "low",
    AIThinkingLevel.MEDIUM: "medium",
    AIThinkingLevel.HIGH: "high",
    AIThinkingLevel.MAX: "xhigh",
}
TUPLE_OPENAI_NO_NONE_PREFIXES: tuple[str, ...] = ("gpt-6-astra", "gpt-6.1-sol")
# The GPT-5 launch family (gpt-5, -mini, -nano, -pro, -codex, dated snapshots):
# ids that are exactly gpt-5 or start with gpt-5- (gpt-5.1 and later are gpt-5.).
TUPLE_OPENAI_GPT5_BASE_PREFIXES: tuple[str, ...] = ("gpt-5-",)


def openai_thinking_capabilities(model_name: str) -> AIThinkingCapabilities | None:
    """The standard levels an OpenAI model honors, or None without a control."""
    str_name: str = model_name.strip().lower()
    if str_name.startswith(("gpt-6", "gpt-5.")):
        list_levels: list[AIThinkingLevel] = [
            AIThinkingLevel.LOW,
            AIThinkingLevel.MEDIUM,
            AIThinkingLevel.HIGH,
            AIThinkingLevel.MAX,
        ]
        if not str_name.startswith(TUPLE_OPENAI_NO_NONE_PREFIXES):
            list_levels.insert(0, AIThinkingLevel.OFF)
        # Early return: GPT-5.1 and later; MAX is xhigh.
        return AIThinkingCapabilities(
            levels=list_levels,
            notes="reasoning_effort none|low|medium|high|xhigh",
        )
    if str_name == "gpt-5" or str_name.startswith(TUPLE_OPENAI_GPT5_BASE_PREFIXES):
        # Early return: the GPT-5 launch models and their snapshots.
        return AIThinkingCapabilities(
            levels=[
                AIThinkingLevel.MINIMAL,
                AIThinkingLevel.LOW,
                AIThinkingLevel.MEDIUM,
                AIThinkingLevel.HIGH,
            ],
            notes="reasoning_effort minimal|low|medium|high",
        )
    if str_name.startswith(("o1", "o3", "o4")):
        # Early return: the o-series.
        return AIThinkingCapabilities(
            levels=[AIThinkingLevel.LOW, AIThinkingLevel.MEDIUM, AIThinkingLevel.HIGH],
            notes="reasoning_effort low|medium|high",
        )
    # Normal return: no reasoning control (gpt-4.1, gpt-4o, and unknown ids).
    return None


# --- Gemini ---
#
# Gemini 3.x takes `thinking_config.thinking_level` (minimal|low|medium|high
# on 3 Flash and later; 3 Pro accepts low|high) and cannot turn thinking
# off. Gemini 2.5 takes `thinking_config.thinking_budget`; Flash and Flash
# Lite accept 0 (off), Pro's floor is 128. Gemini 2.0 and 1.5 have no
# thinking control.
DICT_GEMINI_LEVEL: dict[AIThinkingLevel, str] = {
    AIThinkingLevel.MINIMAL: "MINIMAL",
    AIThinkingLevel.LOW: "LOW",
    AIThinkingLevel.MEDIUM: "MEDIUM",
    AIThinkingLevel.HIGH: "HIGH",
}
DICT_GEMINI_25_BUDGET: dict[AIThinkingLevel, int] = {
    AIThinkingLevel.OFF: 0,
    AIThinkingLevel.LOW: 1024,
    AIThinkingLevel.MEDIUM: 8192,
    AIThinkingLevel.HIGH: 16_384,
    AIThinkingLevel.MAX: 24_576,
}
INT_GEMINI_25_PRO_MAX_BUDGET: int = 32_768


def gemini_thinking_capabilities(model_name: str) -> AIThinkingCapabilities | None:
    """The standard levels a Gemini model honors, or None without a control."""
    str_name: str = model_name.strip().lower()
    if str_name.startswith("gemini-3"):
        if "pro" in str_name:
            # Early return: 3 Pro's two levels.
            return AIThinkingCapabilities(
                levels=[AIThinkingLevel.LOW, AIThinkingLevel.HIGH],
                notes="thinking_config.thinking_level LOW|HIGH",
            )
        # Early return: 3 Flash and later.
        return AIThinkingCapabilities(
            levels=list(DICT_GEMINI_LEVEL),
            notes="thinking_config.thinking_level MINIMAL|LOW|MEDIUM|HIGH",
        )
    if "2.5" in str_name:
        list_levels: list[AIThinkingLevel] = list(DICT_GEMINI_25_BUDGET)
        if "pro" in str_name:
            list_levels.remove(AIThinkingLevel.OFF)
        # Early return: 2.5 budgets.
        return AIThinkingCapabilities(
            levels=list_levels,
            notes="thinking_config.thinking_budget 0|1024|8192|16384|24576 (Pro: no 0, max 32768)",
        )
    # Normal return: no thinking control.
    return None


def gemini_thinking_config(model_name: str, level: AIThinkingLevel) -> dict[str, Any]:
    """The `thinking_config` dict that applies one honored level on a Gemini model."""
    str_name: str = model_name.strip().lower()
    if str_name.startswith("gemini-3"):
        # Normal return with the level name.
        return {"thinking_level": DICT_GEMINI_LEVEL[level]}
    int_budget: int = DICT_GEMINI_25_BUDGET[level]
    if level is AIThinkingLevel.MAX and "pro" in str_name:
        int_budget = INT_GEMINI_25_PRO_MAX_BUDGET
    # Normal return with the budget.
    return {"thinking_budget": int_budget}
