# ai_structured_output_classifier.py
"""
Classification through any completions engine that supports structured output.

The engine answers a one-field JSON schema whose only field is an enum of
the allowed answers (the option values, "yes"/"no", or the level labels),
so every engine the registry lists for structured output can classify. The
delegate completions engine runs the call inside its own observability
lifecycle, so this adapter sets BOOL_PROVIDER_OBSERVED and adds a
`classifier_role` tag to that call instead of emitting a second cost event.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, ClassVar

from ..ai_base import (
    AIBaseClassifier,
    AIBaseCompletions,
    AIClassificationChoice,
    AIClassificationLevel,
    AIClassificationResult,
    AIClassifierCapabilitiesBase,
    AIFinishReason,
    AIStructuredOutputResult,
)
from ..ai_completions_exceptions import StructuredResponseTokenLimitError
from ..thinking import (
    CLASSIFIER_THINKING_KEY,
    DEFAULT_CLASSIFIER_THINKING_LEVEL,
    AIThinkingLevel,
    parse_thinking_level,
)
from ..ai_provider_exceptions import (
    AiProviderCapabilityUnsupportedError,
    AiProviderConfigurationError,
)
from ..middleware.observability_runtime import (
    TOKEN_COUNT_SOURCE_NONE,
    TOKEN_COUNT_SOURCE_PROVIDER,
    ObservabilityContextModel,
    get_observability_context,
    reset_observability_context,
    set_observability_context,
)

# Tag added to the delegate call's observability context so a dashboard can
# split classifier spend from the engine's other completions.
OBSERVABILITY_TAG_CLASSIFIER_ROLE: str = "classifier_role"

# The library's shared structured-output budget. The answer is one short JSON
# object, but reasoning and always-on-thinking models spend thinking tokens
# from the same budget, so a label-sized cap truncates them before the JSON.
DEFAULT_CLASSIFIER_MAX_RESPONSE_TOKENS: int = (
    AIBaseCompletions.STRUCTURED_DEFAULT_MAX_RESPONSE_TOKENS
)

STR_PREDICATE_YES: str = "yes"
STR_PREDICATE_NO: str = "no"

STR_BASE_INSTRUCTIONS: str = (
    "You are a classifier. Read the input and answer the question by "
    "returning only the JSON object the schema allows. Do not explain."
)


class AiStructuredOutputClassifier(AIBaseClassifier):
    """
    Classifies through a completions engine's structured output.

    Args:
        model: Model for the completions engine; blank means that engine's
            COMPLETIONS_MODEL_NAME, then its default.
        engine: Completions engine token the registry maps this classifier
            to (openai, claude, google-gemini, ...). Required unless a
            completions_client is given.
        base_url: Optional API base-URL override for engines that accept one.
        completions_client: An already-built completions engine, for tests
            or for callers who hold one.

    Raises:
        AiProviderConfigurationError: Neither engine nor client was given.
        AiProviderCapabilityUnsupportedError: The engine's model lacks
            structured output.
    """

    BOOL_PROVIDER_OBSERVED: ClassVar[bool] = True

    def __init__(
        self,
        model: str | None = None,
        *,
        engine: str | None = None,
        base_url: str | None = None,
        completions_client: AIBaseCompletions | None = None,
        thinking: AIThinkingLevel | str | None = None,
    ) -> None:
        super().__init__(model=model)
        level_explicit: AIThinkingLevel | None = parse_thinking_level(thinking)
        bool_built_here: bool = completions_client is None
        if completions_client is None:
            if not engine:
                raise AiProviderConfigurationError(
                    "AiStructuredOutputClassifier needs a completions engine token "
                    "or a completions_client."
                )
            # Local import: the factory imports ai_base, and this module is
            # lazy-loaded by the factory, so a top-level import would cycle.
            from ..ai_factory import AIFactory

            completions_client = AIFactory._build_completions_client(
                engine, model or "", base_url
            )
        if not completions_client.capabilities.supports_structured_output:
            raise AiProviderCapabilityUnsupportedError(
                f"{type(completions_client).__name__} model "
                f"'{completions_client.model_name}' does not support structured "
                "output, so it cannot classify. Pick a model whose capabilities "
                "include supports_structured_output."
            )
        if level_explicit is not None or bool_built_here:
            # A client built here classifies at the classifier's level (default
            # off); a caller's client keeps its own unless one was passed.
            completions_client.thinking_level = (
                level_explicit or self._configured_thinking_level()
            )
        self.str_engine: str = engine or ""
        self._completions: AIBaseCompletions = completions_client
        self.model = completions_client.model_name

    @staticmethod
    def _configured_thinking_level() -> AIThinkingLevel:
        """`classifier.thinking` in models.json, else off."""
        from ..util.env_settings import EnvSettings

        # Normal return with the configured level, or the classifier default.
        return (
            parse_thinking_level(
                EnvSettings().get_setting(CLASSIFIER_THINKING_KEY, None)
            )
            or DEFAULT_CLASSIFIER_THINKING_LEVEL
        )

    @property
    def completions_client(self) -> AIBaseCompletions:
        """The delegate completions engine."""
        # Normal return with the delegate.
        return self._completions

    @property
    def list_model_names(self) -> list[str]:
        """Models the delegate engine lists."""
        # Normal return with the delegate's catalog.
        return self._completions.list_model_names

    @property
    def capabilities(self) -> AIClassifierCapabilitiesBase:
        """Every question type; async and pricing follow the delegate."""
        # Normal return with the descriptor derived from the delegate.
        return AIClassifierCapabilitiesBase(
            supports_choice=True,
            supports_predicate=True,
            supports_score=True,
            supports_images=False,
            supports_async=self._completions.capabilities.supports_async,
            max_choices=None,
            pricing=self._completions.capabilities.pricing,
        )

    def _resolve_observability_provider_engine(self) -> str:
        """The delegate's engine token, so events name the real provider."""
        # Normal return with the configured token or the delegate's label.
        return (
            self.str_engine
            or self._completions._resolve_observability_provider_engine()
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
        list_values: list[str] = [choice.value for choice in list_choices]
        str_system: str = self._system_prompt(
            instructions,
            "Pick exactly one option.",
            [(choice.value, choice.description) for choice in list_choices],
        )
        with self._tagged("choice"):
            result: AIStructuredOutputResult = self._completions.send_structured_output(
                str_text,
                response_schema=self._schema("answer", list_values),
                system_prompt=str_system,
                max_response_tokens=DEFAULT_CLASSIFIER_MAX_RESPONSE_TOKENS,
                request_timeout_seconds=request_timeout_seconds,
            )
        # Normal return with the mapped answer.
        return self._map_choice(result, "choice", list_values)

    def _classify_predicate_provider(
        self,
        str_text: str,
        *,
        instructions: str,
        request_timeout_seconds: float | None,
    ) -> AIClassificationResult:
        str_system: str = self._system_prompt(
            instructions,
            f'Answer "{STR_PREDICATE_YES}" when the condition holds, else '
            f'"{STR_PREDICATE_NO}".',
            [],
        )
        with self._tagged("predicate"):
            result: AIStructuredOutputResult = self._completions.send_structured_output(
                str_text,
                response_schema=self._schema(
                    "answer", [STR_PREDICATE_YES, STR_PREDICATE_NO]
                ),
                system_prompt=str_system,
                max_response_tokens=DEFAULT_CLASSIFIER_MAX_RESPONSE_TOKENS,
                request_timeout_seconds=request_timeout_seconds,
            )
        mapped: AIClassificationResult = self._map_choice(
            result, "predicate", [STR_PREDICATE_YES, STR_PREDICATE_NO]
        )
        if mapped.choice is not None:
            # A structured answer is a hard yes or no, so the probability is
            # 1.0 or 0.0; a classifier with real probabilities reports them.
            mapped.probability = 1.0 if mapped.choice == STR_PREDICATE_YES else 0.0
            mapped.choice = None
        # Normal return with the mapped answer.
        return mapped

    def _classify_score_provider(
        self,
        str_text: str,
        list_levels: list[AIClassificationLevel],
        *,
        instructions: str | None,
        request_timeout_seconds: float | None,
    ) -> AIClassificationResult:
        list_labels: list[str] = [level.label for level in list_levels]
        str_system: str = self._system_prompt(
            instructions,
            "Pick exactly one level. The levels are listed lowest first.",
            [(level.label, level.description) for level in list_levels],
        )
        with self._tagged("score"):
            result: AIStructuredOutputResult = self._completions.send_structured_output(
                str_text,
                response_schema=self._schema("answer", list_labels),
                system_prompt=str_system,
                max_response_tokens=DEFAULT_CLASSIFIER_MAX_RESPONSE_TOKENS,
                request_timeout_seconds=request_timeout_seconds,
            )
        mapped: AIClassificationResult = self._map_choice(result, "score", list_labels)
        if mapped.choice is not None:
            # Position on the scale, normalized so two levels map to 0 and 1.
            mapped.score = list_labels.index(mapped.choice) / (len(list_labels) - 1)
        # Normal return with the mapped answer.
        return mapped

    async def _aclassify_choice_provider(
        self,
        str_text: str,
        list_choices: list[AIClassificationChoice],
        *,
        instructions: str | None,
        request_timeout_seconds: float | None,
    ) -> AIClassificationResult:
        list_values: list[str] = [choice.value for choice in list_choices]
        str_system: str = self._system_prompt(
            instructions,
            "Pick exactly one option.",
            [(choice.value, choice.description) for choice in list_choices],
        )
        with self._tagged("choice"):
            result: AIStructuredOutputResult = (
                await self._completions.asend_structured_output(
                    str_text,
                    response_schema=self._schema("answer", list_values),
                    system_prompt=str_system,
                    max_response_tokens=DEFAULT_CLASSIFIER_MAX_RESPONSE_TOKENS,
                    request_timeout_seconds=request_timeout_seconds,
                )
            )
        # Normal return with the mapped answer.
        return self._map_choice(result, "choice", list_values)

    async def _aclassify_predicate_provider(
        self,
        str_text: str,
        *,
        instructions: str,
        request_timeout_seconds: float | None,
    ) -> AIClassificationResult:
        str_system: str = self._system_prompt(
            instructions,
            f'Answer "{STR_PREDICATE_YES}" when the condition holds, else '
            f'"{STR_PREDICATE_NO}".',
            [],
        )
        with self._tagged("predicate"):
            result: AIStructuredOutputResult = (
                await self._completions.asend_structured_output(
                    str_text,
                    response_schema=self._schema(
                        "answer", [STR_PREDICATE_YES, STR_PREDICATE_NO]
                    ),
                    system_prompt=str_system,
                    max_response_tokens=DEFAULT_CLASSIFIER_MAX_RESPONSE_TOKENS,
                    request_timeout_seconds=request_timeout_seconds,
                )
            )
        mapped: AIClassificationResult = self._map_choice(
            result, "predicate", [STR_PREDICATE_YES, STR_PREDICATE_NO]
        )
        if mapped.choice is not None:
            mapped.probability = 1.0 if mapped.choice == STR_PREDICATE_YES else 0.0
            mapped.choice = None
        # Normal return with the mapped answer.
        return mapped

    async def _aclassify_score_provider(
        self,
        str_text: str,
        list_levels: list[AIClassificationLevel],
        *,
        instructions: str | None,
        request_timeout_seconds: float | None,
    ) -> AIClassificationResult:
        list_labels: list[str] = [level.label for level in list_levels]
        str_system: str = self._system_prompt(
            instructions,
            "Pick exactly one level. The levels are listed lowest first.",
            [(level.label, level.description) for level in list_levels],
        )
        with self._tagged("score"):
            result: AIStructuredOutputResult = (
                await self._completions.asend_structured_output(
                    str_text,
                    response_schema=self._schema("answer", list_labels),
                    system_prompt=str_system,
                    max_response_tokens=DEFAULT_CLASSIFIER_MAX_RESPONSE_TOKENS,
                    request_timeout_seconds=request_timeout_seconds,
                )
            )
        mapped: AIClassificationResult = self._map_choice(result, "score", list_labels)
        if mapped.choice is not None:
            mapped.score = list_labels.index(mapped.choice) / (len(list_labels) - 1)
        # Normal return with the mapped answer.
        return mapped

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _schema(str_field: str, list_allowed: list[str]) -> dict[str, Any]:
        """One required enum field; the engine can answer nothing else."""
        # Normal return with the schema.
        return {
            "type": "object",
            "properties": {str_field: {"type": "string", "enum": list_allowed}},
            "required": [str_field],
            "additionalProperties": False,
        }

    @staticmethod
    def _system_prompt(
        instructions: str | None,
        str_task: str,
        list_options: list[tuple[str, str]],
    ) -> str:
        """Base instructions, the caller's guidance, then the options."""
        list_lines: list[str] = [STR_BASE_INSTRUCTIONS, str_task]
        if instructions:
            list_lines.append(instructions.strip())
        if list_options:
            list_lines.append("Options:")
            # Loop over options so each carries its description.
            for str_value, str_description in list_options:
                list_lines.append(
                    f"- {str_value}: {str_description}"
                    if str_description
                    else f"- {str_value}"
                )
        # Normal return with the joined prompt.
        return "\n".join(list_lines)

    def _map_choice(
        self,
        result: AIStructuredOutputResult,
        str_question_type: str,
        list_allowed: list[str],
    ) -> AIClassificationResult:
        """
        Reads the enum answer and the usage off a structured result.

        Raises:
            StructuredResponseTokenLimitError: The delegate truncated the
                answer, so retrying with a larger budget is the fix, not a
                refusal.
        """
        if result.finish_reason == AIFinishReason.LENGTH:
            str_model: str = result.model_name or self.model_name or ""
            raise StructuredResponseTokenLimitError(
                message=(
                    f"Classification on '{str_model}' was truncated before the "
                    f"answer at {DEFAULT_CLASSIFIER_MAX_RESPONSE_TOKENS} "
                    "response tokens."
                ),
                provider_name=(
                    result.provider_engine
                    or self._resolve_observability_provider_engine()
                ),
                model_name=str_model,
                max_response_tokens=DEFAULT_CLASSIFIER_MAX_RESPONSE_TOKENS,
                minimum_supported_tokens=DEFAULT_CLASSIFIER_MAX_RESPONSE_TOKENS,
                finish_reason=str(result.finish_reason),
                raw_output_char_count=len(result.raw_text),
            )
        str_answer: str | None = None
        if isinstance(result.data, dict):
            raw_answer: Any = result.data.get("answer")
            if isinstance(raw_answer, str) and raw_answer in list_allowed:
                str_answer = raw_answer
        bool_refused: bool = result.finish_reason == AIFinishReason.REFUSAL or (
            str_answer is None
        )
        str_source: str = (
            TOKEN_COUNT_SOURCE_PROVIDER
            if result.usage.input_tokens is not None
            else TOKEN_COUNT_SOURCE_NONE
        )
        # Normal return with the mapped result; the template stamps the rest.
        return AIClassificationResult(
            question_type=str_question_type,  # type: ignore[arg-type]
            choice=str_answer,
            refused=bool_refused,
            raw_text=result.raw_text,
            usage=result.usage,
            token_source=str_source,
            provider_engine=result.provider_engine,
            model_name=result.model_name,
        )

    @contextmanager
    def _tagged(self, str_role: str) -> Iterator[None]:
        """Adds the classifier_role tag to the delegate call's context."""
        context: ObservabilityContextModel = get_observability_context()
        token = set_observability_context(
            caller_id=context.caller_id,
            session_id=context.session_id,
            workflow_id=context.workflow_id,
            tags={**dict(context.tags), OBSERVABILITY_TAG_CLASSIFIER_ROLE: str_role},
        )
        try:
            yield
        finally:
            reset_observability_context(token)
