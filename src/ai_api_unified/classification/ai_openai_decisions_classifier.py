# ai_openai_decisions_classifier.py
"""
Classification through OpenAI's Decisions API.

The Decisions API (POST /v1/decisions) answers a fixed set of questions about
one input: pick one of the given choices, say whether a predicate holds, or
rate the input on an ordered scale. It returns probabilities with the answer,
runs about ten times faster than a chat completion, and bills input tokens
only. The call goes through the OpenAI SDK's generic `post`, so the library's
pinned 2.x SDK (which predates `client.decisions`) serves it.

Metering: the response is expected to carry `usage`; when it does not, the
engine estimates input tokens from the request text and reports
`token_source="estimated"` so the cost event is never silently zero. Cost
events price the call under the registry key "decisions/<model>", whose
rates differ from the model's chat rates.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import BaseModel, ConfigDict

from ..ai_base import (
    AIBaseClassifier,
    AIClassificationChoice,
    AIClassificationLevel,
    AIClassificationResult,
    AIClassifierCapabilitiesBase,
    AITokenUsage,
)
from ..ai_openai_base import AIOpenAIBase
from ..ai_provider_exceptions import (
    AiProviderRequestError,
    classify_transport_fallback_reason,
)
from ..middleware.observability_runtime import (
    TOKEN_COUNT_SOURCE_ESTIMATED,
    TOKEN_COUNT_SOURCE_PROVIDER,
)
from ..pricing.pricing_registry import (
    PROVIDER_OPENAI,
    enforce_model_lifecycle,
    get_model_pricing,
)

DECISIONS_API_PATH: str = "/decisions"
DECISIONS_DEFAULT_MODEL: str = "gpt-6-luna"
# Models the Decisions API serves. Public beta: one model so far.
FROZENSET_DECISIONS_MODELS: frozenset[str] = frozenset({DECISIONS_DEFAULT_MODEL})
# Registry key prefix: Decisions bills input only, at its own rate, so it is
# priced apart from the same model's chat rates.
DECISIONS_PRICING_MODEL_PREFIX: str = "decisions/"
# Rough tokens-per-character ratio for English text, used only when the
# response carries no usage block.
DECISIONS_CHARS_PER_TOKEN_ESTIMATE: int = 4
STR_ANSWER_NAME: str = "answer"
STR_ANSWER_TYPE_REFUSAL: str = "refusal"

STR_DEFAULT_CHOICE_INSTRUCTIONS: str = "Which option best describes the input?"
STR_DEFAULT_SCORE_INSTRUCTIONS: str = "Rate the input on the scale, lowest level first."


class _DecisionsAnswer(BaseModel):
    """One answer in a Decisions response; tolerant of fields we do not read."""

    model_config = ConfigDict(extra="allow")

    type: str
    name: str | None = None
    choice: str | None = None
    label: str | None = None
    probability: float | None = None
    score: float | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | list[Any] | None = None


class _DecisionsResponse(BaseModel):
    """A Decisions response; tolerant of fields we do not read."""

    model_config = ConfigDict(extra="allow")

    answers: list[_DecisionsAnswer]
    usage: dict[str, Any] | None = None


def decisions_pricing_key(str_model: str) -> str:
    """The pricing registry key for a Decisions model."""
    # Normal return with the prefixed key.
    return f"{DECISIONS_PRICING_MODEL_PREFIX}{str_model}"


# AIOpenAIBase comes first so its org-identity hooks override AIBase's
# no-op defaults, and so AIBase.__init__ does not chain into it with no
# arguments and build a second SDK client.
class AiOpenAIDecisionsClassifier(AIOpenAIBase, AIBaseClassifier):
    """
    Classifies through the OpenAI Decisions API.

    Args:
        model: Decisions model; blank means gpt-6-luna.
        engine: Accepted for factory uniformity; ignored.
        base_url: Optional API base-URL override.
        retry_policy: Optional retry policy override for the SDK client.
    """

    PROVIDER_ENGINE_TOKEN: ClassVar[str] = "openai-decisions"
    PROVIDER_DISPLAY_NAME: ClassVar[str] = "OpenAI Decisions"

    def __init__(
        self,
        model: str | None = None,
        *,
        engine: str | None = None,
        base_url: str | None = None,
        retry_policy: str | None = None,
    ) -> None:
        AIBaseClassifier.__init__(self, model=model or DECISIONS_DEFAULT_MODEL)
        AIOpenAIBase.__init__(self, retry_policy=retry_policy, base_url=base_url)
        enforce_model_lifecycle(PROVIDER_OPENAI, self.pricing_model_key)

    @property
    def pricing_model_key(self) -> str:
        """Registry key the call is priced under."""
        # Normal return with the Decisions pricing key.
        return decisions_pricing_key(str(self.model))

    @property
    def list_model_names(self) -> list[str]:
        """Models the Decisions API serves."""
        # Normal return with the sorted catalog.
        return sorted(FROZENSET_DECISIONS_MODELS)

    @property
    def capabilities(self) -> AIClassifierCapabilitiesBase:
        """Every question type, async, priced from the Decisions entry."""
        # Normal return with the Decisions descriptor.
        return AIClassifierCapabilitiesBase(
            supports_choice=True,
            supports_predicate=True,
            supports_score=True,
            supports_images=False,
            supports_async=True,
            max_choices=None,
            pricing=get_model_pricing(PROVIDER_OPENAI, self.pricing_model_key),
        )

    def _resolve_observability_provider_engine(self) -> str:
        # Normal return with the engine token.
        return self.PROVIDER_ENGINE_TOKEN

    def _resolve_observability_model_name(self) -> str | None:
        # Cost events look pricing up by model name, so report the key
        # whose rates apply to this call.
        return self.pricing_model_key

    # ── Provider hooks ──────────────────────────────────────────────────────

    def _classify_choice_provider(
        self,
        str_text: str,
        list_choices: list[AIClassificationChoice],
        *,
        instructions: str | None,
        request_timeout_seconds: float | None,
    ) -> AIClassificationResult:
        dict_question: dict[str, Any] = self._choice_question(
            list_choices, instructions
        )
        response: _DecisionsResponse = self._post(
            str_text, dict_question, request_timeout_seconds
        )
        # Normal return with the mapped answer.
        return self._map_choice_answer(response, str_text, dict_question)

    def _classify_predicate_provider(
        self,
        str_text: str,
        *,
        instructions: str,
        request_timeout_seconds: float | None,
    ) -> AIClassificationResult:
        dict_question: dict[str, Any] = self._predicate_question(instructions)
        response: _DecisionsResponse = self._post(
            str_text, dict_question, request_timeout_seconds
        )
        # Normal return with the mapped answer.
        return self._map_predicate_answer(response, str_text, dict_question)

    def _classify_score_provider(
        self,
        str_text: str,
        list_levels: list[AIClassificationLevel],
        *,
        instructions: str | None,
        request_timeout_seconds: float | None,
    ) -> AIClassificationResult:
        dict_question: dict[str, Any] = self._score_question(list_levels, instructions)
        response: _DecisionsResponse = self._post(
            str_text, dict_question, request_timeout_seconds
        )
        # Normal return with the mapped answer.
        return self._map_score_answer(response, str_text, dict_question, list_levels)

    async def _aclassify_choice_provider(
        self,
        str_text: str,
        list_choices: list[AIClassificationChoice],
        *,
        instructions: str | None,
        request_timeout_seconds: float | None,
    ) -> AIClassificationResult:
        dict_question: dict[str, Any] = self._choice_question(
            list_choices, instructions
        )
        response: _DecisionsResponse = await self._apost(
            str_text, dict_question, request_timeout_seconds
        )
        # Normal return with the mapped answer.
        return self._map_choice_answer(response, str_text, dict_question)

    async def _aclassify_predicate_provider(
        self,
        str_text: str,
        *,
        instructions: str,
        request_timeout_seconds: float | None,
    ) -> AIClassificationResult:
        dict_question: dict[str, Any] = self._predicate_question(instructions)
        response: _DecisionsResponse = await self._apost(
            str_text, dict_question, request_timeout_seconds
        )
        # Normal return with the mapped answer.
        return self._map_predicate_answer(response, str_text, dict_question)

    async def _aclassify_score_provider(
        self,
        str_text: str,
        list_levels: list[AIClassificationLevel],
        *,
        instructions: str | None,
        request_timeout_seconds: float | None,
    ) -> AIClassificationResult:
        dict_question: dict[str, Any] = self._score_question(list_levels, instructions)
        response: _DecisionsResponse = await self._apost(
            str_text, dict_question, request_timeout_seconds
        )
        # Normal return with the mapped answer.
        return self._map_score_answer(response, str_text, dict_question, list_levels)

    # ── Request shaping ─────────────────────────────────────────────────────

    @staticmethod
    def _choice_question(
        list_choices: list[AIClassificationChoice], instructions: str | None
    ) -> dict[str, Any]:
        # Normal return with the Decisions choice question.
        return {
            "type": "choice",
            "name": STR_ANSWER_NAME,
            "instructions": instructions or STR_DEFAULT_CHOICE_INSTRUCTIONS,
            "choices": [
                {"value": choice.value, "description": choice.description}
                for choice in list_choices
            ],
        }

    @staticmethod
    def _predicate_question(instructions: str) -> dict[str, Any]:
        # Normal return with the Decisions predicate question.
        return {
            "type": "predicate",
            "name": STR_ANSWER_NAME,
            "instructions": instructions,
        }

    @staticmethod
    def _score_question(
        list_levels: list[AIClassificationLevel], instructions: str | None
    ) -> dict[str, Any]:
        # Normal return with the Decisions score question.
        return {
            "type": "score",
            "name": STR_ANSWER_NAME,
            "instructions": instructions or STR_DEFAULT_SCORE_INSTRUCTIONS,
            "levels": [
                {"label": level.label, "description": level.description}
                for level in list_levels
            ],
        }

    def _body(self, str_text: str, dict_question: dict[str, Any]) -> dict[str, Any]:
        # Normal return with the request body.
        return {"model": self.model, "input": str_text, "questions": [dict_question]}

    def _post(
        self,
        str_text: str,
        dict_question: dict[str, Any],
        request_timeout_seconds: float | None,
    ) -> _DecisionsResponse:
        """Posts one question and parses the response."""
        dict_options: dict[str, Any] = (
            {"timeout": request_timeout_seconds}
            if request_timeout_seconds is not None
            else {}
        )
        try:
            raw: httpx.Response = self.client.post(
                DECISIONS_API_PATH,
                cast_to=httpx.Response,
                body=self._body(str_text, dict_question),
                options=dict_options,
            )
        except Exception as exception:
            self._raise_request_error(exception)
            raise
        # Normal return with the parsed response.
        return _DecisionsResponse.model_validate(raw.json())

    async def _apost(
        self,
        str_text: str,
        dict_question: dict[str, Any],
        request_timeout_seconds: float | None,
    ) -> _DecisionsResponse:
        """Async twin of _post."""
        dict_options: dict[str, Any] = (
            {"timeout": request_timeout_seconds}
            if request_timeout_seconds is not None
            else {}
        )
        try:
            raw: httpx.Response = await self.async_client.post(
                DECISIONS_API_PATH,
                cast_to=httpx.Response,
                body=self._body(str_text, dict_question),
                options=dict_options,
            )
        except Exception as exception:
            self._raise_request_error(exception)
            raise
        # Normal return with the parsed response.
        return _DecisionsResponse.model_validate(raw.json())

    def _raise_request_error(self, exception: Exception) -> None:
        """
        Re-raises one OpenAI SDK transport error as the typed request error,
        classified the way the OpenAI completions engine classifies its own.
        """
        if isinstance(exception, APIStatusError):
            # Local import: the completions module is heavier than this one
            # and only its classifier is needed.
            from ..completions.ai_openai_completions import AiOpenAICompletions

            raise AiProviderRequestError(
                f"{self.PROVIDER_DISPLAY_NAME} request failed with status "
                f"{exception.status_code}: {exception.message}",
                status_code=exception.status_code,
                provider_engine=self.PROVIDER_ENGINE_TOKEN,
                fallback_reason=AiOpenAICompletions._classify_fallback_reason(
                    exception
                ),
            ) from exception
        if isinstance(exception, (APITimeoutError, APIConnectionError)):
            raise AiProviderRequestError(
                f"{self.PROVIDER_DISPLAY_NAME} request failed before a status "
                f"was available: {exception}",
                status_code=None,
                provider_engine=self.PROVIDER_ENGINE_TOKEN,
                fallback_reason=classify_transport_fallback_reason(
                    isinstance(exception, APITimeoutError)
                ),
            ) from exception
        # Normal return so non-transport exceptions propagate unchanged.
        return None

    # ── Response mapping ────────────────────────────────────────────────────

    @staticmethod
    def _first_answer(response: _DecisionsResponse) -> _DecisionsAnswer:
        if not response.answers:
            raise AiProviderRequestError(
                "OpenAI Decisions returned no answers.",
                status_code=None,
                provider_engine=AiOpenAIDecisionsClassifier.PROVIDER_ENGINE_TOKEN,
                fallback_reason=None,
            )
        # Normal return with the one answer to the one question.
        return response.answers[0]

    def _usage(
        self, response: _DecisionsResponse, str_text: str, dict_question: dict[str, Any]
    ) -> tuple[AITokenUsage, str]:
        """Provider usage when present, else an estimate from the request text."""
        dict_usage: dict[str, Any] = response.usage or {}
        raw_input: Any = dict_usage.get("input_tokens", dict_usage.get("prompt_tokens"))
        if isinstance(raw_input, int):
            raw_total: Any = dict_usage.get("total_tokens", raw_input)
            # Early return with the provider's counts; output is free.
            return (
                AITokenUsage(
                    input_tokens=raw_input,
                    output_tokens=0,
                    total_tokens=raw_total if isinstance(raw_total, int) else raw_input,
                ),
                TOKEN_COUNT_SOURCE_PROVIDER,
            )
        str_request: str = str_text + " ".join(
            str(value) for value in self._question_strings(dict_question)
        )
        int_estimate: int = max(
            1, math.ceil(len(str_request) / DECISIONS_CHARS_PER_TOKEN_ESTIMATE)
        )
        # Normal return with the estimate.
        return (
            AITokenUsage(
                input_tokens=int_estimate, output_tokens=0, total_tokens=int_estimate
            ),
            TOKEN_COUNT_SOURCE_ESTIMATED,
        )

    @staticmethod
    def _question_strings(dict_question: dict[str, Any]) -> list[str]:
        list_strings: list[str] = [str(dict_question.get("instructions", ""))]
        # Loop over options or levels so their text counts toward the estimate.
        for item in dict_question.get("choices", []) + dict_question.get("levels", []):
            list_strings.append(" ".join(str(value) for value in item.values()))
        # Normal return with every string the request carries.
        return list_strings

    @staticmethod
    def _probabilities(answer: _DecisionsAnswer) -> dict[str, float] | None:
        if isinstance(answer.probabilities, dict):
            # Early return with the keyed distribution.
            return {
                str(key): float(value) for key, value in answer.probabilities.items()
            }
        if isinstance(answer.probabilities, list):
            dict_out: dict[str, float] = {}
            # Loop over list entries of the {value|label, probability} shape.
            for item in answer.probabilities:
                if isinstance(item, dict):
                    # Score entries carry the level index in "value" and the
                    # level name in "label"; choice entries carry only "value".
                    str_key: Any = item.get("label", item.get("value"))
                    raw_probability: Any = item.get("probability")
                    if str_key is not None and isinstance(
                        raw_probability, (int, float)
                    ):
                        dict_out[str(str_key)] = float(raw_probability)
            # Early return with the converted distribution, or None if empty.
            return dict_out or None
        # Normal return: no distribution reported.
        return None

    def _map_choice_answer(
        self, response: _DecisionsResponse, str_text: str, dict_question: dict[str, Any]
    ) -> AIClassificationResult:
        answer: _DecisionsAnswer = self._first_answer(response)
        usage, str_source = self._usage(response, str_text, dict_question)
        bool_refused: bool = (
            answer.type == STR_ANSWER_TYPE_REFUSAL or answer.choice is None
        )
        # Normal return with the mapped result.
        return AIClassificationResult(
            question_type="choice",
            choice=None if bool_refused else answer.choice,
            confidence=answer.confidence,
            probabilities=self._probabilities(answer),
            refused=bool_refused,
            raw_text=answer.model_dump_json(),
            usage=usage,
            token_source=str_source,
            provider_engine=self.PROVIDER_ENGINE_TOKEN,
            model_name=str(self.model),
        )

    def _map_predicate_answer(
        self, response: _DecisionsResponse, str_text: str, dict_question: dict[str, Any]
    ) -> AIClassificationResult:
        answer: _DecisionsAnswer = self._first_answer(response)
        usage, str_source = self._usage(response, str_text, dict_question)
        bool_refused: bool = (
            answer.type == STR_ANSWER_TYPE_REFUSAL or answer.probability is None
        )
        # Normal return with the mapped result.
        return AIClassificationResult(
            question_type="predicate",
            probability=None if bool_refused else answer.probability,
            confidence=answer.confidence,
            refused=bool_refused,
            raw_text=answer.model_dump_json(),
            usage=usage,
            token_source=str_source,
            provider_engine=self.PROVIDER_ENGINE_TOKEN,
            model_name=str(self.model),
        )

    def _map_score_answer(
        self,
        response: _DecisionsResponse,
        str_text: str,
        dict_question: dict[str, Any],
        list_levels: list[AIClassificationLevel],
    ) -> AIClassificationResult:
        answer: _DecisionsAnswer = self._first_answer(response)
        usage, str_source = self._usage(response, str_text, dict_question)
        list_labels: list[str] = [level.label for level in list_levels]
        dict_probabilities: dict[str, float] | None = self._probabilities(answer)
        float_score: float | None = None
        str_label: str | None = answer.label or answer.choice
        if dict_probabilities and all(
            label in dict_probabilities for label in list_labels
        ):
            # Expected position on the scale, normalized to [0, 1].
            float_score = sum(
                index * dict_probabilities[label]
                for index, label in enumerate(list_labels)
            ) / (len(list_labels) - 1)
            if str_label is None:
                str_label = max(
                    list_labels, key=lambda label: dict_probabilities[label]
                )
        elif str_label in list_labels:
            float_score = list_labels.index(str_label) / (len(list_labels) - 1)
        elif isinstance(answer.score, (int, float)):
            # The API's score is the probability-weighted average level index,
            # so normalize it over the scale and clamp into [0, 1].
            float_score = min(
                max(float(answer.score) / (len(list_labels) - 1), 0.0), 1.0
            )
        bool_refused: bool = (
            answer.type == STR_ANSWER_TYPE_REFUSAL or float_score is None
        )
        # Normal return with the mapped result.
        return AIClassificationResult(
            question_type="score",
            choice=None if bool_refused else str_label,
            score=None if bool_refused else float_score,
            confidence=answer.confidence,
            probabilities=dict_probabilities,
            refused=bool_refused,
            raw_text=answer.model_dump_json(),
            usage=usage,
            token_source=str_source,
            provider_engine=self.PROVIDER_ENGINE_TOKEN,
            model_name=str(self.model),
        )
