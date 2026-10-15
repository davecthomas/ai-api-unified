# tests/test_structured_output_classifier.py
"""
AiStructuredOutputClassifier over a fake completions engine: the schema it
sends, the answers it maps, refusals, usage, the classifier_role tag, and
the capability check at build.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ai_api_unified import (
    AIBaseCompletions,
    AIClassificationChoice,
    AIClassificationLevel,
    AICompletionsCapabilitiesBase,
    AIFinishReason,
    AIStructuredOutputResult,
    AITokenUsage,
)
from ai_api_unified.ai_completions_exceptions import StructuredResponseTokenLimitError
from ai_api_unified.ai_provider_exceptions import (
    AiProviderCapabilityUnsupportedError,
    AiProviderConfigurationError,
)
from ai_api_unified.classification.ai_structured_output_classifier import (
    DEFAULT_CLASSIFIER_MAX_RESPONSE_TOKENS,
    AiStructuredOutputClassifier,
)
from ai_api_unified.middleware.observability_runtime import (
    get_observability_context,
    reset_observability_context,
    set_observability_context,
)

CHOICES: list[AIClassificationChoice] = [
    AIClassificationChoice(value="easy", description="Short and factual."),
    AIClassificationChoice(value="medium", description="Some reasoning."),
    AIClassificationChoice(value="hard", description="Deep reasoning."),
]
LEVELS: list[AIClassificationLevel] = [
    AIClassificationLevel(label="low"),
    AIClassificationLevel(label="mid"),
    AIClassificationLevel(label="high"),
]


class _FakeCompletions(AIBaseCompletions):
    """Answers every structured call with a scripted value."""

    def __init__(
        self,
        answer: Any = "medium",
        *,
        finish: AIFinishReason = AIFinishReason.COMPLETE,
        usage: AITokenUsage | None = None,
        supports_structured_output: bool = True,
    ) -> None:
        super().__init__(model="fake-llm")
        self.answer = answer
        self.finish = finish
        self.usage = (
            usage
            if usage is not None
            else AITokenUsage(input_tokens=20, output_tokens=2, total_tokens=22)
        )
        self._caps = AICompletionsCapabilitiesBase(
            context_window_length=1000,
            supports_structured_output=supports_structured_output,
            supports_async=True,
        )
        self.kwargs: dict[str, Any] = {}
        self.tags_seen: dict[str, str] = {}

    @property
    def capabilities(self) -> AICompletionsCapabilitiesBase:
        return self._caps

    @property
    def list_model_names(self) -> list[str]:
        return ["fake-llm"]

    @property
    def max_context_tokens(self) -> int:
        return 1000

    def _reply(self, **kwargs: Any) -> AIStructuredOutputResult:
        self.kwargs = kwargs
        self.tags_seen = dict(get_observability_context().tags)
        data = None if self.answer is None else {"answer": self.answer}
        return AIStructuredOutputResult(
            data=data,
            finish_reason=self.finish,
            usage=self.usage,
            raw_text=str(self.answer),
        )

    def send_structured_output(self, prompt: Any = None, **kwargs: Any) -> Any:
        return self._reply(prompt=prompt, **kwargs)

    async def asend_structured_output(self, prompt: Any = None, **kwargs: Any) -> Any:
        return self._reply(prompt=prompt, **kwargs)

    # Unused abstract surface.
    def send_prompt(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError

    async def asend_prompt(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError

    def send_prompt_streaming(self, prompt: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    def count_tokens(self, prompt: str, **kwargs: Any) -> int:
        raise NotImplementedError

    def strict_schema_prompt(
        self,
        prompt: str,
        response_model: Any,
        max_response_tokens: int = 2048,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError

    def send_conversation(
        self, system_prompt: str, messages: Any, **kwargs: Any
    ) -> Any:
        raise NotImplementedError

    async def asend_conversation(
        self, system_prompt: str, messages: Any, **kwargs: Any
    ) -> Any:
        raise NotImplementedError


def _classifier(**kwargs: Any) -> tuple[AiStructuredOutputClassifier, _FakeCompletions]:
    engine = _FakeCompletions(**kwargs)
    return (
        AiStructuredOutputClassifier(engine="fake", completions_client=engine),
        engine,
    )


class TestBuild:
    def test_requires_engine_or_client(self) -> None:
        with pytest.raises(AiProviderConfigurationError):
            AiStructuredOutputClassifier()

    def test_rejects_engine_without_structured_output(self) -> None:
        with pytest.raises(AiProviderCapabilityUnsupportedError, match="structured"):
            AiStructuredOutputClassifier(
                completions_client=_FakeCompletions(supports_structured_output=False)
            )

    def test_capabilities_follow_the_delegate(self) -> None:
        classifier, _ = _classifier()
        caps = classifier.capabilities
        assert caps.supports_choice and caps.supports_predicate and caps.supports_score
        assert caps.supports_async is True
        assert classifier.model_name == "fake-llm"
        assert classifier.list_model_names == ["fake-llm"]


class TestChoice:
    def test_sends_an_enum_schema_and_maps_the_answer(self) -> None:
        classifier, engine = _classifier(answer="hard")
        result = classifier.classify_choice(
            "Prove P != NP",
            CHOICES,
            instructions="Rate difficulty.",
            request_timeout_seconds=3.0,
        )
        assert result.choice == "hard" and result.refused is False
        schema = engine.kwargs["response_schema"]
        assert schema["properties"]["answer"]["enum"] == ["easy", "medium", "hard"]
        assert schema["required"] == ["answer"]
        assert (
            engine.kwargs["max_response_tokens"]
            == DEFAULT_CLASSIFIER_MAX_RESPONSE_TOKENS
        )
        assert engine.kwargs["request_timeout_seconds"] == 3.0
        system = engine.kwargs["system_prompt"]
        assert "Rate difficulty." in system and "- hard: Deep reasoning." in system

    def test_usage_comes_from_the_delegate(self) -> None:
        classifier, _ = _classifier()
        result = classifier.classify_choice("hi", CHOICES)
        assert result.usage.input_tokens == 20 and result.token_source == "provider"
        assert result.provider_engine == "fake"

    def test_missing_usage_is_reported_as_none(self) -> None:
        classifier, _ = _classifier(usage=AITokenUsage())
        assert classifier.classify_choice("hi", CHOICES).token_source == "none"

    def test_refusal(self) -> None:
        classifier, _ = _classifier(answer=None, finish=AIFinishReason.REFUSAL)
        result = classifier.classify_choice("hi", CHOICES)
        assert result.refused is True and result.choice is None

    def test_answer_outside_the_enum_is_a_refusal(self) -> None:
        classifier, _ = _classifier(answer="bogus")
        assert classifier.classify_choice("hi", CHOICES).refused is True

    def test_truncation_raises_instead_of_reporting_a_refusal(self) -> None:
        classifier, _ = _classifier(answer=None, finish=AIFinishReason.LENGTH)
        with pytest.raises(StructuredResponseTokenLimitError):
            classifier.classify_choice("hi", CHOICES)

    def test_classifier_role_tag_is_visible_to_the_delegate_and_reset(self) -> None:
        classifier, engine = _classifier()
        token = set_observability_context(caller_id="svc", tags={"team": "x"})
        try:
            classifier.classify_choice("hi", CHOICES)
            assert engine.tags_seen == {"team": "x", "classifier_role": "choice"}
            assert dict(get_observability_context().tags) == {"team": "x"}
        finally:
            reset_observability_context(token)

    def test_async(self) -> None:
        classifier, _ = _classifier(answer="easy")
        assert asyncio.run(classifier.aclassify_choice("hi", CHOICES)).choice == "easy"


class TestPredicateAndScore:
    @pytest.mark.parametrize(("answer", "probability"), [("yes", 1.0), ("no", 0.0)])
    def test_predicate_maps_to_a_hard_probability(
        self, answer: str, probability: float
    ) -> None:
        classifier, engine = _classifier(answer=answer)
        result = classifier.classify_predicate("hi", instructions="Is it a question?")
        assert result.probability == probability and result.choice is None
        assert engine.kwargs["response_schema"]["properties"]["answer"]["enum"] == [
            "yes",
            "no",
        ]
        assert "Is it a question?" in engine.kwargs["system_prompt"]

    @pytest.mark.parametrize(
        ("answer", "score"), [("low", 0.0), ("mid", 0.5), ("high", 1.0)]
    )
    def test_score_is_the_normalized_level_position(
        self, answer: str, score: float
    ) -> None:
        classifier, engine = _classifier(answer=answer)
        result = classifier.classify_score("hi", LEVELS)
        assert result.score == score and result.choice == answer
        assert engine.kwargs["response_schema"]["properties"]["answer"]["enum"] == [
            "low",
            "mid",
            "high",
        ]
        assert engine.tags_seen["classifier_role"] == "score"

    def test_async_predicate_and_score(self) -> None:
        classifier, _ = _classifier(answer="yes")
        assert (
            asyncio.run(
                classifier.aclassify_predicate("hi", instructions="Q?")
            ).probability
            == 1.0
        )
        classifier, _ = _classifier(answer="mid")
        assert asyncio.run(classifier.aclassify_score("hi", LEVELS)).score == 0.5
