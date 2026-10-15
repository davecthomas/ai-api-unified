# tests/test_classification_base.py
"""
Template-method contract of AIBaseClassifier: validation, PII redaction,
truncation, the observability wrapping, result stamping, and the
unmetered warning.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import Mock

import pytest

from ai_api_unified import (
    AIBaseClassifier,
    AIClassificationChoice,
    AIClassificationLevel,
    AIClassificationResult,
    AIClassifierCapabilitiesBase,
    AITokenUsage,
)
from ai_api_unified.ai_base import (
    CLASSIFIER_EVENT_UNMETERED,
    DEFAULT_CLASSIFIER_MAX_INPUT_CHARS,
    truncate_classification_text,
)
from ai_api_unified.ai_provider_exceptions import AiProviderCapabilityUnsupportedError

LOGGER_NAME: str = "ai_api_unified.ai_base"

CHOICES: list[AIClassificationChoice] = [
    AIClassificationChoice(value="easy", description="Short and factual."),
    AIClassificationChoice(value="hard", description="Needs reasoning."),
]
LEVELS: list[AIClassificationLevel] = [
    AIClassificationLevel(label="low"),
    AIClassificationLevel(label="high"),
]


class _FakeClassifier(AIBaseClassifier):
    """Records what the template hands the provider hooks."""

    def __init__(
        self,
        *,
        capabilities: AIClassifierCapabilitiesBase | None = None,
        token_source: str = "provider",
    ) -> None:
        super().__init__(model="fake-1")
        self._caps = capabilities or AIClassifierCapabilitiesBase(
            supports_predicate=True, supports_score=True, supports_async=True
        )
        self.token_source = token_source
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def list_model_names(self) -> list[str]:
        return ["fake-1"]

    @property
    def capabilities(self) -> AIClassifierCapabilitiesBase:
        return self._caps

    def _result(self, question_type: str, **fields: Any) -> AIClassificationResult:
        usage = (
            AITokenUsage(input_tokens=10, output_tokens=1, total_tokens=11)
            if self.token_source != "none"
            else AITokenUsage()
        )
        return AIClassificationResult(
            question_type=question_type,  # type: ignore[arg-type]
            usage=usage,
            token_source=self.token_source,
            **fields,
        )

    def _classify_choice_provider(self, str_text, list_choices, **kwargs):
        self.calls.append(("choice", {"text": str_text, **kwargs}))
        return self._result("choice", choice=list_choices[0].value, confidence=0.9)

    def _classify_predicate_provider(self, str_text, **kwargs):
        self.calls.append(("predicate", {"text": str_text, **kwargs}))
        return self._result("predicate", probability=0.2)

    def _classify_score_provider(self, str_text, list_levels, **kwargs):
        self.calls.append(("score", {"text": str_text, **kwargs}))
        return self._result("score", choice=list_levels[-1].label, score=1.0)

    async def _aclassify_choice_provider(self, str_text, list_choices, **kwargs):
        self.calls.append(("achoice", {"text": str_text, **kwargs}))
        return self._result("choice", choice=list_choices[1].value)


class TestTruncation:
    def test_short_text_is_unchanged(self) -> None:
        assert truncate_classification_text("abc", 10) == "abc"
        assert truncate_classification_text("abc", None) == "abc"

    def test_long_text_keeps_head_and_tail(self) -> None:
        str_text = "H" * 100 + "T" * 50
        str_cut = truncate_classification_text(str_text, 30)
        assert str_cut.startswith("H" * 20)
        assert str_cut.endswith("T" * 10)
        assert "\n...\n" in str_cut

    def test_default_cap_applies(self) -> None:
        classifier = _FakeClassifier()
        classifier.classify_choice(
            "x" * (DEFAULT_CLASSIFIER_MAX_INPUT_CHARS + 100), CHOICES
        )
        assert (
            len(classifier.calls[0][1]["text"])
            < DEFAULT_CLASSIFIER_MAX_INPUT_CHARS + 10
        )


class TestValidation:
    def test_empty_text_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _FakeClassifier().classify_choice("  ", CHOICES)

    def test_one_choice_rejected(self) -> None:
        with pytest.raises(ValueError, match="two choices"):
            _FakeClassifier().classify_choice("hi", CHOICES[:1])

    def test_duplicate_choice_values_rejected(self) -> None:
        with pytest.raises(ValueError, match="distinct"):
            _FakeClassifier().classify_choice("hi", [CHOICES[0], CHOICES[0]])

    def test_max_choices_enforced(self) -> None:
        classifier = _FakeClassifier(
            capabilities=AIClassifierCapabilitiesBase(max_choices=1)
        )
        with pytest.raises(ValueError, match="at most 1"):
            classifier.classify_choice("hi", CHOICES)

    def test_predicate_needs_instructions(self) -> None:
        with pytest.raises(ValueError, match="instructions"):
            _FakeClassifier().classify_predicate("hi", instructions=" ")

    def test_levels_validated(self) -> None:
        with pytest.raises(ValueError, match="two levels"):
            _FakeClassifier().classify_score("hi", LEVELS[:1])

    def test_unsupported_question_type_raises_capability_error(self) -> None:
        classifier = _FakeClassifier(capabilities=AIClassifierCapabilitiesBase())
        with pytest.raises(
            AiProviderCapabilityUnsupportedError, match="classify_predicate"
        ):
            classifier.classify_predicate("hi", instructions="Is it a question?")

    def test_async_without_support_raises(self) -> None:
        classifier = _FakeClassifier(capabilities=AIClassifierCapabilitiesBase())
        with pytest.raises(AiProviderCapabilityUnsupportedError, match="async"):
            asyncio.run(classifier.aclassify_choice("hi", CHOICES))


class TestTemplate:
    def test_pii_redaction_runs_before_the_hook(self) -> None:
        classifier = _FakeClassifier()
        classifier.pii_middleware = Mock(process_input=Mock(return_value="<redacted>"))
        classifier.classify_choice("call me at 555-0100", CHOICES)
        classifier.pii_middleware.process_input.assert_called_once_with(
            "call me at 555-0100"
        )
        assert classifier.calls[0][1]["text"] == "<redacted>"

    def test_result_is_stamped(self) -> None:
        classifier = _FakeClassifier()
        result = classifier.classify_choice("hi", CHOICES, instructions="Pick.")
        assert result.choice == "easy"
        assert result.model_name == "fake-1"
        assert result.provider_engine == "_fakeclassifier"
        assert result.latency_ms > 0
        assert classifier.calls[0][1]["instructions"] == "Pick."

    def test_predicate_and_score_results(self) -> None:
        classifier = _FakeClassifier()
        assert classifier.classify_predicate("hi", instructions="Q?").probability == 0.2
        scored = classifier.classify_score("hi", LEVELS)
        assert scored.score == 1.0 and scored.choice == "high"

    def test_async_choice(self) -> None:
        classifier = _FakeClassifier()
        result = asyncio.run(classifier.aclassify_choice("hi", CHOICES))
        assert result.choice == "hard"
        assert classifier.calls[0][0] == "achoice"

    def test_hook_runs_inside_the_observability_lifecycle(self) -> None:
        classifier = _FakeClassifier()
        captured: dict[str, Any] = {}

        def _observed(**kwargs: Any) -> Any:
            captured.update(kwargs)
            result = kwargs["callable_execute"]()
            captured["summary"] = kwargs["callable_build_result_summary"](result, 12.5)
            return result

        classifier._execute_provider_call_with_observability = _observed  # type: ignore[method-assign]
        classifier.classify_choice("hi", CHOICES)
        assert captured["capability"] == "classification"
        assert captured["operation"] == "classify_choice"
        assert captured["dict_input_metadata"]["option_count"] == 2
        summary = captured["summary"]
        assert summary.provider_prompt_tokens == 10
        assert summary.input_token_count_source == "provider"
        assert summary.provider_elapsed_ms == 12.5
        assert summary.dict_metadata["question_type"] == "choice"

    def test_provider_observed_engines_are_not_wrapped_twice(self) -> None:
        class _Delegating(_FakeClassifier):
            BOOL_PROVIDER_OBSERVED = True

        classifier = _Delegating()
        classifier._execute_provider_call_with_observability = Mock(  # type: ignore[method-assign]
            side_effect=AssertionError("must not wrap")
        )
        assert classifier.classify_choice("hi", CHOICES).choice == "easy"

    def test_unmetered_call_logs_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        classifier = _FakeClassifier(token_source="none")
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            result = classifier.classify_choice("hi", CHOICES)
        assert result.token_source == "none"
        events = [
            r
            for r in caplog.records
            if getattr(r, "ai_classifier_event", "") == CLASSIFIER_EVENT_UNMETERED
        ]
        assert len(events) == 1
        assert events[0].operation == "classify_choice"

    def test_metered_call_logs_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            _FakeClassifier().classify_choice("hi", CHOICES)
        assert caplog.records == []
