# tests/test_openai_decisions_classifier.py
"""
AiOpenAIDecisionsClassifier against a mocked SDK `post`: request body,
answer mapping for the three question types, refusals, provider versus
estimated usage, error classification, and the cost event it emits.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

openai = pytest.importorskip("openai")

from ai_api_unified import (  # noqa: E402
    AIClassificationChoice,
    AIClassificationLevel,
    AiFallbackReason,
    AiProviderRequestError,
)
from ai_api_unified.classification.ai_openai_decisions_classifier import (  # noqa: E402
    DECISIONS_API_PATH,
    AiOpenAIDecisionsClassifier,
    decisions_pricing_key,
)
from ai_api_unified.middleware.middleware_config import (
    ObservabilitySettingsModel,
)  # noqa: E402
from ai_api_unified.middleware.observability import (  # noqa: E402
    COST_LOGGER_NAME,
    LoggerBackedObservabilityMiddleware,
)
from ai_api_unified.pricing.pricing_registry import get_model_pricing  # noqa: E402

CHOICES: list[AIClassificationChoice] = [
    AIClassificationChoice(value="easy", description="Short and factual."),
    AIClassificationChoice(value="hard", description="Deep reasoning."),
]
LEVELS: list[AIClassificationLevel] = [
    AIClassificationLevel(label="low"),
    AIClassificationLevel(label="mid"),
    AIClassificationLevel(label="high"),
]


@contextmanager
def _settings(**overrides: str) -> Iterator[None]:
    dict_all: dict[str, str] = {"OPENAI_API_KEY": "test-key", **overrides}
    with (
        patch(
            "ai_api_unified.util.env_settings.EnvSettings.get_setting",
            side_effect=lambda key, default=None: dict_all.get(key, default),
        ),
        patch.dict(os.environ, dict_all, clear=False),
    ):
        yield


def _response(dict_body: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json=dict_body,
        request=httpx.Request("POST", "https://api.openai.com/v1/decisions"),
    )


def _choice_body(choice: str = "hard", **extra: Any) -> dict[str, Any]:
    return {
        "answers": [
            {
                "type": "choice",
                "name": "answer",
                "choice": choice,
                "confidence": 0.91,
                "probabilities": {"easy": 0.09, "hard": 0.91},
            }
        ],
        **extra,
    }


def _classifier(dict_body: dict[str, Any] | None = None) -> AiOpenAIDecisionsClassifier:
    with _settings():
        classifier = AiOpenAIDecisionsClassifier()
    classifier.client.post = Mock(return_value=_response(dict_body or _choice_body()))  # type: ignore[method-assign]
    classifier._resolve_provider_organization = lambda: (None, None)  # type: ignore[method-assign]
    return classifier


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class TestBuild:
    def test_defaults(self) -> None:
        classifier = _classifier()
        assert classifier.model_name == "gpt-6-luna"
        assert classifier.pricing_model_key == "decisions/gpt-6-luna"
        assert classifier.list_model_names == ["gpt-6-luna"]
        caps = classifier.capabilities
        assert caps.supports_choice and caps.supports_predicate and caps.supports_score
        assert caps.supports_async is True
        assert caps.pricing is not None
        assert caps.pricing.token_rates.input_per_1m == Decimal("0.10")

    def test_pricing_registry_entry(self) -> None:
        pricing = get_model_pricing("openai", decisions_pricing_key("gpt-6-luna"))
        assert pricing is not None
        assert pricing.token_rates.output_per_1m == Decimal("0")

    def test_org_info_hooks_come_from_the_openai_base(self) -> None:
        classifier = _classifier()
        assert classifier._get_org_info_capability_provider().supports_org_id is True

    def test_missing_key_fails_at_build(self) -> None:
        with _settings(OPENAI_API_KEY=""), pytest.raises(ValueError):
            AiOpenAIDecisionsClassifier()


class TestChoice:
    def test_request_body_and_mapping(self) -> None:
        classifier = _classifier()
        result = classifier.classify_choice(
            "Prove P != NP",
            CHOICES,
            instructions="How hard is this?",
            request_timeout_seconds=2.5,
        )
        assert result.choice == "hard" and result.confidence == 0.91
        assert result.probabilities == {"easy": 0.09, "hard": 0.91}
        assert result.refused is False
        assert (
            result.provider_engine == "openai-decisions"
            and result.model_name == "gpt-6-luna"
        )
        call = classifier.client.post.call_args
        assert call.args[0] == DECISIONS_API_PATH
        assert call.kwargs["cast_to"] is httpx.Response
        assert call.kwargs["options"] == {"timeout": 2.5}
        body = call.kwargs["body"]
        assert body["model"] == "gpt-6-luna" and body["input"] == "Prove P != NP"
        question = body["questions"][0]
        assert (
            question["type"] == "choice"
            and question["instructions"] == "How hard is this?"
        )
        assert question["choices"] == [
            {"value": "easy", "description": "Short and factual."},
            {"value": "hard", "description": "Deep reasoning."},
        ]

    def test_default_instructions_when_none_given(self) -> None:
        classifier = _classifier()
        classifier.classify_choice("hi", CHOICES)
        assert classifier.client.post.call_args.kwargs["body"]["questions"][0][
            "instructions"
        ]

    def test_refusal(self) -> None:
        classifier = _classifier({"answers": [{"type": "refusal", "name": "answer"}]})
        result = classifier.classify_choice("hi", CHOICES)
        assert result.refused is True and result.choice is None

    def test_provider_usage_is_preferred(self) -> None:
        classifier = _classifier(
            _choice_body(usage={"input_tokens": 57, "total_tokens": 57})
        )
        result = classifier.classify_choice("hi", CHOICES)
        assert result.usage.input_tokens == 57 and result.usage.output_tokens == 0
        assert result.token_source == "provider"

    def test_missing_usage_is_estimated_from_the_request(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        classifier = _classifier()
        with caplog.at_level(logging.WARNING):
            result = classifier.classify_choice("x" * 400, CHOICES)
        assert result.token_source == "estimated"
        assert (
            result.usage.input_tokens is not None and result.usage.input_tokens >= 100
        )
        # Estimated is metered, so no unmetered warning.
        assert not [r for r in caplog.records if getattr(r, "ai_classifier_event", "")]

    def test_async(self) -> None:
        classifier = _classifier()
        with patch.object(
            type(classifier),
            "async_client",
            new_callable=lambda: property(
                lambda self: Mock(
                    post=AsyncMock(return_value=_response(_choice_body("easy")))
                )
            ),
        ):
            result = asyncio.run(classifier.aclassify_choice("hi", CHOICES))
        assert result.choice == "easy"


class TestPredicateAndScore:
    def test_predicate(self) -> None:
        classifier = _classifier(
            {"answers": [{"type": "predicate", "name": "answer", "probability": 0.83}]}
        )
        result = classifier.classify_predicate(
            "Refund me", instructions="Does the user want a refund?"
        )
        assert result.probability == 0.83 and result.refused is False
        question = classifier.client.post.call_args.kwargs["body"]["questions"][0]
        assert question == {
            "type": "predicate",
            "name": "answer",
            "instructions": "Does the user want a refund?",
        }

    def test_score_uses_the_probability_distribution(self) -> None:
        classifier = _classifier(
            {
                "answers": [
                    {
                        "type": "score",
                        "name": "answer",
                        "score": 1.2,
                        "confidence": 0.7,
                        # The API keys score entries by level index in
                        # "value" and by level name in "label".
                        "probabilities": [
                            {"value": 0, "label": "low", "probability": 0.1},
                            {"value": 1, "label": "mid", "probability": 0.6},
                            {"value": 2, "label": "high", "probability": 0.3},
                        ],
                    }
                ]
            }
        )
        result = classifier.classify_score("hi", LEVELS)
        # Expected index 0*0.1 + 1*0.6 + 2*0.3 = 1.2, over 2 steps.
        assert result.score == pytest.approx(0.6)
        assert result.choice == "mid" and result.confidence == 0.7
        assert result.probabilities == {"low": 0.1, "mid": 0.6, "high": 0.3}
        question = classifier.client.post.call_args.kwargs["body"]["questions"][0]
        assert question["type"] == "score"
        assert [level["label"] for level in question["levels"]] == [
            "low",
            "mid",
            "high",
        ]

    def test_score_falls_back_to_the_label(self) -> None:
        classifier = _classifier(
            {"answers": [{"type": "score", "name": "answer", "label": "high"}]}
        )
        result = classifier.classify_score("hi", LEVELS)
        assert result.score == 1.0 and result.choice == "high"

    def test_score_without_distribution_normalizes_the_raw_score(self) -> None:
        classifier = _classifier(
            {"answers": [{"type": "score", "name": "answer", "score": 1.2}]}
        )
        result = classifier.classify_score("hi", LEVELS)
        # 1.2 is the weighted level index; over 2 steps that is 0.6.
        assert result.score == pytest.approx(0.6) and result.refused is False

    def test_choice_probabilities_list_shape(self) -> None:
        classifier = _classifier(
            {
                "answers": [
                    {
                        "type": "choice",
                        "name": "answer",
                        "choice": "easy",
                        "probabilities": [
                            {"value": "easy", "probability": 0.8},
                            {"value": "hard", "probability": 0.2},
                        ],
                    }
                ]
            }
        )
        result = classifier.classify_choice("hi", CHOICES)
        assert result.probabilities == {"easy": 0.8, "hard": 0.2}

    def test_score_refusal(self) -> None:
        classifier = _classifier({"answers": [{"type": "refusal", "name": "answer"}]})
        assert classifier.classify_score("hi", LEVELS).refused is True


class TestErrors:
    def test_quota_error_is_typed_and_classified(self) -> None:
        classifier = _classifier()
        request = httpx.Request("POST", "https://api.openai.com/v1/decisions")
        error = openai.APIStatusError(
            "no credits",
            response=httpx.Response(429, request=request),
            body={
                "error": {"code": "credit_balance_exhausted", "message": "no credits"}
            },
        )
        error.code = "credit_balance_exhausted"
        classifier.client.post = Mock(side_effect=error)  # type: ignore[method-assign]
        with pytest.raises(AiProviderRequestError) as exc_info:
            classifier.classify_choice("hi", CHOICES)
        assert exc_info.value.fallback_reason is AiFallbackReason.QUOTA_EXHAUSTED
        assert exc_info.value.provider_engine == "openai-decisions"

    def test_timeout_is_typed(self) -> None:
        classifier = _classifier()
        classifier.client.post = Mock(  # type: ignore[method-assign]
            side_effect=openai.APITimeoutError(
                request=httpx.Request("POST", "https://api.openai.com/v1/decisions")
            )
        )
        with pytest.raises(AiProviderRequestError) as exc_info:
            classifier.classify_choice("hi", CHOICES)
        assert exc_info.value.fallback_reason is AiFallbackReason.TIMEOUT

    def test_empty_answers_is_a_request_error(self) -> None:
        classifier = _classifier({"answers": []})
        with pytest.raises(AiProviderRequestError, match="no answers"):
            classifier.classify_choice("hi", CHOICES)


class TestMetering:
    def test_cost_event_prices_under_the_decisions_key(self) -> None:
        handler = _CaptureHandler()
        logger = logging.getLogger(COST_LOGGER_NAME)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            classifier = _classifier(_choice_body(usage={"input_tokens": 100000}))
            classifier._observability_middleware = LoggerBackedObservabilityMiddleware(
                ObservabilitySettingsModel(emit_cost=True)
            )
            classifier.classify_choice("hi", CHOICES)
        finally:
            logger.removeHandler(handler)
        assert len(handler.records) == 1
        fields = handler.records[0].args[1]
        assert fields["model"] == "decisions/gpt-6-luna"
        assert fields["provider"] == "openai"
        assert fields["provider_engine"] == "openai-decisions"
        assert fields["capability"] == "classification"
        assert fields["operation"] == "classify_choice"
        assert fields["input_tokens"] == 100000 and fields["output_tokens"] == 0
        # 100000 * 0.10 / 1M = 0.01
        assert Decimal(fields["usd_cost"]) == Decimal("0.01")

    def test_estimated_tokens_still_price_the_call(self) -> None:
        handler = _CaptureHandler()
        logger = logging.getLogger(COST_LOGGER_NAME)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            classifier = _classifier()
            classifier._observability_middleware = LoggerBackedObservabilityMiddleware(
                ObservabilitySettingsModel(emit_cost=True)
            )
            classifier.classify_choice("x" * 4000, CHOICES)
        finally:
            logger.removeHandler(handler)
        fields = handler.records[0].args[1]
        assert fields["input_tokens"] >= 1000
        assert Decimal(fields["usd_cost"]) > 0
