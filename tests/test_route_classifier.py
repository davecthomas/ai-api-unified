# tests/test_route_classifier.py
"""
Route classification: the tier a request routes to and that tier's
candidates, with no candidate built or called. AiRouterCompletions offers it
as classify_route, and AiRouteClassifier is a router that does nothing else,
so its candidates may name engines the library does not have.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import patch

import pytest

from ai_api_unified import (
    AIFactory,
    AIRouteDecision,
    AiRouteClassifier,
    AIRouterClassifierConfig,
    AIRouterConfig,
    AIThinkingLevel,
)
from ai_api_unified.ai_models_config import AIModelPolicy
from ai_api_unified.ai_provider_exceptions import AiProviderConfigurationError
from tests.test_model_router import (
    _FakeClassifier,
    _router,
    _three,
    _three_tiers,
    _tier,
)

STR_TASK: str = "Add a --dry-run flag to the export command."

HARNESS_TIERS = [
    _tier(
        "standard",
        "harness:pi-lean/claude:claude-sonnet-5",
        "harness:pi-careful/openai:gpt-5.6-luna",
        description="A feature or fix inside one module.",
        thinking="low",
    ),
    _tier(
        "complex",
        "harness:pi-careful/claude:claude-opus-5-5",
        description="Cross-module changes.",
        thinking="high",
    ),
]


def _harness_config(**kwargs: object) -> AIRouterConfig:
    return AIRouterConfig(
        tiers=HARNESS_TIERS,
        default_tier=kwargs.pop("default_tier", "standard"),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


class TestClassifyRouteOnARouter:
    def test_returns_the_tier_and_its_candidates_without_building_them(
        self,
    ) -> None:
        engines = _three()
        router = _router(
            engines,
            tiers=_three_tiers(medium={"thinking": "high"}),
            classifier=_FakeClassifier("medium", confidence=0.9),
        )

        decision = router.classify_route(STR_TASK)

        assert isinstance(decision, AIRouteDecision)
        assert decision.tier == "medium" and decision.method == "classifier"
        assert decision.confidence == 0.9
        assert decision.thinking is AIThinkingLevel.HIGH
        assert [c.label for c in decision.candidates] == ["claude:m"]
        assert decision.classifier_usd == Decimal("0")
        router.builder.assert_not_called()  # type: ignore[attr-defined]
        assert all(engine.calls == [] for engine in engines.values())

    def test_floor_ceiling_and_override_apply(self) -> None:
        router = _router(
            _three(), tiers=_three_tiers(), classifier=_FakeClassifier("easy")
        )

        floored = router.classify_route(
            STR_TASK, provider_options={"route_floor": "hard"}
        )
        capped = router.classify_route(
            STR_TASK,
            provider_options={"route_floor": "medium", "route_ceiling": "medium"},
        )
        named = router.classify_route(STR_TASK, provider_options={"route": "hard"})

        assert (floored.tier, floored.method) == ("hard", "floor")
        assert (capped.tier, capped.method) == ("medium", "floor")
        assert (named.tier, named.method) == ("hard", "override")

    def test_an_unsure_classifier_goes_one_tier_up(self) -> None:
        router = _router(
            _three(),
            tiers=_three_tiers(),
            classifier=_FakeClassifier("easy", confidence=0.4),
            classifier_config=AIRouterClassifierConfig(min_confidence=0.7),
        )

        decision = router.classify_route(STR_TASK)

        assert (decision.tier, decision.method) == ("medium", "low_confidence")

    def test_a_failed_classifier_routes_to_the_default_tier(self) -> None:
        router = _router(
            _three(),
            tiers=_three_tiers(),
            classifier=_FakeClassifier(fail=RuntimeError("down")),
            default_tier="medium",
        )

        decision = router.classify_route(STR_TASK)

        assert (decision.tier, decision.method) == ("medium", "default")
        assert decision.confidence is None

    def test_with_no_classifier_or_default_tier_it_names_no_tier(self) -> None:
        router = _router(_three(), tiers=_three_tiers(), classifier=None)

        decision = router.classify_route(STR_TASK)

        assert decision.tier is None and decision.method == "default"

    def test_an_unknown_floor_raises(self) -> None:
        router = _router(_three(), tiers=_three_tiers(), classifier=None)

        with pytest.raises(ValueError, match="route_floor"):
            router.classify_route(STR_TASK, provider_options={"route_floor": "huge"})

    def test_async_twin_matches(self) -> None:
        router = _router(
            _three(), tiers=_three_tiers(), classifier=_FakeClassifier("hard")
        )

        decision = asyncio.run(router.aclassify_route(STR_TASK))

        assert (decision.tier, decision.method) == ("hard", "classifier")
        assert [c.label for c in decision.candidates] == ["claude:h"]

    def test_classification_does_not_advance_the_ratchet(self) -> None:
        router = _router(
            _three(),
            tiers=_three_tiers(),
            classifier=_FakeClassifier("hard"),
            followup="ratchet",
        )

        router.classify_route(STR_TASK)

        assert router.last_tier is None


class TestRouteClassifier:
    def test_harness_candidates_come_back_as_configured(self) -> None:
        classifier = AiRouteClassifier(
            config=_harness_config(), classifier=_FakeClassifier("complex")
        )

        decision = classifier.classify_route(STR_TASK)

        assert decision.tier == "complex"
        assert decision.thinking is AIThinkingLevel.HIGH
        assert [(c.engine, c.model) for c in decision.candidates] == [
            ("harness", "pi-careful/claude:claude-opus-5-5")
        ]

    def test_without_a_classifier_every_request_takes_the_default_tier(
        self,
    ) -> None:
        classifier = AiRouteClassifier(config=_harness_config(), classifier=None)

        decision = classifier.classify_route(STR_TASK)

        assert (decision.tier, decision.method) == ("standard", "default")
        assert [c.model for c in decision.candidates] == [
            "pi-lean/claude:claude-sonnet-5",
            "pi-careful/openai:gpt-5.6-luna",
        ]

    def test_a_floor_moves_the_default_tier_up(self) -> None:
        classifier = AiRouteClassifier(config=_harness_config(), classifier=None)

        decision = asyncio.run(
            classifier.aclassify_route(
                STR_TASK, provider_options={"route_floor": "complex"}
            )
        )

        assert (decision.tier, decision.method) == ("complex", "floor")

    def test_needs_a_default_tier(self) -> None:
        with pytest.raises(ValueError, match="default_tier"):
            AiRouteClassifier(
                config=AIRouterConfig(tiers=HARNESS_TIERS), classifier=None
            )

    def test_the_default_chain_is_refused(self) -> None:
        classifier = AiRouteClassifier(config=_harness_config(), classifier=None)

        with pytest.raises(ValueError, match="default chain"):
            classifier.classify_route(STR_TASK, provider_options={"route": "default"})

    def test_it_answers_no_completions_call(self) -> None:
        classifier = AiRouteClassifier(config=_harness_config(), classifier=None)

        assert not hasattr(classifier, "send_prompt")
        with pytest.raises(AiProviderConfigurationError, match="classify_route"):
            classifier._router.default_client.send_prompt("hi")

    def test_the_decision_carries_the_config_id(self) -> None:
        classifier = AiRouteClassifier(
            config=_harness_config(), classifier=None, config_id="abc123"
        )

        assert classifier.classify_route(STR_TASK).config_id == "abc123"


class TestFactory:
    def test_builds_over_harness_candidates_without_checking_them(self) -> None:
        config = _harness_config(classifier=AIRouterClassifierConfig(engine="none"))
        with patch(
            "ai_api_unified.ai_factory.AiModelChecker.check_all",
            side_effect=AssertionError("tier candidates must not be probed"),
        ):
            classifier = AIFactory.get_route_classifier(router=config)

        decision = classifier.classify_route(STR_TASK)

        assert classifier.classifier is None
        assert decision.tier == "standard"
        assert decision.config_id == classifier.config_id
        assert classifier.config_id

    def test_a_denied_classifier_is_left_out(self) -> None:
        config = _harness_config(
            classifier=AIRouterClassifierConfig(
                engine="claude", model="claude-haiku-4-5"
            )
        )
        with (
            patch.object(
                AIFactory,
                "_model_policy",
                return_value=AIModelPolicy(deny=["claude:claude-haiku-4-5"]),
            ),
            patch.object(
                AIFactory,
                "get_ai_classifier_client",
                side_effect=AssertionError("a denied classifier must not be built"),
            ),
        ):
            classifier = AIFactory.get_route_classifier(router=config)

        assert classifier.classifier is None
        assert classifier.classify_route(STR_TASK).method == "default"

    def test_no_router_raises(self) -> None:
        with pytest.raises(ValueError, match="No router is configured"):
            AIFactory.get_route_classifier()
