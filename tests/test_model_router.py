# tests/test_model_router.py
"""
AiRouterCompletions over fake engines and a fake classifier: routing by
classification, the default route when the classifier is missing or fails,
overrides, escalation, pinned histories, route tags, streaming, result
stamping, lazy tier builds, and the factory wiring.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import Mock, patch

import pytest

from ai_api_unified import (
    AIBaseClassifier,
    AIBaseCompletions,
    AIClassificationChoice,
    AIClassificationResult,
    AIClassifierCapabilitiesBase,
    AICompletionsCapabilitiesBase,
    AIFactory,
    AIFinishReason,
    AIRouterClassifierConfig,
    AIRouterConfig,
    AIRouterTier,
    AIStructuredOutputResult,
    AITokenUsage,
    AITurnResult,
    AiFallbackCompletions,
    AiFallbackReason as R,
    AiProviderRequestError,
    AiRouterCompletions,
)
from ai_api_unified.ai_provider_exceptions import AiProviderCapabilityUnsupportedError
from ai_api_unified.completions.ai_fallback_completions import (
    DEFAULT_FALLBACK_REASONS,
    AIFallbackCandidate,
)
from ai_api_unified.middleware.observability_runtime import (
    get_observability_context,
    reset_observability_context,
    set_observability_context,
)

LOGGER_NAME: str = "ai_api_unified.completions.ai_router_completions"


def _err(reason: R | None, status: int | None = 503) -> AiProviderRequestError:
    return AiProviderRequestError(
        f"fail:{reason}", status_code=status, fallback_reason=reason
    )


class _FakeEngine(AIBaseCompletions):
    """Returns its name, records calls and the tags it saw, or raises `fail`."""

    def __init__(
        self,
        name: str,
        *,
        fail: Exception | None = None,
        family: str = "anthropic",
        engine: str = "claude",
    ) -> None:
        super().__init__(model=name)
        self.fail = fail
        self.FALLBACK_ENGINE_FAMILY = family
        self.engine_token = engine
        self.calls: list[str] = []
        self.tags_seen: list[dict[str, str]] = []
        self._caps = AICompletionsCapabilitiesBase(
            context_window_length=1000,
            supports_streaming=True,
            supports_tool_use=True,
            supports_structured_output=True,
            supports_async=True,
        )

    def _hit(self, op: str) -> None:
        self.calls.append(op)
        self.tags_seen.append(dict(get_observability_context().tags))
        if self.fail is not None:
            raise self.fail

    @property
    def capabilities(self) -> AICompletionsCapabilitiesBase:
        return self._caps

    @property
    def list_model_names(self) -> list[str]:
        return [str(self.model)]

    @property
    def max_context_tokens(self) -> int:
        return 1000

    def send_prompt(self, prompt: str, **kwargs: Any) -> str:
        self._hit("send_prompt")
        return f"{self.model}:ok"

    async def asend_prompt(self, prompt: str, **kwargs: Any) -> str:
        self._hit("asend_prompt")
        return f"{self.model}:ok"

    def send_prompt_streaming(
        self, prompt: str, *, other_params: Any = None
    ) -> Iterator[str]:
        def _gen() -> Iterator[str]:
            self._hit("send_prompt_streaming")
            yield f"{self.model}-1"
            yield f"{self.model}-2"

        return _gen()

    def count_tokens(self, prompt: str, *, other_params: Any = None) -> int:
        self.calls.append("count_tokens")
        return 7

    def strict_schema_prompt(
        self,
        prompt: str,
        response_model: Any,
        max_response_tokens: int = 2048,
        *,
        other_params: Any = None,
    ) -> Any:
        self.calls.append("strict_schema_prompt")
        return response_model(answer=str(self.model))

    def send_structured_output(self, prompt: Any = None, **kwargs: Any) -> Any:
        self._hit("send_structured_output")
        self.last_kwargs = kwargs
        return AIStructuredOutputResult(
            data={"answer": str(self.model)}, finish_reason=AIFinishReason.COMPLETE
        )

    async def asend_structured_output(self, prompt: Any = None, **kwargs: Any) -> Any:
        self._hit("asend_structured_output")
        self.last_kwargs = kwargs
        return AIStructuredOutputResult(
            data={"answer": str(self.model)}, finish_reason=AIFinishReason.COMPLETE
        )

    def send_conversation(
        self, system_prompt: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AITurnResult:
        self._hit("send_conversation")
        self.last_kwargs = kwargs
        return AITurnResult(
            text=f"{self.model}:turn",
            finish_reason=AIFinishReason.COMPLETE,
            raw_content=[{"type": "text", "text": "x"}],
        )

    async def asend_conversation(
        self, system_prompt: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AITurnResult:
        self._hit("asend_conversation")
        return AITurnResult(
            text=f"{self.model}:turn",
            finish_reason=AIFinishReason.COMPLETE,
            raw_content=[{"type": "text", "text": "x"}],
        )

    def build_tool_result_message(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t"}],
            "engine": str(self.model),
        }

    def extend_messages_with_turn(
        self, messages: list[dict[str, Any]], turn: AITurnResult
    ) -> list[dict[str, Any]]:
        messages.append(
            {"role": "assistant", "content": turn.raw_content, "by": str(self.model)}
        )
        return messages


class _FakeClassifier(AIBaseClassifier):
    """Returns a scripted tier, refuses, or raises."""

    def __init__(
        self,
        choice: str | None = "hard",
        *,
        fail: Exception | None = None,
        supports_async: bool = True,
    ) -> None:
        super().__init__(model="fake-classifier")
        self.choice = choice
        self.fail = fail
        self._async = supports_async
        self.calls: list[dict[str, Any]] = []
        self.tags_seen: list[dict[str, str]] = []

    @property
    def list_model_names(self) -> list[str]:
        return ["fake-classifier"]

    @property
    def capabilities(self) -> AIClassifierCapabilitiesBase:
        return AIClassifierCapabilitiesBase(supports_async=self._async)

    def _classify_choice_provider(
        self, str_text: str, list_choices: list[AIClassificationChoice], **kwargs: Any
    ) -> AIClassificationResult:
        self.calls.append(
            {"text": str_text, "choices": [c.value for c in list_choices], **kwargs}
        )
        self.tags_seen.append(dict(get_observability_context().tags))
        if self.fail is not None:
            raise self.fail
        return AIClassificationResult(
            question_type="choice",
            choice=self.choice,
            refused=self.choice is None,
            confidence=0.8,
            probabilities={"easy": 0.2, "hard": 0.8},
            usage=AITokenUsage(input_tokens=12, output_tokens=0),
            token_source="provider",
        )

    async def _aclassify_choice_provider(
        self, str_text: str, list_choices: list[AIClassificationChoice], **kwargs: Any
    ) -> AIClassificationResult:
        return self._classify_choice_provider(str_text, list_choices, **kwargs)


def _tier(name: str, *models: str, **kwargs: Any) -> AIRouterTier:
    return AIRouterTier(name=name, models=list(models), **kwargs)  # type: ignore[call-arg]


def _router(
    engines: dict[str, _FakeEngine],
    *,
    tiers: list[AIRouterTier] | None = None,
    classifier: AIBaseClassifier | None = None,
    default: _FakeEngine | None = None,
    **config_kwargs: Any,
) -> AiRouterCompletions:
    """Builds a router whose builder hands out `engines` by candidate label."""
    config = AIRouterConfig(
        tiers=tiers or [_tier("easy", "claude:e"), _tier("hard", "claude:h")],
        **config_kwargs,
    )
    builder = Mock(side_effect=lambda candidate: engines[candidate.label])
    router = AiRouterCompletions(
        default_client=default or _FakeEngine("d"),
        config=config,
        classifier=classifier,
        client_builder=builder,
        fallback_on=DEFAULT_FALLBACK_REASONS,
    )
    router.builder = builder  # type: ignore[attr-defined]
    return router


def _engines(*names: str, **fails: Exception) -> dict[str, _FakeEngine]:
    return {f"claude:{name}": _FakeEngine(name, fail=fails.get(name)) for name in names}


def _events(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.ai_router_event for r in caplog.records if hasattr(r, "ai_router_event")]


# ── Routing by classification ───────────────────────────────────────────────


class TestClassifiedRouting:
    def test_send_prompt_goes_to_the_classified_tier(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        engines = _engines("e", "h")
        router = _router(engines, classifier=_FakeClassifier("hard"))
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            assert router.send_prompt("Prove it") == "h:ok"
        assert engines["claude:e"].calls == []
        assert router.last_tier == "hard"
        assert (
            router.last_decision is not None and router.last_decision.choice == "hard"
        )
        routed = [
            r for r in caplog.records if getattr(r, "ai_router_event", None) == "routed"
        ][0]
        assert routed.tier == "hard" and routed.method == "classifier"
        assert routed.confidence == 0.8 and routed.classifier_input_tokens == 12
        assert (
            routed.classifier_token_source == "provider"
            and routed.operation == "send_prompt"
        )
        assert routed.route_id

    def test_classifier_sees_the_tiers_and_the_router_role_tag(self) -> None:
        classifier = _FakeClassifier("easy")
        router = _router(_engines("e", "h"), classifier=classifier)
        router.send_prompt("hi")
        call = classifier.calls[0]
        assert call["choices"] == ["easy", "hard"] and call["text"] == "hi"
        assert "Pick the one tier" in call["instructions"]
        assert classifier.tags_seen[0] == {"router_role": "classifier"}
        # The tag is gone once the call returns.
        assert dict(get_observability_context().tags) == {}

    def test_custom_instructions_and_timeout_reach_the_classifier(self) -> None:
        classifier = _FakeClassifier("easy")
        config = AIRouterConfig(
            tiers=[_tier("easy", "claude:e"), _tier("hard", "claude:h")],
            classifier=AIRouterClassifierConfig(
                engine="x", timeout_seconds=1.5, instructions="Prefer easy."
            ),
        )
        engines = _engines("e", "h")
        router = AiRouterCompletions(
            default_client=_FakeEngine("d"),
            config=config,
            classifier=classifier,
            client_builder=lambda c: engines[c.label],
            fallback_on=DEFAULT_FALLBACK_REASONS,
        )
        router.send_prompt("hi")
        assert classifier.calls[0]["instructions"].startswith("Prefer easy.")
        assert classifier.calls[0]["request_timeout_seconds"] == 1.5

    def test_every_routed_method(self) -> None:
        engines = _engines("e", "h")
        router = _router(engines, classifier=_FakeClassifier("easy"))
        assert router.send_prompt("p") == "e:ok"
        assert asyncio.run(router.asend_prompt("p")) == "e:ok"
        assert router.send_structured_output(
            "p", response_schema={"type": "object"}
        ).data == {"answer": "e"}
        assert asyncio.run(
            router.asend_structured_output("p", response_schema={"type": "object"})
        ).data == {"answer": "e"}
        assert (
            router.send_conversation("s", [{"role": "user", "content": "p"}]).text
            == "e:turn"
        )
        assert (
            asyncio.run(
                router.asend_conversation("s", [{"role": "user", "content": "p"}])
            ).text
            == "e:turn"
        )
        assert list(router.send_prompt_streaming("p")) == ["e-1", "e-2"]
        assert engines["claude:h"].calls == []

    def test_results_are_stamped_by_the_chain(self) -> None:
        router = _router(_engines("e", "h"), classifier=_FakeClassifier("hard"))
        turn = router.send_conversation("s", [{"role": "user", "content": "p"}])
        assert turn.provider_engine == "claude" and turn.model_name == "h"

    def test_tier_chains_are_built_once_and_lazily(self) -> None:
        router = _router(_engines("e", "h"), classifier=_FakeClassifier("hard"))
        router.send_prompt("a")
        router.send_prompt("b")
        assert [c.args[0].label for c in router.builder.call_args_list] == ["claude:h"]
        assert isinstance(router.for_tier("hard"), AiFallbackCompletions)

    def test_route_tags_reach_the_engine_and_are_reset(self) -> None:
        engines = _engines("e", "h")
        router = _router(engines, classifier=_FakeClassifier("hard"))
        token = set_observability_context(caller_id="svc", tags={"team": "x"})
        try:
            router.send_prompt("p")
            tags = engines["claude:h"].tags_seen[0]
            assert (
                tags["team"] == "x"
                and tags["route_tier"] == "hard"
                and tags["route_method"] == "classifier"
            )
            assert tags["route_id"]
            assert dict(get_observability_context().tags) == {"team": "x"}
        finally:
            reset_observability_context(token)

    def test_async_classifier_without_async_support_uses_the_sync_path(self) -> None:
        classifier = _FakeClassifier("easy", supports_async=False)
        router = _router(_engines("e", "h"), classifier=classifier)
        assert asyncio.run(router.asend_prompt("p")) == "e:ok"
        assert len(classifier.calls) == 1


# ── Default route ───────────────────────────────────────────────────────────


class TestDefaultRoute:
    def test_no_classifier_routes_to_the_default_chain(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        default = _FakeEngine("d")
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            router = _router(_engines("e", "h"), classifier=None, default=default)
        assert _events(caplog) == ["classifier_unavailable"]
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            assert router.send_prompt("p") == "d:ok"
        routed = [
            r for r in caplog.records if getattr(r, "ai_router_event", None) == "routed"
        ][0]
        assert routed.tier == "default" and routed.method == "default"
        assert router.last_tier is None
        router.builder.assert_not_called()

    def test_no_classifier_with_default_tier(self) -> None:
        router = _router(_engines("e", "h"), classifier=None, default_tier="easy")
        assert router.send_prompt("p") == "e:ok"

    def test_classifier_exception_routes_to_default(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        default = _FakeEngine("d")
        router = _router(
            _engines("e", "h"),
            classifier=_FakeClassifier(fail=_err(R.QUOTA_EXHAUSTED, 429)),
            default=default,
        )
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            assert router.send_prompt("p") == "d:ok"
        failed = [
            r
            for r in caplog.records
            if getattr(r, "ai_router_event", None) == "classifier_failed"
        ][0]
        assert (
            failed.error == "AiProviderRequestError"
            and failed.fallback_reason == "quota_exhausted"
        )
        assert failed.routed_to == "default"

    def test_refusal_and_unknown_tier_route_to_default(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        for classifier in (_FakeClassifier(None), _FakeClassifier("bogus")):
            router = _router(
                _engines("e", "h"), classifier=classifier, default_tier="easy"
            )
            with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
                assert router.send_prompt("p") == "e:ok"
        errors = [
            r.error
            for r in caplog.records
            if getattr(r, "ai_router_event", None) == "classifier_failed"
        ]
        assert errors == ["refusal", "unknown_tier"]

    def test_empty_text_skips_the_classifier(self) -> None:
        classifier = _FakeClassifier("hard")
        router = _router(_engines("e", "h"), classifier=classifier)
        router.send_conversation("s", [{"role": "assistant", "content": "x"}])
        assert classifier.calls == []


# ── Overrides and bypass ────────────────────────────────────────────────────


class TestOverrides:
    def test_route_option_pins_a_tier_and_is_stripped(self) -> None:
        engines = _engines("e", "h")
        classifier = _FakeClassifier("hard")
        router = _router(engines, classifier=classifier)
        result = router.send_structured_output(
            "p",
            response_schema={"type": "object"},
            provider_options={"route": "easy", "x": 1},
        )
        assert result.data == {"answer": "e"} and classifier.calls == []
        assert engines["claude:e"].last_kwargs["provider_options"] == {"x": 1}

    def test_route_default_pins_the_default_chain(self) -> None:
        default = _FakeEngine("d")
        router = _router(
            _engines("e", "h"), classifier=_FakeClassifier("hard"), default=default
        )
        turn = router.send_conversation(
            "s",
            [{"role": "user", "content": "p"}],
            provider_options={"route": "default"},
        )
        assert turn.text == "d:turn"

    def test_unknown_route_raises(self) -> None:
        router = _router(_engines("e", "h"), classifier=_FakeClassifier("hard"))
        with pytest.raises(ValueError, match="Unknown router tier"):
            router.send_conversation(
                "s",
                [{"role": "user", "content": "p"}],
                provider_options={"route": "expert"},
            )

    def test_for_tier(self) -> None:
        router = _router(_engines("e", "h"), classifier=_FakeClassifier("hard"))
        assert router.for_tier("easy").send_prompt("p") == "e:ok"
        assert router.for_tier("default") is router.default_client
        with pytest.raises(ValueError):
            router.for_tier("expert")

    def test_pass_through_methods_use_the_default_client(self) -> None:
        default = _FakeEngine("d")
        router = _router(
            _engines("e", "h"), classifier=_FakeClassifier("hard"), default=default
        )
        assert router.count_tokens("p") == 7
        assert router.capabilities is default.capabilities
        assert router.list_model_names == ["d"]
        assert "count_tokens" in default.calls


# ── Escalation ──────────────────────────────────────────────────────────────


class TestEscalation:
    def test_trigger_failure_escalates_to_the_next_tier(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        engines = _engines("e", "h", e=_err(R.UNAVAILABLE))
        router = _router(engines, classifier=_FakeClassifier("easy"))
        with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
            assert router.send_prompt("p") == "h:ok"
        exhausted = [
            r
            for r in caplog.records
            if getattr(r, "ai_router_event", None) == "tier_exhausted"
        ][0]
        assert exhausted.tier == "easy" and exhausted.escalate_to == "hard"
        assert exhausted.fallback_reason == "unavailable"
        assert router.last_tier == "hard"

    def test_non_trigger_failure_propagates(self) -> None:
        engines = _engines("e", "h", e=_err(None, 400))
        router = _router(engines, classifier=_FakeClassifier("easy"))
        with pytest.raises(AiProviderRequestError):
            router.send_prompt("p")
        assert engines["claude:h"].calls == []

    def test_raise_mode_does_not_escalate(self) -> None:
        engines = _engines("e", "h", e=_err(R.UNAVAILABLE))
        router = _router(
            engines, classifier=_FakeClassifier("easy"), on_exhausted="raise"
        )
        with pytest.raises(AiProviderRequestError):
            router.send_prompt("p")
        assert engines["claude:h"].calls == []

    def test_top_tier_failure_propagates(self) -> None:
        engines = _engines("e", "h", h=_err(R.UNAVAILABLE))
        router = _router(engines, classifier=_FakeClassifier("hard"))
        with pytest.raises(AiProviderRequestError):
            router.send_prompt("p")

    def test_default_chain_failure_propagates(self) -> None:
        router = _router(
            _engines("e", "h"),
            classifier=None,
            default=_FakeEngine("d", fail=_err(R.UNAVAILABLE)),
        )
        with pytest.raises(AiProviderRequestError):
            router.send_prompt("p")

    def test_capability_error_does_not_escalate(self) -> None:
        engines = _engines("e", "h", e=AiProviderCapabilityUnsupportedError("no"))
        router = _router(engines, classifier=_FakeClassifier("easy"))
        with pytest.raises(AiProviderCapabilityUnsupportedError):
            router.send_prompt("p")

    def test_unbuildable_tier_escalates(self, caplog: pytest.LogCaptureFixture) -> None:
        engines = _engines("h")
        config = AIRouterConfig(
            tiers=[_tier("easy", "claude:e"), _tier("hard", "claude:h")]
        )

        def _build(candidate: AIFallbackCandidate) -> AIBaseCompletions:
            if candidate.label == "claude:e":
                raise RuntimeError("missing key")
            return engines[candidate.label]

        router = AiRouterCompletions(
            default_client=_FakeEngine("d"),
            config=config,
            classifier=_FakeClassifier("easy"),
            client_builder=_build,
            fallback_on=DEFAULT_FALLBACK_REASONS,
        )
        with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
            assert router.send_prompt("p") == "h:ok"
            assert router.send_prompt("p") == "h:ok"
        assert _events(caplog).count("tier_unbuildable") == 1

    def test_async_escalation(self) -> None:
        engines = _engines("e", "h", e=_err(R.RATE_LIMITED, 429))
        router = _router(engines, classifier=_FakeClassifier("easy"))
        assert asyncio.run(router.asend_prompt("p")) == "h:ok"

    def test_streaming_escalates_before_the_first_chunk(self) -> None:
        engines = _engines("e", "h", e=_err(R.UNAVAILABLE))
        router = _router(engines, classifier=_FakeClassifier("easy"))
        assert list(router.send_prompt_streaming("p")) == ["h-1", "h-2"]
        assert router.last_tier == "hard"


# ── Pinned histories ────────────────────────────────────────────────────────

ANTHROPIC_HISTORY: list[dict[str, Any]] = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": [{"type": "text", "text": "x"}]},
    {"role": "user", "content": "more"},
]


class TestPinnedHistory:
    def test_engine_shaped_history_skips_the_classifier_and_stays_in_its_tier(
        self,
    ) -> None:
        engines = _engines("e", "h")
        classifier = _FakeClassifier("hard")
        router = _router(engines, classifier=classifier)
        router.send_conversation(
            "s", [{"role": "user", "content": "p"}]
        )  # hard, classified
        turn = router.send_conversation("s", list(ANTHROPIC_HISTORY))
        assert turn.text == "h:turn" and len(classifier.calls) == 1

    def test_pinned_history_prefers_the_first_eligible_tier_otherwise(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        engines = {
            "openai:o": _FakeEngine("o", family="openai"),
            "claude:c": _FakeEngine("c"),
        }
        router = _router(
            engines,
            tiers=[_tier("easy", "openai:o"), _tier("hard", "claude:c")],
            classifier=_FakeClassifier("easy"),
        )
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            turn = router.send_conversation("s", list(ANTHROPIC_HISTORY))
        assert turn.text == "c:turn"
        routed = [
            r for r in caplog.records if getattr(r, "ai_router_event", None) == "routed"
        ][0]
        assert routed.method == "pinned" and routed.tier == "hard"

    def test_pinned_history_without_an_eligible_tier_uses_the_default_chain(
        self,
    ) -> None:
        engines = {"openai:o": _FakeEngine("o", family="openai")}
        default = _FakeEngine("d")
        router = _router(
            engines,
            tiers=[_tier("easy", "openai:o")],
            classifier=_FakeClassifier("easy"),
            default=default,
        )
        assert router.send_conversation("s", list(ANTHROPIC_HISTORY)).text == "d:turn"

    def test_tool_result_and_turn_extension_follow_the_producing_tier(self) -> None:
        engines = _engines("e", "h")
        router = _router(engines, classifier=_FakeClassifier("hard"))
        messages: list[dict[str, Any]] = [{"role": "user", "content": "p"}]
        turn = router.send_conversation("s", messages)
        router.extend_messages_with_turn(messages, turn)
        assert messages[-1]["by"] == "h"
        message = router.build_tool_result_message(
            tool_call_id="t", result={}, messages=messages
        )
        assert message["engine"] == "h"


# ── Factory ─────────────────────────────────────────────────────────────────

BASE_SETTINGS: dict[str, str] = {
    "ANTHROPIC_API_KEY": "test-key",
    "COMPLETIONS_ENGINE": "claude",
    "COMPLETIONS_MODEL_NAME": "claude-haiku-4-5",
}
ROUTER_JSON: str = (
    '{"tiers": [{"name": "easy", "models": ["claude:claude-haiku-4-5"]}, {"name": "hard", "models": ["claude:claude-opus-5-5"]}]}'
)


@contextmanager
def _settings(dict_settings: dict[str, str]) -> Iterator[None]:
    dict_all = {**BASE_SETTINGS, **dict_settings}
    with (
        patch(
            "ai_api_unified.util.env_settings.EnvSettings.get_setting",
            side_effect=lambda key, default=None: dict_all.get(key, default),
        ),
        patch.dict(os.environ, dict_all, clear=False),
    ):
        yield


class TestFactory:
    def test_no_router_setting_returns_the_plain_client(self) -> None:
        pytest.importorskip("anthropic")
        with _settings({}):
            client = AIFactory.get_ai_completions_client()
        assert not isinstance(client, AiRouterCompletions)

    def test_inline_json_builds_a_router_with_a_classifier(self) -> None:
        pytest.importorskip("anthropic")
        with _settings(
            {
                "COMPLETIONS_ROUTER_CONFIG": ROUTER_JSON,
                "COMPLETIONS_ROUTER_CLASSIFIER": "claude:claude-haiku-4-5",
            }
        ):
            client = AIFactory.get_ai_completions_client()
        assert isinstance(client, AiRouterCompletions)
        assert [t.name for t in client.tiers] == ["easy", "hard"]
        assert (
            client.classifier is not None
            and client.classifier.model_name == "claude-haiku-4-5"
        )

    def test_classifier_none_disables_it(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        pytest.importorskip("anthropic")
        with (
            _settings(
                {
                    "COMPLETIONS_ROUTER_CONFIG": ROUTER_JSON,
                    "COMPLETIONS_ROUTER_CLASSIFIER": "none",
                }
            ),
            caplog.at_level(logging.WARNING, logger=LOGGER_NAME),
        ):
            client = AIFactory.get_ai_completions_client()
        assert isinstance(client, AiRouterCompletions) and client.classifier is None
        assert [
            r.reason
            for r in caplog.records
            if getattr(r, "ai_router_event", None) == "classifier_unavailable"
        ] == ["disabled"]

    def test_unresolvable_classifier_degrades_instead_of_raising(self) -> None:
        pytest.importorskip("anthropic")
        with _settings({"COMPLETIONS_ROUTER_CONFIG": ROUTER_JSON}):
            client = AIFactory.get_ai_completions_client()
        assert isinstance(client, AiRouterCompletions) and client.classifier is None

    def test_router_argument_and_config_classifier(self) -> None:
        pytest.importorskip("anthropic")
        config = AIRouterConfig(
            tiers=[_tier("easy", "claude:claude-haiku-4-5")],
            classifier=AIRouterClassifierConfig(
                engine="claude", model="claude-haiku-4-5"
            ),
        )
        with _settings({}):
            client = AIFactory.get_ai_completions_client(router=config)
        assert isinstance(client, AiRouterCompletions) and client.classifier is not None

    def test_bad_tier_engine_fails_at_build(self) -> None:
        pytest.importorskip("anthropic")
        with (
            _settings(
                {
                    "COMPLETIONS_ROUTER_CONFIG": '{"tiers": [{"name": "easy", "models": ["bogus:x"]}]}'
                }
            ),
            pytest.raises(ValueError, match="bogus"),
        ):
            AIFactory.get_ai_completions_client()

    def test_fallback_chain_becomes_the_default_route(self) -> None:
        pytest.importorskip("anthropic")
        with _settings(
            {
                "COMPLETIONS_ROUTER_CONFIG": ROUTER_JSON,
                "COMPLETIONS_FALLBACKS": "claude:claude-sonnet-5",
                "COMPLETIONS_ROUTER_CLASSIFIER": "none",
            }
        ):
            client = AIFactory.get_ai_completions_client()
        assert isinstance(client, AiRouterCompletions)
        assert isinstance(client.default_client, AiFallbackCompletions)
