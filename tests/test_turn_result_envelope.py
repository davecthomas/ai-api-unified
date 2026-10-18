# tests/test_turn_result_envelope.py
"""
Conversation and structured-output results carry the same envelope as
send_prompt_with_result (provider, route, pin, cost), and
`provider_options["model"]` pins one call.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ai_api_unified import AiFallbackReason as R
from ai_api_unified.ai_base import (
    ENVELOPE_ACTIVE,
    AIBaseCompletions,
    AIFinishReason,
    AIStructuredOutputResult,
    AITurnResult,
)
from ai_api_unified.completions.ai_fallback_completions import (
    DEFAULT_FALLBACK_REASONS,
)
from ai_api_unified.completions.ai_router_completions import AiRouterCompletions
from ai_api_unified.routing.router_config import AIRouterConfig
from tests.test_model_router import _FakeClassifier, _tier
from tests.test_request_cost import (
    CLASSIFIER_COST,
    GPT_COST,
    OPUS_COST,
    _chain,
    _err,
    _PricedClassifier,
    _PricedEngine,
)

SCHEMA: dict[str, Any] = {"type": "object", "properties": {"a": {"type": "string"}}}
MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]


class _TemplateEngine(_PricedEngine):
    """A priced engine that runs the base template methods, as real engines do."""

    def __init__(self, name: str, **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._caps = self._caps.model_copy(update={"context_window_length": 100_000})

    @property
    def max_context_tokens(self) -> int:
        return 100_000

    send_conversation = AIBaseCompletions.send_conversation
    asend_conversation = AIBaseCompletions.asend_conversation
    send_structured_output = AIBaseCompletions.send_structured_output
    asend_structured_output = AIBaseCompletions.asend_structured_output

    def _send_conversation_provider(self, **kwargs: Any) -> AITurnResult:
        self.last_kwargs = kwargs
        return self._observed(
            "send_conversation",
            AITurnResult(
                text=f"{self.model}:turn", finish_reason=AIFinishReason.COMPLETE
            ),
        )

    async def _asend_conversation_provider(self, **kwargs: Any) -> AITurnResult:
        return self._send_conversation_provider(**kwargs)

    def _send_structured_output_provider(
        self, **kwargs: Any
    ) -> AIStructuredOutputResult:
        self.last_kwargs = kwargs
        return self._observed(
            "send_structured_output",
            AIStructuredOutputResult(
                data={"a": str(self.model)}, finish_reason=AIFinishReason.COMPLETE
            ),
        )

    async def _asend_structured_output_provider(
        self, **kwargs: Any
    ) -> AIStructuredOutputResult:
        return self._send_structured_output_provider(**kwargs)


def _template_chain(*engines: _TemplateEngine) -> Any:
    chain = _chain(*engines)
    chain._cooldown = chain._cooldown.model_copy(update={"enabled": False})
    return chain


class TestEnvelope:
    def test_plain_engine_turn_carries_provider_model_and_cost(self) -> None:
        engine = _TemplateEngine("p")
        turn = engine.send_conversation("sys", MESSAGES)
        assert turn.text == "p:turn"
        assert turn.provider == "openai" and turn.provider_engine == "openai"
        assert turn.model_name == "gpt-5.4"
        assert turn.cost is not None and turn.cost.total_usd == GPT_COST
        assert turn.route_tier is None and turn.model_pin is None

    def test_structured_result_carries_cost(self) -> None:
        result = _TemplateEngine("p").send_structured_output(
            "q", response_schema=SCHEMA
        )
        assert result.data == {"a": "p"}
        assert result.cost is not None and result.cost.total_usd == GPT_COST

    def test_chain_turn_reports_the_fallback_and_its_wasted_attempt(self) -> None:
        chain = _template_chain(
            _TemplateEngine("p", fail=_err(R.UNAVAILABLE, 529)),
            _TemplateEngine("f", vendor="anthropic", model="claude-opus-5"),
        )
        turn = chain.send_conversation("sys", MESSAGES)
        assert turn.text == "f:turn"
        assert turn.provider == "anthropic" and turn.provider_engine == "fake"
        assert turn.cost.total_usd == OPUS_COST and turn.cost.failed_calls == 1

    def test_router_structured_reports_tier_and_classifier_cost(self) -> None:
        engines = {"claude:e": _TemplateEngine("e"), "claude:h": _TemplateEngine("h")}
        router = AiRouterCompletions(
            default_client=_TemplateEngine("d"),
            config=AIRouterConfig(
                tiers=[_tier("easy", "claude:e"), _tier("hard", "claude:h")]
            ),
            classifier=_PricedClassifier("hard"),
            client_builder=lambda candidate: engines[candidate.label],
            fallback_on=DEFAULT_FALLBACK_REASONS,
            cooldown=False,
        )
        result = router.send_structured_output("prove it", response_schema=SCHEMA)
        assert result.data == {"a": "h"}
        assert result.route_tier == "hard" and result.route_method == "classifier"
        assert result.cost.total_usd == GPT_COST + CLASSIFIER_COST

    def test_envelope_flag_is_cleared(self) -> None:
        _TemplateEngine("p").send_conversation("sys", MESSAGES)
        assert ENVELOPE_ACTIVE.get() is False


class TestPins:
    def test_chain_conversation_starts_on_the_pin(self) -> None:
        a, b = _TemplateEngine("a"), _TemplateEngine("b")
        chain = _template_chain(a, b)
        turn = chain.send_conversation(
            "sys", MESSAGES, provider_options={"model": "b", "purpose": "x"}
        )
        assert turn.text == "b:turn" and a.calls == []
        assert turn.model_pin == "fake:b" and turn.route_method == "model_pin"
        # The reserved key never reaches the engine; other options do.
        assert "model" not in (b.last_kwargs["provider_options"] or {})

    def test_failed_pin_falls_back(self) -> None:
        a = _TemplateEngine("a")
        b = _TemplateEngine("b", fail=_err(R.UNAVAILABLE, 529))
        chain = _template_chain(a, b)
        result = chain.send_structured_output(
            "q", response_schema=SCHEMA, provider_options={"model": "b"}
        )
        assert result.data == {"a": "a"}
        assert result.cost.failed_calls == 1

    def test_router_pin_skips_classifier_and_route_option(self) -> None:
        engines = {"claude:e": _TemplateEngine("e"), "claude:h": _TemplateEngine("h")}
        classifier = _FakeClassifier("hard")
        router = AiRouterCompletions(
            default_client=_TemplateEngine("d"),
            config=AIRouterConfig(
                tiers=[_tier("easy", "claude:e"), _tier("hard", "claude:h")]
            ),
            classifier=classifier,
            client_builder=lambda candidate: engines[candidate.label],
            fallback_on=DEFAULT_FALLBACK_REASONS,
            cooldown=False,
        )
        turn = router.send_conversation(
            "sys", MESSAGES, provider_options={"model": "claude:e", "route": "hard"}
        )
        assert turn.text == "e:turn" and classifier.calls == []
        assert turn.route_tier == "easy" and turn.route_method == "model_pin"

    def test_plain_engine_pinning_its_own_model(self) -> None:
        engine = _TemplateEngine("p")
        engine._pin_candidate = None
        turn = engine.send_conversation(
            "sys", MESSAGES, provider_options={"model": "claude:p"}
        )
        assert turn.model_pin == "claude:p"
        assert "model" not in (engine.last_kwargs["provider_options"] or {})

    def test_async_pin(self) -> None:
        a, b = _TemplateEngine("a"), _TemplateEngine("b")
        chain = _template_chain(a, b)
        turn = asyncio.run(
            chain.asend_conversation("sys", MESSAGES, provider_options={"model": "b"})
        )
        assert turn.text == "b:turn" and turn.model_pin == "fake:b"
        assert turn.cost.total_usd == GPT_COST

    def test_blank_model_option_is_no_pin(self) -> None:
        turn = _TemplateEngine("p").send_conversation(
            "sys", MESSAGES, provider_options={"model": " "}
        )
        assert turn.model_pin is None

    def test_unknown_bare_pin_raises(self) -> None:
        with pytest.raises(ValueError):
            _template_chain(_TemplateEngine("a")).send_conversation(
                "sys", MESSAGES, provider_options={"model": "nope"}
            )
