# tests/test_model_pin.py
"""
Per-call model pins: `send_prompt_with_result(model=...)` skips routing,
starts on the pinned model, keeps the configured fallback, and marks the
call with the `model_pin` and `route_method` tags and a MODEL PIN log line.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from ai_api_unified import AIFactory, AiFallbackReason as R
from ai_api_unified.ai_models_config import clear_models_config_cache
from ai_api_unified.ai_provider_exceptions import AiProviderConfigurationError
from ai_api_unified.completions.ai_fallback_completions import (
    DEFAULT_FALLBACK_REASONS,
    AIFallbackCandidate,
    AiFallbackCompletions,
)
from ai_api_unified.completions.ai_model_pin import (
    MODEL_PIN,
    ROUTE_METHOD_MODEL_PIN,
    resolve_model_pin,
)
from tests.test_model_router import (
    _engines,
    _FakeClassifier,
    _FakeEngine,
    _router,
)
from tests.test_request_cost import _err


def _chain(
    engines: list[_FakeEngine], extra: dict[str, _FakeEngine] | None = None
) -> AiFallbackCompletions:
    """A claude chain over `engines`; `extra` holds models only a pin builds."""
    dict_by_label: dict[str, _FakeEngine] = {
        f"claude:{engine.model}": engine for engine in engines
    }
    dict_by_label.update(extra or {})
    chain = AiFallbackCompletions(
        primary=engines[0],
        primary_candidate=AIFallbackCandidate(
            engine="claude", model=str(engines[0].model)
        ),
        fallback_candidates=[
            AIFallbackCandidate(
                engine="claude", model=str(engine.model), base_url=f"https://{i}"
            )
            for i, engine in enumerate(engines[1:])
        ],
        client_builder=Mock(side_effect=lambda c: dict_by_label[c.label]),
        fallback_on=DEFAULT_FALLBACK_REASONS,
        cooldown=False,
    )
    return chain


class TestResolve:
    KNOWN = [
        AIFallbackCandidate(engine="claude", model="claude-haiku-4-5"),
        AIFallbackCandidate(
            engine="bedrock", model="amazon.nova-lite-v1:0", base_url=None
        ),
    ]

    def test_engine_and_model(self) -> None:
        pin = resolve_model_pin("openai:gpt-5.4-mini", self.KNOWN)
        assert (pin.engine, pin.model) == ("openai", "gpt-5.4-mini")

    def test_bedrock_ids_keep_their_colons(self) -> None:
        pin = resolve_model_pin("bedrock:amazon.nova-pro-v1:0", self.KNOWN)
        assert (pin.engine, pin.model) == ("bedrock", "amazon.nova-pro-v1:0")

    def test_bare_model_resolves_to_its_configured_candidate(self) -> None:
        assert resolve_model_pin("amazon.nova-lite-v1:0", self.KNOWN) == self.KNOWN[1]
        assert resolve_model_pin("claude-haiku-4-5", self.KNOWN) == self.KNOWN[0]

    def test_a_configured_pair_keeps_its_base_url(self) -> None:
        known = [AIFallbackCandidate(engine="openai", model="m", base_url="https://x")]
        assert resolve_model_pin("openai:m", known).base_url == "https://x"

    @pytest.mark.parametrize("text", ["", "  ", "claude:", "no-such-model"])
    def test_bad_pins_raise(self, text: str) -> None:
        with pytest.raises(ValueError):
            resolve_model_pin(text, self.KNOWN)


class TestFallbackChain:
    def test_pin_starts_on_the_pinned_candidate(self) -> None:
        a, b, c = _FakeEngine("a"), _FakeEngine("b"), _FakeEngine("c")
        chain = _chain([a, b, c])
        result = chain.send_prompt_with_result("hi", model="b")
        assert result.text == "b:ok"
        assert a.calls == [] and b.calls == ["send_prompt"]
        assert result.model_pin == "claude:b"
        assert b.tags_seen[0]["model_pin"] == "claude:b"
        assert b.tags_seen[0]["route_method"] == ROUTE_METHOD_MODEL_PIN

    def test_a_failed_pin_falls_back_along_the_chain_in_order(self) -> None:
        a = _FakeEngine("a", fail=_err(R.UNAVAILABLE, 529))
        b = _FakeEngine("b", fail=_err(R.UNAVAILABLE, 529))
        c = _FakeEngine("c")
        chain = _chain([a, b, c])
        assert chain.send_prompt_with_result("hi", model="b").text == "c:ok"
        assert b.calls == ["send_prompt"] and a.calls == ["send_prompt"]

    def test_a_non_trigger_failure_on_the_pin_raises(self) -> None:
        b = _FakeEngine("b", fail=_err(R.MODEL_UNAVAILABLE, 404))
        a = _FakeEngine("a")
        chain = _chain([a, b])
        with pytest.raises(Exception):
            chain.send_prompt_with_result("hi", model="b")
        assert a.calls == []

    def test_a_pin_outside_the_chain_is_built_once_and_never_used_unpinned(
        self,
    ) -> None:
        a = _FakeEngine("a")
        z = _FakeEngine("z")
        chain = _chain([a], extra={"claude:z": z})
        assert chain.send_prompt_with_result("1", model="claude:z").text == "z:ok"
        assert chain.send_prompt_with_result("2", model="claude:z").text == "z:ok"
        assert chain._client_builder.call_count == 1
        assert [c.label for c in chain.candidates] == ["claude:a"]
        a.fail = _err(R.UNAVAILABLE, 529)
        with pytest.raises(Exception):
            chain.send_prompt("3")
        assert z.calls == ["send_prompt", "send_prompt"]

    def test_a_pin_that_cannot_be_built_raises(self) -> None:
        chain = _chain([_FakeEngine("a")])
        with pytest.raises(KeyError):
            chain.send_prompt_with_result("hi", model="claude:nope")

    def test_the_pin_is_cleared_after_the_call(self) -> None:
        chain = _chain([_FakeEngine("a"), _FakeEngine("b")])
        chain.send_prompt_with_result("hi", model="b")
        assert MODEL_PIN.get() is None
        assert chain.send_prompt_with_result("hi").text == "a:ok"

    def test_pin_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        chain = _chain([_FakeEngine("a"), _FakeEngine("b")])
        with caplog.at_level(logging.INFO):
            chain.send_prompt_with_result("hi", model="b")
        records = [r for r in caplog.records if r.message.startswith("MODEL PIN")]
        assert len(records) == 1
        assert records[0].ai_route_event == "model_pin"  # type: ignore[attr-defined]
        assert records[0].model_pin == "claude:b"  # type: ignore[attr-defined]


class TestRouter:
    def test_pin_skips_the_classifier_and_runs_the_tier_that_holds_it(
        self,
    ) -> None:
        engines = _engines("e", "h")
        classifier = _FakeClassifier("easy")
        router = _router(engines, classifier=classifier)
        result = router.send_prompt_with_result("hi", model="claude:h")
        assert result.text == "h:ok"
        assert classifier.calls == []
        assert result.route_tier == "hard"
        assert result.route_method == ROUTE_METHOD_MODEL_PIN
        assert engines["claude:h"].tags_seen[0]["route_method"] == "model_pin"

    def test_a_pinned_tier_never_escalates(self) -> None:
        engines = _engines("e", "h", e=_err(R.UNAVAILABLE, 529))
        router = _router(engines, classifier=_FakeClassifier("easy"))
        with pytest.raises(Exception):
            router.send_prompt_with_result("hi", model="claude:e")
        assert engines["claude:h"].calls == []

    def test_unpinned_calls_still_escalate(self) -> None:
        engines = _engines("e", "h", e=_err(R.UNAVAILABLE, 529))
        router = _router(engines, classifier=_FakeClassifier("easy"))
        assert router.send_prompt_with_result("hi").text == "h:ok"

    def test_a_pin_in_no_tier_runs_on_the_default_chain(self) -> None:
        default = _FakeEngine("d")
        default._configure_model_pins(
            candidate=AIFallbackCandidate(engine="claude", model="d"),
            client_builder=lambda c: engines[c.label],
            fallback_on=DEFAULT_FALLBACK_REASONS,
            cooldown=False,
        )
        engines = {**_engines("e", "h"), "claude:x": _FakeEngine("x")}
        router = _router(engines, classifier=_FakeClassifier("easy"), default=default)
        result = router.send_prompt_with_result("hi", model="claude:x")
        assert result.text == "x:ok"
        assert result.route_tier == "default"

    def test_bare_model_from_a_tier(self) -> None:
        engines = _engines("e", "h")
        router = _router(engines, classifier=_FakeClassifier("easy"))
        assert router.send_prompt_with_result("hi", model="h").text == "h:ok"

    def test_routed_event_names_the_pin(self, caplog: pytest.LogCaptureFixture) -> None:
        router = _router(_engines("e", "h"), classifier=_FakeClassifier("easy"))
        with caplog.at_level(logging.INFO):
            router.send_prompt_with_result("hi", model="claude:h")
        routed = [r for r in caplog.records if r.message.startswith("ROUTED")]
        assert routed and routed[0].method == "model_pin"  # type: ignore[attr-defined]


class TestSingleClient:
    def _configured(self, extra: dict[str, _FakeEngine]) -> _FakeEngine:
        primary = _FakeEngine("p")
        primary._configure_model_pins(
            candidate=AIFallbackCandidate(engine="claude", model="p"),
            client_builder=lambda c: extra[c.label],
            fallback_on=DEFAULT_FALLBACK_REASONS,
            cooldown=False,
        )
        return primary

    def test_pin_runs_the_pinned_model(self) -> None:
        primary = self._configured({"claude:x": _FakeEngine("x")})
        result = primary.send_prompt_with_result("hi", model="claude:x")
        assert result.text == "x:ok" and primary.calls == []
        assert primary.last_request_cost is not None

    def test_a_failed_pin_falls_back_to_the_configured_model(self) -> None:
        primary = self._configured(
            {"claude:x": _FakeEngine("x", fail=_err(R.UNAVAILABLE, 529))}
        )
        assert primary.send_prompt_with_result("hi", model="claude:x").text == "p:ok"

    def test_pinning_its_own_model_without_factory_support_is_allowed(self) -> None:
        engine = _FakeEngine("p")
        assert engine.send_prompt_with_result("hi", model="claude:p").text == "p:ok"

    def test_another_model_without_factory_support_raises(self) -> None:
        with pytest.raises(
            AiProviderConfigurationError, match="not built by AIFactory"
        ):
            _FakeEngine("p").send_prompt_with_result("hi", model="claude:x")


@pytest.fixture
def models_file(tmp_path: Path) -> Any:
    def _write(dict_config: dict[str, Any]) -> Path:
        path = tmp_path / "models.json"
        path.write_text(json.dumps(dict_config))
        clear_models_config_cache()
        return path

    yield _write
    clear_models_config_cache()


class TestFactory:
    def _build(self, **kwargs: Any) -> Any:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            return AIFactory.get_ai_completions_client(
                completions_engine="claude", model_name="claude-haiku-4-5", **kwargs
            )

    def test_single_client_gets_pin_support(self) -> None:
        client = self._build(fallbacks=[])
        pin = resolve_model_pin(
            "claude:claude-sonnet-4-5", client._known_pin_candidates()
        )
        chain = client._model_pin_target(pin)
        assert isinstance(chain, AiFallbackCompletions)
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            int_index = chain._pin_index(pin)
        assert chain._clients[int_index].model_name == "claude-sonnet-4-5"
        assert chain.config_id == client.config_id

    def test_policy_denies_a_pinned_model(self, models_file: Any) -> None:
        path = models_file(
            {
                "completions": {"engine": "claude", "model": "claude-haiku-4-5"},
                "deny": ["claude:claude-opus-*"],
            }
        )
        with patch.dict(os.environ, {"AI_MODELS_CONFIG": str(path)}):
            client = self._build(fallbacks=[])
            with pytest.raises(ValueError, match="allow/deny"):
                client.send_prompt_with_result("hi", model="claude:claude-opus-5-5")
