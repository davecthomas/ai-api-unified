# tests/test_response_cache.py
"""
The response cache: off until configured; then an identical request gets
the stored answer without reaching the router or the provider, the hit is
metered with the saved cost, per-call options opt out, and the store is
replaceable and fails open.
"""

from __future__ import annotations

import asyncio
import json
import logging
from decimal import Decimal
from typing import Any

import pytest

from ai_api_unified import (
    AIBaseClassifier,
    AIClassificationChoice,
    AIClassificationResult,
    AIFactory,
    AITurnResult,
    AiFallbackReason as R,
    InMemoryCacheStore,
    cache_mode,
    request_cost_scope,
    set_cache_store,
)
from ai_api_unified.ai_base import AIBaseEmbeddings
from ai_api_unified.middleware.cache_store import get_cache_store
from ai_api_unified.middleware.middleware_config import ResponseCacheSettingsModel
from ai_api_unified.middleware.response_cache import (
    RESPONSE_CACHE_ACTIVE,
    build_cache_key,
    reset_response_cache,
)
from tests.test_request_cost import GPT_COST, _chain, _err, _PricedEngine
from tests.test_turn_result_envelope import MESSAGES, _TemplateEngine


@pytest.fixture
def cache_on(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Turns the cache on through models.json with the given settings."""

    def _enable(dict_settings: dict[str, Any] | None = None) -> None:
        entry: dict[str, Any] = {"name": "response_cache", "enabled": True}
        if dict_settings is not None:
            entry["settings"] = dict_settings
        monkeypatch.setenv("AI_MODELS_CONFIG", json.dumps({"middleware": [entry]}))
        from ai_api_unified.ai_models_config import clear_models_config_cache

        clear_models_config_cache()
        reset_response_cache()

    yield _enable
    reset_response_cache()


class _FakeEmbeddings(AIBaseEmbeddings):
    def __init__(self) -> None:
        super().__init__(model="embed-1", dimensions=3)
        self.calls: int = 0

    @property
    def list_model_names(self) -> list[str]:
        return ["embed-1"]

    def generate_embeddings(self, text: str, *, input_type: str | None = None) -> dict:
        self.calls += 1
        return {"text": text, "embedding": [0.1, 0.2, 0.3], "n": self.calls}

    def generate_embeddings_batch(
        self, texts: list[str], *, input_type: str | None = None
    ) -> list[dict]:
        self.calls += 1
        return [{"text": t, "embedding": [0.0]} for t in texts]


class _FakeClassifier(AIBaseClassifier):
    def __init__(self) -> None:
        super().__init__(model="cls")
        self.calls: int = 0

    @property
    def list_model_names(self) -> list[str]:
        return ["cls"]

    def _classify_choice_provider(
        self, str_text: str, list_choices: list[AIClassificationChoice], **_: Any
    ) -> AIClassificationResult:
        self.calls += 1
        return AIClassificationResult(question_type="choice", choice="a")


class TestStore:
    def test_lru_and_ttl(self) -> None:
        clock: list[float] = [0.0]
        store = InMemoryCacheStore(max_entries=2, callable_now=lambda: clock[0])
        store.set("a", b"1", 10)
        store.set("b", b"2", 10)
        assert store.get("a") == b"1"  # a is now most recent
        store.set("c", b"3", 10)  # evicts b
        assert store.get("b") is None and store.get("a") == b"1"
        clock[0] = 11.0
        assert store.get("a") is None and len(store) == 1
        store.set("d", b"4", 0)
        assert store.get("d") is None

    def test_set_cache_store_validates_and_swaps(self) -> None:
        with pytest.raises(TypeError):
            set_cache_store(object())  # type: ignore[arg-type]
        mine = InMemoryCacheStore()
        set_cache_store(mine)
        assert get_cache_store() is mine
        AIFactory.set_cache_store(None)
        assert get_cache_store() is not mine


class TestSettings:
    def test_defaults_and_overrides(self) -> None:
        s = ResponseCacheSettingsModel()
        assert s.ttl_for("embeddings") == 86400 and s.ttl_for("completions") == 3600
        s = ResponseCacheSettingsModel.model_validate(
            {"ttl_seconds": 10, "operations": {"completions": {"enabled": False}}}
        )
        assert s.ttl_for("completions") is None and s.ttl_for("embeddings") == 10
        with pytest.raises(ValueError, match="unknown"):
            ResponseCacheSettingsModel.model_validate({"operations": {"video": {}}})

    def test_off_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _PricedEngine("p")
        assert engine.send_prompt("hi") == "p:ok"
        assert engine.send_prompt("hi") == "p:ok"
        assert engine.calls.count("send_prompt") == 2


class TestKey:
    def test_identical_requests_share_a_key_and_whitespace_is_trimmed(self) -> None:
        engine = _PricedEngine("p")
        k1 = build_cache_key(engine, "completions", "send_prompt", {"prompt": "hi"})
        k2 = build_cache_key(engine, "completions", "send_prompt", {"prompt": "  hi\n"})
        assert k1 == k2 and k1.startswith("rc:v1:")

    def test_case_model_options_and_client_change_the_key(self) -> None:
        a, b = _PricedEngine("p"), _PricedEngine("q", model="other")
        base = build_cache_key(a, "completions", "send_prompt", {"prompt": "hi"})
        assert base != build_cache_key(
            a, "completions", "send_prompt", {"prompt": "HI"}
        )
        assert base != build_cache_key(
            b, "completions", "send_prompt", {"prompt": "hi"}
        )
        assert base != build_cache_key(
            a, "completions", "send_prompt", {"prompt": "hi", "max_response_tokens": 5}
        )

    def test_labels_and_reserved_keys_do_not_change_the_key(self) -> None:
        engine = _TemplateEngine("p")
        k1 = build_cache_key(
            engine,
            "conversation",
            "send_conversation",
            {"messages": MESSAGES, "provider_options": {"temperature": 0.2}},
        )
        k2 = build_cache_key(
            engine,
            "conversation",
            "send_conversation",
            {
                "messages": MESSAGES,
                "provider_options": {
                    "temperature": 0.2,
                    "cache": "bypass",
                    "purpose": "x",
                },
                "request_timeout_seconds": 9,
            },
        )
        assert k1 == k2
        engine.set_config_identity("cfg-1")
        assert (
            build_cache_key(
                engine, "conversation", "send_conversation", {"messages": MESSAGES}
            )
            != k1
        )


class TestHitsAndMisses:
    def test_send_prompt_hit_skips_the_provider_and_is_metered(
        self, cache_on: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        cache_on()
        engine = _PricedEngine("p")
        assert engine.send_prompt("hi") == "p:ok"
        with caplog.at_level(logging.INFO):
            assert engine.send_prompt("hi") == "p:ok"
        assert engine.calls.count("send_prompt") == 1
        report = engine.last_request_cost
        assert report.cache_hit and report.total_usd == 0
        assert report.usd_saved == GPT_COST
        assert any(r.message.startswith("CACHE HIT") for r in caplog.records)

    def test_send_prompt_with_result_hit_carries_the_original_envelope(
        self, cache_on: Any
    ) -> None:
        cache_on()
        chain = _chain(
            _PricedEngine("p", fail=_err(R.UNAVAILABLE, 529)), _PricedEngine("f")
        )
        first = chain.send_prompt_with_result("hi")
        second = chain.send_prompt_with_result("hi")
        assert not first.cache_hit and second.cache_hit
        assert second.text == first.text == "f:ok"
        assert second.model_name == first.model_name
        assert second.usd_saved == first.cost.total_usd == GPT_COST
        assert second.cost.total_usd == 0
        # The async twin shares the entry.
        assert asyncio.run(chain.asend_prompt_with_result("hi")).cache_hit

    def test_different_prompt_misses(self, cache_on: Any) -> None:
        cache_on()
        engine = _PricedEngine("p")
        engine.send_prompt("hi")
        engine.send_prompt("hello")
        assert engine.calls.count("send_prompt") == 2

    def test_conversation_hit_returns_the_envelope(self, cache_on: Any) -> None:
        cache_on()
        engine = _TemplateEngine("p")
        first = engine.send_conversation("sys", MESSAGES)
        second = engine.send_conversation("sys", MESSAGES)
        assert isinstance(second, AITurnResult) and second.text == first.text
        assert not first.cache_hit and second.cache_hit
        assert second.usd_saved == GPT_COST and second.cost.total_usd == 0
        assert second.provider == first.provider == "openai"
        assert engine.calls.count("send_conversation") == 1

    def test_structured_output_hit(self, cache_on: Any) -> None:
        cache_on()
        engine = _TemplateEngine("p")
        schema = {"type": "object"}
        engine.send_structured_output("q", response_schema=schema)
        again = engine.send_structured_output("q", response_schema=schema)
        assert again.cache_hit and again.data == {"a": "p"}
        assert engine.calls.count("send_structured_output") == 1

    def test_embeddings_and_classification(self, cache_on: Any) -> None:
        cache_on()
        emb = _FakeEmbeddings()
        assert emb.generate_embeddings("x") == emb.generate_embeddings("x")
        assert emb.calls == 1
        emb.generate_embeddings_batch(["a", "b"])
        emb.generate_embeddings_batch(["a", "b"])
        assert emb.calls == 2
        cls = _FakeClassifier()
        choices = [AIClassificationChoice(value="a"), AIClassificationChoice(value="b")]
        cls.classify_choice("t", choices)
        hit = cls.classify_choice("t", choices)
        assert cls.calls == 1 and hit.cache_hit

    def test_chain_caches_once_at_the_outermost_client(self, cache_on: Any) -> None:
        cache_on()
        primary = _PricedEngine("p", fail=_err(R.UNAVAILABLE, 529))
        fallback = _PricedEngine("f")
        chain = _chain(primary, fallback)
        assert chain.send_prompt("hi") == "f:ok"
        assert chain.send_prompt("hi") == "f:ok"
        assert primary.calls.count("send_prompt") == 1
        assert fallback.calls.count("send_prompt") == 1
        assert RESPONSE_CACHE_ACTIVE.get() is False

    def test_hit_inside_a_callers_scope_is_a_zero_cost_line(
        self, cache_on: Any
    ) -> None:
        cache_on()
        engine = _PricedEngine("p")
        engine.send_prompt("hi")
        with request_cost_scope("step") as scope:
            engine.send_prompt("hi")
        assert scope.report.total_usd == 0 and scope.report.calls == []

    def test_async_twin(self, cache_on: Any) -> None:
        cache_on()
        engine = _PricedEngine("p")
        asyncio.run(engine.asend_prompt("hi"))
        asyncio.run(engine.asend_prompt("hi"))
        assert engine.calls.count("asend_prompt") == 1

    def test_entries_expire(self, cache_on: Any) -> None:
        cache_on({"ttl_seconds": 1})
        clock: list[float] = [0.0]
        set_cache_store(InMemoryCacheStore(callable_now=lambda: clock[0]))
        engine = _PricedEngine("p")
        engine.send_prompt("hi")
        clock[0] = 2.0
        engine.send_prompt("hi")
        assert engine.calls.count("send_prompt") == 2


class TestOptingOut:
    def test_operation_disabled_in_config(self, cache_on: Any) -> None:
        cache_on({"operations": {"completions": {"enabled": False}}})
        engine = _PricedEngine("p")
        engine.send_prompt("hi")
        engine.send_prompt("hi")
        assert engine.calls.count("send_prompt") == 2

    def test_provider_options_cache_off_and_bypass(self, cache_on: Any) -> None:
        cache_on()
        engine = _TemplateEngine("p")
        engine.send_conversation("sys", MESSAGES, provider_options={"cache": "off"})
        engine.send_conversation("sys", MESSAGES, provider_options={"cache": "off"})
        assert engine.calls.count("send_conversation") == 2
        # The reserved key never reaches the engine.
        assert "cache" not in (engine.last_kwargs.get("provider_options") or {})
        engine.send_conversation("sys", MESSAGES)  # stores
        engine.send_conversation("sys", MESSAGES, provider_options={"cache": "bypass"})
        assert engine.calls.count("send_conversation") == 4
        assert engine.send_conversation("sys", MESSAGES).cache_hit

    def test_other_params_cache_off(self, cache_on: Any) -> None:
        from ai_api_unified.ai_base import AICompletionsPromptParamsBase

        class _Params(AICompletionsPromptParamsBase):
            pass

        cache_on()
        engine = _PricedEngine("p")
        engine.send_prompt("hi", other_params=_Params(cache="off"))
        engine.send_prompt("hi", other_params=_Params(cache="off"))
        assert engine.calls.count("send_prompt") == 2

    def test_cache_mode_block(self, cache_on: Any) -> None:
        cache_on()
        engine = _PricedEngine("p")
        with cache_mode("off"):
            engine.send_prompt("hi")
            engine.send_prompt("hi")
        assert engine.calls.count("send_prompt") == 2
        with pytest.raises(ValueError):
            with cache_mode("sometimes"):
                pass

    def test_media_attachments_are_never_cached(self, cache_on: Any) -> None:
        cache_on()
        engine = _PricedEngine("p")

        class _Media:
            def has_included_media(self) -> bool:
                return True

        engine.send_prompt("hi", other_params=_Media())
        engine.send_prompt("hi", other_params=_Media())
        assert engine.calls.count("send_prompt") == 2


class TestFailsOpen:
    def test_a_broken_store_never_fails_a_call(self, cache_on: Any) -> None:
        cache_on()

        class _Broken:
            def get(self, key: str) -> bytes | None:
                raise RuntimeError("down")

            def set(self, key: str, value: bytes, ttl_seconds: float) -> None:
                raise RuntimeError("down")

            def delete(self, key: str) -> None:
                raise RuntimeError("down")

        set_cache_store(_Broken())
        engine = _PricedEngine("p")
        assert engine.send_prompt("hi") == "p:ok"
        assert engine.send_prompt("hi") == "p:ok"
        assert engine.calls.count("send_prompt") == 2

    def test_a_corrupt_entry_is_a_miss(self, cache_on: Any) -> None:
        cache_on()
        engine = _PricedEngine("p")
        engine.send_prompt("hi")
        store = get_cache_store()
        assert isinstance(store, InMemoryCacheStore)
        key = next(iter(store._entries))
        store.set(key, b"not json", 100)
        assert engine.send_prompt("hi") == "p:ok"
        assert engine.calls.count("send_prompt") == 2


class TestCooldownsShareTheStore:
    def test_benches_live_in_the_cache_store(self) -> None:
        from ai_api_unified.completions.ai_fallback_cooldown import AiCandidateCooldowns

        mine = InMemoryCacheStore()
        set_cache_store(mine)
        AiCandidateCooldowns.bench(
            str_engine="fake",
            str_model="m",
            base_url=None,
            reason=R.RATE_LIMITED,
            bool_account_wide=False,
            float_seconds=30,
            str_label="fake:m",
        )
        assert any(k.startswith("cd:v1:cand") for k in mine._entries)
        rest = AiCandidateCooldowns.resting(
            str_engine="fake", str_model="m", base_url=None
        )
        assert rest is not None and rest.reason is R.RATE_LIMITED
        AiCandidateCooldowns.clear(
            str_engine="fake", str_model="m", base_url=None, str_label="fake:m"
        )
        assert (
            AiCandidateCooldowns.resting(
                str_engine="fake", str_model="m", base_url=None
            )
            is None
        )


def test_isinstance_check_accepts_duck_typed_stores() -> None:
    class _Store:
        def get(self, key: str) -> bytes | None:
            return None

        def set(self, key: str, value: bytes, ttl_seconds: float) -> None:
            return None

        def delete(self, key: str) -> None:
            return None

    set_cache_store(_Store())
    assert get_cache_store().get("x") is None
    assert Decimal("0") == Decimal(0)
