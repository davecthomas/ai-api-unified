# ruff: noqa: E402
# test_model_fallback.py
"""
Tests for AiFallbackCompletions and the factory's fallback chain.

Most cases use a small fake engine so the chain mechanics (eligibility,
lazy construction, capability skips, conversation stickiness, streaming
cut-off, per-call opt-out) are tested without SDK doubles. One end-to-end
case runs the real claude and openai engines with mocked SDK clients, so
the error classification from the engines feeds the wrapper as in
production.
"""

import asyncio
import os
from collections.abc import Iterator
from typing import Any, Type
from unittest.mock import Mock, patch

import httpx
import pytest

from ai_api_unified import (
    DEFAULT_FALLBACK_REASONS,
    AIFactory,
    AIFallbackCandidate,
    AiFallbackCompletions,
    AiFallbackReason,
    AiProviderRequestError,
)
from ai_api_unified.ai_base import (
    AIBaseCompletions,
    AICompletionsCapabilitiesBase,
    AICompletionsPromptParamsBase,
    AIFinishReason,
    AIStructuredOutputResult,
    AIStructuredPrompt,
    AITurnResult,
)
from ai_api_unified.ai_provider_exceptions import (
    AiProviderCapabilityUnsupportedError,
    AiProviderDependencyUnavailableError,
)
from ai_api_unified.completions.ai_fallback_completions import (
    parse_fallback_candidates,
    parse_fallback_reasons,
)

R = AiFallbackReason


class _Answer(AIStructuredPrompt):
    answer: str

    @staticmethod
    def get_prompt() -> str:
        return "hi"


def _err(reason: AiFallbackReason | None, status: int | None = 503) -> Exception:
    return AiProviderRequestError(
        f"fail:{reason}", status_code=status, fallback_reason=reason
    )


def _caps(**overrides: Any) -> AICompletionsCapabilitiesBase:
    dict_fields: dict[str, Any] = {
        "context_window_length": 1000,
        "supports_streaming": True,
        "supports_tool_use": True,
        "supports_structured_output": True,
        "supports_async": True,
    }
    dict_fields.update(overrides)
    return AICompletionsCapabilitiesBase(**dict_fields)


class _FakeEngine(AIBaseCompletions):
    """One candidate: returns its name, or raises `fail` on every call."""

    def __init__(
        self,
        name: str,
        *,
        fail: Exception | None = None,
        capabilities: AICompletionsCapabilitiesBase | None = None,
        chunks: list[str] | None = None,
        fail_after_first_chunk: bool = False,
        family: str = "anthropic",
    ) -> None:
        super().__init__(model=name)
        self.fail = fail
        self._caps = capabilities or _caps()
        self.chunks = chunks if chunks is not None else [f"{name}-1", f"{name}-2"]
        self.fail_after_first_chunk = fail_after_first_chunk
        # Read by engine_family_of, so fakes can stand in for engine families.
        self.FALLBACK_ENGINE_FAMILY = family
        self.calls: list[str] = []
        self.last_kwargs: dict[str, Any] = {}

    def _maybe_fail(self, op: str, **kwargs: Any) -> None:
        self.calls.append(op)
        self.last_kwargs = kwargs
        if op.startswith("a") and not self._caps.supports_async:
            raise AiProviderCapabilityUnsupportedError(f"{self.model}: no async")
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
        self._maybe_fail("send_prompt")
        return f"{self.model}:ok"

    async def asend_prompt(self, prompt: str, **kwargs: Any) -> str:
        self._maybe_fail("asend_prompt")
        return f"{self.model}:ok"

    def send_prompt_streaming(
        self, prompt: str, *, other_params: Any = None
    ) -> Iterator[str]:
        def _gen() -> Iterator[str]:
            self.calls.append("send_prompt_streaming")
            if self.fail is not None and not self.fail_after_first_chunk:
                raise self.fail
            yield self.chunks[0]
            if self.fail is not None:
                raise self.fail
            yield from self.chunks[1:]

        return _gen()

    def count_tokens(self, prompt: str, *, other_params: Any = None) -> int:
        self.calls.append("count_tokens")
        return 42

    def strict_schema_prompt(
        self,
        prompt: str,
        response_model: Type[AIStructuredPrompt],
        max_response_tokens: int = 2048,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> AIStructuredPrompt:
        self._maybe_fail("strict_schema_prompt")
        return response_model(answer=str(self.model))

    def send_structured_output(self, prompt: Any = None, **kwargs: Any) -> Any:
        self._maybe_fail("send_structured_output", **kwargs)
        return AIStructuredOutputResult(
            data={"answer": str(self.model)}, finish_reason=AIFinishReason.COMPLETE
        )

    async def asend_structured_output(self, prompt: Any = None, **kwargs: Any) -> Any:
        self._maybe_fail("asend_structured_output", **kwargs)
        return AIStructuredOutputResult(
            data={"answer": str(self.model)}, finish_reason=AIFinishReason.COMPLETE
        )

    def send_conversation(
        self, system_prompt: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AITurnResult:
        self._maybe_fail("send_conversation", **kwargs)
        return AITurnResult(
            text=f"{self.model}:turn",
            finish_reason=AIFinishReason.COMPLETE,
            raw_content=[{"type": "text", "text": "x"}],
        )

    async def asend_conversation(
        self, system_prompt: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AITurnResult:
        self._maybe_fail("asend_conversation", **kwargs)
        return AITurnResult(
            text=f"{self.model}:turn",
            finish_reason=AIFinishReason.COMPLETE,
            raw_content=[{"type": "text", "text": "x"}],
        )

    def build_tool_result_message(self, **kwargs: Any) -> dict[str, Any]:
        if self.FALLBACK_ENGINE_FAMILY == "openai":
            return {"role": "tool", "tool_call_id": "t", "engine": str(self.model)}
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t"}],
            "engine": str(self.model),
        }

    def extend_messages_with_turn(
        self, messages: list[dict[str, Any]], turn: AITurnResult
    ) -> list[dict[str, Any]]:
        # Append in this family's shape so routing can read it back.
        if self.FALLBACK_ENGINE_FAMILY == "openai":
            messages.append({"role": "assistant", "tool_calls": [{"id": "t"}]})
        else:
            messages.append({"role": "assistant", "content": turn.raw_content})
        return messages


def _chain(*engines: _FakeEngine, **kwargs: Any) -> AiFallbackCompletions:
    """Builds a wrapper whose builder hands out the given engines in order."""
    list_rest: list[_FakeEngine] = list(engines[1:])
    dict_by_label: dict[str, _FakeEngine] = {
        f"fake:{engine.model}": engine for engine in list_rest
    }
    builder: Mock = Mock(side_effect=lambda c: dict_by_label[c.label])
    wrapper = AiFallbackCompletions(
        primary=engines[0],
        primary_candidate=AIFallbackCandidate(
            engine="fake", model=str(engines[0].model)
        ),
        fallback_candidates=[
            AIFallbackCandidate(engine="fake", model=str(engine.model))
            for engine in list_rest
        ],
        client_builder=builder,
        **kwargs,
    )
    wrapper.builder = builder  # type: ignore[attr-defined]
    return wrapper


# ── Parsing and defaults ────────────────────────────────────────────────────


class TestParsing:
    def test_candidates_split_on_first_colon_only(self) -> None:
        list_candidates = parse_fallback_candidates(
            "openai:gpt-5.6-luna, bedrock:amazon.nova-lite-v1:0 ,google-gemini"
        )
        assert [c.engine for c in list_candidates] == [
            "openai",
            "bedrock",
            "google-gemini",
        ]
        assert list_candidates[1].model == "amazon.nova-lite-v1:0"
        assert list_candidates[2].model == ""

    def test_blank_setting_means_no_chain(self) -> None:
        assert parse_fallback_candidates("  ") == []

    def test_reasons_parse_values_and_members(self) -> None:
        assert parse_fallback_reasons("unavailable, QUOTA_EXHAUSTED") == {
            R.UNAVAILABLE,
            R.QUOTA_EXHAUSTED,
        }
        assert parse_fallback_reasons([R.RATE_LIMITED, "unavailable"]) == {
            R.RATE_LIMITED,
            R.UNAVAILABLE,
        }

    def test_unknown_reason_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="bogus"):
            parse_fallback_reasons("bogus")

    def test_default_reasons_exclude_model_unavailable(self) -> None:
        assert DEFAULT_FALLBACK_REASONS == {
            R.UNAVAILABLE,
            R.RATE_LIMITED,
            R.QUOTA_EXHAUSTED,
        }


# ── Chain mechanics ─────────────────────────────────────────────────────────


class TestFailover:
    def test_primary_success_never_builds_a_fallback(self) -> None:
        wrapper = _chain(_FakeEngine("p"), _FakeEngine("f"))
        assert wrapper.send_prompt("hi") == "p:ok"
        wrapper.builder.assert_not_called()
        assert wrapper.last_route.model == "p"

    @pytest.mark.parametrize("reason", sorted(DEFAULT_FALLBACK_REASONS))
    def test_default_reasons_fail_over(self, reason: AiFallbackReason) -> None:
        fallback = _FakeEngine("f")
        wrapper = _chain(_FakeEngine("p", fail=_err(reason)), fallback)
        assert wrapper.send_prompt("hi") == "f:ok"
        assert wrapper.last_route.model == "f"

    def test_fallback_is_built_once(self) -> None:
        wrapper = _chain(_FakeEngine("p", fail=_err(R.UNAVAILABLE)), _FakeEngine("f"))
        wrapper.send_prompt("hi")
        wrapper.send_prompt("hi")
        assert wrapper.builder.call_count == 1

    def test_model_unavailable_does_not_fail_over_by_default(self) -> None:
        wrapper = _chain(
            _FakeEngine("p", fail=_err(R.MODEL_UNAVAILABLE, 404)), _FakeEngine("f")
        )
        with pytest.raises(AiProviderRequestError):
            wrapper.send_prompt("hi")
        wrapper.builder.assert_not_called()

    def test_model_unavailable_fails_over_when_configured(self) -> None:
        wrapper = _chain(
            _FakeEngine("p", fail=_err(R.MODEL_UNAVAILABLE, 404)),
            _FakeEngine("f"),
            fallback_on={R.MODEL_UNAVAILABLE},
        )
        assert wrapper.send_prompt("hi") == "f:ok"

    def test_error_without_reason_propagates(self) -> None:
        wrapper = _chain(_FakeEngine("p", fail=_err(None, 400)), _FakeEngine("f"))
        with pytest.raises(AiProviderRequestError):
            wrapper.send_prompt("hi")
        wrapper.builder.assert_not_called()

    def test_non_request_errors_propagate(self) -> None:
        wrapper = _chain(
            _FakeEngine("p", fail=ValueError("bad prompt")), _FakeEngine("f")
        )
        with pytest.raises(ValueError):
            wrapper.send_prompt("hi")
        wrapper.builder.assert_not_called()

    def test_all_candidates_failing_raises_the_last_error(self) -> None:
        wrapper = _chain(
            _FakeEngine("p", fail=_err(R.UNAVAILABLE)),
            _FakeEngine("f", fail=_err(R.QUOTA_EXHAUSTED, 429)),
        )
        with pytest.raises(AiProviderRequestError) as exc_info:
            wrapper.send_prompt("hi")
        assert exc_info.value.fallback_reason is R.QUOTA_EXHAUSTED

    def test_unbuildable_candidate_is_skipped_and_not_rebuilt(self) -> None:
        third = _FakeEngine("t")
        wrapper = _chain(
            _FakeEngine("p", fail=_err(R.UNAVAILABLE)), _FakeEngine("f"), third
        )
        dict_map = {"fake:t": third}
        wrapper.builder.side_effect = lambda c: dict_map.get(c.label) or (
            _ for _ in ()
        ).throw(AiProviderDependencyUnavailableError("no extra"))
        assert wrapper.send_prompt("hi") == "t:ok"
        wrapper.send_prompt("hi")
        # Built "f" once (failed) and "t" once.
        assert wrapper.builder.call_count == 2

    def test_candidate_missing_the_capability_is_skipped(self) -> None:
        wrapper = _chain(
            _FakeEngine("p", fail=_err(R.UNAVAILABLE)),
            _FakeEngine("f", capabilities=_caps(supports_structured_output=False)),
            _FakeEngine("t"),
        )
        result = wrapper.send_structured_output(
            "hi", response_schema={"type": "object"}
        )
        assert result.data == {"answer": "t"}
        assert result.provider_engine == "fake"
        assert result.model_name == "t"

    def test_strict_schema_prompt_fails_over(self) -> None:
        wrapper = _chain(
            _FakeEngine("p", fail=_err(R.RATE_LIMITED, 429)), _FakeEngine("f")
        )
        assert wrapper.strict_schema_prompt("hi", _Answer).answer == "f"

    def test_async_prompt_fails_over(self) -> None:
        wrapper = _chain(_FakeEngine("p", fail=_err(R.UNAVAILABLE)), _FakeEngine("f"))
        assert asyncio.run(wrapper.asend_prompt("hi")) == "f:ok"

    def test_async_structured_output_skips_a_candidate_without_it(self) -> None:
        wrapper = _chain(
            _FakeEngine("p", fail=_err(R.UNAVAILABLE)),
            _FakeEngine("f", capabilities=_caps(supports_structured_output=False)),
            _FakeEngine("t"),
        )
        result = asyncio.run(
            wrapper.asend_structured_output("hi", response_schema={"type": "object"})
        )
        assert result.data == {"answer": "t"}

    def test_any_build_error_skips_the_candidate_once(self) -> None:
        third = _FakeEngine("t")
        wrapper = _chain(
            _FakeEngine("p", fail=_err(R.UNAVAILABLE)), _FakeEngine("f"), third
        )
        dict_map = {"fake:t": third}

        def _build(candidate: AIFallbackCandidate) -> _FakeEngine:
            if candidate.label in dict_map:
                return dict_map[candidate.label]
            raise RuntimeError("constructor exploded")

        wrapper.builder.side_effect = _build
        assert wrapper.send_prompt("hi") == "t:ok"
        wrapper.send_prompt("hi")
        assert wrapper.builder.call_count == 2

    def test_cost_helpers_price_at_the_serving_candidate(self) -> None:
        fallback = _FakeEngine("f")
        fallback.compute_completion_cost = Mock(return_value=1.5)  # type: ignore[method-assign]
        wrapper = _chain(_FakeEngine("p", fail=_err(R.UNAVAILABLE)), fallback)
        wrapper.send_prompt("hi")
        assert wrapper.compute_completion_cost(input_tokens=10, output_tokens=5) == 1.5
        fallback.compute_completion_cost.assert_called_once_with(
            input_tokens=10,
            output_tokens=5,
            cached_input_tokens=0,
            cache_write_5m_tokens=0,
            cache_write_1h_tokens=0,
        )

    def test_org_info_is_delegated_to_the_primary(self) -> None:
        primary = _FakeEngine("p")
        primary.get_org_info = Mock(return_value="org")  # type: ignore[method-assign]
        wrapper = _chain(primary, _FakeEngine("f"))
        assert wrapper.get_org_info() == "org"

    def test_count_tokens_and_batches_stay_on_the_primary(self) -> None:
        primary = _FakeEngine("p", fail=_err(R.UNAVAILABLE))
        primary.submit_batch = Mock(return_value="job")  # type: ignore[method-assign]
        wrapper = _chain(primary, _FakeEngine("f"))
        assert wrapper.count_tokens("hi") == 42
        assert wrapper.submit_batch([]) == "job"
        wrapper.builder.assert_not_called()


# ── Per-call opt-out ────────────────────────────────────────────────────────


class TestPerCallOptOut:
    def test_provider_options_fallback_none_keeps_the_primary(self) -> None:
        wrapper = _chain(_FakeEngine("p", fail=_err(R.UNAVAILABLE)), _FakeEngine("f"))
        with pytest.raises(AiProviderRequestError):
            wrapper.send_conversation(
                "sys",
                [{"role": "user", "content": "hi"}],
                provider_options={"fallback": "none"},
            )
        wrapper.builder.assert_not_called()

    def test_reserved_key_never_reaches_the_engine(self) -> None:
        primary = _FakeEngine("p")
        wrapper = _chain(primary, _FakeEngine("f"))
        wrapper.send_conversation(
            "sys",
            [{"role": "user", "content": "hi"}],
            provider_options={"fallback": "none", "temperature": 0.1},
        )
        assert primary.last_kwargs["provider_options"] == {"temperature": 0.1}
        wrapper.send_conversation(
            "sys",
            [{"role": "user", "content": "hi"}],
            provider_options={"fallback": "none"},
        )
        assert primary.last_kwargs["provider_options"] is None


# ── Conversations ───────────────────────────────────────────────────────────


class TestConversations:
    def test_neutral_history_fails_over_and_stamps_the_route(self) -> None:
        wrapper = _chain(_FakeEngine("p", fail=_err(R.UNAVAILABLE)), _FakeEngine("f"))
        turn = wrapper.send_conversation("sys", [{"role": "user", "content": "hi"}])
        assert turn.text == "f:turn"
        assert turn.provider_engine == "fake"
        assert turn.model_name == "f"

    def test_engine_shaped_history_routes_to_its_family(self) -> None:
        primary = _FakeEngine("p", fail=_err(R.UNAVAILABLE), family="anthropic")
        fallback = _FakeEngine("f", family="openai")
        wrapper = _chain(primary, fallback)
        list_messages: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
        turn = wrapper.send_conversation("sys", list_messages)
        wrapper.extend_messages_with_turn(list_messages, turn)
        list_messages.append(
            wrapper.build_tool_result_message(
                tool_call_id="t", result={}, messages=list_messages
            )
        )
        assert list_messages[-1]["engine"] == "f"
        # The primary recovers, but the history is OpenAI-shaped, so it
        # stays on "f" with fallback off.
        primary.fail = None
        turn2 = wrapper.send_conversation("sys", list_messages)
        assert turn2.text == "f:turn"
        assert primary.calls.count("send_conversation") == 1

    def test_one_client_serves_conversations_on_different_engines(self) -> None:
        primary = _FakeEngine("p", family="anthropic")
        fallback = _FakeEngine("f", family="openai")
        wrapper = _chain(primary, fallback)
        # Conversation A fails over to "f" on its first turn.
        primary.fail = _err(R.UNAVAILABLE)
        list_a: list[dict[str, Any]] = [{"role": "user", "content": "a"}]
        wrapper.extend_messages_with_turn(
            list_a, wrapper.send_conversation("sys", list_a)
        )
        # Conversation B starts on a recovered "p".
        primary.fail = None
        list_b: list[dict[str, Any]] = [{"role": "user", "content": "b"}]
        wrapper.extend_messages_with_turn(
            list_b, wrapper.send_conversation("sys", list_b)
        )
        # Interleaved second turns each go to the engine that shaped them.
        assert wrapper.send_conversation("sys", list_a).text == "f:turn"
        assert wrapper.send_conversation("sys", list_b).text == "p:turn"
        assert (
            wrapper.build_tool_result_message(
                tool_call_id="t", result={}, messages=list_a
            )["engine"]
            == "f"
        )
        assert (
            wrapper.build_tool_result_message(
                tool_call_id="t", result={}, messages=list_b
            )["engine"]
            == "p"
        )

    def test_structured_output_routes_by_history_shape(self) -> None:
        primary = _FakeEngine("p", family="anthropic")
        fallback = _FakeEngine("f", family="openai")
        wrapper = _chain(primary, fallback)
        primary.fail = _err(R.UNAVAILABLE)
        list_messages: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
        wrapper.extend_messages_with_turn(
            list_messages, wrapper.send_conversation("sys", list_messages)
        )
        primary.fail = None
        result = wrapper.send_structured_output(
            "summarize", response_schema={"type": "object"}, messages=list_messages
        )
        assert result.data == {"answer": "f"}
        assert "send_structured_output" not in primary.calls

    def test_pinned_candidate_without_async_raises_its_own_capability_error(
        self,
    ) -> None:
        primary = _FakeEngine("p", fail=_err(R.UNAVAILABLE), family="anthropic")
        fallback = _FakeEngine(
            "f", family="openai", capabilities=_caps(supports_async=False)
        )
        wrapper = _chain(primary, fallback)
        list_messages: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
        wrapper.extend_messages_with_turn(
            list_messages, wrapper.send_conversation("sys", list_messages)
        )
        with pytest.raises(AiProviderCapabilityUnsupportedError):
            asyncio.run(wrapper.asend_conversation("sys", list_messages))

    def test_engine_shaped_history_does_not_fail_over(self) -> None:
        wrapper = _chain(_FakeEngine("p", fail=_err(R.UNAVAILABLE)), _FakeEngine("f"))
        list_messages: list[dict[str, Any]] = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "x"}]},
            {"role": "user", "content": "more"},
        ]
        with pytest.raises(AiProviderRequestError):
            wrapper.send_conversation("sys", list_messages)
        wrapper.builder.assert_not_called()

    def test_async_conversation_fails_over(self) -> None:
        wrapper = _chain(_FakeEngine("p", fail=_err(R.UNAVAILABLE)), _FakeEngine("f"))
        turn = asyncio.run(
            wrapper.asend_conversation("sys", [{"role": "user", "content": "hi"}])
        )
        assert turn.text == "f:turn"


# ── Streaming ───────────────────────────────────────────────────────────────


class TestStreaming:
    def test_failure_before_first_chunk_fails_over(self) -> None:
        wrapper = _chain(_FakeEngine("p", fail=_err(R.UNAVAILABLE)), _FakeEngine("f"))
        assert list(wrapper.send_prompt_streaming("hi")) == ["f-1", "f-2"]
        assert wrapper.last_route.model == "f"

    def test_failure_after_first_chunk_propagates(self) -> None:
        wrapper = _chain(
            _FakeEngine("p", fail=_err(R.UNAVAILABLE), fail_after_first_chunk=True),
            _FakeEngine("f"),
        )
        iterator = wrapper.send_prompt_streaming("hi")
        assert next(iterator) == "p-1"
        with pytest.raises(AiProviderRequestError):
            next(iterator)
        wrapper.builder.assert_not_called()

    def test_real_openai_stream_error_fails_over(self) -> None:
        openai = pytest.importorskip("openai")
        from ai_api_unified.completions.ai_openai_completions import (
            AiOpenAICompletions,
        )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            primary = AiOpenAICompletions(model="gpt-5.1")
        primary.client = Mock()
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        primary.client.chat.completions.create.side_effect = openai.APIStatusError(
            "overloaded",
            response=httpx.Response(503, request=request),
            body={"error": {"code": None, "message": "overloaded"}},
        )
        fallback = _FakeEngine("f")
        wrapper = AiFallbackCompletions(
            primary=primary,
            primary_candidate=AIFallbackCandidate(engine="openai", model="gpt-5.1"),
            fallback_candidates=[AIFallbackCandidate(engine="fake", model="f")],
            client_builder=lambda candidate: fallback,
        )
        assert list(wrapper.send_prompt_streaming("hi")) == ["f-1", "f-2"]
        assert wrapper.last_route.engine == "fake"


# ── Factory ─────────────────────────────────────────────────────────────────


def _patch_settings(dict_settings: dict[str, str]) -> Any:
    return patch(
        "ai_api_unified.util.env_settings.EnvSettings.get_setting",
        side_effect=lambda key, default=None: dict_settings.get(key, default),
    )


BASE_SETTINGS: dict[str, str] = {
    "COMPLETIONS_ENGINE": "claude",
    "COMPLETIONS_MODEL_NAME": "claude-opus-4-8",
    "ANTHROPIC_API_KEY": "test-key",
    "OPENAI_API_KEY": "test-key",
}


class TestFactory:
    def test_no_chain_returns_the_plain_engine(self) -> None:
        pytest.importorskip("anthropic")
        from ai_api_unified.completions.ai_anthropic_completions import (
            AiAnthropicCompletions,
        )

        with _patch_settings(BASE_SETTINGS), patch.dict(os.environ, BASE_SETTINGS):
            client = AIFactory.get_ai_completions_client()
        assert isinstance(client, AiAnthropicCompletions)

    def test_setting_builds_a_lazy_chain(self) -> None:
        pytest.importorskip("anthropic")
        with (
            _patch_settings(
                {**BASE_SETTINGS, "COMPLETIONS_FALLBACKS": "openai:gpt-5.1"}
            ),
            patch.dict(os.environ, BASE_SETTINGS),
        ):
            client = AIFactory.get_ai_completions_client()
        assert isinstance(client, AiFallbackCompletions)
        assert [c.label for c in client.candidates] == [
            "claude:claude-opus-4-8",
            "openai:gpt-5.1",
        ]
        assert client.fallback_on == DEFAULT_FALLBACK_REASONS
        assert list(client._clients) == [0]

    def test_empty_argument_overrides_the_setting(self) -> None:
        pytest.importorskip("anthropic")
        with (
            _patch_settings(
                {**BASE_SETTINGS, "COMPLETIONS_FALLBACKS": "openai:gpt-5.1"}
            ),
            patch.dict(os.environ, BASE_SETTINGS),
        ):
            client = AIFactory.get_ai_completions_client(fallbacks=[])
        assert not isinstance(client, AiFallbackCompletions)

    def test_unknown_fallback_engine_fails_at_startup(self) -> None:
        with _patch_settings(BASE_SETTINGS), patch.dict(os.environ, BASE_SETTINGS):
            with pytest.raises(ValueError, match="Unsupported COMPLETIONS engine"):
                AIFactory.get_ai_completions_client(fallbacks=[("nope", "x")])

    def test_unknown_reason_setting_fails_at_startup(self) -> None:
        pytest.importorskip("anthropic")
        with (
            _patch_settings(
                {
                    **BASE_SETTINGS,
                    "COMPLETIONS_FALLBACKS": "openai:gpt-5.1",
                    "COMPLETIONS_FALLBACK_ON": "bogus",
                }
            ),
            patch.dict(os.environ, BASE_SETTINGS),
        ):
            with pytest.raises(ValueError, match="bogus"):
                AIFactory.get_ai_completions_client()

    def test_end_to_end_claude_overloaded_serves_from_openai(self) -> None:
        anthropic = pytest.importorskip("anthropic")
        pytest.importorskip("openai")
        with _patch_settings(BASE_SETTINGS), patch.dict(os.environ, BASE_SETTINGS):
            client = AIFactory.get_ai_completions_client(
                fallbacks=[("openai", "gpt-5.1")]
            )
        assert isinstance(client, AiFallbackCompletions)
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        client.primary.client = Mock()
        client.primary.client.messages.create.side_effect = anthropic.APIStatusError(
            "Overloaded",
            response=httpx.Response(529, request=request),
            body={"error": {"type": "overloaded_error", "message": "Overloaded"}},
        )
        # Build the OpenAI fallback through the factory, then mock its SDK.
        openai_client = client._client_at(1)
        assert openai_client is not None
        message = Mock(spec=["content", "tool_calls", "refusal", "model_dump"])
        message.content = "from openai"
        message.tool_calls = None
        message.refusal = None
        openai_client.client = Mock()
        openai_client.client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=message, finish_reason="stop")],
            usage=Mock(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
                prompt_tokens_details=Mock(cached_tokens=None),
            ),
        )
        assert client.send_prompt("hi") == "from openai"
        assert client.last_route.engine == "openai"
