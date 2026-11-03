# test_gemini_thought_signature.py
"""
Gemini tool-use turns carry a thought_signature as raw bytes. The turn's
replay parts must stay JSON-safe so the response cache and HTTP callers can
store them, and the signature must survive the round trip back to Gemini.
A result the cache cannot serialize is skipped, never raised, because the
provider has already billed the call.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from google.genai import types as genai_types

from ai_api_unified.ai_base import AIFinishReason, AIToolCall, AITurnResult
from ai_api_unified.completions.ai_google_gemini_completions import (
    GoogleGeminiCompletions,
)
from ai_api_unified.middleware.response_cache import reset_response_cache
from tests.test_turn_result_envelope import MESSAGES, _TemplateEngine

BYTES_SIGNATURE: bytes = b"\x89\xff\x00signature"


@pytest.fixture
def cache_on(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Turns the response cache on through models.json."""
    monkeypatch.setenv(
        "AI_MODELS_CONFIG",
        json.dumps({"middleware": [{"name": "response_cache", "enabled": True}]}),
    )
    from ai_api_unified.ai_models_config import clear_models_config_cache

    clear_models_config_cache()
    reset_response_cache()
    yield
    reset_response_cache()


def _tool_part() -> genai_types.Part:
    return genai_types.Part(
        function_call=genai_types.FunctionCall(name="write", args={"path": "a"}),
        thought_signature=BYTES_SIGNATURE,
    )


class _TurnEngine(_TemplateEngine):
    """Returns a tool-use turn whose replay parts are set per test."""

    def __init__(self, raw_content: Any) -> None:
        super().__init__("p")
        self.raw_content = raw_content

    def _send_conversation_provider(self, **kwargs: Any) -> AITurnResult:
        return self._observed(
            "send_conversation",
            AITurnResult(
                tool_calls=[AIToolCall(id="write", name="write", input={})],
                finish_reason=AIFinishReason.TOOL_USE,
                raw_content=self.raw_content,
            ),
        )


def test_serialized_part_is_json_safe_and_replays_the_signature() -> None:
    dict_part: dict[str, Any] = GoogleGeminiCompletions._serialize_gemini_part(
        _tool_part()
    )

    assert isinstance(dict_part["thought_signature"], str)
    json.dumps(dict_part)
    replayed = genai_types.Content.model_validate(
        {"role": "model", "parts": [dict_part]}
    )
    assert replayed.parts[0].thought_signature == BYTES_SIGNATURE
    assert replayed.parts[0].function_call.name == "write"


def test_gemini_turn_is_cached_and_the_hit_replays_the_signature(
    cache_on: Any,
) -> None:
    dict_part: dict[str, Any] = GoogleGeminiCompletions._serialize_gemini_part(
        _tool_part()
    )
    engine = _TurnEngine([dict_part])

    engine.send_conversation("sys", MESSAGES)
    hit: AITurnResult = engine.send_conversation("sys", MESSAGES)

    assert hit.cache_hit
    assert engine.calls.count("send_conversation") == 1
    replayed = genai_types.Content.model_validate(
        {"role": "model", "parts": hit.raw_content}
    )
    assert replayed.parts[0].thought_signature == BYTES_SIGNATURE


def test_an_unserializable_result_is_returned_and_not_stored(
    cache_on: Any,
) -> None:
    engine = _TurnEngine([{"thought_signature": BYTES_SIGNATURE}])

    first: AITurnResult = engine.send_conversation("sys", MESSAGES)
    second: AITurnResult = engine.send_conversation("sys", MESSAGES)

    assert first.raw_content == [{"thought_signature": BYTES_SIGNATURE}]
    assert not second.cache_hit
    assert engine.calls.count("send_conversation") == 2
