# ruff: noqa: E402

# test_openai_responses_completions.py
"""
Tests for the OpenAI Responses API completions engine.

Covers factory resolution, text send_prompt, structured output, streaming,
and the text-only capability restriction, all against a mocked SDK client.
"""

import os
from typing import Any
from unittest.mock import Mock, patch

import pytest

pytest.importorskip("openai")

from ai_api_unified.ai_base import AIStructuredPrompt, SupportedDataType
from ai_api_unified.ai_factory import AIFactory
from ai_api_unified.completions.ai_openai_responses_completions import (
    AiOpenAIResponsesCompletions,
)


def _build_client() -> AiOpenAIResponsesCompletions:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        client = AiOpenAIResponsesCompletions(model="gpt-4o-mini")
    client.client = Mock()
    return client


def test_factory_resolves_responses_engine() -> None:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        client = AIFactory.get_ai_completions_client(
            completions_engine="openai-responses", model_name="gpt-4o-mini"
        )
    assert isinstance(client, AiOpenAIResponsesCompletions)


def test_capabilities_are_text_only_with_streaming() -> None:
    client = _build_client()
    capabilities = client.capabilities
    assert capabilities.supported_data_types == [SupportedDataType.TEXT]
    assert capabilities.supports_streaming is True


def test_send_prompt_uses_responses_create() -> None:
    client = _build_client()
    client.client.responses.create.return_value = Mock(
        output_text="Hello there",
        status="completed",
        usage=Mock(input_tokens=5, output_tokens=2, total_tokens=7),
    )

    result: str = client.send_prompt("Greet me")

    assert result == "Hello there"
    create_kwargs = client.client.responses.create.call_args.kwargs
    assert create_kwargs["model"] == "gpt-4o-mini"
    assert create_kwargs["input"] == "Greet me"
    assert "instructions" in create_kwargs


def test_send_prompt_streaming_yields_text_deltas() -> None:
    client = _build_client()
    client.client.responses.create.return_value = iter(
        [
            Mock(type="response.output_text.delta", delta="Hel"),
            Mock(type="response.output_text.delta", delta="lo"),
            Mock(
                type="response.completed", response=Mock(status="completed", usage=None)
            ),
        ]
    )

    list_chunks: list[str] = list(client.send_prompt_streaming("Greet me"))

    assert list_chunks == ["Hel", "lo"]
    create_kwargs = client.client.responses.create.call_args.kwargs
    assert create_kwargs["stream"] is True


class _Person(AIStructuredPrompt):
    name: str | None = None
    age: int | None = None

    @staticmethod
    def get_prompt(input_text: str = "") -> str:
        return f"Extract name and age from: {input_text}"


def test_strict_schema_prompt_parses_json_output() -> None:
    client = _build_client()
    client.client.responses.create.return_value = Mock(
        output_text='{"name": "Alice", "age": 30}',
        status="completed",
        usage=Mock(input_tokens=10, output_tokens=8, total_tokens=18),
    )

    result: Any = client.strict_schema_prompt("Extract", _Person)

    assert result.name == "Alice"
    assert result.age == 30
    create_kwargs = client.client.responses.create.call_args.kwargs
    assert create_kwargs["text"]["format"]["type"] == "json_schema"


def test_strict_schema_prompt_raises_on_invalid_json() -> None:
    client = _build_client()
    client.client.responses.create.return_value = Mock(
        output_text="not json",
        status="completed",
        usage=None,
    )

    with pytest.raises(ValueError, match="Invalid JSON response"):
        client.strict_schema_prompt("Extract", _Person)
