# tests/test_thinking_levels_nonmock.py
"""
Live thinking levels: the configured level reaches a real provider, a
per-call override changes it, and the result records what was applied.

Run with:

    poetry run pytest -m nonmock tests/test_thinking_levels_nonmock.py -q
"""

from __future__ import annotations

import os
import socket

import pytest

from ai_api_unified import (
    AIFactory,
    AIStructuredPrompt,
    AIThinkingLevel,
    AiProviderRequestError,
)

pytestmark = pytest.mark.nonmock

STR_PROMPT: str = "Reply with the single word: pong"


class _Word(AIStructuredPrompt):
    word: str

    @staticmethod
    def get_prompt(input_text: str = "") -> str:
        return input_text


def _skip_without(str_key: str, str_host: str) -> None:
    if str_key not in os.environ:
        pytest.skip(f"Skipping: {str_key} not set")
    try:
        socket.gethostbyname(str_host)
    except OSError as exception:
        pytest.skip(f"Skipping: DNS unavailable for {str_host}: {exception}")


def _skip_on_account(exception: AiProviderRequestError) -> None:
    if exception.fallback_reason is not None and exception.fallback_reason.value in (
        "quota_exhausted",
        "auth_rejected",
    ):
        pytest.skip(f"Skipping: provider account cannot serve ({exception})")
    raise exception


@pytest.mark.parametrize(
    "str_engine, str_model, str_key, str_host, level_off_expected",
    [
        (
            "claude",
            "claude-haiku-4-5",
            "ANTHROPIC_API_KEY",
            "api.anthropic.com",
            AIThinkingLevel.OFF,
        ),
        (
            "google-gemini",
            "gemini-2.5-flash",
            "GOOGLE_GEMINI_API_KEY",
            "generativelanguage.googleapis.com",
            AIThinkingLevel.OFF,
        ),
        (
            "openai",
            "gpt-5.4-mini",
            "OPENAI_API_KEY",
            "api.openai.com",
            AIThinkingLevel.OFF,
        ),
    ],
)
def test_configured_and_per_call_levels_reach_the_provider(
    monkeypatch: pytest.MonkeyPatch,
    str_engine: str,
    str_model: str,
    str_key: str,
    str_host: str,
    level_off_expected: AIThinkingLevel,
) -> None:
    _skip_without(str_key, str_host)
    client = AIFactory.get_ai_completions_client(
        completions_engine=str_engine, model_name=str_model, thinking="low"
    )
    assert client.thinking_level is AIThinkingLevel.LOW
    try:
        text = client.send_prompt(STR_PROMPT, max_response_tokens=4096)
    except AiProviderRequestError as exception:
        _skip_on_account(exception)
    assert text and "pong" in text.lower()
    assert client.last_thinking is not None
    assert client.last_thinking.requested is AIThinkingLevel.LOW
    assert client.last_thinking.applied is AIThinkingLevel.LOW
    # A structured call at high, then off, each recorded on the result.
    result = client.send_structured_output(
        "Answer with the word pong.",
        response_model=_Word,
        provider_options={"thinking_level": "high"},
    )
    assert result.data == {"word": "pong"} or result.data["word"].lower() == "pong"
    assert result.thinking is not None
    assert result.thinking.requested is AIThinkingLevel.HIGH
    result_off = client.send_structured_output(
        "Answer with the word pong.",
        response_model=_Word,
        provider_options={"thinking_level": "off"},
    )
    assert result_off.thinking is not None
    assert result_off.thinking.applied is level_off_expected


def test_always_on_model_applies_nearest_level_and_records_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _skip_without("ANTHROPIC_API_KEY", "api.anthropic.com")
    client = AIFactory.get_ai_completions_client(
        completions_engine="claude", model_name="claude-opus-5-5", thinking="off"
    )
    try:
        text = client.send_prompt(STR_PROMPT, max_response_tokens=4096)
    except AiProviderRequestError as exception:
        _skip_on_account(exception)
    assert text and "pong" in text.lower()
    resolution = client.last_thinking
    assert resolution is not None
    assert resolution.requested is AIThinkingLevel.OFF
    assert resolution.applied is AIThinkingLevel.LOW
    assert not resolution.exact and resolution.reason == "unsupported_level"
