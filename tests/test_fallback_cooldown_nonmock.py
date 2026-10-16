# tests/test_fallback_cooldown_nonmock.py
"""
Live proof that a provider outage at startup does not stop the client, and
that the outage then benches the primary instead of being retried per call.

The Gemini primary points at a closed local port, so every request to it
fails with a refused connection, the way a down provider does. The client
must build, serve the first call from the Claude fallback, and serve the
next calls from Claude without touching Gemini while it rests.

    poetry run pytest -m nonmock tests/test_fallback_cooldown_nonmock.py -q -s
"""

from __future__ import annotations

import logging
import os
import socket

import pytest

from ai_api_unified import AiCandidateCooldowns, AIFactory, AiFallbackCompletions

pytestmark = pytest.mark.nonmock

STR_DEAD_BASE_URL: str = "http://127.0.0.1:9"
LOGGER_FALLBACK: str = "ai_api_unified.completions.ai_fallback_completions"


def test_outage_at_startup_starts_on_the_fallback_and_rests_the_primary(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    for str_key, str_host in (
        ("GOOGLE_GEMINI_API_KEY", "generativelanguage.googleapis.com"),
        ("ANTHROPIC_API_KEY", "api.anthropic.com"),
    ):
        if not os.environ.get(str_key):
            pytest.skip(f"Skipping: {str_key} not set")
        try:
            socket.gethostbyname(str_host)
        except OSError as exception:
            pytest.skip(f"Skipping: DNS unavailable for {str_host}: {exception}")
    monkeypatch.setenv("GOOGLE_AUTH_METHOD", "api_key")
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL_OVERRIDE", STR_DEAD_BASE_URL)
    monkeypatch.setenv("COMPLETIONS_RETRY_POLICY", "none")
    monkeypatch.setenv("AI_MODEL_CHECK", "prune")
    AiCandidateCooldowns.reset()
    with caplog.at_level(logging.WARNING):
        client = AIFactory.get_ai_completions_client(
            completions_engine="google-gemini",
            model_name="gemini-2.5-flash",
            fallbacks=[("claude", "claude-haiku-4-5")],
        )
        assert isinstance(client, AiFallbackCompletions)
        first = client.send_prompt("Reply with the single word: pong")
        print(f"\n  call 1 served by {client.last_route.label}: {first!r}")
        assert client.last_route.label == "claude:claude-haiku-4-5"
        list_events = [
            getattr(r, "ai_fallback_event", None)
            for r in caplog.records
            if r.name == LOGGER_FALLBACK
        ]
        print(f"  fallback events: {list_events}")
        assert "candidate_cooldown" in list_events
        caplog.clear()
        second = client.send_prompt("Reply with the single word: pong")
        print(f"  call 2 served by {client.last_route.label}: {second!r}")
    assert client.last_route.label == "claude:claude-haiku-4-5"
    # Call 2 never tried Gemini: no failover was logged.
    assert not any(
        getattr(r, "ai_fallback_event", None) == "failover" for r in caplog.records
    )
    AiCandidateCooldowns.reset()
