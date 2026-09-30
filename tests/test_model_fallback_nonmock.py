# tests/test_model_fallback_nonmock.py
"""
Live tests for the model fallback chain against real provider APIs.

A real failover needs a real failure. The dependable one is an unknown model
id on the primary, so the chain here starts on a Claude model that does not
exist, turns on MODEL_UNAVAILABLE (off by default), and expects OpenAI to
serve the call. That exercises the engine's live error classification, the
lazy build of the fallback through the factory, and a real second request.
Claude is the primary because the Anthropic engine sends the model id as
given; the Gemini engine replaces an unknown id with its default model, so
it cannot produce this failure.

Run with:

    poetry run pytest -m nonmock tests/test_model_fallback_nonmock.py -q

Requires ANTHROPIC_API_KEY and OPENAI_API_KEY in the environment (.env).
"""

import logging
import os
import socket

import pytest

from ai_api_unified import (
    AIFactory,
    AiFallbackCompletions,
    AiFallbackReason,
    AiProviderRequestError,
)

# The module-level marker keeps these out of `-m "not nonmock"` runs.
pytestmark = pytest.mark.nonmock

ANTHROPIC_HOSTNAME: str = "api.anthropic.com"
OPENAI_HOSTNAME: str = "api.openai.com"

PRIMARY_ENGINE: str = "claude"
PRIMARY_MODEL: str = "claude-haiku-4-5"
# A model id Anthropic does not serve; the primary fails with a 404.
MISSING_MODEL: str = "claude-model-that-does-not-exist"
FALLBACK_ENGINE: str = "openai"
FALLBACK_MODEL: str = "gpt-4o-mini"


def _skip_if_dns_unavailable(hostname: str) -> None:
    try:
        socket.gethostbyname(hostname)
    except OSError as exception:
        pytest.skip(f"Skipping: DNS unavailable for {hostname}: {exception}")


@pytest.fixture
def live_chain_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require both providers' credentials and a route to both hosts."""
    pytest.importorskip("anthropic")
    pytest.importorskip("openai")
    for str_key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        if str_key not in os.environ:
            pytest.skip(f"Skipping: {str_key} not set")
    _skip_if_dns_unavailable(ANTHROPIC_HOSTNAME)
    _skip_if_dns_unavailable(OPENAI_HOSTNAME)


LOGGER_NAME: str = "ai_api_unified.completions.ai_fallback_completions"


def test_unknown_primary_model_fails_over_to_openai(
    live_chain_credentials: None, caplog: pytest.LogCaptureFixture
) -> None:
    client = AIFactory.get_ai_completions_client(
        model_name=MISSING_MODEL,
        completions_engine=PRIMARY_ENGINE,
        fallbacks=[(FALLBACK_ENGINE, FALLBACK_MODEL)],
        fallback_on={AiFallbackReason.MODEL_UNAVAILABLE},
    )
    assert isinstance(client, AiFallbackCompletions)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        str_reply: str = client.send_prompt(
            "Reply with the single word: pong", max_response_tokens=32
        )

    assert "pong" in str_reply.lower()
    assert client.last_route.engine == FALLBACK_ENGINE
    assert client.last_route.model == FALLBACK_MODEL

    # The failover must be visible in logs: one ERROR when the primary fails
    # and the chain moves, one WARNING when the fallback serves the request.
    list_records = [r for r in caplog.records if hasattr(r, "ai_fallback_event")]
    assert [r.ai_fallback_event for r in list_records] == [
        "failover",
        "served_by_fallback",
    ]
    assert list_records[0].levelno == logging.ERROR
    assert list_records[0].fallback_reason == "model_unavailable"
    assert list_records[0].status_code == 404
    assert list_records[1].levelno == logging.WARNING
    print("\nFallback log events observed during this test:")
    for record in list_records:
        print(f"  [{record.levelname}] {record.getMessage()}")


def test_unknown_model_is_classified_and_not_a_default_trigger(
    live_chain_credentials: None,
) -> None:
    # With the default reason set, MODEL_UNAVAILABLE stays off, so the same
    # chain raises the primary's typed error and never builds the fallback.
    client = AIFactory.get_ai_completions_client(
        model_name=MISSING_MODEL,
        completions_engine=PRIMARY_ENGINE,
        fallbacks=[(FALLBACK_ENGINE, FALLBACK_MODEL)],
    )
    assert isinstance(client, AiFallbackCompletions)

    with pytest.raises(AiProviderRequestError) as exc_info:
        client.send_prompt("Reply with the single word: pong", max_response_tokens=32)

    assert exc_info.value.fallback_reason is AiFallbackReason.MODEL_UNAVAILABLE
    assert exc_info.value.provider_engine == PRIMARY_ENGINE
    assert list(client._clients) == [0]


def test_healthy_primary_never_builds_the_fallback(
    live_chain_credentials: None,
) -> None:
    client = AIFactory.get_ai_completions_client(
        model_name=PRIMARY_MODEL,
        completions_engine=PRIMARY_ENGINE,
        fallbacks=[(FALLBACK_ENGINE, FALLBACK_MODEL)],
    )
    assert isinstance(client, AiFallbackCompletions)

    str_reply: str = client.send_prompt(
        "Reply with the single word: pong", max_response_tokens=32
    )

    assert "pong" in str_reply.lower()
    assert client.last_route.engine == PRIMARY_ENGINE
    assert list(client._clients) == [0]


def test_first_conversation_turn_fails_over_and_stamps_the_route(
    live_chain_credentials: None,
) -> None:
    client = AIFactory.get_ai_completions_client(
        model_name=MISSING_MODEL,
        completions_engine=PRIMARY_ENGINE,
        fallbacks=[(FALLBACK_ENGINE, FALLBACK_MODEL)],
        fallback_on={AiFallbackReason.MODEL_UNAVAILABLE},
    )

    turn = client.send_conversation(
        "You answer in one word.",
        [{"role": "user", "content": "Reply with the single word: pong"}],
        max_response_tokens=32,
    )

    assert turn.text is not None and "pong" in turn.text.lower()
    assert turn.provider_engine == FALLBACK_ENGINE
    assert turn.model_name == FALLBACK_MODEL


def test_second_turn_after_failover_replays_on_the_fallback(
    live_chain_credentials: None,
) -> None:
    # Issue 65: the OpenAI turn used to replay with SDK fields (annotations),
    # which read as neutral history, so turn two went back to Claude and
    # failed with a 400 on the unknown key.
    client = AIFactory.get_ai_completions_client(
        model_name=MISSING_MODEL,
        completions_engine=PRIMARY_ENGINE,
        fallbacks=[(FALLBACK_ENGINE, FALLBACK_MODEL)],
        fallback_on={AiFallbackReason.MODEL_UNAVAILABLE},
    )
    list_messages: list[dict[str, str]] = [
        {
            "role": "user",
            "content": "Pick one color: red or blue. Reply with the word only.",
        }
    ]

    turn = client.send_conversation("Be brief.", list_messages, max_response_tokens=32)
    client.extend_messages_with_turn(list_messages, turn)
    list_messages.append(
        {
            "role": "user",
            "content": "Which color did you pick? Reply with the word only.",
        }
    )
    turn2 = client.send_conversation("Be brief.", list_messages, max_response_tokens=32)

    assert turn.provider_engine == FALLBACK_ENGINE
    assert turn2.provider_engine == FALLBACK_ENGINE
    assert turn2.text is not None and turn2.text.strip()
    assert set(list_messages[1]) <= {"role", "content"}
