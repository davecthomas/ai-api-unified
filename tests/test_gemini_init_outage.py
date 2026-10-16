# tests/test_gemini_init_outage.py
"""
The Gemini engine checks its model with models.get while it is built. A
provider outage there (5xx, 429, timeout, refused connection) must not stop
the build, so a fallback chain can start and serve; a configuration error
(bad key, denial, unknown model) still raises.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

import pytest

genai = pytest.importorskip("google.genai")
gerr = pytest.importorskip("google.genai.errors")
httpx = pytest.importorskip("httpx")

from ai_api_unified import ai_google_base  # noqa: E402
from ai_api_unified.ai_google_base import (  # noqa: E402
    AIGoogleBase,
    gemini_init_failure_is_transient,
)

DICT_SETTINGS: dict[str, str] = {
    "GOOGLE_AUTH_METHOD": "api_key",
    "GOOGLE_GEMINI_API_KEY": "test-api-key",
}


def _api_error(cls: Any, int_code: int, str_message: str) -> Exception:
    return cls(int_code, {"error": {"message": str_message, "status": "X"}})


def _get_client(error: Exception) -> Any:
    client = Mock()
    client.models.get.side_effect = error
    with (
        patch.object(ai_google_base.genai, "Client", return_value=client),
        patch.object(ai_google_base, "EnvSettings") as mock_env,
    ):
        mock_env.return_value.get_setting.side_effect = (
            lambda key, default=None: DICT_SETTINGS.get(key, default)
        )
        # Normal return with what get_client returned.
        return client, AIGoogleBase().get_client(model="gemini-2.5-flash")


@pytest.mark.parametrize(
    "error",
    [
        _api_error(gerr.ServerError, 503, "backend error"),
        _api_error(gerr.ClientError, 429, "slow down"),
        httpx.ReadTimeout("timed out"),
        httpx.ConnectError("refused"),
    ],
    ids=["503", "429", "timeout", "connect"],
)
def test_outage_at_init_returns_the_client(error: Exception) -> None:
    assert gemini_init_failure_is_transient(error)
    client, returned = _get_client(error)
    assert returned is client


@pytest.mark.parametrize(
    "error",
    [
        _api_error(gerr.ClientError, 400, "API key not valid"),
        _api_error(gerr.ClientError, 403, "permission denied"),
        _api_error(gerr.ClientError, 404, "model not found"),
        _api_error(gerr.ClientError, 429, "quota exceeded per day"),
    ],
    ids=["bad-key", "denied", "unknown-model", "daily-quota"],
)
def test_configuration_errors_still_stop_the_build(error: Exception) -> None:
    assert not gemini_init_failure_is_transient(error)
    with pytest.raises(RuntimeError):
        _get_client(error)
