# test_gemini_vertex_location.py
"""
Issue #45: Vertex serves Gemini 3.x models only from the global location.
The default location is global, an explicit one is kept, and a 404 from a
regional location says how to fix it.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

import pytest
from google.genai import errors as gerr

from ai_api_unified import ai_google_base
from ai_api_unified.ai_google_base import AIGoogleBase

STR_404_MESSAGE: str = "Publisher model gemini-3.7-flash was not found"


def _env(dict_settings: dict[str, str]) -> Mock:
    env: Mock = Mock()
    env.get_setting.side_effect = lambda key, default=None: dict_settings.get(
        key, default
    )
    return env


def _vertex(str_location: str | None = None) -> dict[str, str]:
    dict_settings: dict[str, str] = {
        "GOOGLE_AUTH_METHOD": "service_account",
        "GOOGLE_PROJECT_ID": "proj",
    }
    if str_location is not None:
        dict_settings["GOOGLE_LOCATION"] = str_location
    return dict_settings


@pytest.mark.parametrize(
    ("str_location", "str_expected"),
    [(None, "global"), ("", "global"), ("us-central1", "us-central1")],
)
def test_vertex_location_defaults_to_global(
    str_location: str | None, str_expected: str
) -> None:
    with (
        patch.object(
            ai_google_base, "EnvSettings", return_value=_env(_vertex(str_location))
        ),
        patch.object(ai_google_base, "genai") as mock_genai,
        patch.object(
            AIGoogleBase, "_load_service_account_credentials", return_value=None
        ),
    ):
        AIGoogleBase().get_client(model="gemini-3.7-flash")
    assert mock_genai.Client.call_args.kwargs["location"] == str_expected


def _not_found() -> gerr.ClientError:
    return gerr.ClientError(
        404,
        {"error": {"code": 404, "message": STR_404_MESSAGE, "status": "NOT_FOUND"}},
    )


def _raise_404() -> Any:
    raise _not_found()


def test_regional_404_names_the_global_location() -> None:
    with patch.object(
        ai_google_base, "EnvSettings", return_value=_env(_vertex("us-central1"))
    ):
        with pytest.raises(RuntimeError) as info:
            AIGoogleBase()._retry_with_exponential_backoff(_raise_404, max_retries=1)
    str_message: str = str(info.value)
    assert STR_404_MESSAGE in str_message
    assert "'us-central1' does not serve this model" in str_message
    assert "GOOGLE_LOCATION" in str_message


@pytest.mark.parametrize(
    "dict_settings",
    [
        _vertex(),
        _vertex("GLOBAL"),
        {"GOOGLE_AUTH_METHOD": "api_key", "GOOGLE_GEMINI_API_KEY": "k"},
    ],
)
def test_no_hint_on_global_or_api_key(dict_settings: dict[str, str]) -> None:
    with patch.object(ai_google_base, "EnvSettings", return_value=_env(dict_settings)):
        with pytest.raises(RuntimeError) as info:
            AIGoogleBase()._retry_with_exponential_backoff(_raise_404, max_retries=1)
    assert "does not serve this model" not in str(info.value)


def test_hint_only_for_404() -> None:
    with patch.object(
        ai_google_base, "EnvSettings", return_value=_env(_vertex("us-central1"))
    ):
        assert AIGoogleBase()._vertex_location_hint(400) == ""
        assert AIGoogleBase()._vertex_location_hint(404) != ""


def test_typed_request_error_keeps_the_hint() -> None:
    from ai_api_unified.ai_provider_exceptions import AiProviderRequestError
    from ai_api_unified.completions.ai_google_gemini_completions import (
        GoogleGeminiCompletions,
    )

    fake_self: Mock = Mock()
    fake_self.PROVIDER_ENGINE_TOKEN = GoogleGeminiCompletions.PROVIDER_ENGINE_TOKEN
    fake_self._vertex_location_hint.side_effect = lambda code: (
        " HINT" if code == 404 else ""
    )
    with pytest.raises(AiProviderRequestError) as info:
        GoogleGeminiCompletions._raise_gemini_request_error(fake_self, _not_found())
    assert str(info.value).endswith(" HINT")
    assert info.value.status_code == 404
