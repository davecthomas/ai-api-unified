# test_google_service_account_key.py
"""
Service-account mode loads the key file GOOGLE_APPLICATION_CREDENTIALS names
through EnvSettings, so a path set only in `.env` reaches the Google clients
(issue: the SDK reads that variable from the process environment alone).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from ai_api_unified import ai_google_base
from ai_api_unified.ai_google_base import AIGoogleBase
from ai_api_unified.ai_provider_exceptions import AiProviderConfigurationValueError


def _env(dict_settings: dict[str, str]) -> Mock:
    env: Mock = Mock()
    env.get_setting.side_effect = lambda key, default=None: dict_settings.get(
        key, default
    )
    return env


def test_key_file_path_from_settings_is_loaded_with_cloud_scope() -> None:
    sentinel: object = object()
    with patch(
        "google.oauth2.service_account.Credentials.from_service_account_file",
        return_value=sentinel,
    ) as mock_load:
        credentials: Any = AIGoogleBase()._load_service_account_credentials(
            _env({"GOOGLE_APPLICATION_CREDENTIALS": " /keys/vertex.json "})
        )
    assert credentials is sentinel
    mock_load.assert_called_once_with(
        "/keys/vertex.json", scopes=[ai_google_base.GOOGLE_CLOUD_PLATFORM_SCOPE]
    )


def test_unset_key_file_falls_back_to_default_credentials() -> None:
    assert AIGoogleBase()._load_service_account_credentials(_env({})) is None


@pytest.mark.parametrize("str_content", [None, "not json", '{"type": "x"}'])
def test_unreadable_key_file_is_a_configuration_error(
    tmp_path: Path, str_content: str | None
) -> None:
    path: Path = tmp_path / "key.json"
    if str_content is not None:
        path.write_text(str_content)
    with pytest.raises(AiProviderConfigurationValueError, match="key.json"):
        AIGoogleBase()._load_service_account_credentials(
            _env({"GOOGLE_APPLICATION_CREDENTIALS": str(path)})
        )


def test_vertex_client_receives_the_key_file_credentials() -> None:
    sentinel: object = object()
    env: Mock = _env(
        {
            "GOOGLE_AUTH_METHOD": "service_account",
            "GOOGLE_PROJECT_ID": "proj",
            "GOOGLE_LOCATION": "us-east1",
        }
    )
    with (
        patch.object(ai_google_base, "EnvSettings", return_value=env),
        patch.object(ai_google_base, "genai") as mock_genai,
        patch.object(
            AIGoogleBase, "_load_service_account_credentials", return_value=sentinel
        ),
    ):
        AIGoogleBase().get_client(model="gemini-2.5-flash")
    mock_genai.Client.assert_called_once_with(
        vertexai=True, project="proj", location="us-east1", credentials=sentinel
    )


def test_api_key_mode_never_reads_the_key_file() -> None:
    env: Mock = _env({"GOOGLE_AUTH_METHOD": "api_key", "GOOGLE_GEMINI_API_KEY": "k"})
    with (
        patch.object(ai_google_base, "EnvSettings", return_value=env),
        patch.object(ai_google_base, "genai") as mock_genai,
        patch.object(AIGoogleBase, "_load_service_account_credentials") as mock_load,
    ):
        AIGoogleBase().get_client(model="gemini-2.5-flash")
    mock_load.assert_not_called()
    mock_genai.Client.assert_called_once_with(api_key="k")


@pytest.mark.parametrize(
    ("str_factory", "str_client"),
    [
        ("_create_tts_client", "texttospeech.TextToSpeechClient"),
        ("_create_stt_client", "speech.SpeechClient"),
    ],
)
def test_voice_clients_receive_the_key_file_credentials(
    str_factory: str, str_client: str
) -> None:
    voice_module: Any = pytest.importorskip("ai_api_unified.voice.ai_voice_google")
    sentinel: object = object()
    fake_self: Mock = Mock()
    fake_self.GOOGLE_AUTH_METHOD_API_KEY = AIGoogleBase.GOOGLE_AUTH_METHOD_API_KEY
    fake_self._get_google_auth_method.return_value = "service_account"
    fake_self._load_service_account_credentials.return_value = sentinel
    str_module, str_class = str_client.split(".")
    with patch.object(getattr(voice_module, str_module), str_class) as mock_client:
        getattr(voice_module.AIVoiceGoogle, str_factory)(fake_self)
    mock_client.assert_called_once_with(credentials=sentinel)
