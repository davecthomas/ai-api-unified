from __future__ import annotations

from pathlib import Path

import pytest

from ai_api_unified.util.env_settings import EnvSettings


def test_azure_voice_reads_credentials_from_env_settings_env_file(
    monkeypatch, tmp_path: Path
) -> None:
    """Azure voice should resolve credentials via EnvSettings, including .env extras."""
    pytest.importorskip("azure.cognitiveservices.speech")
    from ai_api_unified.voice import ai_voice_azure as ai_voice_azure_module

    env_file: Path = tmp_path / ".env"
    env_file.write_text(
        (
            "MICROSOFT_COGNITIVE_SERVICES_API_KEY=azure-key-from-env-file\n"
            "MICROSOFT_COGNITIVE_SERVICES_REGION=westus2\n"
        ),
        encoding="utf-8",
    )

    captured: dict[str, str] = {}
    env_settings: EnvSettings = EnvSettings(_env_file=env_file)

    class FakeSpeechConfig:
        def __init__(self, *, subscription: str, region: str) -> None:
            captured["subscription"] = subscription
            captured["region"] = region

    monkeypatch.setattr(ai_voice_azure_module, "EnvSettings", lambda: env_settings)
    monkeypatch.setattr(ai_voice_azure_module.speechsdk, "SpeechConfig", FakeSpeechConfig)
    monkeypatch.setattr(
        ai_voice_azure_module.AIVoiceAzure,
        "_build_voice_catalog",
        lambda self: [],
    )

    ai_voice_azure_module.AIVoiceAzure(engine="azure")

    assert captured == {
        "subscription": "azure-key-from-env-file",
        "region": "westus2",
    }


def test_elevenlabs_voice_reads_api_key_from_env_settings_env_file(
    monkeypatch, tmp_path: Path
) -> None:
    """ElevenLabs voice should resolve API keys via EnvSettings, including .env extras."""
    pytest.importorskip("elevenlabs")
    from ai_api_unified.voice import ai_voice_elevenlabs as ai_voice_elevenlabs_module

    env_file: Path = tmp_path / ".env"
    env_file.write_text(
        "ELEVEN_LABS_API_KEY=elevenlabs-key-from-env-file\n",
        encoding="utf-8",
    )

    captured: dict[str, str] = {}
    env_settings: EnvSettings = EnvSettings(_env_file=env_file)

    class FakeElevenLabs:
        def __init__(self, *, api_key: str) -> None:
            captured["api_key"] = api_key

    monkeypatch.setattr(
        ai_voice_elevenlabs_module,
        "EnvSettings",
        lambda: env_settings,
    )
    monkeypatch.setattr(ai_voice_elevenlabs_module, "ElevenLabs", FakeElevenLabs)
    monkeypatch.setattr(
        ai_voice_elevenlabs_module.AIVoiceElevenLabs,
        "_list_voices_internal",
        lambda self: [],
    )

    ai_voice_elevenlabs_module.AIVoiceElevenLabs(engine="elevenlabs")

    assert captured == {"api_key": "elevenlabs-key-from-env-file"}
