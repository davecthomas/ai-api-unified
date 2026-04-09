# tests/test_voice_nonmock.py
from __future__ import annotations

import os
import socket

import pytest

from ai_api_unified.ai_provider_exceptions import (
    AiProviderDependencyUnavailableError,
)

# Import interfaces for type hints
from ai_api_unified.voice import (
    AIVoiceBase,
    AIVoiceFactory,
    AIVoiceSelectionBase,
    AudioFormat,
)

VOICE_ENGINE_BY_PROVIDER: dict[str, str] = {
    "openai": "openai",
    "google-gemini": "google",
    "google": "google",
    "azure": "azure",
    "elevenlabs": "elevenlabs",
}
GOOGLE_TTS_HOSTNAME: str = "texttospeech.googleapis.com"


def _skip_if_dns_unavailable(hostname: str) -> None:
    """Skip live provider tests quickly when DNS/network prerequisites are unavailable."""
    try:
        socket.getaddrinfo(hostname, 443)
    except OSError as exception:
        pytest.skip(f"Skipping: DNS unavailable for {hostname}: {exception}")


def _skip_if_google_tts_service_unavailable(exception: Exception) -> None:
    """Skip live Google TTS tests when the backing Cloud TTS service is disabled."""
    message: str = str(exception)
    if (
        "texttospeech.googleapis.com" in message
        or "Cloud Text-to-Speech API has not been used" in message
        or "SERVICE_DISABLED" in message
    ):
        pytest.skip(
            "Skipping Google voice nonmock test because Cloud Text-to-Speech is disabled for the current project."
        )


@pytest.mark.nonmock
def test_voice_nonmock(
    aiprovider: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Live TTS test:
    1) Build a voice client via the factory
    2) Generate speech bytes from a short sentence
    3) Play the audio (no-op in Hex env; local pydub backend otherwise)
    4) Assert basic invariants so the test is meaningful
    """
    if not aiprovider:
        pytest.skip("Skipping voice test because no provider was supplied.")

    voice_engine: str | None = VOICE_ENGINE_BY_PROVIDER.get(aiprovider)
    if voice_engine is None:
        pytest.skip(f"Skipping voice test for unsupported provider: {aiprovider}")

    monkeypatch.setenv("AI_VOICE_ENGINE", voice_engine)
    if voice_engine == "openai" and not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("Skipping voice test because OPENAI_API_KEY is not set.")
    if voice_engine == "google" and not os.environ.get("GOOGLE_GEMINI_API_KEY"):
        pytest.skip("Skipping voice test because GOOGLE_GEMINI_API_KEY is not set.")
    if voice_engine == "google":
        _skip_if_dns_unavailable(GOOGLE_TTS_HOSTNAME)
        monkeypatch.setenv("GOOGLE_AUTH_METHOD", "api_key")
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    if voice_engine == "azure" and not (
        os.environ.get("MICROSOFT_COGNITIVE_SERVICES_API_KEY")
        and os.environ.get("MICROSOFT_COGNITIVE_SERVICES_REGION")
    ):
        pytest.skip(
            "Skipping voice test because Azure speech credentials are not fully configured."
        )
    if voice_engine == "elevenlabs" and not os.environ.get("ELEVEN_LABS_API_KEY"):
        pytest.skip("Skipping voice test because ELEVEN_LABS_API_KEY is not set.")

    try:
        ai_voice_client: AIVoiceBase = AIVoiceFactory.create()
    except AiProviderDependencyUnavailableError as exception:
        pytest.skip(f"Skipping voice test due to missing dependency: {exception}")

    list_available_voices: list[AIVoiceSelectionBase] = (
        ai_voice_client.get_available_voices()
    )
    assert isinstance(list_available_voices, list)
    assert len(list_available_voices) > 0, "No voices available from provider"

    list_voices_by_locale: list[AIVoiceSelectionBase] = (
        ai_voice_client.get_voices_by_locale(locale="en-US")
    )
    assert isinstance(list_voices_by_locale, list)
    assert len(list_voices_by_locale) > 0, "No en-US voices available from provider"

    # Verify that the voice seletion prioritizes embedded locale over the passed locale
    if "gemini" in ai_voice_client.selected_model.name.lower():
        voice: AIVoiceSelectionBase = ai_voice_client.get_voice(
            voice_id="Orus", locale="en-US"
        )
    else:
        voice = ai_voice_client.get_default_voice()

    assert voice.locale == "en-US"

    audio_format: AudioFormat | None = ai_voice_client.default_audio_format
    if audio_format is None:
        list_output_formats: list[AudioFormat] = ai_voice_client.list_output_formats
        if list_output_formats:
            audio_format = list_output_formats[0]
        else:
            raise AssertionError("No audio formats available from provider")
    selected_audio_format: AudioFormat = audio_format

    test_text: str = "my dog has cute ears!"
    try:
        audio_bytes: bytes = ai_voice_client.text_to_voice(
            text_to_convert=test_text,
            voice=voice,
            audio_format=selected_audio_format,
        )
    except RuntimeError as exception:
        _skip_if_google_tts_service_unavailable(exception)
        raise

    assert isinstance(audio_bytes, (bytes, bytearray))
    assert len(audio_bytes) > 256, "Audio output is unexpectedly small"

    voice_id: str = voice.voice_id
    voice_name: str = voice.voice_name
    assert voice_id != ""
    assert voice_name != ""
    if audio_format is not None:
        fmt_key: str = audio_format.key
        assert fmt_key != ""

    ai_voice_client.play(audio_bytes)
    # ai_voice_client.save_generated_audio(audio_bytes, "test_voice_nonmock.wav")

    # test deprecated voice name format with bullets
    if "gemini" in ai_voice_client.selected_model.name.lower():
        test_text2: str = "testing the deprecated voice format with bullets"
        voice_deprecated: AIVoiceSelectionBase = ai_voice_client.get_voice(
            voice_id="en-US • Chirp3-HD • Orus"
        )
        try:
            audio_bytes = ai_voice_client.text_to_voice(
                text_to_convert=test_text2,
                voice=voice_deprecated,
                audio_format=selected_audio_format,
            )
        except RuntimeError as exception:
            _skip_if_google_tts_service_unavailable(exception)
            raise

        assert isinstance(audio_bytes, (bytes, bytearray))
        assert len(audio_bytes) > 256, "Audio output is unexpectedly small"

        voice_id: str = voice.voice_id
        voice_name: str = voice.voice_name
        assert voice_id != ""
        assert voice_name != ""
        if audio_format is not None:
            fmt_key: str = audio_format.key
            assert fmt_key != ""

        ai_voice_client.play(audio_bytes)

        # testing deprecated voice format in spanish
        test_text2: str = "Probando el formato de voz obsoleto con viñetas"
        voice_deprecated: AIVoiceSelectionBase = ai_voice_client.get_voice(
            voice_id="es-MX • Chirp3-HD • Orus"
        )
        try:
            audio_bytes = ai_voice_client.text_to_voice(
                text_to_convert=test_text2,
                voice=voice_deprecated,
                audio_format=selected_audio_format,
            )
        except RuntimeError as exception:
            _skip_if_google_tts_service_unavailable(exception)
            raise

        assert isinstance(audio_bytes, (bytes, bytearray))
        assert len(audio_bytes) > 256, "Audio output is unexpectedly small"
        ai_voice_client.play(audio_bytes)

        # Testing ability to use deprecated voice format with emotion prompt
        try:
            audio_bytes = ai_voice_client.text_to_voice_with_emotion_prompt(
                emotion_prompt="Spoken sardonically and sarcastically",
                text_to_convert=test_text2,
                voice=voice_deprecated,
                audio_format=selected_audio_format,
            )
        except RuntimeError as exception:
            _skip_if_google_tts_service_unavailable(exception)
            raise

        assert isinstance(audio_bytes, (bytes, bytearray))
        assert len(audio_bytes) > 256, "Audio output is unexpectedly small"
        ai_voice_client.play(audio_bytes)

    # Verify that the voice seletion prioritizes embedded locale over the passed locale
    if "gemini" in ai_voice_client.selected_model.name.lower():
        voice_es: AIVoiceSelectionBase = ai_voice_client.get_voice(
            voice_id="Orus", locale="es-MX"
        )
        test_text_es: str = "¡Mi perro tiene orejas lindas!"
        try:
            audio_bytes = ai_voice_client.text_to_voice(
                text_to_convert=test_text_es,
                voice=voice_es,
                audio_format=selected_audio_format,
            )
        except RuntimeError as exception:
            _skip_if_google_tts_service_unavailable(exception)
            raise

        assert isinstance(audio_bytes, (bytes, bytearray))
        assert len(audio_bytes) > 256, "Audio output is unexpectedly small"

        voice_id: str = voice_es.voice_id
        voice_name: str = voice_es.voice_name
        assert voice_id != ""
        assert voice_name != ""
        if audio_format is not None:
            fmt_key: str = audio_format.key
            assert fmt_key != ""

        ai_voice_client.play(audio_bytes)
        # ai_voice_client.save_generated_audio(audio_bytes, "test_voice_nonmock.wav")

        try:
            audio_bytes2: bytes = ai_voice_client.text_to_voice_with_emotion_prompt(
                emotion_prompt="Spoken sardonically and sarcastically",
                text_to_convert="Oh sí, [laughs dryly], "
                + test_text_es
                + " Si eso dices. [sighs dramatically]",
                voice=voice_es,
                audio_format=selected_audio_format,
            )
        except RuntimeError as exception:
            _skip_if_google_tts_service_unavailable(exception)
            raise
        assert isinstance(audio_bytes2, (bytes, bytearray))
        assert len(audio_bytes2) > 256, "Audio output is unexpectedly small"
        ai_voice_client.play(audio_bytes2)

    try:
        audio_bytes2: bytes = ai_voice_client.text_to_voice_with_emotion_prompt(
            emotion_prompt="Spoken sardonically and sarcastically",
            text_to_convert="Oh yeah, [laughs dryly], "
            + test_text
            + " If you say so. [sighs dramatically]",
            voice=voice,
            audio_format=selected_audio_format,
        )
        assert isinstance(audio_bytes2, (bytes, bytearray))
        assert len(audio_bytes2) > 256, "Audio output is unexpectedly small"
        ai_voice_client.play(audio_bytes2)
        # ai_voice_client.save_generated_audio(
        #     audio_bytes2, "test_voice_nonmock_google_emotion_prompt.wav"
        # )
    except NotImplementedError:
        # Some providers do not support emotion prompts; that's OK
        pass
