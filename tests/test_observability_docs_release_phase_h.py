from __future__ import annotations

import logging

import pytest

from ai_api_unified.voice.ai_voice_base import (
    AIVoiceBase,
    AIVoiceCapabilities,
    AIVoiceModelBase,
    AIVoiceSelectionBase,
)
from ai_api_unified.voice.audio_models import AudioFormat

TEST_AUDITION_TEXT: str = "This sample sentence should never appear in logs."


class StubVoiceClient(AIVoiceBase):
    """
    Minimal concrete voice client used to exercise audition logging behavior.

    Args:
        None

    Returns:
        Concrete AIVoiceBase subclass suitable for focused docs-release tests.
    """

    def text_to_voice(
        self,
        text_to_convert: str,
        voice: AIVoiceSelectionBase | None = None,
        audio_format: AudioFormat | None = None,
        speaking_rate: float = 1.0,
        use_ssml: bool = False,
    ) -> bytes:
        """
        Returns static audio bytes for the audition helper test.

        Args:
            text_to_convert: Caller-supplied text converted to speech.
            voice: Optional selected voice metadata.
            audio_format: Optional caller-facing audio-format selection.
            speaking_rate: Requested speaking rate multiplier.
            use_ssml: Whether the caller requested SSML mode.

        Returns:
            Static audio bytes because the test only verifies logging behavior.
        """
        # Normal return with static audio bytes for the audition helper test.
        return b"audio"

    def stream_audio(self, audio_bytes: bytes, chunk_size: int = 4096) -> None:
        """
        Accepts caller audio bytes without performing any playback in the test double.

        Args:
            audio_bytes: Audio payload that would be streamed by a concrete voice client.
            chunk_size: Chunk size requested by the caller for streaming playback.

        Returns:
            None because the test double intentionally skips playback.
        """
        # Normal return because the test double intentionally skips playback work.
        return None

    def speech_to_text(
        self,
        audio_file_path: str,
        language: str = "en",
        transcript_model: str | None = None,
    ) -> str:
        """
        Returns a static transcript so the test double satisfies the abstract base contract.

        Args:
            audio_file_path: Path to the audio file that would be transcribed.
            language: Requested transcription language code.
            transcript_model: Optional transcript model identifier.

        Returns:
            Static transcript string because this helper is unused in the audition test.
        """
        # Normal return with a static transcript because the test does not exercise STT behavior.
        return "transcript"


def test_audition_voices_logs_metadata_not_sample_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Ensures the audition helper logs only sample metadata rather than raw sample text.

    Args:
        caplog: Pytest fixture used to capture voice-base logs.

    Returns:
        None for normal test completion.
    """
    audio_format: AudioFormat = AudioFormat(
        key="wav",
        description="wav",
        file_extension=".wav",
        sample_rate_hz=16_000,
    )
    selected_voice: AIVoiceSelectionBase = AIVoiceSelectionBase(
        voice_id="voice-1",
        voice_name="Voice One",
        locale="en-US",
    )
    voice_client: StubVoiceClient = StubVoiceClient.model_construct(
        engine="stub",
        default_model_id="stub-model",
        default_audio_format=audio_format,
        selected_voice=selected_voice,
        common_vendor_capabilities=AIVoiceCapabilities(supports_ssml=False),
        list_models_capabilities=[
            AIVoiceModelBase(
                name="stub-model",
                description="stub",
                is_default=True,
                capabilities=AIVoiceCapabilities(supports_ssml=False),
            )
        ],
        list_available_voices=[selected_voice],
    )

    def _play_bytes_stub(audio_bytes: bytes) -> None:
        """
        Ignores audio playback during the audition-helper test.

        Args:
            audio_bytes: Audio payload that would otherwise be played.

        Returns:
            None because playback is intentionally skipped in the test.
        """
        # Normal return because the test intentionally skips audio playback.
        return None

    voice_client._play_bytes = _play_bytes_stub

    with caplog.at_level(
        logging.INFO,
        logger="ai_api_unified.voice.ai_voice_base",
    ):
        voice_client.audition_voices(
            ai_voice_client=voice_client,
            text=TEST_AUDITION_TEXT,
            pause_between_auditions=False,
            list_locales=["en-US"],
        )

    assert TEST_AUDITION_TEXT not in caplog.text
    assert "Audition sample sentence metadata" in caplog.text
    assert "char_count=" in caplog.text
