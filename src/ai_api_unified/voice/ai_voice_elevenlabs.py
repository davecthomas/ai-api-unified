# src/ai_api_unified/voice/ai_voice_elevenlabs.py
from __future__ import annotations

import os
import random
import time
from io import BytesIO
from typing import Any, ClassVar, Iterator

ELEVENLABS_DEPENDENCIES_AVAILABLE: bool = False
try:
    from elevenlabs import ElevenLabs, GetVoicesV2Response, Model
    from elevenlabs import play as eleven_play
    from elevenlabs import stream

    ELEVENLABS_DEPENDENCIES_AVAILABLE = True
except ImportError as import_error:
    ELEVENLABS_DEPENDENCIES_AVAILABLE = False


if ELEVENLABS_DEPENDENCIES_AVAILABLE:
    from pydantic import Field
    from requests import ConnectTimeout

    from .ai_voice_base import (
        AIVoiceBase,
        AIVoiceCapabilities,
        AIVoiceModelBase,
        AIVoiceSelectionBase,
    )
    from .audio_models import AudioFormat
    from ..util import is_hex_enabled

    # ──────────────────────────────────────────────────────────────────────────────
    # Voice-selection helper
    # ──────────────────────────────────────────────────────────────────────────────
    class AIVoiceSelectionElevenLabs(AIVoiceSelectionBase):
        """Container for a single ElevenLabs voice selection (id → name)."""

    # ──────────────────────────────────────────────────────────────────────────────
    # ElevenLabs concrete implementation
    # ──────────────────────────────────────────────────────────────────────────────
    class AIVoiceElevenLabs(AIVoiceBase):
        """ElevenLabs implementation of :class:`AIVoiceBase`."""

        V3_MODEL_ID: str = "eleven_v3"  # enterprise-only
        _MODEL_DEFINITIONS: ClassVar[list[dict[str, Any]]] = [
            {
                "name": "eleven_multilingual_v2",
                "display_name": "ElevenLabs Multilingual V2",
                "description": "High-quality multilingual model supporting 32 languages.",
                "is_default": True,
            },
        ]
        default_model_id: str = Field(
            default="eleven_multilingual_v2",
            description="Default model name for ElevenLabs",
        )

        MIN_SPEED: ClassVar[float] = 0.7
        MAX_SPEED: ClassVar[float] = 1.2

        # ------------------------------------------------------------------ #
        # Construction                                                       #
        # ------------------------------------------------------------------ #
        def __init__(self, *, engine: str, **kwargs: Any) -> None:
            # ← call Pydantic’s init so it can set up __pydantic_fields_set__ and default fields
            super().__init__(engine=engine, **kwargs)
            api_key: str = os.getenv("ELEVEN_LABS_API_KEY", "")
            if not api_key:
                raise RuntimeError("ELEVEN_LABS_API_KEY is not set")

            # Client
            self.client: ElevenLabs = ElevenLabs(api_key=api_key)

            # Set capabilities before you get voices!
            self.common_vendor_capabilities = AIVoiceCapabilities(
                supports_ssml=False,
                supports_streaming=True,
                supports_speech_to_text=True,  # corrected: ElevenLabs offers STT
                supported_languages=[
                    # ElevenLabs currently supports ~32 languages; we just need 2 so far
                    "en",
                    "es",
                ],
                supported_locales=[
                    "en-US",
                    "es-US",
                ],
                supported_audio_formats=self.list_output_formats,
                supports_custom_voice=True,
                supports_emotion_control=False,
                supports_prosody_settings=False,
                supports_viseme_events=False,
                supports_word_timestamps=False,
                supports_batch_synthesis=False,
                supports_pronunciation_lexicon=False,
                supports_audio_profiles=False,
                min_speaking_rate=self.MIN_SPEED,
                max_speaking_rate=self.MAX_SPEED,
            )

            # Prefetch voices & models
            self.list_available_voices: list[AIVoiceSelectionElevenLabs] = (
                self._list_voices_internal()
            )
            if self.list_available_voices:
                first_voice = self.list_available_voices[0]
                self.selected_voice = AIVoiceSelectionElevenLabs(
                    voice_id=first_voice.voice_id,
                    voice_name=first_voice.voice_name,
                )
            else:
                self.selected_voice = None

            self.list_models_capabilities = self._initialize_models_and_capabilities()

            # Model selection
            self.selected_model: AIVoiceModelBase = self.get_default_model()

            # ElevenLabs output formats
            self.list_output_formats: list[AudioFormat] = [
                AudioFormat(
                    key="mp3_22050_32",
                    description="MP3 - 22.05 kHz - 32 kbps (lowest bandwidth)",
                    file_extension=".mp3",
                    sample_rate_hz=22_050,
                ),
                AudioFormat(
                    key="mp3_44100_32",
                    description="MP3 - 44.1 kHz - 32 kbps",
                    file_extension=".mp3",
                    sample_rate_hz=44_100,
                ),
                AudioFormat(
                    key="mp3_44100_64",
                    description="MP3 - 44.1 kHz - 64 kbps",
                    file_extension=".mp3",
                    sample_rate_hz=44_100,
                ),
                AudioFormat(
                    key="mp3_44100_96",
                    description="MP3 - 44.1 kHz - 96 kbps",
                    file_extension=".mp3",
                    sample_rate_hz=44_100,
                ),
                AudioFormat(
                    key="mp3_44100_128",
                    description="MP3 - 44.1 kHz - 128 kbps (default, free and starter)",
                    file_extension=".mp3",
                    sample_rate_hz=44_100,
                ),
                AudioFormat(
                    key="mp3_44100_192",
                    description="MP3 - 44.1 kHz - 192 kbps (Creator tier and above)",
                    file_extension=".mp3",
                    sample_rate_hz=44_100,
                ),
                AudioFormat(
                    key="pcm_16000",
                    description="PCM S16LE - 16 kHz (raw WAV compatible)",
                    file_extension=".wav",
                    sample_rate_hz=16_000,
                ),
                AudioFormat(
                    key="pcm_22050",
                    description="PCM S16LE - 22.05 kHz",
                    file_extension=".wav",
                    sample_rate_hz=22_050,
                ),
                AudioFormat(
                    key="pcm_24000",
                    description="PCM S16LE - 24 kHz",
                    file_extension=".wav",
                    sample_rate_hz=24_000,
                ),
                AudioFormat(
                    key="pcm_44100",
                    description="PCM S16LE - 44.1 kHz (Pro tier and above)",
                    file_extension=".wav",
                    sample_rate_hz=44_100,
                ),
                AudioFormat(
                    key="ulaw_8000",
                    description="mu-law - 8 kHz (Twilio friendly)",
                    file_extension=".ulaw",
                    sample_rate_hz=8_000,
                ),
            ]

            # default format is the highest quality MP3
            self.default_audio_format: AudioFormat = next(
                fmt for fmt in self.list_output_formats if fmt.key == "mp3_44100_192"
            )

            self.default_model_id = "eleven_multilingual_v2"

        def _initialize_models_and_capabilities(self) -> list[AIVoiceModelBase]:
            """
            Creates a structured list of models and their specific capabilities.
            """
            model_list = []
            for model_def in self._MODEL_DEFINITIONS:
                # Create a unique, deep copy of the common capabilities for this model.
                model_caps = self.common_vendor_capabilities.model_copy(deep=True)

                model_list.append(
                    AIVoiceModelBase(
                        name=model_def["name"],
                        display_name=model_def.get("display_name"),
                        description=model_def.get("description"),
                        # Assign the customized capabilities object to the model.
                        capabilities=model_caps,
                        is_default=model_def.get("is_default", False),
                    )
                )
            return model_list

        # ------------------------------------------------------------------ #
        # Private helpers                                                    #
        # ------------------------------------------------------------------ #
        def _search_voices(
            self,
            page_size: int,
            next_page_token: str | None = None,
            max_retries: int = 3,
            backoff_factor: float = 1.0,
        ) -> GetVoicesV2Response:
            """
            Wrap ElevenLabs voices.search() with retry logic on ConnectTimeout.
            Uses exponential backoff with jitter.
            """
            for attempt in range(1, max_retries + 1):
                try:
                    return self.client.voices.search(
                        page_size=page_size,
                        next_page_token=next_page_token,
                    )
                except ConnectTimeout:
                    if attempt == max_retries:
                        # no retries left, re-raise
                        raise
                    # exponential backoff: base * 2^(attempt-1) + small jitter
                    sleep_sec: float = backoff_factor * (2 ** (attempt - 1))
                    sleep_sec += random.uniform(0, 0.1)
                    time.sleep(sleep_sec)

        def _list_voices_internal(
            self, default_voices_only: bool = False
        ) -> list[AIVoiceSelectionElevenLabs]:
            voices_by_id: dict[str, AIVoiceSelectionElevenLabs] = {}

            # ---------- 1) Default & Personal voices (v2) ----------

            next_page_token: str | None = None
            max_retries: int = 3
            backoff_factor: float = 1.0
            while True:
                for attempt in range(1, max_retries + 1):
                    try:
                        page_voice_search_resp: GetVoicesV2Response = (
                            self._search_voices(
                                page_size=100,
                                next_page_token=next_page_token,
                            )
                        )
                    except ConnectTimeout:
                        if attempt == max_retries:
                            # no retries left, re-raise
                            raise
                        # exponential backoff: base * 2^(attempt-1) + small jitter
                        sleep_sec: float = backoff_factor * (2 ** (attempt - 1))
                        sleep_sec += random.uniform(0, 0.1)
                        time.sleep(sleep_sec)

                for v in page_voice_search_resp.voices:
                    # Iterate safely even if verified_languages is None
                    for lang in v.verified_languages:
                        if not lang.locale:
                            continue
                        # Normalize and compare for supported languages
                        if (
                            lang.language.lower()
                            in self.common_vendor_capabilities.supported_languages
                            and lang.locale
                            in self.common_vendor_capabilities.supported_locales
                        ):
                            voices_by_id[v.voice_id] = AIVoiceSelectionElevenLabs(
                                voice_id=v.voice_id,
                                voice_name=v.name,
                                language=lang.language or "en",
                                accent=lang.accent or "american",
                                locale=lang.locale or "en-US",
                            )
                            break  # no need to check other languages for this voice

                if not page_voice_search_resp.next_page_token:
                    break
                next_page_token = page_voice_search_resp.next_page_token

            if default_voices_only:
                return list(voices_by_id.values())
            else:
                # ---------- 2) Voice Library voices (v1 shared) ----------

                for lang in self.common_vendor_capabilities.supported_languages:
                    for locale in self.common_vendor_capabilities.supported_locales:
                        page: int = 0
                        while True:
                            lib_resp = self.client.voices.get_shared(
                                page=page,
                                page_size=100,  # max for this endpoint
                                language=lang,
                                locale=locale,
                            )
                            for v in lib_resp.voices:
                                voices_by_id[v.voice_id] = AIVoiceSelectionElevenLabs(
                                    voice_id=v.voice_id,
                                    voice_name=v.name,
                                    language=v.language or "en",
                                    accent=v.accent or "american",
                                    locale=v.locale or "en-US",
                                    gender=(
                                        v.labels.get("gender")
                                        if getattr(v, "labels", None)
                                        else None
                                    ),
                                )

                            if not lib_resp.has_more:
                                break
                            page += 1

                return list(voices_by_id.values())

        # ------------------------------------------------------------------ #
        # AIVoiceBase interface                                              #
        # ------------------------------------------------------------------ #
        def get_models_dict(self) -> dict[str, str]:
            all_models: list[Model] = self.client.models.list()
            dict_response: dict[str, str] = {m.model_id: m.name for m in all_models}
            dict_response[self.V3_MODEL_ID] = "ElevenLabs V3 (Enterprise only)"
            return dict_response

        # ------------- synthesis & playback -------------------------------- #
        def text_to_voice(
            self,
            *,
            text_to_convert: str,
            voice: AIVoiceSelectionElevenLabs,  # keeping signature exactly as requested
            audio_format: AudioFormat,
            voice_settings: dict[str, Any] | None = None,
            use_ssml: bool = False,  # unused, but kept for compatibility
        ) -> bytes:
            """
            ElevenLabs call that turns *text_to_convert* into raw
            audio bytes with the given *voice*, *audio_format*, and *speaking_rate*.
            """
            model_id: str = (
                self.selected_model.name
                if self.selected_model
                else self.default_model_id
            )

            raw_audio = self.client.text_to_speech.convert(
                text=text_to_convert,
                voice_id=voice.voice_id,
                model_id=model_id,
                output_format=audio_format.key,
                voice_settings=voice_settings,
                # language_code=voice.locale,   # V2 model doesn't
            )

            return b"".join(raw_audio) if isinstance(raw_audio, Iterator) else raw_audio

        def stream_audio(
            self,
            text: str,
            voice: AIVoiceSelectionElevenLabs | None = None,
        ) -> bytes:
            """Stream audio data for *text* and return raw bytes."""
            voice = voice or self.selected_voice or self.get_default_voice()
            resolved_voice_id: str = voice.voice_id
            model_id: str = (
                self.selected_model.name
                if self.selected_model
                else self.default_model_id
            )

            audio_stream: Iterator[bytes] = self.client.text_to_speech.stream(
                text=text,
                voice_id=resolved_voice_id,
                model_id=model_id,
            )
            combined: bytes = b"".join(audio_stream)
            # only actually play if Hex support isn’t enabled
            if not is_hex_enabled():
                stream(iter([combined]))
            return combined

        def play(self, audio_bytes: bytes) -> None:
            """Play the audio bytes using ElevenLabs' play function."""
            if not is_hex_enabled():
                eleven_play(audio_bytes)
            else:
                print("Hex support is enabled; skipping playback.")

        # ------------- speech-to-text ------------------------ --------------- #
        def speech_to_text(
            self, audio_bytes: bytes, language: str | None = None
        ) -> str:
            audio_buf: BytesIO = BytesIO(audio_bytes)
            result = self.client.speech_to_text.convert(
                file=audio_buf,
                model_id="scribe_v1",
                tag_audio_events=False,
            )
            return result.text
