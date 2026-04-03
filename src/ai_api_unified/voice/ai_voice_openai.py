# src/ai_api_unified/voice/ai_voice_openai.py
"""OpenAI Text-to-Speech implementation of :class:`AIVoiceBase`."""

from __future__ import annotations

import logging
from io import BytesIO
from time import time
from typing import Any, ClassVar

from openai import BadRequestError, OpenAI, OpenAIError, RateLimitError
from openai.types.audio import (
    TranscriptionCreateResponse,
)
from pydantic import Field
from ai_api_unified.ai_openai_base import AIOpenAIBase
from ai_api_unified.util._lazy_pydub import AudioSegment, CouldntDecodeError

from .ai_voice_base import (
    AiApiObservedTtsResultModel,
    AIVoiceBase,
    AIVoiceCapabilities,
    AIVoiceModelBase,
    AIVoiceSelectionBase,
)
from .audio_models import AudioFormat
from ..util import is_hex_enabled
from ..util.env_settings import EnvSettings


class AIVoiceSelectionOpenAI(AIVoiceSelectionBase):
    """Container describing an OpenAI voice."""


class AIVoiceOpenAI(AIVoiceBase, AIOpenAIBase):
    """OpenAI text-to-speech implementation."""

    DEFAULT_INSTRUCTIONS: ClassVar[str] = (
        "Speak in an energetic and clear way as if you are a radio advertisement broadcast"
    )
    MIN_SPEED: ClassVar[float] = 0.25  # Speed is ignored for gpt-4o-mini-tts
    MAX_SPEED: ClassVar[float] = 4.0
    _MODEL_DEFINITIONS: ClassVar[list[dict[str, str]]] = [
        {
            "name": "tts-1-hd",
            "display_name": "OpenAI High Definition TTS model",
            "description": "High-definition, fastest, default choice.",
            "is_default": True,
        },
        {
            "name": "tts-1",
            "display_name": "OpenAI Standard TTS model",
            "description": "Standard quality, balanced speed and quality.",
            "is_default": False,
        },
        {
            "name": "gpt-4o-mini-tts",
            "display_name": "GPT-4o Mini TTS model",
            "description": "Smaller, lower-latency model.",
            "is_default": False,
        },
    ]

    default_model_id: str = Field("tts-1-hd", description="Default OpenAI TTS model")

    def __init__(self, *, engine: str, **kwargs: Any) -> None:
        super().__init__(engine=engine, **kwargs)
        env: EnvSettings = EnvSettings()
        api_key: str = env.get_setting("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        base_url: str = self.get_api_base_url()
        self.client: OpenAI = OpenAI(api_key=api_key, base_url=base_url)

        # Supported formats – all 24 kHz output
        self.list_output_formats: list[AudioFormat] = [
            AudioFormat(
                key="mp3_24000",
                description="MP3 - 24 kHz",
                file_extension=".mp3",
                sample_rate_hz=24_000,
            ),
            AudioFormat(
                key="opus_24000",
                description="Opus - 24 kHz",
                file_extension=".opus",
                sample_rate_hz=24_000,
            ),
            AudioFormat(
                key="aac_24000",
                description="AAC - 24 kHz",
                file_extension=".aac",
                sample_rate_hz=24_000,
            ),
            AudioFormat(
                key="flac_24000",
                description="FLAC - 24 kHz",
                file_extension=".flac",
                sample_rate_hz=24_000,
            ),
            AudioFormat(
                key="wav_linear16_24000",
                description="WAV (Linear16 - 24 kHz)",
                file_extension=".wav",
                sample_rate_hz=24_000,
            ),
        ]

        self.default_audio_format: AudioFormat = next(
            fmt for fmt in self.list_output_formats if fmt.key == "flac_24000"
        )

        self.common_vendor_capabilities = AIVoiceCapabilities(
            supports_ssml=False,
            supports_streaming=True,
            supports_speech_to_text=False,
            supported_languages=["en", "es"],
            supported_locales=["en-US", "es-US"],
            supported_audio_formats=self.list_output_formats,
            supports_custom_voice=False,
            supports_emotion_control=False,
            supports_prosody_settings=True,
            supports_viseme_events=False,
            supports_word_timestamps=False,
            supports_batch_synthesis=False,
            supports_pronunciation_lexicon=False,
            supports_audio_profiles=False,
            min_speaking_rate=self.MIN_SPEED,
            max_speaking_rate=self.MAX_SPEED,
        )

        self.list_available_voices: list[AIVoiceSelectionOpenAI] = (
            self._build_voice_catalog()
        )
        if self.list_available_voices:
            first_voice = self.list_available_voices[0]
            self.selected_voice = AIVoiceSelectionOpenAI(
                voice_id=first_voice.voice_id, voice_name=first_voice.voice_name
            )
        else:
            self.selected_voice = None

        self.list_models_capabilities = self._initialize_models_and_capabilities()

        self.selected_model: AIVoiceModelBase = self.get_default_model()

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

    def _build_voice_catalog(self) -> list[AIVoiceSelectionOpenAI]:
        catalog: list[AIVoiceSelectionOpenAI] = []
        voices_gender = {
            "alloy": "male",  #
            "ash": "female",  #
            "coral": "female",  #
            "echo": "male",  #
            "fable": "male",  #
            "nova": "female",  #
            "onyx": "male",  #
            "sage": "female",  #
            "shimmer": "female",  #
            # "ballad": "female", # This is not available
        }
        for vid, gender in voices_gender.items():
            catalog.append(
                AIVoiceSelectionOpenAI(
                    voice_id=vid,
                    voice_name=vid.title(),
                    language="en",
                    accent="american",
                    locale="en-US",
                    gender=gender,
                )
            )
        return catalog

    @staticmethod
    def _format_to_response_key(audio_format: AudioFormat) -> str:
        mapping = {
            "mp3": "mp3",
            "opus": "opus",
            "aac": "aac",
            "flac": "flac",
            "wav_linear16": "wav",
        }
        prefix = audio_format.key.split("_")[0]
        if prefix.startswith("wav"):
            prefix = "wav_linear16"
        if prefix not in mapping:
            raise ValueError(f"Unsupported audio format: {audio_format.key}")
        return mapping[prefix]

    def _synthesize_audio_bytes(
        self,
        *,
        text_to_convert: str,
        voice: AIVoiceSelectionOpenAI,
        audio_format: AudioFormat,
        speaking_rate: float,
    ) -> bytes:
        """
        Execute one non-streaming OpenAI TTS request and return the provider audio bytes.

        Args:
            text_to_convert: Input text that should be synthesized into speech.
            voice: Resolved OpenAI voice used for the synthesis request.
            audio_format: Caller-facing audio format that determines the OpenAI response format.
            speaking_rate: Requested OpenAI speech speed multiplier.

        Returns:
            Raw audio bytes returned by the OpenAI SDK for the completed synthesis request.
        """
        response_format: str = self._format_to_response_key(audio_format)
        model_id: str = (
            self.selected_model.name if self.selected_model else self.default_model_id
        )

        try:
            response = self.client.audio.speech.create(
                model=model_id,
                voice=voice.voice_id,
                input=text_to_convert,
                response_format=response_format,
                speed=speaking_rate,
                instructions=self.DEFAULT_INSTRUCTIONS,
            )
            try:
                response_content: bytes = response.content
            except AttributeError:
                response_content = bytes(response)
            # Normal return with the provider audio bytes for the completed synthesis request.
            return response_content
        except OpenAIError as error:
            logging.error(
                "OpenAI TTS API error: model=%s voice=%s error=%s",
                model_id,
                voice.voice_id,
                error,
            )
            raise RuntimeError(f"Text-to-speech failed (API error): {error}") from error
        except Exception as error:
            logging.exception("Unexpected error in text_to_voice")
            raise RuntimeError(
                f"Text-to-speech failed (unexpected): {error}"
            ) from error

    def _stream_audio_bytes(
        self,
        *,
        text_to_convert: str,
        voice: AIVoiceSelectionOpenAI,
        audio_format: AudioFormat,
    ) -> bytes:
        """
        Execute one streaming OpenAI TTS request and return the aggregated audio bytes.

        Args:
            text_to_convert: Input text that should be synthesized into speech.
            voice: Resolved OpenAI voice used for the synthesis request.
            audio_format: Caller-facing audio format that determines the OpenAI response format.

        Returns:
            Combined audio bytes aggregated from the streaming OpenAI SDK response.
        """
        model_id: str = (
            self.selected_model.name if self.selected_model else self.default_model_id
        )
        response = self.client.audio.speech.create(
            model=model_id,
            voice=voice.voice_id,
            input=text_to_convert,
            response_format=self._format_to_response_key(audio_format),
            speed=1.0,
            stream=True,
            instructions=self.DEFAULT_INSTRUCTIONS,
        )
        combined_audio_bytes: bytes = b"".join(chunk for chunk in response.iter_bytes())
        if not is_hex_enabled():
            self._play_bytes(combined_audio_bytes)
        # Normal return with the fully aggregated streaming audio payload.
        return combined_audio_bytes

    def text_to_voice(
        self,
        *,
        text_to_convert: str,
        voice: AIVoiceSelectionOpenAI,
        audio_format: AudioFormat,
        speaking_rate: float = 1.0,
        use_ssml: bool = False,  # ignored
    ) -> bytes:
        dict_input_metadata = self._build_tts_observability_input_metadata(
            text_to_convert=text_to_convert,
            voice=voice,
            audio_format=audio_format,
            speaking_rate=speaking_rate,
            use_ssml=use_ssml,
            bool_is_streaming=False,
        )
        audio_bytes: bytes = self._execute_voice_call_with_observability(
            operation="text_to_voice",
            dict_input_metadata=dict_input_metadata,
            callable_execute=lambda: self._synthesize_audio_bytes(
                text_to_convert=text_to_convert,
                voice=voice,
                audio_format=audio_format,
                speaking_rate=speaking_rate,
            ),
            callable_build_result_summary=lambda output_audio_bytes, provider_elapsed_ms: self._build_tts_observability_result_summary(
                observed_result=AiApiObservedTtsResultModel(
                    return_value=output_audio_bytes,
                    output_audio_byte_count=len(output_audio_bytes),
                    dict_metadata={"output_audio_format": audio_format.key},
                ),
                provider_elapsed_ms=provider_elapsed_ms,
            ),
            legacy_caller_id=self.user,
        )
        # Normal return with caller-facing audio bytes after optional observability wrapping.
        return audio_bytes

    def stream_audio(
        self, text: str, voice: AIVoiceSelectionOpenAI | None = None
    ) -> bytes:
        chosen_voice: AIVoiceSelectionOpenAI = (
            voice or self.selected_voice or self.get_default_voice()
        )
        output_audio_format: AudioFormat = self.default_audio_format
        dict_input_metadata = self._build_tts_observability_input_metadata(
            text_to_convert=text,
            voice=chosen_voice,
            audio_format=output_audio_format,
            speaking_rate=1.0,
            use_ssml=False,
            bool_is_streaming=True,
        )
        audio_bytes: bytes = self._execute_voice_call_with_observability(
            operation="stream_audio",
            dict_input_metadata=dict_input_metadata,
            callable_execute=lambda: self._stream_audio_bytes(
                text_to_convert=text,
                voice=chosen_voice,
                audio_format=output_audio_format,
            ),
            callable_build_result_summary=lambda output_audio_bytes, provider_elapsed_ms: self._build_tts_observability_result_summary(
                observed_result=AiApiObservedTtsResultModel(
                    return_value=output_audio_bytes,
                    output_audio_byte_count=len(output_audio_bytes),
                    dict_metadata={"output_audio_format": output_audio_format.key},
                ),
                provider_elapsed_ms=provider_elapsed_ms,
            ),
            legacy_caller_id=self.user,
        )
        # Normal return with aggregated streaming audio bytes after optional observability wrapping.
        return audio_bytes

    def speech_to_text(self, audio_bytes: bytes, language: str | None = None) -> str:
        """
        Convert audio bytes to text using OpenAI's Whisper model, with automatic
        chunking for large inputs (30 s windows, 5 s overlap).
        :param audio_bytes: The audio data in bytes.
        :param language: Optional language code (e.g., 'en', 'es'). Defaults to 'en'.
        :return: The full stitched transcript.
        """
        logger = logging.getLogger(__name__)
        chunk_length_s = 30
        overlap_s = 5

        # Load audio into a time-addressable segment
        try:
            audio = AudioSegment.from_file(BytesIO(audio_bytes))
        except CouldntDecodeError as exc:
            logger.error("Failed to decode audio: %s", exc)
            raise ValueError(f"Could not decode audio: {exc}") from exc
        except Exception as exc:
            logger.error("Unexpected error loading audio: %s", exc)
            raise RuntimeError(f"Unexpected error loading audio: {exc}") from exc

        total_ms = len(audio)
        chunk_ms = chunk_length_s * 1000
        overlap_ms = overlap_s * 1000

        transcripts: list[str] = []
        start = 0

        while start < total_ms:
            end = min(start + chunk_ms, total_ms)
            segment = audio[start:end]

            # Export slice to WAV in memory
            try:
                buf = BytesIO()
                segment.export(buf, format="wav")
                buf.seek(0)
                buf.name = f"audio_{start}_{end}.wav"
            except OSError as exc:
                logger.error("Failed to export audio segment: %s", exc)
                raise RuntimeError(f"Failed to export audio segment: {exc}") from exc

            max_retries = 3
            attempt = 0
            while True:
                try:
                    resp = self.client.audio.transcriptions.create(
                        file=buf,
                        model="whisper-1",
                        response_format="text",
                        language=language or "en",
                    )
                    transcripts.append(
                        resp.text
                        if isinstance(resp, TranscriptionCreateResponse)
                        else resp
                    )
                    break
                except BadRequestError as exc:
                    logger.error("OpenAI bad request error: %s", exc)
                    raise RuntimeError(f"OpenAI bad request error: {exc}") from exc
                except RateLimitError as exc:
                    attempt += 1
                    if attempt > max_retries:
                        logger.error("Exceeded retry limit due to rate limit: %s", exc)
                        raise RuntimeError(
                            f"Exceeded retry limit ({max_retries}) due to rate limit: {exc}"
                        ) from exc
                    backoff = 2**attempt
                    logger.warning(
                        "Rate limit hit, backing off for %s seconds (attempt %s)",
                        backoff,
                        attempt,
                    )
                    time.sleep(backoff)
                    buf.seek(0)
                    continue
                except OpenAIError as exc:
                    logger.error("OpenAI API error: %s", exc)
                    raise RuntimeError(f"OpenAI API error: {exc}") from exc
                except Exception as exc:
                    logger.error("Unexpected error during transcription: %s", exc)
                    raise RuntimeError(
                        f"Unexpected error during transcription: {exc}"
                    ) from exc

            # Advance window (preserve overlap)
            start += chunk_ms - overlap_ms

        return " ".join(transcripts)
