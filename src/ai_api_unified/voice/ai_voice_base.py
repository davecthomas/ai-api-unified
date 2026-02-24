# src/ai_api_unified/voice/ai_voice_base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from io import BytesIO
from itertools import islice
import logging
from pathlib import Path
import re
import struct
from time import sleep
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field
from ai_api_unified.util._lazy_pydub import AudioSegment, play

from .audio_models import AudioFormat
from ..util.utils import is_hex_enabled

_LOGGER = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Selected-voice container
# ──────────────────────────────────────────────────────────────────────────────
class AIVoiceSelectionBase(BaseModel):
    """Container describing a single vendor voice selection."""

    voice_id: str = Field(..., description="Vendor-specific voice identifier")
    voice_name: str = Field(..., description="Human readable name")
    language: str = Field(
        default="en", description="Primary language code, e.g. 'en' for English"
    )
    accent: str | None = Field(default=None, description="Accent descriptor")
    locale: str = Field(default="en-US", description="Locale code for the voice")
    gender: str | None = Field(
        default=None,
        description="Gender of the voice, e.g. 'male', 'female', 'neutral'",
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    def as_tuple(self) -> tuple[str, str]:
        """Return ``(voice_id, voice_name)`` for convenience."""
        return self.voice_id, self.voice_name

    def __repr__(self) -> str:  # noqa: D401
        vid, vname = self.as_tuple()
        return (
            "AIVoiceSelectionBase("
            f"voice_id={vid!r}, voice_name={vname!r}, language={self.language!r}, "
            f"accent={self.accent!r}, locale={self.locale!r})"
        )


class AIVoiceCapabilities(BaseModel):
    supports_ssml: bool = Field(
        default=False, description="Whether the TTS implementation supports SSML."
    )
    supports_streaming: bool = Field(
        default=False, description="Whether the TTS implementation supports streaming audio."
    )
    supports_speech_to_text: bool = Field(
        default=False, description="Whether the TTS implementation supports built-in STT."
    )

    # ───── Additional capabilities ─────
    supported_languages: list[str] = Field(
        default_factory=list,
        description="List of language codes supported by the vendor.",
    )
    supported_locales: list[str] = Field(
        default_factory=list,
        description="List of locale codes supported by the vendor.",
    )
    supported_audio_formats: list[AudioFormat] = Field(
        default_factory=list, description="See AudioFormat"
    )
    supports_custom_voice: bool = Field(
        default=False, description="Whether you can train or clone custom voices."
    )
    supports_emotion_control: bool = Field(
        default=False, description="Whether you can tweak emotion/style markers."
    )
    supports_prosody_settings: bool = Field(
        default=False, description="Tune pitch/rate/volume via API parameters."
    )
    supports_viseme_events: bool = Field(
        default=False, description="Emit viseme (lip-sync) timing data."
    )
    supports_word_timestamps: bool = Field(
        default=False, description="Return word-level time offsets in the audio."
    )
    supports_batch_synthesis: bool = Field(
        default=False, description="Asynchronous long-form (batch) TTS for audiobooks, etc."
    )
    supports_pronunciation_lexicon: bool = Field(
        default=False, description="Upload or reference custom lexicons for pronunciation."
    )
    supports_audio_profiles: bool = Field(
        default=False,
        description="Vendor-provided audio profiles (e.g. telephone vs headphone).",
    )
    min_speaking_rate: float | None = Field(
        default=None, description="Lowest speaking‑rate multiplier supported."
    )
    max_speaking_rate: float | None = Field(
        default=None, description="Highest speaking‑rate multiplier supported."
    )


class AIVoiceModelBase(BaseModel):
    """A structured container for an AI model's details and capabilities.
    a list of these is kept in the AIVoiceBase subclass instance."""

    name: str = Field(..., description="The unique identifier or name of the model.")
    display_name: str | None = Field(
        default=None, description="A human-friendly name for the model."
    )
    description: str | None = Field(
        default=None,
        description="A brief description of the model's purpose or strengths.",
    )
    capabilities: AIVoiceCapabilities = Field(
        default_factory=AIVoiceCapabilities,
        description="The specific capabilities of this model.",
    )
    is_default: bool = Field(
        default=False, description="Whether this is the default model for the vendor."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Vendor-agnostic TTS interface
# ──────────────────────────────────────────────────────────────────────────────
class AIVoiceBase(BaseModel, ABC):
    """Vendor-agnostic AI Voice interface."""

    MIN_SPEED: ClassVar[float] = 0.5
    MAX_SPEED: ClassVar[float] = 2.0
    _MP3_SAMPLE_RATE_HZ: ClassVar[int] = 24_000
    _OGG_SAMPLE_RATE_HZ: ClassVar[int] = 24_000
    _LINEAR16_SAMPLE_RATE_HZ: ClassVar[int] = 48_000
    _MULAW_SAMPLE_RATE_HZ: ClassVar[int] = 8_000
    _RAW_SAMPLE_RATE_HZ: ClassVar[int] = 48_000
    _RAW_SAMPLE_WIDTH_BYTES: ClassVar[int] = 2
    _RAW_CHANNELS: ClassVar[int] = 1
    _DEFAULT_AUDIO_FORMAT_KEY: ClassVar[str] = "wav_linear16_48000"
    _PROMPT_SEPARATOR: ClassVar[str] = "\n\n"
    _WAV_HEADER_PREFIX: ClassVar[bytes] = b"RIFF"
    _FORMAT_WAV: ClassVar[str] = "wav"
    _FORMAT_MP3: ClassVar[str] = "mp3"
    _FORMAT_OGG: ClassVar[str] = "ogg"
    _OGG_OPUS_CODEC: ClassVar[str] = "libopus"
    _PCM_MULAW_CODEC: ClassVar[str] = "pcm_mulaw"
    _MP3_TARGET_BITRATE: ClassVar[str] = "64k"
    _OPUS_TARGET_BITRATE: ClassVar[str] = "64k"
    _MULAW_AUDIO_KEY: ClassVar[str] = "mulaw_8000"
    _DEFAULT_TAIL_SILENCE_MS: ClassVar[int] = (
        100  # We force 1/10th of a second of silence at the end of TTS audio to prevent abrupt cutoffs.
    )
    _TAIL_SILENCE_THRESHOLD_DBFS: ClassVar[float] = -45.0

    engine: str = Field(
        default="",
        description="Name of the TTS engine, e.g. 'elevenlabs', 'google', etc.",
    )
    # ─── required class overrides ───
    client: AIVoiceBase | None = Field(
        default=None, description="Vendor-specific client instance."
    )
    list_output_formats: list[AudioFormat] = Field(
        default_factory=list,
        description="Supported audio output formats (must override in subclass).",
    )
    default_audio_format: AudioFormat | None = Field(
        default=None,
        description="Default audio output format (must override in subclass).",
    )
    common_vendor_capabilities: AIVoiceCapabilities = Field(
        default_factory=lambda: AIVoiceCapabilities(),
        description="Capabilities of the TTS implementation.",
    )

    list_models_capabilities: list[AIVoiceModelBase] | None = Field(
        default=None,
        description="A list of available models and their specific capabilities.",
    )
    dict_model_prices: dict[str, float] | None = Field(
        default=None,
        description="Mapping of model keys to their prices (optional)",
    )
    default_model_id: str = Field(default="", description="Default TTS model ID.")
    selected_model: AIVoiceModelBase | None = Field(default=None)
    selected_voice: AIVoiceSelectionBase | None = Field(default=None)

    #: cached list of every vendor voice
    list_available_voices: list[AIVoiceSelectionBase] = Field(default_factory=list)

    # Pydantic V2 config
    model_config = {"arbitrary_types_allowed": True}

    def get_default_model(self) -> AIVoiceModelBase:
        """
        Returns the default model by finding the entry flagged as is_default.
        """
        if not self.list_models_capabilities:
            raise RuntimeError("Model capabilities list has not been initialized.")

        # Find the first model in the list where is_default is True.
        for model in self.list_models_capabilities:
            if model.is_default:
                return model  # Return the entire model object

        # This should ideally never be reached if initialization is correct.
        raise RuntimeError("No default model was flagged in the capabilities list.")

    def get_voices_by_locale(self, locale: str) -> list[AIVoiceSelectionBase]:
        """
        Retrieve voices whose locale matches the provided locale string.
        """
        sanitized_locale: str = locale.strip().lower()
        if not sanitized_locale or not self.list_available_voices:
            return []
        list_filtered_voices: list[AIVoiceSelectionBase] = [
            voice_entry
            for voice_entry in self.list_available_voices
            if voice_entry.locale.lower() == sanitized_locale
        ]
        return list_filtered_voices

    # ------------- persistence helpers ---------------------------------- #
    def save_generated_audio(self, audio_bytes: bytes, file_name: str) -> None:
        path: Path = Path("generated/audio") / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio_bytes)

    # ------------- duration helper ------------------------------------- #
    def get_audio_duration(self, audio_bytes: bytes) -> float:
        """Return the duration of *audio_bytes* in seconds."""
        try:
            str_format: str = self.default_audio_format.file_extension.lstrip(".") # type: ignore 
            segment: AudioSegment = AudioSegment.from_file(
                BytesIO(audio_bytes), format=str_format
            )
            return len(segment) / 1000.0
        except Exception:
            pass

        try:
            key: str = self.default_audio_format.key # type: ignore 
            if key.startswith("mp3_"):
                _, _, bitrate_kbps = key.split("_")
                bitrate_bps: float = float(bitrate_kbps) * 1000
                return len(audio_bytes) * 8 / bitrate_bps
            if key.startswith("pcm_"):
                sample_rate: int = int(key.split("_")[1])
                bytes_per_second: int = sample_rate * 2  # 16-bit mono PCM
                return len(audio_bytes) / bytes_per_second
            if key.startswith("ulaw_"):
                sample_rate: int = int(key.split("_")[1])
                return len(audio_bytes) / sample_rate
        except Exception:
            pass

        return 0.0

    # ─────────── Required vendor hooks ───────────
    @abstractmethod
    def text_to_voice(
        self,
        text_to_convert: str,
        voice: AIVoiceSelectionBase | None = None,
        audio_format: AudioFormat | None = None,
        speaking_rate: float = 1.0,
        use_ssml: bool = False,
    ) -> bytes:
        """
        Convert *text_to_convert* → speech bytes.

        If *voice* is supplied it overrides the currently-selected voice.
        """
        ...

    @abstractmethod
    def stream_audio(
        self,
        text: str,
        voice: AIVoiceSelectionBase | None = None,
    ) -> bytes:
        """Stream audio for *text* and return raw bytes."""
        ...

    # ------------- metadata & defaults ---------------------------------- #
    def list_voices(self) -> list[AIVoiceSelectionBase]:
        """Return a copy of all available voices."""
        return list(self.list_available_voices)

    def get_default_voice(self) -> AIVoiceSelectionBase:
        if not self.list_available_voices:
            raise RuntimeError(
                "No voices available. Call get_available_voices() first."
            )
        first_voice: AIVoiceSelectionBase = self.list_available_voices[0]
        return AIVoiceSelectionBase(
            voice_id=first_voice.voice_id,
            voice_name=first_voice.voice_name,
        )


    def get_voice_name(self) -> str:
        """Return the name of the currently-selected voice, or empty if unset."""
        return self.selected_voice.voice_name if self.selected_voice else ""

    def get_voice(
        self,
        voice_id: str,
        locale: str = "en-US",
    ) -> AIVoiceSelectionBase:
        """
        Return the voice that matches both *voice_id* and *voice_locale*.

        Raises
        ------
        ValueError
            If no voice satisfies both criteria.
        """
        normalized_id: str = voice_id.strip()
        normalized_locale: str = locale.strip()

        # Deprecated voices have a bullet (•) separator; handle gracefully.
        if "•" in normalized_id:
            parts: list[str] = [segment.strip() for segment in normalized_id.split("•")]
            if len(parts) == 3:
                logging.warning(
                    "Voice identifier '%s' uses a deprecated format. "
                    "Please move to the '<voice_id>' syntax.",
                    voice_id,
                )
                normalized_locale = parts[0] or locale
                normalized_id = parts[2] or parts[-1]

        # non-en-US voices may start with the locale. Override any passed locale with the embedded locale
        embedded_locale_match: re.Match[str] | None = re.match(
            r"^([a-z]{2}-[A-Z]{2})\b", normalized_id
        )
        if embedded_locale_match:
            normalized_locale = embedded_locale_match.group(1)

        normalized_id = normalized_id.lower()
        normalized_locale = normalized_locale.lower()

        for voice in self.list_available_voices:
            if (
                voice.voice_id.lower() == normalized_id
                and voice.locale.lower() == normalized_locale
            ):
                return voice

        raise ValueError(f"No voice found with id '{voice_id}' and locale '{locale}'.")

    def get_voice_from_id(self, voice_id: str) -> AIVoiceSelectionBase:
        """
        Lookup a voice by its ID. If the ID isn’t found,
        fall back to the currently selected voice (if any),
        or else to the default voice.
        """
        # Exact‐match lookup
        for voice in self.list_available_voices:
            if voice.voice_id == voice_id:
                return AIVoiceSelectionBase(
                    voice_id=voice.voice_id,
                    voice_name=voice.voice_name,
                )

        # Fallback to what’s already selected
        if self.selected_voice is not None:
            return self.selected_voice

        # Finally, grab the first/default
        return self.get_default_voice()

    def get_voices_by_id(
        self, voice_id: str | None = None
    ) -> list[AIVoiceSelectionBase]:
        """Return voices whose IDs contain *voice_id* as a substring."""

        if not voice_id:
            return list(self.list_available_voices)

        target: str = voice_id.lower()
        list_matches: list[AIVoiceSelectionBase] = []
        for voice in self.list_available_voices:
            if target in voice.voice_id.lower():
                list_matches.append(
                    AIVoiceSelectionBase(
                        voice_id=voice.voice_id, voice_name=voice.voice_name
                    )
                )

        return list_matches

    @staticmethod
    def _get_first_pair(entries: dict[Any, Any]) -> dict[Any, Any]:
        return dict(islice(entries.items(), 1))

    def get_voice_by_name(self, voice_name: str | None = None) -> AIVoiceSelectionBase:
        """
        Return a voice selection by its name.

        If no voice matches, raise a ValueError.
        """
        if not voice_name:
            if self.selected_voice:
                return self.selected_voice
            else:  # if no voice is selected, return the default voice
                return self.get_default_voice()
        for voice in self.list_available_voices:
            if voice.voice_name.lower() == voice_name.lower():
                return voice
        raise ValueError(f"No voice found with name '{voice_name}'.")

    def _play_bytes(self, audio_bytes: bytes) -> None:
        try:
            seg: AudioSegment = AudioSegment.from_file(
                BytesIO(audio_bytes),
                format=self.default_audio_format.file_extension[1:], # type: ignore 
            )
            # Resample to 48 kHz for glitch-free output
            seg = seg.set_frame_rate(48_000)
            play(seg)  # This calls pydub.play
        except Exception as exc:  # pragma: no cover - local playback only
            logging.warning("local playback failed: %s", exc)

    def play(self, audio_bytes: bytes) -> None:  # noqa: D401
        if not is_hex_enabled():
            self._play_bytes(audio_bytes)

    @abstractmethod
    def speech_to_text(self, audio_bytes: bytes, language: str | None = None) -> str:
        """Transcribe *audio_bytes* to text and return the result."""
        ...

    def get_audio_file_extensions(self) -> list[str]:
        """Return a list of supported audio file extensions."""
        return [fmt.file_extension for fmt in self.list_output_formats]

    def compute_adjusted_speed(
        self,
        *,
        current_speed: float,
        current_duration_s: float,
        target_duration_s: float,
    ) -> float:
        """
        Return a speaking rate that *should* bring ``current_duration_s``
        close to ``target_duration_s``.
        Default model ⇒ perfectly linear (duration ∝ 1/speed).
        Vendors can override when the relationship is non-linear.
        """
        factor: float = current_duration_s / target_duration_s
        proposed: float = current_speed * factor
        return float(min(self.MAX_SPEED, max(self.MIN_SPEED, proposed)))

    def get_available_voices(self) -> list[AIVoiceSelectionBase]:
        """
        Return a copy of the list of all available voices.

        This is a convenience method that returns the cached list.
        """
        return self.list_available_voices

    def get_voices_by_locales(
        self, list_locales: list[str]
    ) -> list[AIVoiceSelectionBase]:
        """
        Return a list of voices that match the given locales.

        If no voices match, return an empty list.
        """
        voices: list[AIVoiceSelectionBase] = [
            v for v in self.list_available_voices if v.locale in list_locales
        ]
        return voices

    def audition_voices(
        self,
        ai_voice_client: AIVoiceBase,
        text: str = "Hello, I'm excited for my audition.",
        voice_id_group: str | None = None,
        pause_between_auditions: bool = False,
        list_locales: list[str] = ["en-US"],
    ) -> None:
        """
        Interactively step through every voice in the configured AIVoice engine.

        For each:
        1. Wait for Enter (or 'q' to quit).
        2. Print the voice ID + name.
        3. Synthesize & play "The quick brown fox jumps over the lazy dog."
        """

        # 2) Fetch the available voices
        voices_list: list[AIVoiceSelectionBase]
        if list_locales:
            voices_list = ai_voice_client.get_voices_by_locales(list_locales)
        else:
            if voice_id_group:
                voices_list = ai_voice_client.get_voices_by_id(voice_id_group)
            else:
                voices_list = ai_voice_client.list_available_voices

        for voice in voices_list:
            voice_id: str = voice.voice_id
            voice_name: str = voice.voice_name
            if pause_between_auditions:
                user_input: str = (
                    input(
                        f"\nPress Enter to test name: '{voice_name}' id:'{voice_id}', or 'q' to quit: "
                    )
                    .strip()
                    .lower()
                )

                if user_input == "q":
                    print("Aborting voice test.")
                    break

            print(f"\n\tAuditioning voice: Name={voice_name!r}; ID={voice_id!r}")

            spoken_voice_name = (
                voice_name.replace("-", " ").replace("•", "").replace("en US", "")
            )
            sample_sentence: str = text

            intro: str = f"Hi! I'm voice {spoken_voice_name}. "
            sample_sentence = f"{intro} {sample_sentence}"
            try:
                ai_voice_client.selected_voice = voice
                chosen_format: AudioFormat = ai_voice_client.default_audio_format # type: ignore 

                print(f"\t{sample_sentence}\n")

                supports_ssml: bool = getattr(
                    self.common_vendor_capabilities, "supports_ssml", False
                )
                use_ssml: bool = bool(
                    supports_ssml and re.search(r"<\/?[a-z][^>]*>", sample_sentence)
                )
                audio_bytes: bytes = ai_voice_client.text_to_voice(
                    text_to_convert=sample_sentence,
                    voice=voice,
                    audio_format=chosen_format,
                    use_ssml=use_ssml,
                )

                ai_voice_client.play(audio_bytes)  # type: ignore
            except NotImplementedError as e:
                print(f"[!] Playback not supported by this engine: {e}")
            except Exception as e:
                print(f"[!] Error during synthesis/playback: {e}")

            if not pause_between_auditions:
                # pause for 0.5 seconds between voices
                sleep(0.5)

        print("\n✅ Voice auditioning complete.")

    def speech_to_text_from_file(
        self, file_path: str | Path, language: str | None = None
    ) -> str:
        """
        Read an audio file from disk and convert it to text.
        :param file_path: Path to the audio file.
        :param language: Optional language code (e.g., 'en', 'es') to specify the language of the audio.
        :return: The transcribed text.
        """
        # Read the entire file into bytes
        audio_bytes: bytes = Path(file_path).read_bytes()
        # Delegate to your existing method
        return self.speech_to_text(audio_bytes, language)

    def save_transcript_to_file(self, transcript: str, file_path: str | Path) -> None:
        """
        Works with speech to text capabilities.
        Save a transcription string to a text file.
        :param transcript: The transcribed text to save.
        :param file_path: Path where the transcript will be written.
        """
        path = Path(file_path)
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write the transcript out as UTF-8
        path.write_text(transcript, encoding="utf-8")

    def text_to_voice_with_emotion_prompt(
        self,
        *,
        emotion_prompt: str | None,
        text_to_convert: str,
        voice: AIVoiceSelectionBase | None,
        audio_format: AudioFormat | None,
        speaking_rate: float = 1.0,  # Not supported by all vendors
    ) -> bytes:
        """
        Convert *text_to_convert* → speech bytes, with an emotion/style prompt.

        If *voice* is supplied it overrides the currently-selected voice.
        """
        raise NotImplementedError(
            "This TTS engine does not support emotion/style prompts."
        )

    def _export_audio_segment(
        self,
        *,
        audio_segment: AudioSegment,
        audio_format: AudioFormat,
    ) -> bytes:
        """
        Export an AudioSegment to bytes using the requested audio format.
        """
        output_buffer: BytesIO = BytesIO()
        requested_extension: str = audio_format.file_extension.lstrip(".")
        export_kwargs: dict[str, str] = {}
        if requested_extension == self._FORMAT_MP3:
            export_kwargs["bitrate"] = self._MP3_TARGET_BITRATE
        elif (
            requested_extension == self._FORMAT_WAV
            and audio_format.key == self._MULAW_AUDIO_KEY
        ):
            export_kwargs["codec"] = self._PCM_MULAW_CODEC
        elif requested_extension == self._FORMAT_OGG:
            export_kwargs["codec"] = self._OGG_OPUS_CODEC
            export_kwargs["bitrate"] = self._OPUS_TARGET_BITRATE

        audio_segment.export(
            output_buffer,
            format=requested_extension,
            **export_kwargs,
        )
        output_bytes: bytes = output_buffer.getvalue()
        return output_bytes

    def _convert_audio_bytes(
        self,
        *,
        audio_bytes: bytes,
        audio_format: AudioFormat,
        test_mode: bool = False,
    ) -> bytes:
        try:
            if test_mode:
                self._test_audio_bytes(
                    audio_bytes=audio_bytes,
                    audio_format=audio_format,
                )
            input_buffer: BytesIO = BytesIO(audio_bytes)
            input_buffer.seek(0)

            # Use from_file when a RIFF header is present so the header bytes are
            # not misinterpreted as PCM samples (which introduces an audible pop).
            audio_segment: AudioSegment
            if audio_bytes.startswith(self._WAV_HEADER_PREFIX):
                audio_segment = AudioSegment.from_file(
                    input_buffer, format=self._FORMAT_WAV
                )
            else:
                audio_segment = AudioSegment.from_raw(
                    input_buffer,
                    frame_rate=self._RAW_SAMPLE_RATE_HZ,
                    sample_width=self._RAW_SAMPLE_WIDTH_BYTES,
                    channels=self._RAW_CHANNELS,
                )

            if audio_segment.channels != self._RAW_CHANNELS:
                audio_segment = audio_segment.set_channels(self._RAW_CHANNELS)
            if audio_segment.sample_width != self._RAW_SAMPLE_WIDTH_BYTES:
                audio_segment = audio_segment.set_sample_width(
                    self._RAW_SAMPLE_WIDTH_BYTES
                )
            if (
                audio_format.sample_rate_hz
                and audio_format.sample_rate_hz != self._RAW_SAMPLE_RATE_HZ
            ):
                audio_segment = audio_segment.set_frame_rate(
                    audio_format.sample_rate_hz
                )
            output_bytes: bytes = self._export_audio_segment(
                audio_segment=audio_segment,
                audio_format=audio_format,
            )
            output_bytes = self.audio_tail_silence(
                audio_bytes=output_bytes,
                audio_format=audio_format,
            )
            if test_mode:
                self._test_audio_bytes(
                    audio_bytes=output_bytes,
                    audio_format=audio_format,
                )
            return output_bytes
        except Exception as exc:  # pragma: no cover - depends on codecs
            _LOGGER.error(
                "Failed to convert Google TTS audio to %s: %s",
                audio_format.key,
                exc,
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to convert Google TTS audio to {audio_format.key}: {exc}"
            ) from exc

    def audio_tail_silence(
        self,
        *,
        audio_bytes: bytes,
        audio_format: AudioFormat,
        ms_silence: int = _DEFAULT_TAIL_SILENCE_MS,
    ) -> bytes:
        """
        Ensure the audio payload ends with the requested duration of silence.
        """
        if not audio_bytes:
            return audio_bytes

        target_silence_ms: int = max(ms_silence, 0)
        if target_silence_ms == 0:
            return audio_bytes

        try:
            requested_extension: str = audio_format.file_extension.lstrip(".")
            input_buffer: BytesIO = BytesIO(audio_bytes)
            input_buffer.seek(0)
            audio_segment: AudioSegment = AudioSegment.from_file(
                input_buffer, format=requested_extension
            )
        except Exception as exc:
            _LOGGER.warning(
                "Failed to inspect audio tail for %s output: %s",
                audio_format.key,
                exc,
                exc_info=True,
            )
            return audio_bytes

        if len(audio_segment) == 0:
            return audio_bytes

        tail_start_ms: int = max(len(audio_segment) - target_silence_ms, 0)
        tail_segment: AudioSegment = audio_segment[tail_start_ms:]
        tail_dbfs: float | None = tail_segment.dBFS

        if tail_dbfs is None or tail_dbfs <= self._TAIL_SILENCE_THRESHOLD_DBFS:
            return audio_bytes

        silence_segment: AudioSegment = AudioSegment.silent(
            duration=target_silence_ms,
            frame_rate=audio_segment.frame_rate,
        )
        if silence_segment.channels != audio_segment.channels:
            silence_segment = silence_segment.set_channels(audio_segment.channels)
        if silence_segment.sample_width != audio_segment.sample_width:
            silence_segment = silence_segment.set_sample_width(
                audio_segment.sample_width
            )

        padded_segment: AudioSegment = audio_segment.append(
            silence_segment,
            crossfade=0,
        )
        output_bytes: bytes = self._export_audio_segment(
            audio_segment=padded_segment,
            audio_format=audio_format,
        )
        return output_bytes

    def _test_audio_bytes(
        self,
        *,
        audio_bytes: bytes,
        audio_format: AudioFormat,
    ) -> None:
        """
        Inspect the opening samples for suspicious discontinuities or DC offset.

        Designed for interactive debugging; logs diagnostics and lets the caller
        decide how to remediate pops/clicks.
        """
        if not audio_bytes:
            _LOGGER.warning("TTS debug: received empty audio payload.")
            return

        sample_width: int = self._RAW_SAMPLE_WIDTH_BYTES
        if len(audio_bytes) < sample_width:
            _LOGGER.debug("TTS debug: payload shorter than one sample.")
            return

        try:
            # Examine the very first sample (and second, if available) to detect
            # abrupt jumps away from zero, which often manifest as clicks.
            first_sample: int = struct.unpack_from("<h", audio_bytes, 0)[0]
            second_sample: int | None = None
            if len(audio_bytes) >= sample_width * 2:
                second_sample = struct.unpack_from("<h", audio_bytes, sample_width)[0]

            # Evaluate the first ~10 ms of audio to see the max amplitude and mean.
            # A high mean indicates DC offset; a high max compared to the rest can
            # signal a transient pop.
            samples_to_check: int = min(
                len(audio_bytes) // sample_width,
                max(1, self._RAW_SAMPLE_RATE_HZ // 100),  # ≈ first 10 ms
            )
            window_format: str = f"<{samples_to_check}h"
            window_values: tuple[int, ...] = struct.unpack_from(
                window_format, audio_bytes, 0
            )
            max_abs: int = max(abs(value) for value in window_values)
            avg: float = sum(window_values) / len(window_values)

            leading_jump_ratio: float = (
                abs(first_sample) / max_abs if max_abs > 0 else 0.0
            )
            dc_offset_ratio: float = avg / max(1.0, max_abs)

            if leading_jump_ratio > 0.9:
                _LOGGER.info(
                    (
                        "TTS debug: pronounced edge detected (first_sample=%d, "
                        "window_max=%d, jump_ratio=%.2f). "
                        "Consider trimming or fading the opening frame."
                    ),
                    first_sample,
                    max_abs,
                    leading_jump_ratio,
                )
            elif abs(dc_offset_ratio) > 0.05:
                _LOGGER.info(
                    (
                        "TTS debug: DC offset observed (average_first_window=%.2f, "
                        "window_max=%d, offset_ratio=%.2f). "
                        "Consider high-pass filtering."
                    ),
                    avg,
                    max_abs,
                    dc_offset_ratio,
                )
            else:
                _LOGGER.debug(
                    (
                        "TTS debug: opening frame looks healthy "
                        "(first_sample=%d, second_sample=%s, window_max=%d, "
                        "avg=%.2f, samples=%d, format=%s)."
                    ),
                    first_sample,
                    second_sample,
                    max_abs,
                    avg,
                    samples_to_check,
                    audio_format.key,
                )

        except Exception as exc:
            _LOGGER.warning(
                "TTS debug: failed to inspect raw samples: %s",
                exc,
                exc_info=True,
            )
