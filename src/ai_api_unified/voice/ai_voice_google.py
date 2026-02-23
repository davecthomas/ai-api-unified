# src/ai_api_unified/voice/ai_voice_google.py
"""Google Cloud Text-to-Speech speech synthesis implementation.
This module uses the Google Cloud Text-to-Speech API for text-to-speech synthesis.
Authentication relies on Application Default Credentials discovered via the
``GOOGLE_APPLICATION_CREDENTIALS`` environment variable. All voice synthesis requests use
standard TTS input (text).

Reference: https://cloud.google.com/text-to-speech/docs
"""

from __future__ import annotations

from io import BytesIO
import logging
import time
from typing import Any, ClassVar

GOOGLE_DEPENDENCIES_AVAILABLE: bool = False
try:  # pragma: no cover - exercised in integration environments
    from google import genai
    from google.api_core import exceptions as gexc
    from google.api_core.client_options import ClientOptions
    from google.cloud import speech_v1p1beta1 as speech
    from google.cloud import texttospeech
    from ai_api_unified.ai_google_base import AIGoogleBase

    GOOGLE_DEPENDENCIES_AVAILABLE = True
except ImportError:  # pragma: no cover - handled gracefully downstream
    GOOGLE_DEPENDENCIES_AVAILABLE = False

    if GOOGLE_DEPENDENCIES_AVAILABLE:
        from pydantic import PrivateAttr
        from ai_api_unified.util.env_settings import EnvSettings
        from ai_api_unified.util._lazy_pydub import (
            AudioSegment,
            CouldntDecodeError,
        )
        from ai_api_unified.voice.ai_voice_base import (
            AIVoiceBase,
            AIVoiceCapabilities,
            AIVoiceModelBase,
            AIVoiceSelectionBase,
        )
        from ai_api_unified.voice.audio_models import AudioFormat
        from ai_api_unified.voice.ai_voice_google_gemini_voices import (
            GEMINI_LOCALES,
            GEMINI_VOICE_NAMES,
        )

        _LOGGER: logging.Logger = logging.getLogger(__name__)
        # SET logger to DEBUG to see detailed logs from this module
        _LOGGER.setLevel(logging.DEBUG)  # Uncomment for detailed logging

        class AIVoiceSelectionGoogle(AIVoiceSelectionBase):
            """Google Gemini TTS voice selection container."""

        class AIVoiceGoogle(AIVoiceBase, AIGoogleBase):
            """Google AI Gemini TTS implementation."""

            MIN_SPEED: ClassVar[float] = 0.25
            MAX_SPEED: ClassVar[float] = 2.0
            _STATIC_LATENCY_SECONDS: ClassVar[float] = 0.13
            _SMOOTHING_RATIO: ClassVar[float] = 0.75

            DEFAULT_TTS_MODEL: ClassVar[str] = "gemini-2.5-pro-tts"
            FALLBACK_TTS_MODEL: ClassVar[str] = "gemini-2.5-flash-tts"
            LIST_IDENTIFYING_GEMINI_TTS_MODELS: ClassVar[list[str]] = ["gemini", "tts"]
            STT_MAX_SYNC_REQUEST_BYTES: ClassVar[int] = 10 * 1024 * 1024
            STT_DEFAULT_CHUNK_LENGTH_MS: ClassVar[int] = 30_000
            STT_DEFAULT_OVERLAP_MS: ClassVar[int] = 0

            _MODEL_DEFINITIONS: ClassVar[list[dict[str, str]]] = [
                {
                    "name": "gemini-2.5-pro-tts",
                    "display_name": "Gemini 2.5 Pro TTS",
                    "description": "High fidelity Gemini speech with instruction prompts.",
                    "is_default": True,
                },
                {
                    "name": "gemini-2.5-flash-tts",
                    "display_name": "Gemini 2.5 Flash TTS",
                    "description": "Cost-optimized Gemini speech with instruction prompts.",
                    "is_default": False,
                },
            ]

            _DEFAULT_LANGUAGE: ClassVar[str] = "en"
            _DEFAULT_LOCALE: ClassVar[str] = "en-US"
            _DEFAULT_ACCENT: ClassVar[str | None] = None
            _DEFAULT_SPEAKING_RATE: ClassVar[float] = 1.0

            _tts_client: texttospeech.TextToSpeechClient | None = PrivateAttr(default=None)

            def __init__(self, engine: str, **kwargs: Any) -> None:
                super().__init__(engine=engine, **kwargs)
                env: EnvSettings = EnvSettings()
                
                auth_method: str = env.get_setting("GOOGLE_AUTH_METHOD", "service_account").lower()
                api_key: str | None = env.get_setting("GOOGLE_GEMINI_API_KEY", None)

                if self._tts_client is None:
                    try:
                        if auth_method == "api_key":
                            if not api_key:
                                raise RuntimeError("GOOGLE_GEMINI_API_KEY is not set but GOOGLE_AUTH_METHOD=api_key.")
                            self._tts_client = texttospeech.TextToSpeechClient(
                                client_options=ClientOptions(api_key=api_key)
                            )
                        else:
                            self._tts_client = texttospeech.TextToSpeechClient()
                    except Exception as init_error:
                        _LOGGER.error(
                            "Failed to initialize Google Text-to-Speech client: %s",
                            init_error,
                            exc_info=True,
                        )
                        raise

            configured_model_setting: str | None = env.get_setting(
                "DEFAULT_GEMINI_TTS_MODEL",
                self.DEFAULT_TTS_MODEL,
            )
            available_model_names: set[str] = {
                model_definition["name"] for model_definition in self._MODEL_DEFINITIONS
            }
            resolved_model: str = (
                configured_model_setting
                if configured_model_setting
                else self.DEFAULT_TTS_MODEL
            )
            if resolved_model not in available_model_names:
                _LOGGER.warning(
                    "Unknown Gemini TTS model '%s'. Falling back to %s.",
                    resolved_model,
                    self.DEFAULT_TTS_MODEL,
                )
                resolved_model = self.DEFAULT_TTS_MODEL
            self.default_model_id: str = resolved_model

            self.list_output_formats: list[AudioFormat] = self._build_output_formats()
            self.default_audio_format: AudioFormat = self._determine_default_format()

            self.common_vendor_capabilities: AIVoiceCapabilities = (
                self._build_common_capabilities()
            )
            self.list_models_capabilities: list[AIVoiceModelBase] = (
                self._initialize_models_and_capabilities()
            )
            self.selected_model = next(
                (
                    model_entry
                    for model_entry in self.list_models_capabilities
                    if model_entry.name == self.default_model_id
                ),
                self.get_default_model(),
            )
            for model_entry in self.list_models_capabilities:
                model_entry.is_default = model_entry.name == self.default_model_id

            self.list_available_voices: list[AIVoiceSelectionGoogle] = (
                self._build_voice_catalog()
            )
            self.selected_voice: AIVoiceSelectionGoogle = self._resolve_voice(None)

            self.dict_model_prices: dict[str, float] | None = None

        def _build_output_formats(self) -> list[AudioFormat]:
            list_formats: list[AudioFormat] = [
                AudioFormat(
                    key="mp3_24000",
                    description="MP3 - 24 kHz (CBR 64 kbps)",
                    file_extension=".mp3",
                    sample_rate_hz=self._MP3_SAMPLE_RATE_HZ,
                ),
                AudioFormat(
                    key="ogg_opus_24000",
                    description="Ogg-Opus - 24 kHz (64 kbps VBR)",
                    file_extension=".ogg",
                    sample_rate_hz=self._OGG_SAMPLE_RATE_HZ,
                ),
                AudioFormat(
                    key="wav_linear16_24000",
                    description="WAV (Linear16 - 24 kHz)",
                    file_extension=".wav",
                    sample_rate_hz=self._RAW_SAMPLE_RATE_HZ,
                ),
                AudioFormat(
                    key="mulaw_8000",
                    description="mu-law - 8 kHz (telephony)",
                    file_extension=".wav",
                    sample_rate_hz=self._MULAW_SAMPLE_RATE_HZ,
                ),
                AudioFormat(
                    key=self._DEFAULT_AUDIO_FORMAT_KEY,
                    description="WAV (Linear16 - 48 kHz)",
                    file_extension=".wav",
                    sample_rate_hz=self._LINEAR16_SAMPLE_RATE_HZ,
                ),
            ]
            return list_formats

        def _determine_default_format(self) -> AudioFormat:
            for audio_format in self.list_output_formats:
                if audio_format.key == self._DEFAULT_AUDIO_FORMAT_KEY:
                    return audio_format
            return self.list_output_formats[0]

        def _build_common_capabilities(self) -> AIVoiceCapabilities:
            return AIVoiceCapabilities(
                supports_ssml=False,
                supports_streaming=False,
                supports_speech_to_text=True,
                supported_languages=[self._DEFAULT_LANGUAGE],
                supported_locales=[self._DEFAULT_LOCALE],
                supported_audio_formats=self.list_output_formats,
                supports_custom_voice=False,
                supports_emotion_control=True,
                supports_prosody_settings=False,
                supports_viseme_events=False,
                supports_word_timestamps=False,
                supports_batch_synthesis=False,
                supports_pronunciation_lexicon=False,
                supports_audio_profiles=False,
                min_speaking_rate=self.MIN_SPEED,
                max_speaking_rate=self.MAX_SPEED,
            )

        def _initialize_models_and_capabilities(self) -> list[AIVoiceModelBase]:
            list_models: list[AIVoiceModelBase] = []
            for model_def in self._MODEL_DEFINITIONS:
                model_capabilities: AIVoiceCapabilities = (
                    self.common_vendor_capabilities.model_copy(deep=True)
                )
                model_capabilities.supports_emotion_control = True
                list_models.append(
                    AIVoiceModelBase(
                        name=model_def["name"],
                        display_name=model_def.get("display_name"),
                        description=model_def.get("description"),
                        capabilities=model_capabilities,
                        is_default=model_def.get("is_default", False),
                    )
                )
            return list_models

        def _build_voice_catalog(self) -> list[AIVoiceSelectionGoogle]:
            target_model: str | None = (
                self.selected_model.name
                if self.selected_model
                else self.default_model_id
            )
            if self.is_gemini_tts_model(target_model):
                return self._build_gemini_voice_catalog()

            list_voices: list[AIVoiceSelectionGoogle] = []

            try:
                if self._tts_client is None:
                    self._tts_client = texttospeech.TextToSpeechClient()

                response: texttospeech.ListVoicesResponse = (
                    self._tts_client.list_voices()
                )

                for voice_metadata in response.voices:
                    voice_id: str = voice_metadata.name
                    if not voice_id:
                        continue

                    language_codes: list[str] = list(
                        voice_metadata.language_codes or [self._DEFAULT_LOCALE]
                    )
                    ssml_gender: str | None = None
                    if (
                        voice_metadata.ssml_gender
                        and voice_metadata.ssml_gender
                        != texttospeech.SsmlVoiceGender.SSML_VOICE_GENDER_UNSPECIFIED
                    ):
                        ssml_gender = texttospeech.SsmlVoiceGender(
                            voice_metadata.ssml_gender
                        ).name.lower()

                    for language_code in language_codes:
                        normalized_locale: str = language_code or self._DEFAULT_LOCALE
                        normalized_language: str = normalized_locale.split("-")[0]
                        list_voices.append(
                            AIVoiceSelectionGoogle(
                                voice_id=voice_id,
                                voice_name=voice_id,
                                language=normalized_language,
                                accent=None,
                                locale=normalized_locale,
                                gender=ssml_gender,
                            )
                        )

            except Exception as voice_error:
                _LOGGER.error(
                    "Failed to retrieve voices from Google Text-to-Speech: %s",
                    voice_error,
                    exc_info=True,
                )
                raise

            if not list_voices:
                raise RuntimeError(
                    "Google Text-to-Speech returned no available voices."
                )

            return list_voices

        def is_gemini_tts_model(self, model_name: str | None) -> bool:
            """
            Check if the provided model name should use the Gemini static catalog.
            """
            if model_name is None:
                return False
            normalized_name: str = model_name.lower()
            return all(
                token in normalized_name
                for token in self.LIST_IDENTIFYING_GEMINI_TTS_MODELS
            )

        def _build_gemini_voice_catalog(self) -> list[AIVoiceSelectionGoogle]:
            """
            Construct every documented Gemini voice/locale combination.
            """
            list_voices: list[AIVoiceSelectionGoogle] = []
            for voice_name, voice_gender in GEMINI_VOICE_NAMES:
                for locale_code in GEMINI_LOCALES:
                    language_code: str = (
                        locale_code.split("-")[0] if "-" in locale_code else locale_code
                    )
                    list_voices.append(
                        AIVoiceSelectionGoogle(
                            voice_id=voice_name,
                            voice_name=voice_name,
                            language=language_code,
                            accent=None,
                            locale=locale_code,
                            gender=voice_gender,
                        )
                    )
            return list_voices

        def _resolve_voice(
            self,
            voice: AIVoiceSelectionBase | None,
        ) -> AIVoiceSelectionGoogle:
            if voice is None or not getattr(voice, "voice_id", ""):
                if not self.list_available_voices:
                    raise RuntimeError("No voices available for Google Gemini TTS.")
                default_voice: AIVoiceSelectionGoogle = self.list_available_voices[0]
                return default_voice
            return AIVoiceSelectionGoogle(
                voice_id=voice.voice_id,
                voice_name=voice.voice_name,
                language=voice.language if voice.language else self._DEFAULT_LANGUAGE,
                accent=voice.accent,
                locale=voice.locale if voice.locale else self._DEFAULT_LOCALE,
                gender=voice.gender if voice.gender else None,
            )

        def text_to_voice(
            self,
            *,
            text_to_convert: str,
            voice: AIVoiceSelectionGoogle,
            audio_format: AudioFormat,
            speaking_rate: float = _DEFAULT_SPEAKING_RATE,
            use_ssml: bool = False,
        ) -> bytes:
            if use_ssml:
                _LOGGER.warning(
                    "SSML is not supported for Google Gemini TTS; ignoring flag.",
                )
            if speaking_rate != self._DEFAULT_SPEAKING_RATE:
                _LOGGER.warning(
                    "Google Gemini TTS ignores speaking_rate adjustments; value %s will be ignored.",
                    speaking_rate,
                )
            return self.text_to_voice_with_emotion_prompt(
                emotion_prompt=None,
                text_to_convert=text_to_convert,
                voice=voice,
                audio_format=audio_format,
                speaking_rate=self._DEFAULT_SPEAKING_RATE,
            )

        def text_to_voice_with_emotion_prompt(
            self,
            *,
            emotion_prompt: str | None,
            text_to_convert: str,
            voice: AIVoiceSelectionGoogle,
            audio_format: AudioFormat,
            speaking_rate: float = _DEFAULT_SPEAKING_RATE,
        ) -> bytes:
            if speaking_rate != self._DEFAULT_SPEAKING_RATE:
                _LOGGER.warning(
                    "Google Gemini TTS ignores speaking_rate adjustments; value %s will be ignored.",
                    speaking_rate,
                )
            if not text_to_convert or not text_to_convert.strip():
                raise ValueError("text_to_convert must be a non-empty string")

            resolved_voice: AIVoiceSelectionGoogle = self._resolve_voice(voice)
            prompt_parts: list[str] = []
            if emotion_prompt and emotion_prompt.strip():
                prompt_parts.append(emotion_prompt.strip())
            prompt_parts.append(text_to_convert.strip())
            prompt_text: str = self._PROMPT_SEPARATOR.join(prompt_parts)
            prompt_preview: str = (
                prompt_text[:200] + "…" if len(prompt_text) > 200 else prompt_text
            )

            model_name: str = (
                self.selected_model.name
                if self.selected_model
                else self.default_model_id
            )

            def _generate_audio() -> bytes:
                synthesis_kwargs: dict[str, str] = {
                    "text": text_to_convert.strip(),
                }
                if emotion_prompt and emotion_prompt.strip():
                    synthesis_kwargs["prompt"] = emotion_prompt.strip()
                synthesis_input: texttospeech.SynthesisInput = (
                    texttospeech.SynthesisInput(**synthesis_kwargs)
                )

                voice_params: texttospeech.VoiceSelectionParams = (
                    texttospeech.VoiceSelectionParams(
                        language_code=resolved_voice.locale or self._DEFAULT_LOCALE,
                        name=resolved_voice.voice_id,
                        model_name=model_name,
                    )
                )

                audio_config: texttospeech.AudioConfig = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self._RAW_SAMPLE_RATE_HZ,
                )

                try:
                    _LOGGER.debug(
                        "Google TTS request dispatched: model=%s voice=%s prompt_preview=%r",
                        model_name,
                        resolved_voice.voice_id,
                        prompt_preview,
                    )
                    response: texttospeech.SynthesizeSpeechResponse = (
                        self._tts_client.synthesize_speech(
                            input=synthesis_input,
                            voice=voice_params,
                            audio_config=audio_config,
                        )
                    )
                    audio_content: bytes = response.audio_content
                    if not audio_content:
                        raise RuntimeError(
                            "Google Text-to-Speech returned empty audio content."
                        )
                    return audio_content
                except gexc.GoogleAPICallError as api_error:
                    status_code: int | None = api_error.code if api_error else None

                    _LOGGER.error(
                        "Google TTS API error: model=%s voice=%s status=%s prompt_preview=%r error=%s",
                        model_name,
                        resolved_voice.voice_id,
                        status_code,
                        prompt_preview,
                        api_error,
                        exc_info=True,
                    )
                    raise
                except Exception as exc:
                    _LOGGER.error(
                        "Google TTS synthesize_speech failed: model=%s voice=%s prompt_preview=%r error=%s",
                        model_name,
                        resolved_voice.voice_id,
                        prompt_preview,
                        exc,
                        exc_info=True,
                    )
                    raise

            try:
                gemini_audio_bytes: bytes = self._retry_with_exponential_backoff(
                    _generate_audio
                )
            except RuntimeError as runtime_error:
                _LOGGER.error(
                    "Google TTS API call failed: model=%s voice=%s prompt_preview=%r error=%s",
                    model_name,
                    resolved_voice.voice_id,
                    prompt_preview,
                    runtime_error,
                    exc_info=True,
                )
                raise
            except Exception as exc:
                _LOGGER.error(
                    "Google TTS API call raised unexpected error: model=%s voice=%s prompt_preview=%r error=%s",
                    model_name,
                    resolved_voice.voice_id,
                    prompt_preview,
                    exc,
                    exc_info=True,
                )
                raise RuntimeError(f"Gemini TTS API call failed: {exc}") from exc

            return self._convert_audio_bytes(
                audio_bytes=gemini_audio_bytes,
                audio_format=audio_format,
            )

        def stream_audio(
            self,
            text: str,
            voice: AIVoiceSelectionGoogle | None = None,
        ) -> bytes:
            resolved_voice: AIVoiceSelectionGoogle = self._resolve_voice(voice)
            return self.text_to_voice(
                text_to_convert=text,
                voice=resolved_voice,
                audio_format=self.default_audio_format,
            )

        def compute_adjusted_speed(
            self,
            *,
            current_speed: float,
            current_duration_s: float,
            target_duration_s: float,
        ) -> float:
            pure_speech_current: float = max(
                0.01,
                current_duration_s - self._STATIC_LATENCY_SECONDS,
            )
            pure_speech_target: float = max(
                0.01,
                target_duration_s - self._STATIC_LATENCY_SECONDS,
            )
            factor: float = pure_speech_current / pure_speech_target
            smoothed_factor: float = 1.0 + (factor - 1.0) * self._SMOOTHING_RATIO
            proposed_speed: float = current_speed * smoothed_factor
            bounded_speed: float = max(
                self.MIN_SPEED, min(self.MAX_SPEED, proposed_speed)
            )
            return float(bounded_speed)

        def audition_voices(
            self,
            ai_voice_client: AIVoiceBase,
            text: str = "Hello from Google Gemini TTS.",
            voice_id_group: str | None = None,
            pause_between_auditions: bool = False,
            list_locales: list[str] | None = None,
        ) -> None:
            list_locales_resolved: list[str] = (
                list_locales if list_locales is not None else [self._DEFAULT_LOCALE]
            )
            super().audition_voices(
                ai_voice_client=ai_voice_client,
                text=text,
                voice_id_group=voice_id_group,
                pause_between_auditions=pause_between_auditions,
                list_locales=list_locales_resolved,
            )

        def _recognize_with_retry(
            self,
            speech_client: speech.SpeechClient,
            recognition_config: speech.RecognitionConfig,
            recognition_audio: speech.RecognitionAudio,
            logger: logging.Logger,
            max_retries: int = 3,
        ) -> speech.RecognizeResponse:
            """
            Call Google STT once, retrying on quota or transient errors with
            exponential back-off. Payload-size errors are surfaced to the caller.
            """

            retry_count: int = 0
            while True:
                try:
                    return speech_client.recognize(
                        config=recognition_config,
                        audio=recognition_audio,
                    )
                except gexc.ResourceExhausted as exc:
                    if retry_count >= max_retries:
                        raise RuntimeError(
                            "Google Speech-to-Text quota exhausted."
                        ) from exc
                    retry_count += 1
                    sleep_seconds: float = 2**retry_count
                    logger.warning(
                        "Speech-to-Text quota hit; retrying in %.1f s (%d/%d)",
                        sleep_seconds,
                        retry_count,
                        max_retries,
                    )
                    time.sleep(sleep_seconds)
                except gexc.ServiceUnavailable as exc:
                    if retry_count >= max_retries:
                        raise RuntimeError(
                            "Google Speech-to-Text service unavailable."
                        ) from exc
                    retry_count += 1
                    sleep_seconds = 2**retry_count
                    logger.warning(
                        "Speech-to-Text service unavailable; retrying in %.1f s (%d/%d)",
                        sleep_seconds,
                        retry_count,
                        max_retries,
                    )
                    time.sleep(sleep_seconds)

        def speech_to_text(
            self,
            audio_bytes: bytes,
            language: str | None = None,
        ) -> str:
            """
            Transcribe ``audio_bytes`` using Google Speech-to-Text (synchronous).
            """

            logger: logging.Logger = logging.getLogger(__name__)
            language_code: str = (language or "en-US").lower()

            try:
                audio_master: AudioSegment = AudioSegment.from_file(
                    BytesIO(audio_bytes)
                )
            except CouldntDecodeError as exc:
                logger.error("Audio decode failed: %s", exc)
                raise ValueError(f"Could not decode audio: {exc}") from exc

            env = EnvSettings()
            auth_method = env.get_setting("GOOGLE_AUTH_METHOD", "service_account").lower()
            if auth_method == "api_key":
                api_key = env.get_setting("GOOGLE_GEMINI_API_KEY", None)
                if not api_key:
                    raise RuntimeError("GOOGLE_GEMINI_API_KEY is not set but GOOGLE_AUTH_METHOD=api_key.")
                speech_client = speech.SpeechClient(client_options=ClientOptions(api_key=api_key))
            else:
                speech_client = speech.SpeechClient()

            def _build_config(sample_rate_hertz: int) -> speech.RecognitionConfig:
                return speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=sample_rate_hertz,
                    language_code=language_code,
                    enable_automatic_punctuation=True,
                )

            chunk_length_ms: int = self.STT_DEFAULT_CHUNK_LENGTH_MS
            overlap_ms: int = self.STT_DEFAULT_OVERLAP_MS
            list_output_lines: list[str] = []
            start_ms: int = 0

            while start_ms < len(audio_master):
                current_window_ms: int = chunk_length_ms
                while True:
                    end_ms: int = min(start_ms + current_window_ms, len(audio_master))
                    segment_audio: AudioSegment = audio_master[start_ms:end_ms]

                    if segment_audio.duration_seconds > 55:
                        current_window_ms //= 2
                        continue

                    bytes_buffer: BytesIO = BytesIO()
                    segment_audio.export(bytes_buffer, format="wav")
                    bytes_segment: bytes = bytes_buffer.getvalue()

                    if len(bytes_segment) <= self.STT_MAX_SYNC_REQUEST_BYTES:
                        break
                    current_window_ms //= 2
                    if current_window_ms < 5_000:
                        raise RuntimeError(
                            "Unable to create a chunk under 10 MB – consider "
                            "lower sample rate or asynchronous STT."
                        )

                bytes_buffer.seek(0)
                recognition_config: speech.RecognitionConfig = _build_config(
                    sample_rate_hertz=segment_audio.frame_rate
                )
                recognition_audio: speech.RecognitionAudio = speech.RecognitionAudio(
                    content=bytes_buffer.read()
                )

                try:
                    response: speech.RecognizeResponse = self._recognize_with_retry(
                        speech_client,
                        recognition_config,
                        recognition_audio,
                        logger,
                    )
                except gexc.InvalidArgument as exc:
                    if "10485760" in str(exc):
                        chunk_length_ms = max(5_000, chunk_length_ms // 2)
                        logger.warning(
                            "Google flagged payload >10 MB; reducing window to %s ms",
                            chunk_length_ms,
                        )
                        continue
                    raise

                if response.results:
                    for res in response.results:
                        list_output_lines.append(res.alternatives[0].transcript)

                start_ms += current_window_ms - overlap_ms

            return "\n".join(list_output_lines)
