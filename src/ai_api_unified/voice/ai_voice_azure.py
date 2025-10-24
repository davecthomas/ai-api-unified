# src/ai_api_unified/voice/ai_voice_azure.py
from __future__ import annotations

import logging
import os
import secrets
import time
from typing import Any, ClassVar

AZURE_DEPENDENCIES_AVAILABLE: bool = False
try:
    import azure.cognitiveservices.speech as speechsdk

    AZURE_DEPENDENCIES_AVAILABLE = True
except ImportError as import_error:
    AZURE_DEPENDENCIES_AVAILABLE = False

if AZURE_DEPENDENCIES_AVAILABLE:
    from .ai_voice_base import (
        AIVoiceBase,
        AIVoiceCapabilities,
        AIVoiceSelectionBase,
        AIVoiceModelBase,
    )
    from .audio_models import AudioFormat

    logger = logging.getLogger(__name__)

    class AIVoiceSelectionAzure(AIVoiceSelectionBase):
        """Container describing an Azure voice selection."""

    class AIVoiceAzure(AIVoiceBase):
        """Azure Cognitive Speech Services text-to-speech implementation."""

        MIN_SPEED: ClassVar[float] = 0.5
        MAX_SPEED: ClassVar[float] = 2.0

        _MODEL_DEFINITIONS: ClassVar[list[dict[str, Any]]] = [
            {
                "name": "neural",
                "display_name": "Neural (Standard)",
                "description": "High-quality, standard neural voices.",
                "is_default": True,
            },
            {
                "name": "neural_hd",
                "display_name": "Neural HD (DragonHD)",
                "description": "Enterprise-only, high-definition neural voices.",
                "is_default": False,
            },
            {
                "name": "openai_neural",
                "display_name": "OpenAI Neural",
                "description": "Azure-exposed standard OpenAI voices.",
                "is_default": False,
            },
            {
                "name": "openai_hd",
                "display_name": "OpenAI NeuralHD",
                "description": "Azure-exposed high-definition OpenAI voices.",
                "is_default": False,
            },
        ]

        def __init__(self, *, engine: str, **kwargs: Any) -> None:
            super().__init__(engine=engine, **kwargs)
            api_key: str = os.getenv("MICROSOFT_COGNITIVE_SERVICES_API_KEY", "")
            if not api_key:
                raise RuntimeError("MICROSOFT_COGNITIVE_SERVICES_API_KEY is not set")
            region: str | None = os.getenv("MICROSOFT_COGNITIVE_SERVICES_REGION")
            if not region:
                raise RuntimeError("MICROSOFT_COGNITIVE_SERVICES_REGION is not set")

            self._speech_config: speechsdk.SpeechConfig = speechsdk.SpeechConfig(
                subscription=api_key,
                region=region or "",
            )
            self.client: speechsdk.SpeechSynthesizer | None = None

            self.list_output_formats: list[AudioFormat] = [
                AudioFormat(
                    key="pcm_48000", description="PCM – 48 kHz", file_extension=".wav"
                ),
                AudioFormat(
                    key="mp3_48000", description="MP3 – 48 kHz", file_extension=".mp3"
                ),
                AudioFormat(
                    key="pcm_24000", description="PCM – 24 kHz", file_extension=".wav"
                ),
                AudioFormat(
                    key="mp3_24000", description="MP3 – 24 kHz", file_extension=".mp3"
                ),
            ]
            self.default_audio_format = next(
                fmt for fmt in self.list_output_formats if fmt.key == "pcm_48000"
            )

            self.common_vendor_capabilities = AIVoiceCapabilities(
                supports_ssml=True,
                supports_streaming=False,  # Not exposed in this wrapper
                supports_speech_to_text=False,  # Not implemented in this class
                supported_languages=["en", "es"],
                supported_locales=["en-US", "es-US", "es-MX", "es-LA"],
                supported_audio_formats=self.list_output_formats,
                supports_prosody_settings=True,
                supports_pronunciation_lexicon=True,
                min_speaking_rate=self.MIN_SPEED,
                max_speaking_rate=self.MAX_SPEED,
                supports_custom_voice=True,
                supports_emotion_control=False,
                supports_viseme_events=True,
                supports_word_timestamps=True,  # Supported via boundary events
                supports_batch_synthesis=True,
                supports_audio_profiles=True,
            )

            self.list_models_capabilities = self._initialize_models_and_capabilities()
            # Model selection
            self.selected_model: AIVoiceModelBase = self.get_default_model()

            self.list_available_voices: list[AIVoiceSelectionAzure] = (
                self._build_voice_catalog()
            )
            if self.list_available_voices:
                first = self.list_available_voices[0]
                self.selected_voice = AIVoiceSelectionAzure(
                    voice_id=first.voice_id, voice_name=first.voice_name
                )
            else:
                self.selected_voice = None

        def _initialize_models_and_capabilities(self) -> list[AIVoiceModelBase]:
            """Creates a structured list of models from the definitions registry."""
            model_list = []
            for model_def in self._MODEL_DEFINITIONS:
                # Each model gets a unique copy of the common capabilities.
                model_caps = self.common_vendor_capabilities.model_copy(deep=True)

                # NOTE: Add any model-specific capability customizations here.
                # For now, all Azure models share the common capabilities.

                model_list.append(
                    AIVoiceModelBase(
                        name=model_def["name"],
                        display_name=model_def.get("display_name"),
                        description=model_def.get("description"),
                        capabilities=model_caps,
                        is_default=model_def.get("is_default", False),
                    )
                )
            return model_list

        def _build_voice_catalog(self) -> list[AIVoiceSelectionAzure]:
            catalog: list[AIVoiceSelectionAzure] = []
            try:
                if not self.client:
                    self.client = speechsdk.SpeechSynthesizer(
                        speech_config=self._speech_config, audio_config=None
                    )
                voices_result = self.client.get_voices_async().get()
                for voice in voices_result.voices:
                    if (
                        voice.locale
                        in self.common_vendor_capabilities.supported_locales
                    ):
                        catalog.append(
                            AIVoiceSelectionAzure(
                                voice_id=voice.short_name,
                                voice_name=voice.local_name or voice.short_name,
                                language=voice.locale.split("-")[0],
                                accent="american",
                                locale=voice.locale,
                                gender=(
                                    voice.gender.name.lower() if voice.gender else None
                                ),
                            )
                        )
            except Exception as exc:
                logging.error("Unable to list Azure voices: %s", exc)
            return catalog

        def _format_to_sdk(
            self, audio_format: AudioFormat
        ) -> speechsdk.SpeechSynthesisOutputFormat:
            mapping: dict[str, speechsdk.SpeechSynthesisOutputFormat] = {
                "pcm_48000": speechsdk.SpeechSynthesisOutputFormat.Riff48Khz16BitMonoPcm,
                "mp3_48000": speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3,
                "pcm_24000": speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm,
                "mp3_24000": speechsdk.SpeechSynthesisOutputFormat.Audio24Khz160KBitRateMonoMp3,
            }
            if audio_format.key not in mapping:
                raise ValueError(f"Unsupported audio format: {audio_format.key}")
            return mapping[audio_format.key]

        def _should_retry(self, is_rate_limit: bool = False) -> bool:
            """
            Increment attempt_index, decide how long to sleep, log, and return False if maxed out.
            """
            # Note: This retry logic is simplified. A more robust implementation
            # would not rely on instance variables that are reset on each call.
            self.attempt_index += 1
            if self.attempt_index >= self.max_attempts:
                logger.error("Max attempts reached. Aborting...")
                return False

            if is_rate_limit:
                sleep_seconds: float = 30.0
            else:
                jitter: float = 0.5 + secrets.randbelow(100) / 100.0
                sleep_seconds = self.backoff_seconds * jitter
                self.backoff_seconds *= self.backoff_multiplier

            prefix = "Rate limit error. " if is_rate_limit else ""
            logger.warning(
                f"{prefix}Retrying attempt "
                f"{self.attempt_index}/{self.max_attempts} "
                f"after sleeping {sleep_seconds:.1f}s…"
            )
            time.sleep(sleep_seconds)
            return True

        def text_to_voice(
            self,
            *,
            text_to_convert: str,
            voice: AIVoiceSelectionAzure,
            audio_format: AudioFormat,
            speaking_rate: float = 1.0,
            use_ssml: bool = False,
            max_attempts: int = 3,
            initial_backoff_seconds: float = 1.0,
            backoff_multiplier: int = 2,
        ) -> bytes:
            if speaking_rate != 1.0 and not use_ssml:
                raise ValueError(
                    "speaking_rate different from 1.0 requires use_ssml=True"
                )

            self._speech_config.speech_synthesis_voice_name = voice.voice_id
            self._speech_config.speech_synthesis_language = voice.locale
            self._speech_config.set_speech_synthesis_output_format(
                self._format_to_sdk(audio_format)
            )

            # Initialize client here to ensure config is fresh for each call
            self.client = speechsdk.SpeechSynthesizer(
                speech_config=self._speech_config, audio_config=None
            )

            if use_ssml:
                rate_prosody = (
                    f"<prosody rate='{speaking_rate:.0%}'>{text_to_convert}</prosody>"
                    if speaking_rate != 1.0
                    else text_to_convert
                )
                payload = (
                    f"<speak version='1.0' xml:lang='{voice.locale}'>"
                    f"<voice name='{voice.voice_id}'>{rate_prosody}</voice></speak>"
                )
                invoke_synthesis = lambda: self.client.speak_ssml_async(payload).get()
            else:
                payload = text_to_convert
                invoke_synthesis = lambda: self.client.speak_text_async(payload).get()

            attempt_index: int = 0
            last_error_message: str | None = None

            while attempt_index < max_attempts:
                attempt_index += 1
                try:
                    synthesis_result: speechsdk.SpeechSynthesisResult = (
                        invoke_synthesis()
                    )

                    if (
                        synthesis_result.reason
                        == speechsdk.ResultReason.SynthesizingAudioCompleted
                    ):
                        return bytes(synthesis_result.audio_data)

                    cancellation = synthesis_result.cancellation_details
                    last_error_message = (
                        f"Azure TTS canceled: {cancellation.reason}. "
                        f"Details: {cancellation.error_details}"
                    )
                    logger.warning(last_error_message)

                    if (
                        cancellation.reason == speechsdk.CancellationReason.Error
                        and "429" in cancellation.error_details
                    ):
                        # Simple retry for rate limit
                        time.sleep(30)
                        continue

                    raise RuntimeError(last_error_message)

                except Exception as error:
                    last_error_message = str(error)
                    if "401" in last_error_message:
                        raise  # Abort on auth errors

                    if attempt_index >= max_attempts:
                        break

                    # Simplified exponential backoff
                    sleep_seconds = initial_backoff_seconds * (
                        backoff_multiplier ** (attempt_index - 1)
                    )
                    logger.warning(
                        f"Retrying in {sleep_seconds:.1f}s... (Attempt {attempt_index}/{max_attempts})"
                    )
                    time.sleep(sleep_seconds)

            raise RuntimeError(
                f"Azure TTS failed after {max_attempts} attempts. "
                f"Last error: {last_error_message}"
            )

        def stream_audio(
            self, text: str, voice: AIVoiceSelectionAzure | None = None
        ) -> bytes:
            selected_voice = voice or self.selected_voice or self.get_default_voice()
            return self.text_to_voice(
                text_to_convert=text,
                voice=selected_voice,
                audio_format=self.default_audio_format,
            )

        def speech_to_text(
            self, audio_bytes: bytes, language: str | None = None
        ) -> str:
            raise NotImplementedError("speech_to_text() is not supported for Azure TTS")

        def audition_voices(
            self,
            ai_voice_client: AIVoiceBase,
            text: str = "I'm excited for my Azure speech audition. The quick brown fox jumps over the lazy dog.",
            voice_id_group: str | None = None,
            pause_between_auditions: bool = False,
            list_locales: list[str] = ["en-US"],
        ) -> None:
            if "es-US" in list_locales or "es-MX" in list_locales:
                text = "Estoy emocionado por mi audición de voz de Azure. El rápido zorro marrón salta sobre el perro perezoso."
            super().audition_voices(
                ai_voice_client=ai_voice_client,
                text=text,
                voice_id_group=voice_id_group,
                pause_between_auditions=pause_between_auditions,
                list_locales=list_locales,
            )
