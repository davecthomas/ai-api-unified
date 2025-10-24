# src/ai_api_unified/voice/ai_voice_factory.py
import logging

_LOGGER: logging.Logger = logging.getLogger(__name__)

import os

from .ai_voice_base import AIVoiceBase
from .ai_voice_openai import AIVoiceOpenAI

# ElevenLabs voice (optional extra)
try:
    from .ai_voice_elevenlabs import AIVoiceElevenLabs  # type: ignore
except ImportError:
    AIVoiceElevenLabs = None  # type: ignore[assignment]
    ELEVENLABS_AVAILABLE: bool = False
else:
    ELEVENLABS_AVAILABLE: bool = True

# Google voice (optional extra)
try:
    from .ai_voice_google import AIVoiceGoogle  # type: ignore
except ImportError:
    AIVoiceGoogle = None  # type: ignore[assignment]
    GOOGLE_VOICE_AVAILABLE: bool = False
else:
    GOOGLE_VOICE_AVAILABLE: bool = True

# Azure TTS (optional extra)
try:
    from .ai_voice_azure import AIVoiceAzure  # type: ignore
except ImportError:
    AIVoiceAzure = None  # type: ignore[assignment]
    AZURE_TTS_AVAILABLE: bool = False
else:
    AZURE_TTS_AVAILABLE: bool = True


class AIVoiceFactory:
    """Factory to create AI Voice instances based on environment configuration."""

    @staticmethod
    def create() -> AIVoiceBase:
        engine = os.getenv("AI_VOICE_ENGINE", "").strip().lower()
        match engine:
            case "elevenlabs":
                if not ELEVENLABS_AVAILABLE:
                    _LOGGER.warning(
                        "ElevenLabs voice requested but the optional extra is not installed. Install it with: poetry add 'ai-api-unified[elevenlabs]'"
                    )
                    raise RuntimeError(
                        "ElevenLabs voice requested but the optional extra is not installed. "
                        "Install it with: poetry add 'ai-api-unified[elevenlabs]'"
                    )
                return AIVoiceElevenLabs(engine=engine)  # type: ignore[call-arg]

            case "google":
                if not GOOGLE_VOICE_AVAILABLE:
                    _LOGGER.warning(
                        "Google voice requested but the optional extra is not installed. Install it with: poetry add 'ai-api-unified[google_gemini]'"
                    )
                    raise RuntimeError(
                        "Google voice requested but the optional extra is not installed. "
                        "Install it with: poetry add 'ai-api-unified[google_gemini]'"
                    )
                return AIVoiceGoogle(engine=engine)  # type: ignore[call-arg]

            case "openai":
                return AIVoiceOpenAI(engine=engine)

            case "azure":
                if not AZURE_TTS_AVAILABLE:
                    _LOGGER.warning(
                        "Azure TTS requested but the optional extra is not installed. Install it with: poetry add 'ai-api-unified[azure_tts]'"
                    )
                    raise RuntimeError(
                        "Azure TTS requested but the optional extra is not installed. "
                        "Install it with: poetry add 'ai-api-unified[azure_tts]'"
                    )
                return AIVoiceAzure(engine=engine)  # type: ignore[call-arg]
            case _:
                _LOGGER.error(f"Unsupported AI_VOICE_ENGINE: {engine}")
                raise ValueError(f"Unsupported AI_VOICE_ENGINE: {engine}")
