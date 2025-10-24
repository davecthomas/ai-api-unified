# src/ai_api_unified/voice/__init__.py
from __future__ import annotations

from .ai_voice_base import AIVoiceBase, AIVoiceCapabilities, AIVoiceSelectionBase
from .ai_voice_factory import AIVoiceFactory
from .ai_voice_openai import AIVoiceOpenAI, AIVoiceSelectionOpenAI
from .audio_models import AudioFormat
from ..util.utils import is_hex_enabled

has_gemini: bool = False
try:
    from .ai_voice_google import AIVoiceGoogle, AIVoiceSelectionGoogle

    has_gemini = True
except ImportError:
    pass

# ----- Optional providers (import if available; otherwise skip) -----
_has_azure: bool = False
try:
    from .ai_voice_azure import AIVoiceAzure, AIVoiceSelectionAzure  # type: ignore

    _has_azure = True
except ImportError:
    pass

_has_elevenlabs: bool = False
try:
    from .ai_voice_elevenlabs import AIVoiceElevenLabs, AIVoiceSelectionElevenLabs  # type: ignore

    _has_elevenlabs = True
except ImportError:
    pass

__all__: list[str] = [
    # base classes
    "AIVoiceSelectionBase",
    "AIVoiceCapabilities",
    "AIVoiceBase",
    # OpenAI
    "AIVoiceSelectionOpenAI",
    "AIVoiceOpenAI",
    # factory
    "AIVoiceFactory",
    # models & utils
    "AudioFormat",
    "is_hex_enabled",
]

# Export optional providers only if they actually imported (and poetry add 'ai-api-unified[<provider>]' )
if has_gemini:
    __all__.extend(["AIVoiceSelectionGoogle", "AIVoiceGoogle"])
if _has_azure:
    __all__.extend(["AIVoiceSelectionAzure", "AIVoiceAzure"])
if _has_elevenlabs:
    __all__.extend(["AIVoiceSelectionElevenLabs", "AIVoiceElevenLabs"])
