"""
ai_api_unified public API exports.

This root module intentionally exports only stable base interfaces and factory
entry points. Concrete optional provider classes are not re-exported from this
namespace. Use the factories for runtime provider selection, or import concrete
providers directly from their implementation modules when needed.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__: str = _pkg_version(__name__)
except PackageNotFoundError:
    from .__version__ import __version__

from .ai_base import (
    AIBase,
    AIBaseCompletions,
    AIBaseEmbeddings,
    AIBaseImageProperties,
    AIBaseImages,
    AIBaseVideoProperties,
    AIBaseVideos,
    AICompletionsCapabilitiesBase,
    AICompletionsPromptParamsBase,
    AIMediaReference,
    AIVideoArtifact,
    AIVideoGenerationJob,
    AIVideoGenerationResult,
    AIVideoGenerationStatus,
    AIStructuredPrompt,
    SupportedDataType,
)
from .ai_completions_exceptions import StructuredResponseTokenLimitError
from .ai_factory import AIFactory
from .voice.ai_voice_base import AIVoiceBase, AIVoiceCapabilities, AIVoiceSelectionBase
from .voice.ai_voice_factory import AIVoiceFactory
from .voice.audio_models import AudioFormat

__all__: list[str] = [
    "__version__",
    "AIFactory",
    "AIBase",
    "AIBaseEmbeddings",
    "AIBaseCompletions",
    "AIBaseImages",
    "AIBaseImageProperties",
    "AIBaseVideos",
    "AIBaseVideoProperties",
    "AIMediaReference",
    "AIVideoArtifact",
    "AIVideoGenerationJob",
    "AIVideoGenerationResult",
    "AIVideoGenerationStatus",
    "AIStructuredPrompt",
    "AICompletionsCapabilitiesBase",
    "AICompletionsPromptParamsBase",
    "StructuredResponseTokenLimitError",
    "SupportedDataType",
    "AIVoiceSelectionBase",
    "AIVoiceCapabilities",
    "AIVoiceFactory",
    "AIVoiceBase",
    "AudioFormat",
]
