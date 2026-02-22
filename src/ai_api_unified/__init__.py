"""
ai_api_unified · Unified access layer for LLM providers.

Public API surface:
  - __version__             Package version string
  - AIFactory               Factory for completions & embeddings clients
  - AIBase                  Base LLM client abstraction
  - AIBaseEmbeddings        Embeddings-specific abstraction
  - AIBaseCompletions       Completions-specific abstraction
  - AIStructuredPrompt      Base class for structured-prompt models
  - AiOpenAICompletions     OpenAI completions back-end
  - AiBedrockCompletions    Amazon Bedrock completions back-end
  - AiOpenAIEmbeddings      OpenAI embeddings back-end
  - AiTitanEmbeddings       Amazon Titan embeddings back-end
  - GoogleGeminiCompletions Google Gemini completions back-end (if available)
  Voice:
    - AIVoiceFactory          Factory for voice clients
    - AIVoiceBase             Base voice client abstraction
    - AIVoiceSelectionBase    Base voice selection abstraction
    - AIVoiceCapabilities     Voice capabilities description
    - AIVoiceGoogle           Google Text-to-Speech back-end
    - AIVoiceSelectionGoogle  Google Text-to-Speech voice selection
    - AIVoiceOpenAI           OpenAI Whisper back-end
    - AIVoiceSelectionOpenAI  OpenAI Whisper voice selection
    - AIVoiceAzure            Microsoft Azure Text-to-Speech back-end (if available)
    - AIVoiceSelectionAzure   Microsoft Azure Text-to-Speech voice selection (if available)
    - AIVoiceElevenLabs       ElevenLabs Text-to-Speech back-end (if available)
    - AIVoiceSelectionElevenLabs ElevenLabs Text-to-Speech voice selection (if available)
    - AudioFormat             Audio format description
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    # Installed package metadata
    __version__: str = _pkg_version(__name__)
except PackageNotFoundError:
    # Editable/develop mode fallback
    from .__version__ import __version__  # type: ignore

# Factory
# Core abstractions & prompt base
from .ai_base import (
    AIBase,
    AIBaseCompletions,
    AIBaseEmbeddings,
    AIBaseImages,
    AIBaseImageProperties,
    AIStructuredPrompt,
    AICompletionsCapabilitiesBase,
    AICompletionsPromptParamsBase,
    SupportedDataType,
)
from .ai_factory import AIFactory

# Concrete back-ends
from .ai_openai_base import AIOpenAIBase
from .completions.ai_openai_completions import (
    AICompletionsCapabilitiesOpenAI,
    AICompletionsPromptParamsOpenAI,
    AiOpenAICompletions,
)
from .embeddings.ai_openai_embeddings import AiOpenAIEmbeddings

from .voice.ai_voice_base import AIVoiceBase, AIVoiceCapabilities, AIVoiceSelectionBase
from .voice.ai_voice_factory import AIVoiceFactory
from .voice.ai_voice_openai import AIVoiceOpenAI, AIVoiceSelectionOpenAI
from .voice.audio_models import AudioFormat
from .images.ai_openai_images import AIOpenAIImages, AIOpenAIImageProperties

# Optional packages
has_gemini: bool = False
try:
    from .voice.ai_voice_google import AIVoiceGoogle, AIVoiceSelectionGoogle
    from .completions.ai_google_gemini_completions import GoogleGeminiCompletions
    from .completions.ai_google_gemini_capabilities import (
        AICompletionsPromptParamsGoogle,
    )

    has_gemini: bool = True

except ImportError:
    pass

has_bedrock: bool = False
try:
    from .ai_bedrock_base import AIBedrockBase

    from .images.ai_bedrock_images import (
        AINovaCanvasImageProperties,
        AINovaCanvasImages,
    )
    from .completions.ai_bedrock_completions import AiBedrockCompletions
    from .embeddings.ai_titan_embeddings import AiTitanEmbeddings

    has_bedrock = True
except ImportError:
    pass

__all__: list[str] = [
    "__version__",
    "AIFactory",
    "AIBase",
    "AIBaseEmbeddings",
    "AIBaseCompletions",
    "AIBaseImages",
    "AIBaseImageProperties",
    "AIStructuredPrompt",
    "AIBedrockBase",
    "AICompletionsCapabilitiesBase",
    "AICompletionsPromptParamsBase",
    "SupportedDataType",
    "AICompletionsCapabilitiesOpenAI",
    "AICompletionsPromptParamsOpenAI",
    "AiOpenAICompletions",
    "AiOpenAIEmbeddings",
    "AIOpenAIBase",
    "AIOpenAIImages",
    "AIOpenAIImageProperties",
    "AIVoiceSelectionBase",
    "AIVoiceCapabilities",
    "AIVoiceFactory",
    "AIVoiceBase",
    "AIVoiceSelectionOpenAI",
    "AIVoiceOpenAI",
    "AudioFormat",
]
if has_gemini:
    __all__.extend(
        [
            "GoogleGeminiCompletions",
            "AICompletionsPromptParamsGoogle",
            "AIVoiceSelectionGoogle",
            "AIVoiceGoogle",
        ]
    )
if has_bedrock:
    __all__.extend(
        [
            "AiBedrockCompletions",
            "AiTitanEmbeddings",
            "AINovaCanvasImages",
            "AINovaCanvasImageProperties",
        ]
    )
