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
    AIBatchItemStatus,
    AIBatchJob,
    AIBatchRequestItem,
    AIBatchResultItem,
    AIBatchStatus,
    AICompletionsCapabilitiesBase,
    AICompletionsPromptParamsBase,
    AIEmbeddingsCapabilitiesBase,
    AIEmbeddingsMultimodalParams,
    AIFinishReason,
    AIIncludedMediaParamsBase,
    AIMediaReference,
    AIStructuredOutputResult,
    AITokenUsage,
    AITool,
    AIToolCall,
    AITurnResult,
    AIVideoArtifact,
    AIVideoGenerationJob,
    AIVideoGenerationResult,
    AIVideoGenerationStatus,
    AIStructuredPrompt,
    RETRY_POLICY_DEFAULT,
    RETRY_POLICY_NONE,
    SupportedDataType,
)
from .ai_completions_exceptions import StructuredResponseTokenLimitError
from .ai_provider_exceptions import (
    AiProviderCapabilityUnsupportedError,
    AiProviderRequestError,
)
from .ai_factory import AIFactory
from .pricing import (
    AIModelInfo,
    AIModelPricing,
    AITokenRates,
    ModelLifecycleStatus,
    PricingUnit,
    get_model_info,
    get_model_pricing,
)
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
    "AIBatchRequestItem",
    "AIBatchResultItem",
    "AIBatchJob",
    "AIBatchStatus",
    "AIBatchItemStatus",
    "AICompletionsCapabilitiesBase",
    "AICompletionsPromptParamsBase",
    "AIEmbeddingsCapabilitiesBase",
    "AIEmbeddingsMultimodalParams",
    "AIFinishReason",
    "AIIncludedMediaParamsBase",
    "AIStructuredOutputResult",
    "AITokenUsage",
    "AITool",
    "AIToolCall",
    "AITurnResult",
    "AiProviderCapabilityUnsupportedError",
    "AiProviderRequestError",
    "RETRY_POLICY_DEFAULT",
    "RETRY_POLICY_NONE",
    "StructuredResponseTokenLimitError",
    "SupportedDataType",
    "AIModelPricing",
    "AIModelInfo",
    "AITokenRates",
    "ModelLifecycleStatus",
    "PricingUnit",
    "get_model_pricing",
    "get_model_info",
    "AIVoiceSelectionBase",
    "AIVoiceCapabilities",
    "AIVoiceFactory",
    "AIVoiceBase",
    "AudioFormat",
]
