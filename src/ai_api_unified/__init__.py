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
    AIBaseClassifier,
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
    AIClassificationChoice,
    AIClassificationLevel,
    AIClassificationResult,
    AIClassifierCapabilitiesBase,
    AICompletionsCapabilitiesBase,
    AICompletionsPromptParamsBase,
    AIEmbeddingsCapabilitiesBase,
    AIEmbeddingsMultimodalParams,
    AIFinishReason,
    AIIncludedMediaParamsBase,
    AIMediaReference,
    AIPromptCacheHint,
    AIPromptCacheRetention,
    AIPromptResult,
    AIProviderOrgInfoBase,
    AIProviderOrgInfoCapability,
    AIStructuredOutputResult,
    AITokenUsage,
    AITool,
    AIToolCall,
    AITurnResult,
    AIRoutePreview,
    AIVideoArtifact,
    AIVideoGenerationJob,
    AIVideoGenerationResult,
    AIVideoGenerationStatus,
    AIStructuredPrompt,
    RETRY_POLICY_DEFAULT,
    RETRY_POLICY_NONE,
    SupportedDataType,
)
from .ai_completions_exceptions import (
    StructuredOutputValidationError,
    StructuredResponseTokenLimitError,
)
from .ai_provider_exceptions import (
    AiProviderCapabilityUnsupportedError,
    AiFallbackReason,
    AiProviderRequestError,
)
from .ai_factory import AIFactory
from .completions.ai_router_completions import AiRouterCompletions
from .routing.router_config import (
    AIRouterClassifierConfig,
    AIRouterConfig,
    AIRouterTier,
    load_router_config,
)
from .completions.ai_fallback_completions import (
    DEFAULT_FALLBACK_REASONS,
    AIFallbackCandidate,
    AiFallbackCompletions,
)
from .routing.config_checks import AIConfigRequirements
from .completions.ai_fallback_cooldown import (
    AiCandidateCooldowns,
    AIFallbackCooldownPolicy,
)
from .thinking import (
    AIThinkingCapabilities,
    AIThinkingLevel,
    AIThinkingResolution,
)
from .middleware.cache_store import CacheStore, InMemoryCacheStore, set_cache_store
from .middleware.middleware_config import ResponseCacheSettingsModel
from .ai_provider_exceptions import AiGuardrailBlockedError
from .middleware.guardrails import reset_guardrails, set_guardrail_classifier
from .middleware.middleware_config import GuardrailsSettingsModel
from .middleware.response_cache import (
    cache_mode,
    reset_response_cache,
    response_cache_clear,
    response_cache_entries,
    response_cache_stats,
)
from .middleware.request_cost import (
    AIRequestCostLineItem,
    AIRequestCostReport,
    request_cost_scope,
)
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
    "AIFallbackCandidate",
    "AiFallbackCompletions",
    "AiCandidateCooldowns",
    "AIFallbackCooldownPolicy",
    "AIConfigRequirements",
    "AiRouterCompletions",
    "AIRouterClassifierConfig",
    "AIRouterConfig",
    "AIRouterTier",
    "load_router_config",
    "DEFAULT_FALLBACK_REASONS",
    "AIBase",
    "AIBaseClassifier",
    "AIBaseEmbeddings",
    "AIBaseCompletions",
    "AIBaseImages",
    "AIBaseImageProperties",
    "AIBaseVideos",
    "AIBaseVideoProperties",
    "AIClassificationChoice",
    "AIClassificationLevel",
    "AIClassificationResult",
    "AIClassifierCapabilitiesBase",
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
    "AIPromptCacheHint",
    "AIPromptCacheRetention",
    "AIProviderOrgInfoBase",
    "AIProviderOrgInfoCapability",
    "AIPromptResult",
    "AiGuardrailBlockedError",
    "CacheStore",
    "GuardrailsSettingsModel",
    "reset_guardrails",
    "set_guardrail_classifier",
    "InMemoryCacheStore",
    "ResponseCacheSettingsModel",
    "cache_mode",
    "reset_response_cache",
    "response_cache_clear",
    "response_cache_entries",
    "response_cache_stats",
    "set_cache_store",
    "AIRequestCostLineItem",
    "AIRequestCostReport",
    "AIStructuredOutputResult",
    "AIThinkingCapabilities",
    "AIThinkingLevel",
    "AIThinkingResolution",
    "AITokenUsage",
    "AITool",
    "AIToolCall",
    "AITurnResult",
    "AIRoutePreview",
    "AiProviderCapabilityUnsupportedError",
    "AiFallbackReason",
    "AiProviderRequestError",
    "RETRY_POLICY_DEFAULT",
    "RETRY_POLICY_NONE",
    "StructuredOutputValidationError",
    "StructuredResponseTokenLimitError",
    "SupportedDataType",
    "request_cost_scope",
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
