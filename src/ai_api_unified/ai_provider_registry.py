"""
Central AiProvider registry for optional dependency-backed engines.

This module stores provider metadata only. It intentionally does not import
optional provider SDK modules or concrete provider classes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .ai_provider_exceptions import AiProviderConfigurationError

AI_PROVIDER_CAPABILITY_COMPLETIONS: str = "completions"
AI_PROVIDER_CAPABILITY_EMBEDDINGS: str = "embeddings"
AI_PROVIDER_CAPABILITY_IMAGES: str = "images"
AI_PROVIDER_CAPABILITY_VOICE: str = "voice"

TypeAiProviderCapability = Literal[
    "completions",
    "embeddings",
    "images",
    "voice",
]
TypeAiProviderRegistryKey = tuple[TypeAiProviderCapability, str]


class AiProviderSpec(BaseModel):
    """
    Metadata describing how to load one provider implementation lazily.

    Args:
        str_capability: Capability category for this provider.
        str_engine: Engine selector value used by runtime configuration.
        str_module_path: Python module path used by lazy import.
        str_class_name: Class identifier resolved from imported module.
        str_required_extra: Poetry optional dependency extra required for use.
        str_consumer_install_command: Install command for downstream projects.
        str_local_install_command: Install command for local repository use.
        set_str_dependency_roots: Root module names considered valid dependency
            misses for this provider.

    Returns:
        A validated AiProviderSpec instance.
    """

    str_capability: TypeAiProviderCapability
    str_engine: str
    str_module_path: str
    str_class_name: str
    str_required_extra: str
    str_consumer_install_command: str
    str_local_install_command: str
    set_str_dependency_roots: set[str] = Field(default_factory=set)

    @field_validator("str_engine")
    @classmethod
    def _validate_str_engine(cls, value: str) -> str:
        """
        Normalizes engine identifiers to lowercase for deterministic lookup.

        Args:
            value: Raw engine string configured for this provider entry.

        Returns:
            A trimmed lowercase engine token.
        """
        str_engine_normalized: str = value.strip().lower()
        if not str_engine_normalized:
            raise ValueError("Provider engine must be a non-empty string.")
        # Normal return with normalized engine token.
        return str_engine_normalized


DICT_TUPLE_AI_PROVIDER_REGISTRY: dict[TypeAiProviderRegistryKey, AiProviderSpec] = {
    (
        AI_PROVIDER_CAPABILITY_COMPLETIONS,
        "openai",
    ): AiProviderSpec(
        str_capability=AI_PROVIDER_CAPABILITY_COMPLETIONS,
        str_engine="openai",
        str_module_path=("ai_api_unified.completions.ai_openai_completions"),
        str_class_name="AiOpenAICompletions",
        str_required_extra="openai",
        str_consumer_install_command=("poetry add 'ai-api-unified[openai]'"),
        str_local_install_command='poetry install --extras "openai"',
        set_str_dependency_roots={"openai"},
    ),
    (
        AI_PROVIDER_CAPABILITY_COMPLETIONS,
        "google-gemini",
    ): AiProviderSpec(
        str_capability=AI_PROVIDER_CAPABILITY_COMPLETIONS,
        str_engine="google-gemini",
        str_module_path=(
            "ai_api_unified.completions.ai_google_gemini_completions"
        ),
        str_class_name="GoogleGeminiCompletions",
        str_required_extra="google_gemini",
        str_consumer_install_command=(
            "poetry add 'ai-api-unified[google_gemini]'"
        ),
        str_local_install_command='poetry install --extras "google_gemini"',
        set_str_dependency_roots={"google", "google.genai", "google.api_core"},
    ),
    (
        AI_PROVIDER_CAPABILITY_COMPLETIONS,
        "llama",
    ): AiProviderSpec(
        str_capability=AI_PROVIDER_CAPABILITY_COMPLETIONS,
        str_engine="llama",
        str_module_path=(
            "ai_api_unified.completions.ai_bedrock_completions"
        ),
        str_class_name="AiBedrockCompletions",
        str_required_extra="bedrock",
        str_consumer_install_command=(
            "poetry add 'ai-api-unified[bedrock]'"
        ),
        str_local_install_command='poetry install --extras "bedrock"',
        set_str_dependency_roots={"boto3", "botocore"},
    ),
    (
        AI_PROVIDER_CAPABILITY_COMPLETIONS,
        "anthropic",
    ): AiProviderSpec(
        str_capability=AI_PROVIDER_CAPABILITY_COMPLETIONS,
        str_engine="anthropic",
        str_module_path=(
            "ai_api_unified.completions.ai_bedrock_completions"
        ),
        str_class_name="AiBedrockCompletions",
        str_required_extra="bedrock",
        str_consumer_install_command=(
            "poetry add 'ai-api-unified[bedrock]'"
        ),
        str_local_install_command='poetry install --extras "bedrock"',
        set_str_dependency_roots={"boto3", "botocore"},
    ),
    (
        AI_PROVIDER_CAPABILITY_COMPLETIONS,
        "mistral",
    ): AiProviderSpec(
        str_capability=AI_PROVIDER_CAPABILITY_COMPLETIONS,
        str_engine="mistral",
        str_module_path=(
            "ai_api_unified.completions.ai_bedrock_completions"
        ),
        str_class_name="AiBedrockCompletions",
        str_required_extra="bedrock",
        str_consumer_install_command=(
            "poetry add 'ai-api-unified[bedrock]'"
        ),
        str_local_install_command='poetry install --extras "bedrock"',
        set_str_dependency_roots={"boto3", "botocore"},
    ),
    (
        AI_PROVIDER_CAPABILITY_COMPLETIONS,
        "nova",
    ): AiProviderSpec(
        str_capability=AI_PROVIDER_CAPABILITY_COMPLETIONS,
        str_engine="nova",
        str_module_path=(
            "ai_api_unified.completions.ai_bedrock_completions"
        ),
        str_class_name="AiBedrockCompletions",
        str_required_extra="bedrock",
        str_consumer_install_command=(
            "poetry add 'ai-api-unified[bedrock]'"
        ),
        str_local_install_command='poetry install --extras "bedrock"',
        set_str_dependency_roots={"boto3", "botocore"},
    ),
    (
        AI_PROVIDER_CAPABILITY_COMPLETIONS,
        "cohere",
    ): AiProviderSpec(
        str_capability=AI_PROVIDER_CAPABILITY_COMPLETIONS,
        str_engine="cohere",
        str_module_path=(
            "ai_api_unified.completions.ai_bedrock_completions"
        ),
        str_class_name="AiBedrockCompletions",
        str_required_extra="bedrock",
        str_consumer_install_command=(
            "poetry add 'ai-api-unified[bedrock]'"
        ),
        str_local_install_command='poetry install --extras "bedrock"',
        set_str_dependency_roots={"boto3", "botocore"},
    ),
    (
        AI_PROVIDER_CAPABILITY_COMPLETIONS,
        "ai21",
    ): AiProviderSpec(
        str_capability=AI_PROVIDER_CAPABILITY_COMPLETIONS,
        str_engine="ai21",
        str_module_path=(
            "ai_api_unified.completions.ai_bedrock_completions"
        ),
        str_class_name="AiBedrockCompletions",
        str_required_extra="bedrock",
        str_consumer_install_command=(
            "poetry add 'ai-api-unified[bedrock]'"
        ),
        str_local_install_command='poetry install --extras "bedrock"',
        set_str_dependency_roots={"boto3", "botocore"},
    ),
    (
        AI_PROVIDER_CAPABILITY_COMPLETIONS,
        "rerank",
    ): AiProviderSpec(
        str_capability=AI_PROVIDER_CAPABILITY_COMPLETIONS,
        str_engine="rerank",
        str_module_path=(
            "ai_api_unified.completions.ai_bedrock_completions"
        ),
        str_class_name="AiBedrockCompletions",
        str_required_extra="bedrock",
        str_consumer_install_command=(
            "poetry add 'ai-api-unified[bedrock]'"
        ),
        str_local_install_command='poetry install --extras "bedrock"',
        set_str_dependency_roots={"boto3", "botocore"},
    ),
    (
        AI_PROVIDER_CAPABILITY_COMPLETIONS,
        "canvas",
    ): AiProviderSpec(
        str_capability=AI_PROVIDER_CAPABILITY_COMPLETIONS,
        str_engine="canvas",
        str_module_path=(
            "ai_api_unified.completions.ai_bedrock_completions"
        ),
        str_class_name="AiBedrockCompletions",
        str_required_extra="bedrock",
        str_consumer_install_command=(
            "poetry add 'ai-api-unified[bedrock]'"
        ),
        str_local_install_command='poetry install --extras "bedrock"',
        set_str_dependency_roots={"boto3", "botocore"},
    ),
    (
        AI_PROVIDER_CAPABILITY_EMBEDDINGS,
        "openai",
    ): AiProviderSpec(
        str_capability=AI_PROVIDER_CAPABILITY_EMBEDDINGS,
        str_engine="openai",
        str_module_path=("ai_api_unified.embeddings.ai_openai_embeddings"),
        str_class_name="AiOpenAIEmbeddings",
        str_required_extra="openai",
        str_consumer_install_command=("poetry add 'ai-api-unified[openai]'"),
        str_local_install_command='poetry install --extras "openai"',
        set_str_dependency_roots={"openai"},
    ),
    (
        AI_PROVIDER_CAPABILITY_EMBEDDINGS,
        "titan",
    ): AiProviderSpec(
        str_capability=AI_PROVIDER_CAPABILITY_EMBEDDINGS,
        str_engine="titan",
        str_module_path=("ai_api_unified.embeddings.ai_titan_embeddings"),
        str_class_name="AiTitanEmbeddings",
        str_required_extra="bedrock",
        str_consumer_install_command=(
            "poetry add 'ai-api-unified[bedrock]'"
        ),
        str_local_install_command='poetry install --extras "bedrock"',
        set_str_dependency_roots={"boto3", "botocore"},
    ),
    (
        AI_PROVIDER_CAPABILITY_EMBEDDINGS,
        "google-gemini",
    ): AiProviderSpec(
        str_capability=AI_PROVIDER_CAPABILITY_EMBEDDINGS,
        str_engine="google-gemini",
        str_module_path=(
            "ai_api_unified.embeddings.ai_google_gemini_embeddings"
        ),
        str_class_name="GoogleGeminiEmbeddings",
        str_required_extra="google_gemini",
        str_consumer_install_command=(
            "poetry add 'ai-api-unified[google_gemini]'"
        ),
        str_local_install_command='poetry install --extras "google_gemini"',
        set_str_dependency_roots={"google", "google.genai", "google.api_core"},
    ),
    (
        AI_PROVIDER_CAPABILITY_IMAGES,
        "openai",
    ): AiProviderSpec(
        str_capability=AI_PROVIDER_CAPABILITY_IMAGES,
        str_engine="openai",
        str_module_path="ai_api_unified.images.ai_openai_images",
        str_class_name="AIOpenAIImages",
        str_required_extra="openai",
        str_consumer_install_command=("poetry add 'ai-api-unified[openai]'"),
        str_local_install_command='poetry install --extras "openai"',
        set_str_dependency_roots={"openai"},
    ),
    (
        AI_PROVIDER_CAPABILITY_IMAGES,
        "bedrock",
    ): AiProviderSpec(
        str_capability=AI_PROVIDER_CAPABILITY_IMAGES,
        str_engine="bedrock",
        str_module_path="ai_api_unified.images.ai_bedrock_images",
        str_class_name="AINovaCanvasImages",
        str_required_extra="bedrock",
        str_consumer_install_command=(
            "poetry add 'ai-api-unified[bedrock]'"
        ),
        str_local_install_command='poetry install --extras "bedrock"',
        set_str_dependency_roots={"boto3", "botocore"},
    ),
    (
        AI_PROVIDER_CAPABILITY_IMAGES,
        "google-gemini",
    ): AiProviderSpec(
        str_capability=AI_PROVIDER_CAPABILITY_IMAGES,
        str_engine="google-gemini",
        str_module_path="ai_api_unified.images.ai_google_gemini_images",
        str_class_name="AIGoogleGeminiImages",
        str_required_extra="google_gemini",
        str_consumer_install_command=(
            "poetry add 'ai-api-unified[google_gemini]'"
        ),
        str_local_install_command='poetry install --extras "google_gemini"',
        set_str_dependency_roots={"google", "google.genai", "google.api_core"},
    ),
    (
        AI_PROVIDER_CAPABILITY_IMAGES,
        "nova",
    ): AiProviderSpec(
        str_capability=AI_PROVIDER_CAPABILITY_IMAGES,
        str_engine="nova",
        str_module_path="ai_api_unified.images.ai_bedrock_images",
        str_class_name="AINovaCanvasImages",
        str_required_extra="bedrock",
        str_consumer_install_command=(
            "poetry add 'ai-api-unified[bedrock]'"
        ),
        str_local_install_command='poetry install --extras "bedrock"',
        set_str_dependency_roots={"boto3", "botocore"},
    ),
    (
        AI_PROVIDER_CAPABILITY_IMAGES,
        "nova-canvas",
    ): AiProviderSpec(
        str_capability=AI_PROVIDER_CAPABILITY_IMAGES,
        str_engine="nova-canvas",
        str_module_path="ai_api_unified.images.ai_bedrock_images",
        str_class_name="AINovaCanvasImages",
        str_required_extra="bedrock",
        str_consumer_install_command=(
            "poetry add 'ai-api-unified[bedrock]'"
        ),
        str_local_install_command='poetry install --extras "bedrock"',
        set_str_dependency_roots={"boto3", "botocore"},
    ),
    (
        AI_PROVIDER_CAPABILITY_VOICE,
        "openai",
    ): AiProviderSpec(
        str_capability=AI_PROVIDER_CAPABILITY_VOICE,
        str_engine="openai",
        str_module_path="ai_api_unified.voice.ai_voice_openai",
        str_class_name="AIVoiceOpenAI",
        str_required_extra="openai",
        str_consumer_install_command=("poetry add 'ai-api-unified[openai]'"),
        str_local_install_command='poetry install --extras "openai"',
        set_str_dependency_roots={"openai"},
    ),
    (
        AI_PROVIDER_CAPABILITY_VOICE,
        "google",
    ): AiProviderSpec(
        str_capability=AI_PROVIDER_CAPABILITY_VOICE,
        str_engine="google",
        str_module_path="ai_api_unified.voice.ai_voice_google",
        str_class_name="AIVoiceGoogle",
        str_required_extra="google_gemini",
        str_consumer_install_command=(
            "poetry add 'ai-api-unified[google_gemini]'"
        ),
        str_local_install_command='poetry install --extras "google_gemini"',
        set_str_dependency_roots={
            "google",
            "google.genai",
            "google.api_core",
            "google.cloud",
        },
    ),
    (
        AI_PROVIDER_CAPABILITY_VOICE,
        "azure",
    ): AiProviderSpec(
        str_capability=AI_PROVIDER_CAPABILITY_VOICE,
        str_engine="azure",
        str_module_path="ai_api_unified.voice.ai_voice_azure",
        str_class_name="AIVoiceAzure",
        str_required_extra="azure_tts",
        str_consumer_install_command=(
            "poetry add 'ai-api-unified[azure_tts]'"
        ),
        str_local_install_command='poetry install --extras "azure_tts"',
        set_str_dependency_roots={"azure", "azure.cognitiveservices"},
    ),
    (
        AI_PROVIDER_CAPABILITY_VOICE,
        "elevenlabs",
    ): AiProviderSpec(
        str_capability=AI_PROVIDER_CAPABILITY_VOICE,
        str_engine="elevenlabs",
        str_module_path="ai_api_unified.voice.ai_voice_elevenlabs",
        str_class_name="AIVoiceElevenLabs",
        str_required_extra="elevenlabs",
        str_consumer_install_command=(
            "poetry add 'ai-api-unified[elevenlabs]'"
        ),
        str_local_install_command='poetry install --extras "elevenlabs"',
        set_str_dependency_roots={"elevenlabs"},
    ),
}


def get_ai_provider_spec(
    str_capability: TypeAiProviderCapability, str_engine: str
) -> AiProviderSpec:
    """
    Looks up provider metadata for the requested capability and engine pair.

    Args:
        str_capability: Provider capability bucket to resolve.
        str_engine: Engine selector token from runtime configuration.

    Returns:
        Matching AiProviderSpec for the capability/engine pair.
        Raises AiProviderConfigurationError when the pair is not registered.
    """
    str_engine_normalized: str = str_engine.strip().lower()
    tuple_registry_key: TypeAiProviderRegistryKey = (
        str_capability,
        str_engine_normalized,
    )
    ai_provider_spec: AiProviderSpec | None = DICT_TUPLE_AI_PROVIDER_REGISTRY.get(
        tuple_registry_key
    )
    if ai_provider_spec is None:
        raise AiProviderConfigurationError(
            f"Unsupported {str_capability} engine: {str_engine_normalized!r}"
        )
    # Normal return with provider metadata for the requested engine.
    return ai_provider_spec


def get_provider_spec(
    str_capability: TypeAiProviderCapability, str_engine: str
) -> AiProviderSpec:
    """
    Backward-compatible alias for get_ai_provider_spec.

    Args:
        str_capability: AiProvider capability bucket to resolve.
        str_engine: Engine selector token from runtime configuration.

    Returns:
        Matching AiProviderSpec for the capability/engine pair.
    """
    # Normal return using the canonical AiProvider lookup function.
    return get_ai_provider_spec(str_capability, str_engine)
