# ai_factory.py
import logging

_LOGGER: logging.Logger = logging.getLogger(__name__)

from typing import Type, Any

from ai_api_unified.ai_base import AIBaseImages
from ai_api_unified.images.ai_openai_images import AIOpenAIImages

from .ai_base import AIBase, AIBaseCompletions
from .completions.ai_openai_completions import AiOpenAICompletions  # type: ignore

# Bedrock images (optional)
try:
    from .images.ai_bedrock_images import AINovaCanvasImages

    BEDROCK_IMAGES_AVAILABLE: bool = True
except ImportError:
    AINovaCanvasImages: Any = None  # type: ignore[assignment]
    BEDROCK_IMAGES_AVAILABLE: bool = False

# Bedrock completions (optional: requires boto3 via extras "bedrock")
try:
    from .completions.ai_bedrock_completions import AiBedrockCompletions

    BEDROCK_AVAILABLE: bool = True
except ImportError:  # ImportError or any transitive import errors
    AiBedrockCompletions: Any = None  # type: ignore[assignment]
    BEDROCK_AVAILABLE: bool = False

# Titan embeddings (optional: requires boto3 via extras "bedrock")
try:
    from .embeddings.ai_titan_embeddings import AiTitanEmbeddings

    TITAN_AVAILABLE: bool = True
except ImportError:
    AiTitanEmbeddings: Any = None  # type: ignore[assignment]
    TITAN_AVAILABLE: bool = False

from .embeddings.ai_openai_embeddings import AiOpenAIEmbeddings  # type: ignore
from .util.env_settings import EnvSettings  # type: ignore

# Conditionally import Google Gemini classes
try:
    from .completions.ai_google_gemini_completions import GoogleGeminiCompletions
    from .embeddings.ai_google_gemini_embeddings import GoogleGeminiEmbeddings
    from .images.ai_google_gemini_images import AIGoogleGeminiImages

    GOOGLE_GEMINI_AVAILABLE = True
except ImportError:
    GOOGLE_GEMINI_AVAILABLE = False
    GoogleGeminiCompletions: Any = None  # type: ignore[assignment]
    GoogleGeminiEmbeddings: Any = None  # type: ignore[assignment]
    AIGoogleGeminiImages: Any = None  # type: ignore[assignment]


class AIFactory:
    # Mapping of (client_type, engine) to the implementing class
    _CLIENT_MAP: dict[tuple[str, str], Type[AIBase]] = {
        # Always-on core OpenAI engines
        (AIBase.CLIENT_TYPE_COMPLETIONS, "openai"): AiOpenAICompletions,
        (AIBase.CLIENT_TYPE_EMBEDDING, "openai"): AiOpenAIEmbeddings,
    }

    # Bedrock-backed completions (optional)
    if BEDROCK_AVAILABLE:
        _CLIENT_MAP.update(
            {
                (AIBase.CLIENT_TYPE_COMPLETIONS, "llama"): AiBedrockCompletions,
                (AIBase.CLIENT_TYPE_COMPLETIONS, "anthropic"): AiBedrockCompletions,
                (AIBase.CLIENT_TYPE_COMPLETIONS, "mistral"): AiBedrockCompletions,
                (AIBase.CLIENT_TYPE_COMPLETIONS, "nova"): AiBedrockCompletions,
                (AIBase.CLIENT_TYPE_COMPLETIONS, "cohere"): AiBedrockCompletions,
                (AIBase.CLIENT_TYPE_COMPLETIONS, "ai21"): AiBedrockCompletions,
                (AIBase.CLIENT_TYPE_COMPLETIONS, "rerank"): AiBedrockCompletions,
                (AIBase.CLIENT_TYPE_COMPLETIONS, "canvas"): AiBedrockCompletions,
            }
        )  # type: ignore

    # Titan embeddings (optional)
    if TITAN_AVAILABLE:
        _CLIENT_MAP.update(
            {
                (AIBase.CLIENT_TYPE_EMBEDDING, "titan"): AiTitanEmbeddings,
            }
        )  # type: ignore

    # Google Gemini (optional)
    if GOOGLE_GEMINI_AVAILABLE:
        _CLIENT_MAP.update(
            {
                (
                    AIBase.CLIENT_TYPE_COMPLETIONS,
                    "google-gemini",
                ): GoogleGeminiCompletions,
                (AIBase.CLIENT_TYPE_EMBEDDING, "google-gemini"): GoogleGeminiEmbeddings,
                (AIBase.CLIENT_TYPE_IMAGES, "google-gemini"): AIGoogleGeminiImages,
            }
        )  # type: ignore

    @staticmethod
    def get_ai_completions_client(
        model_name: str | None = None,
        completions_engine: str | None = None,
    ) -> AIBaseCompletions:
        """
        Instantiate and return the appropriate AIBaseCompletions subclass.

        :param model_name: optional override for COMPLETIONS_MODEL_NAME
        :param completions_engine: optional override for COMPLETIONS_ENGINE
        """
        env = EnvSettings()

        # 1. Determine engine: explicit override wins, otherwise read from env
        if completions_engine:
            engine = completions_engine.strip().lower()
        else:
            engine = env.get_setting("COMPLETIONS_ENGINE", "openai").strip().lower()

        # 2. Determine model name: explicit override wins, otherwise read from env
        if model_name is None:
            model_name = env.get_setting("COMPLETIONS_MODEL_NAME", "").strip()

        # 4. Dispatch to the correct subclass
        if engine == "openai":
            client: AIBaseCompletions = AiOpenAICompletions(model=model_name or "")
        elif engine == "google-gemini":
            if not GOOGLE_GEMINI_AVAILABLE:
                _LOGGER.warning(
                    "Google Gemini support not available. Install google_gemini extra: poetry add 'ai_api_unified[google_gemini]'"
                )
                raise RuntimeError(
                    "Google Gemini support not available. "
                    "Install google_gemini extra: poetry add 'ai_api_unified[google_gemini]'"
                )
            client = GoogleGeminiCompletions(model=model_name or "")
        # All Bedrock-backed families use AiBedrockCompletions
        elif engine in {
            "nova",  # Amazon Nova (Pro/Micro/Canvas)
            "llama",  # Meta Llama family
            "anthropic",  # Anthropic Claude family
            "mistral",  # Mistral family
            "cohere",  # Cohere Command family
            "ai21",  # AI21 Jamba family
            "rerank",  # Amazon Rerank
            "canvas",  # Nova Canvas
        }:
            if not BEDROCK_AVAILABLE:
                _LOGGER.warning(
                    "Bedrock support not available. Install bedrock extra: poetry add 'ai_api_unified[bedrock]'"
                )
                raise RuntimeError(
                    "Bedrock completions requested but the Bedrock extra is not installed. "
                    "Install with:  poetry add 'ai-api-unified[bedrock]'"
                )
            client = AiBedrockCompletions(model=model_name or "")

        else:
            _LOGGER.error(f"Unsupported COMPLETIONS engine: {engine!r}")
            raise ValueError(f"Unsupported COMPLETIONS engine: {engine!r}")

        return client

    @staticmethod
    def get_ai_embedding_client(
        embedding_engine: str | None = None,
        model_name: str | None = None, # type: ignore
    ) -> AIBase:
        """
        Instantiate and return the appropriate AIBase subclass for embeddings.
        """
        env: EnvSettings = EnvSettings()

        if embedding_engine:
            engine: str = embedding_engine.strip().lower()
        else:
            # Default to Google if not specified
            engine: str = env.get_setting("EMBEDDING_ENGINE", "openai").strip().lower()

        # If model_name is not provided, read from environment
        if model_name is None:
            model_name: str = env.get_setting("EMBEDDING_MODEL_NAME", "").strip()
        dim: int = int(env.get_setting("EMBEDDING_DIMENSIONS", "1024"))

        if engine == "titan":
            if not TITAN_AVAILABLE:
                _LOGGER.warning(
                    "Titan embeddings requested but the Bedrock extra is not installed. Install with: poetry add 'ai-api-unified[bedrock]'"
                )
                raise RuntimeError(
                    "Titan embeddings requested but the Bedrock extra is not installed. "
                    "Install with: poetry add 'ai-api-unified[bedrock]'"
                )
            return AiTitanEmbeddings(model=model_name, dimensions=dim)
        if engine == "openai":
            return AiOpenAIEmbeddings(model=model_name, dimensions=dim)
        if engine == "google-gemini":
            if not GOOGLE_GEMINI_AVAILABLE:
                _LOGGER.warning(
                    "Google Gemini support not available. Install google_gemini extra: poetry add 'ai-api-unified[google_gemini]'"
                )
                raise RuntimeError(
                    "Google Gemini support not available. "
                    "Install google_gemini extra: poetry add 'ai-api-unified[google_gemini]'"
                )
            return GoogleGeminiEmbeddings(model=model_name, dimensions=dim)

        _LOGGER.error(f"Unsupported EMBEDDING engine: {engine}")
        raise ValueError(f"Unsupported EMBEDDING engine: {engine}")

    @staticmethod
    def list_completion_models(client: AIBaseCompletions) -> list[str]:
        """
        Return the list of completion-model names supported by the given client.
        """
        if not isinstance(client, AIBaseCompletions):
            raise TypeError(f"Expected AIBaseCompletions, got {type(client).__name__}")
        return client.list_model_names

    @staticmethod
    def get_ai_images_client(
        image_model: str | None = None,
    ) -> AIBaseImages:
        """
        Instantiate and return the appropriate AIBaseImages subclass.

        :param image_model: optional override for IMAGE_MODEL_NAME
        """
        env = EnvSettings()
        image_engine: str = env.get_setting("IMAGE_ENGINE", "openai").strip().lower()
        if image_model is None:
            image_model = env.get_setting("IMAGE_MODEL_NAME", "").strip()

        if image_engine in {"", "openai"}:
            client: AIBaseImages = AIOpenAIImages(model=image_model or "")
            return client

        if image_engine in {"bedrock", "nova", "nova-canvas"}:
            if not BEDROCK_IMAGES_AVAILABLE:
                _LOGGER.warning(
                    "Bedrock image support not available. "
                    "Install the 'bedrock' extra to enable Nova Canvas."
                )
                raise RuntimeError(
                    "Bedrock Nova Canvas requested but the bedrock extra is not installed."
                )
            client = AINovaCanvasImages(model=image_model or "")
            return client

        if image_engine == "google-gemini":
            if not GOOGLE_GEMINI_AVAILABLE:
                _LOGGER.warning(
                    "Google Gemini image support not available. "
                    "Install the 'google_gemini' extra."
                )
                raise RuntimeError(
                    "Google Gemini images requested but the google_gemini extra is not installed."
                )
            client = AIGoogleGeminiImages(model=image_model or "")
            return client

        _LOGGER.error(f"Unsupported IMAGE engine: {image_engine!r}")
        raise ValueError(f"Unsupported IMAGE engine: {image_engine!r}")
