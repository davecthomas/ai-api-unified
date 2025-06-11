# ai_factory.py

from typing import List, Type, Optional
from .ai_base import AIBase, AIBaseCompletions
from .completions.ai_bedrock_completions import AiBedrockCompletions  # type: ignore
from .completions.ai_openai_completions import AiOpenAICompletions  # type: ignore
from .embeddings.ai_openai_embeddings import AiOpenAIEmbeddings  # type: ignore
from .embeddings.ai_titan_embeddings import AiTitanEmbeddings  # type: ignore
from .util.env_settings import EnvSettings  # type: ignore


class AIFactory:
    # Mapping of (client_type, engine) to the implementing class
    _CLIENT_MAP: dict[tuple[str, str], Type[AIBase]] = {
        (AIBase.CLIENT_TYPE_COMPLETIONS, "nova"): AiBedrockCompletions,
        (AIBase.CLIENT_TYPE_COMPLETIONS, "openai"): AiOpenAICompletions,
        (AIBase.CLIENT_TYPE_EMBEDDING, "titan"): AiTitanEmbeddings,
        (AIBase.CLIENT_TYPE_EMBEDDING, "openai"): AiOpenAIEmbeddings,
    }

    @staticmethod
    def get_ai_client(client_type: str = AIBase.CLIENT_TYPE_COMPLETIONS) -> AIBase:
        """
        Instantiate and return an AIBase client (embeddings or completions)
        based on environment settings.
        """
        env: EnvSettings = EnvSettings()
        # Determine engine and model name from env vars
        if client_type == AIBase.CLIENT_TYPE_COMPLETIONS:
            engine: str = (
                env.get_setting("COMPLETIONS_ENGINE", "openai").strip().lower()
            )
            model: str = env.get_setting("COMPLETIONS_MODEL_NAME", "").strip()
        elif client_type == AIBase.CLIENT_TYPE_EMBEDDING:
            engine: str = env.get_setting("EMBEDDING_ENGINE", "titan").strip().lower()
            model: str = env.get_setting("EMBEDDING_MODEL_NAME", "").strip()
        else:
            raise RuntimeError(f"Unsupported client_type: {client_type!r}")

        # Parse embedding dimensions for both types
        dim: int = int(env.get_setting("EMBEDDING_DIMENSIONS", "1024"))

        # Look up the appropriate client class
        cls: Optional[Type[AIBase]] = AIFactory._CLIENT_MAP.get((client_type, engine))
        if cls is None:
            raise RuntimeError(f"Unsupported {client_type} engine: {engine}")

        # Instantiate with model name and dimensions
        return cls(model=model, dimensions=dim)

    @staticmethod
    def get_ai_completions_client(
        client_type: str = AIBase.CLIENT_TYPE_COMPLETIONS,
        model_name: Optional[str] = None,
    ) -> AIBaseCompletions:
        """
        Instantiate and return the appropriate AIBaseCompletions subclass.
        """
        env: EnvSettings = EnvSettings()
        if client_type != AIBase.CLIENT_TYPE_COMPLETIONS:
            raise RuntimeError(f"Unsupported client_type: {client_type!r}")

        engine: str = env.get_setting("COMPLETIONS_ENGINE", "openai").strip().lower()

        # Allow override of model_name; otherwise read from env
        if model_name is None:
            model_name = env.get_setting("COMPLETIONS_MODEL_NAME", "").strip()

        dim: int = int(env.get_setting("EMBEDDING_DIMENSIONS", "1024"))

        if engine == "nova":
            return AiBedrockCompletions(model=model_name, dimensions=dim)
        if engine == "openai":
            return AiOpenAICompletions(model=model_name, dimensions=dim)

        raise RuntimeError(f"Unsupported COMPLETIONS engine: {engine}")

    @staticmethod
    def get_ai_embedding_client(
        client_type: str = AIBase.CLIENT_TYPE_EMBEDDING,
    ) -> AIBase:
        """
        Instantiate and return the appropriate AIBase subclass for embeddings.
        """
        env: EnvSettings = EnvSettings()

        # Determine embedding engine; allow completions fallback if needed
        if client_type == AIBase.CLIENT_TYPE_EMBEDDING:
            engine: str = env.get_setting("EMBEDDING_ENGINE", "titan").strip().lower()
        elif client_type == AIBase.CLIENT_TYPE_COMPLETIONS:
            engine: str = (
                env.get_setting("COMPLETIONS_ENGINE", "openai").strip().lower()
            )
        else:
            raise RuntimeError(f"Unsupported client_type: {client_type!r}")

        model_name: str = env.get_setting("EMBEDDING_MODEL_NAME", "").strip()
        dim: int = int(env.get_setting("EMBEDDING_DIMENSIONS", "1024"))

        if engine == "titan":
            return AiTitanEmbeddings(model=model_name, dimensions=dim)
        if engine == "openai":
            return AiOpenAIEmbeddings(model=model_name, dimensions=dim)

        raise RuntimeError(f"Unsupported EMBEDDING engine: {engine}")

    @staticmethod
    def list_completion_models(client: AIBaseCompletions) -> List[str]:
        """
        Return the list of completion-model names supported by the given client.
        """
        if not isinstance(client, AIBaseCompletions):
            raise TypeError(f"Expected AIBaseCompletions, got {type(client).__name__}")
        return client.list_model_names
