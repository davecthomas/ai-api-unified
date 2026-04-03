"""
Factory utilities for selecting provider implementations by runtime configuration.

This module resolves providers through the centralized AiProvider registry and
lazy loader to avoid import-time failures when optional extras are not installed.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import logging

from .ai_base import AIBaseCompletions, AIBaseEmbeddings, AIBaseImages
from .ai_provider_exceptions import (
    AiProviderConfigurationError,
    AiProviderDependencyUnavailableError,
    AiProviderRuntimeError,
)
from .ai_provider_loader import load_ai_provider_class
from .ai_provider_registry import (
    AiProviderSpec,
    AI_PROVIDER_CAPABILITY_COMPLETIONS,
    AI_PROVIDER_CAPABILITY_EMBEDDINGS,
    AI_PROVIDER_CAPABILITY_IMAGES,
    TypeAiProviderCapability,
    get_ai_provider_spec,
)
from .util.env_settings import EnvSettings

_LOGGER: logging.Logger = logging.getLogger(__name__)

DEFAULT_COMPLETIONS_ENGINE: str = ""
DEFAULT_EMBEDDING_ENGINE: str = ""
DEFAULT_IMAGE_ENGINE: str = ""
DEFAULT_EMBEDDING_DIMENSIONS: str = "0"
COMPLETIONS_MODEL_NAME_KEY: str = "COMPLETIONS_MODEL_NAME"
COMPLETIONS_ENGINE_KEY: str = "COMPLETIONS_ENGINE"
EMBEDDING_MODEL_NAME_KEY: str = "EMBEDDING_MODEL_NAME"
EMBEDDING_ENGINE_KEY: str = "EMBEDDING_ENGINE"
EMBEDDING_DIMENSIONS_KEY: str = "EMBEDDING_DIMENSIONS"
IMAGE_MODEL_NAME_KEY: str = "IMAGE_MODEL_NAME"
IMAGE_ENGINE_KEY: str = "IMAGE_ENGINE"


def _is_python_module_available(str_module_name: str) -> bool:
    """
    Determines whether a Python module can be discovered without importing it.

    Args:
        str_module_name: Fully qualified module path to probe in import metadata.

    Returns:
        True when the module can be resolved by import machinery, otherwise False.
    """
    list_name_parts: list[str] = str_module_name.split(".")
    if not list_name_parts:
        # Early return because empty module names cannot be resolved.
        return False

    str_current_full_name: str = list_name_parts[0]
    try:
        module_spec: importlib.machinery.ModuleSpec | None = importlib.util.find_spec(
            str_current_full_name
        )
    except (ModuleNotFoundError, ValueError):
        # Early return because malformed or missing modules are unavailable.
        return False
    if module_spec is None:
        # Early return because top-level module is unavailable.
        return False

    # Loop over dotted module segments without importing parent packages.
    for str_name_part in list_name_parts[1:]:
        if module_spec.submodule_search_locations is None:
            # Early return because the current segment is not a package.
            return False
        str_current_full_name = f"{str_current_full_name}.{str_name_part}"
        module_spec = importlib.machinery.PathFinder.find_spec(
            str_current_full_name,
            module_spec.submodule_search_locations,
        )
        if module_spec is None:
            # Early return because dotted submodule segment was not found.
            return False

    # Normal return because module path was resolved without importing it.
    return True


def _is_ai_provider_available(
    str_capability: TypeAiProviderCapability,
    str_engine: str,
) -> bool:
    """
    Probes whether required dependency roots for a provider exist in this runtime.

    Args:
        str_capability: Capability bucket used for provider lookup.
        str_engine: Engine selector token under the capability bucket.

    Returns:
        True when all dependency roots for the provider can be discovered, otherwise False.
    """
    try:
        ai_provider_spec: AiProviderSpec = get_ai_provider_spec(
            str_capability, str_engine
        )
    except AiProviderConfigurationError:
        # Early return because provider is unavailable in this runtime.
        return False

    # Loop over dependency roots and fail-fast on the first missing dependency.
    for str_dependency_root in ai_provider_spec.set_str_dependency_roots:
        if not _is_python_module_available(str_dependency_root):
            # Early return because at least one required dependency root is unavailable.
            return False

    # Normal return because every dependency root for this provider is available.
    return True


# Legacy availability flags are eagerly defined as plain module attributes to keep
# introspection simple and avoid module-level __getattr__ indirection.
GOOGLE_GEMINI_AVAILABLE: bool = _is_ai_provider_available(
    AI_PROVIDER_CAPABILITY_COMPLETIONS,
    "google-gemini",
)
BEDROCK_AVAILABLE: bool = _is_ai_provider_available(
    AI_PROVIDER_CAPABILITY_COMPLETIONS,
    "nova",
)
TITAN_AVAILABLE: bool = _is_ai_provider_available(
    AI_PROVIDER_CAPABILITY_EMBEDDINGS,
    "titan",
)
BEDROCK_IMAGES_AVAILABLE: bool = _is_ai_provider_available(
    AI_PROVIDER_CAPABILITY_IMAGES,
    "nova-canvas",
)


class AIFactory:
    """
    Factory for creating provider clients for completions, embeddings, and images.
    """

    @staticmethod
    def _normalize_engine(
        str_engine_override: str | None, str_default_engine: str
    ) -> str:
        """
        Resolves and normalizes engine values from explicit overrides.

        Args:
            str_engine_override: Optional engine override from method caller.
            str_default_engine: Default engine fallback when override is empty.

        Returns:
            Lowercase normalized engine token.
        """
        str_engine_normalized: str = str_default_engine
        if str_engine_override is not None and str_engine_override.strip():
            str_engine_normalized = str_engine_override.strip().lower()
        # Normal return with normalized engine selector.
        return str_engine_normalized

    @staticmethod
    def _resolve_required_engine(
        *,
        env_settings: EnvSettings,
        str_engine_key: str,
        str_engine_override: str | None = None,
    ) -> str:
        """
        Resolves a required engine selector from override or environment config.

        Args:
            env_settings: Shared environment/settings accessor.
            str_engine_key: Environment variable name for the target capability.
            str_engine_override: Optional explicit caller override.

        Returns:
            Lowercase normalized engine token.

        Raises:
            ValueError: When the engine selector is not configured.
        """
        if str_engine_override is not None and str_engine_override.strip():
            return str_engine_override.strip().lower()

        object_engine_value: object = env_settings.get_setting(str_engine_key, "")
        str_engine_value: str = (
            str(object_engine_value) if object_engine_value is not None else ""
        )
        str_engine_normalized: str = str_engine_value.strip().lower()
        if not str_engine_normalized:
            raise ValueError(
                f"{str_engine_key} must be configured explicitly; there is no default provider."
            )
        return str_engine_normalized

    @staticmethod
    def _translate_config_exception(
        exception: AiProviderConfigurationError,
        str_capability: str,
        str_engine: str,
    ) -> ValueError:
        """
        Converts registry configuration misses into legacy ValueError behavior.

        Args:
            exception: Original provider configuration exception.
            str_capability: Capability bucket that failed lookup.
            str_engine: Engine token that failed lookup.

        Returns:
            ValueError matching historical unsupported-engine behavior.
        """
        if str_capability == AI_PROVIDER_CAPABILITY_COMPLETIONS:
            # Normal return with legacy completions unsupported-engine error.
            return ValueError(f"Unsupported COMPLETIONS engine: {str_engine!r}")
        if str_capability == AI_PROVIDER_CAPABILITY_EMBEDDINGS:
            # Normal return with legacy embeddings unsupported-engine error.
            return ValueError(f"Unsupported EMBEDDING engine: {str_engine}")
        if str_capability == AI_PROVIDER_CAPABILITY_IMAGES:
            # Normal return with legacy images unsupported-engine error.
            return ValueError(f"Unsupported IMAGE engine: {str_engine!r}")
        # Normal return with generic unsupported engine error.
        return ValueError(str(exception))

    @staticmethod
    def get_ai_completions_client(
        model_name: str | None = None,
        completions_engine: str | None = None,
    ) -> AIBaseCompletions:
        """
        Instantiates the configured completions client.

        Args:
            model_name: Optional model override; falls back to environment config.
            completions_engine: Optional engine override; falls back to environment config.

        Returns:
            Concrete AIBaseCompletions implementation for the requested engine.
            Raises ValueError for unsupported engines and RuntimeError-derived
            provider exceptions for dependency/runtime loading failures.
        """
        env_settings: EnvSettings = EnvSettings()
        str_engine: str = AIFactory._resolve_required_engine(
            env_settings=env_settings,
            str_engine_key=COMPLETIONS_ENGINE_KEY,
            str_engine_override=completions_engine,
        )

        str_model_name: str
        if model_name is None:
            str_model_name = env_settings.get_setting(COMPLETIONS_MODEL_NAME_KEY, "")
        else:
            str_model_name = model_name

        try:
            # Resolve provider metadata first so engine-to-module mapping is centralized
            # in the registry and does not require hardcoded imports in the factory.
            ai_provider_spec: AiProviderSpec = get_ai_provider_spec(
                AI_PROVIDER_CAPABILITY_COMPLETIONS, str_engine
            )
            # Lazy-load and validate the provider class at runtime. This is the core
            # refactor behavior that keeps optional SDK imports out of package import-time
            # while still enforcing the AIBaseCompletions interface contract.
            class_completions_client: type[AIBaseCompletions] = load_ai_provider_class(
                ai_provider_spec,
                AIBaseCompletions,
            )
            # Instantiate the resolved class only after lazy-load validation succeeds.
            completions_client: AIBaseCompletions = class_completions_client(
                model=str_model_name
            )
            # Normal return with configured completions provider client.
            return completions_client
        except AiProviderConfigurationError as exception:
            raise AIFactory._translate_config_exception(
                exception,
                AI_PROVIDER_CAPABILITY_COMPLETIONS,
                str_engine,
            ) from exception
        except AiProviderDependencyUnavailableError as exception:
            _LOGGER.warning(str(exception))
            raise
        except AiProviderRuntimeError:
            raise

    @staticmethod
    def get_ai_embedding_client(
        embedding_engine: str | None = None,
        model_name: str | None = None,
    ) -> AIBaseEmbeddings:
        """
        Instantiates the configured embeddings client.

        Args:
            embedding_engine: Optional engine override; falls back to environment config.
            model_name: Optional model override; falls back to environment config.

        Returns:
            Concrete AIBaseEmbeddings implementation for the requested engine.
            Raises ValueError for unsupported engines and RuntimeError-derived
            provider exceptions for dependency/runtime loading failures.
        """
        env_settings: EnvSettings = EnvSettings()
        str_engine: str = AIFactory._resolve_required_engine(
            env_settings=env_settings,
            str_engine_key=EMBEDDING_ENGINE_KEY,
            str_engine_override=embedding_engine,
        )

        str_model_name: str
        if model_name is None:
            str_model_name = env_settings.get_setting(EMBEDDING_MODEL_NAME_KEY, "")
        else:
            str_model_name = model_name

        raw_embedding_dimensions: str | int | None = env_settings.get_setting(
            EMBEDDING_DIMENSIONS_KEY, DEFAULT_EMBEDDING_DIMENSIONS
        )
        int_dimensions: int = (
            0
            if raw_embedding_dimensions in (None, "")
            else int(raw_embedding_dimensions)
        )

        try:
            # Resolve provider metadata first so engine-to-module mapping is centralized
            # in the registry and does not require hardcoded imports in the factory.
            ai_provider_spec: AiProviderSpec = get_ai_provider_spec(
                AI_PROVIDER_CAPABILITY_EMBEDDINGS, str_engine
            )
            # Lazy-load and validate the provider class at runtime. This is the core
            # refactor behavior that keeps optional SDK imports out of package import-time
            # while still enforcing the AIBaseEmbeddings interface contract.
            class_embeddings_client: type[AIBaseEmbeddings] = load_ai_provider_class(
                ai_provider_spec,
                AIBaseEmbeddings,
            )
            # Instantiate the resolved class only after lazy-load validation succeeds.
            embeddings_client: AIBaseEmbeddings = class_embeddings_client(
                model=str_model_name,
                dimensions=int_dimensions,
            )
            # Normal return with configured embeddings provider client.
            return embeddings_client
        except AiProviderConfigurationError as exception:
            raise AIFactory._translate_config_exception(
                exception,
                AI_PROVIDER_CAPABILITY_EMBEDDINGS,
                str_engine,
            ) from exception
        except AiProviderDependencyUnavailableError as exception:
            _LOGGER.warning(str(exception))
            raise
        except AiProviderRuntimeError:
            raise

    @staticmethod
    def list_completion_models(client: AIBaseCompletions) -> list[str]:
        """
        Returns all completion model names supported by the given client.

        Args:
            client: Concrete AIBaseCompletions implementation instance.

        Returns:
            list of supported completion model identifiers.
            Raises TypeError when the object is not an AIBaseCompletions instance.
        """
        if not isinstance(client, AIBaseCompletions):
            raise TypeError(f"Expected AIBaseCompletions, got {type(client).__name__}")
        # Normal return with provider model name collection.
        return client.list_model_names

    @staticmethod
    def get_ai_images_client(
        image_model: str | None = None,
    ) -> AIBaseImages:
        """
        Instantiates the configured image-generation client.

        Args:
            image_model: Optional model override; falls back to environment config.

        Returns:
            Concrete AIBaseImages implementation for the requested image engine.
            Raises ValueError for unsupported engines and RuntimeError-derived
            provider exceptions for dependency/runtime loading failures.
        """
        env_settings: EnvSettings = EnvSettings()
        str_image_engine: str = AIFactory._resolve_required_engine(
            env_settings=env_settings,
            str_engine_key=IMAGE_ENGINE_KEY,
        )

        str_image_model_name: str | None
        if image_model is None:
            raw_image_model_name: str | None = env_settings.get_setting(
                IMAGE_MODEL_NAME_KEY, None
            )
            if raw_image_model_name is None or raw_image_model_name.strip() == "":
                str_image_model_name = None
            else:
                str_image_model_name = raw_image_model_name
        else:
            str_image_model_name = image_model

        try:
            # Resolve provider metadata first so engine-to-module mapping is centralized
            # in the registry and does not require hardcoded imports in the factory.
            ai_provider_spec: AiProviderSpec = get_ai_provider_spec(
                AI_PROVIDER_CAPABILITY_IMAGES, str_image_engine
            )
            # Lazy-load and validate the provider class at runtime. This is the core
            # refactor behavior that keeps optional SDK imports out of package import-time
            # while still enforcing the AIBaseImages interface contract.
            class_ai_images_client: type[AIBaseImages] = load_ai_provider_class(
                ai_provider_spec,
                AIBaseImages,
            )
            # Instantiate the resolved class only after lazy-load validation succeeds.
            images_client: AIBaseImages = class_ai_images_client(
                model=str_image_model_name
            )
            # Normal return with configured images provider client.
            return images_client
        except AiProviderConfigurationError as exception:
            raise AIFactory._translate_config_exception(
                exception,
                AI_PROVIDER_CAPABILITY_IMAGES,
                str_image_engine,
            ) from exception
        except AiProviderDependencyUnavailableError as exception:
            _LOGGER.warning(str(exception))
            raise
        except AiProviderRuntimeError:
            raise
