"""Tests for AIFactory provider loading orchestration."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from ai_api_unified.ai_factory import AIFactory
from ai_api_unified.ai_provider_exceptions import (
    AiProviderConfigurationError,
    AiProviderDependencyUnavailableError,
)
from ai_api_unified.ai_provider_registry import (
    AI_PROVIDER_CAPABILITY_COMPLETIONS,
    AI_PROVIDER_CAPABILITY_IMAGES,
)


class FakeCompletionsClient:
    """Minimal fake completions client constructor target for factory tests."""

    def __init__(self, model: str) -> None:
        self.model: str = model


class FakeImagesClient:
    """Minimal fake image client constructor target for factory tests."""

    def __init__(self, model: str | None) -> None:
        self.model: str | None = model


class FakeEmbeddingsClient:
    """Minimal fake embeddings client constructor target for factory tests."""

    def __init__(self, model: str, dimensions: int) -> None:
        self.model: str = model
        self.dimensions: int = dimensions


class TestAiFactoryProviderLoading:
    """Validate AIFactory selection + lazy-loader integration behavior."""

    @staticmethod
    def _build_env_settings_mock(
        *,
        completions_engine: str = "openai",
        embedding_engine: str = "openai",
        image_engine: str = "openai",
        completions_model_name: str = "gpt-4o-mini",
        embedding_model_name: str = "text-embedding-3-small",
        image_model_name: str = "gpt-image-1",
        embedding_dimensions: str = "1536",
    ) -> Mock:
        """Creates a deterministic EnvSettings mock for factory tests."""
        mock_env_settings: Mock = Mock()

        def get_setting_side_effect(str_key: str, str_default: str) -> str:
            """Resolves test configuration values by key with default fallback."""
            dict_settings: dict[str, str] = {
                "COMPLETIONS_ENGINE": completions_engine,
                "COMPLETIONS_MODEL_NAME": completions_model_name,
                "EMBEDDING_ENGINE": embedding_engine,
                "EMBEDDING_MODEL_NAME": embedding_model_name,
                "IMAGE_ENGINE": image_engine,
                "IMAGE_MODEL_NAME": image_model_name,
                "EMBEDDING_DIMENSIONS": embedding_dimensions,
            }
            return dict_settings.get(str_key, str_default)

        mock_env_settings.get_setting.side_effect = get_setting_side_effect
        return mock_env_settings

    def test_get_ai_completions_client_loads_selected_ai_provider(self) -> None:
        """Completions factory path should resolve registry spec and instantiate loaded class."""
        mock_env_settings: Mock = self._build_env_settings_mock()
        mock_ai_provider_spec: Mock = Mock()

        with patch(
            "ai_api_unified.ai_factory.EnvSettings",
            return_value=mock_env_settings,
        ):
            with patch(
                "ai_api_unified.ai_factory.get_ai_provider_spec",
                return_value=mock_ai_provider_spec,
            ) as mock_get_ai_provider_spec:
                with patch(
                    "ai_api_unified.ai_factory.load_ai_provider_class",
                    return_value=FakeCompletionsClient,
                ) as mock_load_ai_provider_class:
                    completions_client: FakeCompletionsClient = (
                        AIFactory.get_ai_completions_client()
                    )

        assert isinstance(completions_client, FakeCompletionsClient)
        assert completions_client.model == "gpt-4o-mini"
        mock_get_ai_provider_spec.assert_called_once_with(
            AI_PROVIDER_CAPABILITY_COMPLETIONS,
            "openai",
        )
        mock_load_ai_provider_class.assert_called_once()

    def test_get_ai_completions_client_translates_unknown_engine_to_value_error(
        self,
    ) -> None:
        """Unsupported completions engines should preserve legacy ValueError contract."""
        mock_env_settings: Mock = self._build_env_settings_mock()

        with patch(
            "ai_api_unified.ai_factory.EnvSettings",
            return_value=mock_env_settings,
        ):
            with patch(
                "ai_api_unified.ai_factory.get_ai_provider_spec",
                side_effect=AiProviderConfigurationError("unknown engine"),
            ):
                with pytest.raises(
                    ValueError,
                    match="Unsupported COMPLETIONS engine",
                ):
                    AIFactory.get_ai_completions_client(completions_engine="unknown")

    def test_get_ai_completions_client_raises_dependency_error_when_extra_missing(
        self,
    ) -> None:
        """Dependency unavailable errors should propagate unchanged from loader."""
        mock_env_settings: Mock = self._build_env_settings_mock()
        mock_ai_provider_spec: Mock = Mock()

        with patch(
            "ai_api_unified.ai_factory.EnvSettings",
            return_value=mock_env_settings,
        ):
            with patch(
                "ai_api_unified.ai_factory.get_ai_provider_spec",
                return_value=mock_ai_provider_spec,
            ):
                with patch(
                    "ai_api_unified.ai_factory.load_ai_provider_class",
                    side_effect=AiProviderDependencyUnavailableError("missing extra"),
                ):
                    with pytest.raises(
                        AiProviderDependencyUnavailableError,
                        match="missing extra",
                    ):
                        AIFactory.get_ai_completions_client()

    def test_get_ai_images_client_loads_selected_ai_provider(self) -> None:
        """Images factory path should resolve registry spec and instantiate loaded class."""
        mock_env_settings: Mock = self._build_env_settings_mock()
        mock_ai_provider_spec: Mock = Mock()

        with patch(
            "ai_api_unified.ai_factory.EnvSettings",
            return_value=mock_env_settings,
        ):
            with patch(
                "ai_api_unified.ai_factory.get_ai_provider_spec",
                return_value=mock_ai_provider_spec,
            ) as mock_get_ai_provider_spec:
                with patch(
                    "ai_api_unified.ai_factory.load_ai_provider_class",
                    return_value=FakeImagesClient,
                ) as mock_load_ai_provider_class:
                    images_client: FakeImagesClient = AIFactory.get_ai_images_client()

        assert isinstance(images_client, FakeImagesClient)
        assert images_client.model == "gpt-image-1"
        mock_get_ai_provider_spec.assert_called_once_with(
            AI_PROVIDER_CAPABILITY_IMAGES,
            "openai",
        )
        mock_load_ai_provider_class.assert_called_once()

    def test_get_ai_images_client_loads_google_gemini_provider_when_selected(self) -> None:
        """Gemini images should route through the centralized lazy-loading image factory."""
        mock_env_settings: Mock = self._build_env_settings_mock(
            image_engine="google-gemini"
        )
        mock_ai_provider_spec: Mock = Mock()

        with patch(
            "ai_api_unified.ai_factory.EnvSettings",
            return_value=mock_env_settings,
        ):
            with patch(
                "ai_api_unified.ai_factory.get_ai_provider_spec",
                return_value=mock_ai_provider_spec,
            ) as mock_get_ai_provider_spec:
                with patch(
                    "ai_api_unified.ai_factory.load_ai_provider_class",
                    return_value=FakeImagesClient,
                ):
                    images_client: FakeImagesClient = AIFactory.get_ai_images_client()

        assert isinstance(images_client, FakeImagesClient)
        assert images_client.model == "gpt-image-1"
        mock_get_ai_provider_spec.assert_called_once_with(
            AI_PROVIDER_CAPABILITY_IMAGES,
            "google-gemini",
        )

    def test_get_ai_embedding_client_preserves_provider_default_model_and_dimensions(
        self,
    ) -> None:
        """Embeddings factory should not inject cross-provider defaults when config is unset."""
        mock_env_settings: Mock = self._build_env_settings_mock(
            embedding_engine="google-gemini",
            embedding_model_name="",
            embedding_dimensions="",
        )
        mock_ai_provider_spec: Mock = Mock()

        with patch(
            "ai_api_unified.ai_factory.EnvSettings",
            return_value=mock_env_settings,
        ):
            with patch(
                "ai_api_unified.ai_factory.get_ai_provider_spec",
                return_value=mock_ai_provider_spec,
            ):
                with patch(
                    "ai_api_unified.ai_factory.load_ai_provider_class",
                    return_value=FakeEmbeddingsClient,
                ):
                    embeddings_client: FakeEmbeddingsClient = (
                        AIFactory.get_ai_embedding_client()
                    )

        assert isinstance(embeddings_client, FakeEmbeddingsClient)
        assert embeddings_client.model == ""
        assert embeddings_client.dimensions == 0

    def test_get_ai_images_client_preserves_provider_default_model_when_config_is_unset(
        self,
    ) -> None:
        """Images factory should pass None so provider implementations can apply their own default model."""
        mock_env_settings: Mock = self._build_env_settings_mock(
            image_engine="google-gemini",
            image_model_name="",
        )
        mock_ai_provider_spec: Mock = Mock()

        with patch(
            "ai_api_unified.ai_factory.EnvSettings",
            return_value=mock_env_settings,
        ):
            with patch(
                "ai_api_unified.ai_factory.get_ai_provider_spec",
                return_value=mock_ai_provider_spec,
            ):
                with patch(
                    "ai_api_unified.ai_factory.load_ai_provider_class",
                    return_value=FakeImagesClient,
                ):
                    images_client: FakeImagesClient = AIFactory.get_ai_images_client()

        assert isinstance(images_client, FakeImagesClient)
        assert images_client.model is None

    def test_get_ai_completions_client_raises_when_engine_is_missing(self) -> None:
        """Completions factory should require explicit engine configuration."""
        mock_env_settings: Mock = self._build_env_settings_mock(completions_engine="")

        with patch(
            "ai_api_unified.ai_factory.EnvSettings",
            return_value=mock_env_settings,
        ):
            with pytest.raises(
                ValueError,
                match="COMPLETIONS_ENGINE must be configured explicitly",
            ):
                AIFactory.get_ai_completions_client()

    def test_get_ai_embedding_client_raises_when_engine_is_missing(self) -> None:
        """Embeddings factory should require explicit engine configuration."""
        mock_env_settings: Mock = self._build_env_settings_mock(embedding_engine="")

        with patch(
            "ai_api_unified.ai_factory.EnvSettings",
            return_value=mock_env_settings,
        ):
            with pytest.raises(
                ValueError,
                match="EMBEDDING_ENGINE must be configured explicitly",
            ):
                AIFactory.get_ai_embedding_client()

    def test_get_ai_images_client_raises_when_engine_is_missing(self) -> None:
        """Images factory should require explicit engine configuration."""
        mock_env_settings: Mock = self._build_env_settings_mock(image_engine="")

        with patch(
            "ai_api_unified.ai_factory.EnvSettings",
            return_value=mock_env_settings,
        ):
            with pytest.raises(
                ValueError,
                match="IMAGE_ENGINE must be configured explicitly",
            ):
                AIFactory.get_ai_images_client()
