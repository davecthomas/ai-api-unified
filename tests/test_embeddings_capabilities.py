# test_embeddings_capabilities.py
"""
Tests for embeddings capabilities descriptors and multimodal embeddings support.

Covers:
    - AIEmbeddingsMultimodalParams validation (aligned lists, MIME/type checks)
    - AIBaseEmbeddings text-only defaults and capability-gated multimodal errors
    - Per-provider capabilities descriptors (Google, OpenAI, Titan)
    - GoogleGeminiEmbeddings multimodal call shape with a mocked client
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest

from ai_api_unified.ai_base import (
    AIBaseEmbeddings,
    AIEmbeddingsCapabilitiesBase,
    AIEmbeddingsMultimodalParams,
    SupportedDataType,
)
from ai_api_unified.ai_provider_exceptions import AiProviderCapabilityUnsupportedError

FAKE_PNG_BYTES: bytes = b"\x89PNG-fake-image-bytes"
FAKE_MP4_BYTES: bytes = b"fake-mp4-bytes"


class _TextOnlyEmbeddings(AIBaseEmbeddings):
    """Minimal concrete embeddings client relying on base-class defaults."""

    def __init__(self) -> None:
        super().__init__(model="text-only-model", dimensions=256)

    @property
    def list_model_names(self) -> list[str]:
        return ["text-only-model"]

    def generate_embeddings(self, text: str) -> dict[str, Any]:
        return {"embedding": [0.0], "text": text, "dimensions": 1}

    def generate_embeddings_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        return [self.generate_embeddings(text) for text in texts]


class TestMultimodalParamsValidation:
    """AIEmbeddingsMultimodalParams input validation."""

    def test_requires_text_or_media(self) -> None:
        with pytest.raises(ValueError, match="requires text, media attachments"):
            AIEmbeddingsMultimodalParams(text="   ")

    def test_rejects_misaligned_lists(self) -> None:
        with pytest.raises(ValueError, match="must be the same length"):
            AIEmbeddingsMultimodalParams(
                included_types=[SupportedDataType.IMAGE],
                included_data=[FAKE_PNG_BYTES, FAKE_PNG_BYTES],
                included_mime_types=["image/png"],
            )

    def test_rejects_mime_type_mismatch(self) -> None:
        with pytest.raises(ValueError, match="does not match declared type"):
            AIEmbeddingsMultimodalParams(
                included_types=[SupportedDataType.VIDEO],
                included_data=[FAKE_MP4_BYTES],
                included_mime_types=["image/png"],
            )

    def test_rejects_text_as_attachment(self) -> None:
        with pytest.raises(ValueError, match="Text belongs in the 'text' field"):
            AIEmbeddingsMultimodalParams(
                included_types=[SupportedDataType.TEXT],
                included_data=[b"hello"],
                included_mime_types=["text/plain"],
            )

    def test_rejects_empty_media_bytes(self) -> None:
        with pytest.raises(ValueError, match="bytes cannot be empty"):
            AIEmbeddingsMultimodalParams(
                included_types=[SupportedDataType.IMAGE],
                included_data=[b""],
                included_mime_types=["image/png"],
            )

    def test_accepts_interleaved_text_and_media(self) -> None:
        params = AIEmbeddingsMultimodalParams(
            text="a red bicycle",
            included_types=[SupportedDataType.IMAGE, SupportedDataType.PDF],
            included_data=[FAKE_PNG_BYTES, b"%PDF-fake"],
            included_mime_types=["image/png", "application/pdf"],
        )
        assert params.has_included_media is True
        list_media: list[tuple[int, SupportedDataType, bytes, str]] = list(
            params.iter_included_media()
        )
        assert len(list_media) == 2
        assert list_media[0][1] is SupportedDataType.IMAGE

    def test_accepts_text_only(self) -> None:
        params = AIEmbeddingsMultimodalParams(text="just text")
        assert params.has_included_media is False

    def test_rejects_oversized_combined_media(self) -> None:
        bytes_eleven_mb: bytes = b"x" * 11_000_000
        with pytest.raises(ValueError, match="Combined media attachments"):
            AIEmbeddingsMultimodalParams(
                included_types=[SupportedDataType.IMAGE, SupportedDataType.IMAGE],
                included_data=[bytes_eleven_mb, bytes_eleven_mb],
                included_mime_types=["image/png", "image/png"],
            )


class TestBaseEmbeddingsCapabilities:
    """Base-class capability defaults and multimodal gating."""

    def test_default_capabilities_are_text_only(self) -> None:
        client = _TextOnlyEmbeddings()
        capabilities: AIEmbeddingsCapabilitiesBase = client.capabilities
        assert capabilities.supported_data_types == [SupportedDataType.TEXT]
        assert capabilities.default_dimensions == 256

    def test_multimodal_raises_capability_error(self) -> None:
        client = _TextOnlyEmbeddings()
        params = AIEmbeddingsMultimodalParams(
            included_types=[SupportedDataType.IMAGE],
            included_data=[FAKE_PNG_BYTES],
            included_mime_types=["image/png"],
        )
        with pytest.raises(
            AiProviderCapabilityUnsupportedError, match="does not support multimodal"
        ):
            client.generate_embeddings_multimodal(params)


class TestGoogleEmbeddingsCapabilities:
    """Google capabilities descriptor and multimodal implementation."""

    @staticmethod
    def _build_client(model: str, dimensions: int, mock_client: Mock) -> Any:
        pytest.importorskip("google.genai")
        from ai_api_unified.embeddings.ai_google_gemini_embeddings import (
            GoogleGeminiEmbeddings,
        )

        if isinstance(mock_client.vertexai, Mock):
            # Bare Mock attributes are truthy; default to API-key mode unless
            # a test explicitly opts into Vertex mode.
            mock_client.vertexai = False
        with patch.object(
            GoogleGeminiEmbeddings,
            "_initialize_client",
            lambda self: setattr(self, "client", mock_client),
        ):
            # Normal return with a Gemini embeddings client whose SDK client is mocked.
            return GoogleGeminiEmbeddings(model=model, dimensions=dimensions)

    def test_capabilities_for_embedding_001_are_text_only(self) -> None:
        pytest.importorskip("google.genai")
        from ai_api_unified.embeddings.ai_google_gemini_embeddings import (
            AIEmbeddingsCapabilitiesGoogle,
        )

        capabilities = AIEmbeddingsCapabilitiesGoogle.for_model("gemini-embedding-001")
        assert capabilities.supported_data_types == [SupportedDataType.TEXT]
        assert capabilities.min_dimensions == 768
        assert capabilities.max_dimensions == 3072

    def test_capabilities_for_embedding_2_are_multimodal(self) -> None:
        pytest.importorskip("google.genai")
        from ai_api_unified.embeddings.ai_google_gemini_embeddings import (
            AIEmbeddingsCapabilitiesGoogle,
        )

        capabilities = AIEmbeddingsCapabilitiesGoogle.for_model("gemini-embedding-2")
        assert SupportedDataType.IMAGE in capabilities.supported_data_types
        assert SupportedDataType.VIDEO in capabilities.supported_data_types
        assert SupportedDataType.AUDIO in capabilities.supported_data_types
        assert SupportedDataType.PDF in capabilities.supported_data_types
        assert capabilities.min_dimensions == 128
        assert capabilities.max_images_per_request == 6
        assert capabilities.max_video_seconds == 120

    def test_text_only_model_rejects_media_input(self) -> None:
        client = self._build_client("gemini-embedding-001", 768, Mock())
        params = AIEmbeddingsMultimodalParams(
            included_types=[SupportedDataType.IMAGE],
            included_data=[FAKE_PNG_BYTES],
            included_mime_types=["image/png"],
        )
        with pytest.raises(
            AiProviderCapabilityUnsupportedError,
            match="does not support multimodal embeddings",
        ):
            client.generate_embeddings_multimodal(params)

    def test_text_only_model_rejects_text_only_multimodal_call(self) -> None:
        client = self._build_client("gemini-embedding-001", 768, Mock())
        params = AIEmbeddingsMultimodalParams(text="plain text")
        with pytest.raises(
            AiProviderCapabilityUnsupportedError,
            match="does not support multimodal embeddings",
        ):
            client.generate_embeddings_multimodal(params)

    def test_vertex_mode_rejects_media_attachments(self) -> None:
        mock_client = Mock()
        mock_client.vertexai = True
        client = self._build_client("gemini-embedding-2", 3072, mock_client)
        params = AIEmbeddingsMultimodalParams(
            text="a red bicycle",
            included_types=[SupportedDataType.IMAGE],
            included_data=[FAKE_PNG_BYTES],
            included_mime_types=["image/png"],
        )
        with pytest.raises(NotImplementedError, match="GOOGLE_AUTH_METHOD=api_key"):
            client.generate_embeddings_multimodal(params)

    def test_multimodal_model_rejects_too_many_images(self) -> None:
        client = self._build_client("gemini-embedding-2", 3072, Mock())
        int_image_count: int = 7
        params = AIEmbeddingsMultimodalParams(
            included_types=[SupportedDataType.IMAGE] * int_image_count,
            included_data=[FAKE_PNG_BYTES] * int_image_count,
            included_mime_types=["image/png"] * int_image_count,
        )
        with pytest.raises(
            AiProviderCapabilityUnsupportedError, match="at most 6 images"
        ):
            client.generate_embeddings_multimodal(params)

    def test_multimodal_embedding_call_shape(self) -> None:
        pytest.importorskip("google.genai")
        from google.genai.types import Content

        mock_client = Mock()
        mock_client.models.embed_content.return_value = Mock(
            embeddings=[Mock(values=[0.1] * 1536)],
            usage_metadata=None,
        )
        client = self._build_client("gemini-embedding-2", 1536, mock_client)

        params = AIEmbeddingsMultimodalParams(
            text="a red bicycle",
            included_types=[SupportedDataType.IMAGE],
            included_data=[FAKE_PNG_BYTES],
            included_mime_types=["image/png"],
        )
        with patch.object(
            client,
            "_retry_with_exponential_backoff",
            side_effect=lambda func: func(),
        ):
            result: dict[str, Any] = client.generate_embeddings_multimodal(params)

        assert result["model"] == "gemini-embedding-2"
        assert result["dimensions"] == 1536
        assert result["text"] == "a red bicycle"
        assert result["included_media_count"] == 1

        embed_kwargs: dict[str, Any] = mock_client.models.embed_content.call_args.kwargs
        assert embed_kwargs["model"] == "gemini-embedding-2"
        content: Content = embed_kwargs["contents"]
        assert isinstance(content, Content)
        assert len(content.parts) == 2
        assert content.parts[0].text == "a red bicycle"
        assert content.parts[1].inline_data.mime_type == "image/png"
        assert content.parts[1].inline_data.data == FAKE_PNG_BYTES
        # Non-default dimensions must be forwarded as output_dimensionality.
        assert embed_kwargs["config"].output_dimensionality == 1536

    def test_multimodal_default_dimensions_omit_config(self) -> None:
        mock_client = Mock()
        mock_client.models.embed_content.return_value = Mock(
            embeddings=[Mock(values=[0.1] * 3072)],
            usage_metadata=None,
        )
        client = self._build_client("gemini-embedding-2", 3072, mock_client)

        params = AIEmbeddingsMultimodalParams(text="only text this time")
        with patch.object(
            client,
            "_retry_with_exponential_backoff",
            side_effect=lambda func: func(),
        ):
            client.generate_embeddings_multimodal(params)

        embed_kwargs: dict[str, Any] = mock_client.models.embed_content.call_args.kwargs
        assert "config" not in embed_kwargs


class TestOpenAIEmbeddingsFactoryDefaults:
    """Factory-style falsy model/dimensions inputs resolve to model defaults."""

    @staticmethod
    def _build_client(model: str, dimensions: int) -> Any:
        pytest.importorskip("openai")
        import os
        from ai_api_unified.embeddings.ai_openai_embeddings import AiOpenAIEmbeddings

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            # Normal return with an OpenAI embeddings client built factory-style.
            return AiOpenAIEmbeddings(model=model, dimensions=dimensions)

    def test_zero_dimensions_resolves_to_model_default(self) -> None:
        client = self._build_client("text-embedding-3-small", 0)
        assert client.dimensions == 1536

    def test_empty_model_resolves_to_default_model(self) -> None:
        client = self._build_client("", 0)
        assert client.embedding_model == "text-embedding-3-small"
        assert client.dimensions == 1536

    def test_explicit_values_win(self) -> None:
        client = self._build_client("text-embedding-3-large", 256)
        assert client.embedding_model == "text-embedding-3-large"
        assert client.dimensions == 256


class TestOpenAIEmbeddingsCapabilities:
    """OpenAI capabilities descriptor."""

    def test_for_model_three_large(self) -> None:
        pytest.importorskip("openai")
        from ai_api_unified.embeddings.ai_openai_embeddings import (
            AIEmbeddingsCapabilitiesOpenAI,
        )

        capabilities = AIEmbeddingsCapabilitiesOpenAI.for_model(
            "text-embedding-3-large"
        )
        assert capabilities.supported_data_types == [SupportedDataType.TEXT]
        assert capabilities.default_dimensions == 3072
        assert capabilities.max_batch_size == 2048

    def test_for_model_ada_has_fixed_dimensions(self) -> None:
        pytest.importorskip("openai")
        from ai_api_unified.embeddings.ai_openai_embeddings import (
            AIEmbeddingsCapabilitiesOpenAI,
        )

        capabilities = AIEmbeddingsCapabilitiesOpenAI.for_model(
            "text-embedding-ada-002"
        )
        assert capabilities.default_dimensions == 1536
        assert capabilities.min_dimensions is None
        assert capabilities.max_dimensions is None


class TestTitanEmbeddingsCapabilities:
    """Titan capabilities descriptor."""

    def test_for_model_v2_supports_reduced_dimensions(self) -> None:
        pytest.importorskip("boto3")
        from ai_api_unified.embeddings.ai_titan_embeddings import (
            AIEmbeddingsCapabilitiesTitan,
        )

        capabilities = AIEmbeddingsCapabilitiesTitan.for_model(
            "amazon.titan-embed-text-v2:0"
        )
        assert capabilities.supported_data_types == [SupportedDataType.TEXT]
        assert capabilities.recommended_dimensions == [256, 512, 1024]

    def test_for_model_v1_has_fixed_dimensions(self) -> None:
        pytest.importorskip("boto3")
        from ai_api_unified.embeddings.ai_titan_embeddings import (
            AIEmbeddingsCapabilitiesTitan,
        )

        capabilities = AIEmbeddingsCapabilitiesTitan.for_model(
            "amazon.titan-embed-text-v1"
        )
        assert capabilities.default_dimensions == 1536
