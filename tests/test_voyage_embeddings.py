# test_voyage_embeddings.py
"""
Mocked tests for the Voyage AI embeddings engine.

The voyageai SDK is optional (the `voyage` extra) and not installed in the
default dev environment; the engine's lazy client makes that testable —
construction needs only VOYAGE_API_KEY, and tests inject Mock clients or
patch sys.modules to simulate the installed/missing SDK.
"""

import os
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ai_api_unified.ai_base import AIBaseEmbeddings
from ai_api_unified.ai_provider_exceptions import (
    AiProviderCapabilityUnsupportedError,
    AiProviderDependencyUnavailableError,
    AiProviderRequestError,
)
from ai_api_unified.ai_provider_registry import (
    AI_PROVIDER_CAPABILITY_EMBEDDINGS,
    get_ai_provider_spec,
)
from ai_api_unified.embeddings.ai_voyage_embeddings import (
    AiVoyageEmbeddings,
    AIEmbeddingsCapabilitiesVoyage,
)


def _build_client(model: str = "voyage-3", **kwargs) -> AiVoyageEmbeddings:
    with patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"}):
        return AiVoyageEmbeddings(model=model, **kwargs)


def _embed_result(vectors: list[list[float]], total_tokens: int = 7):
    return SimpleNamespace(embeddings=vectors, total_tokens=total_tokens)


class _FakeVoyageError(Exception):
    """Duck-typed stand-in for a voyageai SDK error."""

    __module__ = "voyageai.error"

    def __init__(self, message: str, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status


class TestRegistrationAndConstruction:
    def test_provider_spec_registered_with_voyage_extra(self):
        spec = get_ai_provider_spec(AI_PROVIDER_CAPABILITY_EMBEDDINGS, "voyage")
        assert spec.str_class_name == "AiVoyageEmbeddings"
        assert spec.str_required_extra == "voyage"
        assert spec.set_str_dependency_roots == {"voyageai"}

    def test_missing_api_key_rejected(self):
        with patch.dict(os.environ, {"VOYAGE_API_KEY": ""}):
            with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
                AiVoyageEmbeddings(model="voyage-3")

    def test_default_model_and_dimensions(self):
        # Clear EMBEDDING_MODEL_NAME so the engine's voyage-3 default applies
        # (a .env in the working tree would otherwise supply another model).
        with patch.dict(
            os.environ,
            {"VOYAGE_API_KEY": "test-key", "EMBEDDING_MODEL_NAME": ""},
        ):
            client = AiVoyageEmbeddings(model="")
        assert client.model_name == "voyage-3"
        assert client.dimensions == 1024
        lite = _build_client(model="voyage-3-lite")
        assert lite.dimensions == 512

    def test_capabilities_per_model(self):
        client = _build_client(model="voyage-law-2")
        assert client.capabilities.max_input_tokens == 16_000
        assert client.capabilities.max_batch_size == 128
        assert client.capabilities.supports_async is True
        assert client.capabilities.pricing is not None

    def test_observability_vendor_and_engine_resolution(self):
        client = _build_client()
        assert client._resolve_observability_provider_vendor() == "voyage"
        assert client._resolve_observability_provider_engine() == "voyage"

    def test_cost_uses_registry_pricing(self):
        client = _build_client(model="voyage-3")
        cost = client.compute_embedding_cost(input_tokens=1_000_000)
        assert Decimal(str(cost)) == Decimal("0.06")


class TestMissingExtra:
    def test_first_use_raises_typed_error_naming_extra(self):
        client = _build_client()
        with patch.dict("sys.modules", {"voyageai": None}):
            with pytest.raises(
                AiProviderDependencyUnavailableError, match="voyage"
            ) as exc_info:
                client.generate_embeddings("hello")
        assert "pip install" in str(exc_info.value)


class TestEmbed:
    def test_single_embedding_shape_matches_library_convention(self):
        client = _build_client()
        client._client = Mock()
        client._client.embed.return_value = _embed_result([[0.1, 0.2, 0.3]])
        result = client.generate_embeddings("hello world")
        # Same keys as the openai/titan/gemini engines.
        assert set(result) == {"embedding", "text", "dimensions"}
        assert result["embedding"] == [0.1, 0.2, 0.3]
        assert result["text"] == "hello world"
        assert result["dimensions"] == 3
        kwargs = client._client.embed.call_args.kwargs
        assert kwargs["model"] == "voyage-3"
        assert "input_type" not in kwargs

    def test_input_type_hint_forwarded(self):
        client = _build_client()
        client._client = Mock()
        client._client.embed.return_value = _embed_result([[0.5]])
        client.generate_embeddings("find docs about pricing", input_type="query")
        assert client._client.embed.call_args.kwargs["input_type"] == "query"

    def test_output_dimension_only_on_supporting_models(self):
        default_client = _build_client(model="voyage-3", dimensions=512)
        default_client._client = Mock()
        default_client._client.embed.return_value = _embed_result([[0.5]])
        default_client.generate_embeddings("x")
        assert "output_dimension" not in default_client._client.embed.call_args.kwargs

        large_client = _build_client(model="voyage-3-large", dimensions=512)
        large_client._client = Mock()
        large_client._client.embed.return_value = _embed_result([[0.5] * 512])
        large_client.generate_embeddings("x")
        assert large_client._client.embed.call_args.kwargs["output_dimension"] == 512

    def test_empty_text_rejected(self):
        client = _build_client()
        with pytest.raises(ValueError, match="non-empty"):
            client.generate_embeddings("   ")


class TestBatch:
    def test_batch_chunks_at_provider_cap_and_sums_tokens(self):
        client = _build_client()
        client._client = Mock()
        texts = [f"text-{i}" for i in range(130)]
        client._client.embed.side_effect = [
            _embed_result([[float(i)] for i in range(128)], total_tokens=256),
            _embed_result([[float(i)] for i in range(2)], total_tokens=4),
        ]
        results = client.generate_embeddings_batch(texts, input_type="document")
        assert len(results) == 130
        assert results[0]["text"] == "text-0"
        assert results[129]["text"] == "text-129"
        assert client._client.embed.call_count == 2
        first_kwargs = client._client.embed.call_args_list[0].kwargs
        second_kwargs = client._client.embed.call_args_list[1].kwargs
        assert len(first_kwargs["texts"]) == 128
        assert len(second_kwargs["texts"]) == 2
        assert first_kwargs["input_type"] == "document"

    def test_count_mismatch_raises_typed_error(self):
        client = _build_client()
        client._client = Mock()
        client._client.embed.return_value = _embed_result([[0.1]])
        with pytest.raises(AiProviderRequestError, match="embeddings for"):
            client.generate_embeddings_batch(["a", "b"])


class TestAsync:
    @pytest.mark.asyncio
    async def test_agenerate_embeddings(self):
        client = _build_client()
        async_client = Mock()
        async_client.embed = AsyncMock(return_value=_embed_result([[0.9, 0.8]]))
        client._async_client = async_client
        result = await client.agenerate_embeddings("hello", input_type="query")
        assert result["embedding"] == [0.9, 0.8]
        assert async_client.embed.call_args.kwargs["input_type"] == "query"

    @pytest.mark.asyncio
    async def test_agenerate_embeddings_batch_chunks(self):
        client = _build_client()
        async_client = Mock()
        async_client.embed = AsyncMock(
            side_effect=[
                _embed_result([[0.1]] * 128, total_tokens=128),
                _embed_result([[0.2]] * 2, total_tokens=2),
            ]
        )
        client._async_client = async_client
        results = await client.agenerate_embeddings_batch([f"t{i}" for i in range(130)])
        assert len(results) == 130
        assert async_client.embed.call_count == 2

    @pytest.mark.asyncio
    async def test_async_gate_on_engines_without_support(self):
        class _SyncOnlyEmbeddings(AIBaseEmbeddings):
            @property
            def list_model_names(self) -> list[str]:
                return ["sync-only"]

            def generate_embeddings(self, text, *, input_type=None):
                return {}

            def generate_embeddings_batch(self, texts, *, input_type=None):
                return []

        sync_only = _SyncOnlyEmbeddings(model="sync-only", dimensions=3)
        with pytest.raises(AiProviderCapabilityUnsupportedError):
            await sync_only.agenerate_embeddings("hello")


class TestErrorsAndRetries:
    def test_sdk_error_wrapped_with_status_code(self):
        client = _build_client()
        client._client = Mock()
        client._client.embed.side_effect = _FakeVoyageError("rate limited", 429)
        with pytest.raises(AiProviderRequestError) as exc_info:
            client.generate_embeddings("hello")
        assert exc_info.value.status_code == 429
        assert exc_info.value.provider_engine == "voyage"

    def test_non_sdk_error_propagates_unchanged(self):
        client = _build_client()
        client._client = Mock()
        client._client.embed.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            client.generate_embeddings("hello")

    def test_retry_policy_none_disables_sdk_retries(self):
        mock_voyageai = Mock()
        client = _build_client(retry_policy="none")
        with patch.dict("sys.modules", {"voyageai": mock_voyageai}):
            _ = client.client
        assert mock_voyageai.Client.call_args.kwargs["max_retries"] == 0

    def test_default_retry_policy_keeps_sdk_retries(self):
        mock_voyageai = Mock()
        client = _build_client()
        with patch.dict("sys.modules", {"voyageai": mock_voyageai}):
            _ = client.client
        assert "max_retries" not in mock_voyageai.Client.call_args.kwargs

    def test_env_retry_policy_none(self):
        mock_voyageai = Mock()
        with patch.dict(
            os.environ,
            {"VOYAGE_API_KEY": "test-key", "COMPLETIONS_RETRY_POLICY": "none"},
        ):
            client = AiVoyageEmbeddings(model="voyage-3")
        with patch.dict("sys.modules", {"voyageai": mock_voyageai}):
            _ = client.client
        assert mock_voyageai.Client.call_args.kwargs["max_retries"] == 0

    def test_invalid_retry_policy_rejected(self):
        with pytest.raises(ValueError, match="Unsupported retry policy"):
            _build_client(retry_policy="sometimes")


class TestCapabilitiesCatalog:
    def test_all_catalogued_models_have_dimensions_and_pricing(self):
        for model_name in AIEmbeddingsCapabilitiesVoyage.DICT_MODEL_DIMENSIONS:
            caps = AIEmbeddingsCapabilitiesVoyage.for_model(model_name)
            assert caps.default_dimensions > 0, model_name
            assert caps.max_input_tokens, model_name
            assert caps.pricing is not None, model_name
            assert caps.pricing.token_rates is not None, model_name
