# ai_voyage_embeddings.py

"""
Voyage AI embeddings via the official `voyageai` SDK.

Voyage is an embeddings/reranking specialist; Anthropic recommends it for
embeddings since Anthropic serves none. Registered as the `voyage` embeddings
engine (extra: `voyage`), authenticating with VOYAGE_API_KEY.

The engine implements the shared AIBaseEmbeddings surface — identical
signatures and return shapes ({"embedding", "text", "dimensions"}) to the
openai/titan/google-gemini engines — plus the async variants and the
provider-neutral input_type retrieval hint ("query" | "document"), which this
engine forwards to the API. The voyageai import is deferred to first client
use so the extra stays opt-in; a missing dependency raises the typed error
naming the extra to install.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from ..ai_base import (
    AIBaseEmbeddings,
    AiApiObservedEmbeddingsResultModel,
    AIEmbeddingsCapabilitiesBase,
    RETRY_POLICY_DEFAULT,
    RETRY_POLICY_NONE,
    SupportedDataType,
    normalize_retry_policy,
)
from ..ai_provider_exceptions import (
    AiProviderDependencyUnavailableError,
    AiProviderRequestError,
)
from ..middleware.observability_runtime import ObservabilityMetadataValue
from ..pricing.pricing_registry import (
    PROVIDER_VOYAGE,
    enforce_model_lifecycle,
    get_model_pricing,
)
from ..util.env_settings import EnvSettings

_LOGGER: logging.Logger = logging.getLogger(__name__)

VOYAGE_EXTRA_INSTALL_HINT: str = (
    "The voyage embeddings engine requires the 'voyage' extra: "
    "pip install 'ai-api-unified[voyage]' (the voyageai SDK)."
)


class AIEmbeddingsCapabilitiesVoyage(AIEmbeddingsCapabilitiesBase):
    """Voyage-specific embeddings capabilities.

    Based on https://docs.voyageai.com/docs/embeddings (model lineup,
    dimensions, and context lengths).
    """

    DICT_MODEL_DIMENSIONS: ClassVar[dict[str, int]] = {
        "voyage-3-lite": 512,
        "voyage-3": 1024,
        "voyage-3-large": 1024,
        "voyage-code-3": 1024,
        "voyage-finance-2": 1024,
        "voyage-law-2": 1024,
    }
    DICT_MODEL_MAX_INPUT_TOKENS: ClassVar[dict[str, int]] = {
        "voyage-3-lite": 32_000,
        "voyage-3": 32_000,
        "voyage-3-large": 32_000,
        "voyage-code-3": 32_000,
        "voyage-finance-2": 32_000,
        "voyage-law-2": 16_000,
    }
    # Models accepting an output_dimension override (256/512/1024/2048).
    SET_CUSTOM_DIMENSION_MODELS: ClassVar[set[str]] = {
        "voyage-3-large",
        "voyage-code-3",
    }
    DEFAULT_DIMENSIONS: ClassVar[int] = 1024
    # The Voyage embed endpoint accepts at most 128 texts per request.
    MAX_BATCH_SIZE: ClassVar[int] = 128

    @classmethod
    def for_model(cls, model_name: str) -> "AIEmbeddingsCapabilitiesVoyage":
        """Create capabilities instance for the requested Voyage model."""
        normalized_name: str = model_name.strip().lower()
        # Normal return with per-model dimensions, limits, and registry pricing.
        return cls(
            supported_data_types=[SupportedDataType.TEXT],
            default_dimensions=cls.DICT_MODEL_DIMENSIONS.get(
                normalized_name, cls.DEFAULT_DIMENSIONS
            ),
            recommended_dimensions=(
                [256, 512, 1024, 2048]
                if normalized_name in cls.SET_CUSTOM_DIMENSION_MODELS
                else []
            ),
            max_input_tokens=cls.DICT_MODEL_MAX_INPUT_TOKENS.get(
                normalized_name, 32_000
            ),
            max_batch_size=cls.MAX_BATCH_SIZE,
            supports_async=True,
            pricing=get_model_pricing(PROVIDER_VOYAGE, normalized_name),
        )


class AiVoyageEmbeddings(AIBaseEmbeddings):
    """
    Embeddings client for Voyage AI, with batching, input_type retrieval
    hints, and async variants.
    """

    DEFAULT_EMBEDDING_MODEL: ClassVar[str] = "voyage-3"
    EMBEDDING_BATCH_MAX_SIZE: ClassVar[int] = (
        AIEmbeddingsCapabilitiesVoyage.MAX_BATCH_SIZE
    )

    def __init__(
        self,
        model: str = "",
        dimensions: int = 0,
        *,
        retry_policy: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the Voyage embeddings client.

        Args:
            model: Voyage model name; falls back to EMBEDDING_MODEL_NAME, then
                the voyage-3 default.
            dimensions: Optional output-dimension override, honored only on
                models that support it (voyage-3-large, voyage-code-3).
            retry_policy: "default" keeps the voyageai SDK's built-in retries;
                "none" disables them (max_retries=0). Falls back to the
                COMPLETIONS_RETRY_POLICY environment setting, then "default".
        """
        self.env: EnvSettings = EnvSettings()
        self.api_key: str = self.env.get_setting("VOYAGE_API_KEY", "")
        if not self.api_key or not self.api_key.strip():
            raise ValueError("VOYAGE_API_KEY environment variable must be set.")
        resolved_model: str = (
            model
            or self.env.get_setting("EMBEDDING_MODEL_NAME", "")
            or self.DEFAULT_EMBEDDING_MODEL
        )
        self.embedding_model: str = resolved_model.strip().lower()
        enforce_model_lifecycle(PROVIDER_VOYAGE, self.embedding_model)
        self._capabilities: AIEmbeddingsCapabilitiesVoyage = (
            AIEmbeddingsCapabilitiesVoyage.for_model(self.embedding_model)
        )
        int_dimensions: int = dimensions or self._capabilities.default_dimensions
        super().__init__(model=self.embedding_model, dimensions=int_dimensions)
        str_retry_candidate: str = (
            retry_policy
            if retry_policy is not None
            else str(
                self.env.get_setting("COMPLETIONS_RETRY_POLICY", RETRY_POLICY_DEFAULT)
            )
        )
        self.retry_policy: str = normalize_retry_policy(str_retry_candidate)
        # SDK clients are created lazily so the voyageai import happens only
        # on first use; construction works without the extra installed.
        self._client: Any | None = None
        self._async_client: Any | None = None

    # ── SDK access (lazy, extra-gated) ──────────────────────────────────────

    def _import_voyageai(self) -> Any:
        """
        Imports the voyageai SDK, raising the typed dependency error when the
        voyage extra is not installed.
        """
        try:
            import voyageai
        except ImportError as exception:
            raise AiProviderDependencyUnavailableError(
                VOYAGE_EXTRA_INSTALL_HINT
            ) from exception
        # Normal return with the imported SDK module.
        return voyageai

    def _build_client_kwargs(self) -> dict[str, Any]:
        """
        Builds SDK client kwargs shared by the sync and async clients.
        """
        dict_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.retry_policy == RETRY_POLICY_NONE:
            dict_kwargs["max_retries"] = 0
        # Normal return with the SDK client kwargs.
        return dict_kwargs

    @property
    def client(self) -> Any:
        """Returns the lazily created voyageai.Client."""
        if self._client is None:
            voyageai = self._import_voyageai()
            self._client = voyageai.Client(**self._build_client_kwargs())
        # Normal return with the shared sync client instance.
        return self._client

    @property
    def async_client(self) -> Any:
        """Returns the lazily created voyageai.AsyncClient."""
        if self._async_client is None:
            voyageai = self._import_voyageai()
            self._async_client = voyageai.AsyncClient(**self._build_client_kwargs())
        # Normal return with the shared async client instance.
        return self._async_client

    # ── Shared plumbing ─────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        """Identifier of the Voyage model in use."""
        return self.embedding_model

    @property
    def list_model_names(self) -> list[str]:
        """Catalogued Voyage embedding model identifiers."""
        return list(AIEmbeddingsCapabilitiesVoyage.DICT_MODEL_DIMENSIONS)

    @property
    def capabilities(self) -> AIEmbeddingsCapabilitiesVoyage:
        """Return the resolved capabilities for the current Voyage model."""
        return self._capabilities

    def _raise_request_error(self, exception: Exception) -> None:
        """
        Re-raises one voyageai SDK error as the typed request error.

        Carries the HTTP status code (when available) so caller-owned backoff
        can classify 429/5xx uniformly across providers. Non-SDK exceptions
        propagate unchanged.
        """
        str_module: str = type(exception).__module__ or ""
        if not str_module.startswith("voyageai"):
            # Normal return so non-SDK exceptions propagate unchanged.
            return None
        raw_status: Any = getattr(exception, "http_status", None)
        if raw_status is None:
            raw_status = getattr(exception, "status_code", None)
        raise AiProviderRequestError(
            f"Voyage request failed: {exception}",
            status_code=raw_status if isinstance(raw_status, int) else None,
            provider_engine=self.PROVIDER_ENGINE_VOYAGE,
        ) from exception

    def _embed_request_kwargs(
        self, texts: list[str], input_type: str | None
    ) -> dict[str, Any]:
        """
        Builds the SDK embed kwargs shared by sync and async calls.
        """
        dict_kwargs: dict[str, Any] = {
            "texts": texts,
            "model": self.embedding_model,
        }
        if input_type is not None:
            dict_kwargs["input_type"] = input_type
        if (
            self.dimensions
            and self.embedding_model
            in AIEmbeddingsCapabilitiesVoyage.SET_CUSTOM_DIMENSION_MODELS
            and self.dimensions != self._capabilities.default_dimensions
        ):
            dict_kwargs["output_dimension"] = self.dimensions
        # Normal return with the embed request kwargs.
        return dict_kwargs

    def _build_result_dicts(
        self, texts: list[str], list_embeddings: list[list[float]]
    ) -> list[dict[str, Any]]:
        """
        Maps SDK vectors onto the library-wide embeddings return shape.
        """
        if len(list_embeddings) != len(texts):
            raise AiProviderRequestError(
                f"Voyage returned {len(list_embeddings)} embeddings for "
                f"{len(texts)} texts.",
                status_code=None,
                provider_engine=self.PROVIDER_ENGINE_VOYAGE,
            )
        # Normal return with one result dict per input text.
        return [
            {
                "embedding": list(vector),
                "text": text,
                "dimensions": len(vector),
            }
            for text, vector in zip(texts, list_embeddings)
        ]

    @staticmethod
    def _validate_texts(texts: list[str]) -> None:
        """
        Rejects empty inputs before any provider call.
        """
        if not texts:
            raise ValueError("texts cannot be empty.")
        # Loop over inputs so blank entries fail fast with a clear index.
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"texts[{index}] must be a non-empty string.")
        # Normal return after validation.
        return None

    @staticmethod
    def _chunk_texts(texts: list[str], int_chunk_size: int) -> list[list[str]]:
        """
        Splits texts into provider-sized chunks (Voyage caps requests at 128).
        """
        # Normal return with contiguous chunks preserving input order.
        return [
            texts[index : index + int_chunk_size]
            for index in range(0, len(texts), int_chunk_size)
        ]

    def _observed_batch_result(
        self,
        list_result_dicts: list[dict[str, Any]],
        int_input_tokens: int | None,
        *,
        return_single: bool,
    ) -> AiApiObservedEmbeddingsResultModel[Any]:
        """
        Wraps result dicts as an observed embeddings result with usage.
        """
        int_dimensions: int | None = (
            int(list_result_dicts[0]["dimensions"]) if list_result_dicts else None
        )
        return_value: Any = list_result_dicts[0] if return_single else list_result_dicts
        # Normal return with the observed embeddings result and usage metadata.
        return AiApiObservedEmbeddingsResultModel(
            return_value=return_value,
            embedding_count=len(list_result_dicts),
            returned_dimensions=int_dimensions,
            provider_input_tokens=int_input_tokens,
            provider_total_tokens=int_input_tokens,
        )

    # ── Public surface ──────────────────────────────────────────────────────

    def generate_embeddings(
        self, text: str, *, input_type: str | None = None
    ) -> dict[str, Any]:
        """
        Generates one embedding via the Voyage embed endpoint.

        Args:
            text: Text to embed.
            input_type: Optional retrieval hint ("query" or "document"),
                forwarded to the API to improve retrieval quality.
        """
        self._validate_texts([text])
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = (
            self._build_embeddings_observability_input_metadata(
                list_texts=[text],
                bool_is_batch=False,
                requested_dimensions=self.dimensions,
            )
        )

        def _execute_single() -> AiApiObservedEmbeddingsResultModel[dict[str, Any]]:
            try:
                result = self.client.embed(
                    **self._embed_request_kwargs([text], input_type)
                )
            except Exception as exception:
                self._raise_request_error(exception)
                raise
            list_result_dicts = self._build_result_dicts(
                [text], list(getattr(result, "embeddings", None) or [])
            )
            # Normal return with the observed single-embedding result.
            return self._observed_batch_result(
                list_result_dicts,
                getattr(result, "total_tokens", None),
                return_single=True,
            )

        observed_result: AiApiObservedEmbeddingsResultModel[dict[str, Any]] = (
            self._execute_provider_call_with_observability(
                capability=self.CLIENT_TYPE_EMBEDDING,
                operation="generate_embeddings",
                dict_input_metadata=dict_input_metadata,
                callable_execute=_execute_single,
                callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_embeddings_observability_result_summary(
                    observed_result=result,
                    provider_elapsed_ms=provider_elapsed_ms,
                ),
            )
        )
        # Normal return with the caller-facing embedding dict.
        return observed_result.return_value

    def generate_embeddings_batch(
        self, texts: list[str], *, input_type: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Generates embeddings for many texts, chunking internally at the
        provider's 128-text request cap.

        Args:
            texts: Texts to embed (any length; chunked internally).
            input_type: Optional retrieval hint ("query" or "document").
        """
        self._validate_texts(texts)
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = (
            self._build_embeddings_observability_input_metadata(
                list_texts=texts,
                bool_is_batch=True,
                requested_dimensions=self.dimensions,
            )
        )

        def _execute_batch() -> (
            AiApiObservedEmbeddingsResultModel[list[dict[str, Any]]]
        ):
            list_all_dicts: list[dict[str, Any]] = []
            int_total_tokens: int = 0
            bool_any_tokens: bool = False
            # Loop over provider-sized chunks so any batch size works.
            for list_chunk in self._chunk_texts(texts, self.EMBEDDING_BATCH_MAX_SIZE):
                try:
                    result = self.client.embed(
                        **self._embed_request_kwargs(list_chunk, input_type)
                    )
                except Exception as exception:
                    self._raise_request_error(exception)
                    raise
                list_all_dicts.extend(
                    self._build_result_dicts(
                        list_chunk,
                        list(getattr(result, "embeddings", None) or []),
                    )
                )
                int_chunk_tokens: Any = getattr(result, "total_tokens", None)
                if isinstance(int_chunk_tokens, int):
                    int_total_tokens += int_chunk_tokens
                    bool_any_tokens = True
            # Normal return with the observed batch result.
            return self._observed_batch_result(
                list_all_dicts,
                int_total_tokens if bool_any_tokens else None,
                return_single=False,
            )

        observed_result: AiApiObservedEmbeddingsResultModel[list[dict[str, Any]]] = (
            self._execute_provider_call_with_observability(
                capability=self.CLIENT_TYPE_EMBEDDING,
                operation="generate_embeddings_batch",
                dict_input_metadata=dict_input_metadata,
                callable_execute=_execute_batch,
                callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_embeddings_observability_result_summary(
                    observed_result=result,
                    provider_elapsed_ms=provider_elapsed_ms,
                ),
            )
        )
        # Normal return with the caller-facing embedding dict list.
        return observed_result.return_value

    async def _agenerate_embeddings_provider(
        self, text: str, *, input_type: str | None
    ) -> dict[str, Any]:
        """
        Async twin of generate_embeddings via voyageai.AsyncClient.
        """
        self._validate_texts([text])
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = (
            self._build_embeddings_observability_input_metadata(
                list_texts=[text],
                bool_is_batch=False,
                requested_dimensions=self.dimensions,
            )
        )

        async def _execute_single() -> (
            AiApiObservedEmbeddingsResultModel[dict[str, Any]]
        ):
            try:
                result = await self.async_client.embed(
                    **self._embed_request_kwargs([text], input_type)
                )
            except Exception as exception:
                self._raise_request_error(exception)
                raise
            list_result_dicts = self._build_result_dicts(
                [text], list(getattr(result, "embeddings", None) or [])
            )
            # Normal return with the observed single-embedding result.
            return self._observed_batch_result(
                list_result_dicts,
                getattr(result, "total_tokens", None),
                return_single=True,
            )

        observed_result: AiApiObservedEmbeddingsResultModel[dict[str, Any]] = (
            await self._execute_provider_acall_with_observability(
                capability=self.CLIENT_TYPE_EMBEDDING,
                operation="agenerate_embeddings",
                dict_input_metadata=dict_input_metadata,
                callable_execute=_execute_single,
                callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_embeddings_observability_result_summary(
                    observed_result=result,
                    provider_elapsed_ms=provider_elapsed_ms,
                ),
            )
        )
        # Normal return with the caller-facing embedding dict.
        return observed_result.return_value

    async def _agenerate_embeddings_batch_provider(
        self, texts: list[str], *, input_type: str | None
    ) -> list[dict[str, Any]]:
        """
        Async twin of generate_embeddings_batch via voyageai.AsyncClient.
        """
        self._validate_texts(texts)
        dict_input_metadata: dict[str, ObservabilityMetadataValue] = (
            self._build_embeddings_observability_input_metadata(
                list_texts=texts,
                bool_is_batch=True,
                requested_dimensions=self.dimensions,
            )
        )

        async def _execute_batch() -> (
            AiApiObservedEmbeddingsResultModel[list[dict[str, Any]]]
        ):
            list_all_dicts: list[dict[str, Any]] = []
            int_total_tokens: int = 0
            bool_any_tokens: bool = False
            # Loop over provider-sized chunks so any batch size works.
            for list_chunk in self._chunk_texts(texts, self.EMBEDDING_BATCH_MAX_SIZE):
                try:
                    result = await self.async_client.embed(
                        **self._embed_request_kwargs(list_chunk, input_type)
                    )
                except Exception as exception:
                    self._raise_request_error(exception)
                    raise
                list_all_dicts.extend(
                    self._build_result_dicts(
                        list_chunk,
                        list(getattr(result, "embeddings", None) or []),
                    )
                )
                int_chunk_tokens: Any = getattr(result, "total_tokens", None)
                if isinstance(int_chunk_tokens, int):
                    int_total_tokens += int_chunk_tokens
                    bool_any_tokens = True
            # Normal return with the observed batch result.
            return self._observed_batch_result(
                list_all_dicts,
                int_total_tokens if bool_any_tokens else None,
                return_single=False,
            )

        observed_result: AiApiObservedEmbeddingsResultModel[list[dict[str, Any]]] = (
            await self._execute_provider_acall_with_observability(
                capability=self.CLIENT_TYPE_EMBEDDING,
                operation="agenerate_embeddings_batch",
                dict_input_metadata=dict_input_metadata,
                callable_execute=_execute_batch,
                callable_build_result_summary=lambda result, provider_elapsed_ms: self._build_embeddings_observability_result_summary(
                    observed_result=result,
                    provider_elapsed_ms=provider_elapsed_ms,
                ),
            )
        )
        # Normal return with the caller-facing embedding dict list.
        return observed_result.return_value
