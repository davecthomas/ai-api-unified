# ruff: noqa: E402

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from collections.abc import Callable

import pytest

pytest.importorskip("boto3")
pytest.importorskip("google.genai")
pytest.importorskip("openai")

from ai_api_unified.ai_base import AiApiObservedEmbeddingsResultModel
from ai_api_unified.embeddings.ai_google_gemini_embeddings import (
    GoogleGeminiEmbeddings,
)
from ai_api_unified.embeddings.ai_openai_embeddings import (
    AiOpenAIEmbeddings,
)
from ai_api_unified.embeddings.ai_titan_embeddings import AiTitanEmbeddings
from ai_api_unified.middleware.observability import (
    AiApiObservabilityMiddleware,
)
from ai_api_unified.middleware.observability_runtime import (
    AiApiCallContextModel,
    AiApiCallResultSummaryModel,
    reset_observability_context,
    set_observability_context,
)

TEST_EMBEDDING_MODEL: str = "test-embedding-model"
TEST_OPENAI_CUSTOM_DIMENSIONS_MODEL: str = "text-embedding-3-small"
TEST_LEGACY_USER: str = "legacy-openai-user"
TEST_EXPLICIT_CALLER_ID: str = "app-caller-123"
TEST_OPENAI_TEXT: str = "embed this text"


class RecordingObservabilityMiddleware(AiApiObservabilityMiddleware):
    """
    Records lifecycle events so embedding-provider tests can assert ordering and metadata.

    Args:
        None

    Returns:
        Enabled middleware test double that stores before, after, and error events.
    """

    def __init__(self) -> None:
        """
        Initializes empty event collections used by embeddings observability tests.

        Args:
            None

        Returns:
            None after the middleware test double is ready to record events.
        """
        self.list_before_contexts: list[AiApiCallContextModel] = []
        self.list_after_events: list[
            tuple[AiApiCallContextModel, AiApiCallResultSummaryModel]
        ] = []
        self.list_error_events: list[tuple[AiApiCallContextModel, Exception, float]] = (
            []
        )

    @property
    def bool_enabled(self) -> bool:
        """
        Indicates that the recording middleware is enabled for all tests.

        Args:
            None

        Returns:
            True because tests should exercise observability hooks.
        """
        # Normal return because the recording middleware is always enabled in tests.
        return True

    def before_call(self, call_context: AiApiCallContextModel) -> None:
        """
        Records one input lifecycle event.

        Args:
            call_context: Immutable call-context metadata emitted before provider execution.

        Returns:
            None after storing the input event payload.
        """
        self.list_before_contexts.append(call_context)
        # Normal return after recording the input lifecycle event.
        return None

    def after_call(
        self,
        call_context: AiApiCallContextModel,
        call_result_summary: AiApiCallResultSummaryModel,
    ) -> None:
        """
        Records one output lifecycle event.

        Args:
            call_context: Immutable call-context metadata emitted after provider execution.
            call_result_summary: Metadata-only summary derived from the provider output.

        Returns:
            None after storing the output event payload.
        """
        self.list_after_events.append((call_context, call_result_summary))
        # Normal return after recording the output lifecycle event.
        return None

    def on_error(
        self,
        call_context: AiApiCallContextModel,
        exception: Exception,
        float_elapsed_ms: float,
    ) -> None:
        """
        Records one error lifecycle event.

        Args:
            call_context: Immutable call-context metadata emitted for the error event.
            exception: Provider exception propagated through the wrapper.
            float_elapsed_ms: Elapsed milliseconds measured before the failure was surfaced.

        Returns:
            None after storing the error event payload.
        """
        self.list_error_events.append((call_context, exception, float_elapsed_ms))
        # Normal return after recording the error lifecycle event.
        return None


def _build_openai_embedding_response(
    *,
    list_embeddings: list[list[float]],
    prompt_tokens: int,
    total_tokens: int,
) -> Any:
    """
    Builds a minimal OpenAI embeddings response object for unit tests.

    Args:
        list_embeddings: Ordered provider vectors returned by the fake SDK response.
        prompt_tokens: Provider-reported input token count.
        total_tokens: Provider-reported total token count.

    Returns:
        Simple namespace object matching the attributes accessed by the provider code.
    """
    response: Any = SimpleNamespace(
        data=[SimpleNamespace(embedding=embedding) for embedding in list_embeddings],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
        ),
    )
    # Normal return with a fake OpenAI embeddings response object.
    return response


def _build_gemini_embedding_response(
    *,
    list_embeddings: list[list[float]],
    prompt_tokens: int,
    total_tokens: int,
) -> Any:
    """
    Builds a minimal Gemini embeddings response object for unit tests.

    Args:
        list_embeddings: Ordered provider vectors returned by the fake SDK response.
        prompt_tokens: Provider-reported input token count.
        total_tokens: Provider-reported total token count.

    Returns:
        Simple namespace object matching the attributes accessed by the provider code.
    """
    response: Any = SimpleNamespace(
        embeddings=[SimpleNamespace(values=embedding) for embedding in list_embeddings],
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            total_token_count=total_tokens,
        ),
    )
    # Normal return with a fake Gemini embeddings response object.
    return response


def _build_openai_embeddings_client(
    *,
    middleware: RecordingObservabilityMiddleware,
    callable_create: Callable[..., Any],
    dimensions: int = 3,
) -> AiOpenAIEmbeddings:
    """
    Builds a partially initialized OpenAI embeddings client for observability tests.

    Args:
        middleware: Recording observability middleware used to capture lifecycle events.
        callable_create: Fake OpenAI embeddings create callable used by the provider code.
        dimensions: Requested embedding dimensions reported by the test client.

    Returns:
        AiOpenAIEmbeddings instance with fake SDK dependencies injected.
    """
    ai_openai_embeddings: AiOpenAIEmbeddings = AiOpenAIEmbeddings.__new__(
        AiOpenAIEmbeddings
    )
    ai_openai_embeddings.embedding_model = TEST_EMBEDDING_MODEL
    ai_openai_embeddings.dimensions = dimensions
    ai_openai_embeddings.client = SimpleNamespace(
        embeddings=SimpleNamespace(create=callable_create)
    )
    ai_openai_embeddings.backoff_delays = [0]
    ai_openai_embeddings.user = TEST_LEGACY_USER
    ai_openai_embeddings._observability_middleware = middleware
    # Normal return with a test-configured OpenAI embeddings client.
    return ai_openai_embeddings


def _build_gemini_embeddings_client(
    *,
    middleware: RecordingObservabilityMiddleware,
    callable_embed_content: Callable[..., Any],
    dimensions: int = 3,
    max_batch_size: int = 2,
) -> GoogleGeminiEmbeddings:
    """
    Builds a partially initialized Gemini embeddings client for observability tests.

    Args:
        middleware: Recording observability middleware used to capture lifecycle events.
        callable_embed_content: Fake Gemini embeddings callable used by the provider code.
        dimensions: Requested embedding dimensions reported by the test client.
        max_batch_size: Provider-side batch slice size used to force multi-request batches.

    Returns:
        GoogleGeminiEmbeddings instance with fake SDK dependencies injected.
    """
    ai_google_embeddings: GoogleGeminiEmbeddings = GoogleGeminiEmbeddings.__new__(
        GoogleGeminiEmbeddings
    )
    ai_google_embeddings.embedding_model = TEST_EMBEDDING_MODEL
    ai_google_embeddings.dimensions = dimensions
    ai_google_embeddings.client = SimpleNamespace(
        models=SimpleNamespace(embed_content=callable_embed_content)
    )
    ai_google_embeddings.max_retries = 1
    ai_google_embeddings.initial_delay = 0.0
    ai_google_embeddings.backoff_multiplier = 1.0
    ai_google_embeddings.max_jitter = 0.0
    ai_google_embeddings.MAX_BATCH_SIZE = max_batch_size
    ai_google_embeddings._observability_middleware = middleware
    # Normal return with a test-configured Gemini embeddings client.
    return ai_google_embeddings


def _build_titan_embeddings_client(
    *,
    middleware: RecordingObservabilityMiddleware,
    dimensions: int = 3,
) -> AiTitanEmbeddings:
    """
    Builds a partially initialized Titan embeddings client for observability tests.

    Args:
        middleware: Recording observability middleware used to capture lifecycle events.
        dimensions: Requested embedding dimensions reported by the test client.

    Returns:
        AiTitanEmbeddings instance with fake dependencies injected.
    """
    ai_titan_embeddings: AiTitanEmbeddings = AiTitanEmbeddings.__new__(
        AiTitanEmbeddings
    )
    ai_titan_embeddings.embedding_model = TEST_EMBEDDING_MODEL
    ai_titan_embeddings.dimensions = dimensions
    ai_titan_embeddings.backoff_delays = [0]
    ai_titan_embeddings._observability_middleware = middleware
    # Normal return with a test-configured Titan embeddings client.
    return ai_titan_embeddings


def test_openai_single_embeddings_emit_metadata_and_explicit_user_propagation() -> None:
    """
    Verify one OpenAI single embeddings call emits one metadata-only sequence and propagates explicit caller context.

    Args:
        None

    Returns:
        None after asserting input metadata, output metadata, and OpenAI `user` propagation.
    """
    middleware = RecordingObservabilityMiddleware()
    dict_captured_params: dict[str, Any] = {}

    def _fake_create(**params: Any) -> Any:
        """
        Capture OpenAI request parameters and return one fake embeddings response.

        Args:
            **params: Provider request parameters supplied by the client under test.

        Returns:
            Fake OpenAI embeddings response object for unit-test assertions.
        """
        dict_captured_params.update(params)
        # Normal return with a deterministic fake embeddings response.
        return _build_openai_embedding_response(
            list_embeddings=[[0.1, 0.2, 0.3]],
            prompt_tokens=8,
            total_tokens=8,
        )

    client = _build_openai_embeddings_client(
        middleware=middleware,
        callable_create=_fake_create,
    )

    context_token = set_observability_context(caller_id=TEST_EXPLICIT_CALLER_ID)
    try:
        dict_embedding_result = client.generate_embeddings(TEST_OPENAI_TEXT)
    finally:
        reset_observability_context(context_token)

    assert dict_embedding_result["text"] == TEST_OPENAI_TEXT
    assert dict_embedding_result["dimensions"] == 3
    assert dict_captured_params["user"] == TEST_EXPLICIT_CALLER_ID
    assert len(middleware.list_before_contexts) == 1
    assert len(middleware.list_after_events) == 1

    before_context = middleware.list_before_contexts[0]
    after_context, result_summary = middleware.list_after_events[0]

    assert before_context.operation == "generate_embeddings"
    assert before_context.dict_metadata["input_text_count"] == 1
    assert before_context.dict_metadata["input_text_total_chars"] == len(
        TEST_OPENAI_TEXT
    )
    assert before_context.dict_metadata["requested_dimensions"] == 3
    assert before_context.originating_caller_id == TEST_EXPLICIT_CALLER_ID
    assert after_context.direction == "output"
    assert result_summary.input_token_count == 8
    assert result_summary.provider_total_tokens == 8
    assert result_summary.dict_metadata["embedding_count"] == 1
    assert result_summary.dict_metadata["returned_dimensions"] == 3


def test_openai_batch_embeddings_emit_one_sequence() -> None:
    """
    Verify one OpenAI batch embeddings call emits one metadata-only lifecycle sequence.

    Args:
        None

    Returns:
        None after asserting batch metadata and output summary fields.
    """
    middleware = RecordingObservabilityMiddleware()

    def _fake_create(**_: Any) -> Any:
        """
        Return a fake OpenAI batch embeddings response.

        Args:
            **_: Ignored provider request parameters.

        Returns:
            Fake OpenAI embeddings response object for unit-test assertions.
        """
        # Normal return with a deterministic fake batch embeddings response.
        return _build_openai_embedding_response(
            list_embeddings=[[0.1, 0.2], [0.3, 0.4]],
            prompt_tokens=11,
            total_tokens=11,
        )

    client = _build_openai_embeddings_client(
        middleware=middleware,
        callable_create=_fake_create,
        dimensions=2,
    )

    list_results = client.generate_embeddings_batch(["alpha", "beta"])

    assert len(list_results) == 2
    assert len(middleware.list_before_contexts) == 1
    assert len(middleware.list_after_events) == 1

    before_context = middleware.list_before_contexts[0]
    _, result_summary = middleware.list_after_events[0]

    assert before_context.operation == "generate_embeddings_batch"
    assert before_context.dict_metadata["batch_mode"] is True
    assert before_context.dict_metadata["input_text_count"] == 2
    assert result_summary.input_token_count == 11
    assert result_summary.dict_metadata["embedding_count"] == 2
    assert result_summary.dict_metadata["returned_dimensions"] == 2


def test_gemini_batch_embeddings_aggregate_slice_metadata_into_one_sequence() -> None:
    """
    Verify one Gemini batch embeddings call aggregates multiple provider slices into one observability sequence.

    Args:
        None

    Returns:
        None after asserting aggregated token metadata and slice-count metadata.
    """
    middleware = RecordingObservabilityMiddleware()
    list_fake_responses: list[Any] = [
        _build_gemini_embedding_response(
            list_embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            prompt_tokens=5,
            total_tokens=5,
        ),
        _build_gemini_embedding_response(
            list_embeddings=[[0.7, 0.8, 0.9]],
            prompt_tokens=4,
            total_tokens=4,
        ),
    ]

    def _fake_embed_content(**_: Any) -> Any:
        """
        Return the next fake Gemini slice response.

        Args:
            **_: Ignored provider request parameters.

        Returns:
            Fake Gemini embeddings response object for unit-test assertions.
        """
        # Normal return with the next deterministic fake Gemini slice response.
        return list_fake_responses.pop(0)

    client = _build_gemini_embeddings_client(
        middleware=middleware,
        callable_embed_content=_fake_embed_content,
        dimensions=3,
        max_batch_size=2,
    )

    list_results = client.generate_embeddings_batch(["alpha", "beta", "gamma"])

    assert len(list_results) == 3
    assert len(middleware.list_before_contexts) == 1
    assert len(middleware.list_after_events) == 1

    before_context = middleware.list_before_contexts[0]
    _, result_summary = middleware.list_after_events[0]

    assert before_context.dict_metadata["input_text_count"] == 3
    assert before_context.dict_metadata["batch_mode"] is True
    assert result_summary.input_token_count == 9
    assert result_summary.provider_total_tokens == 9
    assert result_summary.dict_metadata["embedding_count"] == 3
    assert result_summary.dict_metadata["returned_dimensions"] == 3
    assert result_summary.dict_metadata["provider_request_count"] == 2


def test_titan_batch_embeddings_emit_one_sequence_for_thread_pool_batch() -> None:
    """
    Verify one Titan batch embeddings call emits one observability sequence even though worker threads do the provider work.

    Args:
        None

    Returns:
        None after asserting aggregated token metadata and single-sequence emission.
    """
    middleware = RecordingObservabilityMiddleware()
    client = _build_titan_embeddings_client(middleware=middleware, dimensions=3)

    def _fake_generate_single_embedding_raw(
        text: str,
    ) -> tuple[dict[str, Any], int | None]:
        """
        Return one fake Titan embedding payload for a worker-thread task.

        Args:
            text: Input text assigned to the current worker-thread task.

        Returns:
            Tuple containing a fake Titan embedding payload and provider token count.
        """
        dict_embedding_result: dict[str, Any] = {
            "embedding": [0.1, 0.2, 0.3],
            "text": text,
            "dimensions": 3,
            "input_tokens": len(text),
        }
        # Normal return with a deterministic fake Titan worker result.
        return dict_embedding_result, len(text)

    client._generate_single_embedding_raw = _fake_generate_single_embedding_raw

    list_results = client.generate_embeddings_batch(["a", "bb", "ccc"])

    assert len(list_results) == 3
    assert len(middleware.list_before_contexts) == 1
    assert len(middleware.list_after_events) == 1
    assert middleware.list_error_events == []

    before_context = middleware.list_before_contexts[0]
    _, result_summary = middleware.list_after_events[0]

    assert before_context.operation == "generate_embeddings_batch"
    assert before_context.dict_metadata["input_text_total_chars"] == 6
    assert result_summary.input_token_count == 6
    assert result_summary.provider_total_tokens == 6
    assert result_summary.dict_metadata["embedding_count"] == 3
    assert result_summary.dict_metadata["returned_dimensions"] == 3


def test_observed_embeddings_result_metadata_is_immutable() -> None:
    """
    Verify observed embeddings result metadata is wrapped in an immutable mapping.

    Args:
        None

    Returns:
        None after asserting metadata writes are rejected.
    """
    observed_result = AiApiObservedEmbeddingsResultModel[dict[str, Any]](
        return_value={"embedding": [0.1], "text": "alpha", "dimensions": 1},
        embedding_count=1,
        returned_dimensions=1,
        dict_metadata={"provider_request_count": 1},
    )

    with pytest.raises(TypeError):
        observed_result.dict_metadata["provider_request_count"] = 2


def test_openai_single_embeddings_raise_clear_error_for_empty_response() -> None:
    """
    Verify OpenAI single embeddings fail with a clear error when no vectors are returned.

    Args:
        None

    Returns:
        None after asserting the provider response shape is validated explicitly.
    """
    middleware = RecordingObservabilityMiddleware()
    client = _build_openai_embeddings_client(
        middleware=middleware,
        callable_create=lambda **_: _build_openai_embedding_response(
            list_embeddings=[],
            prompt_tokens=0,
            total_tokens=0,
        ),
    )

    with pytest.raises(RuntimeError, match="did not contain any embeddings"):
        client.generate_embeddings(TEST_OPENAI_TEXT)


def test_openai_batch_embeddings_raise_clear_error_for_count_mismatch() -> None:
    """
    Verify OpenAI batch embeddings fail when the provider returns fewer vectors than requested.

    Args:
        None

    Returns:
        None after asserting count mismatches do not silently truncate outputs.
    """
    middleware = RecordingObservabilityMiddleware()
    client = _build_openai_embeddings_client(
        middleware=middleware,
        callable_create=lambda **_: _build_openai_embedding_response(
            list_embeddings=[[0.1, 0.2, 0.3]],
            prompt_tokens=3,
            total_tokens=3,
        ),
    )

    with pytest.raises(RuntimeError, match="response count mismatch"):
        client.generate_embeddings_batch(["alpha", "beta"])


def test_openai_single_embeddings_include_requested_dimensions_in_request() -> None:
    """
    Verify OpenAI single embeddings pass configured dimensions to models that support them.

    Args:
        None

    Returns:
        None after asserting the provider request includes the requested dimensions.
    """
    middleware = RecordingObservabilityMiddleware()
    dict_captured_params: dict[str, Any] = {}

    def _fake_create(**params: Any) -> Any:
        """
        Capture OpenAI single embeddings request parameters for assertion.

        Args:
            **params: Provider request parameters supplied by the client under test.

        Returns:
            Fake OpenAI embeddings response matching the requested dimensions.
        """
        dict_captured_params.update(params)
        # Normal return with a deterministic fake embeddings response.
        return _build_openai_embedding_response(
            list_embeddings=[[0.1, 0.2, 0.3]],
            prompt_tokens=3,
            total_tokens=3,
        )

    client = _build_openai_embeddings_client(
        middleware=middleware,
        callable_create=_fake_create,
        dimensions=3,
    )
    client.embedding_model = TEST_OPENAI_CUSTOM_DIMENSIONS_MODEL

    dict_embedding_result = client.generate_embeddings("alpha")

    assert dict_captured_params["dimensions"] == 3
    assert dict_embedding_result["dimensions"] == 3


def test_openai_batch_embeddings_include_requested_dimensions_in_request() -> None:
    """
    Verify OpenAI batch embeddings pass configured dimensions to models that support them.

    Args:
        None

    Returns:
        None after asserting the provider request includes the requested dimensions.
    """
    middleware = RecordingObservabilityMiddleware()
    dict_captured_params: dict[str, Any] = {}

    def _fake_create(**params: Any) -> Any:
        """
        Capture OpenAI batch embeddings request parameters for assertion.

        Args:
            **params: Provider request parameters supplied by the client under test.

        Returns:
            Fake OpenAI embeddings response matching the requested dimensions.
        """
        dict_captured_params.update(params)
        # Normal return with a deterministic fake embeddings response.
        return _build_openai_embedding_response(
            list_embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            prompt_tokens=6,
            total_tokens=6,
        )

    client = _build_openai_embeddings_client(
        middleware=middleware,
        callable_create=_fake_create,
        dimensions=3,
    )
    client.embedding_model = TEST_OPENAI_CUSTOM_DIMENSIONS_MODEL

    list_embedding_results = client.generate_embeddings_batch(["alpha", "beta"])

    assert dict_captured_params["dimensions"] == 3
    assert all(
        dict_embedding_result["dimensions"] == 3
        for dict_embedding_result in list_embedding_results
    )


def test_gemini_single_embeddings_raise_clear_error_for_empty_response() -> None:
    """
    Verify Gemini single embeddings fail with a clear error when no vectors are returned.

    Args:
        None

    Returns:
        None after asserting the provider response shape is validated explicitly.
    """
    middleware = RecordingObservabilityMiddleware()
    client = _build_gemini_embeddings_client(
        middleware=middleware,
        callable_embed_content=lambda **_: _build_gemini_embedding_response(
            list_embeddings=[],
            prompt_tokens=0,
            total_tokens=0,
        ),
    )

    with pytest.raises(RuntimeError, match="did not contain any embeddings"):
        client.generate_embeddings("alpha")


def test_gemini_batch_embeddings_raise_clear_error_for_count_mismatch() -> None:
    """
    Verify Gemini batch embeddings fail when one provider slice returns too few vectors.

    Args:
        None

    Returns:
        None after asserting count mismatches do not silently truncate outputs.
    """
    middleware = RecordingObservabilityMiddleware()
    client = _build_gemini_embeddings_client(
        middleware=middleware,
        callable_embed_content=lambda **_: _build_gemini_embedding_response(
            list_embeddings=[[0.1, 0.2, 0.3]],
            prompt_tokens=3,
            total_tokens=3,
        ),
        max_batch_size=2,
    )

    with pytest.raises(RuntimeError, match="response count mismatch"):
        client.generate_embeddings_batch(["alpha", "beta"])


def test_titan_batch_embeddings_reject_blank_texts() -> None:
    """
    Verify Titan batch embeddings preserve non-empty text validation before worker submission.

    Args:
        None

    Returns:
        None after asserting blank batch inputs fail fast.
    """
    middleware = RecordingObservabilityMiddleware()
    client = _build_titan_embeddings_client(middleware=middleware, dimensions=3)

    with pytest.raises(ValueError, match="All input texts must be non-empty strings"):
        client.generate_embeddings_batch(["alpha", "   "])


def test_titan_raw_embeddings_error_omits_input_text_for_empty_body() -> None:
    """
    Verify Titan empty-body failures do not leak the input text in the raised error.

    Args:
        None

    Returns:
        None after asserting the error message is metadata-only.
    """
    middleware = RecordingObservabilityMiddleware()
    client = _build_titan_embeddings_client(middleware=middleware, dimensions=3)
    client.client = SimpleNamespace(
        invoke_model=lambda **_: {"body": None},
    )

    with pytest.raises(
        RuntimeError, match="Empty response body from Titan"
    ) as exc_info:
        client._generate_single_embedding_raw("highly sensitive text")

    assert "highly sensitive text" not in str(exc_info.value)


def test_titan_raw_embeddings_report_dimension_mismatch_explicitly() -> None:
    """
    Verify Titan dimension mismatches report expected and actual vector widths.

    Args:
        None

    Returns:
        None after asserting mismatched vectors raise a precise error.
    """
    middleware = RecordingObservabilityMiddleware()
    client = _build_titan_embeddings_client(middleware=middleware, dimensions=3)
    client.client = SimpleNamespace(
        invoke_model=lambda **_: {
            "body": SimpleNamespace(
                read=lambda: b'{"embedding": [0.1, 0.2], "inputTextTokenCount": 2}'
            )
        },
    )

    with pytest.raises(
        RuntimeError,
        match="Expected 3 but received 2",
    ):
        client._generate_single_embedding_raw("alpha")


def test_titan_batch_embeddings_preserve_missing_token_counts_as_none() -> None:
    """
    Verify Titan batch observability does not coerce missing provider token counts to zero.

    Args:
        None

    Returns:
        None after asserting missing worker token counts propagate as None.
    """
    middleware = RecordingObservabilityMiddleware()
    client = _build_titan_embeddings_client(middleware=middleware, dimensions=3)

    def _fake_generate_single_embedding_raw(
        text: str,
    ) -> tuple[dict[str, Any], int | None]:
        """
        Return one fake Titan embedding payload without provider token counts.

        Args:
            text: Input text assigned to the current worker-thread task.

        Returns:
            Tuple containing a fake Titan embedding payload and no provider token count.
        """
        dict_embedding_result: dict[str, Any] = {
            "embedding": [0.1, 0.2, 0.3],
            "text": text,
            "dimensions": 3,
            "input_tokens": None,
        }
        # Normal return with a fake Titan worker result that omits token counts.
        return dict_embedding_result, None

    client._generate_single_embedding_raw = _fake_generate_single_embedding_raw

    list_results = client.generate_embeddings_batch(["a", "bb"])

    assert len(list_results) == 2
    _, result_summary = middleware.list_after_events[0]
    assert result_summary.input_token_count is None
    assert result_summary.provider_total_tokens is None
