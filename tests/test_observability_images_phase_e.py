# ruff: noqa: E402

from __future__ import annotations

import base64
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("boto3")
pytest.importorskip("openai")

from ai_api_unified.images.ai_bedrock_images import (
    AINovaCanvasImageProperties,
    AINovaCanvasImages,
)
from ai_api_unified.images.ai_openai_images import (
    AIOpenAIImageProperties,
    AIOpenAIImages,
)
from ai_api_unified.middleware.observability import (
    AiApiObservabilityMiddleware,
)
from ai_api_unified.middleware.observability_runtime import (
    AiApiCallContextModel,
    AiApiCallResultSummaryModel,
    reset_observability_context,
    set_observability_context,
)

TEST_IMAGE_MODEL: str = "gpt-image-1"
TEST_BEDROCK_IMAGE_MODEL: str = "amazon.nova-canvas-v1:0"
TEST_IMAGE_PROMPT: str = "Draw a lighthouse at sunset."
TEST_EXPLICIT_CALLER_ID: str = "image-caller-123"
TEST_IMAGE_BYTES_ONE: bytes = b"image-bytes-one"
TEST_IMAGE_BYTES_TWO: bytes = b"image-bytes-two"


class RecordingObservabilityMiddleware(AiApiObservabilityMiddleware):
    """
    Records lifecycle events so image-provider tests can assert ordering and metadata.

    Args:
        None

    Returns:
        Enabled middleware test double that stores before, after, and error events.
    """

    def __init__(self) -> None:
        """
        Initializes empty event collections used by image observability tests.

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


def _build_openai_images_client(
    *,
    middleware: RecordingObservabilityMiddleware,
    callable_generate: Callable[..., Any],
) -> AIOpenAIImages:
    """
    Builds a partially initialized OpenAI images client for observability tests.

    Args:
        middleware: Recording observability middleware used to capture lifecycle events.
        callable_generate: Fake OpenAI image generation callable used by the provider code.

    Returns:
        AIOpenAIImages instance with fake SDK dependencies injected.
    """
    ai_openai_images: AIOpenAIImages = AIOpenAIImages.__new__(AIOpenAIImages)
    ai_openai_images.image_model_name = TEST_IMAGE_MODEL
    ai_openai_images.backoff_delays = [0]
    ai_openai_images.user = "legacy-openai-user"
    ai_openai_images.client = SimpleNamespace(
        images=SimpleNamespace(generate=callable_generate)
    )
    ai_openai_images._observability_middleware = middleware
    # Normal return with a test-configured OpenAI images client.
    return ai_openai_images


def _build_bedrock_images_client(
    *,
    middleware: RecordingObservabilityMiddleware,
    callable_invoke: Callable[..., dict[str, Any]],
) -> AINovaCanvasImages:
    """
    Builds a partially initialized Bedrock Nova Canvas client for observability tests.

    Args:
        middleware: Recording observability middleware used to capture lifecycle events.
        callable_invoke: Fake Bedrock JSON invocation callable used by the provider code.

    Returns:
        AINovaCanvasImages instance with fake provider dependencies injected.
    """
    ai_bedrock_images: AINovaCanvasImages = AINovaCanvasImages.__new__(
        AINovaCanvasImages
    )
    ai_bedrock_images.image_model_name = TEST_BEDROCK_IMAGE_MODEL
    ai_bedrock_images.region = "us-east-1"
    ai_bedrock_images._invoke_bedrock_json = callable_invoke
    ai_bedrock_images._observability_middleware = middleware
    # Normal return with a test-configured Bedrock images client.
    return ai_bedrock_images


def test_openai_images_emit_metadata_and_explicit_user_propagation() -> None:
    """
    Verify one OpenAI image call emits one metadata-only sequence and propagates explicit caller context.

    Args:
        None

    Returns:
        None after asserting input metadata, output metadata, and OpenAI `user` propagation.
    """
    middleware = RecordingObservabilityMiddleware()
    dict_captured_params: dict[str, Any] = {}

    def _fake_generate(**params: Any) -> Any:
        """
        Capture OpenAI image request parameters and return fake base64 image data.

        Args:
            **params: Provider request parameters supplied by the client under test.

        Returns:
            Fake OpenAI images response object for unit-test assertions.
        """
        dict_captured_params.update(params)
        # Normal return with a deterministic fake OpenAI images response.
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    b64_json=base64.b64encode(TEST_IMAGE_BYTES_ONE).decode("ascii")
                ),
                SimpleNamespace(
                    b64_json=base64.b64encode(TEST_IMAGE_BYTES_TWO).decode("ascii")
                ),
            ],
            usage=SimpleNamespace(input_tokens=7, total_tokens=7),
        )

    client = _build_openai_images_client(
        middleware=middleware,
        callable_generate=_fake_generate,
    )
    image_properties = AIOpenAIImageProperties(
        width=1024,
        height=1024,
        format="png",
        quality="medium",
        num_images=2,
    )

    context_token = set_observability_context(caller_id=TEST_EXPLICIT_CALLER_ID)
    try:
        list_images = client.generate_images(TEST_IMAGE_PROMPT, image_properties)
    finally:
        reset_observability_context(context_token)

    assert list_images == [TEST_IMAGE_BYTES_ONE, TEST_IMAGE_BYTES_TWO]
    assert dict_captured_params["user"] == TEST_EXPLICIT_CALLER_ID
    assert len(middleware.list_before_contexts) == 1
    assert len(middleware.list_after_events) == 1
    assert middleware.list_error_events == []

    before_context = middleware.list_before_contexts[0]
    _, result_summary = middleware.list_after_events[0]

    assert before_context.capability == "images"
    assert before_context.operation == "generate_images"
    assert before_context.dict_metadata["prompt_char_count"] == len(TEST_IMAGE_PROMPT)
    assert before_context.dict_metadata["requested_width"] == 1024
    assert before_context.dict_metadata["requested_height"] == 1024
    assert before_context.dict_metadata["requested_num_images"] == 2
    assert result_summary.input_token_count == 7
    assert result_summary.provider_total_tokens == 7
    assert result_summary.dict_metadata["generated_image_count"] == 2
    assert result_summary.dict_metadata["total_output_bytes"] == (
        len(TEST_IMAGE_BYTES_ONE) + len(TEST_IMAGE_BYTES_TWO)
    )
    assert result_summary.dict_metadata["output_format"] == "png"


def test_openai_images_use_legacy_caller_for_local_observability_only() -> None:
    """
    Verify OpenAI images fall back to the configured legacy user for local caller correlation only.

    Args:
        None

    Returns:
        None after asserting observability uses the legacy caller id without sending it to the provider.
    """
    middleware = RecordingObservabilityMiddleware()
    dict_captured_params: dict[str, Any] = {}

    def _fake_generate(**params: Any) -> Any:
        """
        Capture OpenAI image request parameters for legacy-caller fallback assertions.

        Args:
            **params: Provider request parameters supplied by the client under test.

        Returns:
            Fake OpenAI images response object for unit-test assertions.
        """
        dict_captured_params.update(params)
        # Normal return with a deterministic fake OpenAI images response.
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    b64_json=base64.b64encode(TEST_IMAGE_BYTES_ONE).decode("ascii")
                )
            ]
        )

    client = _build_openai_images_client(
        middleware=middleware,
        callable_generate=_fake_generate,
    )
    image_properties = AIOpenAIImageProperties(
        width=1024,
        height=1024,
        format="png",
        quality="medium",
        num_images=1,
    )

    list_images = client.generate_images(TEST_IMAGE_PROMPT, image_properties)

    assert list_images == [TEST_IMAGE_BYTES_ONE]
    assert "user" not in dict_captured_params
    assert len(middleware.list_before_contexts) == 1

    before_context = middleware.list_before_contexts[0]
    assert before_context.originating_caller_id == "legacy-openai-user"
    assert before_context.originating_caller_id_source == "legacy_setting"


def test_openai_images_emit_error_event_for_invalid_response() -> None:
    """
    Verify OpenAI image generation emits one error event when the provider returns no data entries.

    Args:
        None

    Returns:
        None after asserting the wrapper surfaces one error lifecycle event.
    """
    middleware = RecordingObservabilityMiddleware()
    client = _build_openai_images_client(
        middleware=middleware,
        callable_generate=lambda **_: SimpleNamespace(data=[]),
    )
    image_properties = AIOpenAIImageProperties(
        width=1024,
        height=1024,
        format="png",
        quality="medium",
        num_images=1,
    )

    with pytest.raises(ValueError, match="returned no data entries"):
        client.generate_images(TEST_IMAGE_PROMPT, image_properties)

    assert len(middleware.list_before_contexts) == 1
    assert middleware.list_after_events == []
    assert len(middleware.list_error_events) == 1


def test_openai_images_raise_clear_error_for_count_mismatch() -> None:
    """
    Verify OpenAI image generation fails when the provider returns fewer images than requested.

    Args:
        None

    Returns:
        None after asserting count mismatches do not silently truncate outputs.
    """
    middleware = RecordingObservabilityMiddleware()
    client = _build_openai_images_client(
        middleware=middleware,
        callable_generate=lambda **_: SimpleNamespace(
            data=[
                SimpleNamespace(
                    b64_json=base64.b64encode(TEST_IMAGE_BYTES_ONE).decode("ascii")
                )
            ]
        ),
    )
    image_properties = AIOpenAIImageProperties(
        width=1024,
        height=1024,
        format="png",
        quality="medium",
        num_images=2,
    )

    with pytest.raises(ValueError, match="unexpected number of images"):
        client.generate_images(TEST_IMAGE_PROMPT, image_properties)

    assert len(middleware.list_before_contexts) == 1
    assert middleware.list_after_events == []
    assert len(middleware.list_error_events) == 1


def test_bedrock_images_emit_metadata_for_nova_canvas_generation() -> None:
    """
    Verify one Nova Canvas image call emits one metadata-only sequence with request and output summaries.

    Args:
        None

    Returns:
        None after asserting input metadata and output byte-count summaries.
    """
    middleware = RecordingObservabilityMiddleware()

    def _fake_invoke(**_: Any) -> dict[str, Any]:
        """
        Return one fake Nova Canvas response with base64 image artifacts.

        Args:
            **_: Provider invocation parameters supplied by the client under test.

        Returns:
            Fake Bedrock Nova Canvas response for unit-test assertions.
        """
        # Normal return with a deterministic fake Nova Canvas response.
        return {
            "requestId": "request-123",
            "images": [
                base64.b64encode(TEST_IMAGE_BYTES_ONE).decode("ascii"),
                base64.b64encode(TEST_IMAGE_BYTES_TWO).decode("ascii"),
            ],
        }

    client = _build_bedrock_images_client(
        middleware=middleware,
        callable_invoke=_fake_invoke,
    )
    image_properties = AINovaCanvasImageProperties(
        width=1536,
        height=1024,
        format="png",
        quality="high",
        background="transparent",
        num_images=2,
        negative_text="no text overlay",
    )

    list_images = client.generate_images(TEST_IMAGE_PROMPT, image_properties)

    assert list_images == [TEST_IMAGE_BYTES_ONE, TEST_IMAGE_BYTES_TWO]
    assert len(middleware.list_before_contexts) == 1
    assert len(middleware.list_after_events) == 1
    assert middleware.list_error_events == []

    before_context = middleware.list_before_contexts[0]
    _, result_summary = middleware.list_after_events[0]

    assert before_context.provider_vendor == "bedrock"
    assert before_context.dict_metadata["prompt_char_count"] == len(TEST_IMAGE_PROMPT)
    assert before_context.dict_metadata["requested_width"] == 1536
    assert before_context.dict_metadata["requested_height"] == 1024
    assert before_context.dict_metadata["requested_num_images"] == 2
    assert before_context.dict_metadata["negative_prompt_char_count"] == len(
        "no text overlay"
    )
    assert result_summary.input_token_count is None
    assert result_summary.dict_metadata["generated_image_count"] == 2
    assert result_summary.dict_metadata["total_output_bytes"] == (
        len(TEST_IMAGE_BYTES_ONE) + len(TEST_IMAGE_BYTES_TWO)
    )
    assert result_summary.dict_metadata["request_id"] == "request-123"
    assert result_summary.dict_metadata["output_format"] == "png"
