# ruff: noqa: E402

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("elevenlabs")
pytest.importorskip("openai")
pytest.importorskip("azure.cognitiveservices.speech")
pytest.importorskip("google.genai")
pytest.importorskip("google.cloud.texttospeech")
pytest.importorskip("google.cloud.speech_v1p1beta1")

import ai_api_unified.voice.ai_voice_elevenlabs as ai_voice_elevenlabs_module
import ai_api_unified.voice.ai_voice_openai as ai_voice_openai_module
from ai_api_unified.middleware.observability import (
    AiApiObservabilityMiddleware,
)
from ai_api_unified.middleware.observability_runtime import (
    AiApiCallContextModel,
    AiApiCallResultSummaryModel,
    ORIGINATING_CALLER_ID_SOURCE_LEGACY_SETTING,
)
from ai_api_unified.voice.ai_voice_azure import (
    AIVoiceAzure,
    AIVoiceSelectionAzure,
)
from ai_api_unified.voice.ai_voice_elevenlabs import (
    AIVoiceElevenLabs,
    AIVoiceSelectionElevenLabs,
)
from ai_api_unified.voice.ai_voice_google import (
    AIVoiceGoogle,
    AIVoiceSelectionGoogle,
)
from ai_api_unified.voice.ai_voice_openai import (
    AIVoiceOpenAI,
    AIVoiceSelectionOpenAI,
)
from ai_api_unified.voice.audio_models import AudioFormat

TEST_OPENAI_MODEL: str = "tts-1-hd"
TEST_GOOGLE_MODEL: str = "gemini-2.5-pro-tts"
TEST_AZURE_MODEL: str = "neural"
TEST_ELEVENLABS_MODEL: str = "eleven_multilingual_v2"
TEST_TEXT_TO_CONVERT: str = "Convert this sentence into speech."
TEST_EMOTION_PROMPT: str = "Speak with a calm, warm tone."
TEST_LEGACY_CALLER_ID: str = "legacy-voice-caller"
TEST_OPENAI_AUDIO_BYTES: bytes = b"openai-audio"
TEST_OPENAI_STREAM_BYTES: bytes = b"openai-stream-audio"
TEST_GOOGLE_AUDIO_BYTES: bytes = b"google-audio"
TEST_AZURE_AUDIO_BYTES: bytes = b"azure-audio"
TEST_ELEVENLABS_AUDIO_BYTES: bytes = b"elevenlabs-audio"


class RecordingObservabilityMiddleware(AiApiObservabilityMiddleware):
    """
    Records lifecycle events so TTS-provider tests can assert ordering and metadata.

    Args:
        None

    Returns:
        Enabled middleware test double that stores before, after, and error events.
    """

    def __init__(self) -> None:
        """
        Initializes empty event collections used by TTS observability tests.

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


def _build_audio_format(
    *,
    key: str,
    file_extension: str,
    sample_rate_hz: int,
) -> AudioFormat:
    """
    Builds one minimal AudioFormat instance for TTS observability tests.

    Args:
        key: Stable audio-format identifier used by provider code.
        file_extension: File extension associated with the audio payload.
        sample_rate_hz: Sample rate attached to the audio-format definition.

    Returns:
        AudioFormat instance suitable for partially initialized provider clients.
    """
    audio_format: AudioFormat = AudioFormat(
        key=key,
        description=key,
        file_extension=file_extension,
        sample_rate_hz=sample_rate_hz,
    )
    # Normal return with the requested test audio-format definition.
    return audio_format


def _build_openai_voice_client(
    *,
    middleware: RecordingObservabilityMiddleware,
    callable_create: Callable[..., Any],
) -> AIVoiceOpenAI:
    """
    Builds a partially initialized OpenAI voice client for observability tests.

    Args:
        middleware: Recording observability middleware used to capture lifecycle events.
        callable_create: Fake OpenAI speech creation callable used by the provider code.

    Returns:
        AIVoiceOpenAI instance with fake SDK dependencies injected.
    """
    selected_voice: AIVoiceSelectionOpenAI = AIVoiceSelectionOpenAI(
        voice_id="alloy",
        voice_name="Alloy",
        locale="en-US",
    )
    default_audio_format: AudioFormat = _build_audio_format(
        key="flac_24000",
        file_extension=".flac",
        sample_rate_hz=24_000,
    )
    ai_voice_openai: AIVoiceOpenAI = AIVoiceOpenAI.model_construct(
        engine="openai",
        default_model_id=TEST_OPENAI_MODEL,
        selected_model=SimpleNamespace(name=TEST_OPENAI_MODEL),
        default_audio_format=default_audio_format,
        selected_voice=selected_voice,
    )
    object.__setattr__(ai_voice_openai, "user", TEST_LEGACY_CALLER_ID)
    object.__setattr__(
        ai_voice_openai,
        "client",
        SimpleNamespace(
            audio=SimpleNamespace(speech=SimpleNamespace(create=callable_create))
        ),
    )
    object.__setattr__(ai_voice_openai, "_observability_middleware", middleware)
    object.__setattr__(ai_voice_openai, "_play_bytes", lambda audio_bytes: None)
    # Normal return with a test-configured OpenAI voice client.
    return ai_voice_openai


def _build_google_voice_client(
    *,
    middleware: RecordingObservabilityMiddleware,
    callable_synthesize: Callable[..., Any],
) -> AIVoiceGoogle:
    """
    Builds a partially initialized Google voice client for observability tests.

    Args:
        middleware: Recording observability middleware used to capture lifecycle events.
        callable_synthesize: Fake Google synthesize callable used by the provider code.

    Returns:
        AIVoiceGoogle instance with fake SDK dependencies injected.
    """
    selected_voice: AIVoiceSelectionGoogle = AIVoiceSelectionGoogle(
        voice_id="Kore",
        voice_name="Kore",
        locale="en-US",
    )
    default_audio_format: AudioFormat = _build_audio_format(
        key="wav_linear16_48000",
        file_extension=".wav",
        sample_rate_hz=48_000,
    )
    ai_voice_google: AIVoiceGoogle = AIVoiceGoogle.model_construct(
        engine="google-gemini",
        default_model_id=TEST_GOOGLE_MODEL,
        selected_model=SimpleNamespace(name=TEST_GOOGLE_MODEL),
        default_audio_format=default_audio_format,
        selected_voice=selected_voice,
    )
    object.__setattr__(
        ai_voice_google,
        "_tts_client",
        SimpleNamespace(synthesize_speech=callable_synthesize),
    )
    object.__setattr__(
        ai_voice_google,
        "_retry_with_exponential_backoff",
        lambda operation: operation(),
    )
    object.__setattr__(
        ai_voice_google,
        "_convert_audio_bytes",
        lambda *, audio_bytes, audio_format: audio_bytes,
    )
    object.__setattr__(ai_voice_google, "_observability_middleware", middleware)
    # Normal return with a test-configured Google voice client.
    return ai_voice_google


def _build_azure_voice_client(
    *,
    middleware: RecordingObservabilityMiddleware,
    callable_synthesize_audio_bytes: Callable[..., bytes],
) -> AIVoiceAzure:
    """
    Builds a partially initialized Azure voice client for observability tests.

    Args:
        middleware: Recording observability middleware used to capture lifecycle events.
        callable_synthesize_audio_bytes: Fake synthesis helper used by the provider code.

    Returns:
        AIVoiceAzure instance with fake synthesis behavior injected.
    """
    selected_voice: AIVoiceSelectionAzure = AIVoiceSelectionAzure(
        voice_id="en-US-TestNeural",
        voice_name="Test Neural",
        locale="en-US",
    )
    default_audio_format: AudioFormat = _build_audio_format(
        key="pcm_48000",
        file_extension=".wav",
        sample_rate_hz=48_000,
    )
    ai_voice_azure: AIVoiceAzure = AIVoiceAzure.model_construct(
        engine="azure",
        default_model_id=TEST_AZURE_MODEL,
        selected_model=SimpleNamespace(name=TEST_AZURE_MODEL),
        default_audio_format=default_audio_format,
        selected_voice=selected_voice,
    )
    object.__setattr__(
        ai_voice_azure,
        "_synthesize_audio_bytes",
        callable_synthesize_audio_bytes,
    )
    object.__setattr__(ai_voice_azure, "_observability_middleware", middleware)
    # Normal return with a test-configured Azure voice client.
    return ai_voice_azure


def _build_elevenlabs_voice_client(
    *,
    middleware: RecordingObservabilityMiddleware,
    callable_stream: Callable[..., Any],
) -> AIVoiceElevenLabs:
    """
    Builds a partially initialized ElevenLabs voice client for observability tests.

    Args:
        middleware: Recording observability middleware used to capture lifecycle events.
        callable_stream: Fake ElevenLabs streaming callable used by the provider code.

    Returns:
        AIVoiceElevenLabs instance with fake SDK dependencies injected.
    """
    selected_voice: AIVoiceSelectionElevenLabs = AIVoiceSelectionElevenLabs(
        voice_id="voice-123",
        voice_name="Rachel",
        locale="en-US",
    )
    default_audio_format: AudioFormat = _build_audio_format(
        key="mp3_44100_128",
        file_extension=".mp3",
        sample_rate_hz=44_100,
    )
    ai_voice_elevenlabs: AIVoiceElevenLabs = AIVoiceElevenLabs.model_construct(
        engine="elevenlabs",
        default_model_id=TEST_ELEVENLABS_MODEL,
        selected_model=SimpleNamespace(name=TEST_ELEVENLABS_MODEL),
        default_audio_format=default_audio_format,
        selected_voice=selected_voice,
    )
    object.__setattr__(
        ai_voice_elevenlabs,
        "client",
        SimpleNamespace(text_to_speech=SimpleNamespace(stream=callable_stream)),
    )
    object.__setattr__(ai_voice_elevenlabs, "_observability_middleware", middleware)
    # Normal return with a test-configured ElevenLabs voice client.
    return ai_voice_elevenlabs


def test_openai_text_to_voice_emits_metadata_and_legacy_caller_fallback() -> None:
    """
    Verify one OpenAI TTS call emits one metadata-only sequence and resolves legacy caller fallback.

    Args:
        None

    Returns:
        None after asserting input metadata, output metadata, and caller-resolution fields.
    """
    middleware = RecordingObservabilityMiddleware()
    dict_captured_params: dict[str, Any] = {}

    def _fake_create(**params: Any) -> Any:
        """
        Capture OpenAI TTS request parameters and return deterministic audio bytes.

        Args:
            **params: Provider request parameters supplied by the client under test.

        Returns:
            Fake OpenAI audio response object for unit-test assertions.
        """
        dict_captured_params.update(params)
        # Normal return with a deterministic fake OpenAI audio response.
        return SimpleNamespace(content=TEST_OPENAI_AUDIO_BYTES)

    ai_voice_client: AIVoiceOpenAI = _build_openai_voice_client(
        middleware=middleware,
        callable_create=_fake_create,
    )
    selected_voice: AIVoiceSelectionOpenAI = ai_voice_client.selected_voice
    audio_format: AudioFormat = ai_voice_client.default_audio_format

    audio_bytes: bytes = ai_voice_client.text_to_voice(
        text_to_convert=TEST_TEXT_TO_CONVERT,
        voice=selected_voice,
        audio_format=audio_format,
        speaking_rate=1.25,
    )

    assert audio_bytes == TEST_OPENAI_AUDIO_BYTES
    assert dict_captured_params["model"] == TEST_OPENAI_MODEL
    assert dict_captured_params["voice"] == selected_voice.voice_id
    assert dict_captured_params["response_format"] == "flac"
    assert len(middleware.list_before_contexts) == 1
    assert len(middleware.list_after_events) == 1
    assert middleware.list_error_events == []

    before_context = middleware.list_before_contexts[0]
    _, result_summary = middleware.list_after_events[0]

    assert before_context.operation == "text_to_voice"
    assert before_context.originating_caller_id == TEST_LEGACY_CALLER_ID
    assert (
        before_context.originating_caller_id_source
        == ORIGINATING_CALLER_ID_SOURCE_LEGACY_SETTING
    )
    assert before_context.dict_metadata["input_text_char_count"] == len(
        TEST_TEXT_TO_CONVERT
    )
    assert before_context.dict_metadata["voice_id"] == selected_voice.voice_id
    assert before_context.dict_metadata["requested_audio_format"] == audio_format.key
    assert result_summary.dict_metadata["output_audio_byte_count"] == len(
        TEST_OPENAI_AUDIO_BYTES
    )
    assert result_summary.dict_metadata["output_audio_format"] == audio_format.key


def test_openai_stream_audio_emits_streaming_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify one OpenAI streaming TTS call emits one metadata-only sequence and captures streaming mode.

    Args:
        monkeypatch: Pytest monkeypatch fixture used to disable local playback side effects.

    Returns:
        None after asserting streaming input metadata and output byte counts.
    """
    middleware = RecordingObservabilityMiddleware()
    dict_captured_params: dict[str, Any] = {}

    def _fake_create(**params: Any) -> Any:
        """
        Capture OpenAI streaming request parameters and return deterministic byte chunks.

        Args:
            **params: Provider request parameters supplied by the client under test.

        Returns:
            Fake streaming response object exposing an `iter_bytes` iterator.
        """
        dict_captured_params.update(params)
        # Normal return with a deterministic fake streaming response object.
        return SimpleNamespace(
            iter_bytes=lambda: iter([b"openai-", b"stream-", b"audio"])
        )

    monkeypatch.setattr(ai_voice_openai_module, "is_hex_enabled", lambda: True)
    ai_voice_client: AIVoiceOpenAI = _build_openai_voice_client(
        middleware=middleware,
        callable_create=_fake_create,
    )

    audio_bytes: bytes = ai_voice_client.stream_audio(TEST_TEXT_TO_CONVERT)

    assert audio_bytes == TEST_OPENAI_STREAM_BYTES
    assert dict_captured_params["stream"] is True
    assert len(middleware.list_before_contexts) == 1
    assert len(middleware.list_after_events) == 1
    assert middleware.list_before_contexts[0].dict_metadata["streaming_mode"] is True


def test_google_emotion_prompt_emits_metadata_only_once() -> None:
    """
    Verify one Google emotion-prompt TTS call emits one metadata-only lifecycle sequence.

    Args:
        None

    Returns:
        None after asserting emotion metadata and output byte counts.
    """
    middleware = RecordingObservabilityMiddleware()

    def _fake_synthesize_speech(**_: Any) -> Any:
        """
        Return deterministic Google audio bytes for one fake synthesis request.

        Args:
            **_: Ignored keyword arguments supplied by the provider code.

        Returns:
            Fake Google synthesize response object with deterministic audio bytes.
        """
        # Normal return with deterministic fake Google audio content.
        return SimpleNamespace(audio_content=TEST_GOOGLE_AUDIO_BYTES)

    ai_voice_client: AIVoiceGoogle = _build_google_voice_client(
        middleware=middleware,
        callable_synthesize=_fake_synthesize_speech,
    )
    selected_voice: AIVoiceSelectionGoogle = ai_voice_client.selected_voice
    audio_format: AudioFormat = ai_voice_client.default_audio_format

    audio_bytes: bytes = ai_voice_client.text_to_voice_with_emotion_prompt(
        emotion_prompt=TEST_EMOTION_PROMPT,
        text_to_convert=TEST_TEXT_TO_CONVERT,
        voice=selected_voice,
        audio_format=audio_format,
    )

    assert audio_bytes == TEST_GOOGLE_AUDIO_BYTES
    assert len(middleware.list_before_contexts) == 1
    assert len(middleware.list_after_events) == 1
    assert middleware.list_error_events == []
    assert middleware.list_before_contexts[0].operation == (
        "text_to_voice_with_emotion_prompt"
    )
    assert middleware.list_before_contexts[0].dict_metadata[
        "emotion_prompt_char_count"
    ] == len(TEST_EMOTION_PROMPT)
    assert middleware.list_after_events[0][1].dict_metadata[
        "output_audio_byte_count"
    ] == len(TEST_GOOGLE_AUDIO_BYTES)


def test_azure_stream_audio_emits_one_sequence() -> None:
    """
    Verify one Azure streaming wrapper call emits one metadata-only sequence without nested duplication.

    Args:
        None

    Returns:
        None after asserting streaming metadata and output byte counts.
    """
    middleware = RecordingObservabilityMiddleware()

    def _fake_synthesize_audio_bytes(**kwargs: Any) -> bytes:
        """
        Return deterministic Azure audio bytes for one fake synthesis request.

        Args:
            **kwargs: Keyword arguments supplied by the Azure provider wrapper.

        Returns:
            Deterministic Azure audio bytes for unit-test assertions.
        """
        assert kwargs["use_ssml"] is False
        # Normal return with deterministic fake Azure audio bytes.
        return TEST_AZURE_AUDIO_BYTES

    ai_voice_client: AIVoiceAzure = _build_azure_voice_client(
        middleware=middleware,
        callable_synthesize_audio_bytes=_fake_synthesize_audio_bytes,
    )

    audio_bytes: bytes = ai_voice_client.stream_audio(TEST_TEXT_TO_CONVERT)

    assert audio_bytes == TEST_AZURE_AUDIO_BYTES
    assert len(middleware.list_before_contexts) == 1
    assert len(middleware.list_after_events) == 1
    assert middleware.list_before_contexts[0].operation == "stream_audio"
    assert middleware.list_before_contexts[0].dict_metadata["streaming_mode"] is True
    assert middleware.list_after_events[0][1].dict_metadata[
        "output_audio_byte_count"
    ] == len(TEST_AZURE_AUDIO_BYTES)


def test_elevenlabs_stream_audio_emits_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify one ElevenLabs streaming TTS call emits one metadata-only sequence and captures byte counts.

    Args:
        monkeypatch: Pytest monkeypatch fixture used to disable local playback side effects.

    Returns:
        None after asserting streaming metadata and output byte counts.
    """
    middleware = RecordingObservabilityMiddleware()
    dict_captured_params: dict[str, Any] = {}

    def _fake_stream(**params: Any) -> Any:
        """
        Capture ElevenLabs streaming request parameters and return deterministic byte chunks.

        Args:
            **params: Provider request parameters supplied by the client under test.

        Returns:
            Iterator of deterministic audio chunks for unit-test assertions.
        """
        dict_captured_params.update(params)
        # Normal return with a deterministic fake ElevenLabs streaming iterator.
        return iter([b"eleven", b"labs-", b"audio"])

    monkeypatch.setattr(ai_voice_elevenlabs_module, "is_hex_enabled", lambda: True)
    ai_voice_client: AIVoiceElevenLabs = _build_elevenlabs_voice_client(
        middleware=middleware,
        callable_stream=_fake_stream,
    )

    audio_bytes: bytes = ai_voice_client.stream_audio(TEST_TEXT_TO_CONVERT)

    assert audio_bytes == TEST_ELEVENLABS_AUDIO_BYTES
    assert dict_captured_params["model_id"] == TEST_ELEVENLABS_MODEL
    assert len(middleware.list_before_contexts) == 1
    assert len(middleware.list_after_events) == 1
    assert middleware.list_before_contexts[0].dict_metadata["streaming_mode"] is True
    assert middleware.list_after_events[0][1].dict_metadata[
        "output_audio_byte_count"
    ] == len(TEST_ELEVENLABS_AUDIO_BYTES)
