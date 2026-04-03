# ruff: noqa: E402

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("boto3")
pytest.importorskip("google.genai")
pytest.importorskip("google.cloud.texttospeech")
pytest.importorskip("google.cloud.speech_v1p1beta1")

from ai_api_unified.ai_base import (
    AIStructuredPrompt,
    AICompletionsPromptParamsBase,
)
from ai_api_unified.completions.ai_bedrock_completions import (
    AiBedrockCompletions,
)
from ai_api_unified.completions.ai_google_gemini_completions import (
    GoogleGeminiCompletions,
)
from ai_api_unified.middleware.observability import (
    get_observability_middleware,
)
from ai_api_unified.voice.ai_voice_google import (
    AIVoiceGoogle,
    AIVoiceSelectionGoogle,
)
from ai_api_unified.voice.audio_models import AudioFormat

TEST_COMPLETIONS_MODEL: str = "test-model"
TEST_TTS_MODEL: str = "gemini-2.5-pro-tts"
TEST_SECRET_VALUE: str = "secret-raw-output"
TEST_SECRET_PROMPT: str = "do not log this prompt"
TEST_SECRET_EMOTION_PROMPT: str = "secret style prompt"
TEST_PROVIDER_EXCEPTION_MESSAGE: str = "provider boom"


class IdentityPiiMiddleware:
    """
    Minimal PII middleware test double that preserves inputs and outputs unchanged.

    Args:
        None

    Returns:
        Simple middleware test double suitable for targeted logging-cleanup tests.
    """

    def process_input(self, text: str) -> str:
        """
        Returns the provided input text unchanged.

        Args:
            text: Input text supplied by the provider method under test.

        Returns:
            Original input text without any transformation.
        """
        # Normal return with unchanged input text for targeted logging tests.
        return text

    def process_output(self, text: str) -> str:
        """
        Returns the provided output text unchanged.

        Args:
            text: Output text supplied by the provider method under test.

        Returns:
            Original output text without any transformation.
        """
        # Normal return with unchanged output text for targeted logging tests.
        return text


class FakeStructuredIntegerResponse(AIStructuredPrompt):
    """
    Minimal structured response model that intentionally requires an integer field.

    Args:
        answer: Integer field used to trigger validation failures from string payloads.

    Returns:
        Structured prompt subclass suitable for validation-error cleanup tests.
    """

    answer: int

    @staticmethod
    def get_prompt() -> str | None:
        """
        Returns a static prompt string required by the structured prompt interface.

        Args:
            None

        Returns:
            Static placeholder prompt string for test-model compatibility.
        """
        # Normal return with a placeholder prompt string for test construction.
        return "Return one integer answer."


def _build_google_gemini_client() -> GoogleGeminiCompletions:
    """
    Builds a partially initialized Gemini completions client for targeted log-cleanup tests.

    Args:
        None

    Returns:
        GoogleGeminiCompletions instance with identity middleware and disabled observability.
    """
    gemini_client: GoogleGeminiCompletions = object.__new__(GoogleGeminiCompletions)
    gemini_client.completions_model = TEST_COMPLETIONS_MODEL
    gemini_client.model = TEST_COMPLETIONS_MODEL
    gemini_client.pii_middleware = IdentityPiiMiddleware()
    gemini_client._observability_middleware = get_observability_middleware()
    gemini_client._coerce_params = lambda other_params: AICompletionsPromptParamsBase()
    gemini_client._build_contents = lambda prompt, params: [prompt]
    gemini_client._build_config = (
        lambda params, system_prompt, max_output_tokens, response_schema: {}
    )
    gemini_client._did_stop_on_max_tokens = lambda response: False
    # Normal return with a test-configured Gemini completions client.
    return gemini_client


def _build_bedrock_client() -> AiBedrockCompletions:
    """
    Builds a partially initialized Bedrock completions client for targeted log-cleanup tests.

    Args:
        None

    Returns:
        AiBedrockCompletions instance with identity middleware and disabled observability.
    """
    bedrock_client: AiBedrockCompletions = object.__new__(AiBedrockCompletions)
    bedrock_client.model = TEST_COMPLETIONS_MODEL
    bedrock_client.completions_model = TEST_COMPLETIONS_MODEL
    bedrock_client.backoff_delays = [0.0]
    bedrock_client.pii_middleware = IdentityPiiMiddleware()
    bedrock_client._observability_middleware = get_observability_middleware()
    bedrock_client._sleep_with_backoff = lambda delay: None
    # Normal return with a test-configured Bedrock completions client.
    return bedrock_client


def _build_google_voice_client(*, callable_synthesize_speech: Any) -> AIVoiceGoogle:
    """
    Builds a partially initialized Google TTS client for targeted log-cleanup tests.

    Args:
        callable_synthesize_speech: Fake Google TTS SDK callable used by the provider code.

    Returns:
        AIVoiceGoogle instance with fake synthesis behavior and disabled observability.
    """
    selected_voice: AIVoiceSelectionGoogle = AIVoiceSelectionGoogle(
        voice_id="Kore",
        voice_name="Kore",
        locale="en-US",
    )
    default_audio_format: AudioFormat = AudioFormat(
        key="wav_linear16_48000",
        description="wav",
        file_extension=".wav",
        sample_rate_hz=48_000,
    )
    ai_voice_google: AIVoiceGoogle = AIVoiceGoogle.model_construct(
        engine="google-gemini",
        default_model_id=TEST_TTS_MODEL,
        selected_model=SimpleNamespace(name=TEST_TTS_MODEL),
        default_audio_format=default_audio_format,
        selected_voice=selected_voice,
    )
    object.__setattr__(
        ai_voice_google,
        "_tts_client",
        SimpleNamespace(synthesize_speech=callable_synthesize_speech),
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
    object.__setattr__(
        ai_voice_google,
        "_observability_middleware",
        get_observability_middleware(),
    )
    # Normal return with a test-configured Google TTS client.
    return ai_voice_google


def test_gemini_structured_validation_logs_omit_raw_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Verify Gemini structured-validation logs and raised errors omit raw provider response text.

    Args:
        caplog: Pytest log-capture fixture used to inspect provider log output.

    Returns:
        None after asserting the secret response text never appears in logs or the raised error.
    """
    gemini_client: GoogleGeminiCompletions = _build_google_gemini_client()
    gemini_client.client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=lambda **kwargs: SimpleNamespace(
                text=f'{{"answer":"{TEST_SECRET_VALUE}"}}'
            )
        )
    )

    with caplog.at_level(
        logging.WARNING,
        logger="ai_api_unified.completions.ai_google_gemini_completions",
    ):
        with pytest.raises(ValueError) as exc_info:
            gemini_client.strict_schema_prompt(
                prompt="Return an integer answer.",
                response_model=FakeStructuredIntegerResponse,
            )

    assert TEST_SECRET_VALUE not in caplog.text
    assert TEST_SECRET_VALUE not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert "response_char_count" in caplog.text
    assert "error_count" in caplog.text


def test_bedrock_structured_validation_logs_and_exception_omit_raw_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Verify Bedrock structured-validation logs and raised errors omit raw JSON and parsed payload content.

    Args:
        caplog: Pytest log-capture fixture used to inspect provider log output.

    Returns:
        None after asserting the secret raw payload never appears in logs or the raised error.
    """
    bedrock_client: AiBedrockCompletions = _build_bedrock_client()
    bedrock_client.client = SimpleNamespace(
        converse=lambda **kwargs: {
            "stopReason": "end_turn",
            "usage": {"inputTokens": 7, "outputTokens": 3, "totalTokens": 10},
            "output": {
                "message": {
                    "content": [
                        {"text": f'{{"answer":"{TEST_SECRET_VALUE}"}}'},
                    ]
                }
            },
        },
        exceptions=SimpleNamespace(ModelErrorException=RuntimeError),
    )
    bedrock_client._extract_json_text_from_converse_response = (
        lambda response: response["output"]["message"]["content"][0]["text"]
    )

    with caplog.at_level(
        logging.ERROR,
        logger="ai_api_unified.completions.ai_bedrock_completions",
    ):
        with pytest.raises(RuntimeError) as exc_info:
            bedrock_client.strict_schema_prompt(
                prompt="Return an integer answer.",
                response_model=FakeStructuredIntegerResponse,
            )

    assert TEST_SECRET_VALUE not in caplog.text
    assert TEST_SECRET_VALUE not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert "raw_json_char_count" in caplog.text
    assert "parsed_type" in caplog.text


def test_google_tts_error_logs_omit_prompt_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Verify Google TTS error logs omit prompt text and emotion prompts while preserving metadata-only fields.

    Args:
        caplog: Pytest log-capture fixture used to inspect provider log output.

    Returns:
        None after asserting secret prompt values never appear in logs.
    """
    ai_voice_google: AIVoiceGoogle = _build_google_voice_client(
        callable_synthesize_speech=lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError(TEST_PROVIDER_EXCEPTION_MESSAGE)
        )
    )

    with caplog.at_level(
        logging.ERROR,
        logger="ai_api_unified.voice.ai_voice_google",
    ):
        with pytest.raises(RuntimeError):
            ai_voice_google.text_to_voice_with_emotion_prompt(
                emotion_prompt=TEST_SECRET_EMOTION_PROMPT,
                text_to_convert=TEST_SECRET_PROMPT,
                voice=ai_voice_google.selected_voice,
                audio_format=ai_voice_google.default_audio_format,
            )

    assert TEST_SECRET_PROMPT not in caplog.text
    assert TEST_SECRET_EMOTION_PROMPT not in caplog.text
    assert "prompt_char_count" in caplog.text
    assert "emotion_prompt_char_count" in caplog.text
