# ruff: noqa: E402
# test_fallback_reason.py
"""
Tests for AiFallbackReason, the "another model might serve this" signal on
AiProviderRequestError.

Covers the status-only baseline, the per-engine mapping from provider error
codes (Anthropic error types, OpenAI error codes, Bedrock Converse codes,
Gemini messages), the timeout-versus-connection distinction, and the
short-circuit that stops in-engine retries on a quota or unknown-model error.
All against mocked SDK clients.
"""

import os
from typing import Any
from unittest.mock import Mock, patch

import httpx
import pytest

pytest.importorskip("anthropic")
pytest.importorskip("openai")
pytest.importorskip("boto3")

import anthropic
import openai
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError

from ai_api_unified import AiFallbackReason, AiProviderRequestError
from ai_api_unified.ai_base import AIStructuredPrompt
from ai_api_unified.ai_google_base import classify_gemini_fallback_reason
from ai_api_unified.ai_provider_exceptions import (
    FROZENSET_TRANSIENT_FALLBACK_REASONS,
    classify_fallback_reason_by_status,
)
from ai_api_unified.completions.ai_anthropic_completions import (
    AiAnthropicCompletions,
)
from ai_api_unified.completions.ai_bedrock_completions import AiBedrockCompletions
from ai_api_unified.completions.ai_openai_completions import AiOpenAICompletions

R = AiFallbackReason


# ── Builders ────────────────────────────────────────────────────────────────


def _anthropic() -> AiAnthropicCompletions:
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        client = AiAnthropicCompletions(model="claude-opus-4-8")
    client.client = Mock()
    return client


def _openai() -> AiOpenAICompletions:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        client = AiOpenAICompletions(model="gpt-5.1")
    client.client = Mock()
    return client


def _bedrock() -> AiBedrockCompletions:
    with patch("ai_api_unified.ai_bedrock_base.boto3"):
        client = AiBedrockCompletions(model="us.anthropic.claude-opus-5")
    client.client = Mock()
    client.backoff_delays = [0.0, 0.0, 0.0]
    client._sleep_with_backoff = Mock()
    return client


def _anthropic_error(status: int, error_type: str, message: str = "x") -> Exception:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIStatusError(
        message,
        response=httpx.Response(status, request=request),
        body={"error": {"type": error_type, "message": message}},
    )


def _openai_error(status: int, code: str | None) -> Exception:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    error = openai.APIStatusError(
        "x",
        response=httpx.Response(status, request=request),
        body={"error": {"code": code, "message": "x"}},
    )
    error.code = code
    return error


def _bedrock_error(code: str, status: int, message: str = "x") -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "Converse",
    )


# ── Baseline ────────────────────────────────────────────────────────────────


class TestBaseline:
    @pytest.mark.parametrize(
        ("status", "reason"),
        [
            (None, None),
            (429, R.RATE_LIMITED),
            (404, R.MODEL_UNAVAILABLE),
            (500, R.UNAVAILABLE),
            (503, R.UNAVAILABLE),
            (529, R.UNAVAILABLE),
            (400, None),
            (401, None),
            (403, None),
        ],
    )
    def test_status_only_classification(
        self, status: int | None, reason: AiFallbackReason | None
    ) -> None:
        assert classify_fallback_reason_by_status(status) is reason

    def test_error_derives_reason_from_status_by_default(self) -> None:
        assert AiProviderRequestError("x", status_code=503).fallback_reason is (
            R.UNAVAILABLE
        )

    def test_explicit_none_marks_no_fallback(self) -> None:
        error = AiProviderRequestError("x", status_code=429, fallback_reason=None)
        assert error.fallback_reason is None
        assert error.is_transient is False

    def test_transient_set(self) -> None:
        assert FROZENSET_TRANSIENT_FALLBACK_REASONS == {R.UNAVAILABLE, R.RATE_LIMITED}
        assert (
            AiProviderRequestError(
                "x", status_code=429, fallback_reason=R.QUOTA_EXHAUSTED
            ).is_transient
            is False
        )


# ── Anthropic ───────────────────────────────────────────────────────────────


class TestAnthropic:
    @pytest.mark.parametrize(
        ("status", "error_type", "message", "reason"),
        [
            (529, "overloaded_error", "Overloaded", R.UNAVAILABLE),
            (429, "rate_limit_error", "slow down", R.RATE_LIMITED),
            (404, "not_found_error", "model: nope", R.MODEL_UNAVAILABLE),
            (
                400,
                "invalid_request_error",
                "Your credit balance is too low to access the Anthropic API.",
                R.QUOTA_EXHAUSTED,
            ),
            (400, "invalid_request_error", "messages: bad shape", None),
            (401, "authentication_error", "invalid x-api-key", None),
            (
                403,
                "permission_error",
                "Your organization has been disabled. Contact billing support.",
                None,
            ),
        ],
    )
    def test_status_error_mapping(
        self, status: int, error_type: str, message: str, reason: Any
    ) -> None:
        client = _anthropic()
        client.client.messages.create.side_effect = _anthropic_error(
            status, error_type, message
        )
        with pytest.raises(AiProviderRequestError) as exc_info:
            client.send_prompt("hi")
        assert exc_info.value.fallback_reason is reason
        assert exc_info.value.provider_engine == "claude"

    def test_timeout_is_not_a_fallback_trigger(self) -> None:
        client = _anthropic()
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        client.client.messages.create.side_effect = anthropic.APITimeoutError(
            request=request
        )
        with pytest.raises(AiProviderRequestError) as exc_info:
            client.send_prompt("hi")
        assert exc_info.value.fallback_reason is None

    def test_connection_error_is_unavailable(self) -> None:
        client = _anthropic()
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        client.client.messages.create.side_effect = anthropic.APIConnectionError(
            request=request
        )
        with pytest.raises(AiProviderRequestError) as exc_info:
            client.send_prompt("hi")
        assert exc_info.value.fallback_reason is R.UNAVAILABLE


# ── OpenAI ──────────────────────────────────────────────────────────────────


class _Answer(AIStructuredPrompt):
    answer: str

    @staticmethod
    def get_prompt() -> str:
        return "hi"


class TestOpenAI:
    @pytest.mark.parametrize(
        ("status", "code", "reason"),
        [
            (429, "insufficient_quota", R.QUOTA_EXHAUSTED),
            (429, "rate_limit_exceeded", R.RATE_LIMITED),
            (429, None, R.RATE_LIMITED),
            (404, "model_not_found", R.MODEL_UNAVAILABLE),
            (503, None, R.UNAVAILABLE),
            (400, "invalid_request_error", None),
        ],
    )
    def test_status_error_mapping(
        self, status: int, code: str | None, reason: Any
    ) -> None:
        client = _openai()
        client.client.chat.completions.create.side_effect = _openai_error(status, code)
        with pytest.raises(AiProviderRequestError) as exc_info:
            client.send_prompt("hi")
        assert exc_info.value.fallback_reason is reason

    def test_strict_schema_stops_retrying_on_quota(self) -> None:
        client = _openai()
        client.client.chat.completions.create.side_effect = _openai_error(
            429, "insufficient_quota"
        )
        with patch("ai_api_unified.completions.ai_openai_completions.time.sleep") as s:
            with pytest.raises(AiProviderRequestError) as exc_info:
                client.strict_schema_prompt("hi", _Answer)
        assert exc_info.value.fallback_reason is R.QUOTA_EXHAUSTED
        assert client.client.chat.completions.create.call_count == 1
        s.assert_not_called()

    def test_strict_schema_still_retries_a_rate_limit(self) -> None:
        client = _openai()
        client.client.chat.completions.create.side_effect = _openai_error(
            429, "rate_limit_exceeded"
        )
        with patch("ai_api_unified.completions.ai_openai_completions.time.sleep"):
            with pytest.raises(AiProviderRequestError) as exc_info:
                client.strict_schema_prompt("hi", _Answer)
        assert exc_info.value.fallback_reason is R.RATE_LIMITED
        assert client.client.chat.completions.create.call_count == 3


# ── Bedrock ─────────────────────────────────────────────────────────────────


class TestBedrock:
    @pytest.mark.parametrize(
        ("code", "status", "message", "reason"),
        [
            ("ThrottlingException", 429, "slow down", R.RATE_LIMITED),
            ("ServiceUnavailableException", 503, "x", R.UNAVAILABLE),
            ("ModelNotReadyException", 429, "x", R.UNAVAILABLE),
            ("ServiceQuotaExceededException", 400, "x", R.QUOTA_EXHAUSTED),
            ("ResourceNotFoundException", 404, "x", R.MODEL_UNAVAILABLE),
            (
                "ValidationException",
                400,
                "The provided model identifier is invalid.",
                R.MODEL_UNAVAILABLE,
            ),
            ("ValidationException", 400, "messages: bad", None),
            ("AccessDeniedException", 403, "denied", None),
            ("TooManyRequestsException", 429, "slow down", R.RATE_LIMITED),
            ("ServiceUnavailable", 503, "x", R.UNAVAILABLE),
        ],
    )
    def test_client_error_mapping(
        self, code: str, status: int, message: str, reason: Any
    ) -> None:
        client = _bedrock()
        client.client.converse.side_effect = _bedrock_error(code, status, message)
        with pytest.raises(AiProviderRequestError) as exc_info:
            client.send_conversation(
                "sys", [{"role": "user", "content": [{"text": "hi"}]}]
            )
        assert exc_info.value.fallback_reason is reason

    def test_read_timeout_is_not_a_fallback_trigger(self) -> None:
        client = _bedrock()
        client.client.converse.side_effect = ReadTimeoutError(endpoint_url="x")
        with pytest.raises(AiProviderRequestError) as exc_info:
            client.send_conversation(
                "sys", [{"role": "user", "content": [{"text": "hi"}]}]
            )
        assert exc_info.value.fallback_reason is None

    def test_endpoint_connection_error_is_unavailable(self) -> None:
        client = _bedrock()
        client.client.converse.side_effect = EndpointConnectionError(endpoint_url="x")
        with pytest.raises(AiProviderRequestError) as exc_info:
            client.send_conversation(
                "sys", [{"role": "user", "content": [{"text": "hi"}]}]
            )
        assert exc_info.value.fallback_reason is R.UNAVAILABLE

    def test_send_prompt_stops_retrying_on_quota(self) -> None:
        client = _bedrock()
        client.client.converse.side_effect = _bedrock_error(
            "ServiceQuotaExceededException", 400
        )
        with pytest.raises(AiProviderRequestError) as exc_info:
            client.send_prompt("hi")
        assert exc_info.value.fallback_reason is R.QUOTA_EXHAUSTED
        assert client.client.converse.call_count == 1
        client._sleep_with_backoff.assert_not_called()

    def test_strict_schema_exits_once_on_model_error(self) -> None:
        client = _bedrock()
        client.client.converse.side_effect = _bedrock_error("ModelErrorException", 424)
        with pytest.raises(AiProviderRequestError) as exc_info:
            client.strict_schema_prompt("hi", _Answer)
        assert exc_info.value.fallback_reason is R.UNAVAILABLE
        assert client.client.converse.call_count == 1

    def test_strict_schema_reports_connection_errors_as_typed(self) -> None:
        client = _bedrock()
        client.client.converse.side_effect = EndpointConnectionError(endpoint_url="x")
        with pytest.raises(AiProviderRequestError) as exc_info:
            client.strict_schema_prompt("hi", _Answer)
        assert exc_info.value.fallback_reason is R.UNAVAILABLE

    def test_send_prompt_still_retries_throttling(self) -> None:
        client = _bedrock()
        client.client.converse.side_effect = _bedrock_error("ThrottlingException", 429)
        with pytest.raises(AiProviderRequestError) as exc_info:
            client.send_prompt("hi")
        assert exc_info.value.fallback_reason is R.RATE_LIMITED
        assert client.client.converse.call_count == 3


# ── Gemini ──────────────────────────────────────────────────────────────────


# Google's live RESOURCE_EXHAUSTED text: both carry the billing sentence, and
# only the quota id separates a per-minute limit from a daily one.
GEMINI_PER_MINUTE_MESSAGE: str = (
    "429 RESOURCE_EXHAUSTED. You exceeded your current quota, please check your "
    "plan and billing details. quota_metric: generativelanguage.googleapis.com/"
    "generate_content_requests, quota_id: GenerateRequestsPerMinutePerProjectPerModel "
    "retry_delay { seconds: 30 }"
)
GEMINI_PER_DAY_MESSAGE: str = GEMINI_PER_MINUTE_MESSAGE.replace(
    "PerMinutePerProjectPerModel", "PerDayPerProjectPerModel"
)


class TestGemini:
    @pytest.mark.parametrize(
        ("status", "message", "reason"),
        [
            (429, GEMINI_PER_MINUTE_MESSAGE, R.RATE_LIMITED),
            (429, GEMINI_PER_DAY_MESSAGE, R.QUOTA_EXHAUSTED),
            (
                429,
                "Quota exceeded: generate_content_requests_per_day",
                R.QUOTA_EXHAUSTED,
            ),
            (503, "The model is overloaded.", R.UNAVAILABLE),
            (404, "not found", R.MODEL_UNAVAILABLE),
            (
                400,
                "Gemini 1.0 Pro is not supported. Use a newer model.",
                R.MODEL_UNAVAILABLE,
            ),
            (400, "Invalid argument: contents", None),
        ],
    )
    def test_message_classification(
        self, status: int, message: str, reason: Any
    ) -> None:
        assert classify_gemini_fallback_reason(status, message) is reason

    def test_backoff_loop_stops_on_hard_quota(self) -> None:
        genai_errors = pytest.importorskip("google.genai.errors")
        from ai_api_unified.completions.ai_google_gemini_completions import (
            GoogleGeminiCompletions,
        )

        with patch.object(
            GoogleGeminiCompletions,
            "_initialize_client",
            lambda self: setattr(self, "client", Mock()),
        ):
            client = GoogleGeminiCompletions(model="gemini-2.5-flash")
        error = genai_errors.ClientError(
            429,
            {
                "error": {
                    "message": GEMINI_PER_DAY_MESSAGE,
                    "status": "RESOURCE_EXHAUSTED",
                }
            },
        )
        operation: Mock = Mock(side_effect=error)
        with patch("ai_api_unified.ai_google_base.time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError) as exc_info:
                client._retry_with_exponential_backoff(operation, max_retries=5)
        # Wrapped like the non-retryable branch, with the SDK error as cause.
        assert exc_info.value.__cause__ is error
        assert operation.call_count == 1
        mock_sleep.assert_not_called()

    def test_send_prompt_reports_the_typed_error(self) -> None:
        genai_errors = pytest.importorskip("google.genai.errors")
        from ai_api_unified.completions.ai_google_gemini_completions import (
            GoogleGeminiCompletions,
        )

        with patch.object(
            GoogleGeminiCompletions,
            "_initialize_client",
            lambda self: setattr(self, "client", Mock()),
        ):
            client = GoogleGeminiCompletions(model="gemini-2.5-flash")
        client.client.models.generate_content.side_effect = genai_errors.ClientError(
            429,
            {
                "error": {
                    "message": GEMINI_PER_DAY_MESSAGE,
                    "status": "RESOURCE_EXHAUSTED",
                }
            },
        )
        with patch("ai_api_unified.ai_google_base.time.sleep"):
            with pytest.raises(AiProviderRequestError) as exc_info:
                client.send_prompt("hi")
        assert exc_info.value.fallback_reason is R.QUOTA_EXHAUSTED

    def test_backoff_loop_still_retries_a_rate_limit(self) -> None:
        genai_errors = pytest.importorskip("google.genai.errors")
        from ai_api_unified.completions.ai_google_gemini_completions import (
            GoogleGeminiCompletions,
        )

        with patch.object(
            GoogleGeminiCompletions,
            "_initialize_client",
            lambda self: setattr(self, "client", Mock()),
        ):
            client = GoogleGeminiCompletions(model="gemini-2.5-flash")
        error = genai_errors.ClientError(
            429, {"error": {"message": GEMINI_PER_MINUTE_MESSAGE}}
        )
        operation: Mock = Mock(side_effect=error)
        with patch("ai_api_unified.ai_google_base.time.sleep"):
            with pytest.raises(RuntimeError):
                client._retry_with_exponential_backoff(operation, max_retries=2)
        assert operation.call_count == 3
