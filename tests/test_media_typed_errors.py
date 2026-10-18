# tests/test_media_typed_errors.py
"""
Embeddings, images, videos, and voice raise AiProviderRequestError for a
provider failure, with the status and fallback reason completions engines
report, instead of the SDK's own exception or a RuntimeError around it.
"""

from __future__ import annotations

from typing import Any

import httpx
import openai
import pytest
from botocore.exceptions import ClientError, ReadTimeoutError

from ai_api_unified import AiFallbackReason as R, AiProviderRequestError
from ai_api_unified.ai_provider_error_mapping import map_provider_exception
from ai_api_unified.middleware.request_cost import request_cost_scope
from tests.test_observability_embeddings_phase_d import (
    TEST_OPENAI_TEXT,
    RecordingObservabilityMiddleware,
    _build_openai_embeddings_client,
)

ENGINE: str = "test-engine"


def _openai_status(status: int, code: str | None = None) -> openai.APIStatusError:
    response = httpx.Response(
        status, request=httpx.Request("POST", "https://api.openai.test")
    )
    return openai.APIStatusError(
        "failed", response=response, body={"code": code} if code else None
    )


def _client_error(code: str, status: int) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": "nope"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "InvokeModel",
    )


def _wrapped(inner: Exception) -> RuntimeError:
    """A RuntimeError raised from an SDK error, as the media clients do."""
    try:
        try:
            raise inner
        except Exception as exception:
            raise RuntimeError("provider call failed") from exception
    except RuntimeError as wrapper:
        return wrapper


class TestMapper:
    def test_openai_status_and_code(self) -> None:
        typed = map_provider_exception(_openai_status(503), provider_engine=ENGINE)
        assert typed is not None and typed.status_code == 503
        assert typed.fallback_reason is R.UNAVAILABLE
        assert typed.provider_engine == ENGINE
        quota = map_provider_exception(
            _openai_status(429, "insufficient_quota"), provider_engine=ENGINE
        )
        assert quota.fallback_reason is R.QUOTA_EXHAUSTED

    def test_openai_timeout(self) -> None:
        typed = map_provider_exception(
            openai.APITimeoutError(request=httpx.Request("POST", "https://x")),
            provider_engine=ENGINE,
        )
        assert typed is not None and typed.status_code is None
        assert typed.fallback_reason is R.TIMEOUT

    def test_botocore_codes(self) -> None:
        typed = map_provider_exception(
            _client_error("ThrottlingException", 429), provider_engine=ENGINE
        )
        assert typed.fallback_reason is R.RATE_LIMITED and typed.status_code == 429
        quota = map_provider_exception(
            _client_error("ServiceQuotaExceededException", 400),
            provider_engine=ENGINE,
        )
        # Bedrock quotas are per model, so the account stays usable.
        assert quota.fallback_reason is R.QUOTA_EXHAUSTED and not quota.account_wide
        timeout = map_provider_exception(
            ReadTimeoutError(endpoint_url="https://bedrock.test"),
            provider_engine=ENGINE,
        )
        assert timeout.fallback_reason is R.TIMEOUT

    def test_google_api_core(self) -> None:
        from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

        assert (
            map_provider_exception(
                ServiceUnavailable("down"), provider_engine=ENGINE
            ).fallback_reason
            is R.UNAVAILABLE
        )
        limited = map_provider_exception(
            ResourceExhausted("slow down"), provider_engine=ENGINE
        )
        assert limited.status_code == 429 and limited.fallback_reason is R.RATE_LIMITED

    def test_httpx(self) -> None:
        request = httpx.Request("GET", "https://x")
        status = httpx.HTTPStatusError(
            "bad", request=request, response=httpx.Response(502, request=request)
        )
        assert (
            map_provider_exception(status, provider_engine=ENGINE).fallback_reason
            is R.UNAVAILABLE
        )
        assert (
            map_provider_exception(
                httpx.ConnectError("refused", request=request), provider_engine=ENGINE
            ).fallback_reason
            is R.UNAVAILABLE
        )

    def test_sdk_error_inside_a_runtime_error_is_found(self) -> None:
        typed = map_provider_exception(
            _wrapped(_openai_status(503)), provider_engine=ENGINE
        )
        assert typed is not None and typed.fallback_reason is R.UNAVAILABLE

    def test_typed_error_deeper_in_the_chain_is_copied(self) -> None:
        inner = AiProviderRequestError(
            "x", status_code=429, fallback_reason=R.RATE_LIMITED
        )
        wrapper = _wrapped(inner)
        typed = map_provider_exception(wrapper, provider_engine=ENGINE)
        assert typed is not inner and typed.fallback_reason is R.RATE_LIMITED
        assert map_provider_exception(inner, provider_engine=ENGINE) is inner

    @pytest.mark.parametrize(
        "exception", [ValueError("bad input"), RuntimeError("no vectors"), KeyError()]
    )
    def test_library_errors_are_not_mapped(self, exception: Exception) -> None:
        assert map_provider_exception(exception, provider_engine=ENGINE) is None


class TestEmbeddingsClient:
    def _client(self, error: Exception) -> Any:
        def _create(**_: Any) -> Any:
            raise error

        return _build_openai_embeddings_client(
            middleware=RecordingObservabilityMiddleware(), callable_create=_create
        )

    def test_provider_failure_raises_typed_error_with_cause(self) -> None:
        client = self._client(_openai_status(503))
        with pytest.raises(AiProviderRequestError) as raised:
            client.generate_embeddings(TEST_OPENAI_TEXT)
        assert raised.value.fallback_reason is R.UNAVAILABLE
        assert raised.value.status_code == 503
        assert raised.value.__cause__ is not None

    def test_line_item_records_the_reason(self) -> None:
        client = self._client(_openai_status(429, "rate_limit_exceeded"))
        with request_cost_scope("embed") as scope:
            with pytest.raises(AiProviderRequestError):
                client.generate_embeddings(TEST_OPENAI_TEXT)
        (item,) = scope.report.calls
        assert item.outcome == "error" and item.fallback_reason == "rate_limited"

    def test_library_validation_error_is_unchanged(self) -> None:
        client = self._client(ValueError("bad input"))
        with pytest.raises(Exception) as raised:
            client.generate_embeddings(TEST_OPENAI_TEXT)
        assert not isinstance(raised.value, AiProviderRequestError)
