# tests/test_ai_model_probes.py
"""
Per-provider probes against mocked SDK clients: the one metadata call each
makes, and how 404, 401/403, missing keys, and missing extras map.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import Mock, patch

import httpx
import pytest

from ai_api_unified.util import ai_model_probes as probes
from ai_api_unified.util.ai_model_checker import AiModelCheckStatus as S


@contextmanager
def _settings(**values: str) -> Iterator[None]:
    with (
        patch(
            "ai_api_unified.util.env_settings.EnvSettings.get_setting",
            side_effect=lambda key, default=None: values.get(key, default),
        ),
        patch.dict(os.environ, values, clear=False),
    ):
        yield


def _openai_status_error(status: int) -> Exception:
    openai = pytest.importorskip("openai")
    request = httpx.Request("GET", "https://api.openai.com/v1/models/x")
    return openai.APIStatusError(
        "x", response=httpx.Response(status, request=request), body=None
    )


class TestOpenAI:
    def test_available(self) -> None:
        pytest.importorskip("openai")
        client = Mock()
        with (
            _settings(OPENAI_API_KEY="k"),
            patch("openai.OpenAI", return_value=client) as ctor,
        ):
            status, detail = probes.probe_openai("gpt-4.1-mini", None)
        assert status is S.AVAILABLE
        client.models.retrieve.assert_called_once_with("gpt-4.1-mini")
        assert ctor.call_args.kwargs["max_retries"] == 0

    @pytest.mark.parametrize(
        ("code", "expected"),
        [(404, S.MISSING), (401, S.UNREACHABLE), (403, S.UNVERIFIABLE)],
    )
    def test_status_mapping(self, code: int, expected: S) -> None:
        client = Mock()
        client.models.retrieve.side_effect = _openai_status_error(code)
        with _settings(OPENAI_API_KEY="k"), patch("openai.OpenAI", return_value=client):
            assert probes.probe_openai("x", None)[0] is expected

    def test_server_error_propagates(self) -> None:
        client = Mock()
        client.models.retrieve.side_effect = _openai_status_error(500)
        with _settings(OPENAI_API_KEY="k"), patch("openai.OpenAI", return_value=client):
            with pytest.raises(Exception):
                probes.probe_openai("x", None)

    def test_no_key_and_blank_model(self) -> None:
        pytest.importorskip("openai")
        with _settings():
            assert probes.probe_openai("x", None) == (
                S.UNREACHABLE,
                "OPENAI_API_KEY not set",
            )
        assert probes.probe_openai("", None)[0] is S.UNVERIFIABLE

    def test_base_url_override_reaches_the_client(self) -> None:
        pytest.importorskip("openai")
        with (
            _settings(OPENAI_API_KEY="k"),
            patch("openai.OpenAI", return_value=Mock()) as ctor,
        ):
            probes.probe_openai("x", "https://proxy.test/v1")
        assert ctor.call_args.kwargs["base_url"] == "https://proxy.test/v1"


class TestDecisions:
    def test_unsupported_model_is_missing_without_a_call(self) -> None:
        with patch("openai.OpenAI") as ctor:
            status, detail = probes.probe_openai_decisions("gpt-4.1-mini", None)
        assert status is S.MISSING and "Decisions" in detail
        ctor.assert_not_called()

    def test_supported_model_is_retrieved(self) -> None:
        client = Mock()
        with _settings(OPENAI_API_KEY="k"), patch("openai.OpenAI", return_value=client):
            assert probes.probe_openai_decisions("", None)[0] is S.AVAILABLE
        client.models.retrieve.assert_called_once_with("gpt-6-luna")


class TestOpenAICompatible:
    def _client(self, ids: list[str] | Exception) -> Mock:
        client = Mock()
        if isinstance(ids, Exception):
            client.models.list.side_effect = ids
        else:
            client.models.list.return_value = [Mock(id=i) for i in ids]
        return client

    def test_listed_and_unlisted(self) -> None:
        pytest.importorskip("openai")
        with (
            _settings(OPENAI_COMPATIBLE_BASE_URL="http://localhost:11434/v1"),
            patch("openai.OpenAI", return_value=self._client(["a", "b"])),
        ):
            assert probes.probe_openai_compatible("a", None)[0] is S.AVAILABLE
            assert probes.probe_openai_compatible("zzz", None)[0] is S.MISSING

    def test_no_models_endpoint_or_empty_list_is_unverifiable(self) -> None:
        pytest.importorskip("openai")
        with (
            _settings(OPENAI_COMPATIBLE_BASE_URL="http://x"),
            patch("openai.OpenAI", return_value=self._client(RuntimeError("404"))),
        ):
            assert probes.probe_openai_compatible("a", None)[0] is S.UNVERIFIABLE
        with (
            _settings(OPENAI_COMPATIBLE_BASE_URL="http://x"),
            patch("openai.OpenAI", return_value=self._client([])),
        ):
            assert probes.probe_openai_compatible("a", None)[0] is S.UNVERIFIABLE

    def test_no_base_url(self) -> None:
        pytest.importorskip("openai")
        with _settings():
            assert probes.probe_openai_compatible("a", None)[0] is S.UNREACHABLE

    def test_rejected_key_is_unreachable(self) -> None:
        client = Mock()
        client.models.list.side_effect = _openai_status_error(401)
        with (
            _settings(OPENAI_COMPATIBLE_BASE_URL="http://x"),
            patch("openai.OpenAI", return_value=client),
        ):
            assert probes.probe_openai_compatible("a", None)[0] is S.UNREACHABLE


class TestAnthropic:
    def _error(self, status: int) -> Exception:
        anthropic = pytest.importorskip("anthropic")
        request = httpx.Request("GET", "https://api.anthropic.com/v1/models/x")
        return anthropic.APIStatusError(
            "x", response=httpx.Response(status, request=request), body=None
        )

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (None, S.AVAILABLE),
            (404, S.MISSING),
            (401, S.UNREACHABLE),
            (403, S.UNVERIFIABLE),
        ],
    )
    def test_mapping(self, code: int | None, expected: S) -> None:
        client = Mock()
        if code is not None:
            client.models.retrieve.side_effect = self._error(code)
        with (
            _settings(ANTHROPIC_API_KEY="k"),
            patch("anthropic.Anthropic", return_value=client),
        ):
            assert probes.probe_anthropic("claude-x", None)[0] is expected
        client.models.retrieve.assert_called_once_with("claude-x")

    def test_no_key(self) -> None:
        pytest.importorskip("anthropic")
        with _settings():
            assert probes.probe_anthropic("x", None) == (
                S.UNREACHABLE,
                "ANTHROPIC_API_KEY not set",
            )


class TestGemini:
    def _error(self, code: int, str_reason: str = "NOT_FOUND") -> Exception:
        gerr = pytest.importorskip("google.genai.errors")
        return gerr.ClientError(
            code, {"error": {"message": f"x {str_reason}", "status": str_reason}}
        )

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (None, S.AVAILABLE),
            (404, S.MISSING),
            (401, S.UNREACHABLE),
            (403, S.UNVERIFIABLE),
        ],
    )
    def test_mapping_probes_the_raw_id(self, code: int | None, expected: S) -> None:
        genai = pytest.importorskip("google.genai")
        client = Mock()
        if code is not None:
            client.models.get.side_effect = self._error(code)
        with (
            _settings(GOOGLE_GEMINI_API_KEY="k"),
            patch.object(genai, "Client", return_value=client),
        ):
            assert probes.probe_google_gemini("gemini-nope", None)[0] is expected
        client.models.get.assert_called_once_with(model="gemini-nope")

    def test_invalid_key_400_is_unreachable(self) -> None:
        genai = pytest.importorskip("google.genai")
        client = Mock()
        client.models.get.side_effect = self._error(400, "API_KEY_INVALID")
        with (
            _settings(GOOGLE_GEMINI_API_KEY="k"),
            patch.object(genai, "Client", return_value=client),
        ):
            assert probes.probe_google_gemini("x", None)[0] is S.UNREACHABLE

    def test_vertex_auth_is_unverifiable_and_no_key_unreachable(self) -> None:
        pytest.importorskip("google.genai")
        with _settings(GOOGLE_AUTH_METHOD="vertex"):
            assert probes.probe_google_gemini("x", None)[0] is S.UNVERIFIABLE
        with _settings():
            assert probes.probe_google_gemini("x", None) == (
                S.UNREACHABLE,
                "GOOGLE_GEMINI_API_KEY not set",
            )


class TestBedrock:
    def _client_error(self, code: str) -> Exception:
        botocore = pytest.importorskip("botocore.exceptions")
        return botocore.ClientError(
            {
                "Error": {"Code": code, "Message": "x"},
                "ResponseMetadata": {"HTTPStatusCode": 400},
            },
            "GetFoundationModel",
        )

    def test_foundation_model_and_profile_routing(self) -> None:
        pytest.importorskip("boto3")
        client = Mock()
        with (
            _settings(AWS_REGION="eu-west-1"),
            patch("boto3.client", return_value=client) as ctor,
        ):
            assert probes.probe_bedrock("amazon.nova-lite-v1:0", None)[0] is S.AVAILABLE
            assert (
                probes.probe_bedrock("us.anthropic.claude-sonnet-4-5", None)[0]
                is S.AVAILABLE
            )
        client.get_foundation_model.assert_called_once_with(
            modelIdentifier="amazon.nova-lite-v1:0"
        )
        client.get_inference_profile.assert_called_once_with(
            inferenceProfileIdentifier="us.anthropic.claude-sonnet-4-5"
        )
        assert (
            ctor.call_args.args[0] == "bedrock"
            and ctor.call_args.kwargs["region_name"] == "eu-west-1"
        )

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("ResourceNotFoundException", S.MISSING),
            ("ValidationException", S.MISSING),
            ("AccessDeniedException", S.UNVERIFIABLE),
            ("ExpiredTokenException", S.UNREACHABLE),
            ("UnrecognizedClientException", S.UNREACHABLE),
        ],
    )
    def test_client_error_mapping(self, code: str, expected: S) -> None:
        client = Mock()
        client.get_foundation_model.side_effect = self._client_error(code)
        with _settings(), patch("boto3.client", return_value=client):
            assert probes.probe_bedrock("x", None)[0] is expected

    def test_no_credentials(self) -> None:
        botocore = pytest.importorskip("botocore.exceptions")
        client = Mock()
        client.get_foundation_model.side_effect = botocore.NoCredentialsError()
        with _settings(), patch("boto3.client", return_value=client):
            assert probes.probe_bedrock("x", None) == (
                S.UNREACHABLE,
                "no AWS credentials",
            )


def test_default_registration_covers_every_completions_engine() -> None:
    from ai_api_unified.ai_provider_registry import DICT_TUPLE_AI_PROVIDER_REGISTRY
    from ai_api_unified.util.ai_model_checker import AiModelChecker

    tokens = {
        engine
        for (cap, engine) in DICT_TUPLE_AI_PROVIDER_REGISTRY
        if cap == "completions"
    }
    missing = sorted(t for t in tokens if AiModelChecker.probe_for(t) is None)
    assert missing == [], f"engines without a model probe: {missing}"
    assert AiModelChecker.probe_for("openai-decisions") is not None
