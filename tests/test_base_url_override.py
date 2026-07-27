# test_base_url_override.py
"""
Tests for per-engine API base-URL overrides (anthropic, openai, gemini).

The override redirects provider traffic — including credentials — so it is
validated as https (loopback exempt) and always passed to the SDK explicitly,
which keeps the SDKs' own base-URL environment variables from taking effect
unvalidated.
"""

import os
from unittest.mock import Mock, patch

import pytest

from ai_api_unified.ai_base import (
    resolve_base_url_override,
    validate_base_url_override,
)
from ai_api_unified.ai_provider_exceptions import (
    AiProviderCapabilityUnsupportedError,
    AiProviderConfigurationError,
)


class _StubEnv:
    def __init__(self, **values: str) -> None:
        self._values = values

    def get_setting(self, key: str, default=None):
        return self._values.get(key, default)


class TestValidation:
    def test_https_accepted(self):
        assert (
            validate_base_url_override("https://gw.internal/v1", str_env_key="X")
            == "https://gw.internal/v1"
        )

    def test_whitespace_trimmed(self):
        assert (
            validate_base_url_override("  https://gw.internal/v1  ", str_env_key="X")
            == "https://gw.internal/v1"
        )

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:8000/v1",
            "http://127.0.0.1:11434/v1",
            "http://[::1]:8080/v1",
        ],
    )
    def test_loopback_http_accepted(self, url):
        assert validate_base_url_override(url, str_env_key="X") == url

    def test_plaintext_remote_rejected(self):
        with pytest.raises(AiProviderConfigurationError, match="https"):
            validate_base_url_override("http://gw.example.com/v1", str_env_key="X")

    @pytest.mark.parametrize("url", ["not-a-url", "ftp://host/v1", "/relative/path"])
    def test_malformed_rejected(self, url):
        with pytest.raises(AiProviderConfigurationError):
            validate_base_url_override(url, str_env_key="X")

    def test_error_names_the_setting(self):
        with pytest.raises(AiProviderConfigurationError, match="MY_OVERRIDE"):
            validate_base_url_override("http://evil.example", str_env_key="MY_OVERRIDE")


class TestErrorRedaction:
    def test_credentials_in_url_are_not_echoed(self):
        # Gateway URLs can carry secrets in userinfo or query strings, and
        # config errors land in logs and tracebacks.
        with pytest.raises(AiProviderConfigurationError) as exc_info:
            validate_base_url_override(
                "http://user:sk-secret-token@gw.example.com/v1?api-key=sk-another",
                str_env_key="X",
            )
        str_message = str(exc_info.value)
        assert "sk-secret-token" not in str_message
        assert "sk-another" not in str_message
        assert "gw.example.com" in str_message


class TestResolution:
    def test_explicit_argument_wins_over_env(self):
        env = _StubEnv(X="https://from-env/v1")
        assert (
            resolve_base_url_override(
                env, str_env_key="X", str_explicit="https://from-arg/v1"
            )
            == "https://from-arg/v1"
        )

    def test_env_used_when_no_argument(self):
        env = _StubEnv(X="https://from-env/v1")
        assert resolve_base_url_override(env, str_env_key="X") == "https://from-env/v1"

    def test_none_when_unconfigured(self):
        assert resolve_base_url_override(_StubEnv(), str_env_key="X") is None

    def test_blank_env_treated_as_unset(self):
        assert resolve_base_url_override(_StubEnv(X="   "), str_env_key="X") is None

    def test_env_value_is_validated(self):
        with pytest.raises(AiProviderConfigurationError):
            resolve_base_url_override(
                _StubEnv(X="http://evil.example"), str_env_key="X"
            )


# ── Anthropic ───────────────────────────────────────────────────────────────

anthropic = pytest.importorskip("anthropic")

from ai_api_unified.completions.ai_anthropic_completions import (  # noqa: E402
    AiAnthropicCompletions,
)


class TestAnthropicOverride:
    def test_default_base_url_passed_explicitly(self):
        # Explicit passing is what keeps the SDK's native ANTHROPIC_BASE_URL
        # from taking effect without validation.
        with patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "k", "ANTHROPIC_BASE_URL_OVERRIDE": ""},
        ):
            with patch("ai_api_unified.ai_anthropic_base.Anthropic") as mock_cls:
                AiAnthropicCompletions(model="claude-opus-4-8")
        assert mock_cls.call_args.kwargs["base_url"] == "https://api.anthropic.com"

    def test_env_override_applied(self):
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "k",
                "ANTHROPIC_BASE_URL_OVERRIDE": "https://gw.internal/anthropic",
            },
        ):
            with patch("ai_api_unified.ai_anthropic_base.Anthropic") as mock_cls:
                client = AiAnthropicCompletions(model="claude-opus-4-8")
        assert mock_cls.call_args.kwargs["base_url"] == "https://gw.internal/anthropic"
        assert client.base_url == "https://gw.internal/anthropic"

    def test_constructor_argument_wins(self):
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "k",
                "ANTHROPIC_BASE_URL_OVERRIDE": "https://from-env/v1",
            },
        ):
            with patch("ai_api_unified.ai_anthropic_base.Anthropic") as mock_cls:
                AiAnthropicCompletions(
                    model="claude-opus-4-8", base_url="https://from-arg/v1"
                )
        assert mock_cls.call_args.kwargs["base_url"] == "https://from-arg/v1"

    def test_plaintext_override_rejected_at_construction(self):
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "k",
                "ANTHROPIC_BASE_URL_OVERRIDE": "http://evil.example/v1",
            },
        ):
            with pytest.raises(AiProviderConfigurationError):
                AiAnthropicCompletions(model="claude-opus-4-8")

    def test_admin_key_does_not_follow_the_inference_override(self):
        # The admin key grants org-wide read/write. Routing inference through
        # a gateway must not silently hand that gateway an administration
        # credential, so the admin lookup stays on the vendor host.
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "k",
                "ANTHROPIC_ADMIN_KEY": "admin",
                "ANTHROPIC_BASE_URL_OVERRIDE": "https://gw.internal/anthropic",
                "ANTHROPIC_ADMIN_BASE_URL_OVERRIDE": "",
            },
        ):
            with patch("ai_api_unified.ai_anthropic_base.Anthropic"):
                client = AiAnthropicCompletions(model="claude-opus-4-8")
            response = Mock(status_code=200)
            response.json.return_value = {"id": "org_1", "name": "Acme"}
            with patch(
                "ai_api_unified.ai_anthropic_base.httpx.get", return_value=response
            ) as mock_get:
                client.get_org_info()
        assert (
            mock_get.call_args.args[0]
            == "https://api.anthropic.com/v1/organizations/me"
        )

    def test_admin_lookup_follows_its_own_opt_in_override(self):
        # Egress-restricted networks can route the admin call too, explicitly.
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "k",
                "ANTHROPIC_ADMIN_KEY": "admin",
                "ANTHROPIC_BASE_URL_OVERRIDE": "https://gw.internal/anthropic",
                "ANTHROPIC_ADMIN_BASE_URL_OVERRIDE": "https://admin-gw.internal",
            },
        ):
            with patch("ai_api_unified.ai_anthropic_base.Anthropic"):
                client = AiAnthropicCompletions(model="claude-opus-4-8")
            response = Mock(status_code=200)
            response.json.return_value = {"id": "org_1", "name": "Acme"}
            with patch(
                "ai_api_unified.ai_anthropic_base.httpx.get", return_value=response
            ) as mock_get:
                client.get_org_info()
        assert (
            mock_get.call_args.args[0]
            == "https://admin-gw.internal/v1/organizations/me"
        )


# ── OpenAI ──────────────────────────────────────────────────────────────────

openai_sdk = pytest.importorskip("openai")

from ai_api_unified.completions.ai_openai_completions import (  # noqa: E402
    AiOpenAICompletions,
)


class TestOpenAIOverride:
    def test_env_override_wins_over_geo_residency(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "k",
                "OPENAI_BASE_URL_OVERRIDE": "https://gw.internal/openai",
                "AI_API_GEO_RESIDENCY": "US",
                "OPENAI_BASE_URL": "",
            },
        ):
            with patch("ai_api_unified.ai_openai_base.OpenAI") as mock_cls:
                client = AiOpenAICompletions(model="gpt-4o-mini")
        assert mock_cls.call_args.kwargs["base_url"] == "https://gw.internal/openai"
        assert client.base_url == "https://gw.internal/openai"

    def test_deprecated_openai_base_url_is_validated(self):
        # The SDK's own variable name is commonly set process-wide by other
        # tooling; it must not smuggle a plaintext destination past the guard.
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "k",
                "OPENAI_BASE_URL_OVERRIDE": "",
                "OPENAI_BASE_URL": "http://gw.corp.internal/v1",
            },
        ):
            with pytest.raises(AiProviderConfigurationError):
                AiOpenAICompletions(model="gpt-4o-mini")

    def test_deprecated_openai_base_url_still_works_over_https(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "k",
                "OPENAI_BASE_URL_OVERRIDE": "",
                "OPENAI_BASE_URL": "https://legacy-gw.internal/v1",
            },
        ):
            with patch("ai_api_unified.ai_openai_base.OpenAI") as mock_cls:
                AiOpenAICompletions(model="gpt-4o-mini")
        assert mock_cls.call_args.kwargs["base_url"] == "https://legacy-gw.internal/v1"

    def test_geo_residency_still_applies_without_override(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "k",
                "OPENAI_BASE_URL_OVERRIDE": "",
                "OPENAI_BASE_URL": "",
                "AI_API_GEO_RESIDENCY": "US",
            },
        ):
            with patch("ai_api_unified.ai_openai_base.OpenAI") as mock_cls:
                AiOpenAICompletions(model="gpt-4o-mini")
        assert (
            mock_cls.call_args.kwargs["base_url"]
            == AiOpenAICompletions.OPENAI_US_BASE_URL
        )

    def test_openai_compatible_local_server(self):
        # The headline use case: point the openai engine at any
        # OpenAI-compatible server (Ollama, vLLM, LiteLLM).
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "k",
                "OPENAI_BASE_URL_OVERRIDE": "http://localhost:11434/v1",
            },
        ):
            with patch("ai_api_unified.ai_openai_base.OpenAI") as mock_cls:
                AiOpenAICompletions(model="gpt-4o-mini")
        assert mock_cls.call_args.kwargs["base_url"] == "http://localhost:11434/v1"

    def test_account_org_lookup_follows_override(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "k",
                "OPENAI_BASE_URL_OVERRIDE": "https://gw.internal/openai",
            },
        ):
            with patch("ai_api_unified.ai_openai_base.OpenAI"):
                client = AiOpenAICompletions(model="gpt-4o-mini")
            response = Mock(status_code=200)
            response.json.return_value = {
                "orgs": {"data": [{"id": "org-1", "title": "Acme", "is_default": True}]}
            }
            with patch(
                "ai_api_unified.ai_openai_base.httpx.get", return_value=response
            ) as mock_get:
                client.get_org_info()
        assert mock_get.call_args.args[0] == "https://gw.internal/openai/me"


# ── Gemini ──────────────────────────────────────────────────────────────────

pytest.importorskip("google.genai")

from ai_api_unified.ai_google_base import AIGoogleBase  # noqa: E402


class TestGeminiOverride:
    def test_http_options_carry_override(self):
        base = AIGoogleBase()
        with patch.dict(
            os.environ,
            {
                "GOOGLE_GEMINI_API_KEY": "k",
                "GOOGLE_AUTH_METHOD": "api_key",
                "GOOGLE_GEMINI_BASE_URL_OVERRIDE": "https://gw.internal/gemini",
            },
        ):
            with patch("ai_api_unified.ai_google_base.genai") as mock_genai:
                mock_genai.types.HttpOptions = lambda **kw: kw
                base.get_client(model="gemini-2.5-flash")
        kwargs = mock_genai.Client.call_args.kwargs
        assert kwargs["http_options"] == {"base_url": "https://gw.internal/gemini"}

    def test_no_http_options_without_override(self):
        base = AIGoogleBase()
        with patch.dict(
            os.environ,
            {
                "GOOGLE_GEMINI_API_KEY": "k",
                "GOOGLE_AUTH_METHOD": "api_key",
                "GOOGLE_GEMINI_BASE_URL_OVERRIDE": "",
            },
        ):
            with patch("ai_api_unified.ai_google_base.genai") as mock_genai:
                base.get_client(model="gemini-2.5-flash")
        assert "http_options" not in mock_genai.Client.call_args.kwargs

    def test_plaintext_override_rejected(self):
        base = AIGoogleBase()
        with patch.dict(
            os.environ,
            {
                "GOOGLE_GEMINI_API_KEY": "k",
                "GOOGLE_AUTH_METHOD": "api_key",
                "GOOGLE_GEMINI_BASE_URL_OVERRIDE": "http://evil.example/v1",
            },
        ):
            with pytest.raises(AiProviderConfigurationError):
                base.get_client(model="gemini-2.5-flash")


# ── Factory ─────────────────────────────────────────────────────────────────

from ai_api_unified import AIFactory  # noqa: E402


class TestFactoryOverride:
    def test_factory_passes_override_to_supported_engine(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k"}):
            with patch("ai_api_unified.ai_anthropic_base.Anthropic") as mock_cls:
                AIFactory.get_ai_completions_client(
                    model_name="claude-opus-4-8",
                    completions_engine="claude",
                    base_url="https://gw.internal/anthropic",
                )
        assert mock_cls.call_args.kwargs["base_url"] == "https://gw.internal/anthropic"

    def test_unsupported_engine_rejects_override(self):
        # Silently ignoring the argument would send traffic to the vendor
        # while the caller believed it was proxied.
        with pytest.raises(AiProviderCapabilityUnsupportedError, match="base-URL"):
            AIFactory.get_ai_completions_client(
                model_name="amazon.nova-lite-v1:0",
                completions_engine="nova",
                base_url="https://gw.internal/bedrock",
            )
