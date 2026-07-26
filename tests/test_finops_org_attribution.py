# test_finops_org_attribution.py
"""
Tests for organization-level finops attribution (anthropic, openai,
google, bedrock).

With ANTHROPIC_ADMIN_KEY set, the claude engine resolves the org id and name
from the Admin API; without it, one free count_tokens probe captures the org
id from the anthropic-organization-id response header. Identity is cached per
client, fails open, and lands on cost-topic events as org_id / org_name.
"""

import logging
import os
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

anthropic = pytest.importorskip("anthropic")

from ai_api_unified.completions.ai_anthropic_completions import (  # noqa: E402
    AiAnthropicCompletions,
)
from ai_api_unified.middleware.middleware_config import (  # noqa: E402
    ObservabilitySettingsModel,
)
from ai_api_unified.middleware.observability import (  # noqa: E402
    COST_LOGGER_NAME,
    LoggerBackedObservabilityMiddleware,
)
from ai_api_unified.middleware.observability_runtime import (  # noqa: E402
    AiApiCallContextModel,
    AiApiCallResultSummaryModel,
)


def _build_client(**env) -> AiAnthropicCompletions:
    dict_env = {"ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_ADMIN_KEY": "", **env}
    with patch.dict(os.environ, dict_env):
        client = AiAnthropicCompletions(model="claude-opus-4-8")
    client.client = Mock()
    return client


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class TestEnvSettingsDeclaration:
    def test_admin_key_is_a_declared_settings_field(self):
        # get_setting resolves declared fields case-exactly; undeclared .env
        # keys land in model_extra lowercased and are missed. Live testing
        # caught ANTHROPIC_ADMIN_KEY silently unresolved until declared.
        from ai_api_unified.util.env_settings import EnvSettings

        assert "ANTHROPIC_ADMIN_KEY" in EnvSettings.model_fields


class TestAdminKeyResolution:
    def test_admin_api_resolves_org_id_and_name(self):
        client = _build_client(ANTHROPIC_ADMIN_KEY="admin-test-key")
        response = Mock(status_code=200)
        response.json.return_value = {
            "id": "org_123",
            "type": "organization",
            "name": "Acme Robotics",
        }
        with patch(
            "ai_api_unified.ai_anthropic_base.httpx.get", return_value=response
        ) as mock_get:
            org_id, org_name = client._resolve_provider_organization()
        assert (org_id, org_name) == ("org_123", "Acme Robotics")
        kwargs = mock_get.call_args.kwargs
        assert kwargs["headers"]["x-api-key"] == "admin-test-key"
        assert "anthropic-version" in kwargs["headers"]

    def test_resolution_is_cached_after_first_attempt(self):
        client = _build_client(ANTHROPIC_ADMIN_KEY="admin-test-key")
        response = Mock(status_code=200)
        response.json.return_value = {"id": "org_123", "name": "Acme"}
        with patch(
            "ai_api_unified.ai_anthropic_base.httpx.get", return_value=response
        ) as mock_get:
            client._resolve_provider_organization()
            client._resolve_provider_organization()
        assert mock_get.call_count == 1

    def test_admin_api_error_status_fails_open(self):
        client = _build_client(ANTHROPIC_ADMIN_KEY="admin-test-key")
        with patch(
            "ai_api_unified.ai_anthropic_base.httpx.get",
            return_value=Mock(status_code=401),
        ):
            org_id, org_name = client._resolve_provider_organization()
        assert (org_id, org_name) == (None, None)

    def test_admin_api_exception_fails_open(self):
        client = _build_client(ANTHROPIC_ADMIN_KEY="admin-test-key")
        with patch(
            "ai_api_unified.ai_anthropic_base.httpx.get",
            side_effect=RuntimeError("network down"),
        ):
            org_id, org_name = client._resolve_provider_organization()
        assert (org_id, org_name) == (None, None)


class TestHeaderProbeResolution:
    def test_header_probe_resolves_org_id_only(self):
        client = _build_client()
        raw = SimpleNamespace(headers={"anthropic-organization-id": "org_hdr_9"})
        client.client.messages.with_raw_response.count_tokens.return_value = raw
        org_id, org_name = client._resolve_provider_organization()
        assert org_id == "org_hdr_9"
        assert org_name is None
        kwargs = client.client.messages.with_raw_response.count_tokens.call_args.kwargs
        assert kwargs["model"] == "claude-opus-4-8"

    def test_header_probe_runs_once(self):
        client = _build_client()
        raw = SimpleNamespace(headers={"anthropic-organization-id": "org_hdr_9"})
        client.client.messages.with_raw_response.count_tokens.return_value = raw
        client._resolve_provider_organization()
        client._resolve_provider_organization()
        assert client.client.messages.with_raw_response.count_tokens.call_count == 1

    def test_header_probe_failure_fails_open(self):
        client = _build_client()
        client.client.messages.with_raw_response.count_tokens.side_effect = (
            RuntimeError("boom")
        )
        org_id, org_name = client._resolve_provider_organization()
        assert (org_id, org_name) == (None, None)


class TestGetOrgInfo:
    def test_admin_path_returns_full_identity_with_source(self):
        client = _build_client(ANTHROPIC_ADMIN_KEY="admin-test-key")
        response = Mock(status_code=200)
        response.json.return_value = {"id": "org_123", "name": "Acme Robotics"}
        with patch("ai_api_unified.ai_anthropic_base.httpx.get", return_value=response):
            info = client.get_org_info()
        assert info.org_id == "org_123"
        assert info.org_name == "Acme Robotics"
        assert info.source == "admin_api"

    def test_header_probe_returns_id_only_with_source(self):
        client = _build_client()
        raw = SimpleNamespace(headers={"anthropic-organization-id": "org_hdr_9"})
        client.client.messages.with_raw_response.count_tokens.return_value = raw
        info = client.get_org_info()
        assert info.org_id == "org_hdr_9"
        assert info.org_name is None
        assert info.source == "response_header"

    def test_rejected_admin_key_raises_typed_error_with_status(self):
        from ai_api_unified.ai_provider_exceptions import AiProviderRequestError

        client = _build_client(ANTHROPIC_ADMIN_KEY="admin-test-key")
        with patch(
            "ai_api_unified.ai_anthropic_base.httpx.get",
            return_value=Mock(status_code=401),
        ):
            with pytest.raises(AiProviderRequestError) as exc_info:
                client.get_org_info()
        assert exc_info.value.status_code == 401

    def test_probe_failure_raises_typed_error(self):
        from ai_api_unified.ai_provider_exceptions import AiProviderRequestError

        client = _build_client()
        client.client.messages.with_raw_response.count_tokens.side_effect = (
            RuntimeError("boom")
        )
        with pytest.raises(AiProviderRequestError):
            client.get_org_info()

    def test_on_demand_retries_after_failed_enrichment(self):
        # Enrichment negative-caches a failure; an explicit call retries and
        # succeeds without being poisoned by the cached failure.
        client = _build_client()
        probe = client.client.messages.with_raw_response.count_tokens
        probe.side_effect = RuntimeError("transient")
        assert client._resolve_provider_organization() == (None, None)
        probe.side_effect = None
        probe.return_value = SimpleNamespace(
            headers={"anthropic-organization-id": "org_retry"}
        )
        info = client.get_org_info()
        assert info.org_id == "org_retry"
        # And enrichment now sees the cached success.
        assert client._resolve_provider_organization() == ("org_retry", None)

    def test_default_engines_report_source_none(self):
        from ai_api_unified.ai_base import AIBaseEmbeddings

        class _PlainEmbeddings(AIBaseEmbeddings):
            @property
            def list_model_names(self):
                return ["plain"]

            def generate_embeddings(self, text, *, input_type=None):
                return {}

            def generate_embeddings_batch(self, texts, *, input_type=None):
                return []

        info = _PlainEmbeddings(model="plain", dimensions=3).get_org_info()
        assert info.org_id is None
        assert info.org_name is None
        assert info.source == "none"


class TestContextAndCostEvent:
    def test_call_context_carries_org_identity(self):
        client = _build_client(ANTHROPIC_ADMIN_KEY="admin-test-key")
        response = Mock(status_code=200)
        response.json.return_value = {"id": "org_123", "name": "Acme"}
        with patch("ai_api_unified.ai_anthropic_base.httpx.get", return_value=response):
            context = client._build_observability_call_context(
                capability="completions", operation="send_prompt"
            )
        assert context.provider_org_id == "org_123"
        assert context.provider_org_name == "Acme"

    def test_context_builds_when_resolution_hook_raises(self):
        client = _build_client()
        with patch.object(
            client,
            "_resolve_provider_organization",
            side_effect=RuntimeError("boom"),
        ):
            context = client._build_observability_call_context(
                capability="completions", operation="send_prompt"
            )
        assert context.provider_org_id is None
        assert context.provider_org_name is None

    def test_cost_event_includes_org_fields(self):
        handler = _CaptureHandler()
        cost_logger = logging.getLogger(COST_LOGGER_NAME)
        cost_logger.addHandler(handler)
        cost_logger.setLevel(logging.DEBUG)
        try:
            middleware = LoggerBackedObservabilityMiddleware(
                ObservabilitySettingsModel(emit_cost=True)
            )
            call_context = AiApiCallContextModel(
                call_id="call-org-1",
                event_time_utc=datetime.now(timezone.utc),
                capability="completions",
                operation="send_prompt",
                provider_vendor="anthropic",
                provider_engine="claude",
                model_name="claude-opus-4-8",
                model_version=None,
                direction="output",
                provider_org_id="org_123",
                provider_org_name="Acme Robotics",
            )
            summary = AiApiCallResultSummaryModel(
                provider_elapsed_ms=5.0,
                provider_prompt_tokens=1000,
                provider_completion_tokens=100,
            )
            middleware._maybe_emit_cost_event(
                call_context=call_context, call_result_summary=summary
            )
            assert len(handler.records) == 1
            dict_cost_fields = handler.records[0].args[1]
            assert dict_cost_fields["org_id"] == "org_123"
            assert dict_cost_fields["org_name"] == "Acme Robotics"
        finally:
            cost_logger.removeHandler(handler)

    def test_cost_event_omits_identity_as_none_when_unresolved(self):
        handler = _CaptureHandler()
        cost_logger = logging.getLogger(COST_LOGGER_NAME)
        cost_logger.addHandler(handler)
        cost_logger.setLevel(logging.DEBUG)
        try:
            middleware = LoggerBackedObservabilityMiddleware(
                ObservabilitySettingsModel(emit_cost=True)
            )
            call_context = AiApiCallContextModel(
                call_id="call-org-2",
                event_time_utc=datetime.now(timezone.utc),
                capability="completions",
                operation="send_prompt",
                provider_vendor="anthropic",
                provider_engine="claude",
                model_name="claude-opus-4-8",
                model_version=None,
                direction="output",
            )
            summary = AiApiCallResultSummaryModel(
                provider_elapsed_ms=5.0,
                provider_prompt_tokens=1000,
                provider_completion_tokens=100,
            )
            middleware._maybe_emit_cost_event(
                call_context=call_context, call_result_summary=summary
            )
            dict_cost_fields = handler.records[0].args[1]
            assert dict_cost_fields["org_id"] is None
            assert dict_cost_fields["org_name"] is None
        finally:
            cost_logger.removeHandler(handler)


# ── OpenAI, Google, Bedrock org identity ────────────────────────────────────

openai_sdk = pytest.importorskip("openai")

from ai_api_unified.completions.ai_openai_completions import (  # noqa: E402
    AiOpenAICompletions,
)


def _build_openai_client() -> AiOpenAICompletions:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        client = AiOpenAICompletions(model="gpt-4o-mini")
    client.client = Mock()
    return client


def _me_response(orgs: list) -> Mock:
    response = Mock(status_code=200)
    response.json.return_value = {"orgs": {"object": "list", "data": orgs}}
    return response


class TestOpenAIOrgInfo:
    def test_account_api_resolves_default_org(self):
        client = _build_openai_client()
        orgs = [
            {"id": "org-aaa", "title": "Personal", "is_default": False},
            {"id": "org-bbb", "title": "Acme Inc", "is_default": True},
        ]
        with patch(
            "ai_api_unified.ai_openai_base.httpx.get",
            return_value=_me_response(orgs),
        ):
            info = client.get_org_info()
        assert info.org_id == "org-bbb"
        assert info.org_name == "Acme Inc"
        assert info.source == "account_api"

    def test_falls_back_to_header_probe_when_account_api_fails(self):
        client = _build_openai_client()
        client.client.models.with_raw_response.list.return_value = SimpleNamespace(
            headers={"openai-organization": "org-hdr"}
        )
        with patch(
            "ai_api_unified.ai_openai_base.httpx.get",
            return_value=Mock(status_code=403),
        ):
            info = client.get_org_info()
        assert info.org_id == "org-hdr"
        assert info.org_name is None
        assert info.source == "response_header"

    def test_both_paths_failing_raises_primary_error(self):
        from ai_api_unified.ai_provider_exceptions import AiProviderRequestError

        client = _build_openai_client()
        client.client.models.with_raw_response.list.side_effect = RuntimeError("x")
        with patch(
            "ai_api_unified.ai_openai_base.httpx.get",
            return_value=Mock(status_code=401),
        ):
            with pytest.raises(AiProviderRequestError) as exc_info:
                client.get_org_info()
        assert exc_info.value.status_code == 401

    def test_capability_declares_full_identity(self):
        client = _build_openai_client()
        capability = client.get_org_info_capability()
        assert capability.supports_org_id is True
        assert capability.supports_org_name is True
        assert capability.requirement is None


class TestGoogleOrgInfo:
    def test_configured_project_becomes_org_id(self):
        from ai_api_unified.ai_google_base import AIGoogleBase

        base = AIGoogleBase()
        with patch.dict(os.environ, {"GOOGLE_PROJECT_ID": "my-gcp-project"}):
            info = base._get_org_info_provider()
            capability = base._get_org_info_capability_provider()
        assert info.org_id == "my-gcp-project"
        assert info.org_name is None
        assert info.source == "configuration"
        assert capability.supports_org_id is True
        assert capability.supports_org_name is False

    def test_no_project_reports_source_none_with_requirement(self):
        from ai_api_unified.ai_google_base import AIGoogleBase

        base = AIGoogleBase()
        with patch.dict(os.environ, {"GOOGLE_PROJECT_ID": ""}):
            info = base._get_org_info_provider()
            capability = base._get_org_info_capability_provider()
        assert info.org_id is None
        assert info.source == "none"
        assert capability.supports_org_id is False
        assert "GOOGLE_PROJECT_ID" in (capability.requirement or "")


pytest.importorskip("boto3")

from ai_api_unified.completions.ai_bedrock_completions import (  # noqa: E402
    AiBedrockCompletions,
)


def _build_bedrock_client_with_aws(dict_clients: dict):
    def _client_factory(service_name, **kwargs):
        return dict_clients.get(service_name, Mock())

    with patch("ai_api_unified.ai_bedrock_base.boto3") as mock_boto3:
        mock_boto3.client.side_effect = _client_factory
        client = AiBedrockCompletions(model="amazon.nova-lite-v1:0")
        return client, mock_boto3


class TestBedrockOrgInfo:
    def test_sts_account_and_iam_alias(self):
        sts = Mock()
        sts.get_caller_identity.return_value = {"Account": "123456789012"}
        iam = Mock()
        iam.list_account_aliases.return_value = {"AccountAliases": ["acme-prod"]}
        client, mock_boto3 = _build_bedrock_client_with_aws(
            {"sts": sts, "iam": iam, "bedrock-runtime": Mock()}
        )
        with patch("ai_api_unified.ai_bedrock_base.boto3", mock_boto3):
            info = client.get_org_info()
        assert info.org_id == "123456789012"
        assert info.org_name == "acme-prod"
        assert info.source == "account_api"

    def test_missing_alias_permission_yields_id_only(self):
        sts = Mock()
        sts.get_caller_identity.return_value = {"Account": "123456789012"}
        iam = Mock()
        iam.list_account_aliases.side_effect = RuntimeError("AccessDenied")
        client, mock_boto3 = _build_bedrock_client_with_aws(
            {"sts": sts, "iam": iam, "bedrock-runtime": Mock()}
        )
        with patch("ai_api_unified.ai_bedrock_base.boto3", mock_boto3):
            info = client.get_org_info()
        assert info.org_id == "123456789012"
        assert info.org_name is None

    def test_sts_failure_raises_typed_error(self):
        from ai_api_unified.ai_provider_exceptions import AiProviderRequestError

        sts = Mock()
        sts.get_caller_identity.side_effect = RuntimeError("expired token")
        client, mock_boto3 = _build_bedrock_client_with_aws(
            {"sts": sts, "bedrock-runtime": Mock()}
        )
        with patch("ai_api_unified.ai_bedrock_base.boto3", mock_boto3):
            with pytest.raises(AiProviderRequestError):
                client.get_org_info()

    def test_capability_names_the_iam_requirement(self):
        client, _ = _build_bedrock_client_with_aws({"bedrock-runtime": Mock()})
        capability = client.get_org_info_capability()
        assert capability.supports_org_id is True
        assert "iam:ListAccountAliases" in (capability.requirement or "")


class TestAnthropicCapability:
    def test_capability_reflects_admin_key_presence(self):
        with_key = _build_client(ANTHROPIC_ADMIN_KEY="admin-test-key")
        assert with_key.get_org_info_capability().supports_org_name is True
        without_key = _build_client()
        capability = without_key.get_org_info_capability()
        assert capability.supports_org_id is True
        assert capability.supports_org_name is False
        assert "ANTHROPIC_ADMIN_KEY" in (capability.requirement or "")
