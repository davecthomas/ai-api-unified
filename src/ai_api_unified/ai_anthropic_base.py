# ai_anthropic_base.py

"""
Shared client setup for providers backed by the native Anthropic API
(api.anthropic.com). Claude models are also reachable through Amazon Bedrock
via the `anthropic` completions engine; this base is only for the direct
Anthropic API path (`claude` engine), which authenticates with
ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from anthropic import Anthropic, AsyncAnthropic

from ai_api_unified.ai_base import (
    RETRY_POLICY_DEFAULT,
    RETRY_POLICY_NONE,
    normalize_retry_policy,
)
from ai_api_unified.util.env_settings import EnvSettings

_LOGGER: logging.Logger = logging.getLogger(__name__)

RETRY_POLICY_KEY: str = "COMPLETIONS_RETRY_POLICY"
ANTHROPIC_ADMIN_KEY_SETTING: str = "ANTHROPIC_ADMIN_KEY"
ANTHROPIC_ADMIN_ORGANIZATION_URL: str = "https://api.anthropic.com/v1/organizations/me"
ANTHROPIC_API_VERSION: str = "2023-06-01"
ANTHROPIC_ORG_ID_RESPONSE_HEADER: str = "anthropic-organization-id"
ORG_RESOLUTION_TIMEOUT_SECONDS: float = 10.0


class AIAnthropicBase:
    """
    Base class for native Anthropic API interactions.

    Retry behavior: the Anthropic SDK retries transient failures (408, 409,
    429, and 5xx) twice by default with exponential backoff. Pass
    retry_policy="none" (or set COMPLETIONS_RETRY_POLICY=none) to disable SDK
    retries so caller-owned backoff is the only retry layer.
    """

    def __init__(self, *, retry_policy: str | None = None, **kwargs: Any):
        """
        Initialize the AIAnthropicBase class with environment settings and API credentials.

        Args:
            retry_policy: "default" keeps the Anthropic SDK's built-in retries;
                "none" disables them (max_retries=0). Falls back to the
                COMPLETIONS_RETRY_POLICY environment setting, then "default".
        """
        self.env = EnvSettings()
        self.api_key = self.env.get_setting("ANTHROPIC_API_KEY")
        self.user = self.env.get_setting("ANTHROPIC_USER", "default_user")
        if not self.api_key or self.api_key.strip() == "":
            raise ValueError("ANTHROPIC_API_KEY environment variable must be set.")

        self.retry_policy: str = self._resolve_retry_policy(retry_policy)
        int_max_retries: int | None = (
            0 if self.retry_policy == RETRY_POLICY_NONE else None
        )
        if int_max_retries is None:
            self.client = Anthropic(api_key=self.api_key)
        else:
            self.client = Anthropic(api_key=self.api_key, max_retries=int_max_retries)
        # The async client is created lazily so purely synchronous consumers
        # never pay for an unused event-loop-bound transport.
        self._async_client: AsyncAnthropic | None = None
        self.backoff_delays = [1, 2, 4, 8, 16]
        # Optional Admin API key for organization-level finops attribution.
        self.admin_key: str = str(
            self.env.get_setting(ANTHROPIC_ADMIN_KEY_SETTING, "") or ""
        )
        # Organization identity is resolved lazily, once per client, and
        # cached (including failed attempts) so cost enrichment never adds
        # per-call overhead or repeated probes.
        self._bool_org_resolution_attempted: bool = False
        self._str_org_id: str | None = None
        self._str_org_name: str | None = None

    def _resolve_retry_policy(self, retry_policy: str | None) -> str:
        """
        Resolves the effective retry policy from the constructor or environment.

        Args:
            retry_policy: Optional explicit constructor override.

        Returns:
            Normalized retry policy token ("default" or "none").

        Raises:
            ValueError: When an unrecognized retry policy value is supplied.
        """
        str_candidate: str = (
            retry_policy
            if retry_policy is not None
            else str(self.env.get_setting(RETRY_POLICY_KEY, RETRY_POLICY_DEFAULT))
        )
        # Normal return with the normalized retry policy token.
        return normalize_retry_policy(str_candidate)

    @property
    def async_client(self) -> AsyncAnthropic:
        """
        Returns the lazily created AsyncAnthropic client for async variants.
        """
        if self._async_client is None:
            int_max_retries: int | None = (
                0 if self.retry_policy == RETRY_POLICY_NONE else None
            )
            if int_max_retries is None:
                self._async_client = AsyncAnthropic(api_key=self.api_key)
            else:
                self._async_client = AsyncAnthropic(
                    api_key=self.api_key, max_retries=int_max_retries
                )
        # Normal return with the shared async client instance.
        return self._async_client

    def _resolve_provider_organization(self) -> tuple[str | None, str | None]:
        """
        Resolves the Anthropic organization identity for finops attribution.

        With ANTHROPIC_ADMIN_KEY set, the Admin API supplies the organization
        id and name; without it, one free count_tokens probe captures the
        organization id from the anthropic-organization-id response header.
        Resolution is attempted once per client and fails open: cost events
        simply omit the fields when identity is unavailable.

        Returns:
            Tuple of (org_id, org_name); either element may be None.
        """
        if self._bool_org_resolution_attempted:
            # Early return with the cached (possibly empty) identity.
            return self._str_org_id, self._str_org_name
        self._bool_org_resolution_attempted = True
        try:
            if self.admin_key and self.admin_key.strip():
                self._resolve_organization_via_admin_api()
            else:
                self._resolve_organization_id_via_header_probe()
        except Exception as exception:
            _LOGGER.warning(
                "Anthropic organization resolution failed (%s); cost events "
                "will omit org identity.",
                exception.__class__.__name__,
            )
        # Normal return with whatever identity resolution produced.
        return self._str_org_id, self._str_org_name

    def _resolve_organization_via_admin_api(self) -> None:
        """
        Fetches organization id and name from the Anthropic Admin API.
        """
        response: httpx.Response = httpx.get(
            ANTHROPIC_ADMIN_ORGANIZATION_URL,
            headers={
                "x-api-key": self.admin_key,
                "anthropic-version": ANTHROPIC_API_VERSION,
            },
            timeout=ORG_RESOLUTION_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            _LOGGER.warning(
                "Anthropic Admin API organization lookup returned %s; cost "
                "events will omit org identity. Verify ANTHROPIC_ADMIN_KEY.",
                response.status_code,
            )
            # Early return leaving identity unset.
            return None
        dict_payload: dict = response.json()
        raw_org_id = dict_payload.get("id")
        raw_org_name = dict_payload.get("name")
        self._str_org_id = str(raw_org_id) if raw_org_id else None
        self._str_org_name = str(raw_org_name) if raw_org_name else None
        # Normal return after caching the admin-resolved identity.
        return None

    def _resolve_organization_id_via_header_probe(self) -> None:
        """
        Captures the organization id from one free count_tokens response.

        The Messages API stamps anthropic-organization-id on responses; the
        count_tokens endpoint runs no inference and bills nothing, so one
        probe per client is a cost-free way to attribute spend by org id.
        """
        str_model: str = str(
            getattr(self, "completions_model", "") or "claude-haiku-4-5"
        )
        raw_response = self.client.messages.with_raw_response.count_tokens(
            model=str_model,
            messages=[{"role": "user", "content": "."}],
        )
        raw_org_id = raw_response.headers.get(ANTHROPIC_ORG_ID_RESPONSE_HEADER)
        self._str_org_id = str(raw_org_id) if raw_org_id else None
        # Normal return after caching the header-resolved org id.
        return None
