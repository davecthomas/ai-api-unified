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
    AIProviderOrgInfoBase,
    ORG_INFO_SOURCE_ADMIN_API,
    ORG_INFO_SOURCE_NONE,
    ORG_INFO_SOURCE_RESPONSE_HEADER,
    RETRY_POLICY_DEFAULT,
    RETRY_POLICY_NONE,
    normalize_retry_policy,
)
from ai_api_unified.ai_provider_exceptions import AiProviderRequestError
from ai_api_unified.util.env_settings import EnvSettings

_LOGGER: logging.Logger = logging.getLogger(__name__)

RETRY_POLICY_KEY: str = "COMPLETIONS_RETRY_POLICY"
ANTHROPIC_ADMIN_KEY_SETTING: str = "ANTHROPIC_ADMIN_KEY"
ANTHROPIC_ADMIN_ORGANIZATION_URL: str = "https://api.anthropic.com/v1/organizations/me"
ANTHROPIC_API_VERSION: str = "2023-06-01"
ANTHROPIC_ORG_ID_RESPONSE_HEADER: str = "anthropic-organization-id"
ORG_RESOLUTION_TIMEOUT_SECONDS: float = 10.0


class AIProviderOrgInfoAnthropic(AIProviderOrgInfoBase):
    """
    Anthropic organization identity.

    Carries the normalized base fields today; Anthropic-native additions
    (for example workspace detail) belong here rather than on the base.
    """


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
        # Organization identity is resolved lazily and cached per client.
        # Success caches the info object; failure sets a negative-cache flag
        # honored only by fail-open enrichment, so an explicit get_org_info
        # call still retries and surfaces the typed error.
        self._org_info_cache: AIProviderOrgInfoAnthropic | None = None
        self._bool_org_resolution_failed: bool = False

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

    def _get_org_info_provider(self) -> AIProviderOrgInfoAnthropic:
        """
        Resolves Anthropic organization identity, raising on failure.

        With ANTHROPIC_ADMIN_KEY set, the Admin API supplies org id and name;
        without it, one free count_tokens probe captures the org id from the
        anthropic-organization-id response header. Success is cached per
        client; an explicit call after a failed background attempt retries.

        Raises:
            AiProviderRequestError: When resolution fails, carrying the HTTP
                status code when one was available.
        """
        if self._org_info_cache is not None:
            # Early return with the cached successful identity.
            return self._org_info_cache
        if self.admin_key and self.admin_key.strip():
            org_info: AIProviderOrgInfoAnthropic = self._fetch_org_info_via_admin_api()
        else:
            org_info = self._fetch_org_id_via_header_probe()
        self._org_info_cache = org_info
        self._bool_org_resolution_failed = False
        # Normal return with the freshly resolved identity.
        return org_info

    def _resolve_provider_organization(self) -> tuple[str | None, str | None]:
        """
        Fail-open enrichment resolution with negative caching.

        A failed attempt is remembered so cost enrichment does not retry on
        every call; get_org_info bypasses the negative cache and retries.
        """
        if self._org_info_cache is not None:
            # Early return with the cached successful identity.
            return self._org_info_cache.org_id, self._org_info_cache.org_name
        if self._bool_org_resolution_failed:
            # Early return because a prior attempt failed; enrichment does
            # not retry (get_org_info does).
            return None, None
        try:
            org_info: AIProviderOrgInfoAnthropic = self._get_org_info_provider()
        except Exception as exception:
            self._bool_org_resolution_failed = True
            _LOGGER.warning(
                "Anthropic organization resolution failed (%s); cost events "
                "will omit org identity.",
                exception.__class__.__name__,
            )
            # Early return with no identity because enrichment fails open.
            return None, None
        # Normal return with the resolved identity fields.
        return org_info.org_id, org_info.org_name

    def _fetch_org_info_via_admin_api(self) -> AIProviderOrgInfoAnthropic:
        """
        Fetches organization id and name from the Anthropic Admin API.

        Raises:
            AiProviderRequestError: On transport failure (status_code=None)
                or a non-200 response (status_code set), for example a
                rejected ANTHROPIC_ADMIN_KEY.
        """
        try:
            response: httpx.Response = httpx.get(
                ANTHROPIC_ADMIN_ORGANIZATION_URL,
                headers={
                    "x-api-key": self.admin_key,
                    "anthropic-version": ANTHROPIC_API_VERSION,
                },
                timeout=ORG_RESOLUTION_TIMEOUT_SECONDS,
            )
        except Exception as exception:
            raise AiProviderRequestError(
                "Anthropic Admin API organization lookup failed before a "
                f"status was available: {exception.__class__.__name__}",
                status_code=None,
                provider_engine="claude",
            ) from exception
        if response.status_code != 200:
            raise AiProviderRequestError(
                "Anthropic Admin API organization lookup failed with status "
                f"{response.status_code}. Verify ANTHROPIC_ADMIN_KEY.",
                status_code=response.status_code,
                provider_engine="claude",
            )
        dict_payload: dict = response.json()
        raw_org_id = dict_payload.get("id")
        raw_org_name = dict_payload.get("name")
        # Normal return with the admin-resolved identity.
        return AIProviderOrgInfoAnthropic(
            org_id=str(raw_org_id) if raw_org_id else None,
            org_name=str(raw_org_name) if raw_org_name else None,
            source=ORG_INFO_SOURCE_ADMIN_API,
        )

    def _fetch_org_id_via_header_probe(self) -> AIProviderOrgInfoAnthropic:
        """
        Captures the organization id from one free count_tokens response.

        The Messages API stamps anthropic-organization-id on responses; the
        count_tokens endpoint runs no inference and bills nothing.

        Raises:
            AiProviderRequestError: When the probe request fails.
        """
        str_model: str = str(
            getattr(self, "completions_model", "") or "claude-haiku-4-5"
        )
        try:
            raw_response = self.client.messages.with_raw_response.count_tokens(
                model=str_model,
                messages=[{"role": "user", "content": "."}],
            )
        except Exception as exception:
            raise AiProviderRequestError(
                "Anthropic organization header probe failed: "
                f"{exception.__class__.__name__}",
                status_code=(
                    getattr(exception, "status_code", None)
                    if isinstance(getattr(exception, "status_code", None), int)
                    else None
                ),
                provider_engine="claude",
            ) from exception
        raw_org_id = raw_response.headers.get(ANTHROPIC_ORG_ID_RESPONSE_HEADER)
        # Normal return with the header-resolved identity (id only).
        return AIProviderOrgInfoAnthropic(
            org_id=str(raw_org_id) if raw_org_id else None,
            org_name=None,
            source=(
                ORG_INFO_SOURCE_RESPONSE_HEADER if raw_org_id else ORG_INFO_SOURCE_NONE
            ),
        )
