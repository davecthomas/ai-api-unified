from __future__ import annotations

import logging
from typing import Any

import httpx
from openai import AsyncOpenAI, OpenAI
from ai_api_unified.ai_base import (
    AIProviderOrgInfoBase,
    resolve_base_url_override,
    validate_base_url_override,
    AIProviderOrgInfoCapability,
    ORG_INFO_SOURCE_ACCOUNT_API,
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
OPENAI_BASE_URL_OVERRIDE_SETTING: str = "OPENAI_BASE_URL_OVERRIDE"
OPENAI_USER_SETTING_KEY: str = "OPENAI_USER"
DEFAULT_OPENAI_USER: str = "default_user"
OPENAI_ME_PATH: str = "/me"
OPENAI_ORG_ID_RESPONSE_HEADER: str = "openai-organization"
ORG_RESOLUTION_TIMEOUT_SECONDS: float = 10.0


class AIProviderOrgInfoOpenAI(AIProviderOrgInfoBase):
    """OpenAI organization identity (id and title from the account API)."""


class AIOpenAIBase:
    """
    Base class for OpenAI API interactions.

    Retry behavior: the OpenAI SDK retries transient failures (408, 409, 429,
    and 5xx) twice by default with exponential backoff. Pass
    retry_policy="none" (or set COMPLETIONS_RETRY_POLICY=none) to disable SDK
    retries so caller-owned backoff is the only retry layer.
    """

    DEFAULT_OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_US_BASE_URL: str = "https://us.api.openai.com/v1"

    def __init__(
        self,
        *,
        retry_policy: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the AIOpenAIBase class with environment settings and API credentials.

        Args:
            retry_policy: "default" keeps the OpenAI SDK's built-in retries;
                "none" disables them (max_retries=0). Falls back to the
                COMPLETIONS_RETRY_POLICY environment setting, then "default".
            base_url: Optional API base-URL override (gateways, proxies,
                OpenAI-compatible servers). Falls back to
                OPENAI_BASE_URL_OVERRIDE, the deprecated OPENAI_BASE_URL,
                geo-residency routing, then the OpenAI default. Must be
                https:// unless it targets a loopback host.
        """
        self.env = EnvSettings()
        self.api_key = self.env.get_setting("OPENAI_API_KEY")
        # Blank is unconfigured here too: a present-but-blank OPENAI_USER would
        # otherwise resolve to "" and drop caller attribution to None, while the
        # documented contract promises the sentinel.
        str_configured_user: str = str(
            self.env.get_setting(OPENAI_USER_SETTING_KEY, DEFAULT_OPENAI_USER) or ""
        ).strip()
        self.user = str_configured_user or DEFAULT_OPENAI_USER
        if not self.api_key or self.api_key.strip() == "":
            raise ValueError("OPENAI_API_KEY environment variable must be set.")
        self.base_url = self.get_api_base_url(base_url=base_url)

        # get_setting returns "" for a key that is present but blank, not the
        # default, and "" is not a valid policy. Treat blank as unconfigured so a
        # stray `COMPLETIONS_RETRY_POLICY=` line cannot break client construction.
        object_configured_policy: object = (
            retry_policy
            if retry_policy is not None
            else self.env.get_setting(RETRY_POLICY_KEY, RETRY_POLICY_DEFAULT)
        )
        str_candidate: str = (
            str(object_configured_policy).strip()
            if object_configured_policy is not None
            else ""
        )
        self.retry_policy: str = normalize_retry_policy(
            str_candidate or RETRY_POLICY_DEFAULT
        )
        dict_client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
        }
        if self.retry_policy == RETRY_POLICY_NONE:
            dict_client_kwargs["max_retries"] = 0
        self.client = OpenAI(**dict_client_kwargs)
        # The async client is created lazily so purely synchronous consumers
        # never pay for an unused event-loop-bound transport.
        self._async_client: AsyncOpenAI | None = None
        self.backoff_delays = [1, 2, 4, 8, 16]

    @property
    def async_client(self) -> AsyncOpenAI:
        """
        Returns the lazily created AsyncOpenAI client for async variants.
        """
        if self._async_client is None:
            dict_client_kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "base_url": self.base_url,
            }
            if self.retry_policy == RETRY_POLICY_NONE:
                dict_client_kwargs["max_retries"] = 0
            self._async_client = AsyncOpenAI(**dict_client_kwargs)
        # Normal return with the shared async client instance.
        return self._async_client

    def get_api_base_url(self, *, base_url: str | None = None) -> str:
        """
        Resolve the OpenAI base URL.

        Precedence: the caller argument, then OPENAI_BASE_URL_OVERRIDE, then
        the deprecated OPENAI_BASE_URL, then geo-residency routing, then the
        OpenAI default. The override is validated (https, or loopback) and
        always passed to the SDK explicitly, so the SDK's own OPENAI_BASE_URL
        variable cannot take effect unvalidated.
        """
        env: EnvSettings = EnvSettings()

        str_validated_override: str | None = resolve_base_url_override(
            env,
            str_env_key=OPENAI_BASE_URL_OVERRIDE_SETTING,
            str_explicit=base_url,
        )
        if str_validated_override:
            # Early return with the validated override.
            return str_validated_override

        override_url: str | None = env.get_setting("OPENAI_BASE_URL")
        if override_url:
            _LOGGER.warning(
                "OPENAI_BASE_URL is deprecated. Please use "
                "OPENAI_BASE_URL_OVERRIDE (or AI_API_GEO_RESIDENCY) instead.",
            )
            # Validated like the new override: this is the SDK's own variable
            # name, so other tooling may set it process-wide, and it feeds
            # credential-bearing requests including the org lookup.
            return validate_base_url_override(
                override_url, str_env_key="OPENAI_BASE_URL"
            )

        geo_residency: str | None = (
            env.get_geo_residency()
        )  # On success, this normalizes to "US"
        if geo_residency == "US":
            return self.OPENAI_US_BASE_URL

        return self.DEFAULT_OPENAI_BASE_URL

    def _get_org_info_provider(self) -> AIProviderOrgInfoOpenAI:
        """
        Resolves OpenAI organization identity, raising on failure.

        Primary: the account API (/v1/me) returns the key's organizations
        with id and title, using the regular API key. Fallback: one free
        models.list probe captures the org id from the openai-organization
        response header. Caching lives on AIBase.

        Raises:
            AiProviderRequestError: When both resolution paths fail.
        """
        try:
            # Normal return with the account-API identity (id and name).
            return self._fetch_org_info_via_account_api()
        except AiProviderRequestError as primary_error:
            try:
                # Normal return with the header-resolved identity (id only).
                return self._fetch_org_id_via_header_probe()
            except Exception:
                raise primary_error

    def _get_org_info_capability_provider(self) -> AIProviderOrgInfoCapability:
        """
        Declares OpenAI org-identity resolvability: id and name resolve from
        the account API with the regular API key.
        """
        # Normal return with the configured capability declaration.
        return AIProviderOrgInfoCapability(
            supports_org_id=True,
            supports_org_name=True,
            requirement=None,
        )

    def _fetch_org_info_via_account_api(self) -> AIProviderOrgInfoOpenAI:
        """
        Fetches organization id and title from the OpenAI account API.

        Raises:
            AiProviderRequestError: On transport failure, a non-200
                response, or a payload without organizations.
        """
        try:
            response: httpx.Response = httpx.get(
                f"{self.base_url.rstrip('/')}{OPENAI_ME_PATH}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=ORG_RESOLUTION_TIMEOUT_SECONDS,
            )
        except Exception as exception:
            raise AiProviderRequestError(
                "OpenAI account API organization lookup failed before a "
                f"status was available: {exception.__class__.__name__}",
                status_code=None,
                provider_engine="openai",
            ) from exception
        if response.status_code != 200:
            raise AiProviderRequestError(
                "OpenAI account API organization lookup failed with status "
                f"{response.status_code}.",
                status_code=response.status_code,
                provider_engine="openai",
            )
        try:
            dict_payload: dict = response.json()
        except ValueError as exception:
            # A non-JSON 200 body still follows the typed-error contract, so
            # the caller's header-probe fallback can fire.
            raise AiProviderRequestError(
                "OpenAI account API returned a non-JSON organization payload.",
                status_code=response.status_code,
                provider_engine="openai",
            ) from exception
        list_orgs: list = (dict_payload.get("orgs") or {}).get("data") or []
        if not list_orgs:
            raise AiProviderRequestError(
                "OpenAI account API returned no organizations for this key.",
                status_code=None,
                provider_engine="openai",
            )
        # Prefer the key's default organization when the flag is present.
        dict_org: dict = next(
            (org for org in list_orgs if org.get("is_default")), list_orgs[0]
        )
        raw_org_id = dict_org.get("id")
        raw_org_name = dict_org.get("title")
        # Normal return with the account-resolved identity.
        return AIProviderOrgInfoOpenAI(
            org_id=str(raw_org_id) if raw_org_id else None,
            org_name=str(raw_org_name) if raw_org_name else None,
            source=ORG_INFO_SOURCE_ACCOUNT_API,
        )

    def _fetch_org_id_via_header_probe(self) -> AIProviderOrgInfoOpenAI:
        """
        Captures the organization id from one free models.list response.
        """
        try:
            raw_response = self.client.models.with_raw_response.list()
        except Exception as exception:
            raise AiProviderRequestError(
                "OpenAI organization header probe failed: "
                f"{exception.__class__.__name__}",
                status_code=(
                    getattr(exception, "status_code", None)
                    if isinstance(getattr(exception, "status_code", None), int)
                    else None
                ),
                provider_engine="openai",
            ) from exception
        raw_org_id = raw_response.headers.get(OPENAI_ORG_ID_RESPONSE_HEADER)
        # Normal return with the header-resolved identity (id only).
        return AIProviderOrgInfoOpenAI(
            org_id=str(raw_org_id) if raw_org_id else None,
            org_name=None,
            source=(
                ORG_INFO_SOURCE_RESPONSE_HEADER if raw_org_id else ORG_INFO_SOURCE_NONE
            ),
        )
