from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI, OpenAI
from ai_api_unified.ai_base import (
    RETRY_POLICY_DEFAULT,
    RETRY_POLICY_NONE,
    normalize_retry_policy,
)
from ai_api_unified.util.env_settings import EnvSettings

_LOGGER: logging.Logger = logging.getLogger(__name__)

RETRY_POLICY_KEY: str = "COMPLETIONS_RETRY_POLICY"


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

    def __init__(self, *, retry_policy: str | None = None, **kwargs: Any):
        """
        Initialize the AIOpenAIBase class with environment settings and API credentials.

        Args:
            retry_policy: "default" keeps the OpenAI SDK's built-in retries;
                "none" disables them (max_retries=0). Falls back to the
                COMPLETIONS_RETRY_POLICY environment setting, then "default".
        """
        self.env = EnvSettings()
        self.api_key = self.env.get_setting("OPENAI_API_KEY")
        self.user = self.env.get_setting("OPENAI_USER", "default_user")
        if not self.api_key or self.api_key.strip() == "":
            raise ValueError("OPENAI_API_KEY environment variable must be set.")
        self.base_url = self.get_api_base_url()

        str_candidate: str = (
            retry_policy
            if retry_policy is not None
            else str(self.env.get_setting(RETRY_POLICY_KEY, RETRY_POLICY_DEFAULT))
        )
        self.retry_policy: str = normalize_retry_policy(str_candidate)
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

    def get_api_base_url(self) -> str:
        """Resolve the OpenAI base URL based on data residency constraints."""
        env: EnvSettings = EnvSettings()

        override_url: str | None = env.get_setting("OPENAI_BASE_URL")
        if override_url:
            _LOGGER.warning(
                "OPENAI_BASE_URL is deprecated. Please use AI_API_GEO_RESIDENCY instead.",
            )
            return override_url

        geo_residency: str | None = (
            env.get_geo_residency()
        )  # On success, this normalizes to "US"
        if geo_residency == "US":
            return self.OPENAI_US_BASE_URL

        return self.DEFAULT_OPENAI_BASE_URL
