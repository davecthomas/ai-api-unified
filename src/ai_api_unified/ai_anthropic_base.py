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

from anthropic import Anthropic, AsyncAnthropic

from ai_api_unified.ai_base import RETRY_POLICY_DEFAULT, RETRY_POLICY_NONE
from ai_api_unified.util.env_settings import EnvSettings

_LOGGER: logging.Logger = logging.getLogger(__name__)

RETRY_POLICY_KEY: str = "COMPLETIONS_RETRY_POLICY"


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
        str_normalized: str = str_candidate.strip().lower()
        if str_normalized not in (RETRY_POLICY_DEFAULT, RETRY_POLICY_NONE):
            raise ValueError(
                f"Unsupported retry policy {str_candidate!r}; "
                f"expected '{RETRY_POLICY_DEFAULT}' or '{RETRY_POLICY_NONE}'."
            )
        # Normal return with the normalized retry policy token.
        return str_normalized

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
