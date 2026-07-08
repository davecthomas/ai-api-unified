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

from anthropic import Anthropic

from ai_api_unified.util.env_settings import EnvSettings

_LOGGER: logging.Logger = logging.getLogger(__name__)


class AIAnthropicBase:
    """
    Base class for native Anthropic API interactions.
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize the AIAnthropicBase class with environment settings and API credentials.
        """
        self.env = EnvSettings()
        self.api_key = self.env.get_setting("ANTHROPIC_API_KEY")
        self.user = self.env.get_setting("ANTHROPIC_USER", "default_user")
        if not self.api_key or self.api_key.strip() == "":
            raise ValueError("ANTHROPIC_API_KEY environment variable must be set.")

        self.client = Anthropic(api_key=self.api_key)
        self.backoff_delays = [1, 2, 4, 8, 16]
