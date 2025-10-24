from __future__ import annotations

import logging
from typing import Any
from openai import OpenAI
from ai_api_unified.util.env_settings import EnvSettings

_LOGGER: logging.Logger = logging.getLogger(__name__)


class AIOpenAIBase:
    """
    Base class for OpenAI API interactions.
    """

    DEFAULT_OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_US_BASE_URL: str = "https://us.api.openai.com/v1"

    def __init__(self, **kwargs: Any):
        """
        Initialize the AIOpenAIBase class with environment settings and API credentials.
        """
        self.env = EnvSettings()
        self.api_key = self.env.get_setting("OPENAI_API_KEY")
        self.user = self.env.get_setting("OPENAI_USER", "default_user")
        if not self.api_key or self.api_key.strip() == "":
            raise ValueError("OPENAI_API_KEY environment variable must be set.")
        self.base_url = self.get_api_base_url()

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.backoff_delays = [1, 2, 4, 8, 16]

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
