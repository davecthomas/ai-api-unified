from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from ai_api_unified.completions.ai_openai_completions import (
    AICompletionsCapabilitiesOpenAI,
    AiOpenAICompletions,
)
from ai_api_unified.embeddings.ai_openai_embeddings import AiOpenAIEmbeddings
from ai_api_unified.voice.ai_voice_openai import AIVoiceOpenAI


class TestOpenAIUSEndpoint:
    """Test that OpenAI clients use the US-specific endpoint by default."""

    @patch("ai_api_unified.completions.ai_openai_completions.OpenAI")
    def test_completions_uses_us_endpoint_when_geo_constrained(self, mock_openai):
        """Test that AiOpenAICompletions switches to the US endpoint when configured."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key", "AI_API_GEO_RESIDENCY": "US"},
            clear=True,
        ):
            AiOpenAICompletions()
            mock_openai.assert_called_once()
            args, kwargs = mock_openai.call_args
            assert "base_url" in kwargs
            assert kwargs["base_url"] == "https://us.api.openai.com/v1"

    @patch("ai_api_unified.embeddings.ai_openai_embeddings.OpenAI")
    def test_embeddings_use_us_endpoint_when_geo_constrained(self, mock_openai):
        """Test that AiOpenAIEmbeddings switches to the US endpoint when configured."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key", "AI_API_GEO_RESIDENCY": "United States"},
            clear=True,
        ):
            AiOpenAIEmbeddings()
            mock_openai.assert_called_once()
            args, kwargs = mock_openai.call_args
            assert "base_url" in kwargs
            assert kwargs["base_url"] == "https://us.api.openai.com/v1"

    def test_capabilities_report_data_residency_support(self):
        """Ensure OpenAI capabilities indicate data residency support."""

        capabilities = AICompletionsCapabilitiesOpenAI.for_model("gpt-4o-mini")
        assert capabilities.supports_data_residency_constraint is True
        assert capabilities.context_window_length == 128_000

    @patch("ai_api_unified.completions.ai_openai_completions.OpenAI")
    def test_invalid_geo_logs_warning_and_defaults(
        self, mock_openai, caplog: pytest.LogCaptureFixture
    ):
        """Unsupported geo values should log a warning and fall back to the global endpoint."""

        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key", "AI_API_GEO_RESIDENCY": "europe"},
            clear=True,
        ):
            with caplog.at_level("WARNING"):
                AiOpenAICompletions()
        mock_openai.assert_called_once()
        args, kwargs = mock_openai.call_args
        assert kwargs["base_url"] == "https://api.openai.com/v1"
        assert any(
            "Unsupported AI_API_GEO_RESIDENCY" in record.message
            for record in caplog.records
        )
