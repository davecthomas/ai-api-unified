# src/ai_api_unified/completions/ai_google_gemini_capabilities.py
from datetime import date

from pydantic import Field

from ..ai_base import (
    AICompletionsCapabilitiesBase,
    AICompletionsPromptParamsBase,
    SupportedDataType,
)


class AICompletionsCapabilitiesGoogle(AICompletionsCapabilitiesBase):
    """
    Google Gemini-specific completions capabilities.

    Based on https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash-lite
    """

    @classmethod
    def for_model(cls, model_name: str) -> "AICompletionsCapabilitiesGoogle":
        """Create capabilities instance for specific Gemini model."""
        # Default capabilities for all Gemini models
        capabilities = {
            "reasoning": False,
            "supported_data_types": [
                SupportedDataType.TEXT,
                SupportedDataType.IMAGE,
                SupportedDataType.VIDEO,
                SupportedDataType.AUDIO,
                SupportedDataType.PDF,
            ],
        }

        # Model-specific capabilities
        if "2.5" in model_name:
            capabilities.update(
                {
                    "context_window_length": 1048576,
                    "knowledge_cutoff_date": date(2024, 10, 1),  # Approximate
                    "reasoning": True,  # Gemini 2.5 supports reasoning
                }
            )
        elif "2.0" in model_name:
            capabilities.update(
                {
                    "context_window_length": 1048576,
                    "knowledge_cutoff_date": date(2024, 10, 1),  # Approximate
                    "reasoning": False,
                }
            )
        elif "1.5-pro" in model_name:
            capabilities.update(
                {
                    "context_window_length": 2097152,
                    "knowledge_cutoff_date": date(2024, 4, 1),  # Approximate
                    "reasoning": False,
                }
            )
        elif "1.5-flash" in model_name:
            capabilities.update(
                {
                    "context_window_length": 1048576,
                    "knowledge_cutoff_date": date(2024, 4, 1),  # Approximate
                    "reasoning": False,
                }
            )
        else:
            # Default for unknown models
            capabilities.update(
                {
                    "context_window_length": 1048576,
                    "knowledge_cutoff_date": None,
                    "reasoning": False,
                }
            )

        return cls(**capabilities)


class AICompletionsPromptParamsGoogle(AICompletionsPromptParamsBase):
    """
    Google Gemini-specific prompt parameters.
    """

    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(
        default=1, ge=1, le=40
    )  # Default is 1 per Gemini API docs; max allowed is 40
    reasoning_effort: bool = Field(default=False)
    max_output_tokens: int = Field(default=2048, ge=1, le=8192)
