"""
Completion-model back-ends (OpenAI, Bedrock, …).
"""

from .ai_bedrock_completions import AiBedrockCompletions
from .ai_openai_completions import AiOpenAICompletions

__all__ = [
    "AiBedrockCompletions",
    "AiOpenAICompletions",
]
