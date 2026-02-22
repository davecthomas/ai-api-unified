"""
Completion-model back-ends (OpenAI, Bedrock, …).
"""

from .ai_openai_completions import AiOpenAICompletions

has_google_gemini: bool = False
try:
    from .ai_google_gemini_completions import GoogleGeminiCompletions

    has_google_gemini = True
except ImportError:
    pass

has_bedrock: bool = False
try:
    from .ai_bedrock_completions import AiBedrockCompletions

    has_bedrock = True
except ImportError:
    pass

__all__ = [
    "AiOpenAICompletions",
]
if has_google_gemini:
    __all__.append("GoogleGeminiCompletions")
if has_bedrock:
    __all__.append("AiBedrockCompletions")
