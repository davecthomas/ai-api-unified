"""
Embedding-model back-ends (OpenAI, Titan, Gemini…).
"""

from .ai_openai_embeddings import AiOpenAIEmbeddings

has_google_gemini = False
try:
    from .ai_google_gemini_embeddings import GoogleGeminiEmbeddings

    has_google_gemini = True
except ImportError:
    pass

has_bedrock = False
try:
    from .ai_titan_embeddings import AiTitanEmbeddings

    has_bedrock = True
except ImportError:
    pass

__all__ = [
    "AiOpenAIEmbeddings",
]
if has_bedrock:
    __all__.append("AiTitanEmbeddings")
if has_google_gemini:
    __all__.append("GoogleGeminiEmbeddings")
