"""
Embedding-model back-ends (OpenAI, Titan, …).
"""

from .ai_openai_embeddings import AiOpenAIEmbeddings
from .ai_titan_embeddings import AiTitanEmbeddings

__all__ = [
    "AiOpenAIEmbeddings",
    "AiTitanEmbeddings",
]
