"""
ai_api_unified · Unified access layer for LLM providers.

Public API surface:
  - __version__             Package version string
  - AIFactory               Factory for completions & embeddings clients
  - AIBase                  Base LLM client abstraction
  - AIBaseEmbeddings        Embeddings-specific abstraction
  - AIBaseCompletions       Completions-specific abstraction
  - AIStructuredPrompt      Base class for structured-prompt models
  - AiOpenAICompletions     OpenAI completions back-end
  - AiBedrockCompletions    Amazon Bedrock completions back-end
  - AiOpenAIEmbeddings      OpenAI embeddings back-end
  - AiTitanEmbeddings       Amazon Titan embeddings back-end
"""

from __future__ import annotations
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    # Installed package metadata
    __version__: str = _pkg_version(__name__)
except PackageNotFoundError:
    # Editable/develop mode fallback
    from .__version__ import __version__  # type: ignore

# Factory
from .ai_factory import AIFactory

# Core abstractions & prompt base
from .ai_base import AIBase, AIBaseEmbeddings, AIBaseCompletions, AIStructuredPrompt

# Concrete back-ends
from .completions.ai_openai_completions import AiOpenAICompletions
from .completions.ai_bedrock_completions import AiBedrockCompletions
from .embeddings.ai_openai_embeddings import AiOpenAIEmbeddings
from .embeddings.ai_titan_embeddings import AiTitanEmbeddings

__all__: list[str] = [
    "__version__",
    "AIFactory",
    "AIBase",
    "AIBaseEmbeddings",
    "AIBaseCompletions",
    "AIStructuredPrompt",
    "AiOpenAICompletions",
    "AiBedrockCompletions",
    "AiOpenAIEmbeddings",
    "AiTitanEmbeddings",
]
# Public API surface for ai_api_unified package
