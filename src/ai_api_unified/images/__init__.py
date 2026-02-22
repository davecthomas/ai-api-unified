# src/ai_api_unified/images/__init__.py
from __future__ import annotations

from .ai_openai_images import AIOpenAIImages

__all__: list[str] = [
    "AIOpenAIImages",
]

# ----- Optional providers (import if available; otherwise skip) -----

has_bedrock: bool = False
try:
    from .ai_bedrock_images import AINovaCanvasImages

    has_bedrock = True
except ImportError:
    pass

if has_bedrock:
    __all__.append("AINovaCanvasImages")
