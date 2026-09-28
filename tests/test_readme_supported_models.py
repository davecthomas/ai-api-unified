# test_readme_supported_models.py
"""
Keeps the README "Supported models" table in step with the engines.

Every model an engine catalogs must appear in the table, so adding a model
to an engine without documenting it fails here. Retired models and
non-model IDs are the listed exceptions.
"""

from pathlib import Path

import pytest

README_PATH: Path = Path(__file__).resolve().parents[1] / "README.md"
TABLE_START: str = "<!-- supported-models:start -->"
TABLE_END: str = "<!-- supported-models:end -->"
# "o4-mini-high" is a reasoning setting on o4-mini, not an OpenAI model ID.
FROZENSET_EXCLUDED: frozenset[str] = frozenset({"o4-mini-high"})


def _table_text() -> str:
    str_readme: str = README_PATH.read_text()
    int_start: int = str_readme.index(TABLE_START)
    int_end: int = str_readme.index(TABLE_END)
    return str_readme[int_start:int_end]


def _catalogs() -> dict[str, list[str]]:
    pytest.importorskip("openai")
    pytest.importorskip("anthropic")
    pytest.importorskip("boto3")
    pytest.importorskip("google.genai")
    pytest.importorskip("voyageai")
    from ai_api_unified.completions.ai_anthropic_completions import (
        AiAnthropicCompletions,
    )
    from ai_api_unified.completions.ai_bedrock_completions import (
        AiBedrockCompletions,
    )
    from ai_api_unified.completions.ai_google_gemini_completions import (
        GEMINI_MODEL_SPECS,
    )
    from ai_api_unified.completions.ai_openai_completions import AiOpenAICompletions
    from ai_api_unified.embeddings.ai_google_gemini_embeddings import (
        GoogleGeminiEmbeddings,
    )
    from ai_api_unified.embeddings.ai_openai_embeddings import AiOpenAIEmbeddings
    from ai_api_unified.embeddings.ai_titan_embeddings import AiTitanEmbeddings
    from ai_api_unified.embeddings.ai_voyage_embeddings import (
        AIEmbeddingsCapabilitiesVoyage,
    )
    from ai_api_unified.images.ai_google_gemini_images import AIGoogleGeminiImages
    from ai_api_unified.images.ai_openai_images import AIOpenAIImages
    from ai_api_unified.videos.ai_google_gemini_videos import AIGoogleGeminiVideos

    def _property(cls: type, str_name: str) -> list[str]:
        # The catalogs are plain lists; read them without running __init__.
        return list(getattr(cls, str_name).fget(cls.__new__(cls)))

    # Normal return with every engine catalog keyed by a readable label.
    return {
        "openai completions": _property(AiOpenAICompletions, "list_model_names"),
        "claude completions": _property(AiAnthropicCompletions, "list_model_names"),
        "gemini completions": list(GEMINI_MODEL_SPECS),
        "bedrock completions": _property(AiBedrockCompletions, "list_model_names"),
        "openai embeddings": _property(AiOpenAIEmbeddings, "list_model_names"),
        "gemini embeddings": _property(GoogleGeminiEmbeddings, "list_model_names"),
        "titan embeddings": _property(AiTitanEmbeddings, "list_model_names"),
        "voyage embeddings": list(AIEmbeddingsCapabilitiesVoyage.DICT_MODEL_DIMENSIONS),
        "openai images": list(AIOpenAIImages.SUPPORTED_IMAGE_MODELS),
        "gemini images": list(AIGoogleGeminiImages.SUPPORTED_IMAGE_MODELS),
        "gemini video": list(AIGoogleGeminiVideos.SUPPORTED_VIDEO_MODELS),
    }


def test_every_cataloged_model_is_in_the_readme_table() -> None:
    str_table: str = _table_text()
    list_missing: list[str] = [
        f"{str_label}: {str_model}"
        for str_label, list_models in _catalogs().items()
        for str_model in list_models
        if str_model not in FROZENSET_EXCLUDED and f"`{str_model}`" not in str_table
    ]
    assert not list_missing, (
        "Models cataloged in code but missing from the README 'Supported "
        f"models' table: {list_missing}"
    )
