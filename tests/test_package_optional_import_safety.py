"""Regression tests ensuring package imports remain safe with optional provider extras."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

PYTHON_PROBE_TIMEOUT_SECONDS: int = 30
PROBE_CODE_SNIPPET_MAX_CHARS: int = 500
PROBE_CODE_TRUNCATION_SUFFIX: str = "...(truncated)"


def _run_python_probe(str_python_code: str) -> dict[str, Any]:
    """
    Runs a short Python snippet in a fresh interpreter and parses JSON output.

    Args:
        str_python_code: Python source code executed with ``python -c``.

    Returns:
        A dictionary of probe results emitted by the child process.
        Raises AssertionError when the child process fails or emits invalid output.
    """
    try:
        completed_process: subprocess.CompletedProcess[str] = subprocess.run(
            [sys.executable, "-c", str_python_code],
            check=False,
            capture_output=True,
            text=True,
            timeout=PYTHON_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exception:
        str_stdout: str = exception.stdout if isinstance(exception.stdout, str) else ""
        str_stderr: str = exception.stderr if isinstance(exception.stderr, str) else ""
        str_python_code_snippet: str = str_python_code.strip()
        if len(str_python_code_snippet) > PROBE_CODE_SNIPPET_MAX_CHARS:
            str_python_code_snippet = (
                f"{str_python_code_snippet[:PROBE_CODE_SNIPPET_MAX_CHARS]}"
                f"{PROBE_CODE_TRUNCATION_SUFFIX}"
            )
        raise AssertionError(
            "Probe process timed out. "
            f"timeout_seconds={PYTHON_PROBE_TIMEOUT_SECONDS}, "
            f"stderr={str_stderr!r}, stdout={str_stdout!r}, "
            f"python_code_snippet={str_python_code_snippet!r}"
        ) from exception

    if completed_process.returncode != 0:
        raise AssertionError(
            "Probe process failed. "
            f"stderr={completed_process.stderr!r}, stdout={completed_process.stdout!r}"
        )

    object_probe_results: object = json.loads(completed_process.stdout.strip())
    if not isinstance(object_probe_results, dict):
        raise AssertionError(
            f"Expected probe output dictionary, got {type(object_probe_results).__name__}."
        )
    dict_probe_results: dict[str, Any] = object_probe_results
    # Normal return with parsed probe results.
    return dict_probe_results


def test_root_package_import_does_not_eager_load_openai_modules() -> None:
    """
    Verifies root package import does not eagerly import optional OpenAI provider modules.

    Args:
        None

    Returns:
        None
    """
    str_python_code: str = """
import json
import sys
import ai_api_unified
print(json.dumps({
    "openai_base": "ai_api_unified.ai_openai_base" in sys.modules,
    "openai_completions": "ai_api_unified.completions.ai_openai_completions" in sys.modules,
    "openai_embeddings": "ai_api_unified.embeddings.ai_openai_embeddings" in sys.modules,
    "openai_images": "ai_api_unified.images.ai_openai_images" in sys.modules,
    "openai_voice": "ai_api_unified.voice.ai_voice_openai" in sys.modules
}))
"""
    dict_probe_results: dict[str, bool] = _run_python_probe(str_python_code)

    assert dict_probe_results["openai_base"] is False
    assert dict_probe_results["openai_completions"] is False
    assert dict_probe_results["openai_embeddings"] is False
    assert dict_probe_results["openai_images"] is False
    assert dict_probe_results["openai_voice"] is False


def test_root_package_import_does_not_eager_load_any_registry_provider_modules() -> (
    None
):
    """
    Verifies root package import does not eagerly import any provider module in the AiProvider registry.

    Args:
        None

    Returns:
        None
    """
    str_python_code: str = """
import json
import sys
from ai_api_unified.ai_provider_registry import DICT_TUPLE_AI_PROVIDER_REGISTRY
import ai_api_unified

list_str_provider_modules = sorted({
    ai_provider_spec.str_module_path
    for ai_provider_spec in DICT_TUPLE_AI_PROVIDER_REGISTRY.values()
})

print(json.dumps({
    str_module_name: str_module_name in sys.modules
    for str_module_name in list_str_provider_modules
}))
"""
    dict_probe_results: dict[str, Any] = _run_python_probe(str_python_code)
    for bool_loaded in dict_probe_results.values():
        assert bool_loaded is False


def test_ai_factory_import_and_availability_flags_do_not_eager_load_optional_modules() -> (
    None
):
    """
    Verifies ai_factory import and legacy availability flags avoid eager optional module imports.

    Args:
        None

    Returns:
        None
    """
    str_python_code: str = """
import json
import sys
import ai_api_unified.ai_factory as ai_factory

list_str_optional_modules = [
    "openai",
    "boto3",
    "botocore",
    "google",
    "google.genai",
    "google.api_core",
    "google.cloud",
    "azure",
    "azure.cognitiveservices",
    "elevenlabs",
    "ai_api_unified.ai_openai_base",
    "ai_api_unified.ai_google_base",
    "ai_api_unified.ai_bedrock_base",
    "ai_api_unified.completions.ai_openai_completions",
    "ai_api_unified.completions.ai_google_gemini_completions",
    "ai_api_unified.completions.ai_bedrock_completions",
    "ai_api_unified.embeddings.ai_openai_embeddings",
    "ai_api_unified.embeddings.ai_google_gemini_embeddings",
    "ai_api_unified.embeddings.ai_titan_embeddings",
    "ai_api_unified.images.ai_openai_images",
    "ai_api_unified.images.ai_google_gemini_images",
    "ai_api_unified.images.ai_bedrock_images",
    "ai_api_unified.voice.ai_voice_openai",
    "ai_api_unified.voice.ai_voice_google",
    "ai_api_unified.voice.ai_voice_azure",
    "ai_api_unified.voice.ai_voice_elevenlabs",
]

dict_before: dict[str, bool] = {
    str_module_name: str_module_name in sys.modules
    for str_module_name in list_str_optional_modules
}

dict_flag_values: dict[str, bool] = {
    "GOOGLE_GEMINI_AVAILABLE": ai_factory.GOOGLE_GEMINI_AVAILABLE,
    "BEDROCK_AVAILABLE": ai_factory.BEDROCK_AVAILABLE,
    "TITAN_AVAILABLE": ai_factory.TITAN_AVAILABLE,
    "BEDROCK_IMAGES_AVAILABLE": ai_factory.BEDROCK_IMAGES_AVAILABLE,
}

dict_after: dict[str, bool] = {
    str_module_name: str_module_name in sys.modules
    for str_module_name in list_str_optional_modules
}

print(json.dumps({
    "before": dict_before,
    "after": dict_after,
    "all_flags_are_bool": all(isinstance(bool_value, bool) for bool_value in dict_flag_values.values()),
}))
"""
    dict_probe_results: dict[str, Any] = _run_python_probe(str_python_code)
    object_before: object = dict_probe_results["before"]
    object_after: object = dict_probe_results["after"]
    object_flags_are_bool: object = dict_probe_results["all_flags_are_bool"]

    if not isinstance(object_before, dict):
        raise AssertionError("Expected 'before' to be a dictionary.")
    if not isinstance(object_after, dict):
        raise AssertionError("Expected 'after' to be a dictionary.")
    if not isinstance(object_flags_are_bool, bool):
        raise AssertionError("Expected 'all_flags_are_bool' to be a bool.")

    dict_before: dict[str, bool] = object_before
    dict_after: dict[str, bool] = object_after
    bool_all_flags_are_bool: bool = object_flags_are_bool

    for bool_loaded in dict_before.values():
        assert bool_loaded is False
    for bool_loaded in dict_after.values():
        assert bool_loaded is False
    assert bool_all_flags_are_bool is True


def test_completions_package_import_does_not_eager_load_openai_completions() -> None:
    """
    Verifies completions package import does not eagerly import optional OpenAI completions module.

    Args:
        None

    Returns:
        None
    """
    str_python_code: str = """
import json
import sys
import ai_api_unified.completions
print(json.dumps({
    "openai_completions": "ai_api_unified.completions.ai_openai_completions" in sys.modules
}))
"""
    dict_probe_results: dict[str, bool] = _run_python_probe(str_python_code)
    assert dict_probe_results["openai_completions"] is False


def test_embeddings_package_import_does_not_eager_load_openai_embeddings() -> None:
    """
    Verifies embeddings package import does not eagerly import optional OpenAI embeddings module.

    Args:
        None

    Returns:
        None
    """
    str_python_code: str = """
import json
import sys
import ai_api_unified.embeddings
print(json.dumps({
    "openai_embeddings": "ai_api_unified.embeddings.ai_openai_embeddings" in sys.modules
}))
"""
    dict_probe_results: dict[str, bool] = _run_python_probe(str_python_code)
    assert dict_probe_results["openai_embeddings"] is False


def test_images_package_import_does_not_eager_load_openai_images() -> None:
    """
    Verifies images package import does not eagerly import optional OpenAI images module.

    Args:
        None

    Returns:
        None
    """
    str_python_code: str = """
import json
import sys
import ai_api_unified.images
print(json.dumps({
    "openai_images": "ai_api_unified.images.ai_openai_images" in sys.modules
}))
"""
    dict_probe_results: dict[str, bool] = _run_python_probe(str_python_code)
    assert dict_probe_results["openai_images"] is False


def test_voice_package_import_does_not_eager_load_openai_voice() -> None:
    """
    Verifies voice package import does not eagerly import optional OpenAI voice module.

    Args:
        None

    Returns:
        None
    """
    str_python_code: str = """
import json
import sys
import ai_api_unified.voice
print(json.dumps({
    "openai_voice": "ai_api_unified.voice.ai_voice_openai" in sys.modules
}))
"""
    dict_probe_results: dict[str, bool] = _run_python_probe(str_python_code)
    assert dict_probe_results["openai_voice"] is False


def test_root_star_import_does_not_eager_load_optional_provider_modules() -> None:
    """
    Verifies root star import succeeds without loading optional provider implementation modules.

    Args:
        None

    Returns:
        None
    """
    str_python_code: str = """
import json
from ai_api_unified import *  # noqa: F401,F403
print(json.dumps({
    "google_completions_exported": "GoogleGeminiCompletions" in globals(),
    "bedrock_completions_exported": "AiBedrockCompletions" in globals(),
    "titan_embeddings_exported": "AiTitanEmbeddings" in globals(),
    "nova_images_exported": "AINovaCanvasImages" in globals(),
    "google_voice_exported": "AIVoiceGoogle" in globals()
}))
"""
    dict_probe_results: dict[str, bool] = _run_python_probe(str_python_code)

    assert dict_probe_results["google_completions_exported"] is False
    assert dict_probe_results["bedrock_completions_exported"] is False
    assert dict_probe_results["titan_embeddings_exported"] is False
    assert dict_probe_results["nova_images_exported"] is False
    assert dict_probe_results["google_voice_exported"] is False


def test_completions_star_import_does_not_eager_load_optional_providers() -> None:
    """
    Verifies completions star import succeeds without loading optional provider modules.

    Args:
        None

    Returns:
        None
    """
    str_python_code: str = """
import json
from ai_api_unified.completions import *  # noqa: F401,F403
print(json.dumps({
    "openai_completions_exported": "AiOpenAICompletions" in globals(),
    "google_completions_exported": "GoogleGeminiCompletions" in globals(),
    "bedrock_completions_exported": "AiBedrockCompletions" in globals()
}))
"""
    dict_probe_results: dict[str, bool] = _run_python_probe(str_python_code)

    assert dict_probe_results["openai_completions_exported"] is False
    assert dict_probe_results["google_completions_exported"] is False
    assert dict_probe_results["bedrock_completions_exported"] is False


def test_embeddings_star_import_does_not_eager_load_optional_providers() -> None:
    """
    Verifies embeddings star import succeeds without loading optional provider modules.

    Args:
        None

    Returns:
        None
    """
    str_python_code: str = """
import json
from ai_api_unified.embeddings import *  # noqa: F401,F403
print(json.dumps({
    "openai_embeddings_exported": "AiOpenAIEmbeddings" in globals(),
    "google_embeddings_exported": "GoogleGeminiEmbeddings" in globals(),
    "titan_embeddings_exported": "AiTitanEmbeddings" in globals()
}))
"""
    dict_probe_results: dict[str, bool] = _run_python_probe(str_python_code)

    assert dict_probe_results["openai_embeddings_exported"] is False
    assert dict_probe_results["google_embeddings_exported"] is False
    assert dict_probe_results["titan_embeddings_exported"] is False


def test_images_star_import_does_not_eager_load_optional_providers() -> None:
    """
    Verifies images star import succeeds without loading optional provider modules.

    Args:
        None

    Returns:
        None
    """
    str_python_code: str = """
import json
from ai_api_unified.images import *  # noqa: F401,F403
print(json.dumps({
    "openai_images_exported": "AIOpenAIImages" in globals(),
    "bedrock_images_exported": "AINovaCanvasImages" in globals()
}))
"""
    dict_probe_results: dict[str, bool] = _run_python_probe(str_python_code)

    assert dict_probe_results["openai_images_exported"] is False
    assert dict_probe_results["bedrock_images_exported"] is False


def test_voice_star_import_does_not_eager_load_optional_providers() -> None:
    """
    Verifies voice star import succeeds without loading optional provider modules.

    Args:
        None

    Returns:
        None
    """
    str_python_code: str = """
import json
from ai_api_unified.voice import *  # noqa: F401,F403
print(json.dumps({
    "openai_voice_exported": "AIVoiceOpenAI" in globals(),
    "google_voice_exported": "AIVoiceGoogle" in globals(),
    "azure_voice_exported": "AIVoiceAzure" in globals(),
    "elevenlabs_voice_exported": "AIVoiceElevenLabs" in globals()
}))
"""
    dict_probe_results: dict[str, bool] = _run_python_probe(str_python_code)

    assert dict_probe_results["openai_voice_exported"] is False
    assert dict_probe_results["google_voice_exported"] is False
    assert dict_probe_results["azure_voice_exported"] is False
    assert dict_probe_results["elevenlabs_voice_exported"] is False
