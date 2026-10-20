# tests/test_factory_config_errors.py
"""
Issue #47: a configuration error a provider class raises while it is built
(a retired model, a missing key) reaches the caller with its own message,
not as "Unsupported <capability> engine". A genuinely unknown engine still
gets the legacy message. Both are ValueErrors, as this path always raised.
"""

from __future__ import annotations

from typing import Any, Callable
from unittest.mock import patch

import pytest

from ai_api_unified import AIFactory, AiProviderConfigurationValueError
from ai_api_unified.ai_provider_exceptions import AiProviderConfigurationError

MESSAGE: str = (
    "Model 'gemini-2.0-flash' (google) is retired; use 'gemini-2.5-flash' instead."
)


class _Retired:
    """A provider class whose constructor rejects its configuration."""

    def __init__(self, **_: Any) -> None:
        raise AiProviderConfigurationError(MESSAGE)


BUILDERS: list[tuple[str, Callable[[], Any]]] = [
    (
        "completions",
        lambda: AIFactory._build_completions_client(
            "google-gemini", "gemini-2.0-flash", None
        ),
    ),
    (
        "embeddings",
        lambda: AIFactory.get_ai_embedding_client(
            embedding_engine="google-gemini", model_name="m"
        ),
    ),
    (
        "classification",
        lambda: AIFactory.get_ai_classifier_client(
            classifier_engine="google-gemini", model_name="m"
        ),
    ),
    ("images", lambda: AIFactory.get_ai_images_client(image_model="m")),
    ("videos", lambda: AIFactory.get_ai_video_client(model_name="m")),
]


@pytest.fixture(autouse=True)
def _engines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_ENGINE", "google-gemini")
    monkeypatch.setenv("VIDEO_ENGINE", "google-gemini")


@pytest.mark.parametrize(("str_capability", "build"), BUILDERS)
def test_a_construction_error_keeps_its_own_message(
    str_capability: str, build: Callable[[], Any]
) -> None:
    with patch(
        "ai_api_unified.ai_factory.load_ai_provider_class", return_value=_Retired
    ):
        with pytest.raises(AiProviderConfigurationValueError) as raised:
            build()
    assert MESSAGE in str(raised.value)
    assert "google-gemini" in str(raised.value) and str_capability in str(raised.value)
    assert "Unsupported" not in str(raised.value)
    # Still both of the types a caller may already catch.
    assert isinstance(raised.value, ValueError)
    assert isinstance(raised.value, AiProviderConfigurationError)
    assert isinstance(raised.value.__cause__, AiProviderConfigurationError)


def test_an_unknown_engine_keeps_the_legacy_message() -> None:
    with pytest.raises(
        ValueError, match="Unsupported COMPLETIONS engine: 'nope'"
    ) as raised:
        AIFactory._build_completions_client("nope", "m", None)
    assert not isinstance(raised.value, AiProviderConfigurationValueError)
    with pytest.raises(ValueError, match="Unsupported EMBEDDING engine"):
        AIFactory.get_ai_embedding_client(embedding_engine="nope", model_name="m")
