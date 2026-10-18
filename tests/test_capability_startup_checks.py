# tests/test_capability_startup_checks.py
"""
The factory checks the configured embeddings, images, and videos model at
build, as it checks the completions primary, and stamps the configuration
identity on every client it builds, voice and classifier included.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import Mock, patch

import pytest

from ai_api_unified import AIFactory
from ai_api_unified.ai_base import AIBase
from ai_api_unified.ai_provider_exceptions import AiProviderConfigurationError
from ai_api_unified.util.ai_model_checker import AiModelChecker
from ai_api_unified.util.ai_model_checker import AiModelCheckStatus as S
from ai_api_unified.voice.ai_voice_base import AIVoiceBase


class _FakeClient:
    """Stands in for a provider class; carries the base identity methods."""

    set_config_identity = AIBase.set_config_identity
    config_id = AIBase.config_id
    config_version = AIBase.config_version
    _dict_config_tags: Any = {}

    def __init__(self, model: Any = None, dimensions: int = 0, **_: Any) -> None:
        self.model = model
        self.dimensions = dimensions


@pytest.fixture(autouse=True)
def _fake_probes() -> Iterator[list[tuple[str, str]]]:
    """openai and bedrock answer from the model id: 'gone' missing, 'badkey' unreachable."""
    dict_saved = dict(AiModelChecker._DICT_PROBE_BY_ENGINE)
    AiModelChecker.reset_cache()
    list_calls: list[tuple[str, str]] = []

    def _probe_for(str_engine: str) -> Any:
        def _probe(model: str, base_url: str | None) -> Any:
            list_calls.append((str_engine, model))
            if "gone" in model:
                return S.MISSING, "fake 404"
            if "badkey" in model:
                return S.UNREACHABLE, "fake 401"
            return S.AVAILABLE, "ok"

        return _probe

    for token in ("openai", "bedrock", "google-gemini"):
        AiModelChecker.register_probe(token, _probe_for(token))
    yield list_calls
    AiModelChecker._DICT_PROBE_BY_ENGINE.clear()
    AiModelChecker._DICT_PROBE_BY_ENGINE.update(dict_saved)
    AiModelChecker.reset_cache()


def _env(**overrides: str) -> Mock:
    dict_settings: dict[str, str] = {
        "AI_MODEL_CHECK": "prune",
        "EMBEDDING_ENGINE": "openai",
        "EMBEDDING_MODEL_NAME": "text-embedding-3-small",
        "EMBEDDING_DIMENSIONS": "1536",
        "IMAGE_ENGINE": "openai",
        "IMAGE_MODEL_NAME": "gpt-image-1",
        "VIDEO_ENGINE": "openai",
        "VIDEO_MODEL_NAME": "sora-2",
        **overrides,
    }
    env = Mock()
    env.get_setting.side_effect = lambda key, default=None: dict_settings.get(
        key, default
    )
    env.get_models_config.return_value = None
    return env


def _build(builder: str, env: Mock, **kwargs: Any) -> Any:
    with (
        patch("ai_api_unified.ai_factory.EnvSettings", return_value=env),
        patch("ai_api_unified.ai_factory.get_ai_provider_spec", return_value=Mock()),
        patch(
            "ai_api_unified.ai_factory.load_ai_provider_class", return_value=_FakeClient
        ),
    ):
        return getattr(AIFactory, builder)(**kwargs)


class TestModelCheck:
    @pytest.mark.parametrize(
        ("builder", "setting", "expected"),
        [
            ("get_ai_embedding_client", "EMBEDDING_MODEL_NAME", "embeddings.model"),
            ("get_ai_images_client", "IMAGE_MODEL_NAME", "images.model"),
            ("get_ai_video_client", "VIDEO_MODEL_NAME", "videos.model"),
        ],
    )
    def test_missing_model_stops_the_build_and_names_the_setting(
        self, builder: str, setting: str, expected: str
    ) -> None:
        with pytest.raises(ValueError, match=expected.replace(".", r"\.")):
            _build(builder, _env(**{setting: "model-gone"}))

    def test_unreachable_model_raises_configuration_error(self) -> None:
        with pytest.raises(AiProviderConfigurationError, match="cannot be reached"):
            _build("get_ai_images_client", _env(IMAGE_MODEL_NAME="badkey-model"))

    def test_available_model_builds(self, _fake_probes: list[tuple[str, str]]) -> None:
        client = _build("get_ai_embedding_client", _env())
        assert client.model == "text-embedding-3-small"
        assert ("openai", "text-embedding-3-small") in _fake_probes

    def test_bedrock_media_engines_use_the_bedrock_probe(
        self, _fake_probes: list[tuple[str, str]]
    ) -> None:
        _build(
            "get_ai_embedding_client",
            _env(EMBEDDING_ENGINE="titan", EMBEDDING_MODEL_NAME="amazon.titan-v2"),
        )
        assert ("bedrock", "amazon.titan-v2") in _fake_probes

    def test_engine_without_a_probe_is_not_checked(
        self, _fake_probes: list[tuple[str, str]]
    ) -> None:
        _build(
            "get_ai_embedding_client",
            _env(EMBEDDING_ENGINE="voyage", EMBEDDING_MODEL_NAME="voyage-gone"),
        )
        assert _fake_probes == []

    def test_engine_default_model_is_not_checked(
        self, _fake_probes: list[tuple[str, str]]
    ) -> None:
        _build("get_ai_images_client", _env(IMAGE_MODEL_NAME=""))
        assert _fake_probes == []

    def test_mode_off_skips_the_check(self) -> None:
        _build(
            "get_ai_images_client",
            _env(AI_MODEL_CHECK="off", IMAGE_MODEL_NAME="model-gone"),
        )


class TestIdentity:
    def test_media_clients_carry_a_config_id_that_tracks_the_model(self) -> None:
        first = _build("get_ai_embedding_client", _env())
        again = _build("get_ai_embedding_client", _env())
        other = _build(
            "get_ai_embedding_client",
            _env(EMBEDDING_MODEL_NAME="text-embedding-3-large"),
        )
        assert first.config_id and first.config_id == again.config_id
        assert other.config_id != first.config_id
        images = _build("get_ai_images_client", _env())
        assert images.config_id and images.config_id != first.config_id

    def test_media_calls_carry_the_identity_tags(self) -> None:
        from ai_api_unified.middleware.request_cost import request_cost_scope

        client = _FakeClient()
        client.set_config_identity("abc", "v1", "svc")
        with request_cost_scope("x", dict_default_tags=client._dict_config_tags) as s:
            pass
        assert s.report.tags["config_id"] == "abc"

    def test_voice_client_gets_identity_and_tags_its_calls(self) -> None:
        class _Voice(AIVoiceBase):
            def text_to_voice(self, *a: Any, **k: Any) -> bytes:
                return b""

            def stream_audio(self, *a: Any, **k: Any) -> Any:
                return None

            def speech_to_text(self, *a: Any, **k: Any) -> str:
                return ""

        voice = _Voice.model_construct()
        voice.set_config_identity("vid", "v2", "svc")
        assert voice.config_id == "vid"
        seen: dict[str, str] = {}

        def _execute() -> bytes:
            from ai_api_unified.middleware.observability_runtime import (
                get_observability_context,
            )

            seen.update(get_observability_context().tags)
            return b"x"

        with (
            patch.object(
                AIVoiceBase,
                "_build_observability_call_context",
                return_value=Mock(),
            ),
            patch(
                "ai_api_unified.voice.ai_voice_base.execute_observed_call",
                side_effect=lambda **kw: kw["callable_execute"](),
            ),
        ):
            voice._execute_voice_call_with_observability(
                operation="text_to_voice",
                dict_input_metadata=None,
                callable_execute=_execute,
                callable_build_result_summary=Mock(),
            )
        assert seen["config_id"] == "vid" and seen["service"] == "svc"
