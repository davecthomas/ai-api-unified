# tests/test_config_checks.py
"""
Config identity (config_id, config_version) and the startup requirements
check (completions.require): the hash, the tags every request carries, the
factory stamping every client, and capability and second-provider checks.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from ai_api_unified import AIConfigRequirements, AIFactory
from ai_api_unified.ai_base import AIBase
from ai_api_unified.ai_models_config import clear_models_config_cache
from ai_api_unified.completions.ai_fallback_completions import AIFallbackCandidate
from ai_api_unified.middleware.observability_runtime import (
    get_observability_context,
    reset_observability_context,
    set_observability_context,
)
from ai_api_unified.routing.config_checks import (
    compute_config_id,
    enforce_requirements,
    parse_config_requirements,
    provider_of,
    unmet_requirements,
)

LOGGER_CHECKS: str = "ai_api_unified.routing.config_checks"


def _c(engine: str, model: str, base_url: str | None = None) -> AIFallbackCandidate:
    return AIFallbackCandidate(engine=engine, model=model, base_url=base_url)


class _TagProbe(AIBase):
    """Runs one request scope and records the tags it saw."""

    def __init__(self) -> None:
        super().__init__(model="probe")
        self.tags_seen: dict[str, str] = {}

    @property
    def list_model_names(self) -> list[str]:
        return ["probe"]

    def run(self) -> None:
        def _inside() -> None:
            self.tags_seen = dict(get_observability_context().tags)

        self._with_request_cost_scope("probe", _inside)


@pytest.fixture
def models_file(tmp_path: Path) -> Any:
    def _write(dict_config: dict[str, Any]) -> Path:
        path = tmp_path / "models.json"
        path.write_text(json.dumps(dict_config))
        clear_models_config_cache()
        return path

    yield _write
    clear_models_config_cache()


class TestConfigId:
    def test_stable_and_key_order_independent(self) -> None:
        assert compute_config_id({"a": 1, "b": [1, 2]}) == compute_config_id(
            {"b": [1, 2], "a": 1}
        )
        assert len(compute_config_id({"a": 1})) == 12
        assert compute_config_id({"a": 1}) != compute_config_id({"a": 2})

    def test_request_tags_carry_the_identity_and_caller_tags_win(self) -> None:
        probe = _TagProbe()
        probe.set_config_identity("abc123", "2026-10-16")
        probe.run()
        assert probe.tags_seen["config_id"] == "abc123"
        assert probe.tags_seen["config_version"] == "2026-10-16"
        assert probe.last_request_cost is not None
        assert probe.last_request_cost.tags["config_id"] == "abc123"
        token = set_observability_context(tags={"config_version": "caller"})
        try:
            probe.run()
        finally:
            reset_observability_context(token)
        assert probe.tags_seen["config_version"] == "caller"

    def test_unstamped_client_adds_no_tags(self) -> None:
        probe = _TagProbe()
        probe.run()
        assert "config_id" not in probe.tags_seen
        assert probe.config_id is None and probe.config_version is None


class TestFactoryIdentity:
    def _build(self, **kwargs: Any) -> Any:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            return AIFactory.get_ai_completions_client(
                completions_engine="claude", model_name="claude-haiku-4-5", **kwargs
            )

    def test_every_client_gets_the_same_id_and_it_tracks_the_config(self) -> None:
        chain = self._build(fallbacks=[("claude", "claude-sonnet-5")])
        again = self._build(fallbacks=[("claude", "claude-sonnet-5")])
        other = self._build(fallbacks=[("claude", "claude-opus-5-5")])
        assert chain.config_id and chain.config_id == again.config_id
        assert chain.primary.config_id == chain.config_id
        assert other.config_id != chain.config_id
        plain = self._build(fallbacks=[])
        assert plain.config_id and plain.config_id != chain.config_id

    def test_config_version_comes_from_models_json(self, models_file: Any) -> None:
        path = models_file(
            {
                "config_version": "release-42",
                "completions": {"engine": "claude", "model": "claude-haiku-4-5"},
            }
        )
        with patch.dict(os.environ, {"AI_MODELS_CONFIG": str(path)}):
            client = self._build()
        assert client.config_version == "release-42"


class TestRequirements:
    def test_parse_and_reject_unknown_capability(self) -> None:
        req = parse_config_requirements(
            {"capabilities": ["Supports_Tool_Use", "async"]}
        )
        assert req.capabilities == ["tool_use", "async"]
        assert parse_config_requirements(None).empty
        with pytest.raises(ValueError, match="Unknown capabilities"):
            parse_config_requirements({"capabilities": ["telepathy"]})

    def test_primary_must_have_each_capability(self) -> None:
        req = AIConfigRequirements(capabilities=["structured_output", "async"])
        problems = unmet_requirements(
            req,
            [
                ("default", [_c("claude", "claude-haiku-4-5")]),
                ("easy", [_c("bedrock", "amazon.nova-lite-v1:0"), _c("claude", "x")]),
            ],
        )
        assert problems == [
            "route easy: primary bedrock:amazon.nova-lite-v1:0 lacks "
            "structured_output, async"
        ]

    def test_unknown_engine_or_default_model_is_not_a_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        req = AIConfigRequirements(capabilities=["tool_use"])
        with caplog.at_level(logging.INFO, logger=LOGGER_CHECKS):
            problems = unmet_requirements(
                req,
                [
                    ("a", [_c("openai-compatible", "llama", "http://x")]),
                    ("b", [_c("claude", "")]),
                ],
            )
        assert problems == []
        assert (
            sum("cannot check capabilities" in r.getMessage() for r in caplog.records)
            == 2
        )

    def test_second_provider(self) -> None:
        req = AIConfigRequirements(second_provider=True)
        assert unmet_requirements(
            req,
            [
                ("ok", [_c("claude", "a"), _c("openai", "b")]),
                ("bedrock-is-not-anthropic", [_c("claude", "a"), _c("anthropic", "b")]),
                ("same", [_c("openai", "a"), _c("openai-responses", "b")]),
            ],
        ) == [
            "route same: every model is on openai; add a fallback on another provider"
        ]
        assert provider_of(_c("openai-compatible", "m", "http://a")) != provider_of(
            _c("openai-compatible", "m", "http://b")
        )

    def test_warn_logs_and_strict_raises(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        routes = [("default", [_c("claude", "a")])]
        with caplog.at_level(logging.WARNING, logger=LOGGER_CHECKS):
            enforce_requirements(AIConfigRequirements(second_provider=True), routes)
        assert [getattr(r, "ai_config_event", None) for r in caplog.records] == [
            "requirement_unmet"
        ]
        with pytest.raises(ValueError, match="strict"):
            enforce_requirements(
                AIConfigRequirements(second_provider=True, mode="strict"), routes
            )

    def test_factory_checks_the_chain_and_every_tier(self) -> None:
        from ai_api_unified import AIRouterConfig

        router = AIRouterConfig.model_validate(
            {
                "tiers": [
                    {
                        "name": "easy",
                        "models": ["claude:claude-haiku-4-5", "openai:gpt-5.4-mini"],
                    },
                    {"name": "hard", "models": ["claude:claude-opus-5-5"]},
                ],
                "classifier": {"engine": "none"},
            }
        )
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with pytest.raises(ValueError, match="route hard"):
                AIFactory.get_ai_completions_client(
                    completions_engine="claude",
                    model_name="claude-haiku-4-5",
                    fallbacks=[("openai", "gpt-5.4-mini")],
                    router=router,
                    require={"second_provider": True, "mode": "strict"},
                )

    def test_models_json_require_is_validated_at_load(self, models_file: Any) -> None:
        from ai_api_unified.ai_models_config import load_models_config

        good = models_file({"completions": {"require": {"capabilities": ["tool_use"]}}})
        assert load_models_config(str(good)).completions.require.capabilities == [
            "tool_use"
        ]
        bad = models_file({"completions": {"require": {"capabilities": ["x"]}}})
        with pytest.raises(ValueError):
            load_models_config(str(bad))
