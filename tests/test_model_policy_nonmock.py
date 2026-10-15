# tests/test_model_policy_nonmock.py
"""
Live proof that a models.json deny list bites, against the real providers.

The test reads tests/fixtures/models_test.json by default, whose hard tier
deliberately lists a model its own deny list rejects. Point it at any file:

    poetry run pytest -m nonmock tests/test_model_policy_nonmock.py -q -s
    poetry run pytest -m nonmock tests/test_model_policy_nonmock.py -q -s --models-config ./models.json

With your own file the test injects a known denied model into the top tier
(in memory, never on disk) when the file does not already contain one.
"""

from __future__ import annotations

import logging

import pytest

from ai_api_unified import AIFactory, AiRouterCompletions
from ai_api_unified.ai_models_config import AIModelsConfig, load_models_config
from ai_api_unified.completions.ai_fallback_completions import AIFallbackCandidate
from ai_api_unified.routing.router_config import AIRouterConfig

pytestmark = pytest.mark.nonmock

LOGGER_FACTORY: str = "ai_api_unified.ai_factory"

# Known ids across providers, so a deny pattern can be matched to a concrete
# model without guessing from the glob.
TUPLE_KNOWN_MODELS: tuple[tuple[str, str], ...] = (
    ("claude", "claude-fable-5-1"),
    ("claude", "claude-fable-5"),
    ("claude", "claude-opus-5-5"),
    ("openai", "gpt-5.6-luna"),
    ("openai", "gpt-4.1-mini"),
    ("openai-compatible", "moonshotai/kimi-k2"),
    ("bedrock", "moonshot.kimi-k2-v1:0"),
    ("google-gemini", "gemini-2.5-flash"),
    ("nova", "amazon.nova-lite-v1:0"),
)
TUPLE_ENV_OVERRIDES: tuple[str, ...] = (
    "COMPLETIONS_ROUTER_CONFIG",
    "COMPLETIONS_ROUTER_CLASSIFIER",
    "COMPLETIONS_FALLBACKS",
    "COMPLETIONS_FALLBACK_ON",
    "AI_MODEL_CHECK",
    "COMPLETIONS_ENGINE",
    "COMPLETIONS_MODEL_NAME",
)


@pytest.fixture
def config(models_config_path: str, monkeypatch: pytest.MonkeyPatch) -> AIModelsConfig:
    """The file under test, with the suite's env overrides lifted so it rules."""
    for str_key in TUPLE_ENV_OVERRIDES:
        monkeypatch.delenv(str_key, raising=False)
    monkeypatch.setenv("AI_MODELS_CONFIG", models_config_path)
    config = load_models_config(models_config_path)
    if not config.deny:
        pytest.skip(f"Skipping: {models_config_path} has no deny patterns")
    if config.router is None:
        pytest.skip(f"Skipping: {models_config_path} has no router section")
    return config


def _router_with_a_denied_model(config: AIModelsConfig) -> tuple[AIRouterConfig, str]:
    """The file's router, holding at least one denied candidate; returns its label."""
    assert config.router is not None
    # Loop over tiers for a candidate the file already denies.
    for tier in config.router.tiers:
        for candidate in tier.candidates:
            if config.policy.denial_reason(candidate.engine, candidate.model):
                return config.router, candidate.label
    # Otherwise inject a known denied model into the top tier, in memory.
    for str_engine, str_model in TUPLE_KNOWN_MODELS:
        if config.policy.denial_reason(str_engine, str_model):
            top_tier = config.router.tiers[-1]
            injected = top_tier.model_copy(
                update={
                    "candidates": [
                        AIFallbackCandidate(engine=str_engine, model=str_model),
                        *top_tier.candidates,
                    ]
                }
            )
            return (
                config.router.model_copy(
                    update={"tiers": [*config.router.tiers[:-1], injected]}
                ),
                f"{str_engine}:{str_model}",
            )
    pytest.skip("Skipping: no known model matches the file's deny patterns")


def test_deny_list_prunes_a_denied_tier_model(
    config: AIModelsConfig, caplog: pytest.LogCaptureFixture
) -> None:
    router, str_denied = _router_with_a_denied_model(config)
    print(f"\n  denied candidate under test: {str_denied}")
    with caplog.at_level(logging.WARNING, logger=LOGGER_FACTORY):
        client = AIFactory.get_ai_completions_client(router=router)
    assert isinstance(client, AiRouterCompletions)
    list_denied = [
        r
        for r in caplog.records
        if getattr(r, "ai_model_policy_event", "") == "model_denied"
    ]
    assert list_denied, "no model_denied warning was logged"
    print(f"  {list_denied[0].getMessage()}")
    assert f"{str_denied} is denied by policy pattern" in list_denied[0].getMessage()
    for tier in client.tiers:
        assert all(
            c.label != str_denied for c in tier.candidates
        ), f"{str_denied} survived in tier {tier.name}"
    assert client.tiers, "every tier was pruned"
    print("  tiers after pruning: " + ", ".join(t.label for t in client.tiers))
    # The live check still verified the survivors.
    assert any(
        r.status.value == "available" for r in AIFactory.last_model_check_report()
    )


def test_explicit_request_for_a_denied_model_is_refused(config: AIModelsConfig) -> None:
    _, str_denied = _router_with_a_denied_model(config)
    str_engine, _, str_model = str_denied.partition(":")
    with pytest.raises(ValueError, match="denied by policy pattern"):
        AIFactory.get_ai_completions_client(
            completions_engine=str_engine, model_name=str_model
        )


def test_file_configures_nothing_it_denies_except_on_purpose(
    config: AIModelsConfig,
) -> None:
    """Your own file should not deny its primary; the fixture denies a tier model on purpose."""
    if config.completions.engine:
        assert (
            config.policy.denial_reason(
                config.completions.engine, config.completions.model or ""
            )
            is None
        )
