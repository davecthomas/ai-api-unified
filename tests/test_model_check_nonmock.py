# tests/test_model_check_nonmock.py
"""
Live proof that the configured-model check prunes models that do not exist.

Reads tests/fixtures/models_missing_test.json by default: its fallback chain
and easy tier each list an id no provider serves. Against the real
providers, both must be reported missing and pruned with a warning, while
the chain and the tier survive on their remaining models. Point it at any
file; with your own, a bogus id is injected in memory, never on disk:

    poetry run pytest -m nonmock tests/test_model_check_nonmock.py -q -s
    poetry run pytest -m nonmock tests/test_model_check_nonmock.py -q -s --models-config ./models.json
"""

from __future__ import annotations

import logging

import pytest

from ai_api_unified import AIFactory, AiFallbackCompletions, AiRouterCompletions
from ai_api_unified.ai_models_config import AIModelsConfig, load_models_config
from ai_api_unified.completions.ai_fallback_completions import (
    AIFallbackCandidate,
    parse_fallback_candidates,
)
from ai_api_unified.routing.router_config import AIRouterConfig
from ai_api_unified.util.ai_model_checker import AiModelChecker, AiModelCheckStatus

pytestmark = pytest.mark.nonmock

LOGGER_CHECK: str = "ai_api_unified.util.ai_model_checker"
LOGGER_FACTORY: str = "ai_api_unified.ai_factory"
STR_BOGUS_MODEL: str = "claude-nope-9"
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
def models_config_path(request: pytest.FixtureRequest) -> str:
    """--models-config when given, else the missing-model fixture."""
    str_option: str | None = request.config.getoption("--models-config")
    from pathlib import Path

    return str_option or str(
        Path(__file__).resolve().parent / "fixtures" / "models_missing_test.json"
    )


@pytest.fixture
def config(models_config_path: str, monkeypatch: pytest.MonkeyPatch) -> AIModelsConfig:
    for str_key in TUPLE_ENV_OVERRIDES:
        monkeypatch.delenv(str_key, raising=False)
    monkeypatch.setenv("AI_MODELS_CONFIG", models_config_path)
    monkeypatch.setenv("AI_MODEL_CHECK", "prune")
    AiModelChecker.reset_cache()
    config = load_models_config(models_config_path)
    if not config.completions.engine or config.router is None:
        pytest.skip(
            f"Skipping: {models_config_path} needs a completions engine and a router"
        )
    return config


def _with_a_bogus_model(
    config: AIModelsConfig,
) -> tuple[list[AIFallbackCandidate], AIRouterConfig, str]:
    """The file's chain and router, each holding a model no provider serves."""
    assert config.router is not None
    list_fallbacks = parse_fallback_candidates(",".join(config.completions.fallbacks))
    bogus = AIFallbackCandidate(engine="claude", model=STR_BOGUS_MODEL)
    if not any("nope" in c.model for c in list_fallbacks):
        list_fallbacks = [bogus, *list_fallbacks]
    router = config.router
    if not any("nope" in c.model for tier in router.tiers for c in tier.candidates):
        first = router.tiers[0]
        router = router.model_copy(
            update={
                "tiers": [
                    first.model_copy(update={"candidates": [bogus, *first.candidates]}),
                    *router.tiers[1:],
                ]
            }
        )
    return list_fallbacks, router, bogus.label


def test_missing_models_are_pruned_and_the_configuration_still_builds(
    config: AIModelsConfig, caplog: pytest.LogCaptureFixture
) -> None:
    list_fallbacks, router, str_bogus = _with_a_bogus_model(config)
    with caplog.at_level(logging.WARNING):
        client = AIFactory.get_ai_completions_client(
            completions_engine=config.completions.engine,
            model_name=config.completions.model,
            fallbacks=list_fallbacks,
            router=router,
        )
    report = AIFactory.last_model_check_report()
    print()
    for result in report:
        print(f"  {result.label:45} {result.status.value:13} {result.detail}")
    list_missing = [r for r in report if r.status is AiModelCheckStatus.MISSING]
    assert list_missing, "no configured model was reported missing"
    assert all("nope" in r.model for r in list_missing), [r.label for r in list_missing]
    list_warned = [
        r.getMessage()
        for r in caplog.records
        if getattr(r, "ai_model_check_event", "") == "model_missing"
    ]
    assert list_warned, "no model_missing warning was logged"
    print("  " + "\n  ".join(list_warned))
    assert any("is not available and will be ignored" in m for m in list_warned)
    assert isinstance(client, AiRouterCompletions)
    for tier in client.tiers:
        assert all(
            "nope" not in c.model for c in tier.candidates
        ), f"a missing model survived in tier {tier.name}"
    if isinstance(client.default_client, AiFallbackCompletions):
        assert all("nope" not in c.model for c in client.default_client.candidates)
        print(
            "  chain after pruning: "
            + ", ".join(c.label for c in client.default_client.candidates)
        )
    else:
        # Every fallback was pruned, so the default route is the plain engine.
        print(
            f"  chain after pruning: {client.default_client.model_name} (no fallbacks left)"
        )
    print("  tiers after pruning: " + ", ".join(t.label for t in client.tiers))
    assert any(r.status is AiModelCheckStatus.AVAILABLE for r in report)


def test_a_missing_primary_is_refused(config: AIModelsConfig) -> None:
    with pytest.raises(ValueError, match="not available and cannot be pruned"):
        AIFactory.get_ai_completions_client(
            completions_engine="claude",
            model_name=STR_BOGUS_MODEL,
            fallbacks=[("claude", "claude-haiku-4-5")],
        )


def test_models_are_checked_once_per_process(config: AIModelsConfig) -> None:
    list_fallbacks, router, _ = _with_a_bogus_model(config)
    AIFactory.get_ai_completions_client(
        completions_engine=config.completions.engine,
        model_name=config.completions.model,
        fallbacks=list_fallbacks,
        router=router,
    )
    int_before = len(AiModelChecker._DICT_CACHE)
    AIFactory.get_ai_completions_client(
        completions_engine=config.completions.engine,
        model_name=config.completions.model,
        fallbacks=list_fallbacks,
        router=router,
    )
    assert (
        len(AiModelChecker._DICT_CACHE) == int_before
    ), "the second build probed again"
