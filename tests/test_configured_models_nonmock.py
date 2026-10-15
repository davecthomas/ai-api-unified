# tests/test_configured_models_nonmock.py
"""
Live check of the models YOUR configuration names: the completions LLM,
COMPLETIONS_FALLBACKS, every router tier in COMPLETIONS_ROUTER_CONFIG, and
the router's classifier. Fails on any model a provider reports missing.

Run with:

    poetry run pytest -m nonmock tests/test_configured_models_nonmock.py -q -s
"""

from __future__ import annotations


import pytest

from ai_api_unified import AIFactory
from ai_api_unified.completions.ai_fallback_completions import (
    AIFallbackCandidate,
    parse_fallback_candidates,
)
from ai_api_unified.routing.router_config import load_router_config
from ai_api_unified.util import ai_model_probes  # noqa: F401
from ai_api_unified.util.ai_model_checker import AiModelChecker, AiModelCheckStatus
from ai_api_unified.util.env_settings import EnvSettings

pytestmark = pytest.mark.nonmock

TUPLE_CONFIG_KEYS: tuple[str, ...] = (
    "COMPLETIONS_ROUTER_CONFIG",
    "COMPLETIONS_ROUTER_CLASSIFIER",
    "COMPLETIONS_FALLBACKS",
    "COMPLETIONS_FALLBACK_ON",
    "AI_MODEL_CHECK",
)


def test_every_configured_model_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    # The suite blanks these so mocked tests never see a developer's config;
    # this test exists to read exactly that config, from .env.
    for str_key in TUPLE_CONFIG_KEYS:
        monkeypatch.delenv(str_key, raising=False)
    env = EnvSettings()
    str_engine = str(env.get_setting("COMPLETIONS_ENGINE", "") or "")
    if not str_engine:
        pytest.skip("Skipping: COMPLETIONS_ENGINE not set")
    list_candidates: list[AIFallbackCandidate] = [
        AIFallbackCandidate(
            engine=str_engine,
            model=str(env.get_setting("COMPLETIONS_MODEL_NAME", "") or ""),
        )
    ]
    list_candidates += parse_fallback_candidates(
        str(env.get_setting("COMPLETIONS_FALLBACKS", "") or "")
    )
    str_router = str(env.get_setting("COMPLETIONS_ROUTER_CONFIG", "") or "").strip()
    if str_router:
        config = load_router_config(str_router)
        for tier in config.tiers:
            list_candidates += tier.candidates
        classifier = AIFactory._router_classifier_candidate(
            env_settings=env, router_config=config
        )
        if classifier is not None:
            list_candidates.append(classifier)
    AiModelChecker.reset_cache()
    list_results = AiModelChecker().check_all(list_candidates)
    print()
    for result in list_results:
        print(f"  {result.label:50} {result.status.value:13} {result.detail}")
    list_missing = [r for r in list_results if r.status is AiModelCheckStatus.MISSING]
    assert not list_missing, "configured models not available: " + ", ".join(
        f"{r.label} ({r.detail})" for r in list_missing
    )
    assert any(
        r.status is AiModelCheckStatus.AVAILABLE for r in list_results
    ), "nothing was verifiable"
