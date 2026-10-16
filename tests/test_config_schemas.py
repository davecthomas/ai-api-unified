# tests/test_config_schemas.py
"""
The published JSON Schemas: the shipped files match what the pydantic
models generate, every config file the library loads passes them, they
reject what the library rejects, and the validate command reports both.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

jsonschema = pytest.importorskip("jsonschema")

from ai_api_unified import config_schema  # noqa: E402
from ai_api_unified.ai_models_config import AIModelsConfig  # noqa: E402

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
LIST_MODELS_FILES: list[Path] = [
    REPO_ROOT / "models_sample.json",
    REPO_ROOT / "tests" / "fixtures" / "models_test.json",
    REPO_ROOT / "tests" / "fixtures" / "models_missing_test.json",
]
PATH_MIDDLEWARE_EXAMPLE: Path = (
    REPO_ROOT / "docs" / "observability_middleware_example.yaml"
)


def _validator(dict_schema: dict[str, Any]) -> Any:
    jsonschema.Draft202012Validator.check_schema(dict_schema)
    return jsonschema.Draft202012Validator(dict_schema)


@pytest.mark.parametrize(
    ("str_file", "callable_schema"),
    [
        (config_schema.SCHEMA_FILE_MODELS, config_schema.models_config_schema),
        (config_schema.SCHEMA_FILE_MIDDLEWARE, config_schema.middleware_config_schema),
    ],
)
def test_shipped_schema_matches_the_models(str_file: str, callable_schema: Any) -> None:
    path: Path = config_schema.PATH_SCHEMAS_DIR / str_file
    assert json.loads(path.read_text()) == callable_schema(), (
        f"{str_file} is stale; run: poetry run python -m "
        "ai_api_unified.config_schema write"
    )


@pytest.mark.parametrize("path", LIST_MODELS_FILES, ids=lambda p: p.name)
def test_every_models_file_passes_the_schema(path: Path) -> None:
    validator = _validator(config_schema.models_config_schema())
    list_errors = [
        e.message for e in validator.iter_errors(json.loads(path.read_text()))
    ]
    assert list_errors == []
    AIModelsConfig.model_validate(json.loads(path.read_text()))


def test_every_accepted_spelling_passes() -> None:
    validator = _validator(config_schema.models_config_schema())
    dict_config: dict[str, Any] = {
        "completions": {
            "engine": "claude",
            "fallbacks": ["openai:gpt-5.4-mini"],
            "fallback_on": "unavailable,rate_limited",
            "cooldown": False,
            "thinking": "low",
        },
        "router": {
            "fallback_on": ["unavailable"],
            "tiers": [
                {
                    "name": "easy",
                    "models": ["claude:a", {"engine": "openai", "model": "b"}],
                },
                {"name": "hard", "candidates": [{"engine": "claude", "model": "c"}]},
            ],
        },
    }
    assert [e.message for e in validator.iter_errors(dict_config)] == []
    AIModelsConfig.model_validate(dict_config)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.update({"completion": {}}),
        lambda c: c["completions"].update({"thinking": "lots"}),
        lambda c: c["completions"].update({"require": {"capabilities": ["x"]}}),
        lambda c: c["router"]["tiers"][0].pop("models"),
        lambda c: c["model_check"].update({"mode": "sometimes"}),
    ],
    ids=[
        "unknown-section",
        "bad-thinking",
        "bad-capability",
        "tier-without-models",
        "bad-mode",
    ],
)
def test_schema_and_loader_reject_the_same_mistakes(mutate: Any) -> None:
    dict_config: dict[str, Any] = copy.deepcopy(
        json.loads((REPO_ROOT / "models_sample.json").read_text())
    )
    mutate(dict_config)
    validator = _validator(config_schema.models_config_schema())
    assert list(validator.iter_errors(dict_config)), "schema accepted it"
    with pytest.raises(ValueError):
        AIModelsConfig.model_validate(dict_config)


def test_middleware_example_passes_and_a_wrong_name_fails() -> None:
    validator = _validator(config_schema.middleware_config_schema())
    dict_example: dict[str, Any] = yaml.safe_load(PATH_MIDDLEWARE_EXAMPLE.read_text())
    assert [e.message for e in validator.iter_errors(dict_example)] == []
    dict_bad: dict[str, Any] = {"middleware": [{"name": "pii", "enabled": True}]}
    assert list(validator.iter_errors(dict_bad))


def test_validate_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path_bad: Path = tmp_path / "bad.json"
    path_bad.write_text(json.dumps({"completions": {"thinking": "lots"}}))
    assert (
        config_schema.main(
            [
                "validate",
                str(REPO_ROOT / "models_sample.json"),
                str(PATH_MIDDLEWARE_EXAMPLE),
            ]
        )
        == 0
    )
    assert config_schema.main(["validate", str(path_bad)]) == 1
    str_out: str = capsys.readouterr().out
    assert "models_sample.json: valid" in str_out and "bad.json: INVALID" in str_out
    assert config_schema.main(["nonsense"]) == 2
