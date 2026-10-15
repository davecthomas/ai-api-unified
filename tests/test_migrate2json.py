# tests/test_migrate2json.py
"""scripts/migrate2json.py: .env keys land in their sections, defaults fill in, policy stays empty."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT: Path = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "migrate2json", REPO_ROOT / "scripts" / "migrate2json.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_env_keys_land_in_sections_with_sample_defaults(tmp_path: Path) -> None:
    (tmp_path / "router.json").write_text(
        json.dumps({"tiers": [{"name": "easy", "models": ["claude:claude-haiku-4-5"]}]})
    )
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=secret\nCOMPLETIONS_ENGINE=google-gemini\nCOMPLETIONS_MODEL_NAME=gemini-2.5-flash\n"
        "COMPLETIONS_FALLBACKS=claude:claude-sonnet-5, openai:gpt-5.6-luna\nCOMPLETIONS_ROUTER_CONFIG=./router.json\n"
        "COMPLETIONS_ROUTER_CLASSIFIER=openai-decisions:gpt-6-luna\nEMBEDDING_ENGINE=openai\nEMBEDDING_DIMENSIONS=1536\n"
        "AI_API_GEO_RESIDENCY=US\n"
    )
    script = _load_script()
    out = tmp_path / "models.json"
    assert (
        script.main(
            ["--env", str(tmp_path / ".env"), "--out", str(out), "--comment-env"]
        )
        == 0
    )
    payload = json.loads(out.read_text())
    assert payload["allow"] == [] and payload["deny"] == []
    assert payload["completions"]["engine"] == "google-gemini"
    assert payload["completions"]["fallbacks"] == [
        "claude:claude-sonnet-5",
        "openai:gpt-5.6-luna",
    ]
    assert payload["completions"]["fallback_on"]  # structural default from the sample
    assert payload["classifier"] == {
        "engine": "openai-decisions",
        "model": "gpt-6-luna",
    }
    assert (
        payload["router"]["tiers"][0]["name"] == "easy"
        and payload["router"]["on_exhausted"] == "escalate"
    )
    assert payload["model_check"]["mode"] == "prune"
    assert payload["embeddings"] == {"engine": "openai", "dimensions": 1536}
    assert payload["geo_residency"] == "US"
    assert "voice" not in payload and "images" not in payload
    str_env = (tmp_path / ".env").read_text()
    assert "OPENAI_API_KEY=secret" in str_env
    assert "# moved to models.json: COMPLETIONS_ENGINE=google-gemini" in str_env
    assert "# moved to models.json: COMPLETIONS_ROUTER_CONFIG=./router.json" in str_env
    # The generated file is valid for the library.
    from ai_api_unified.ai_models_config import load_models_config

    assert load_models_config(str(out)).router is not None


def test_refuses_to_overwrite_without_force_and_handles_a_bare_env(
    tmp_path: Path,
) -> None:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=secret\n")
    script = _load_script()
    out = tmp_path / "models.json"
    assert script.main(["--env", str(tmp_path / ".env"), "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert set(payload) == {"allow", "deny", "model_check"}
    assert script.main(["--env", str(tmp_path / ".env"), "--out", str(out)]) == 2
    assert (
        script.main(["--env", str(tmp_path / ".env"), "--out", str(out), "--force"])
        == 0
    )


def test_missing_env_file(tmp_path: Path) -> None:
    assert (
        _load_script().main(
            ["--env", str(tmp_path / "nope.env"), "--out", str(tmp_path / "m.json")]
        )
        == 2
    )


def test_the_repo_sample_is_the_default_source_of_defaults() -> None:
    script = _load_script()
    assert Path(script.PATH_SAMPLE) == REPO_ROOT / "models_sample.json"
    assert (
        json.loads(Path(script.PATH_SAMPLE).read_text())["model_check"]["mode"]
        == "prune"
    )
