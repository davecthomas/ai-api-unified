#!/usr/bin/env python3
"""
Generates models.json from the current directory's .env.

Reads every setting in .env that is not a secret (engines, models, the
fallback chain and its reasons, the retry policy, the classifier, the router
config, the model check, embeddings, images, videos, voice, geo residency,
provider base URLs and Vertex project, the OpenAI-compatible server, the
Azure speech region, strict deprecations) and the middleware YAML that
AI_MIDDLEWARE_CONFIG_PATH names, places each in its models.json section,
and fills the structural defaults from
models_sample.json (model_check mode and timeout, fallback_on, the router's
on_exhausted). The allow and deny lists are left empty: a policy is a
decision, not a default. A router file named by COMPLETIONS_ROUTER_CONFIG
is embedded as the router section.

Usage:

    poetry run python scripts/migrate2json.py            # writes ./models.json
    poetry run python scripts/migrate2json.py --force    # overwrites an existing one
    poetry run python scripts/migrate2json.py --out cfg/models.json --env deploy.env
    poetry run python scripts/migrate2json.py --comment-env   # also comments the moved keys out of .env

Secrets (API keys, AWS credentials, GOOGLE_APPLICATION_CREDENTIALS) are
never moved; they stay in .env. AWS_REGION also stays: AWS hosts set it
themselves, and models.json's providers.aws.region is read only when the
environment leaves it unset.
"""

from __future__ import annotations

import argparse
import json
import sys

import yaml
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

PATH_HERE: Path = Path(__file__).resolve()
PATH_SAMPLE: Path = PATH_HERE.parents[1] / "models_sample.json"

# .env key -> (section, field). Value transforms are applied below.
DICT_KEY_TO_FIELD: dict[str, tuple[str, str]] = {
    "COMPLETIONS_ENGINE": ("completions", "engine"),
    "COMPLETIONS_MODEL_NAME": ("completions", "model"),
    "COMPLETIONS_FALLBACKS": ("completions", "fallbacks"),
    "COMPLETIONS_FALLBACK_ON": ("completions", "fallback_on"),
    "COMPLETIONS_RETRY_POLICY": ("completions", "retry_policy"),
    "CLASSIFIER_ENGINE": ("classifier", "engine"),
    "CLASSIFIER_MODEL_NAME": ("classifier", "model"),
    "AI_MODEL_CHECK": ("model_check", "mode"),
    "AI_MODEL_CHECK_TIMEOUT_SECONDS": ("model_check", "timeout_seconds"),
    "EMBEDDING_ENGINE": ("embeddings", "engine"),
    "EMBEDDING_MODEL_NAME": ("embeddings", "model"),
    "EMBEDDING_DIMENSIONS": ("embeddings", "dimensions"),
    "IMAGE_ENGINE": ("images", "engine"),
    "IMAGE_MODEL_NAME": ("images", "model"),
    "VIDEO_ENGINE": ("videos", "engine"),
    "VIDEO_MODEL_NAME": ("videos", "model"),
    "VIDEO_OUTPUT_DIR": ("videos", "output_dir"),
    "VIDEO_POLL_INTERVAL_SECONDS": ("videos", "poll_interval_seconds"),
    "VIDEO_TIMEOUT_SECONDS": ("videos", "timeout_seconds"),
    "AI_VOICE_ENGINE": ("voice", "engine"),
    "AI_VOICE_LANGUAGE": ("voice", "language"),
    "DEFAULT_GEMINI_TTS_MODEL": ("voice", "tts_model"),
    "BEDROCK_VIDEO_OUTPUT_S3_URI": ("videos", "bedrock_output_s3_uri"),
}
# .env key -> (provider, field) under the providers section.
DICT_PROVIDER_KEY_TO_FIELD: dict[str, tuple[str, str]] = {
    "OPENAI_BASE_URL_OVERRIDE": ("openai", "base_url"),
    "OPENAI_USER": ("openai", "user"),
    "ANTHROPIC_BASE_URL_OVERRIDE": ("anthropic", "base_url"),
    "ANTHROPIC_ADMIN_BASE_URL_OVERRIDE": ("anthropic", "admin_base_url"),
    "ANTHROPIC_USER": ("anthropic", "user"),
    "GOOGLE_AUTH_METHOD": ("google", "auth_method"),
    "GOOGLE_PROJECT_ID": ("google", "project_id"),
    "GOOGLE_LOCATION": ("google", "location"),
    "GOOGLE_GEMINI_BASE_URL_OVERRIDE": ("google", "base_url"),
    "OPENAI_COMPATIBLE_BASE_URL": ("openai_compatible", "base_url"),
    "OPENAI_COMPATIBLE_STRUCTURED_OUTPUT": ("openai_compatible", "structured_output"),
    "OPENAI_COMPATIBLE_CONTEXT_WINDOW": ("openai_compatible", "context_window"),
    "MICROSOFT_COGNITIVE_SERVICES_REGION": ("azure_speech", "region"),
}
STR_MIDDLEWARE_KEY: str = "AI_MIDDLEWARE_CONFIG_PATH"
STR_STRICT_KEY: str = "AI_STRICT_DEPRECATIONS"
STR_ROUTER_KEY: str = "COMPLETIONS_ROUTER_CONFIG"
STR_ROUTER_CLASSIFIER_KEY: str = "COMPLETIONS_ROUTER_CLASSIFIER"
STR_GEO_KEY: str = "AI_API_GEO_RESIDENCY"
FROZENSET_LIST_FIELDS: frozenset[str] = frozenset({"fallbacks"})
# (section, field) pairs the schema types as integers; every other
# *_seconds field is a float.
FROZENSET_INT_FIELDS: frozenset[tuple[str, str]] = frozenset(
    {
        ("embeddings", "dimensions"),
        ("videos", "poll_interval_seconds"),
        ("videos", "timeout_seconds"),
        ("openai_compatible", "context_window"),
    }
)
# Structural defaults copied from the sample when .env says nothing.
TUPLE_SAMPLE_DEFAULTS: tuple[tuple[str, str], ...] = (
    ("model_check", "mode"),
    ("model_check", "timeout_seconds"),
    ("completions", "fallback_on"),
    ("completions", "thinking"),
)
TUPLE_SECTION_ORDER: tuple[str, ...] = (
    "allow",
    "deny",
    "completions",
    "classifier",
    "router",
    "model_check",
    "embeddings",
    "images",
    "videos",
    "voice",
    "geo_residency",
    "providers",
    "middleware",
    "strict_deprecations",
)


def _coerce(str_section: str, str_field: str, str_value: str) -> Any:
    """Turns a .env string into the JSON type the section expects."""
    str_stripped: str = str_value.strip()
    if str_field in FROZENSET_LIST_FIELDS:
        # Early return with a list for the chain.
        return [item.strip() for item in str_stripped.split(",") if item.strip()]
    if (str_section, str_field) in FROZENSET_INT_FIELDS:
        try:
            # Early return with an integer.
            return int(float(str_stripped))
        except ValueError:
            return str_stripped
    if str_field.endswith("_seconds"):
        try:
            # Early return with a float timeout.
            return float(str_stripped)
        except ValueError:
            return str_stripped
    # Normal return with the string.
    return str_stripped


def _load_router(str_value: str, path_env_dir: Path) -> dict[str, Any] | None:
    """The router section from COMPLETIONS_ROUTER_CONFIG: inline JSON or a file."""
    str_stripped: str = str_value.strip()
    if not str_stripped:
        # Early return: no router.
        return None
    if str_stripped.startswith("{"):
        # Early return with the inline JSON.
        return json.loads(str_stripped)
    path: Path = Path(str_stripped).expanduser()
    if not path.is_absolute():
        path = path_env_dir / path
    if not path.is_file():
        print(
            f"warning: {STR_ROUTER_KEY} names {path}, which does not exist; router omitted",
            file=sys.stderr,
        )
        # Early return: nothing to embed.
        return None
    # Normal return with the file's content.
    return json.loads(path.read_text(encoding="utf-8"))


def _load_middleware(str_value: str, path_env_dir: Path) -> list[Any] | None:
    """The middleware list from the YAML that AI_MIDDLEWARE_CONFIG_PATH names."""
    str_stripped: str = str_value.strip()
    if not str_stripped:
        # Early return: no middleware file.
        return None
    path: Path = Path(str_stripped).expanduser()
    if not path.is_absolute():
        path = path_env_dir / path
    if not path.is_file():
        print(
            f"warning: {STR_MIDDLEWARE_KEY} names {path}, which does not exist; "
            "middleware omitted",
            file=sys.stderr,
        )
        # Early return: nothing to embed.
        return None
    loaded: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    list_middleware: Any = (
        loaded.get("middleware") if isinstance(loaded, dict) else None
    )
    # Normal return with the list, or None when the file holds none.
    return list_middleware if isinstance(list_middleware, list) else None


def build_models_config(
    dict_env: dict[str, str | None], *, path_env_dir: Path, dict_sample: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """
    Builds the models.json payload from .env values and sample defaults.

    Args:
        dict_env: Parsed .env (dotenv_values).
        path_env_dir: Directory the .env lives in, for relative router paths.
        dict_sample: Parsed models_sample.json.

    Returns:
        (payload, .env keys that were moved into it).
    """
    dict_out: dict[str, Any] = {"allow": [], "deny": []}
    list_moved: list[str] = []
    # Loop over the mapped keys so each present value lands in its section.
    for str_key, (str_section, str_field) in DICT_KEY_TO_FIELD.items():
        raw: str | None = dict_env.get(str_key)
        if raw is None or not str(raw).strip():
            continue
        dict_out.setdefault(str_section, {})[str_field] = _coerce(
            str_section, str_field, str(raw)
        )
        list_moved.append(str_key)
    str_router_classifier: str = str(
        dict_env.get(STR_ROUTER_CLASSIFIER_KEY) or ""
    ).strip()
    if str_router_classifier:
        str_engine, _, str_model = str_router_classifier.partition(":")
        dict_classifier: dict[str, Any] = dict_out.setdefault("classifier", {})
        dict_classifier["engine"] = str_engine.strip().lower()
        if str_model.strip():
            dict_classifier["model"] = str_model.strip()
        list_moved.append(STR_ROUTER_CLASSIFIER_KEY)
    dict_router: dict[str, Any] | None = _load_router(
        str(dict_env.get(STR_ROUTER_KEY) or ""), path_env_dir
    )
    if dict_router is not None:
        dict_router.setdefault(
            "on_exhausted",
            dict_sample.get("router", {}).get("on_exhausted", "escalate"),
        )
        dict_out["router"] = dict_router
        list_moved.append(STR_ROUTER_KEY)
    str_geo: str = str(dict_env.get(STR_GEO_KEY) or "").strip()
    if str_geo:
        dict_out["geo_residency"] = str_geo
        list_moved.append(STR_GEO_KEY)
    # Loop over provider keys so each lands under providers.<provider>.
    for str_key, (str_provider, str_field) in DICT_PROVIDER_KEY_TO_FIELD.items():
        raw_provider: str | None = dict_env.get(str_key)
        if raw_provider is None or not str(raw_provider).strip():
            continue
        dict_out.setdefault("providers", {}).setdefault(str_provider, {})[str_field] = (
            _coerce(str_provider, str_field, str(raw_provider))
        )
        list_moved.append(str_key)
    list_middleware: list[Any] | None = _load_middleware(
        str(dict_env.get(STR_MIDDLEWARE_KEY) or ""), path_env_dir
    )
    if list_middleware is not None:
        dict_out["middleware"] = list_middleware
        list_moved.append(STR_MIDDLEWARE_KEY)
    str_strict: str = str(dict_env.get(STR_STRICT_KEY) or "").strip().lower()
    if str_strict:
        dict_out["strict_deprecations"] = str_strict in {"1", "true", "yes", "on"}
        list_moved.append(STR_STRICT_KEY)
    # Loop over structural defaults the sample provides.
    for str_section, str_field in TUPLE_SAMPLE_DEFAULTS:
        if str_field not in dict_out.get(
            str_section, {}
        ) and str_field in dict_sample.get(str_section, {}):
            if str_section == "completions" and "completions" not in dict_out:
                continue
            dict_out.setdefault(str_section, {})[str_field] = dict_sample[str_section][
                str_field
            ]
    dict_ordered: dict[str, Any] = {
        str_section: dict_out[str_section]
        for str_section in TUPLE_SECTION_ORDER
        if str_section in dict_out
    }
    # Normal return with the payload and the moved keys.
    return dict_ordered, list_moved


def comment_out_keys(path_env: Path, list_keys: list[str]) -> int:
    """Comments the moved keys out of .env; returns how many lines changed."""
    list_lines: list[str] = path_env.read_text(encoding="utf-8").split("\n")
    set_keys: set[str] = set(list_keys)
    int_changed: int = 0
    # Loop over lines so each active moved key becomes a comment.
    for int_index, str_line in enumerate(list_lines):
        if "=" in str_line and not str_line.lstrip().startswith("#"):
            str_key: str = str_line.split("=", 1)[0].strip()
            if str_key in set_keys:
                list_lines[int_index] = f"# moved to models.json: {str_line}"
                int_changed += 1
    path_env.write_text("\n".join(list_lines), encoding="utf-8")
    # Normal return with the count.
    return int_changed


def main(list_argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate models.json from .env.")
    parser.add_argument(
        "--env", default=".env", help="the .env to read (default ./.env)"
    )
    parser.add_argument(
        "--out", default="models.json", help="where to write (default ./models.json)"
    )
    parser.add_argument(
        "--sample", default=str(PATH_SAMPLE), help="models_sample.json for defaults"
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing output file"
    )
    parser.add_argument(
        "--comment-env", action="store_true", help="comment the moved keys out of .env"
    )
    args = parser.parse_args(list_argv)
    path_env: Path = Path(args.env)
    path_out: Path = Path(args.out)
    if not path_env.is_file():
        print(f"error: {path_env} not found", file=sys.stderr)
        return 2
    if path_out.exists() and not args.force:
        print(f"error: {path_out} exists; pass --force to overwrite", file=sys.stderr)
        return 2
    dict_sample: dict[str, Any] = json.loads(
        Path(args.sample).read_text(encoding="utf-8")
    )
    dict_payload, list_moved = build_models_config(
        dotenv_values(path_env),
        path_env_dir=path_env.resolve().parent,
        dict_sample=dict_sample,
    )
    # Validate through the library before writing.
    from ai_api_unified.ai_models_config import load_models_config

    load_models_config(json.dumps(dict_payload))
    path_out.write_text(json.dumps(dict_payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path_out} with sections: {', '.join(dict_payload)}")
    if list_moved:
        print("moved from .env: " + ", ".join(list_moved))
    if args.comment_env and list_moved:
        print(
            f"commented {comment_out_keys(path_env, list_moved)} line(s) out of {path_env}"
        )
    elif list_moved:
        print(
            "these keys still override the file while set; rerun with --comment-env to comment them out"
        )
    print(
        "add AI_MODELS_CONFIG=./models.json to .env (or leave it unset: ./models.json is picked up)"
    )
    # Normal return: success.
    return 0


if __name__ == "__main__":
    sys.exit(main())
