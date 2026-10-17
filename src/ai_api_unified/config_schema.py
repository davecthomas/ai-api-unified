# config_schema.py
"""
JSON Schemas for the library's configuration files, and a validator.

The schemas are generated from the pydantic models that load the files, so
they cannot drift from what the library accepts:

- models.schema.json for models.json (AIModelsConfig)
- middleware.schema.json for the middleware YAML (AI_MIDDLEWARE_CONFIG_PATH)

Both ship in the package under ai_api_unified/schemas/ and are published at
the URLs in SCHEMA_URL_MODELS and SCHEMA_URL_MIDDLEWARE. Point an editor at
one with a `"$schema"` key at the top of models.json, or a
`# yaml-language-server: $schema=<url>` line at the top of the YAML.

Where a loader accepts more than one spelling (a tier's `models` list or
its `candidates`, `fallback_on` as a list or a comma-separated string, a
middleware setting a validator normalizes), the schema accepts each
spelling, so a file the library loads never fails the schema.

Command line:

    python -m ai_api_unified.config_schema validate models.json
    python -m ai_api_unified.config_schema validate middleware.yaml
    python -m ai_api_unified.config_schema print models
    python -m ai_api_unified.config_schema write   # regenerate the shipped files
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

SCHEMA_DIALECT: str = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_URL_BASE: str = (
    "https://raw.githubusercontent.com/davecthomas/ai-api-unified/main/"
    "src/ai_api_unified/schemas/"
)
SCHEMA_FILE_MODELS: str = "models.schema.json"
SCHEMA_FILE_MIDDLEWARE: str = "middleware.schema.json"
SCHEMA_URL_MODELS: str = SCHEMA_URL_BASE + SCHEMA_FILE_MODELS
SCHEMA_URL_MIDDLEWARE: str = SCHEMA_URL_BASE + SCHEMA_FILE_MIDDLEWARE
PATH_SCHEMAS_DIR: Path = Path(__file__).resolve().parent / "schemas"

# fallback_on: a list of reasons, or one comma-separated string of them.
DICT_FALLBACK_ON_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {"type": "array", "items": {"$ref": "#/$defs/AiFallbackReason"}},
        {
            "type": "string",
            "description": "Comma-separated AiFallbackReason values.",
        },
        {"type": "null"},
    ],
    "default": None,
    "title": "Fallback On",
}


def _patch_models_schema(dict_schema: dict[str, Any]) -> dict[str, Any]:
    """Widens the generated schema to every spelling the loaders accept."""
    dict_defs: dict[str, Any] = dict_schema["$defs"]
    tier: dict[str, Any] = dict_defs["AIRouterTier"]
    tier["properties"]["models"] = {
        "description": (
            "The tier's models in order, the primary first: 'engine:model' "
            "strings (comma-separated pairs allowed) or {engine, model, "
            "base_url} objects. Same as candidates."
        ),
        "type": "array",
        "minItems": 1,
        "items": {
            "anyOf": [
                {"type": "string"},
                {"$ref": "#/$defs/AIFallbackCandidate"},
            ]
        },
    }
    tier["required"] = ["name"]
    tier["oneOf"] = [{"required": ["models"]}, {"required": ["candidates"]}]
    tier["properties"]["fallback_on"] = copy.deepcopy(DICT_FALLBACK_ON_SCHEMA)
    dict_defs["AIRouterConfig"]["properties"]["fallback_on"] = copy.deepcopy(
        DICT_FALLBACK_ON_SCHEMA
    )
    dict_completions: dict[str, Any] = dict_defs["AICompletionsSelection"]["properties"]
    dict_completions["fallback_on"] = copy.deepcopy(DICT_FALLBACK_ON_SCHEMA)
    dict_completions["thinking"] = {
        "anyOf": [{"$ref": "#/$defs/AIThinkingLevel"}, {"type": "null"}],
        "default": None,
        "title": "Thinking",
    }
    from .routing.config_checks import FROZENSET_CAPABILITY_FIELDS

    dict_defs["AIConfigRequirements"]["properties"]["capabilities"]["items"] = {
        "type": "string",
        "enum": sorted(FROZENSET_CAPABILITY_FIELDS),
    }
    dict_defs["AIClassifierSelection"]["properties"]["thinking"] = {
        "anyOf": [{"$ref": "#/$defs/AIThinkingLevel"}, {"type": "null"}],
        "default": None,
        "title": "Thinking",
    }
    dict_completions["fallbacks"]["items"] = {
        "type": "string",
        "description": "An 'engine:model' pair; the split is on the first colon.",
    }
    # Normal return with the widened schema.
    return dict_schema


def models_config_schema() -> dict[str, Any]:
    """The JSON Schema for models.json."""
    from .ai_models_config import AIModelsConfig

    dict_schema: dict[str, Any] = AIModelsConfig.model_json_schema(
        by_alias=True, mode="validation"
    )
    dict_schema = _patch_models_schema(dict_schema)
    # The middleware section is the middleware YAML's list, entry for entry.
    dict_middleware_defs, dict_entry = _middleware_defs_and_entry()
    dict_schema["$defs"].update(dict_middleware_defs)
    dict_schema["$defs"]["MiddlewareEntry"] = dict_entry
    dict_schema["properties"]["middleware"] = {
        "anyOf": [
            {"type": "array", "items": {"$ref": "#/$defs/MiddlewareEntry"}},
            {"type": "null"},
        ],
        "default": None,
        "title": "Middleware",
        "description": (
            "PII redaction and observability middleware; the same list the "
            "middleware YAML holds under `middleware`."
        ),
    }
    # Normal return with the dialect and id first, for readers.
    return {
        "$schema": SCHEMA_DIALECT,
        "$id": SCHEMA_URL_MODELS,
        **dict_schema,
    }


def _relax_before_validated(
    model: type[BaseModel], dict_schema: dict[str, Any]
) -> dict[str, Any]:
    """
    Drops the type constraint from fields a "before" validator normalizes,
    since those accept more spellings than the field's type; the title,
    description, and default stay.
    """
    set_relaxed: set[str] = set()
    # Loop over the model's field validators for the "before" ones.
    for decorator in model.__pydantic_decorators__.field_validators.values():
        if decorator.info.mode == "before":
            set_relaxed.update(decorator.info.fields)
    dict_properties: dict[str, Any] = dict_schema.get("properties", {})
    # Loop over the relaxed fields, keeping only their documentation.
    for str_field in set_relaxed:
        str_key: str = model.model_fields[str_field].alias or str_field
        dict_property: dict[str, Any] | None = dict_properties.get(str_key)
        if dict_property is None:
            continue
        dict_properties[str_key] = {
            key: value
            for key, value in dict_property.items()
            if key in ("title", "description", "default")
        }
    # Normal return with the relaxed schema.
    return dict_schema


def _middleware_defs_and_entry() -> tuple[dict[str, Any], dict[str, Any]]:
    """
    The middleware settings definitions and the schema of one `middleware`
    entry, shared by the middleware YAML schema and models.json's section.
    """
    from .middleware.middleware_config import (
        OBSERVABILITY,
        PII_REDACTION,
        ObservabilitySettingsModel,
        PiiRedactionSettingsModel,
    )

    dict_defs: dict[str, Any] = {}
    dict_settings_refs: dict[str, str] = {}
    # Loop over the settings models so each middleware's block is described.
    for str_name, model in (
        (PII_REDACTION, PiiRedactionSettingsModel),
        (OBSERVABILITY, ObservabilitySettingsModel),
    ):
        dict_model: dict[str, Any] = model.model_json_schema(
            by_alias=True, mode="validation", ref_template="#/$defs/{model}"
        )
        dict_defs.update(dict_model.pop("$defs", {}))
        dict_defs[model.__name__] = _relax_before_validated(model, dict_model)
        dict_settings_refs[str_name] = f"#/$defs/{model.__name__}"
    # Loop over nested models too, so their normalized fields are relaxed.
    from .middleware import middleware_config as module

    for str_def, dict_def in list(dict_defs.items()):
        nested: Any = getattr(module, str_def, None)
        if isinstance(nested, type) and issubclass(nested, BaseModel):
            dict_defs[str_def] = _relax_before_validated(nested, dict_def)
    dict_entry: dict[str, Any] = {
        "type": "object",
        "required": ["name"],
        "properties": {
            "name": {"type": "string", "enum": sorted(dict_settings_refs)},
            "enabled": {"type": "boolean", "default": False},
            "settings": {"description": "Settings for the named middleware."},
        },
        "allOf": [
            {
                "if": {"properties": {"name": {"const": str_name}}},
                "then": {"properties": {"settings": {"$ref": str_ref}}},
            }
            for str_name, str_ref in sorted(dict_settings_refs.items())
        ],
    }
    # Normal return with the definitions and the entry schema.
    return dict_defs, dict_entry


def middleware_config_schema() -> dict[str, Any]:
    """The JSON Schema for the middleware YAML."""
    from .middleware.middleware_config import MIDDLEWARE_KEY

    dict_defs, dict_entry = _middleware_defs_and_entry()
    # Normal return with the whole file's schema.
    return {
        "$schema": SCHEMA_DIALECT,
        "$id": SCHEMA_URL_MIDDLEWARE,
        "title": "ai_api_unified middleware configuration",
        "type": "object",
        "properties": {
            MIDDLEWARE_KEY: {"type": "array", "items": dict_entry},
        },
        "$defs": dict_defs,
    }


def _render(dict_schema: dict[str, Any]) -> str:
    # Normal return with stable, diff-friendly JSON.
    return json.dumps(dict_schema, indent=2, sort_keys=False) + "\n"


def write_schemas(path_dir: Path = PATH_SCHEMAS_DIR) -> list[Path]:
    """Regenerates the shipped schema files; returns their paths."""
    path_dir.mkdir(parents=True, exist_ok=True)
    list_written: list[Path] = []
    # Loop over both schemas.
    for str_file, callable_schema in (
        (SCHEMA_FILE_MODELS, models_config_schema),
        (SCHEMA_FILE_MIDDLEWARE, middleware_config_schema),
    ):
        path_file: Path = path_dir / str_file
        path_file.write_text(_render(callable_schema()))
        list_written.append(path_file)
    # Normal return with the written paths.
    return list_written


def validate_config_file(str_path: str) -> list[str]:
    """
    Loads a models.json/.yaml or middleware YAML file the way the library
    does and returns its errors; empty when the file is valid.
    """
    path: Path = Path(str_path)
    if not path.is_file():
        # Early return: nothing to read.
        return [f"{str_path}: no such file"]
    import yaml

    str_text: str = path.read_text()
    try:
        raw: Any = (
            json.loads(str_text)
            if path.suffix.lower() == ".json"
            else yaml.safe_load(str_text)
        )
    except (json.JSONDecodeError, yaml.YAMLError) as exception:
        # Early return: not parseable.
        return [f"{str_path}: {exception}"]
    if isinstance(raw, dict) and set(raw) <= {"middleware"}:
        # Early return: a middleware YAML holds only the middleware list.
        return _middleware_errors(raw.get("middleware"))
    from .ai_models_config import AIModelsConfig

    try:
        config = AIModelsConfig.model_validate(raw)
    except ValidationError as error:
        # Early return with the models.json errors.
        return [str(error)]
    # Normal return with the middleware section's errors, if any.
    return _middleware_errors(config.middleware) if config.middleware else []


def _middleware_errors(list_middleware: Any) -> list[str]:
    """Each middleware entry checked by its own settings model."""
    from .middleware.middleware_config import (
        MiddlewareConfigurationModel,
        ObservabilitySettingsModel,
        PiiRedactionSettingsModel,
    )

    try:
        configuration = MiddlewareConfigurationModel.model_validate(
            {"middleware": list_middleware or []}
        )
    except ValidationError as error:
        # Early return with the structural errors.
        return [str(error)]
    dict_models: dict[str, type[BaseModel]] = {
        "pii_redaction": PiiRedactionSettingsModel,
        "observability": ObservabilitySettingsModel,
    }
    list_errors: list[str] = []
    # Loop over entries so each block is checked by its own model.
    for entry in configuration.list_middleware:
        model: type[BaseModel] | None = dict_models.get(entry.name)
        if model is None:
            list_errors.append(f"unknown middleware {entry.name!r}")
            continue
        try:
            model.model_validate(entry.settings or {})
        except ValidationError as error:
            list_errors.append(f"{entry.name}: {error}")
    # Normal return with every entry's errors.
    return list_errors


def main(list_args: list[str]) -> int:
    """Command-line entry: validate, print, or write."""
    if len(list_args) >= 2 and list_args[0] == "validate":
        int_status: int = 0
        # Loop over the files named.
        for str_path in list_args[1:]:
            list_errors: list[str] = validate_config_file(str_path)
            if list_errors:
                int_status = 1
                print(f"{str_path}: INVALID")
                for str_error in list_errors:
                    print(f"  {str_error}")
            else:
                print(f"{str_path}: valid")
        # Early return with 1 when any file failed.
        return int_status
    if len(list_args) == 2 and list_args[0] == "print":
        callable_schema: Any = {
            "models": models_config_schema,
            "middleware": middleware_config_schema,
        }.get(list_args[1])
        if callable_schema is not None:
            sys.stdout.write(_render(callable_schema()))
            # Early return after printing.
            return 0
    if list_args == ["write"]:
        for path in write_schemas():
            print(f"wrote {path}")
        # Early return after writing.
        return 0
    print(
        "usage: python -m ai_api_unified.config_schema "
        "validate FILE... | print models|middleware | write"
    )
    # Normal return: bad usage.
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
