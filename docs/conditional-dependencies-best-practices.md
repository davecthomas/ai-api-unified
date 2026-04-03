# Conditional Dependencies Best Practices

## TL;DR: Old vs New
- Old: optional providers are imported eagerly in many modules, guarded by global availability flags, and failure handling is spread across the codebase.
- New: providers are loaded only when selected via a single centralized lazy-loader boundary, with one consistent typed dependency error path.
- Result: stable factory API for clients, fewer import-time failures, clearer errors, and much less `# type: ignore` pressure.

## Context
This document proposes a cleaner pattern for optional AI provider dependencies in this repository. The goal is to prevent runtime import failures, reduce `# type: ignore` usage, and make provider enablement predictable for developers.

The current codebase has multiple forms of conditional imports:
- Module-level `try/except ImportError` with `*_DEPENDENCIES_AVAILABLE` flags.
- Conditional class definitions (`if DEP_AVAILABLE: class ...`).
- Top-level package imports that attempt to pull optional provider classes into `__init__.py`.
- Factory modules importing optional implementations eagerly, then assigning `None` fallbacks.

These patterns work, but they degrade developer experience and static analysis quality.

## Compatibility Contract
This design preserves factory/base-interface API shape but intentionally removes
root/subpackage re-exports of concrete provider classes.

Breaking change statement:
- OpenAI is no longer part of the base install.
- Consumers must explicitly install the `openai` extra to use OpenAI-backed clients.
- This is a SemVer major change and should be released as `2.0.0` (or later major).

API compatibility guarantees:
- Existing factory entry points remain unchanged:
  - `AIFactory.get_ai_completions_client(...)`
  - `AIFactory.get_ai_embedding_client(...)`
  - `AIFactory.get_ai_images_client(...)`
  - `AIVoiceFactory.create()`
- Existing base interfaces remain unchanged:
  - `AIBaseCompletions`, `AIBaseEmbeddings`, `AIBaseImages`, `AIVoiceBase`
- Existing environment selectors remain unchanged:
  - `COMPLETIONS_ENGINE`, `EMBEDDING_ENGINE`, `IMAGE_ENGINE`, `AI_VOICE_ENGINE`
- Runtime semantics are intentionally tightened:
  - If a provider extra is missing, provider selection fails fast with a clear dependency error.
  - If the provider extra is installed, client behavior is unchanged.

Implementation constraint:
- Keep factories and base classes as the stable contract.
- Do not re-export optional concrete provider classes from package `__init__.py` modules.
- Require direct module imports for concrete provider classes when callers opt out of factories.

## Problems With The Current Pattern
Observed issues in the current structure:

- Optional provider modules are imported too early.
  - Example locations: `src/ai_api_unified/__init__.py`, `ai_factory.py`, `voice/ai_voice_factory.py`, and `*/__init__.py` files.
- Import failures are handled in many places instead of one place.
  - This duplicates behavior and creates inconsistent error messages.
- Optional classes are not always present in module scope.
  - Conditional class definitions can make imports non-deterministic for users and type checkers.
- Broad `ImportError` handling can hide real defects.
  - A transitive internal import bug can look identical to "provider SDK not installed."
- Type-checking quality drops.
  - Many `# type: ignore` entries are compensating for conditional symbol existence.
- Top-level exports become fragile.
  - The package namespace can differ based on what is installed locally.

## What Good Looks Like
Target behavior:

- Importing `ai_api_unified` succeeds with no optional provider extras installed.
- Provider classes are imported only when the specific provider is requested.
- Missing extras produce one deterministic typed exception with a precise install command.
- Base interfaces and factories remain stable.
- Optional provider code can use normal imports and normal typing, with minimal `# type: ignore`.

## Recommended Architecture
Use one runtime loading boundary for optional providers.

### AiProvider Registry
The AiProvider registry is a single in-process metadata map used by factories and the lazy loader.
It is not a network registry or plugin discovery service. It is just validated configuration.

In practical terms, the AiProvider registry is:
- One Pydantic v2 model describing a provider entry.
- One dictionary keyed by `(capability, engine)` that stores those entries.
- The single source of truth for module path, class identifier, required extra, and install guidance.

`AiProviderSpec` field structure in this design:
- `str_capability`: one of `completions`, `embeddings`, `images`, `voice`.
- `str_engine`: normalized engine value used at runtime, such as `openai`, `google-gemini`, `titan`, `nova-canvas`, `azure`, `elevenlabs`.
- `str_module_path`: fully qualified module import path for lazy loading.
- `str_class_name`: provider class identifier (string) resolved from the loaded module after lazy import.
- `str_required_extra`: Poetry extra required to enable this provider.
- `str_consumer_install_command`: install command for downstream client projects.
- `str_local_install_command`: install command for local development in this repository.
- `set_str_dependency_roots`: module root names used to validate whether a `ModuleNotFoundError` is truly a missing optional dependency versus an internal code defect.

Important design note:
- `str_class_name` is intentionally stored as a string instead of a class object.
- Storing a class object would force importing optional provider modules during registry construction, which defeats lazy loading.
- The loader resolves this string to the actual class at runtime, then validates it is a subclass of the capability's expected base interface.

Exact dictionary structure in this design:
- Dictionary key: a 2-tuple `(capability, engine)`.
- Dictionary value: one `AiProviderSpec` object.
- Full type meaning: map each unique capability + engine pair to exactly one provider spec.

Key semantics:
- Key element 1 (`capability`): one of `completions`, `embeddings`, `images`, `voice`.
- Key element 2 (`engine`): normalized runtime engine string from config, for example `openai`, `google-gemini`, `titan`, `nova-canvas`, `azure`, `elevenlabs`.
- Value: one `AiProviderSpec` object for exactly that `(capability, engine)` pair.

Registry example in plain language:
- Entry key `("completions", "openai")` points to the spec for `AiOpenAICompletions` with required extra `openai`.
- Entry key `("completions", "google-gemini")` points to the spec for `GoogleGeminiCompletions` with required extra `google_gemini`.
- Entry key `("embeddings", "google-gemini")` points to the spec for `GoogleGeminiEmbeddings` with required extra `google_gemini`.

How it is consumed:
1. Factory resolves `(capability, engine)` from runtime config.
2. Factory loads `AiProviderSpec` from the registry.
3. Central loader imports `str_module_path`, resolves `str_class_name` to a class object, validates subclass compatibility with the capability base class, and instantiates.
4. If dependency import fails, loader raises one typed dependency exception using install commands from that same `AiProviderSpec`.

Why this improves clarity:
- No scattered hardcoded install strings.
- No duplicated module/class metadata.
- Registry mistakes fail fast through Pydantic validation.
- Adding a provider means adding one validated entry, not editing import guards across many files.

### Central Lazy Loader
Add one loader utility that:
- Imports the selected provider implementation with `importlib.import_module`.
- Resolves the class through the module namespace dictionary using the registry class name.
- Converts missing dependency failures into a custom typed exception.
- Does not swallow unrelated internal errors.

Key rule:
- Catch `ModuleNotFoundError` narrowly and validate whether the missing module belongs to the provider SDK stack.
- Re-raise non-dependency import failures as internal runtime errors.

### Factory Behavior
Factories should:
- Read engine selection.
- Resolve provider metadata from the registry.
- Lazy-load the class only for that selected provider.
- Instantiate and return the base interface type.

Factories should not:
- Import all provider implementations at module import time.
- Keep provider-specific availability booleans in global state.

### Package Export Strategy
The top-level package should export:
- Base abstractions.
- Factory classes.
- Shared typed exceptions.

Do not re-export optional provider concrete classes from package `__init__.py`.
This keeps package imports simple, deterministic, and free of implicit dynamic
attribute resolution paths.

### Typing Strategy
Use typing patterns that avoid runtime dependency coupling:
- Base interfaces return abstract/base types from factories.
- `TYPE_CHECKING` imports only for static typing where needed.
- Avoid assigning optional class symbols to `None` with ignore comments.

### Error Model
Introduce common exceptions for all optional providers:
- `AiProviderDependencyUnavailableError`
- `AiProviderConfigurationError`
- `AiProviderRuntimeError`

Dependency errors should include:
- Provider key.
- Missing package/module.
- Exactly which Poetry command to run.

## Comparison With Middleware Redaction Pattern
The middleware PII redaction approach is directionally correct and should be treated as the baseline pattern:
- It uses a single dynamic import boundary in `middleware/pii_redactor.py`.
- It maps missing optional dependencies to a typed domain exception.
- It supports strict mode behavior explicitly.

Improvements to carry over for provider loading:
- Keep exception handling narrow (`ModuleNotFoundError` vs broad `ImportError`).
- Use one shared dependency exception family across providers.
- Standardize install guidance text from a registry instead of hardcoded per-call strings.

## Required `pyproject.toml` Changes
To meet the requirement that each provider must be explicitly enabled:

- Move provider SDKs out of base dependencies and into provider extras.
  - This includes moving OpenAI into an explicit `openai` extra.
- Keep base install provider-neutral (core abstractions and shared utilities only).
- Preserve or refine existing provider extras:
  - `openai`
  - `bedrock`
  - `google_gemini`
  - `azure_tts`
  - `elevenlabs`
- Keep middleware extras independent:
  - `middleware-pii-redaction`
  - `middleware-pii-redaction-small`
  - `middleware-pii-redaction-large`
- Optionally add a convenience umbrella extra for local integration testing, while keeping provider-specific extras as the main path.

## Required `README.md` Changes
Update installation docs so behavior is explicit and consistent:

- State clearly that base install includes no provider SDKs.
- Add a provider install table with required commands:
  - Consumer project form: `poetry add 'ai-api-unified[<extra>]'`
  - Local repository form: `poetry install --extras "<extra>"`
- Add explicit mapping from environment engine values to extras.
- Add a troubleshooting section for dependency exceptions with examples.
- Remove or revise guidance that implies providers are available without explicit extras.

## Required `env_template.txt` Changes
`env_template.txt` should evolve to mirror the provider-extra model and reduce setup confusion.

What should change:
- Keep a small base section that always applies:
  - `COMPLETIONS_ENGINE`
  - `EMBEDDING_ENGINE`
  - `IMAGE_ENGINE`
  - `AI_VOICE_ENGINE`
- Split provider credentials into clearly labeled provider blocks:
  - OpenAI block
  - Bedrock block
  - Google Gemini block
  - Azure TTS block
  - ElevenLabs block
- For each provider block, include:
  - Required env vars.
  - Optional env vars.
  - The exact extra install command for that provider.
  - A note that these variables are ignored unless the matching engine is selected.

Recommended documentation behavior in `env_template.txt`:
- Keep placeholder-only values, never real secrets.
- Add one short note per provider block stating the dependency gate, for example:
  - "Requires `openai` extra."
  - "Requires `bedrock` extra."
  - "Requires `google_gemini` extra."
- Keep middleware environment examples in a separate middleware section so provider and middleware dependency gates stay distinct.

What should not change:
- Existing environment variable names used by current public APIs should remain stable for backward compatibility.
- Existing engine selector names should remain stable (`openai`, `google-gemini`, `titan`, `nova-canvas`, `azure`, `elevenlabs`, etc.).

Net effect:
- `env_template.txt` remains one file, but becomes capability- and provider-scoped documentation rather than one flat mixed list.
- Developers can identify required variables and install prerequisites per selected provider without scanning unrelated settings.

## Migration Plan
Recommended phased rollout:

- Phase A
  - Introduce the AiProvider registry and shared dependency exceptions.
  - Add lazy-loader utility and wire one factory path first.
- Phase B
  - Remove eager optional imports from package and subpackage `__init__.py`.
  - Migrate all factories to the centralized loader.
- Phase C
  - Simplify provider modules to normal imports and normal class definitions.
  - Remove legacy `*_DEPENDENCIES_AVAILABLE` flags and `None` fallback assignments.
- Phase D
  - Move OpenAI SDK to an explicit extra and update docs/install matrix.
  - Expand tests for base install and provider-extra installs.

## Validation Checklist
After migration, these conditions should hold:

- `import ai_api_unified` succeeds in a base-only install.
- Selecting an uninstalled provider fails with `AiProviderDependencyUnavailableError` and a correct Poetry command.
- Selecting an installed provider succeeds without extra import guards in unrelated modules.
- Type-check and lint noise from optional dependency branching is materially reduced.
- Public factory APIs remain stable.

## Summary
The best path is to centralize optional dependency loading in one import boundary and keep provider SDK imports out of global module initialization. The middleware redaction strategy demonstrates the right direction. Applying that pattern consistently across OpenAI, Bedrock, Gemini, Azure, and ElevenLabs will produce a simpler codebase, better type safety, and a much better developer experience.
