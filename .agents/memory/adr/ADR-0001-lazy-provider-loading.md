---
adr: "0001"
title: Registry-based lazy provider loading with optional extras
status: Accepted
date: "2026-04-03"
tags: [architecture, providers, dependency-management, semver]
must_read: true
supersedes: ~
superseded_by: ~
---

# ADR-0001: Registry-based lazy provider loading with optional extras

## Status
Accepted

## Context
The codebase previously used scattered module-level `try/except ImportError` blocks with `*_DEPENDENCIES_AVAILABLE` flags to handle optional provider SDKs. This pattern degraded static analysis quality, created non-deterministic symbol availability, and made it hard to produce clear errors when a provider dependency was missing.

## Decision
Providers are loaded only when selected at runtime via a centralized registry (`ai_provider_registry.py`) and lazy loader (`ai_provider_loader.py`). All provider SDK dependencies are declared as optional `[extras]` in `pyproject.toml`.

**Released as SemVer 2.0.0** — OpenAI and all other provider SDKs are no longer base dependencies.

**Stable API contract:**
- Factory entrypoints (`AIFactory.get_ai_completions_client`, `get_ai_embedding_client`, `get_ai_images_client`, `AIVoiceFactory.create`) are unchanged.
- Base interfaces (`AIBaseCompletions`, `AIBaseEmbeddings`, `AIBaseImages`, `AIVoiceBase`) are unchanged.
- Environment engine selectors (`COMPLETIONS_ENGINE`, `EMBEDDING_ENGINE`, `IMAGE_ENGINE`, `AI_VOICE_ENGINE`) are unchanged.
- Missing provider extras produce a fast-fail `AIProviderDependencyError`.
- Concrete provider classes are NOT re-exported from package `__init__.py`; direct module imports required when bypassing factories.

## Consequences
- Fewer import-time failures; consistent, typed error path for missing dependencies.
- Reduced `# type: ignore` burden across the codebase.
- Breaking change: callers must add provider extras to their install (e.g., `pip install ai-api-unified[openai]`).

## Evidence
`docs/conditional-dependencies-best-practices.md`; commit `7748caf`; `src/ai_api_unified/ai_provider_registry.py`, `ai_provider_loader.py`, `ai_provider_exceptions.py`.
