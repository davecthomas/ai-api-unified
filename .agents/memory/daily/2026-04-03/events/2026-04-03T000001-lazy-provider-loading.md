---
type: decision_candidate
decision_candidate: true
timestamp: "2026-04-03T00:00:01-07:00"
bootstrapped_at: "2026-04-03T21:00:00Z"
title: Registry-based lazy provider loading with optional extras
tags: [architecture, providers, dependency-management, semver]
---

# Registry-based lazy provider loading with optional extras

Providers are loaded only when selected at runtime via a centralized registry (`ai_provider_registry.py`) and lazy loader (`ai_provider_loader.py`). All provider SDK dependencies are declared as optional `[extras]` in `pyproject.toml` rather than as base dependencies.

**Breaking contract (2.0.0):** OpenAI is no longer part of the base install. Consumers must install the `openai` extra explicitly. All provider SDKs follow this pattern.

**Stable API contract preserved:**
- `AIFactory.get_ai_completions_client(...)`, `get_ai_embedding_client(...)`, `get_ai_images_client(...)`, `AIVoiceFactory.create()` remain unchanged.
- Base interfaces (`AIBaseCompletions`, `AIBaseEmbeddings`, `AIBaseImages`, `AIVoiceBase`) remain unchanged.
- Environment selectors (`COMPLETIONS_ENGINE`, `EMBEDDING_ENGINE`, `IMAGE_ENGINE`, `AI_VOICE_ENGINE`) remain unchanged.
- If a provider extra is missing, the factory fails fast with a typed `AIProviderDependencyError`.

**Rationale:** Eliminated module-level `try/except ImportError` with `*_DEPENDENCIES_AVAILABLE` flags scattered across the codebase. Reduces import-time failures, removes `# type: ignore` pressure, and produces clearer errors.

**Evidence:** `docs/conditional-dependencies-best-practices.md` (§ Compatibility Contract, § Problems With The Current Pattern); commit `7748caf`; `src/ai_api_unified/ai_provider_registry.py`, `ai_provider_loader.py`, `ai_provider_exceptions.py`.
