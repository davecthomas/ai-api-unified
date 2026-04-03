---
type: decision_candidate
decision_candidate: true
timestamp: "2026-04-03T00:00:02-07:00"
bootstrapped_at: "2026-04-03T21:00:00Z"
title: YAML-driven middleware profile as sole source of truth; opaque to callers
tags: [architecture, middleware, pii, configuration]
---

# YAML-driven middleware profile as sole source of truth; opaque to callers

Middleware enablement and configuration is controlled exclusively via a YAML file pointed to by `AI_MIDDLEWARE_CONFIG_PATH`. The middleware chain is completely opaque to calling applications — callers pass no flags, decorators, or objects to activate or bypass middleware. If the env var is unset or the file is missing, all middleware remains disabled. YAML parse failures fall back to an empty (disabled) config.

**Execution order:** middleware processes inputs in declaration order and outputs in reverse order.

**Scope:** currently applies to Completions only (prompts and outputs). Embeddings, Voice (TTS/STT), and image generation are explicitly out of scope for PII redaction middleware.

**Rationale:** Replaces ad-hoc boolean environment variables for complex middleware behavior. Provides a single, policy-driven enforcement point so application code never needs awareness of redaction or observability instrumentation.

**Evidence:** `docs/pii_redaction_design.md` (§ Configuration Rule, § Middleware Configuration Profile); `docs/observability_middleware_design.md` (§ Design Goals); commit `7748caf`; `src/ai_api_unified/middleware/`.
