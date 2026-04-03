---
adr: "0002"
title: YAML-driven middleware profile as sole source of truth; opaque to callers
status: Accepted
date: "2026-04-03"
tags: [architecture, middleware, configuration, pii]
must_read: true
supersedes: ~
superseded_by: ~
---

# ADR-0002: YAML-driven middleware profile as sole source of truth; opaque to callers

## Status
Accepted

## Context
Middleware enablement was previously controlled by ad-hoc boolean environment variables scattered across provider implementations, making consistent policy enforcement difficult.

## Decision
Middleware enablement and all per-component configuration is controlled exclusively via a YAML file at `AI_MIDDLEWARE_CONFIG_PATH`. The middleware chain is completely opaque to calling applications — callers pass no flags or objects to activate or bypass it.

**Resolution rules:**
- If `AI_MIDDLEWARE_CONFIG_PATH` is unset or the file is missing: all middleware stays disabled.
- YAML parse/decode failures fall back to empty (disabled) config; they do not raise.
- Inputs are processed in declaration order; outputs in reverse order.

**Scope:** PII redaction applies to Completions text only (prompts and responses). Embeddings, voice (TTS/STT), and image generation are explicitly excluded from PII redaction middleware.

## Consequences
- A single policy-enforced boundary; application code never needs awareness of redaction or observability.
- Replaces ad-hoc boolean env vars for multi-component behavior.
- Middleware can be added or reordered purely via YAML without code changes.

## Evidence
`docs/pii_redaction_design.md` (§ Configuration Rule, § Middleware Configuration Profile); `docs/observability_middleware_design.md` (§ Design Goals); commit `7748caf`; `src/ai_api_unified/middleware/`.
