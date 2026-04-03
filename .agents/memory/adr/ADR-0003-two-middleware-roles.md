---
adr: "0003"
title: Two distinct middleware roles — text-transform vs lifecycle/observability
status: Accepted
date: "2026-04-03"
tags: [architecture, middleware, observability, design-pattern]
must_read: false
supersedes: ~
superseded_by: ~
---

# ADR-0003: Two distinct middleware roles — text-transform vs lifecycle/observability

## Status
Accepted

## Context
The middleware architecture needed to accommodate two fundamentally different behavioral contracts: content modification (PII redaction) where failures must be loud, and observability emission where failures must never break provider calls.

## Decision
Two non-overlapping middleware roles are defined:

1. **Text-transform middleware** (e.g., `AiApiPiiMiddleware`): receives text, returns modified text via `process_input` / `process_output`. Failures raise explicit typed exceptions and propagate to the caller.

2. **Lifecycle/observability middleware** (e.g., `AiApiObservabilityMiddleware`): emits metadata-only log events at provider-call boundaries without modifying content. Designed **fail-open** — observability failures must never interrupt provider calls. Per-request context (caller, session, workflow) is threaded via `contextvars`.

**Extensibility pattern:** unsupported detection behaviors are added via pluggable recognizer/factory composition (e.g., `CustomRecognizerFactory`) behind stable base interfaces (`BaseRedactorLayer`), never via interface changes.

## Consequences
- Text-transform correctness guarantees and observability fail-open semantics cannot interfere with each other.
- New middleware types must be classified into one of the two roles at design time.
- Recognizer extensions are additive and do not require changes to existing middleware contracts.

## Evidence
`docs/observability_middleware_design.md` (§ Middleware Roles); `docs/middleware-extensibility-pattern-pii-poc.md` (§ Scope and Design Goal); commit `7748caf`.
