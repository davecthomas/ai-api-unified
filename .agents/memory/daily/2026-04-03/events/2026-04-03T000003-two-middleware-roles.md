---
type: decision_candidate
decision_candidate: true
timestamp: "2026-04-03T00:00:03-07:00"
bootstrapped_at: "2026-04-03T21:00:00Z"
title: Two distinct middleware roles — text-transform vs lifecycle/observability
tags: [architecture, middleware, observability, design-pattern]
---

# Two distinct middleware roles — text-transform vs lifecycle/observability

The middleware layer is split into two non-overlapping roles:

1. **Text-transform middleware** (e.g., PII redaction): receives prompt/response text and returns modified text. Has a `process_input` / `process_output` interface. Failures raise explicit exceptions.

2. **Lifecycle/observability middleware** (e.g., `AiApiObservabilityMiddleware`): emits metadata-only events (logs) around provider calls without modifying content. Designed to be **fail-open** — observability failures must never break provider calls. Context (caller, session, workflow) flows via `contextvars`.

**Extensibility pattern:** unsupported detection behaviors in either role are added via pluggable recognizer/factory composition (e.g., `CustomRecognizerFactory`) behind the stable `BaseRedactorLayer` / observability interface — not via interface changes.

**Rationale:** Keeps text-transform correctness guarantees separate from observability fail-open semantics, preventing either from compromising the other.

**Evidence:** `docs/observability_middleware_design.md` (§ Middleware Roles); `docs/middleware-extensibility-pattern-pii-poc.md` (§ Scope and Design Goal); commit `7748caf`.
