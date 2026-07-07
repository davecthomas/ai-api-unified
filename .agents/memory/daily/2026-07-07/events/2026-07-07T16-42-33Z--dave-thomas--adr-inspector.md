---
timestamp: "2026-07-07T16:42:33Z"
author: "dave-thomas"
branch: "main"
thread_id: "adr-inspector"
turn_id: "finops-adr-b-observe-only"
decision_candidate: true
ai_generated: true
ai_model: "claude-fable-5[1m]"
ai_tool: "claude"
ai_surface: "claude-code"
ai_executor: "local-agent"
related_adrs:
files_touched:
  - "docs/finops_middleware_design.md"
design_docs_touched:
  - "docs/finops_middleware_design.md"
verification:
---

## Why

- Cost tracking could plausibly enforce policy — cap spend, throttle, or block a call once a budget is exceeded. The decision is to explicitly *exclude* that from the library: v1 (and the finops role generally) is observe-only. It computes cost per call and emits an event; it never affects program flow. Budgets, alerting, and enforcement are pushed to a separate downstream layer that consumes the emitted event stream, outside this library.
- This boundary is load-bearing for three reasons. First, it keeps the middleware faithful to the lifecycle/observability role (ADR-0003), which observes a completed call and emits metadata, changing nothing about the request or response. Second, it preserves the fail-open guarantee: an observer that can only emit can never break a provider call, whereas an enforcer that blocks calls inherently can. Third, budget semantics differ per consumer (per-caller, per-workflow, hard vs soft caps), so embedding one policy in a shared library would be wrong for most callers; leaving enforcement to the consumer keeps the library policy-free and the event stream the integration contract.

## What changed

- `docs/finops_middleware_design.md` confirms the scope decision: finops v1 is observe-only and emits per-call cost events, with budgets and enforcement deferred to a later layer that consumes the event stream rather than living in the library.

## Evidence

- `docs/finops_middleware_design.md` § "Decisions (confirmed 2026-07-07)" item 1 — "Scope: observe-only. v1 computes cost per call and emits events; it never affects program flow. Budgets and enforcement are a later layer that consumes the event stream."
- `docs/finops_middleware_design.md` § "Where it fits" — "Cost tracking is a lifecycle/observability concern — it observes a completed call and emits metadata, changing nothing about the request or response."
- `docs/finops_middleware_design.md` § "Phased plan" — "Later — budgets/alerting on the event stream, if wanted," placing enforcement strictly downstream of emission.

## Next

- Promote to ADR. Treat the emitted event stream as the seam between this library (observation) and any future enforcement layer.
