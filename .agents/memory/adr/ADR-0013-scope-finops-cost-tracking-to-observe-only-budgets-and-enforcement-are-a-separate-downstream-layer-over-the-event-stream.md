# ADR-0013 Scope finops cost tracking to observe-only; budgets and enforcement are a separate downstream layer over the event stream

Status: accepted
Date: 2026-07-07
Owners: dave-thomas
Must read: true
Supersedes: 
Superseded by: 
ai-generated: True
ai-model: claude-fable-5[1m]
ai-tool: claude
ai-surface: claude-code
ai-executor: local-agent

Purpose: Scope finops cost tracking to observe-only; budgets and enforcement are a separate downstream layer over the event stream
Derived from: [2026-07-07T16-42-33Z--dave-thomas--adr-inspector](../daily/2026-07-07/events/2026-07-07T16-42-33Z--dave-thomas--adr-inspector.md)

## Context

- Cost tracking could plausibly enforce policy — cap spend, throttle, or block a call once a budget is exceeded. The decision is to explicitly *exclude* that from the library: v1 (and the finops role generally) is observe-only. It computes cost per call and emits an event; it never affects program flow. Budgets, alerting, and enforcement are pushed to a separate downstream layer that consumes the emitted event stream, outside this library.
- This boundary is load-bearing for three reasons. First, it keeps the middleware faithful to the lifecycle/observability role (ADR-0003), which observes a completed call and emits metadata, changing nothing about the request or response. Second, it preserves the fail-open guarantee: an observer that can only emit can never break a provider call, whereas an enforcer that blocks calls inherently can. Third, budget semantics differ per consumer (per-caller, per-workflow, hard vs soft caps), so embedding one policy in a shared library would be wrong for most callers; leaving enforcement to the consumer keeps the library policy-free and the event stream the integration contract.

## Decision

- `docs/finops_middleware_design.md` confirms the scope decision: finops v1 is observe-only and emits per-call cost events, with budgets and enforcement deferred to a later layer that consumes the event stream rather than living in the library.

## Consequences

- Promote to ADR. Treat the emitted event stream as the seam between this library (observation) and any future enforcement layer.

## Source memory events

- [2026-07-07T16-42-33Z--dave-thomas--adr-inspector](../daily/2026-07-07/events/2026-07-07T16-42-33Z--dave-thomas--adr-inspector.md)

## Related code paths

- docs/finops_middleware_design.md
