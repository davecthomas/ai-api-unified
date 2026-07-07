# ADR-0012 Generalize the single observability slot into a fail-open composite chain so cost tracking coexists with logging

Status: superseded
Date: 2026-07-07
Owners: dave-thomas
Must read: true
Supersedes: 
Superseded by: ADR-0015
ai-generated: True
ai-model: claude-fable-5[1m]
ai-tool: claude
ai-surface: claude-code
ai-executor: local-agent

Purpose: Generalize the single observability slot into a fail-open composite chain so cost tracking coexists with logging
Derived from: [2026-07-07T16-42-32Z--dave-thomas--adr-inspector](../daily/2026-07-07/events/2026-07-07T16-42-32Z--dave-thomas--adr-inspector.md)

## Context

- Today `get_observability_middleware()` resolves to exactly one middleware in the observability slot — either a no-op or the logger-backed one. Finops cost tracking is a lifecycle/observability concern (ADR-0003), so it belongs in that same slot, but it must run *alongside* logging rather than displace it. Special-casing "logger + cost" would bake finops into the runtime and make every future observer another special case. The decision is to generalize the single slot into a `CompositeObservabilityMiddleware` that holds N children and fans `before_call` / `after_call` / `on_error` to each. The composite is enabled when any child is enabled, so an all-disabled profile stays a true no-op.
- The composite reuses the fail-open guarantee already implemented in `execute_observed_call`: a child that raises is logged and skipped, and provider behavior is never affected. This is the property that lets observers be additive without risk — adding cost tracking cannot break a provider call even if pricing lookup throws. The change is deliberately confined to how the slot is *built*; provider code and the call runtime are untouched, which is why it is called out as "the one architectural change."

## Decision

- `docs/finops_middleware_design.md` specifies replacing the single-middleware observability slot with a `CompositeObservabilityMiddleware` built by `get_observability_middleware()` from the YAML profile (ADR-0002). The existing logger-backed middleware becomes one child; the new cost middleware becomes another. Fan-out to children preserves the existing fail-open semantics of `execute_observed_call`.

## Consequences

- Promote to ADR. Phase 1 implements `CompositeObservabilityMiddleware` and rewires `get_observability_middleware()` as a behavior-preserving refactor before any cost logic lands.

## Source memory events

- [2026-07-07T16-42-32Z--dave-thomas--adr-inspector](../daily/2026-07-07/events/2026-07-07T16-42-32Z--dave-thomas--adr-inspector.md)

## Related code paths

- docs/finops_middleware_design.md
