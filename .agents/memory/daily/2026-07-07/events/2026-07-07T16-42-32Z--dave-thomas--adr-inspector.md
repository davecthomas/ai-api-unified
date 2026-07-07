---
timestamp: "2026-07-07T16:42:32Z"
author: "dave-thomas"
branch: "main"
thread_id: "adr-inspector"
turn_id: "finops-adr-a-observability-chain"
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

- Today `get_observability_middleware()` resolves to exactly one middleware in the observability slot — either a no-op or the logger-backed one. Finops cost tracking is a lifecycle/observability concern (ADR-0003), so it belongs in that same slot, but it must run *alongside* logging rather than displace it. Special-casing "logger + cost" would bake finops into the runtime and make every future observer another special case. The decision is to generalize the single slot into a `CompositeObservabilityMiddleware` that holds N children and fans `before_call` / `after_call` / `on_error` to each. The composite is enabled when any child is enabled, so an all-disabled profile stays a true no-op.
- The composite reuses the fail-open guarantee already implemented in `execute_observed_call`: a child that raises is logged and skipped, and provider behavior is never affected. This is the property that lets observers be additive without risk — adding cost tracking cannot break a provider call even if pricing lookup throws. The change is deliberately confined to how the slot is *built*; provider code and the call runtime are untouched, which is why it is called out as "the one architectural change."

## What changed

- `docs/finops_middleware_design.md` specifies replacing the single-middleware observability slot with a `CompositeObservabilityMiddleware` built by `get_observability_middleware()` from the YAML profile (ADR-0002). The existing logger-backed middleware becomes one child; the new cost middleware becomes another. Fan-out to children preserves the existing fail-open semantics of `execute_observed_call`.

## Evidence

- `docs/finops_middleware_design.md` § "The one architectural change: an observability chain" — the composite diagram (`CompositeObservabilityMiddleware(bool_enabled = any child enabled)` over `LoggerBackedObservabilityMiddleware` and `CostObservabilityMiddleware`) and the statement that it "fans `before_call` / `after_call` / `on_error` to each child under the same fail-open guarantee already in `execute_observed_call`. A child that raises is logged and skipped; provider behavior is never affected."
- `docs/finops_middleware_design.md` § "Where it fits" — grounds the slot as the lifecycle/observability role per ADR-0003 and YAML-driven per ADR-0002.

## Next

- Promote to ADR. Phase 1 implements `CompositeObservabilityMiddleware` and rewires `get_observability_middleware()` as a behavior-preserving refactor before any cost logic lands.
