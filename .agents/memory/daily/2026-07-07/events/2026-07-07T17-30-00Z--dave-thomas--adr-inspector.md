---
timestamp: "2026-07-07T17:30:00Z"
author: "dave-thomas"
branch: "main"
thread_id: "adr-inspector"
turn_id: "finops-adr-v1-enrichment-no-composite"
decision_candidate: true
ai_generated: true
ai_model: "claude-fable-5[1m]"
ai_tool: "claude"
ai_surface: "claude-code"
ai_executor: "local-agent"
related_adrs:
  - "ADR-0012"
  - "ADR-0002"
  - "ADR-0003"
files_touched:
  - "docs/finops_middleware_design.md"
design_docs_touched:
  - "docs/finops_middleware_design.md"
verification:
---

## Why

- The finops design doc was revised on 2026-07-07 and now reverses its earlier plan for a `CompositeObservabilityMiddleware` that chained a separate cost middleware (the decision captured in ADR-0012). The revised framing is that cost is not a new subsystem but a *derived field* on the observability event the logger-backed middleware already emits in `after_call`: the same model, provider, and token counts, plus a computed `usd_cost`. Because observe-only v1 introduces no new data, hook, or lifecycle, a composite that fans lifecycle calls to N children is scaffolding for a separation that does not yet exist. Building it now would add a runtime abstraction ahead of any second observer that needs it.
- The chosen v1 shape is a config-gated enrichment *inside* the existing logger-backed observability `after_call`, with no new middleware class and no change to how the observability slot is built. Placement in the observability emit path (rather than the per-provider result-summary builder) is deliberate: the `emit_cost` gate means the Decimal pricing math runs only when observability is enabled, and pricing lookup stays out of the provider call path entirely. The composite is not discarded but deferred — it "earns its keep" only when a second, genuinely distinct observer or a non-log destination forces the separation. This keeps v1 minimal while preserving ADR-0003's lifecycle/observability role and ADR-0002's YAML-driven, single-middleware model (finops is one `emit_cost` flag on the existing observability settings, not a new profile entry).

## What changed

- `docs/finops_middleware_design.md` now specifies that finops v1 adds **no new middleware class and no composite chain**. Cost is computed as a config-gated enrichment within the existing logger-backed observability `after_call` when `emit_cost` is true: resolve `AIModelPricing` for `(provider, model)`, read the provider-reported prompt/completion tokens, compute `usd_cost` via `compute_token_cost`, and emit it on the observability event. This supersedes ADR-0012's decision to generalize the single observability slot into a `CompositeObservabilityMiddleware`; that composite is explicitly deferred until a separation earns its keep.

## Evidence

- `docs/finops_middleware_design.md` § "Framing: cost is a topic on the observability event, not a new subsystem" — "Cost is a *derived field* on that same event — there is no new data, hook, or lifecycle" and "This replaces an earlier plan for a `CompositeObservabilityMiddleware` chaining a separate cost middleware. That composite is scaffolding for a separation that does not exist in observe-only v1, so it is deferred."
- `docs/finops_middleware_design.md` § "Design → Middleware — no new class in v1" — "Cost is a config-gated enrichment inside the existing logger-backed observability `after_call`, not a new middleware," including the gating rationale that keeps pricing out of the provider call path.
- `docs/finops_middleware_design.md` § "Where it fits" and § "Config — one flag on the observability settings" — grounds finops in ADR-0003's lifecycle/observability role and ADR-0002's YAML-driven model, with `ObservabilitySettingsModel.emit_cost` as the single enabling flag.
- `docs/finops_middleware_design.md` § "When separation earns its keep (deferred)" — the composite/pluggable separation is a Phase-2 concern, added only when a non-log destination forces it.

## Next

- Promote to ADR superseding ADR-0012. Phase 1 implements the `emit_cost`-gated enrichment inside the existing observability `after_call` and adds `ObservabilitySettingsModel.emit_cost`; no observability-slot refactor lands in v1.
