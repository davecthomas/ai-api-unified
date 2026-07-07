---
timestamp: "2026-07-07T16:42:34Z"
author: "dave-thomas"
branch: "main"
thread_id: "adr-inspector"
turn_id: "finops-adr-c-event-sink-provenance"
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

- Given that cost is computed from real usage and split rates (ADR-0011), a second decision governs how the result leaves the library: what the emitted unit is, where it goes, and how much the library does with it. The unit is a self-describing `AiApiCostEvent` per call that carries not only `usd_cost`, token counts, and call identity, but the *pricing provenance* — `pricing_effective_date`, `pricing_source`, `pricing_confidence`. Carrying provenance on the event is what makes it auditable: a stored event explains exactly which rate produced it, with no need to re-derive or reconcile against a moving pricing table later.
- Events go to a pluggable `AiApiCostSink` (`record(event) -> None`) whose default writes a structured log line, consistent with current logger-backed observability; applications register their own sink to push to a metrics or billing system. The library deliberately does *no* in-library aggregation — per-call events only, correlated via the `caller_id` / session / workflow ids already on the call context; rollups are the consumer's job. Together these fix the ownership boundary: the library owns emitting accurate, auditable, per-call facts through a stable sink interface, and leaves storage, routing, and aggregation to the application. This avoids baking a storage backend or rollup semantics into a library whose consumers integrate with different billing systems.

## What changed

- `docs/finops_middleware_design.md` specifies the `AiApiCostEvent` shape (including pricing provenance fields), a pluggable `AiApiCostSink` abstraction with a default log sink, and the decision that the library performs no aggregation — emitting per-call events correlated by ids already on the context.

## Evidence

- `docs/finops_middleware_design.md` § "Cost event shape" — the `AiApiCostEvent` fields including `pricing_effective_date`, `pricing_source`, `pricing_confidence`, and the note "Carrying the pricing provenance … on each event is what makes the cost auditable — a stored event explains exactly which rate produced it."
- `docs/finops_middleware_design.md` § "Sink" and "Decisions (confirmed 2026-07-07)" item 2 — the `AiApiCostSink` ABC with default log sink and "Applications register their own sink to push events to a metrics or billing system."
- `docs/finops_middleware_design.md` § "Decisions (confirmed 2026-07-07)" item 4 — "Aggregation: none in-library. Per-call events only, correlated via the `caller_id` / session / workflow ids already on the context. Rollups are the consumer's job."

## Next

- Promote to ADR. Phase 2 implements `AiApiCostEvent`, `AiApiCostSink` + default log sink, and the YAML settings to enable finops and select the sink.
