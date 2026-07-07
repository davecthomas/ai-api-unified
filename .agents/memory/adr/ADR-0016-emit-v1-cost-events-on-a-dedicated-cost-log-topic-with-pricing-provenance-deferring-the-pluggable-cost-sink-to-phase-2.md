# ADR-0016 Emit v1 cost events on a dedicated cost log topic with pricing provenance, deferring the pluggable cost sink to Phase 2

Status: accepted
Date: 2026-07-07
Owners: dave-thomas
Must read: true
Supersedes: ADR-0014
Superseded by: 
ai-generated: True
ai-model: claude-fable-5[1m]
ai-tool: claude
ai-surface: claude-code
ai-executor: local-agent

Purpose: Emit v1 cost events on a dedicated cost log topic with pricing provenance, deferring the pluggable cost sink to Phase 2
Derived from: [2026-07-07T17-30-01Z--dave-thomas--adr-inspector](../daily/2026-07-07/events/2026-07-07T17-30-01Z--dave-thomas--adr-inspector.md)

## Context

- The revised finops design doc changes how the cost result *leaves* the library. ADR-0014 made a pluggable `AiApiCostSink` (`record(event) -> None`) the v1 emission mechanism, with a default log sink. The revision recognizes that in observe-only v1 the sink abstraction is premature: Python's standard logging already provides topic-based routing via logger names, handlers, and filters, so a dedicated cost log topic gives applications everything they need to filter and route cost events to a separate destination without any new abstraction. The pluggable sink only becomes justified when events must reach a **billing-grade numeric destination in-process** (a Prometheus counter, a cost database, a billing API) where pushing Decimal money through log strings and re-parsing is fragile. That is a destination choice, not a reason to add the interface up front.
- So v1 emits the cost event on a dedicated cost topic — a child logger (`ai_api_unified.observability.cost`) or an `event_type: "cost"` field on the record — and defers `AiApiCostSink` to Phase 2, added as an alternate emitter (`cost_sink` selector) on the same middleware only when a non-log destination forces it. The two properties ADR-0014 established that remain unchanged: each event carries pricing provenance (`pricing_effective_date`, `pricing_source`, `pricing_confidence`) so a stored event is self-explaining and auditable, and the library does **no** in-library aggregation — per-call events only, correlated by the `caller_id` / session / workflow ids already on the context. The change is narrowly the default emission target: log topic now, pluggable sink later.

## Decision

- `docs/finops_middleware_design.md` now specifies that finops v1 emits the cost event on a **dedicated cost log topic** (a `ai_api_unified.observability.cost` child logger or an `event_type: "cost"` field), relying on standard Python logging routing, and that the pluggable `AiApiCostSink` plus a `cost_sink` config selector is **deferred to Phase 2**, added only when a billing/metrics destination requires structured numeric events. This supersedes ADR-0014's decision that v1 emits to a pluggable sink by default. The event still carries pricing provenance and the library still performs no aggregation.

## Consequences

- Promote to ADR superseding ADR-0014. Phase 1 emits cost events on the cost log topic with provenance and `ObservabilitySettingsModel` topic override; `AiApiCostSink` + `cost_sink` selector are Phase 2, added when a non-log destination forces it.

## Source memory events

- [2026-07-07T17-30-01Z--dave-thomas--adr-inspector](../daily/2026-07-07/events/2026-07-07T17-30-01Z--dave-thomas--adr-inspector.md)

## Related code paths

- docs/finops_middleware_design.md
