# ADR-0014 Emit per-call cost events carrying pricing provenance to a pluggable sink, with no in-library aggregation

Status: superseded
Date: 2026-07-07
Owners: dave-thomas
Must read: true
Supersedes: 
Superseded by: ADR-0016
ai-generated: True
ai-model: claude-fable-5[1m]
ai-tool: claude
ai-surface: claude-code
ai-executor: local-agent

Purpose: Emit per-call cost events carrying pricing provenance to a pluggable sink, with no in-library aggregation
Derived from: [2026-07-07T16-42-34Z--dave-thomas--adr-inspector](../daily/2026-07-07/events/2026-07-07T16-42-34Z--dave-thomas--adr-inspector.md)

## Context

- Given that cost is computed from real usage and split rates (ADR-0011), a second decision governs how the result leaves the library: what the emitted unit is, where it goes, and how much the library does with it. The unit is a self-describing `AiApiCostEvent` per call that carries not only `usd_cost`, token counts, and call identity, but the *pricing provenance* — `pricing_effective_date`, `pricing_source`, `pricing_confidence`. Carrying provenance on the event is what makes it auditable: a stored event explains exactly which rate produced it, with no need to re-derive or reconcile against a moving pricing table later.
- Events go to a pluggable `AiApiCostSink` (`record(event) -> None`) whose default writes a structured log line, consistent with current logger-backed observability; applications register their own sink to push to a metrics or billing system. The library deliberately does *no* in-library aggregation — per-call events only, correlated via the `caller_id` / session / workflow ids already on the call context; rollups are the consumer's job. Together these fix the ownership boundary: the library owns emitting accurate, auditable, per-call facts through a stable sink interface, and leaves storage, routing, and aggregation to the application. This avoids baking a storage backend or rollup semantics into a library whose consumers integrate with different billing systems.

## Decision

- `docs/finops_middleware_design.md` specifies the `AiApiCostEvent` shape (including pricing provenance fields), a pluggable `AiApiCostSink` abstraction with a default log sink, and the decision that the library performs no aggregation — emitting per-call events correlated by ids already on the context.

## Consequences

- Promote to ADR. Phase 2 implements `AiApiCostEvent`, `AiApiCostSink` + default log sink, and the YAML settings to enable finops and select the sink.

## Source memory events

- [2026-07-07T16-42-34Z--dave-thomas--adr-inspector](../daily/2026-07-07/events/2026-07-07T16-42-34Z--dave-thomas--adr-inspector.md)

## Related code paths

- docs/finops_middleware_design.md
