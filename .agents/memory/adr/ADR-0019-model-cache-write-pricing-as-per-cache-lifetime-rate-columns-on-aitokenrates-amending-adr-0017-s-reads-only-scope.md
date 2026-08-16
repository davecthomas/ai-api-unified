# ADR-0019 Model cache-write pricing as per-cache-lifetime rate columns on AITokenRates, amending ADR-0017's reads-only scope

Status: accepted
Date: 2026-08-16
Owners: dave-thomas
Must read: true
Supersedes: 
Superseded by: 
ai-generated: True
ai-tool: claude
ai-surface: claude-code
ai-executor: local-agent

Purpose: Model cache-write pricing as per-cache-lifetime rate columns on AITokenRates, amending ADR-0017's reads-only scope
Derived from: [2026-08-16T17-29-00Z--dave-thomas--adr-inspector](../daily/2026-08-16/events/2026-08-16T17-29-00Z--dave-thomas--adr-inspector.md)

## Context

- ADR-0017 deliberately scoped cache-token billing to cache *reads* only,
  because the pricing registry (ADR-0010) had no cache-write rate column —
  "rather than model a rate that does not exist, the decision defers write
  billing until that column lands." Its Consequences section committed to
  revisiting once such a column existed. That column now exists, so the
  deferral is lifted and the shape of the write-rate model is itself a
  decision worth recording.
- Cache writes cannot be a single rate: providers price cache priming per
  cache lifetime. Anthropic bills 1.25x base input for the 5-minute TTL and
  2x for the 1-hour TTL. A single `cache_write_per_1m` column would force the
  same midpoint-blending mistake the pricing redesign (ADR-0009) exists to
  eliminate — the rate depends on which TTL the caller requested, and the
  provider reports write tokens per TTL.
- The decision: model cache-write pricing as two explicit per-lifetime
  columns on `AITokenRates` — `cache_write_5m_per_1m` and
  `cache_write_1h_per_1m` — both `Decimal | None`, where `None` means the
  provider does not bill writes above base input (writes are free or
  inapplicable). Provider hooks capture the matching per-TTL token counts
  (`provider_cache_write_5m_tokens` / `provider_cache_write_1h_tokens`) on
  the result summary, and cost computation multiplies each write-token bucket
  by its own rate, consistent with the real-usage-times-real-rates rule of
  ADR-0011.
- The accepted tradeoff: the columns enumerate the two TTLs Anthropic
  documents today rather than introducing a generic
  `dict[lifetime, Decimal]`. Explicit columns keep the descriptor flat,
  typed, and auditable per entry; a new provider lifetime would require a new
  column, which is acceptable because billing lifetimes change on provider
  timescales, not caller timescales.

## Decision

- `docs/pricing_research.md` (text-completions notes) now records: "Cache
  writes bill above base input and are priced per cache lifetime — 1.25x for
  the 5-minute TTL and 2x for the 1-hour TTL — modeled since 2.24.0 as
  `cache_write_5m_per_1m` / `cache_write_1h_per_1m` on `AITokenRates`,"
  replacing the prior wording "5-minute cache writes bill 1.25x input (not
  modeled)."
- The doc's API recommendation (`AITokenRates` sketch) gained the two
  columns: `cache_write_5m_per_1m: Decimal | None = None  # None where writes
  are free` and `cache_write_1h_per_1m: Decimal | None = None`.

## Consequences

- Promote this candidate to an ADR capturing the per-cache-lifetime
  cache-write rate columns and their billing rule, amending ADR-0017's
  reads-only scope.
- Bedrock exposes an undifferentiated `cacheWriteInputTokens` (no TTL split);
  confirm how it maps onto the per-TTL columns, and extend if a provider
  introduces a third cache lifetime.

## Source memory events

- [2026-08-16T17-29-00Z--dave-thomas--adr-inspector](../daily/2026-08-16/events/2026-08-16T17-29-00Z--dave-thomas--adr-inspector.md)

## Related code paths

- docs/pricing_research.md
