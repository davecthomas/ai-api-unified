# 2026-08-16 summary

## Snapshot

- Captured 2 memory events.
- Main work: docs/pricing_research.md now records that cache writes are priced per cache lifetime and modeled since 2.24.0 as cache_write_5m_per_1m and cache_write_1h_per_1m on AITokenRates (None where writes are free), extending the ADR-0009 Decimal-typed descriptor. The prose caveat that 5-minute cache writes were 'not modeled' is gone, and the schema listing in the doc matches the shipped model.
- Top decision: Prompt-cache writes bill above base input on Anthropic (1.25x for the 5-minute TTL, 2x for the 1-hour TTL), and until 2.24.0 the pricing model only covered cache reads, leaving a real cost invisible to finops attribution. The pricing redesign is the prerequisite for the middleware observability layer, so keeping the design doc's schema in sync with the shipped descriptor preserves the doc as a trustworthy source of truth for future pricing work. ([2026-08-16 17:28:24 UTC by 2355287-davecthomas](events/2026-08-16T17-28-24Z--2355287-davecthomas--thread_bcd551a6-ba5b-4ac4-8fb6-6b3182463480--turn_af34c9d1166aee819.md))
- Blockers: Cache writes cannot be a single rate: providers price cache priming per

| Metric | Value |
|---|---|
| Memory events captured | 2 |
| Repo files changed | 2 |
| Decision candidates | 2 |
| Active blockers | 1 |

## Major work completed

- docs/pricing_research.md now records that cache writes are priced per cache lifetime and modeled since 2.24.0 as cache_write_5m_per_1m and cache_write_1h_per_1m on AITokenRates (None where writes are free), extending the ADR-0009 Decimal-typed descriptor. The prose caveat that 5-minute cache writes were 'not modeled' is gone, and the schema listing in the doc matches the shipped model.
- `docs/pricing_research.md` (text-completions notes) now records: "Cache

## Why this mattered

- Prompt-cache writes bill above base input on Anthropic (1.25x for the 5-minute TTL, 2x for the 1-hour TTL), and until 2.24.0 the pricing model only covered cache reads, leaving a real cost invisible to finops attribution. The pricing redesign is the prerequisite for the middleware observability layer, so keeping the design doc's schema in sync with the shipped descriptor preserves the doc as a trustworthy source of truth for future pricing work.
- ADR-0017 deliberately scoped cache-token billing to cache *reads* only,

## Active blockers

- Cache writes cannot be a single rate: providers price cache priming per

## Decision candidates

- Prompt-cache writes bill above base input on Anthropic (1.25x for the 5-minute TTL, 2x for the 1-hour TTL), and until 2.24.0 the pricing model only covered cache reads, leaving a real cost invisible to finops attribution. The pricing redesign is the prerequisite for the middleware observability layer, so keeping the design doc's schema in sync with the shipped descriptor preserves the doc as a trustworthy source of truth for future pricing work. ([2026-08-16 17:28:24 UTC by 2355287-davecthomas](events/2026-08-16T17-28-24Z--2355287-davecthomas--thread_bcd551a6-ba5b-4ac4-8fb6-6b3182463480--turn_af34c9d1166aee819.md))
- ADR-0017 deliberately scoped cache-token billing to cache *reads* only, ([2026-08-16 17:29:00 UTC by dave-thomas](events/2026-08-16T17-29-00Z--dave-thomas--adr-inspector.md))

## Next likely steps

- Ship feat/cache-write-billing via PR with its 2.24.0 minor bump; release flow still requires the version-sync test and the full mocked regression before tagging.
- Promote this candidate to an ADR capturing the per-cache-lifetime
- cache-write rate columns and their billing rule, amending ADR-0017's
- reads-only scope.
- Bedrock exposes an undifferentiated `cacheWriteInputTokens` (no TTL split);
- confirm how it maps onto the per-TTL columns, and extend if a provider
- introduces a third cache lifetime.

## Relevant event shards

- [2026-08-16 17:28:24 UTC by 2355287-davecthomas](events/2026-08-16T17-28-24Z--2355287-davecthomas--thread_bcd551a6-ba5b-4ac4-8fb6-6b3182463480--turn_af34c9d1166aee819.md)
- [2026-08-16 17:29:00 UTC by dave-thomas](events/2026-08-16T17-29-00Z--dave-thomas--adr-inspector.md)
