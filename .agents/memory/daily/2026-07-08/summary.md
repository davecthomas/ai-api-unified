# 2026-07-08 summary

## Snapshot

- Captured 3 memory events.
- Main work: Added provider_cached_input_tokens to the observability event's provider-usage key set and to the after_call emit gate, so the field is written to the structured event whenever the result summary reports a non-None cache-read count. Emission is conditional, matching the existing prompt/completion/total token handling.
- Top decision: v1 finops billed every input token at the full input rate, which systematically ([2026-07-08 12:21:54 UTC by dave-thomas](events/2026-07-08T12-21-54Z--dave-thomas--adr-inspector.md))
- Blockers: None.

| Metric | Value |
|---|---|
| Memory events captured | 3 |
| Repo files changed | 3 |
| Decision candidates | 1 |
| Active blockers | 0 |

## Major work completed

- Added provider_cached_input_tokens to the observability event's provider-usage key set and to the after_call emit gate, so the field is written to the structured event whenever the result summary reports a non-None cache-read count. Emission is conditional, matching the existing prompt/completion/total token handling.
- The result summary now carries a cached-input-token count, and the Bedrock hook extracts its cache-read count (cacheReadInputTokens) and folds it into the prompt-token total so the cached subset is uniform across providers. The observability cost enrichment surfaces cached_input_tokens on the cost event. Scope is deliberately cache reads only.
- The finops middleware design (`docs/finops_middleware_design.md`) records

## Why this mattered

- Phase 3 of the finops observability plan (docs/finops_middleware_design.md) makes cost reflect provider cache discounts. Surfacing the cached-input token count on the observability event is the prerequisite for billing the cached subset at the cached rate rather than the full input rate, so downstream cost/billing consumers see accurate, auditable usage.
- This advances Phase 3 (2.12.0) of the finops design, which makes cost auditable by billing the cached input subset at the cached rate and the remainder at the full input rate. Without cache-read capture, cached calls are overbilled in the emitted cost events, so this closes an accuracy gap in the cost-attribution stream that billing and budgeting layers downstream depend on.
- v1 finops billed every input token at the full input rate, which systematically

## Active blockers

- None

## Decision candidates

- v1 finops billed every input token at the full input rate, which systematically ([2026-07-08 12:21:54 UTC by dave-thomas](events/2026-07-08T12-21-54Z--dave-thomas--adr-inspector.md))

## Next likely steps

- Ensure the cost-topic event and cost computation split the cached subset at the cached rate and the remainder at the full input rate, per design Decision 3; confirm each provider hook populates provider_cached_input_tokens on the result summary.
- Cache *writes* (Bedrock cacheWriteInputTokens, billed above the base input rate) are not yet captured or billed because the pricing registry has no cache-write rate column; that is the follow-on once such a column lands.
- Promote this candidate to an ADR capturing the uniform cached-input-token
- representation and split-rate billing decision.
- Revisit when a cache-write rate column is added to the pricing registry, which
- would unblock capturing and billing `cacheWriteInputTokens` and supersede the
- reads-only scope stated here.

## Relevant event shards

- [2026-07-07 18:32:22 UTC by 2355287-davecthomas](events/2026-07-08T12-19-54Z--2355287-davecthomas--thread_451c3cb5-1bf1-4f06-911e-bb274923e3b1--turn_afb4e15d9a5aa56d4.md)
- [2026-07-08 12:21:10 UTC by 2355287-davecthomas](events/2026-07-08T12-21-10Z--2355287-davecthomas--thread_5772aded-b79f-4a97-9afb-27036fa135be--turn_0601d61c0f.md)
- [2026-07-08 12:21:54 UTC by dave-thomas](events/2026-07-08T12-21-54Z--dave-thomas--adr-inspector.md)
