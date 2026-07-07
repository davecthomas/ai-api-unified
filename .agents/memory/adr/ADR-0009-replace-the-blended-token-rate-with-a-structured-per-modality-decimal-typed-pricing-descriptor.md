# ADR-0009 Replace the blended token rate with a structured per-modality, Decimal-typed pricing descriptor

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

Purpose: Replace the blended token rate with a structured per-modality, Decimal-typed pricing descriptor
Derived from: [2026-07-07T11-51-24Z--dave-thomas--adr-inspector](../daily/2026-07-07/events/2026-07-07T11-51-24Z--dave-thomas--adr-inspector.md)

## Context

- The current cost model exposes one blended `price_per_1k_tokens` float per model and multiplies it by a raw token count. The research shows this is structurally wrong, not merely imprecise: output is priced 4–8× input (gpt-5.5 is $5 in / $30 out), cached input is now a first-class rate roughly 10× cheaper than fresh input, non-text modalities do not price per token at all (images per image, video per second, TTS per character), and some models tier by context length (gemini-2.5-pro doubles above 200K input tokens). A single scalar cannot express any of these, so blended figures are already fiction — the OpenAI "0.0100" entries are hand-picked midpoints between input and output list prices.
- The decision is to make the price a structured, per-modality descriptor (`AIModelPricing`) whose canonical unit is chosen per modality — tokens per 1M for text/embeddings, characters per 1M for character-billed speech, per-image, per-second of video, per-minute of audio — because there is no honest single cross-modal unit (a token and a rendered second are not convertible). Token pricing splits into input / cached-input / output, and context or quality/size tiers are modeled explicitly rather than flattened into one number. Money is carried as `Decimal`, not `float`, because these values feed a cost ledger where float rounding is unacceptable. This is the load-bearing data-model choice that everything in the pricing redesign and the downstream finops layer builds on.

## Decision

- `pricing_research.md` documents the replacement of the single blended rate with the `AIModelPricing` / `AIPricingTier` / `AITokenRates` descriptor: a per-modality `unit`, split token rates (input, cached_input, output), per-image / per-second / per-minute fields, an optional `tiers` list for context-length and quality/size tiers, and `Decimal`-typed money. Normalized pricing tables for text, embeddings, images, video, and voice establish the canonical unit per modality.

## Consequences

- Promote to ADR; then design the migration that keeps `price_per_1k_tokens` / `calculate_cost` as deprecated blended shims for one release.

## Source memory events

- [2026-07-07T11-51-24Z--dave-thomas--adr-inspector](../daily/2026-07-07/events/2026-07-07T11-51-24Z--dave-thomas--adr-inspector.md)

## Related code paths

- pricing_research.md
