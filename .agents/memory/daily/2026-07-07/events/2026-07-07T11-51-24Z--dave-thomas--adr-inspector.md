---
timestamp: "2026-07-07T11:51:24Z"
author: "dave-thomas"
branch: "main"
thread_id: "adr-inspector"
turn_id: "pricing-adr-a"
decision_candidate: true
ai_generated: true
ai_model: "claude-fable-5[1m]"
ai_tool: "claude"
ai_surface: "claude-code"
ai_executor: "local-agent"
related_adrs:
files_touched:
  - "pricing_research.md"
design_docs_touched:
  - "pricing_research.md"
verification:
---

## Why

- The current cost model exposes one blended `price_per_1k_tokens` float per model and multiplies it by a raw token count. The research shows this is structurally wrong, not merely imprecise: output is priced 4–8× input (gpt-5.5 is $5 in / $30 out), cached input is now a first-class rate roughly 10× cheaper than fresh input, non-text modalities do not price per token at all (images per image, video per second, TTS per character), and some models tier by context length (gemini-2.5-pro doubles above 200K input tokens). A single scalar cannot express any of these, so blended figures are already fiction — the OpenAI "0.0100" entries are hand-picked midpoints between input and output list prices.
- The decision is to make the price a structured, per-modality descriptor (`AIModelPricing`) whose canonical unit is chosen per modality — tokens per 1M for text/embeddings, characters per 1M for character-billed speech, per-image, per-second of video, per-minute of audio — because there is no honest single cross-modal unit (a token and a rendered second are not convertible). Token pricing splits into input / cached-input / output, and context or quality/size tiers are modeled explicitly rather than flattened into one number. Money is carried as `Decimal`, not `float`, because these values feed a cost ledger where float rounding is unacceptable. This is the load-bearing data-model choice that everything in the pricing redesign and the downstream finops layer builds on.

## What changed

- `pricing_research.md` documents the replacement of the single blended rate with the `AIModelPricing` / `AIPricingTier` / `AITokenRates` descriptor: a per-modality `unit`, split token rates (input, cached_input, output), per-image / per-second / per-minute fields, an optional `tiers` list for context-length and quality/size tiers, and `Decimal`-typed money. Normalized pricing tables for text, embeddings, images, video, and voice establish the canonical unit per modality.

## Evidence

- `pricing_research.md` § "The core finding: the current cost model cannot support finops" (six enumerated ways the blended float breaks).
- `pricing_research.md` § "Normalized pricing tables" (canonical unit per modality; input / cached-input / output split; gemini-2.5-pro ≤200K vs >200K tiers).
- `pricing_research.md` § "API recommendation" — the `AIModelPricing` model definition and the "Decimal, not float" design point.

## Next

- Promote to ADR; then design the migration that keeps `price_per_1k_tokens` / `calculate_cost` as deprecated blended shims for one release.
