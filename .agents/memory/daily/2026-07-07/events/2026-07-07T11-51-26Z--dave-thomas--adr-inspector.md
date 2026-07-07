---
timestamp: "2026-07-07T11:51:26Z"
author: "dave-thomas"
branch: "main"
thread_id: "adr-inspector"
turn_id: "pricing-adr-c"
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

- Cost today is computed by `calculate_cost(num_tokens)` — one blended rate times a single token count. Once input, cached-input, and output carry asymmetric rates, blending is wrong regardless of how good the rate table is, because it discards the actual input/output mix of each call. The decision is to change the cost-computation contract: cost is derived from the real, provider-reported usage — the `provider_prompt_tokens` and `provider_completion_tokens` the observability runtime already captures — multiplied by the split rates from the pricing descriptor, yielding an auditable cost event rather than an estimate. This is why the pricing redesign must land before the finops middleware: the middleware hooks `after_call`, reads actual token counts, looks up `AIModelPricing`, and emits `{model, input_tokens, output_tokens, cached_tokens, usd_cost, effective_date, source}`; without split rates and real usage there is nothing auditable to emit.
- The decision also fixes the migration policy: because dropping `price_per_1k_tokens` / `calculate_cost` is a breaking public-API change, they are retained as thin deprecated shims (blended = input+output midpoint) for exactly one release so callers migrate without a break, consistent with the library's semver conventions. This bounds the compatibility surface to a single release rather than leaving the old blended path indefinitely.

## What changed

- `pricing_research.md` specifies replacing `calculate_cost(num_tokens)` with a cost function driven by real provider-reported input/output token counts and the split per-modality rates, consumed by the planned finops observability middleware on `after_call`, and keeping the old blended API as deprecated one-release shims.

## Evidence

- `pricing_research.md` § "API recommendation" — design point "Compute cost from real usage, not a blend … takes the provider-reported input and output token counts and applies the split rates," and the deprecation-shim paragraph ("Keep the old `price_per_1k_tokens` / `calculate_cost` as thin deprecated shims … for one release").
- `pricing_research.md` § "Finops connection (next, per backlog)" — the `after_call` cost-event contract and the ordering rationale (pricing first, middleware second).

## Next

- Promote to ADR; wire the finops middleware onto the observability lifecycle once the pricing registry exists.
