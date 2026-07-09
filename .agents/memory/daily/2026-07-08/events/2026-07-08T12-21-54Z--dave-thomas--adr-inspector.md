---
agentmemory_version: "0.4.4"
timestamp: "2026-07-08T12:21:54Z"
author: "dave-thomas"
branch: "finops-cached-tokens-pii-fixes"
thread_id: "adr-inspector-finops-cached-tokens"
turn_id: "adr-inspector"
workstream_id: "thread-adr-inspector-finops-cached-tokens"
workstream_scope: "thread"
episode_id: "episode-finops-cached-tokens"
episode_scope: "mixed"
decision_candidate: true
enriched: true
ai_generated: true
ai_tool: "claude"
ai_surface: "claude-code"
ai_executor: "local-agent"
related_adrs:
files_touched:
  - "docs/finops_middleware_design.md"
design_docs_touched:
  - "docs/finops_middleware_design.md"
---

## Why

- v1 finops billed every input token at the full input rate, which systematically
  overcharges any call that hits a provider prompt cache — cache reads are billed
  far below the base input rate, and for cache-heavy workloads they are a large
  share of input tokens. Ignoring them makes the "real cost" the middleware emits
  wrong in the one direction that matters for budgeting: too high.
- The hard part is that providers report cache reads inconsistently. OpenAI
  Chat exposes `prompt_tokens_details.cached_tokens`, OpenAI Responses exposes
  `input_tokens_details.cached_tokens`, Anthropic exposes `cache_read_input_tokens`,
  Bedrock exposes `cacheReadInputTokens`, and Gemini exposes
  `cached_content_token_count`. Some providers count cache reads *inside* the
  input token count; others report them *separately*. Billing correctly requires
  one uniform representation the cost middleware can reason about without
  per-provider branching at compute time.
- The decision: capture a single `provider_cached_input_tokens` field on the
  result summary, have each provider hook extract its own cache-read count, and
  normalize so the cached subset is always contained within
  `provider_prompt_tokens` (Anthropic and Bedrock, which report reads separately,
  fold them in). The cost middleware then splits the input: cached subset at the
  cached rate, remainder at the full input rate, and emits `cached_input_tokens`
  on the cost event for auditability.
- The accepted tradeoff: scope is cache *reads* only. Cache *writes* (priming a
  cache, billed above the base input rate, exposed by Bedrock as
  `cacheWriteInputTokens`) are deliberately not captured or billed, because the
  pricing registry (ADR-0010) has no cache-write rate column. Rather than model a
  rate that does not exist, the decision defers write billing until that column
  lands. This keeps the emitted cost accurate for the reads case and honest about
  what it does not yet cover.

## What changed

- The finops middleware design (`docs/finops_middleware_design.md`) records
  Phase 3 (shipped, 2.12.0): the result summary carries
  `provider_cached_input_tokens`; each provider hook extracts its cache-read count
  from that provider's usage payload; providers that report reads separately fold
  them into `provider_prompt_tokens` so the cached subset is uniform across
  providers; the cost middleware bills the cached subset at the cached rate and
  the remainder at the full input rate; the cost event carries
  `cached_input_tokens`. Cache writes are explicitly out of scope pending a
  cache-write rate in the pricing registry.

## Evidence

- `docs/finops_middleware_design.md`, "Decisions (confirmed 2026-07-07)" item 3
  ("Cached-token accuracy: shipped in Phase 3 (2.12.0)"), lists the exact
  per-provider extraction keys, the fold-into-`provider_prompt_tokens`
  normalization for Anthropic and Bedrock, the split-rate billing rule, and the
  cache-writes-deferred tradeoff citing the missing `cacheWriteInputTokens` rate.
- Same doc, "Phased plan" Phase 3 entry, confirms
  `provider_cached_input_tokens` on the result summary and the cached/full-rate
  input split as the shipped 2.12.0 behavior.
- Builds on ADR-0010 (versioned pricing registry keyed by provider and model) —
  the absence of a cache-write rate column there is the stated reason writes are
  deferred.

## Next

- Promote this candidate to an ADR capturing the uniform cached-input-token
  representation and split-rate billing decision.
- Revisit when a cache-write rate column is added to the pricing registry, which
  would unblock capturing and billing `cacheWriteInputTokens` and supersede the
  reads-only scope stated here.
