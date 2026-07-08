# ADR-0017 Normalize provider cache-read tokens into a uniform cached-input subset and bill it at the cached rate, deferring cache-write billing

Status: accepted
Date: 2026-07-08
Owners: dave-thomas
Must read: true
Supersedes: 
Superseded by: 
ai-generated: True
ai-tool: claude
ai-surface: claude-code
ai-executor: local-agent

Purpose: Normalize provider cache-read tokens into a uniform cached-input subset and bill it at the cached rate, deferring cache-write billing
Derived from: [2026-07-08T12-21-54Z--dave-thomas--adr-inspector](../daily/2026-07-08/events/2026-07-08T12-21-54Z--dave-thomas--adr-inspector.md)

## Context

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

## Decision

- The finops middleware design (`docs/finops_middleware_design.md`) records
  Phase 3 (shipped, 2.12.0): the result summary carries
  `provider_cached_input_tokens`; each provider hook extracts its cache-read count
  from that provider's usage payload; providers that report reads separately fold
  them into `provider_prompt_tokens` so the cached subset is uniform across
  providers; the cost middleware bills the cached subset at the cached rate and
  the remainder at the full input rate; the cost event carries
  `cached_input_tokens`. Cache writes are explicitly out of scope pending a
  cache-write rate in the pricing registry.

## Consequences

- Promote this candidate to an ADR capturing the uniform cached-input-token
  representation and split-rate billing decision.
- Revisit when a cache-write rate column is added to the pricing registry, which
  would unblock capturing and billing `cacheWriteInputTokens` and supersede the
  reads-only scope stated here.

## Source memory events

- [2026-07-08T12-21-54Z--dave-thomas--adr-inspector](../daily/2026-07-08/events/2026-07-08T12-21-54Z--dave-thomas--adr-inspector.md)

## Related code paths

- docs/finops_middleware_design.md
