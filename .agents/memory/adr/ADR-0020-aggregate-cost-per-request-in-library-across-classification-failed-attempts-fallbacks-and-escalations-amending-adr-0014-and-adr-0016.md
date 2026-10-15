# ADR-0020 Aggregate cost per request in-library across classification, failed attempts, fallbacks, and escalations, amending ADR-0014 and ADR-0016

Status: accepted
Date: 2026-10-15
Owners: dave-thomas
Must read: true
Supersedes: 
Superseded by: 
ai-generated: True
ai-tool: claude
ai-surface: claude-code
ai-executor: local-agent

Purpose: Keep a per-request ledger of every provider call so one request has one total, amending the "no in-library aggregation" rule of ADR-0014 and ADR-0016 while leaving per-call events unchanged
Derived from: GitHub issue #78 (2026-10-15)

## Context

- ADR-0014 and ADR-0016 decided that the library does no aggregation: per-call
  `ai_api_call_cost` events only, correlated by the `caller_id` / session /
  workflow ids on the context, with rollups the consumer's job. That held
  while one public call was one provider call.
- The fallback chain (2.x) and the complexity router (3.0.0) broke that
  equivalence. One `send_structured_output` can be four provider calls: a
  classifier call, a 529 on the first tier, a fallback on the same tier, and
  an escalation that answers. The library emitted four unrelated events and
  no total; a failed call emitted only `ai_api_call_error` with no usage; the
  wrappers emitted nothing on the cost topic; and the router priced the
  classifier ad hoc in its own log line, ignoring cache tokens.
- The request total is the number the routing cost rationale depends on
  (classifier spend plus how often escalation buys a second answer), the unit
  teams budget and charge back by, the measure of waste (fallback and
  escalation spend on answers nobody uses), and the input any future budget
  limit needs. Rebuilding it from log events per request is the consumer's
  job only if the library tells them which four events belong together; it
  did not.

## Decision

- A **request scope** lives in a contextvar. The outermost public call opens
  it: `AiRouterCompletions` and `AiFallbackCompletions` around their attempt
  loops (sync, async, and streaming), and a plain engine's observed call when
  no scope is active. Callers open one with `request_cost_scope(name)` to
  roll several library calls into one logical request; nested scopes join the
  outer one.
- **One line item per provider call**, recorded in `execute_observed_call`
  and its async and streaming twins from the same result summary the per-call
  event uses, priced by the same `compute_call_cost` function so the two
  cannot disagree. A failed call is a line item with no cost and no tokens;
  it still counts as a call. Line items are recorded whether or not the
  logger-backed middleware is enabled, so the report is always available
  programmatically (`client.last_request_cost`, `AIRequestCostReport`).
- **Served and wasted.** A wrapper marks the last successful attempt as
  served when it returns. `wasted_usd` is the cost of billed attempts that
  are neither served nor the classifier.
- **Emission.** When the outermost scope closes, normally or on an exception,
  one `ai_api_request_cost` event goes to the cost topic when `emit_cost` is
  on. Per-call events are unchanged and carry `tag_request_id`; the router's
  `route_id` is the request id, and the classifier call now carries the
  `route_id` tag too.
- **What stays the consumer's job:** rollups across requests (per day, per
  team, per workflow). The library aggregates within a request and nowhere
  else; ADR-0013's observe-only scope and ADR-0016's cost-topic emission are
  unchanged.

## Consequences

- One request has one total, one status, and one `served_by`, joined to its
  per-call events by `request_id`. A dashboard can show cost per request,
  classifier overhead, and wasted spend without reconstructing call groups.
- Failed attempts are visible on the cost topic for the first time, as
  zero-cost line items inside the request event.
- Billed failures that raise inside the observed callable (invalid JSON, a
  truncated structured answer) are costed from 3.4.0: the exception carries
  the provider's `usage` (`StructuredOutputValidationError`,
  `StructuredResponseTokenLimitError`), the runtime builds the line item and
  a cost event (`outcome: "error"`) from it through the non-abstract
  `on_billed_error` hook, and `AiFallbackReason.INVALID_OUTPUT` (opt-in) lets
  the chain and the router move on, leaving the discarded answer as a wasted
  line item. Exceptions carrying usage were chosen over restructuring every
  engine to parse outside the observed call: one shared parser in the base
  raises the typed error, and the engines' observed callables are unchanged.
- The scope is a contextvar, so a streaming request holds it open between
  chunks in the consumer's context; a library call interleaved with stream
  consumption joins that request. The scope resets when the stream ends or is
  abandoned.

## Related code paths

- src/ai_api_unified/middleware/request_cost.py
- src/ai_api_unified/middleware/observability_runtime.py
- src/ai_api_unified/middleware/observability.py
- src/ai_api_unified/ai_base.py
- src/ai_api_unified/completions/ai_fallback_completions.py
- src/ai_api_unified/completions/ai_router_completions.py
- docs/finops_middleware_design.md
