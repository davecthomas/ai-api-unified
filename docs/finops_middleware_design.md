# Financial-ops observability middleware — design

Prepared 2026-07-07, revised the same day. Builds on the model pricing work
(2.9.0) to emit auditable per-call cost events. This is a design for review; no
code is written yet.

## Goal

Attach real USD cost to every provider call, computed from the tokens the
provider actually reported, and emit it as a structured, auditable event. Cost
events feed billing, budgeting, and per-caller attribution downstream.

## Framing: cost is a topic on the observability event, not a new subsystem

The logger-backed observability middleware already emits a structured event in
`after_call` carrying the model, provider, and token counts. Cost is a *derived
field* on that same event — there is no new data, hook, or lifecycle. So v1
finops is, in effect, a special-topic logger: the existing observability event
plus a `usd_cost` field, emitted on a cost topic that handlers can route
separately.

This replaces an earlier plan for a `CompositeObservabilityMiddleware` chaining
a separate cost middleware. That composite is scaffolding for a separation that
does not exist in observe-only v1, so it is deferred (see "When separation earns
its keep" below).

## Where it fits

Per ADR-0003 the library has two middleware roles: text-transform (PII) and
lifecycle/observability. Cost tracking is a lifecycle/observability concern — it
observes a completed call and emits metadata, changing nothing about the request
or response. Per ADR-0002 middleware is YAML-driven, so finops is a flag on the
existing observability settings, not a new middleware profile entry.

The observability lifecycle already exposes the exact hook and data needed.
`AiApiObservabilityMiddleware.after_call(call_context, call_result_summary)`
receives:

- `call_context`: `provider_vendor`, `provider_engine`, `model_name`,
  `capability`, `operation`, `call_id`, and the resolved `originating_caller_id`.
- `call_result_summary`: `provider_prompt_tokens`, `provider_completion_tokens`,
  `provider_total_tokens`, `finish_reason`, `provider_elapsed_ms`, and
  `dict_metadata`.

That is enough to look up `AIModelPricing` for `(provider, model)` and call
`compute_token_cost(input_tokens=..., output_tokens=...)`.

## Design

### Middleware — no new class in v1

Cost is a config-gated enrichment inside the existing logger-backed
observability `after_call`, not a new middleware. When `emit_cost` is on, the
middleware:

1. Resolves pricing: maps `call_context.provider_vendor` to the registry
   provider label and looks up `get_model_pricing(provider, model_name)`. Skips
   silently when the model is unpriced.
2. Reads `provider_prompt_tokens` and `provider_completion_tokens`.
3. Computes `usd_cost = pricing.compute_token_cost(input_tokens=prompt,
   output_tokens=completion)`.
4. Emits the cost event on the cost topic (see below).

Gating on `emit_cost` is why the computation lives in the observability emit
path rather than the per-provider result-summary builder: it runs only when
observability is enabled, and pricing stays out of the provider call path. The
Decimal math is cheap either way, so the config gate decides placement.

### Logger — a dedicated cost topic

The cost event rides the existing structured event on a dedicated topic: a child
logger (`ai_api_unified.observability.cost`) or an `event_type: "cost"` field on
the emitted record. Either lets an application's logging handlers filter and
route cost events to a separate destination (a cost index, a metrics exporter)
without any new abstraction. This is the "special-topic logger": Python logging
already provides topic-based routing via logger names, handlers, and filters.

The cost event fields:

```
call_id, event_time_utc, provider, model, capability, operation, caller_id,
input_tokens, output_tokens, cached_input_tokens (None in v1),
usd_cost (Decimal), currency,
pricing_effective_date, pricing_source, pricing_confidence
```

Carrying the pricing provenance (effective date, source, confidence) on each
event is what makes the cost auditable — a stored event explains exactly which
rate produced it.

### Config — one flag on the observability settings

No new middleware profile entry. `ObservabilitySettingsModel` gains:

- `emit_cost: bool` (default false) — turn cost enrichment on.
- an optional cost topic / logger-name override.

Enabling finops is one flag under the existing observability block, consistent
with the YAML-driven, single-middleware model.

## When separation earns its keep (deferred)

Cost stops being "just a log field" when events must reach a **billing-grade
numeric destination in-process** — a metrics counter (Prometheus), a cost
database, a billing API. Pushing Decimal money values through log strings and
re-parsing them downstream is fragile; those systems want structured numeric
events. That is the real justification for a pluggable sink:

```python
class AiApiCostSink(ABC):
    def record(self, event: AiApiCostEvent) -> None: ...
```

But this is a *destination* choice, not a reason for a second middleware, and
observe-only v1 does not need it. It becomes a Phase-2 config option (a
`cost_sink` selector) — an alternate emitter on the same middleware — added when
a non-log destination actually forces it. The default remains the cost log
topic.

## Decisions (confirmed 2026-07-07)

1. **Scope: observe-only.** v1 computes cost per call and emits events; it never
   affects program flow. Budgets and enforcement are a later layer that consumes
   the event stream.

2. **Emission: cost log topic by default.** A pluggable `AiApiCostSink` for
   non-log destinations is deferred to Phase 2, added when a billing/metrics
   destination requires it.

3. **Cached-token accuracy: shipped in Phase 3 (2.12.0).** The result summary
   now carries `provider_cached_input_tokens`, and each provider hook extracts
   its cache-read count (OpenAI `prompt_tokens_details.cached_tokens`, OpenAI
   Responses `input_tokens_details.cached_tokens`, Anthropic
   `cache_read_input_tokens`, Bedrock `cacheReadInputTokens`, Gemini
   `cached_content_token_count`). Providers that report cache reads separately
   from the input count (Anthropic, Bedrock) fold them into
   `provider_prompt_tokens` so the cached subset is uniform across providers.
   The cost middleware bills the cached subset at the cached rate and the
   remainder at the full input rate, and the cost event carries
   `cached_input_tokens`. Scope is cache *reads*: cache *writes* (priming a
   cache, billed above the base input rate — Bedrock exposes them as
   `cacheWriteInputTokens`) are not captured or billed yet, because the pricing
   registry does not model a cache-write rate. That is a follow-on once a
   cache-write rate column lands.

4. **Aggregation: none in-library.** Per-call events only, correlated via the
   `caller_id` / session / workflow ids already on the context. Rollups are the
   consumer's job.

## Phased plan

- **Phase 1 (v1)** — cost enrichment in the logger-backed observability
  `after_call`, gated by `emit_cost`; cost event on a dedicated log topic with
  pricing provenance; `ObservabilitySettingsModel.emit_cost` + topic override.
  No new middleware class.
- **Phase 2** — `AiApiCostSink` + a `cost_sink` config selector for non-log
  destinations, when one is needed. The default stays the cost log topic.
- **Phase 3 (shipped, 2.12.0)** — cached-token capture in the result summary
  (per provider) so cost reflects cache discounts. `provider_cached_input_tokens`
  on the result summary; the cost middleware splits cached from full-rate input.
- **Later** — budgets/alerting on the event stream, if wanted.
