# ADR-0027 Attribute cost by service, tenant, and purpose through standard observability tags

Status: accepted
Date: 2026-10-17
Owners: dave-thomas
Must read: true
Supersedes: 
Superseded by: 
ai-generated: True
ai-tool: claude
ai-surface: claude-code
ai-executor: local-agent

Purpose: Make cost attributable by service, tenant, and purpose without new event schemas, and keep metered tokens equal to what the provider bills
Derived from: owner requirements audit, token and cost metering (2026-10-17)

## Context

- Cost events and the request ledger (ADR-0020) carried provider, model,
  and route tier, but service, tenant, and purpose had no standard names;
  each application invented its own tags.
- Gemini reports thinking tokens apart from the answer and bills both as
  output; the engine counted the answer only, so cost understated every
  thinking call.

## Decision

- **Three standard tags** ride on the existing observability context, so
  every per-call event (`tag_<name>`) and the request report carry them:
  `service` from models.json, stamped on clients like `config_id`
  (ADR-0025); `tenant` set by the application; `purpose` from a reserved
  per-call `provider_options["purpose"]` key that the engine strips before
  the provider call and wrappers push before opening the request scope.
  The report exposes all three as fields.
- **Metered output is billed output.** Each engine's output count is what
  the provider bills, thinking included; where the provider breaks thinking
  out, the share is reported as `reasoning_tokens`, a subset that changes no
  cost.
- **Every line item names its rates**: `pricing_source` and
  `pricing_effective_date`, matching the per-call cost event.

## Consequences

- Dashboards group spend by `tag_service`, `tag_tenant`, `tag_purpose`,
  `route_tier`, provider, and model with no custom tagging.
- Gemini spend recorded by 3.5.0 through 3.13.0 is understated; the
  changelog says so.

## Related code paths

- src/ai_api_unified/middleware/request_cost.py
- src/ai_api_unified/middleware/observability_runtime.py
- src/ai_api_unified/ai_base.py
- src/ai_api_unified/completions/ai_google_gemini_completions.py
- src/ai_api_unified/ai_factory.py
