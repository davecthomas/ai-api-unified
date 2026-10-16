# ADR-0025 Identify the routing configuration on every call, and check declared route requirements at build

Status: accepted
Date: 2026-10-16
Owners: dave-thomas
Must read: true
Supersedes: 
Superseded by: 
ai-generated: True
ai-tool: claude
ai-surface: claude-code
ai-executor: local-agent

Purpose: Let any call's record name the configuration that routed it, and let operators declare what every route must offer so a gap fails at startup
Derived from: owner requirements audit, configuration section (2026-10-16)

## Context

- A call's events named the tier, the method, and the model, but nothing
  tied them to the configuration in force, so a change in cost or quality
  could not be attributed to a configuration change.
- The model check (ADR-0022) proves each model exists and can be reached;
  it cannot know which capabilities callers need, or whether a route
  depends on a single provider.

## Decision

- **config_id is a hash of the configured routing**: primary, fallbacks,
  trigger reasons, router, cooldown, thinking level. Taken before policy
  and model-check pruning, so the same configuration gives the same id in
  every process even when a model is down at startup. 12 hex characters
  of SHA-256. **config_version** is an optional human label at the top of
  models.json.
- **Carried as observability tags.** Every client the factory builds is
  stamped; each request scope it opens pushes `config_id` and
  `config_version` as default tags, so per-call events, cost events, and
  the request cost report carry them with no change to the event schema.
  Caller-set tags win. The router and fallback log records add them as
  fields.
- **Requirements are declared, not inferred.** `completions.require` lists
  capabilities each route's primary must support and whether each route
  needs a second provider. The primary is what counts, since a call starts
  there and a primary without a capability raises. Capabilities come from
  each engine's static model table (`for_model`), so the check builds no
  client; an engine without a table is logged and not failed. `warn` by
  default, `strict` raises.

## Consequences

- Dashboards can group cost and errors by `tag_config_id` and compare
  configurations directly.
- A route that silently depends on one provider, or whose primary lacks a
  capability callers use, is reported at startup instead of at the first
  failing call.
- The capability check is as good as the engines' model tables; a wrong
  table entry gives a wrong answer, the same exposure the engines already
  have at call time.

## Related code paths

- src/ai_api_unified/routing/config_checks.py
- src/ai_api_unified/ai_factory.py
- src/ai_api_unified/ai_base.py
- src/ai_api_unified/middleware/request_cost.py
- src/ai_api_unified/ai_models_config.py
