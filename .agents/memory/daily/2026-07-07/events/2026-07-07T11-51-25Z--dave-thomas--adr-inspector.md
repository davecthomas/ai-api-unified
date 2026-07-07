---
timestamp: "2026-07-07T11:51:25Z"
author: "dave-thomas"
branch: "main"
thread_id: "adr-inspector"
turn_id: "pricing-adr-b"
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

- Today rate data is embedded inside the provider/model classes. That conflates two things that change on completely different schedules and for different reasons: a model class describes fixed behavior and capabilities, whereas a rate is external data the provider revises on its own timeline. The research calls out the consequence — the in-code numbers are stale and partly invented, several advertised models are retired or past their shutdown date, and no rate carries an effective date or source, so no cost figure is auditable. A finops layer cannot stand on unauditable, drift-prone numbers.
- The decision is to store pricing outside the model classes in a dedicated versioned pricing registry keyed by `(provider, model)`, where every entry carries `effective_date`, `source` (the official pricing URL), and a `confidence` level. Provenance-per-entry is what makes a cost auditable and is called out as the single most important change for finops. The registry is surfaced through the existing capabilities descriptors — a `pricing: AIModelPricing | None` field resolved by the same `for_model()` path already in place — so callers read `client.capabilities.pricing` without the pricing data leaking back into the behavior classes. This preserves the separation of "what a model does" from "what a model costs right now."

## What changed

- `pricing_research.md` establishes pricing as data owned by a versioned registry keyed by `(provider, model)`, separate from provider classes, with per-entry `effective_date` / `source` / `confidence` for auditability, exposed via the capabilities `for_model()` resolution path. The doc also inventories model health (retired gemini-1.5 models, deprecated gemini-2.0 family, imagen-4.0 shutdown) that the registry's provenance is meant to prevent recurring.

## Evidence

- `pricing_research.md` § "API recommendation" — design point "Store pricing outside the model classes … a dedicated versioned pricing registry keyed by `(provider, model)`, with `effective_date` and `source` per entry … the single most important change for finops," and "Expose it through capabilities … resolved by the same `for_model()` path."
- `pricing_research.md` § "The core finding" items 5 (no provenance) and 6 (stale and partly invented).
- `pricing_research.md` § "Model health" (retired / deprecated models currently advertised in code).

## Next

- Promote to ADR; define the registry storage format and seed it from the sourced tables in this research.
