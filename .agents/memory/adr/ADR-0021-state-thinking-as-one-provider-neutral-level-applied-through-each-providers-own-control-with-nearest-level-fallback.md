# ADR-0021 State thinking as one provider-neutral level, applied through each provider's own control, with nearest-level fallback

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

Purpose: Configure how much a model thinks as a standard level from off to max, map each level to the provider's control inside the engine, and apply the nearest honored level when a model cannot honor the requested one
Derived from: owner-specified feature requirements (2026-10-15)

## Context

- Every provider now exposes a thinking control, and none agree on its
  shape: Anthropic adaptive thinking with `output_config.effort`
  (low|medium|high|xhigh|max) on 4.6 and later and `budget_tokens` before
  that, with models whose thinking cannot be turned off; OpenAI
  `reasoning_effort` (none|minimal|low|medium|high|xhigh|max, per model);
  Gemini `thinking_level` on 3.x and `thinking_budget` on 2.5; Bedrock the
  Claude fields under `additionalModelRequestFields`.
- Until now the library had no thinking setting. A caller who wanted it
  passed a provider-specific `provider_options` key, which a fallback chain
  or router then dropped or forwarded to the wrong engine (#53), and the
  default was whatever the provider did when the field was omitted: off on
  some models, on at the provider's default on others.

## Decision

- **One level, six values.** `AIThinkingLevel`: off, minimal, low, medium,
  high, max. Configuration (`COMPLETIONS_THINKING`, `completions.thinking`
  in models.json), the factory (`thinking=`), the client
  (`thinking_level`), and a call (`other_params.thinking`, or the reserved
  `thinking_level` provider option) all speak this type. Neither
  application code nor configuration names a provider's own field.
- **Engines own the mapping.** The provider tables live in one module
  (`thinking.py`) with no SDK imports, so the Anthropic engine and the
  Bedrock engine share the Claude mapping. Each engine merges the fields it
  derives before `provider_options`, so a caller's native option still
  wins.
- **Capabilities declare honored levels.** `capabilities.thinking` lists the
  standard levels a model applies exactly (OFF absent means always on;
  None means no control). A requested level outside the list resolves to
  the nearest one, ties to the lower level, since the caller asked for no
  more than the requested level. Both levels are recorded on the result
  (`AIThinkingResolution`: requested, applied, exact, reason), on every
  completions observability event, and in an `ai_thinking_event` log line.
- **Default medium.** The library asks for medium when nothing is
  configured, and each provider's medium is what it maps to. This turns
  thinking on at medium for models where omitting the field meant off; the
  changelog says so, and `off` restores the old shape where the model
  allows it.
- **A caller's native field is recorded, never silent** (amended
  2026-10-17, 3.12.0). A provider thinking field in `provider_options` is
  still honored, but the layer then applies nothing of its own, records the
  call as `applied` empty with reason `provider_option`, and warns once per
  client and field, naming the neutral option.
- **Classifiers run at their own level, default off** (amended 2026-10-17):
  `classifier.thinking`, so routing decisions do not spend thinking tokens.
- **Batches apply the level per item** (amended 2026-10-17), and the
  submitted job records each request's resolution.
- **Engine-side exceptions are recorded, never silent.** A forced
  `tool_choice`, an assistant prefill, or a `max_tokens` too small for the
  minimum budget turns thinking off on models that allow it, with the
  reason on the resolution.

## Consequences

- One config line sets thinking across a fallback chain or router whose
  tiers span providers, and a request that fails over keeps its level.
- The per-provider tables are a maintenance point: a new model family lands
  in `thinking.py`. A wrong entry surfaces as a provider 400, not a silent
  downgrade, which is the honest failure; the live test exercises the
  configured models.
- Cost and latency rise for models that were off by default; the request
  cost ledger (ADR-0020) shows the difference per request.

## Related code paths

- src/ai_api_unified/thinking.py
- src/ai_api_unified/ai_base.py
- src/ai_api_unified/ai_factory.py
- src/ai_api_unified/ai_models_config.py
- src/ai_api_unified/completions/ai_anthropic_completions.py
- src/ai_api_unified/completions/ai_openai_completions.py
- src/ai_api_unified/completions/ai_openai_responses_completions.py
- src/ai_api_unified/completions/ai_google_gemini_completions.py
- src/ai_api_unified/completions/ai_bedrock_completions.py
