# ADR-0022 Check every configured completions model for existence and reachability at client build, memoized by credentials

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

Purpose: Make the configured-model check prove that each model can be called, not only that it exists, and define exactly when it runs and when a result goes stale
Derived from: owner direction and an independent audit (2026-10-16)

## Context

- The configured-model check (3.1.0) ran only when a fallback chain or a
  router was configured, and asked only whether the provider serves the
  model. A missing key, a rejected key, or a missing extra was
  "unverifiable" and the model was kept with an INFO line.
- Fallbacks and tiers are built lazily. A fallback that could not be
  reached therefore stayed in the configured chain until the first
  failover, where its build failed and it was dropped. The chain was
  shorter than configured, and nobody found out until an outage.
- The owner's expectation: when configuration is loaded, every model is
  checked to ensure the process can reach it, and the result is cached
  until the configuration changes.

## Decision

- **Every completions build runs the check**, a lone primary included,
  over the primary, each fallback, each tier candidate, and the router's
  classifier. `model_check.mode` keeps its three values.
- **A new status, `unreachable`**: no key, a key the provider rejects as
  invalid (401, Gemini `API_KEY_INVALID`, Bedrock `UnrecognizedClient`,
  `InvalidSignature`, `ExpiredToken`, no AWS credentials), no base URL for
  an OpenAI-compatible server, or the extra not installed. It prunes like
  `missing` (WARNING in `prune`, raises in `strict`). An unreachable
  primary raises `AiProviderConfigurationError` after the engine's own
  constructor has had the chance to raise its typed error.
- **A 403 stays `unverifiable`.** A permission denial on the catalog call
  (OpenAI, Anthropic, Gemini 403, Bedrock `AccessDenied` on the control
  plane) does not prove the inference call fails, so pruning on it could
  remove a working model.
- **The memo key is (engine, model, base_url, credential fingerprint).**
  Each probe registers the settings it reads; the checker hashes their
  values (SHA-256, truncated) into the key. A model is probed again
  exactly when its id, base URL, or credentials change; a changed
  `models.json` re-probes only the models it changes. Keys never appear in
  the memo.
- **`error` results are not memoized**, so a transient outage at startup
  does not hide a broken model for the life of the process.

## Consequences

- A process with a lone primary now makes one metadata call per model at
  its first factory call; later calls are free.
- A fallback chain at startup is the chain that can actually serve.
- AWS credentials that come from an instance role or an SSO cache are not
  in the fingerprint; a rotation there does not force a recheck unless
  `AWS_PROFILE` or the key variables change, or `reset_cache()` is called.
- The mocked suite replaces every registered probe with an offline one, so
  no mocked test reaches a provider from the check.

## Related code paths

- src/ai_api_unified/util/ai_model_checker.py
- src/ai_api_unified/util/ai_model_probes.py
- src/ai_api_unified/ai_factory.py
- src/ai_api_unified/completions/ai_fallback_completions.py
- tests/conftest.py
