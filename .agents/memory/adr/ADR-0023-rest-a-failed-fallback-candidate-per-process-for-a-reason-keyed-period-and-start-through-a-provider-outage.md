# ADR-0023 Rest a failed fallback candidate per process for a reason-keyed period, and start through a provider outage

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

Purpose: Stop every request from re-paying a failing candidate's backoff, and keep a provider outage at startup from becoming a service outage
Derived from: owner requirements audit (2026-10-16)

## Context

- Each call through a fallback chain started at the primary. During an
  outage every call paid the primary's full retry backoff and sent one
  more doomed request before failing over.
- ADR-0022 made the build refuse a primary that is missing or cannot be
  reached with its credentials. The audit agreed for configuration errors
  but asked that a provider that is down at startup not stop the client.
- Every engine except Gemini is built without calling its provider. The
  Gemini engine calls models.get while it is built and raised on any
  failure, so a Gemini outage at startup stopped the build.

## Decision

- **Cooldown per candidate.** A trigger failure (a reason in the chain's
  fallback_on) benches the failed candidate for a period keyed to the
  reason: 30 s for UNAVAILABLE, RATE_LIMITED, and TIMEOUT; 5 min for
  QUOTA_EXHAUSTED and MODEL_UNAVAILABLE; 1 h for AUTH_REJECTED; none for
  INVALID_OUTPUT. An account-wide failure benches the account. Configured
  as `completions.cooldown` in models.json or the factory's `cooldown=`;
  on by default.
- **Reorder, never remove.** A resting candidate moves to the end of a
  call's order. It is still tried when everything ahead of it fails, so a
  cooldown cannot fail a call that would otherwise be served; with all
  candidates resting the chain's own order applies.
- **Process-wide and credential-aware.** One registry is shared by every
  chain and router tier. Each bench records the engine's credential
  fingerprint (ADR-0022); a bench taken with other credentials is dropped.
  A candidate that serves a call ends its own bench.
- **Startup through an outage.** The Gemini engine treats a 5xx, a 429
  rate limit, a timeout, or a refused connection on its init-time
  models.get as an outage: it logs a WARNING and returns the client. A bad
  key, a denial, an unknown model, or a daily quota still raises. With
  that, no engine needs its provider up to be built, and ADR-0022's model
  check already keeps an `error` result. A primary that is down at startup
  serves through its chain and is retried each time its rest ends; it is
  never replaced for the life of the process.

## Consequences

- During an outage, one call per rest period pays the primary's backoff
  instead of every call.
- A primary that recovers mid-rest waits out the rest (at most 30 s for
  outage reasons) before it serves again, unless every other candidate
  fails first.
- Tests must reset the registry; the mocked suite's conftest does it per
  test.

## Related code paths

- src/ai_api_unified/completions/ai_fallback_cooldown.py
- src/ai_api_unified/completions/ai_fallback_completions.py
- src/ai_api_unified/completions/ai_router_completions.py
- src/ai_api_unified/ai_factory.py
- src/ai_api_unified/ai_google_base.py
- src/ai_api_unified/ai_models_config.py
