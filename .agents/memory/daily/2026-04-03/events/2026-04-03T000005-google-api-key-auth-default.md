---
type: decision_candidate
decision_candidate: true
timestamp: "2026-04-03T00:00:05-07:00"
bootstrapped_at: "2026-04-03T21:00:00Z"
title: Google API-key auth is the default for all Google-backed providers
tags: [google, gemini, authentication, configuration]
---

# Google API-key auth is the default for all Google-backed providers

`GOOGLE_AUTH_METHOD` defaults to API-key authentication (`GOOGLE_API_KEY`) for all Google-backed providers (completions, embeddings, images, voice). Service-account credential-file auth (`GOOGLE_APPLICATION_CREDENTIALS`) remains available via explicit configuration but is no longer the assumed default.

**Rationale:** Simplifies onboarding and aligns with typical OSS/developer usage of Gemini, where API keys are the standard credential mechanism. Service-account flow is preserved for GCP-hosted deployments that require it.

**Evidence:** commit `122586a` ("Introduce `GOOGLE_AUTH_METHOD` environment variable"); commit `7748caf` ("align provider defaults with current models"); `env_template`; `src/ai_api_unified/ai_google_base.py`.
