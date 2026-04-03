---
adr: "0005"
title: Google API-key auth is the default for all Google-backed providers
status: Accepted
date: "2026-04-03"
tags: [google, gemini, authentication, configuration]
must_read: false
supersedes: ~
superseded_by: ~
---

# ADR-0005: Google API-key auth is the default for all Google-backed providers

## Status
Accepted

## Context
Originally, all Google-backed providers assumed service-account credential-file authentication (`GOOGLE_APPLICATION_CREDENTIALS`). This created unnecessary friction for OSS developers and Gemini API users who use API keys.

## Decision
`GOOGLE_AUTH_METHOD` defaults to API-key authentication using `GOOGLE_API_KEY` across all Google-backed providers (completions, embeddings, images, voice). Service-account credential-file auth remains available via explicit `GOOGLE_AUTH_METHOD` configuration and is not removed.

## Consequences
- Simplified onboarding for the common Gemini/OSS developer use case.
- GCP-hosted deployments that require service-account auth must explicitly set `GOOGLE_AUTH_METHOD`.
- Introduced in commit `122586a`; made the library-wide default in `7748caf`.

## Evidence
Commit `122586a` ("Introduce `GOOGLE_AUTH_METHOD` environment variable"); commit `7748caf`; `env_template`; `src/ai_api_unified/ai_google_base.py`.
