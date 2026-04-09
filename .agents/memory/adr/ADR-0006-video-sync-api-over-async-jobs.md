---
adr: "0006"
title: Sync-friendly public video API wrapping provider async job models
status: Accepted
date: "2026-04-04"
tags: [architecture, video, api-design, async, providers]
must_read: true
supersedes: ~
superseded_by: ~
---

# ADR-0006: Sync-friendly public video API wrapping provider async job models

## Status
Accepted

## Context
All three video generation providers in scope (OpenAI Sora, Google Veo, Bedrock Nova Reel) use asynchronous job patterns — submit, poll, download. Exposing a native async Python API would force callers into `asyncio`, which conflicts with the library's existing synchronous contract for completions, images, and voice.

## Decision
The `AIBaseVideos` interface is synchronous at the public API surface. A blocking `generate_video(...)` convenience method internally submits a job, polls until terminal, and downloads artifacts. The underlying async job reality is exposed through explicit `submit_video_generation(...)`, `wait_for_video_generation(...)`, and `download_video_result(...)` methods — all synchronous, all returning normalized `AIVideoGenerationJob` and `AIVideoGenerationResult` models.

The library normalizes provider job states into a shared `AIVideoGenerationStatus` enum: `queued`, `running`, `completed`, `failed`, `cancelled`.

## Consequences
- The simple path is one method call: `client.generate_video("prompt")`.
- Callers needing finer control can use the explicit job methods without switching to async Python.
- Provider-specific job/operation IDs are preserved in `provider_metadata` for debugging.
- No async Python surface in v1; can be added later without breaking the sync contract.

## Evidence
`docs/video_generation_design.md` (§ Proposed Public API, § Key Design Decision); `docs/video_generation_implementation_plan.md` (Phase C).
