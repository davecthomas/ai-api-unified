---
adr: "0008"
title: Explicit VIDEO_ENGINE selection required with provider-owned model defaults
status: Accepted
date: "2026-04-04"
tags: [architecture, video, configuration, providers]
must_read: false
supersedes: ~
superseded_by: ~
---

# ADR-0008: Explicit VIDEO_ENGINE selection required with provider-owned model defaults

## Status
Accepted

## Context
The library's existing pattern requires explicit engine selection (`COMPLETIONS_ENGINE`, `IMAGE_ENGINE`, etc.) rather than implicit provider fallback. Video generation adds three providers with very different cost, capability, and infrastructure profiles — OpenAI Sora (deprecated, shutting down 2026-09-24), Google Veo, and Bedrock Nova Reel (requires caller-owned S3). An implicit default would create surprising cost or infrastructure requirements.

## Decision
`VIDEO_ENGINE` is required when video is used. `VIDEO_MODEL_NAME` is optional — when unset, each provider applies its own default model optimized for ease of experimentation: OpenAI defaults to `sora-2`, Google to `veo-3.1-lite-generate-preview`, Bedrock to `amazon.nova-reel-v1:1`. Provider defaults live in the concrete provider classes, not the factory.

Bedrock additionally requires `BEDROCK_VIDEO_OUTPUT_S3_URI` (or per-request `s3_output_uri`) because the provider mandates a caller-owned S3 destination. The library fails fast with a clear message when this is missing.

## Consequences
- No accidental cost surprises from implicit provider selection.
- Provider model defaults favor low-friction experimentation over maximum quality.
- Factory stays small; cross-provider defaults don't leak into unrelated engines.
- Consistent with ADR-0001's explicit-engine pattern.

## Evidence
`docs/video_generation_design.md` (§ Defaulting Strategy, § Provider Default Profiles, § Why Bedrock Still Needs One Explicit Blank Filled In, § Why Provider Defaults Should Stay in Provider Classes).
