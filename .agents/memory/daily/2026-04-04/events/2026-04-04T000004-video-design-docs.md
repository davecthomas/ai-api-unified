---
timestamp: "2026-04-04T00:00:04-07:00"
author: "davidcthomas"
branch: "codex/video-generation-design-plan"
thread_id: "video-design"
turn_id: "video-design-docs"
decision_candidate: false
ai_generated: true
ai_model: "claude-sonnet-4-6"
ai_tool: "claude"
ai_surface: "claude-code"
ai_executor: "local-agent"
related_adrs:
  - "0006"
  - "0007"
  - "0008"
files_touched:
  - "docs/video_generation_design.md"
  - "docs/video_generation_implementation_plan.md"
verification:
---

## Why

- First-class video generation support designed for ai_api_unified using the same stable-contract approach as completions, embeddings, images, and voice.
- Three providers: OpenAI Sora (deprecated 2026-04-04, shutdown 2026-09-24), Google Veo (primary), Bedrock Nova Reel.
- Phased rollout plan (A through I): shared contracts first, then Google Veo, Bedrock, OpenAI, frame helpers, observability, tests/docs.

## What changed

- Added `docs/video_generation_design.md` — full design covering `AIBaseVideos`, normalized job/result models, provider mappings, request/result types, observability, frame extraction helpers.
- Added `docs/video_generation_implementation_plan.md` — phased implementation plan.

## Evidence

- OpenAI Sora deprecated as of 2026-04-04, shutdown scheduled 2026-09-24.
- Google Veo models: veo-3.1-generate-preview, veo-3.0-generate-001, veo-2.0-generate-001.
- Bedrock Nova Reel: amazon.nova-reel-v1:1, requires caller-owned S3 output.

## Next

- Begin Phase A: shared video types, factory wiring, registry entries.
- Merge design docs from `codex/video-generation-design-plan` branch.
