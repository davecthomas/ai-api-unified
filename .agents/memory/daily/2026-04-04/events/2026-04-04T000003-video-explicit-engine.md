---
timestamp: "2026-04-04T00:00:03-07:00"
author: "davidcthomas"
branch: "codex/video-generation-design-plan"
thread_id: "video-design"
turn_id: "video-adr-0008"
decision_candidate: true
ai_generated: true
ai_model: "claude-sonnet-4-6"
ai_tool: "claude"
ai_surface: "claude-code"
ai_executor: "local-agent"
related_adrs:
  - "0008"
files_touched:
  - "docs/video_generation_design.md"
verification:
---

## Why

- `VIDEO_ENGINE` is required — no implicit provider default, consistent with ADR-0001.
- `VIDEO_MODEL_NAME` is optional; each provider owns its default model optimized for experimentation.
- Bedrock additionally requires `BEDROCK_VIDEO_OUTPUT_S3_URI` because the provider mandates caller-owned S3.

## What changed

- Design decision documented in `docs/video_generation_design.md`.
- Promoted to ADR-0008.

## Evidence

- `docs/video_generation_design.md` (§ Defaulting Strategy, § Provider Default Profiles)

## Next

- None
