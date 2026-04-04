---
timestamp: "2026-04-04T00:00:02-07:00"
author: "davidcthomas"
branch: "codex/video-generation-design-plan"
thread_id: "video-design"
turn_id: "video-adr-0007"
decision_candidate: true
ai_generated: true
ai_model: "claude-sonnet-4-6"
ai_tool: "claude"
ai_surface: "claude-code"
ai_executor: "local-agent"
related_adrs:
  - "0007"
files_touched:
  - "docs/video_generation_design.md"
verification:
---

## Why

- Video payloads are tens to hundreds of MB; returning raw `bytes` by default would cause memory spikes.
- Bedrock naturally delivers results to S3; file-backed output matches that model.
- `AIVideoArtifact` uses `file_path` as primary accessor with `read_bytes()` convenience for in-memory access.

## What changed

- Design decision documented in `docs/video_generation_design.md`.
- Promoted to ADR-0007.

## Evidence

- `docs/video_generation_design.md` (§ Why File-Backed Artifacts Are The Default)

## Next

- None
