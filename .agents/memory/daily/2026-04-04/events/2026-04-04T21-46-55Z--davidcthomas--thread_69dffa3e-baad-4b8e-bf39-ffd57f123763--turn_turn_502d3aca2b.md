---
timestamp: "2026-04-04T21:46:55Z"
author: "davidcthomas"
branch: "codex/video-generation-design-plan"
thread_id: "69dffa3e-baad-4b8e-bf39-ffd57f123763"
turn_id: "turn_502d3aca2b"
decision_candidate: true
ai_generated: true
ai_model: "claude-unknown"
ai_tool: "claude"
ai_surface: "claude-code"
ai_executor: "local-agent"
related_adrs:
  - "ADR-0006"
  - "ADR-0007"
  - "ADR-0008"
files_touched:
  - "README.md"
  - "env_template"
  - "poetry.lock"
  - "pyproject.toml"
  - "src/ai_api_unified/__init__.py"
  - "src/ai_api_unified/ai_base.py"
  - "src/ai_api_unified/ai_factory.py"
  - "src/ai_api_unified/ai_provider_registry.py"
  - "src/ai_api_unified/middleware/middleware_config.py"
  - "src/ai_api_unified/middleware/observability.py"
  - "src/ai_api_unified/util/env_settings.py"
  - "tests/test_ai_provider_registry.py"
  - "tests/test_observability_middleware_phase_a.py"
verification:
  - "git diff:  13 files changed, 944 insertions(+), 10 deletions(-); `ai-api-unified` is a unified Python library for AI completions, embeddings, image generation, video generation, and voice. Application code targets stable base interfaces and factory entry points while concrete providers are selected at runtime from environment configuration.; - `COMPLETIONS_ENGINE`, `EMBEDDING_ENGINE`, `IMAGE_ENGINE`, `VIDEO_ENGINE`, and `AI_VOICE_ENGINE` must be set explicitly when that capability is used.; | Videos      | `AIBaseVideos`      | `openai`, `google-gemini`, `nova-reel` and Bedrock video aliases                                                              | `openai`, `google_gemini`, `bedrock`, `video_frames` |"
---

## Why

- You're right — this project uses the shared repo memory under `.agents/memory/`, not the `~/.claude/` user-level memory. Everything I've written so far is already in the right place (`.agents/memory/adr/` and `.agents/memory/daily/`). There's nothing to create under `~/.claude/`.

## Repo changes

- Updated README.md
- Updated env_template
- Updated poetry.lock
- Updated pyproject.toml
- Updated src/ai_api_unified/__init__.py
- Updated src/ai_api_unified/ai_base.py
- Updated src/ai_api_unified/ai_factory.py
- Updated src/ai_api_unified/ai_provider_registry.py
- Updated src/ai_api_unified/middleware/middleware_config.py
- Updated src/ai_api_unified/middleware/observability.py
- Updated src/ai_api_unified/util/env_settings.py
- Updated tests/test_ai_provider_registry.py
- Updated tests/test_observability_middleware_phase_a.py

## Evidence

- git diff:  13 files changed, 944 insertions(+), 10 deletions(-); `ai-api-unified` is a unified Python library for AI completions, embeddings, image generation, video generation, and voice. Application code targets stable base interfaces and factory entry points while concrete providers are selected at runtime from environment configuration.; - `COMPLETIONS_ENGINE`, `EMBEDDING_ENGINE`, `IMAGE_ENGINE`, `VIDEO_ENGINE`, and `AI_VOICE_ENGINE` must be set explicitly when that capability is used.; | Videos      | `AIBaseVideos`      | `openai`, `google-gemini`, `nova-reel` and Bedrock video aliases                                                              | `openai`, `google_gemini`, `bedrock`, `video_frames` |

## Next

- Review the generated shard and summary, then explicitly commit and push them with the related code changes if ready.
