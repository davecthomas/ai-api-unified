# ADR index

| ADR | Title | Status | Date | Tags | Must Read | Supersedes | Superseded By |
|---|---|---|---|---|---|---|---|
| [0001](ADR-0001-lazy-provider-loading.md) | Registry-based lazy provider loading with optional extras | Accepted | 2026-04-03 | architecture, providers, dependency-management, semver | yes | - | - |
| [0002](ADR-0002-middleware-yaml-driven.md) | YAML-driven middleware profile as sole source of truth; opaque to callers | Accepted | 2026-04-03 | architecture, middleware, configuration, pii | yes | - | - |
| [0003](ADR-0003-two-middleware-roles.md) | Two distinct middleware roles — text-transform vs lifecycle/observability | Accepted | 2026-04-03 | architecture, middleware, observability, design-pattern | no | - | - |
| [0004](ADR-0004-tool-calling-abstraction.md) | Unified tool-calling abstraction with intentional per-vendor API choices | Accepted | 2026-04-03 | architecture, tool-calling, openai, gemini, bedrock | yes | - | - |
| [0005](ADR-0005-google-api-key-auth-default.md) | Google API-key auth is the default for all Google-backed providers | Accepted | 2026-04-03 | google, gemini, authentication, configuration | no | - | - |
| [0006](ADR-0006-video-sync-api-over-async-jobs.md) | Sync-friendly public video API wrapping provider async job models | Accepted | 2026-04-04 | architecture, video, api-design, async, providers | yes | - | - |
| [0007](ADR-0007-video-file-backed-artifacts.md) | File-backed video artifacts as default output representation | Accepted | 2026-04-04 | architecture, video, artifacts, memory-management | no | - | - |
| [0008](ADR-0008-video-explicit-engine-selection.md) | Explicit VIDEO_ENGINE selection required with provider-owned model defaults | Accepted | 2026-04-04 | architecture, video, configuration, providers | no | - | - |
