# ADR index

| ADR | Title | Status | Date | Tags | Must Read | Supersedes | Superseded By |
|---|---|---|---|---|---|---|---|
| "0001" | [lazy provider loading with optional extras](ADR-0001-lazy-provider-loading.md) | Accepted | "2026-04-03" | poc | true | ~ |  |
| "0002" | [middleware profile as sole source of truth; opaque to callers](ADR-0002-middleware-yaml-driven.md) | Accepted | "2026-04-03" | poc | true | ~ |  |
| "0003" | [distinct middleware roles — text-transform vs lifecycle/observability](ADR-0003-two-middleware-roles.md) | Accepted | "2026-04-03" | poc | true | ~ |  |
| "0004" | [tool-calling abstraction with intentional per-vendor API choices](ADR-0004-tool-calling-abstraction.md) | Accepted | "2026-04-03" | poc | true | ~ |  |
| "0005" | [API-key auth is the default for all Google-backed providers](ADR-0005-google-api-key-auth-default.md) | Accepted | "2026-04-03" | poc | true | ~ |  |
| "0006" | [public video API wrapping provider async job models](ADR-0006-video-sync-api-over-async-jobs.md) | Accepted | "2026-04-04" | poc | true | ~ |  |
| "0007" | [video artifacts as default output representation](ADR-0007-video-file-backed-artifacts.md) | Accepted | "2026-04-04" | poc | true | ~ |  |
| "0008" | [VIDEO_ENGINE selection required with provider-owned model defaults](ADR-0008-video-explicit-engine-selection.md) | Accepted | "2026-04-04" | poc | true | ~ |  |
| ADR-0009 | [Replace the blended token rate with a structured per-modality, Decimal-typed pricing descriptor](ADR-0009-replace-the-blended-token-rate-with-a-structured-per-modality-decimal-typed-pricing-descriptor.md) | accepted | 2026-07-07 | pricing_research.md | true |  |  |
| ADR-0010 | [Store pricing in a versioned registry keyed by provider and model, separate from model classes, with per-entry provenance](ADR-0010-store-pricing-in-a-versioned-registry-keyed-by-provider-and-model-separate-from-model-classes-with-per-entry-provenance.md) | accepted | 2026-07-07 | pricing_research.md | true |  |  |
| ADR-0011 | [Compute cost from real provider-reported token usage and split rates, deprecating the blended calculate_cost for one release](ADR-0011-compute-cost-from-real-provider-reported-token-usage-and-split-rates-deprecating-the-blended-calculate-cost-for-one-release.md) | accepted | 2026-07-07 | pricing_research.md | true |  |  |
