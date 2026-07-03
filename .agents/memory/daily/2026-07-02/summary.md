# 2026-07-02 summary

## Snapshot

- Captured 2 memory events.
- Main work: Added AIEmbeddingsCapabilitiesBase, AIEmbeddingsMultimodalParams, SupportedDataType, and AIMediaReference contracts to the base layer, and a new AiProviderCapabilityUnsupportedError. Google embeddings gained AIEmbeddingsCapabilitiesGoogle.for_model with gemini-embedding-001 as text-only and gemini-embedding-2 as multimodal (interleaved text, images, video, audio, PDFs, with per-model limits such as a 6-image cap); OpenAI and Titan clients declare text-only capabilities and reject multimodal requests. Public exports, README Multimodal Embeddings docs, and a Google SDK dependency refresh accompany the feature.
- Top decision: The library needs multimodal embeddings without leaking per-provider differences to callers. Rather than letting unsupported inputs fail deep inside vendor SDKs, every embeddings client now declares what it supports up front via a capabilities descriptor, and unsupported input types raise a typed AiProviderCapabilityUnsupportedError before any API call. This mirrors the repo's established pattern of unified abstractions with intentional per-vendor behavior (ADR-0004) and unblocks the streaming-completions backlog queued behind this PR. ([2026-07-02 21:38:22 UTC by 2355287-davecthomas](events/2026-07-02T21-38-22Z--2355287-davecthomas--thread_451c3cb5-1bf1-4f06-911e-bb274923e3b1--turn_8d3e8bf539.md))
- Blockers: None.

| Metric | Value |
|---|---|
| Memory events captured | 2 |
| Repo files changed | 2 |
| Decision candidates | 1 |
| Active blockers | 0 |

## Major work completed

- Added AIEmbeddingsCapabilitiesBase, AIEmbeddingsMultimodalParams, SupportedDataType, and AIMediaReference contracts to the base layer, and a new AiProviderCapabilityUnsupportedError. Google embeddings gained AIEmbeddingsCapabilitiesGoogle.for_model with gemini-embedding-001 as text-only and gemini-embedding-2 as multimodal (interleaved text, images, video, audio, PDFs, with per-model limits such as a 6-image cap); OpenAI and Titan clients declare text-only capabilities and reject multimodal requests. Public exports, README Multimodal Embeddings docs, and a Google SDK dependency refresh accompany the feature.
- AICompletionsPromptParamsBase's inline image-only media validation was generalized into a new AIIncludedMediaParamsBase: subclasses declare accepted media types via DICT_ALLOWED_MIME_PREFIXES and an optional per-attachment-and-combined byte cap via MAX_MEDIA_BYTES, while the shared validator enforces list alignment, MIME/type consistency, non-empty bytes, and total-payload limits. The Gemini embeddings client's duplicated _embed_single closure was collapsed into a shared _embed_one_content_observed helper on the same retry/observability path, and the new base class joined the package's public API.

## Why this mattered

- The library needs multimodal embeddings without leaking per-provider differences to callers. Rather than letting unsupported inputs fail deep inside vendor SDKs, every embeddings client now declares what it supports up front via a capabilities descriptor, and unsupported input types raise a typed AiProviderCapabilityUnsupportedError before any API call. This mirrors the repo's established pattern of unified abstractions with intentional per-vendor behavior (ADR-0004) and unblocks the streaming-completions backlog queued behind this PR.
- Immediately after the capability-gated multimodal embeddings feature merged (PR #18), the codebase carried two divergent copies of media-attachment validation: the completions prompt params validated image-only attachments inline, while the new multimodal embeddings params re-implemented similar aligned-list checks. Consolidating them into one subclass-parameterized base keeps the capabilities-descriptor contract (see the 2026-07-02 decision candidate mirroring ADR-0004) enforceable in a single place, so future providers or media types extend ClassVar declarations instead of forking validator logic.

## Active blockers

- None

## Decision candidates

- The library needs multimodal embeddings without leaking per-provider differences to callers. Rather than letting unsupported inputs fail deep inside vendor SDKs, every embeddings client now declares what it supports up front via a capabilities descriptor, and unsupported input types raise a typed AiProviderCapabilityUnsupportedError before any API call. This mirrors the repo's established pattern of unified abstractions with intentional per-vendor behavior (ADR-0004) and unblocks the streaming-completions backlog queued behind this PR. ([2026-07-02 21:38:22 UTC by 2355287-davecthomas](events/2026-07-02T21-38-22Z--2355287-davecthomas--thread_451c3cb5-1bf1-4f06-911e-bb274923e3b1--turn_8d3e8bf539.md))

## Next likely steps

- Merge the multimodal embeddings PR, then start the queued streaming-completions workstream; consider promoting the capabilities-descriptor contract into an ADR alongside ADR-0004.
- Add or update tests covering the generalized media validation (MIME-prefix mismatches, byte caps, combined-payload limit) and open the dedup PR; the streaming-completions workstream remains queued behind the multimodal embeddings work.

## Relevant event shards

- [2026-07-02 21:38:22 UTC by 2355287-davecthomas](events/2026-07-02T21-38-22Z--2355287-davecthomas--thread_451c3cb5-1bf1-4f06-911e-bb274923e3b1--turn_8d3e8bf539.md)
- [2026-07-02 23:37:56 UTC by 2355287-davecthomas](events/2026-07-02T23-37-56Z--2355287-davecthomas--thread_451c3cb5-1bf1-4f06-911e-bb274923e3b1--turn_a350662c6f.md)
