# Video Generation Implementation Plan

## Summary

This plan breaks first-class video generation support into small phases that fit
the current repository architecture and dependency model.

The rollout covers:

- a new `videos` capability with a stable base interface
- provider implementations for OpenAI, Google Gemini API, and Bedrock Nova Reel
- explicit provider default profiles so developers can call the API without
  filling in every optional field
- helper utilities for extracting image frames from a video buffer and saving
  image buffers as files
- observability, tests, and end-user documentation

## Scope

In scope:

- add `AIBaseVideos` and shared video request/result models
- add `AIFactory.get_ai_video_client(...)`
- add `videos` provider registry entries and lazy loading
- add provider default profiles for OpenAI, Google, and Bedrock
- add blocking and job-oriented public methods
- add frame extraction and image-buffer persistence helpers
- add `videos` observability support
- add README, `env_template`, and docs updates

Out of scope for the first implementation:

- webhook-driven completion flows
- async Python client methods
- cancellation support unless the provider SDK path makes it trivial
- broad in-memory video payload defaults
- full parity for every provider-only advanced editing workflow on day one

## Design Guardrails

- preserve the current explicit-engine pattern; `VIDEO_ENGINE` must be set when
  video is used
- keep provider SDKs optional and behind the registry/lazy-loader boundary
- keep the simple path synchronous at the public API surface
- keep the default result file-backed, not raw-video-bytes-backed
- keep frame decoding behind its own optional dependency boundary
- reuse existing naming and helper patterns where practical

## Proposed Deliverables

### Public API

- `AIBaseVideos`
- `AIBaseVideoProperties`
- `AIVideoGenerationJob`
- `AIVideoGenerationResult`
- `AIVideoArtifact`
- `AIFactory.get_ai_video_client(...)`

### Helper Utilities

- `AIBaseVideos.extract_image_frames_from_video_buffer(...)`
- `AIBaseVideos.save_image_buffers_as_files(...)`
- `AIVideoArtifact.read_bytes()`

### Provider Implementations

- `AIOpenAIVideos`
- `AIGoogleGeminiVideos`
- `AINovaReelVideos`

## Phase Plan

### Phase A: Shared Contracts and Factory Wiring

Deliverables:

- add shared video types and status enum to `ai_base.py`
- add `AIBaseVideos`
- add `VIDEO_ENGINE` and `VIDEO_MODEL_NAME` support to `EnvSettings`
- add `AIFactory.get_ai_video_client(...)`
- add provider registry capability bucket and video provider entries
- export video base types from the package root

Targeted files:

- `src/ai_api_unified/ai_base.py`
- `src/ai_api_unified/ai_factory.py`
- `src/ai_api_unified/ai_provider_registry.py`
- `src/ai_api_unified/util/env_settings.py`
- `src/ai_api_unified/__init__.py`
- `src/ai_api_unified/videos/__init__.py`

Exit criteria:

- callers can request a video client through the factory
- missing engine selection raises a clear error
- video providers are resolved lazily like the existing capabilities

### Phase B: Default Profile Resolution

Deliverables:

- codify the provider default profiles described in the design doc
- keep portable defaults in `AIBaseVideoProperties`
- keep provider model defaults in concrete provider classes
- add shared helpers for resolving output directories, poll intervals, and
  timeout values
- add Bedrock-specific validation that requires an S3 output URI from either the
  request or environment

Targeted files:

- `src/ai_api_unified/ai_base.py`
- `src/ai_api_unified/util/env_settings.py`
- new shared helper module under `src/ai_api_unified/videos/`
- provider modules under `src/ai_api_unified/videos/`

Exit criteria:

- `client.generate_video("...")` has a deterministic default profile for each
  provider
- provider defaults are documented in code and tests
- Bedrock fails fast with a targeted message when no S3 output destination is
  available

### Phase C: Shared Job Polling and Artifact Normalization

Deliverables:

- add shared polling loop in `AIBaseVideos.wait_for_video_generation(...)`
- add status normalization helpers
- add output-directory materialization helpers
- add file-backed `AIVideoArtifact` behavior
- add `AIVideoArtifact.read_bytes()`

Targeted files:

- `src/ai_api_unified/ai_base.py`
- new shared helper module under `src/ai_api_unified/videos/`

Exit criteria:

- all providers can normalize into the same job/result shape
- the blocking public path is provider-agnostic
- artifacts are persisted locally when `download_outputs=True`

### Phase D: Google Gemini API Veo Provider

Deliverables:

- implement `AIGoogleGeminiVideos`
- wire `client.models.generate_videos(...)`
- wire long-running operation polling through `client.operations.get(...)`
- wire file download through `client.files.download(...)`
- support text-to-video
- support image-guided generation
- support reference images and video extension where the request shape already
  supports them cleanly

Targeted files:

- `src/ai_api_unified/videos/ai_google_gemini_videos.py`

Exit criteria:

- Google is the first fully working video provider
- the default no-extra-arguments path works with `VIDEO_ENGINE=google-gemini`

### Phase E: Bedrock Nova Reel Provider

Deliverables:

- implement `AINovaReelVideos`
- wire `start_async_invoke`
- wire async job retrieval
- add request-shape selection for:
  - single-shot text-to-video
  - image-guided generation
  - multi-shot automated
  - multi-shot manual
- add S3 output download normalization into local file artifacts

Targeted files:

- `src/ai_api_unified/videos/ai_bedrock_videos.py`
- `src/ai_api_unified/ai_bedrock_base.py` if shared helpers are needed

Exit criteria:

- Bedrock Nova Reel works through the same blocking and job-based interfaces
- the only required non-default blank is the caller-owned S3 destination

### Phase F: OpenAI Video Provider

Deliverables:

- implement `AIOpenAIVideos`
- wire the Videos API create/retrieve/content flows
- support plain text-to-video
- support image-guided generation in the initial rollout
- carry a clear code comment and README note that the provider is deprecated as
  of April 4, 2026, with shutdown scheduled for September 24, 2026

Targeted files:

- `src/ai_api_unified/videos/ai_openai_videos.py`

Exit criteria:

- OpenAI video generation works through the same normalized interface
- deprecation state is documented clearly without blocking use

### Phase G: Frame Extraction and Image Persistence Helpers

Deliverables:

- add one optional helper dependency boundary for video-frame decoding
- implement `extract_image_frames_from_video_buffer(...)`
- support:
  - `time_offsets_seconds`
  - `frame_indices`
- reject invalid selector combinations
- return image buffers in request order
- implement `save_image_buffers_as_files(...)`
- reuse the repository's conditional-dependency approach so missing helper
  extras raise a clear targeted error

Recommended dependency direction:

- add a new optional extra for frame extraction backed by a lazy-loaded
  `imageio` + `imageio-ffmpeg` helper stack

Targeted files:

- `src/ai_api_unified/videos/frame_helpers.py`
- possibly a new lazy helper module under `src/ai_api_unified/util/`
- `pyproject.toml`

Exit criteria:

- callers can extract frames by time offset or frame index from a video buffer
- callers can persist the resulting image buffers without extra dependencies
- missing frame-decoder dependencies produce clear install guidance

### Phase H: Observability Integration

Deliverables:

- add `videos` as a supported observability capability
- add a video observed-result container
- add video input/output metadata builders
- add optional `include_video_byte_count` config support
- instrument provider calls using the shared observability wrapper pattern

Targeted files:

- `src/ai_api_unified/ai_base.py`
- `src/ai_api_unified/middleware/middleware_config.py`
- `src/ai_api_unified/middleware/observability_runtime.py`
- `src/ai_api_unified/middleware/observability.py`
- provider files under `src/ai_api_unified/videos/`

Exit criteria:

- video calls emit metadata-only observability events when enabled
- existing config tests are updated to treat `videos` as valid

### Phase I: Tests, Docs, and Release Follow-Through

Deliverables:

- add unit and mocked-provider tests
- add optional live tests guarded by credentials and account availability
- update `README.md`
- update `env_template`
- add any needed troubleshooting notes
- review package version bump requirements

Targeted files:

- `tests/test_ai_factory_provider_loading.py`
- `tests/test_ai_provider_registry.py`
- new tests under `tests/` for each provider and helper area
- `README.md`
- `env_template`
- `docs/video_generation_design.md`

Exit criteria:

- docs match the shipped API and default profiles
- the helper APIs are covered by unit tests
- provider tests follow the same mock-first pattern already used elsewhere in
  the repository

## Test Plan Additions

Add unit coverage for:

- factory resolution for `videos`
- provider default model pass-through behavior
- provider default property selection
- normalized status mapping
- blocking wait timeout behavior
- output directory resolution
- Bedrock S3 output URI validation
- artifact `read_bytes()`
- frame extraction by time offsets
- frame extraction by frame indices
- image-buffer file persistence
- observability metadata for `videos`

Add optional live coverage for:

- Google Veo blocking generation path
- Bedrock Nova Reel blocking generation path
- OpenAI Sora blocking generation path while the provider remains available

## Dependency Review Checklist

- confirm the minimum `openai` version that exposes the required videos surface
- confirm the pinned `google-genai` version supports the exact Veo methods used
- confirm the pinned `boto3` version supports the required Nova Reel async APIs
- choose and pin the optional frame-decoder extra carefully to keep base installs
  small

## Recommended Sequencing

Recommended implementation order:

1. Phase A
2. Phase B
3. Phase C
4. Phase D
5. Phase E
6. Phase F
7. Phase G
8. Phase H
9. Phase I

This order lands the shared contract before providers, the primary Google path
before Bedrock/OpenAI, and the frame helper after the core video capability is
stable.
