# Video Generation Design

## Summary

This document proposes first-class video generation support in
`ai_api_unified` using the same stable-contract approach already used for
completions, embeddings, images, and voice.

The design adds:

- a new `videos` capability with a stable `AIBaseVideos` interface
- one factory entrypoint: `AIFactory.get_ai_video_client()`
- a provider registry bucket for `videos`
- provider implementations for:
  - OpenAI video generation
  - Google Gemini API Veo video generation
  - Amazon Bedrock Nova Reel video generation
- a single model-agnostic return model that hides provider-specific job and
  artifact handling
- configuration defaults that keep the simple path short while still allowing
  provider-specific overrides
- cross-provider helper utilities for:
  - extracting image frames from a resulting video buffer by time offset or
    frame index
  - saving a list of image buffers as image files

This design is intentionally sync-friendly at the public API layer, even though
every currently relevant provider uses an asynchronous job pattern under the
hood.

## Design Goals

- add first-class video generation without changing the established factory and
  base-interface architecture
- keep application code provider-agnostic for common text-to-video and
  image-guided generation flows
- hide provider-specific job, operation, and download mechanics behind one
  normalized interface
- avoid returning large video payloads as raw `bytes` by default
- preserve explicit provider selection through environment configuration
- keep optional provider dependencies optional
- fit cleanly into existing observability and lazy-loading patterns

## Non-Goals

- do not add implicit provider defaults; `VIDEO_ENGINE` remains explicit
- do not add webhook orchestration in v1
- do not attempt to unify every provider-only creative feature into the base
  request shape on day one
- do not change existing completions, embeddings, images, or voice public APIs
- do not make video generation an async Python API surface in v1

## Provider Surface in Scope

This section reflects the provider documentation reviewed on April 4, 2026.

### OpenAI

OpenAI exposes video generation through the Videos API and Sora 2 model family.
The current documented models are:

- `sora-2`
- `sora-2-pro`

Important lifecycle note:

- OpenAI documents the Videos API and Sora 2 models as deprecated and scheduled
  to shut down on September 24, 2026.

Documented capabilities relevant to this design:

- asynchronous job creation and polling
- MP4 download after completion
- image-guided generation
- video extension
- video editing

### Google Gemini API

Google exposes video generation through `client.models.generate_videos(...)`
with long-running operations. The documented Gemini API model set includes:

- `veo-3.1-generate-preview`
- `veo-3.1-fast-generate-preview`
- `veo-3.1-lite-generate-preview`
- `veo-3.0-generate-001`
- `veo-3.0-fast-generate-001`
- `veo-2.0-generate-001`

Documented capabilities relevant to this design:

- asynchronous operation polling
- file download after completion
- text-to-video
- image-guided generation
- reference images
- last-frame interpolation
- video extension
- aspect ratio and resolution controls

### Amazon Bedrock

Amazon Nova Reel uses Bedrock async invocation. The relevant in-scope model is:

- `amazon.nova-reel-v1:1`

Documented capabilities relevant to this design:

- async invoke only
- S3-backed output storage
- text-to-video
- text-and-image-to-video
- automated multi-shot generation
- manual multi-shot generation
- 6-second increments up to two minutes
- fixed `1280x720` resolution at `24` FPS

## Why Existing Image Patterns Are Not Sufficient

The current image interface is:

- `generate_images(prompt, properties) -> list[bytes]`

That pattern does not translate well to video for three reasons:

1. Video generation is asynchronous across all providers in scope.
2. Video payloads are materially larger than image payloads, so returning raw
   `bytes` by default is a bad default for memory pressure and ergonomics.
3. Provider completion artifacts are delivered in three different ways:
   - OpenAI: content endpoint download
   - Google: generated file handle download
   - Bedrock: S3 output location

The video design therefore needs both:

- a blocking convenience method for the common case
- a normalized job model for the underlying async reality

## Proposed Public API

### New Stable Interface

Add a new base class in `src/ai_api_unified/ai_base.py`:

```python
class AIBaseVideos(AIBase):
    def generate_video(
        self,
        video_prompt: str,
        video_properties: AIBaseVideoProperties = AIBaseVideoProperties(),
    ) -> AIVideoGenerationResult:
        ...

    @abstractmethod
    def submit_video_generation(
        self,
        video_prompt: str,
        video_properties: AIBaseVideoProperties = AIBaseVideoProperties(),
    ) -> AIVideoGenerationJob:
        ...

    @abstractmethod
    def get_video_generation_job(
        self,
        job: str | AIVideoGenerationJob,
    ) -> AIVideoGenerationJob:
        ...

    @abstractmethod
    def download_video_result(
        self,
        job: str | AIVideoGenerationJob,
    ) -> AIVideoGenerationResult:
        ...

    def wait_for_video_generation(
        self,
        job: str | AIVideoGenerationJob,
        *,
        timeout_seconds: int | None = None,
        poll_interval_seconds: int | None = None,
    ) -> AIVideoGenerationJob:
        ...

    @staticmethod
    def extract_image_frames_from_video_buffer(
        video_buffer: bytes,
        *,
        time_offsets_seconds: list[float] | None = None,
        frame_indices: list[int] | None = None,
        image_format: str = "png",
    ) -> list[bytes]:
        ...

    @staticmethod
    def save_image_buffers_as_files(
        image_buffers: list[bytes],
        *,
        output_dir: Path,
        root_file_name: str = "video_frame",
        image_format: str = "png",
    ) -> list[Path]:
        ...
```

### Public Behavior

The intended behavior is:

- `generate_video(...)`
  - submit provider job
  - poll until terminal state
  - download or materialize final artifacts
  - return a normalized result object
- `submit_video_generation(...)`
  - return immediately with a normalized job handle
- `wait_for_video_generation(...)`
  - poll provider state using normalized status mapping
- `download_video_result(...)`
  - fetch provider artifacts and normalize them into local/file-backed outputs
- `extract_image_frames_from_video_buffer(...)`
  - decode a video buffer and return image buffers for requested frame captures
- `save_image_buffers_as_files(...)`
  - persist arbitrary image buffers as sequential image files

This keeps the simple path to one method call while still exposing the async job
surface when callers need it.

### Frame Helper Semantics

The frame helper behavior should be:

- accept either:
  - `time_offsets_seconds`
  - `frame_indices`
- reject calls that provide both or neither
- return image buffers in the same order requested by the caller
- default to `png` output buffers
- raise a clear dependency error when the optional frame-decoding extra is not
  installed

The file-writing helper should:

- take the list of image buffers returned by the frame helper
- write sequential files such as `video_frame_1.png`, `video_frame_2.png`, and
  so on
- return the written `Path` objects
- require no extra dependency beyond the standard library

### Normalized Result Types

Add shared video models in `src/ai_api_unified/ai_base.py`.

```python
class AIVideoGenerationStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AIVideoArtifact(BaseModel):
    mime_type: str = "video/mp4"
    file_path: Path | None = None
    remote_uri: str | None = None
    width: int | None = None
    height: int | None = None
    duration_seconds: int | None = None
    fps: int | None = None
    has_audio: bool | None = None
    provider_metadata: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict
    )


class AIVideoGenerationJob(BaseModel):
    job_id: str
    provider_job_id: str
    status: AIVideoGenerationStatus
    progress_percent: float | None = None
    submitted_at_utc: datetime | None = None
    completed_at_utc: datetime | None = None
    error_message: str | None = None
    provider_engine: str
    provider_model_name: str | None = None
    provider_metadata: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict
    )


class AIVideoGenerationResult(BaseModel):
    job: AIVideoGenerationJob
    artifacts: list[AIVideoArtifact]
    provider_metadata: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict
    )
```

### Why File-Backed Artifacts Are The Default

`AIVideoArtifact` should be file-first rather than `bytes`-first.

Rationale:

- it avoids surprising memory spikes
- it matches how Bedrock naturally delivers results
- it still works for OpenAI and Google after library-managed downloads
- it makes downstream developer usage simple:
  - inspect `artifact.file_path`
  - copy or upload that file elsewhere
  - optionally call a helper like `read_bytes()` later if needed

The library should still allow opt-in in-memory materialization later, but that
should not be the default behavior.

To keep the frame helper ergonomic with this file-first design, `AIVideoArtifact`
should also expose a simple convenience method:

```python
def read_bytes(self) -> bytes:
    ...
```

That allows a caller to do this without learning provider-specific download
behavior:

```python
video_buffer = result.artifacts[0].read_bytes()
frames = AIBaseVideos.extract_image_frames_from_video_buffer(
    video_buffer,
    time_offsets_seconds=[0.0, 1.5, 3.0],
)
```

## Request Model

### Portable Base Properties

Add a new shared `AIBaseVideoProperties` model:

```python
class AIBaseVideoProperties(BaseModel):
    duration_seconds: int | None = None
    aspect_ratio: str | None = None
    resolution: str | None = None
    fps: int | None = None
    num_videos: int = 1
    seed: int | None = None
    output_format: str = "mp4"
    poll_interval_seconds: int = 10
    timeout_seconds: int = 900
    output_dir: Path | None = None
    download_outputs: bool = True
```

These fields intentionally cover the portable core only.

### Portable Caller Defaults

When the caller omits `video_properties`, the library should behave as if it
received:

```python
AIBaseVideoProperties(
    duration_seconds=None,
    aspect_ratio=None,
    resolution=None,
    fps=None,
    num_videos=1,
    seed=None,
    output_format="mp4",
    poll_interval_seconds=10,
    timeout_seconds=900,
    output_dir=None,
    download_outputs=True,
)
```

Those portable defaults are then completed by provider-specific defaults when a
provider field remains unset.

### Provider-Specific Property Subclasses

Like images, video should allow provider-specific property subclasses when the
provider surface is wider than the portable base.

#### OpenAI

```python
class AIOpenAIVideoProperties(AIBaseVideoProperties):
    reference_image: AIMediaReference | None = None
    source_video: AIMediaReference | None = None
```

#### Google Gemini

```python
class AIGoogleGeminiVideoProperties(AIBaseVideoProperties):
    reference_images: list[AIMediaReference] = Field(default_factory=list)
    image: AIMediaReference | None = None
    source_video: AIMediaReference | None = None
    last_frame_image: AIMediaReference | None = None
    person_generation: str | None = None
```

#### Bedrock Nova Reel

```python
class AIVideoShot(BaseModel):
    text: str
    image: AIMediaReference | None = None


class AINovaReelVideoProperties(AIBaseVideoProperties):
    shots: list[AIVideoShot] = Field(default_factory=list)
    negative_prompt: str | None = None
    s3_output_uri: str | None = None
```

### Shared Media Reference Model

To avoid repeating raw `bytes | Path | str` unions throughout the API, introduce
one small helper model:

```python
class AIMediaReference(BaseModel):
    bytes_data: bytes | None = None
    file_path: Path | None = None
    remote_uri: str | None = None
    mime_type: str
```

This is useful for video, and later it can be reused by images or multimodal
completions if the repository wants to converge more media inputs on one type.

## Example Usage

### Common Blocking Path

```python
from ai_api_unified import AIFactory, AIBaseVideos, AIBaseVideoProperties

client: AIBaseVideos = AIFactory.get_ai_video_client()
result = client.generate_video(
    "A cinematic tracking shot of a neon train passing through the desert at dusk.",
    AIBaseVideoProperties(),
)

print(result.artifacts[0].file_path)
```

### Explicit Async Job Control

```python
from ai_api_unified import AIFactory, AIBaseVideos, AIBaseVideoProperties

client: AIBaseVideos = AIFactory.get_ai_video_client()
job = client.submit_video_generation(
    "A stop-motion paper city waking up at sunrise.",
    AIBaseVideoProperties(timeout_seconds=1800),
)

job = client.wait_for_video_generation(job)
result = client.download_video_result(job)
print(result.artifacts[0].file_path)
```

## Factory and Registry Changes

### `AIFactory`

Add:

- `AIFactory.get_ai_video_client(...)`

Signature:

```python
@staticmethod
def get_ai_video_client(
    model_name: str | None = None,
    video_engine: str | None = None,
) -> AIBaseVideos:
    ...
```

### New Environment Selectors

Add:

- `VIDEO_ENGINE`
- `VIDEO_MODEL_NAME`

### Provider Registry

Add a new provider capability bucket:

- `AI_PROVIDER_CAPABILITY_VIDEOS = "videos"`

Registry entries should cover:

- `("videos", "openai")`
- `("videos", "google-gemini")`
- `("videos", "bedrock")`
- `("videos", "nova")`
- `("videos", "nova-reel")`

Suggested concrete classes:

- `ai_api_unified.videos.ai_openai_videos.AIOpenAIVideos`
- `ai_api_unified.videos.ai_google_gemini_videos.AIGoogleGeminiVideos`
- `ai_api_unified.videos.ai_bedrock_videos.AINovaReelVideos`

## Configuration Changes

### New `EnvSettings` Fields

Add at minimum:

- `VIDEO_ENGINE: str | None = None`
- `VIDEO_MODEL_NAME: str | None = None`
- `VIDEO_POLL_INTERVAL_SECONDS: int | None = None`
- `VIDEO_TIMEOUT_SECONDS: int | None = None`
- `VIDEO_OUTPUT_DIR: str | None = None`
- `BEDROCK_VIDEO_OUTPUT_S3_URI: str | None = None`

Optional later additions:

- `BEDROCK_VIDEO_OUTPUT_S3_KMS_KEY_ID`
- `VIDEO_DOWNLOAD_OUTPUTS`

### Defaulting Strategy

The defaulting policy should reduce friction without reintroducing implicit
provider selection.

Rules:

- `VIDEO_ENGINE` is required when video is used
- `VIDEO_MODEL_NAME` is optional
- when `VIDEO_MODEL_NAME` is unset, each provider may apply a provider-specific
  default model
- provider defaults should optimize for ease of experimentation rather than
  maximum quality or maximum cost

Recommended provider defaults:

- OpenAI: `sora-2`
- Google Gemini: `veo-3.1-lite-generate-preview`
- Bedrock: `amazon.nova-reel-v1:1`

Rationale:

- OpenAI default should favor faster iteration
- Veo default should favor lower friction and lower cost for first use
- Nova Reel currently has a single practical default model in scope

### Provider Default Profiles

The implementation should make the no-extra-arguments path explicit and
predictable. When a developer writes only:

```python
client = AIFactory.get_ai_video_client()
result = client.generate_video("...")
```

the library should select the following defaults.

#### OpenAI Default Profile

- `VIDEO_ENGINE=openai`
- `VIDEO_MODEL_NAME` default: `sora-2`
- request mode: plain text-to-video
- duration default: `8` seconds
- size default: `1280x720`
- aspect ratio default: `16:9`
- number of videos default: `1`
- poll interval default: `10` seconds
- timeout default: `900` seconds
- output location default:
  - `VIDEO_OUTPUT_DIR` when configured
  - otherwise a deterministic temporary directory managed by the library

#### Google Gemini Default Profile

- `VIDEO_ENGINE=google-gemini`
- `VIDEO_MODEL_NAME` default: `veo-3.1-lite-generate-preview`
- request mode: plain text-to-video
- duration default:
  - omit the field and rely on the provider model default clip length
  - as of April 4, 2026, Veo 3.1 docs describe the current default path as
    producing 8-second clips
- resolution default: `720p`
- aspect ratio default: `16:9`
- number of videos default: `1`
- poll interval default: `10` seconds
- timeout default: `900` seconds
- output location default:
  - `VIDEO_OUTPUT_DIR` when configured
  - otherwise a deterministic temporary directory managed by the library

#### Bedrock Nova Reel Default Profile

- `VIDEO_ENGINE=nova-reel` or `VIDEO_ENGINE=bedrock`
- `VIDEO_MODEL_NAME` default: `amazon.nova-reel-v1:1`
- request mode default: single-shot text-to-video
- duration default: `6` seconds
- resolution default: `1280x720`
- FPS default: `24`
- number of videos default: `1`
- seed default: provider-generated random seed
- poll interval default: `10` seconds
- timeout default: `1800` seconds
- output S3 destination default:
  - `BEDROCK_VIDEO_OUTPUT_S3_URI` when configured
  - otherwise fail fast because Bedrock requires a caller-owned S3 destination
- local download location default:
  - `VIDEO_OUTPUT_DIR` when configured
  - otherwise a deterministic temporary directory managed by the library

### Why Bedrock Still Needs One Explicit Blank Filled In

OpenAI and Google can complete the blocking flow with only credentials and
engine selection.

Bedrock cannot be fully zero-config because the provider requires an output S3
destination that belongs to the caller. The design should therefore optimize
everything else, but keep this one requirement explicit and well-documented.

### Why Provider Defaults Should Stay in Provider Classes

This matches the current images and embeddings pattern where the factory passes
through `None` and lets the provider own model selection. That keeps the
factory small and avoids cross-provider defaults leaking into unrelated engines.

## Provider Mapping Strategy

### OpenAI Mapping

`AIOpenAIVideos` should normalize:

- `submit_video_generation(...)`
  - `client.videos.create(...)`
- `get_video_generation_job(...)`
  - `client.videos.retrieve(...)` or equivalent SDK call
- `download_video_result(...)`
  - content download endpoint into a local file

Portable field mapping:

- `duration_seconds` -> provider `seconds`
- `aspect_ratio` / `resolution` -> provider `size`
- `reference_image` -> image-guided generation input
- `source_video` -> extension or edit flow when applicable

Important note:

- because OpenAI video generation is already deprecated as of April 4, 2026,
  implementation should mark the provider docstrings and README notes clearly,
  but still support it while it remains available

### Google Mapping

`AIGoogleGeminiVideos` should normalize:

- `submit_video_generation(...)`
  - `client.models.generate_videos(...)`
- `get_video_generation_job(...)`
  - `client.operations.get(...)`
- `download_video_result(...)`
  - `client.files.download(...)`

Portable field mapping:

- `aspect_ratio` -> `GenerateVideosConfig.aspect_ratio`
- `resolution` -> `GenerateVideosConfig.resolution`
- `num_videos` -> `GenerateVideosConfig.number_of_videos`
- `reference_images` -> `GenerateVideosConfig.reference_images`
- `last_frame_image` -> `GenerateVideosConfig.last_frame`
- `source_video` -> extension flow

### Bedrock Mapping

`AINovaReelVideos` should normalize:

- `submit_video_generation(...)`
  - `bedrock-runtime.start_async_invoke(...)`
- `get_video_generation_job(...)`
  - `bedrock-runtime.get_async_invoke(...)`
- `download_video_result(...)`
  - download final artifact from the provider S3 output location

Portable field mapping:

- `duration_seconds`
  - for single-shot requests, clamp or validate to provider-supported increments
- `fps`
  - validate to `24`
- `resolution`
  - validate to `1280x720`
- `shots`
  - select multi-shot automated or manual payload shapes

Bedrock-specific friction reduction:

- if `s3_output_uri` is not set on the properties object, fall back to
  `BEDROCK_VIDEO_OUTPUT_S3_URI`
- if neither is set, fail fast with a clear message that explains Bedrock video
  outputs require an S3 destination

## Model Validation Strategy

Video model identifiers change quickly, especially for preview releases.

To avoid turning the library into a stale allow-list, the design should use this
rule:

- `list_model_names()` returns the curated model IDs the library knows about and
  tests against
- non-empty override model names are allowed even when they are not in that
  curated list
- providers may log a warning for unknown model IDs, but they should not reject
  them solely because the local list is older than the provider

This is especially important for Google preview models and dated OpenAI model
revisions.

## Return Normalization Rules

The normalized result object should hide provider differences using these rules.

### Job IDs

- OpenAI: use provider video ID as `provider_job_id`
- Google: use operation name as `provider_job_id`
- Bedrock: use invocation ARN or invocation ID as `provider_job_id`

### Status Mapping

Normalize to:

- `queued`
- `running`
- `completed`
- `failed`
- `cancelled`

Store raw provider status in `provider_metadata`.

### Artifact Materialization

When `download_outputs=True`:

- OpenAI downloads completed MP4 content into `output_dir`
- Google downloads generated file content into `output_dir`
- Bedrock downloads the completed S3 artifact into `output_dir`

When `download_outputs=False`:

- `AIVideoArtifact.file_path` may be `None`
- `AIVideoArtifact.remote_uri` should still be populated when available
- frame extraction remains available when the caller first materializes a video
  buffer from the artifact or remote content

### Output Directory Behavior

Recommended default behavior:

- if `output_dir` is provided, save there
- else if `VIDEO_OUTPUT_DIR` is set, save there
- else save into a deterministic temporary directory and return the resolved file
  path

This keeps the one-call API useful without forcing callers to manage paths
up-front.

## Observability

Video generation should become a first-class observability capability.

### New Capability Token

Add:

- `CAPABILITY_VIDEOS = "videos"`

This affects:

- middleware capability validation
- observability runtime metadata
- docs and YAML examples

### New Result Container

Add:

- `AiApiObservedVideosResultModel`

Suggested fields:

- `return_value`
- `generated_video_count`
- `total_output_bytes`
- `provider_input_tokens`
- `provider_total_tokens`
- `dict_metadata`

### New Optional Middleware Setting

Add:

- `include_video_byte_count: bool = True`

Recommended input-side metadata:

- `prompt_char_count`
- `requested_duration_seconds`
- `requested_aspect_ratio`
- `requested_resolution`
- `requested_num_videos`
- `reference_image_count`
- `has_source_video`
- `download_outputs`

Recommended output-side metadata:

- `generated_video_count`
- `total_output_bytes`
- `artifact_count`
- `job_id`
- `provider_status`
- `has_audio`

## Error Model

Video should follow the current provider error model.

Expected failure categories:

- missing provider extra
- missing engine selector
- unsupported engine selector
- missing provider credentials
- provider job failed
- provider job timed out
- provider returned no artifact
- Bedrock S3 output location not configured

Recommended typed additions:

- `AiVideoGenerationTimeoutError`
- `AiVideoGenerationFailedError`

These can remain capability-specific exceptions layered on top of the existing
provider dependency/configuration/runtime exceptions.

## Repository Layout

Add:

- `src/ai_api_unified/videos/__init__.py`
- `src/ai_api_unified/videos/ai_openai_videos.py`
- `src/ai_api_unified/videos/ai_google_gemini_videos.py`
- `src/ai_api_unified/videos/ai_bedrock_videos.py`
- `src/ai_api_unified/videos/frame_helpers.py`

Update:

- `src/ai_api_unified/ai_base.py`
- `src/ai_api_unified/ai_factory.py`
- `src/ai_api_unified/ai_provider_registry.py`
- `src/ai_api_unified/util/env_settings.py`
- `src/ai_api_unified/__init__.py`
- `README.md`
- `env_template`
- middleware config and observability files

Root exports should include at minimum:

- `AIBaseVideos`
- `AIBaseVideoProperties`
- `AIVideoGenerationJob`
- `AIVideoGenerationResult`

## Dependency Review

The existing provider extras are directionally correct, but implementation
should verify minimum versions for video-capable SDK surfaces.

Specific review points:

- `openai`
  - confirm the minimum version that exposes the videos resource used by the
    implementation
- `google-genai`
  - confirm the current pinned minimum already supports
    `client.models.generate_videos(...)`
- `boto3`
  - confirm the current pinned minimum covers Nova Reel async invocation and
    job retrieval
- frame extraction helper
  - add one new optional extra for video-frame decoding, preferably isolated
    from provider extras
  - recommended direction: a small helper stack built around `imageio` and
    `imageio-ffmpeg`, lazy-loaded behind one helper boundary

If any floor is too old, update only the relevant optional extra.

## Testing Strategy

### Unit Tests

Add tests for:

- provider registry coverage for `videos`
- `AIFactory.get_ai_video_client(...)`
- provider default model pass-through behavior when `VIDEO_MODEL_NAME` is unset
- status normalization for OpenAI, Google, and Bedrock
- timeout behavior in `wait_for_video_generation(...)`
- Bedrock S3 output URI validation
- artifact download normalization into `AIVideoArtifact`
- frame extraction helper by time offset
- frame extraction helper by frame index
- image-buffer file persistence helper
- observability metadata emission for `videos`

### Mocked Provider Tests

Add provider-specific mocked tests similar to current image tests:

- `tests/test_openai_videos.py`
- `tests/test_google_gemini_videos.py`
- `tests/test_nova_reel_videos.py`
- `tests/test_observability_videos.py`

### Live Tests

Add optional `nonmock` coverage guarded by provider credentials:

- OpenAI Sora live test
- Google Veo live test
- Bedrock Nova Reel live test

These should remain opt-in and resilient to quota or account-surface limitations.

### Existing Test Adjustments

Current observability config tests treat `videos` as an invalid capability
token. Those tests must be updated once video becomes a supported capability.

## README and Environment Documentation Changes

Update the README capability table to add:

- `Videos` | `AIBaseVideos` | `openai`, `google-gemini`, `nova-reel` aliases | `openai`, `google_gemini`, `bedrock`

Update the quickstart section with a blocking video example.

Update `env_template` with:

- `VIDEO_ENGINE`
- `VIDEO_MODEL_NAME`
- `VIDEO_OUTPUT_DIR`
- `BEDROCK_VIDEO_OUTPUT_S3_URI`
- comments that document each provider's default profile so developers know what
  happens when they leave optional blanks unset

Recommended OSS defaults:

```dotenv
VIDEO_ENGINE=google-gemini
VIDEO_MODEL_NAME=veo-3.1-lite-generate-preview
VIDEO_OUTPUT_DIR=./generated_videos
```

Bedrock-specific example:

```dotenv
VIDEO_ENGINE=nova-reel
VIDEO_MODEL_NAME=amazon.nova-reel-v1:1
BEDROCK_VIDEO_OUTPUT_S3_URI=s3://your-bucket/ai-api-unified/video-output/
```

## Migration and Delivery Plan

### Phase 1

- add shared video base models and factory support
- add provider registry entries
- implement Google Veo provider first

Rationale:

- Google already matches the repository's `google-genai` direction
- the API shape is relatively direct
- it exercises the new job/result model clearly

### Phase 2

- add Bedrock Nova Reel provider
- add S3 artifact normalization

### Phase 3

- add OpenAI Sora provider with explicit deprecation note

### Phase 4

- wire observability for `videos`
- update README and env docs
- add live tests where feasible

### Phase 5

- add frame extraction helper
- add image-buffer file persistence helper
- document the optional helper dependency boundary

## Key Design Decision

The central design decision is:

- keep the public interface synchronous and simple
- represent the underlying provider reality with a normalized job model
- return file-backed artifacts instead of raw video bytes

That gives the repository a clean extension of its current architecture without
pretending that video behaves like image generation.

## References

- OpenAI video generation with Sora:
  `https://developers.openai.com/api/docs/guides/video-generation`
- Google Gemini API video generation with Veo:
  `https://ai.google.dev/gemini-api/docs/video`
- Amazon Nova Reel video generation:
  `https://docs.aws.amazon.com/nova/latest/userguide/video-generation.html`
- Amazon Nova Reel code examples:
  `https://docs.aws.amazon.com/nova/latest/userguide/video-gen-code-examples2.html`
