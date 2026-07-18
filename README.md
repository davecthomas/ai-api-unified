# ai-api-unified 2.15.0

`ai-api-unified` is a unified Python library for AI completions, embeddings, image generation, video generation, and voice. Application code targets stable base interfaces and factory entry points while concrete providers are selected at runtime from environment configuration.

Author: Dave Thomas  
Install name: `ai-api-unified`  
Import path: `ai_api_unified`  
Python: `>=3.11,<3.14`  
License: MIT

The current package architecture is registry-backed and lazy-loaded:

- provider SDKs are optional extras, not base dependencies
- providers are resolved only when a factory selects them
- package `__init__` modules export stable interfaces only
- missing provider selectors are configuration errors, not implicit fallbacks

## Overview

Use this library when you want one consistent interface across multiple AI providers without binding application code to a single SDK. The library currently covers:

- text completions
- embeddings
- image generation
- video generation
- text-to-speech and selected speech-to-text flows

The public entry points are the stable base interfaces and factories:

- `AIFactory.get_ai_completions_client()`
- `AIFactory.get_ai_embedding_client()`
- `AIFactory.get_ai_images_client()`
- `AIFactory.get_ai_video_client()`
- `AIVoiceFactory.create()`

## Capabilities

| Capability  | Stable interface    | Engines                                                                                                                       | Required extra(s)                                    |
| ----------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| Completions | `AIBaseCompletions` | `openai`, `openai-responses`, `claude`, `google-gemini`, Bedrock-routed aliases such as `nova`, `anthropic`, `llama`, `mistral`, `cohere`, `ai21`, `rerank` | `openai`, `anthropic`, `google_gemini`, `bedrock`    |
| Embeddings  | `AIBaseEmbeddings`  | `openai`, `titan`, `google-gemini`                                                                                            | `openai`, `bedrock`, `google_gemini`                 |
| Images      | `AIBaseImages`      | `openai`, `google-gemini`, `nova-canvas` and Bedrock image aliases                                                            | `openai`, `google_gemini`, `bedrock`                 |
| Videos      | `AIBaseVideos`      | `openai`, `google-gemini`, `nova-reel` and Bedrock video aliases                                                              | `openai`, `google_gemini`, `bedrock`                 |
| Voice TTS   | `AIVoiceBase`       | `openai`, `google`, `azure`, `elevenlabs`                                                                                     | `openai`, `google_gemini`, `azure_tts`, `elevenlabs` |
| Voice STT   | `AIVoiceBase`       | provider-specific support such as Google and ElevenLabs                                                                       | `google_gemini`, `elevenlabs`                        |

Default model guidance in the checked-in OSS env files:

- Anthropic completions: `claude-opus-4-8`
- Google completions: `gemini-2.5-flash`
- Google embeddings: `gemini-embedding-001` (text-only) or `gemini-embedding-2` (multimodal)
- Google images: `imagen-4.0-generate-001`
- Google videos: `veo-3.1-lite-generate-preview`
- Google voice: `gemini-2.5-pro-tts`

## Installation

### Python Requirements

This package requires Python `>=3.11,<3.14`.

### Install as a Dependency

Base package only:

```bash
poetry add ai-api-unified
```

Install with one or more provider extras:

```bash
poetry add 'ai-api-unified[google_gemini]'
poetry add 'ai-api-unified[openai]'
poetry add 'ai-api-unified[anthropic]'
poetry add 'ai-api-unified[bedrock,google_gemini]'
poetry add 'ai-api-unified[google_gemini,video_frames]'
poetry add 'ai-api-unified[openai,video_frames]'
```

### Install in a Local Clone

Base install:

```bash
poetry install
```

Common local development installs:

```bash
poetry install --with dev
poetry install --extras "google_gemini"
poetry install --extras "openai"
poetry install --extras "google_gemini" --extras "video_frames" --with dev
poetry install --extras "openai" --extras "video_frames" --with dev
poetry install --all-extras --with dev
```

### Optional Extras

| Extra                            | Installs                                                               |
| -------------------------------- | ---------------------------------------------------------------------- |
| `openai`                         | OpenAI completions, embeddings, images, and voice                      |
| `anthropic`                      | Anthropic Claude completions via the native Anthropic API (`claude` engine) |
| `google_gemini`                  | Google Gemini completions, embeddings, images, and Google voice        |
| `bedrock`                        | AWS Bedrock completions, Titan embeddings, Bedrock image providers, and Bedrock video providers |
| `video_frames`                   | Optional frame extraction helpers backed by ImageIO + Pillow           |
| `azure_tts`                      | Azure Cognitive Services TTS                                           |
| `elevenlabs`                     | ElevenLabs TTS and STT                                                 |
| `middleware-pii-redaction`       | Presidio + spaCy + `usaddress`; install the required spaCy model separately |
| `middleware-pii-redaction-small` | Compatibility alias for PII redaction deps; pair with separate `en_core_web_sm` install |
| `middleware-pii-redaction-large` | Compatibility alias for PII redaction deps; pair with separate `en_core_web_lg` install |
| `similarity_score`               | NumPy-based similarity helpers                                         |
| `dev`                            | Optional dev dependencies from `[project.optional-dependencies]`       |

### Environment File

Copy [`env_template`](env_template) to `.env` and fill in only the providers you use.

The OSS template now defaults to Google API-key auth:

```dotenv
COMPLETIONS_ENGINE=google-gemini
EMBEDDING_ENGINE=google-gemini
IMAGE_ENGINE=google-gemini
VIDEO_ENGINE=google-gemini
AI_VOICE_ENGINE=google

GOOGLE_GEMINI_API_KEY=...
GOOGLE_AUTH_METHOD=api_key

COMPLETIONS_MODEL_NAME=gemini-2.5-flash
EMBEDDING_MODEL_NAME=gemini-embedding-001
IMAGE_MODEL_NAME=imagen-4.0-generate-001
VIDEO_MODEL_NAME=veo-3.1-lite-generate-preview
DEFAULT_GEMINI_TTS_MODEL=gemini-2.5-pro-tts
```

Leave `EMBEDDING_DIMENSIONS` unset unless you deliberately want a provider-specific override. The library now preserves provider defaults instead of forcing a generic value.

### Smoke Test

```bash
python -c "import ai_api_unified; print(ai_api_unified.__version__)"
```

## Quickstart

The examples below assume the Google API-key-first OSS defaults shown above. The same APIs work with OpenAI, Bedrock, Azure, or ElevenLabs by changing env selectors and installing the matching extras.

### Completions

```python
from ai_api_unified import AIFactory, AIBaseCompletions

client: AIBaseCompletions = AIFactory.get_ai_completions_client()
response: str = client.send_prompt("Say hello in one short sentence.")
print(response)
```

### Streaming Completions

Every completions client exposes a `capabilities` descriptor with a
`supports_streaming` flag set per model. Models without streaming support raise
`AiProviderCapabilityUnsupportedError` from `send_prompt_streaming`. OpenAI,
Anthropic, Google Gemini, and Bedrock chat models all stream:

```python
from ai_api_unified import AIFactory

client = AIFactory.get_ai_completions_client()

if client.capabilities.supports_streaming:
    for chunk in client.send_prompt_streaming("Tell me a short story."):
        print(chunk, end="", flush=True)
```

Streaming is unavailable while the PII redaction middleware is enabled
(`AiProviderConfigurationError`): redaction cannot be guaranteed across chunk
boundaries, so use `send_prompt` in PII-redacting deployments.

### Token Counting

Providers whose capabilities include `supports_token_counting` can return a
provider-counted input token total without running inference. Bedrock supports
this via its `CountTokens` operation and the native Anthropic API via its
`count_tokens` endpoint; other providers raise
`AiProviderCapabilityUnsupportedError`.

```python
client = AIFactory.get_ai_completions_client(completions_engine="claude")

if client.capabilities.supports_token_counting:
    print(client.count_tokens("How many tokens is this prompt?"))
```

### OpenAI Responses engine

OpenAI exposes two completions engines. The default `openai` engine uses Chat
Completions; the `openai-responses` engine (`COMPLETIONS_ENGINE=openai-responses`)
uses the Responses API, OpenAI's successor to Chat Completions. Both implement
`send_prompt`, `strict_schema_prompt`, and
`send_prompt_streaming`. The Responses engine is text-only for now; use the
`openai` engine for image inputs.

### Anthropic Claude engines

Claude models are reachable through two completions engines:

- `claude` — the native Anthropic API (api.anthropic.com) via the official
  `anthropic` SDK. Requires the `anthropic` extra and `ANTHROPIC_API_KEY`.
- `anthropic` — Claude on Amazon Bedrock via the Converse API. Requires the
  `bedrock` extra and AWS credentials.

The two engines expose the same core caller-facing API (`send_prompt`,
`strict_schema_prompt`, `send_prompt_streaming`, `count_tokens`) plus the
capability-gated conversation/structured-output surface (see the support
matrix above); switching between them is a configuration change. The `claude`
engine additionally implements the async variants and batch completions. Use
`claude` for the current model lineup on Anthropic's own endpoint; use
`anthropic` when your Claude access is provisioned through AWS.

```dotenv
COMPLETIONS_ENGINE=claude
COMPLETIONS_MODEL_NAME=claude-opus-4-8
ANTHROPIC_API_KEY=...
```

Models catalogued for the `claude` engine (alias model IDs):
`claude-fable-5`, `claude-opus-4-8` (default), `claude-opus-4-7`,
`claude-opus-4-6`, `claude-sonnet-4-6`, and `claude-haiku-4-5`. Capabilities
per model include the context window (1M tokens except `claude-haiku-4-5` at
200K), streaming, provider-side token counting, image inputs, and registry
pricing. Structured output uses the Messages API JSON-schema response format,
so `strict_schema_prompt` works on every catalogued model. On
`claude-fable-5`, whose thinking is always on and counts against `max_tokens`,
pass a `max_response_tokens` well above the 2048 default so the budget covers
thinking plus the JSON body. Image attachments are capped at Anthropic's 5MB
per-image limit.

### Send prompt options

`send_prompt` accepts three optional parameters. Omitting them leaves prior
behavior unchanged.

```python
from ai_api_unified import AIFactory

client = AIFactory.get_ai_completions_client()
text = client.send_prompt(
    "Generate a workflow document from this instruction: ...",
    system_prompt="You write workflow documents.",
    max_response_tokens=9000,
    request_timeout_seconds=30.0,
)
```

All three parameters map to native provider fields on the `claude`,
`openai`, `openai-responses`, and `google-gemini` engines. Bedrock-routed
engines map `system_prompt` and `max_response_tokens` but raise
`AiProviderCapabilityUnsupportedError` for `request_timeout_seconds` (boto3
has no per-call timeout). See the support matrix below.

#### Feature support by engine

| Feature | `claude` | `openai` | `openai-responses` | `google-gemini` | Bedrock-routed |
|---|---|---|---|---|---|
| `send_prompt` extended params | yes | yes | yes | yes | partial (no per-call timeout) |
| `send_structured_output` | yes | yes | yes | yes | per-model (AWS structured-outputs list: Claude 4.5+) |
| `send_conversation` tool loop | yes | yes | yes | yes | Nova + Claude families |
| Async variants (`asend_*`) | yes | yes | yes | yes | no (boto3 has no official async client) |
| `retry_policy` / `AiProviderRequestError` | yes | yes | yes | yes | yes (engine loop; SDK retries via `AWS_MAX_ATTEMPTS`) |

Unsupported combinations raise the typed `AiProviderCapabilityUnsupportedError`
and each engine's `client.capabilities` flags report support at runtime.

### Structured output extraction (send_structured_output)

`send_structured_output` is the single-shot extraction call: prose in, a parsed
JSON object out. It accepts either an `AIStructuredPrompt` subclass
(`response_model`) or a raw JSON Schema (`response_schema`) — for example a
hand-written schema with `anyOf` variants per node type. Engines declare
support via `capabilities.supports_structured_output`. Provider mappings:
`claude` uses the Messages API JSON-schema response format (and streams and
accumulates internally above its non-streaming budget); `openai` and
`openai-responses` use the `json_schema` response format in schema-guided
mode; `google-gemini` uses `response_json_schema`; Bedrock uses Converse
`outputConfig` on the models AWS supports (Claude 4.5+).

```python
result = client.send_structured_output(
    "Compile this prose into a workflow graph: ...",
    response_schema=GRAPH_SCHEMA,          # raw JSON Schema owned by the caller
    system_prompt="You are a workflow compiler.",
    max_response_tokens=32_000,            # up to the model context limit
    request_timeout_seconds=120.0,
)
if result.finish_reason == "complete":
    graph = result.data                     # parsed dict
elif result.finish_reason == "length":
    ...                                     # truncated: retry with a larger budget
elif result.finish_reason == "refusal":
    ...                                     # model declined: abort
print(result.usage.input_tokens, result.usage.output_tokens)
```

Every result carries a normalized `finish_reason`
(`complete | length | tool_use | refusal`, one enum across engines) and token
usage, so truncation and refusal are distinguishable in code. `data` is `None`
on `length` and `refusal`. Multi-turn correction is supported through
`messages`: replay a prior model output plus feedback as extra
`{role, content}` turns, and `prompt` may then be omitted.

```python
result = client.send_structured_output(
    "The kind 'bogus' is invalid; use 'task' or 'gate'.",
    response_schema=GRAPH_SCHEMA,
    messages=[
        {"role": "user", "content": original_prompt},
        {"role": "assistant", "content": previous_bad_output},
    ],
)
```

Budgets above the engine's non-streaming request limit are handled inside the
engine (the `claude` engine streams and accumulates); the caller sees one
blocking call either way. `provider_options` is a per-engine escape hatch
merged into the underlying request; the `claude` engine honors the reserved key
`retry_policy` and merges every other key verbatim into the Messages request.

### Tool-use conversations (send_conversation)

`send_conversation` sends one conversation turn and returns the model's turn;
the caller owns the tool loop. Engines declare support via
`capabilities.supports_tool_use`.

```python
from ai_api_unified import AITool

tools = [
    AITool(
        name="lookup_ticket",
        description="Fetch a ticket by id.",
        input_schema={
            "type": "object",
            "properties": {"ticket_id": {"type": "string"}},
            "required": ["ticket_id"],
        },
        strict=True,               # schema-exact tool inputs where supported
    )
]

messages = [{"role": "user", "content": "Summarize ticket VL-123."}]
for _ in range(MAX_ITERATIONS):
    turn = client.send_conversation(
        "You are a support agent.",
        messages,
        tools=tools,
        # tool_choice="lookup_ticket",   # force a named tool for this turn
        max_response_tokens=4096,
        request_timeout_seconds=60.0,
    )
    if turn.finish_reason != "tool_use":
        break
    client.extend_messages_with_turn(messages, turn)   # engine-shaped replay
    for tool_call in turn.tool_calls:
        output = execute_tool(tool_call.name, tool_call.input)   # caller-side (e.g. MCP)
        messages.append(
            client.build_tool_result_message(
                tool_call_id=tool_call.id, result=output, is_error=False
            )
        )
```

Each `AITurnResult` carries `text`, `tool_calls` (`id`, `name`, `input`), the
normalized `finish_reason`, `usage` for that turn, and `raw_content` —
engine-specific content replayable as the next assistant turn. Because the
assistant-turn wire shape differs per engine, `extend_messages_with_turn`
appends it for you and `build_tool_result_message` produces the engine's
tool-result shape, so the loop above runs unchanged on `claude`, `openai`,
`openai-responses`, `google-gemini`, and tool-capable Bedrock models
(Nova and Claude families). On Gemini, tool calls carry no provider ids, so
`AIToolCall.id` is the function name.

### Async variants

Engines whose SDK has an async client expose `a`-prefixed variants with the
same signatures: `asend_prompt`, `asend_structured_output`, and
`asend_conversation`. Support is declared via `capabilities.supports_async`:
`claude` (lazy `AsyncAnthropic`), `openai` and `openai-responses` (lazy
`AsyncOpenAI`), and `google-gemini` (`client.aio`) implement all three;
Bedrock does not (boto3 has no official async client). Gemini async calls
run a single attempt — the engine backoff loop is synchronous — so pair them
with caller-owned backoff on `AiProviderRequestError`. Sync methods are
unchanged.

```python
turn = await client.asend_conversation("system", messages, tools=tools)
```

### Retry policy and typed request errors

Engine retry behavior:

- `claude` / `openai` / `openai-responses` — the provider SDK retries
  transient failures (408, 409, 429, 5xx) twice by default with exponential
  backoff; `retry_policy="none"` constructs the client with `max_retries=0`.
- `google-gemini` — the engine's exponential-backoff loop (5 attempts);
  `retry_policy="none"` collapses it to a single attempt.
- Bedrock-routed engines — the engine's backoff schedule;
  `retry_policy="none"` collapses it to a single attempt. botocore's own
  client-level retries are configured separately via the standard
  `AWS_MAX_ATTEMPTS` environment variable.

Every engine accepts `retry_policy` at the constructor, through configuration
(`COMPLETIONS_RETRY_POLICY=none`), or per call
(`provider_options={"retry_policy": "none"}`; on Bedrock the per-call
override collapses that call's schedule). HTTP-level failures raise
`AiProviderRequestError`, whose `status_code` attribute lets caller backoff
classify 429/5xx/529 uniformly across engines; it is `None` when the failure
happened before a status was available (connection error or client-side
timeout).

```python
from ai_api_unified import AiProviderRequestError

try:
    turn = client.send_conversation("system", messages, tools=tools)
except AiProviderRequestError as error:
    if error.status_code in (429, 529):
        backoff_and_retry()
```

### Batch completions (Anthropic)

The `claude` engine can process many prompts as one asynchronous batch through
Anthropic's Message Batches API. Batches run in the background (most finish well
under an hour, up to a 24-hour ceiling) at roughly half the per-token cost of
individual calls — use them for bulk work that isn't latency-sensitive, such as
classification, extraction, or evaluation runs.

Batch support is capability-gated like streaming and token counting: check
`capabilities.supports_batch` before calling. Every catalogued `claude` model
supports it; other engines raise `AiProviderCapabilityUnsupportedError`.

Each request carries a `custom_id` you choose. Results come back keyed by that
`custom_id` (in arbitrary order), so you correlate results to requests yourself
rather than relying on position.

The blocking convenience path submits, polls, and returns results in one call:

```python
from ai_api_unified import AIBatchRequestItem, AIFactory

client = AIFactory.get_ai_completions_client(completions_engine="claude")

requests = [
    AIBatchRequestItem(custom_id="a", prompt="Summarize: the cat sat on the mat."),
    AIBatchRequestItem(custom_id="b", prompt="Translate to French: good morning."),
]

if client.capabilities.supports_batch:
    results = client.run_batch(requests, poll_interval_seconds=30)
    for item in results:
        print(item.custom_id, item.status, item.text)
```

For explicit control instead of the blocking wrapper, drive the lifecycle
directly — submit, poll status, then fetch results once the batch has ended:

```python
job = client.submit_batch(requests)
print(job.batch_id, job.status)          # AIBatchStatus.IN_PROGRESS

job = client.get_batch(job)               # refresh status + per-state counts
if job.is_terminal:                        # ENDED, FAILED, EXPIRED, or CANCELED
    for item in client.get_batch_results(job):
        if item.status.value == "succeeded":
            print(item.custom_id, item.text)
        else:
            print(item.custom_id, "failed:", item.error_message)

# client.cancel_batch(job)                 # request cancellation while in progress
```

Request prompts are PII-redacted (when the redaction middleware is enabled)
before submission, exactly as `send_prompt` redacts. Each `AIBatchRequestItem`
also accepts an optional `system_prompt` and `max_response_tokens`. Successful
result items carry the per-request `provider_prompt_tokens` /
`provider_completion_tokens` for cost attribution.

### Model pricing

Each model's rates are exposed through `capabilities.pricing` as a structured
`AIModelPricing`: separate per-1M input, output, and cached-input rates (a
`Decimal`), with an effective date, source, and confidence. Compute the cost of
a call from the token counts the provider reports:

```python
client = AIFactory.get_ai_completions_client(model_name="gpt-5.4")
pricing = client.capabilities.pricing
print(pricing.token_rates.input_per_1m, pricing.token_rates.output_per_1m)

usd = client.compute_completion_cost(input_tokens=1200, output_tokens=800)
```

Embeddings clients expose `compute_embedding_cost(input_tokens=...)`. The rate
tables live in a single pricing registry (`ai_api_unified.pricing`) keyed by
`(provider, model)`; see `docs/pricing_research.md` for the full table and
sources. The blended `price_per_1k_tokens` and `calculate_cost` are deprecated
shims over the split rates.

### Deprecated and retired models

The pricing registry also carries model lifecycle status. Requesting a
**retired** model (one the provider no longer serves) raises
`AiProviderConfigurationError` at construction and names a replacement,
surfacing the problem at setup. Requesting a **deprecated** model logs a
warning and emits a `DeprecationWarning` once per process, naming the sunset
date and replacement, then proceeds. Set `AI_STRICT_DEPRECATIONS=1` to escalate
deprecated models to the same construction-time error (useful in CI).

### Embeddings

```python
from ai_api_unified import AIFactory, AIBaseEmbeddings

client: AIBaseEmbeddings = AIFactory.get_ai_embedding_client()
result: dict[str, object] = client.generate_embeddings("hello world")
embedding = result.get("embedding")
print(len(embedding) if embedding else None)
```

### Multimodal Embeddings

Every embeddings client exposes a `capabilities` descriptor stating which input
types the configured model supports. Text-only models raise
`AiProviderCapabilityUnsupportedError` from `generate_embeddings_multimodal`.
Google `gemini-embedding-2` embeds interleaved text, images, video, audio, and
PDFs into one vector space (set `EMBEDDING_MODEL_NAME=gemini-embedding-2`):

```python
from ai_api_unified import (
    AIEmbeddingsMultimodalParams,
    AIFactory,
    SupportedDataType,
)

client = AIFactory.get_ai_embedding_client()  # EMBEDDING_MODEL_NAME=gemini-embedding-2

if SupportedDataType.IMAGE in client.capabilities.supported_data_types:
    with open("bicycle.png", "rb") as file_image:
        params = AIEmbeddingsMultimodalParams(
            text="a red bicycle",
            included_types=[SupportedDataType.IMAGE],
            included_data=[file_image.read()],
            included_mime_types=["image/png"],
        )
    result = client.generate_embeddings_multimodal(params)
    print(result["dimensions"])
```

Media attachments are capped at 20MB per attachment and per request. They also
require API-key auth (`GOOGLE_AUTH_METHOD=api_key`): the google-genai SDK sends
only text parts to the Vertex embedding endpoint, so in service-account mode
the client rejects media attachments up front.

### Image Generation

```python
from ai_api_unified import AIFactory, AIBaseImageProperties, AIBaseImages

client: AIBaseImages = AIFactory.get_ai_images_client()
images: list[bytes] = client.generate_images(
    "A watercolor skyline at sunrise.",
    AIBaseImageProperties(width=1024, height=1024, format="png", num_images=1),
)

with open("generated_image.png", "wb") as generated_file:
    generated_file.write(images[0])
```

### Video Generation

The blocking convenience path is:

```python
from pathlib import Path

from ai_api_unified import AIFactory, AIBaseVideoProperties, AIBaseVideos

client: AIBaseVideos = AIFactory.get_ai_video_client()
result = client.generate_video(
    "A cinematic tracking shot of a neon train crossing the desert at dusk.",
    AIBaseVideoProperties(output_dir=Path("./generated_videos")),
)

video_bytes: bytes = result.artifacts[0].read_bytes()
frames: list[bytes] = AIBaseVideos.extract_image_frames_from_video_buffer(
    video_bytes,
    time_offsets_seconds=[0.0, 1.0],
)
AIBaseVideos.save_image_buffers_as_files(
    frames,
    output_dir=Path("./generated_frames"),
)
```

If you want explicit job control instead of the blocking wrapper:

```python
from ai_api_unified import AIFactory, AIBaseVideos

client: AIBaseVideos = AIFactory.get_ai_video_client()
job = client.submit_video_generation(
    "A stop-motion paper city waking up at sunrise."
)
job = client.wait_for_video_generation(job)
result = client.download_video_result(job)
print(result.job.status, result.artifacts[0].file_path)
```

#### Kick Off Google Gemini Video Generation

Environment:

```dotenv
VIDEO_ENGINE=google-gemini
VIDEO_MODEL_NAME=veo-3.1-lite-generate-preview
GOOGLE_GEMINI_API_KEY=...
GOOGLE_AUTH_METHOD=api_key
```

Code:

```python
from pathlib import Path

from ai_api_unified import AIFactory, AIBaseVideoProperties, AIBaseVideos

client: AIBaseVideos = AIFactory.get_ai_video_client()
result = client.generate_video(
    "A cinematic dolly shot of a red vintage train moving through a desert at sunset.",
    AIBaseVideoProperties(
        output_dir=Path("./generated_videos/google"),
        timeout_seconds=1200,
        poll_interval_seconds=10,
    ),
)

print(result.artifacts[0].file_path)
```

#### Kick Off OpenAI Video Generation

Environment:

```dotenv
VIDEO_ENGINE=openai
VIDEO_MODEL_NAME=sora-2
OPENAI_API_KEY=...
```

Code:

```python
from pathlib import Path

from ai_api_unified import AIFactory, AIBaseVideoProperties, AIBaseVideos

client: AIBaseVideos = AIFactory.get_ai_video_client()
result = client.generate_video(
    "A wide cinematic shot of gentle ocean waves meeting a rocky coastline at golden hour.",
    AIBaseVideoProperties(
        output_dir=Path("./generated_videos/openai"),
        timeout_seconds=1200,
        poll_interval_seconds=10,
    ),
)

print(result.artifacts[0].file_path)
```

Frame extraction requires the optional `video_frames` extra.

### Voice

```python
from ai_api_unified import AIVoiceBase, AIVoiceFactory

voice: AIVoiceBase = AIVoiceFactory.create()
audio_bytes: bytes = voice.text_to_speech("Hello from ai-api-unified")

with open("out.wav", "wb") as output_file:
    output_file.write(audio_bytes)
```

## Configuration

### Required Engine Selectors

There is no implicit default provider. Set the selector for each capability you use.

| Environment variable | Valid values                                                                                                                  |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `COMPLETIONS_ENGINE` | `openai`, `openai-responses`, `claude`, `google-gemini`, Bedrock-routed aliases such as `nova`, `anthropic`, `llama`, `mistral`, `cohere`, `ai21`, `rerank` |
| `EMBEDDING_ENGINE`   | `openai`, `titan`, `google-gemini`                                                                                            |
| `IMAGE_ENGINE`       | `openai`, `google-gemini`, `nova-canvas`, `bedrock`, `nova`                                                                   |
| `VIDEO_ENGINE`       | `openai`, `google-gemini`, `bedrock`, `nova`, `nova-reel`                                                                     |
| `AI_VOICE_ENGINE`    | `openai`, `google`, `azure`, `elevenlabs`                                                                                     |

### Common Model Settings

| Environment variable        | Notes                                                                                       |
| --------------------------- | ------------------------------------------------------------------------------------------- |
| `COMPLETIONS_MODEL_NAME`    | Optional completions model override                                                         |
| `EMBEDDING_MODEL_NAME`      | Optional embeddings model override. `gemini-embedding-2` enables multimodal embeddings.     |
| `IMAGE_MODEL_NAME`          | Optional image model override                                                               |
| `VIDEO_MODEL_NAME`          | Optional video model override                                                               |
| `VIDEO_OUTPUT_DIR`          | Optional local output directory for materialized video artifacts                            |
| `VIDEO_POLL_INTERVAL_SECONDS` | Optional default poll interval for video job waits                                        |
| `VIDEO_TIMEOUT_SECONDS`     | Optional default timeout for blocking video generation                                      |
| `BEDROCK_VIDEO_OUTPUT_S3_URI` | Required for Nova Reel unless provided per request                                        |
| `DEFAULT_GEMINI_TTS_MODEL`  | Optional Google voice model override                                                        |
| `EMBEDDING_DIMENSIONS`      | Optional embeddings dimension override. Leave unset for provider defaults.                  |
| `AI_API_GEO_RESIDENCY`      | Optional geo hint. `US`, `USA`, or `United States` normalize to US routing where supported. |
| `AI_MIDDLEWARE_CONFIG_PATH` | Optional YAML config path for observability and PII middleware                              |

### Provider Authentication

#### OpenAI

Required:

- `OPENAI_API_KEY`

Common optional settings:

- `OPENAI_BASE_URL`
- `COMPLETIONS_MODEL_NAME`
- `EMBEDDING_MODEL_NAME`
- `IMAGE_MODEL_NAME`
- `VIDEO_MODEL_NAME`
- `EMBEDDING_DIMENSIONS`
- `AI_API_GEO_RESIDENCY`

#### Anthropic (native Claude API)

Required for the `claude` completions engine:

- `ANTHROPIC_API_KEY`

Common optional settings:

- `COMPLETIONS_MODEL_NAME` (defaults to `claude-opus-4-8`)

Claude via Amazon Bedrock (the `anthropic` engine) does not use
`ANTHROPIC_API_KEY`; it authenticates with AWS credentials like the other
Bedrock engines below.

For current model IDs and pricing, use the Anthropic documentation:

- [Claude model overview](https://platform.claude.com/docs/en/about-claude/models/overview)
- [Claude API pricing](https://platform.claude.com/docs/en/pricing)

#### AWS Bedrock and Titan

Required:

- `AWS_REGION`
- standard AWS credentials in environment or runtime IAM context

Common optional settings:

- `COMPLETIONS_MODEL_NAME`
- `EMBEDDING_MODEL_NAME`
- `IMAGE_MODEL_NAME`
- `VIDEO_MODEL_NAME`
- `BEDROCK_VIDEO_OUTPUT_S3_URI`
- `EMBEDDING_DIMENSIONS`
- `AI_API_GEO_RESIDENCY`

For current Bedrock model IDs, use the AWS documentation:

- [Supported foundation models in Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html)
- [Amazon Bedrock foundation model reference](https://docs.aws.amazon.com/bedrock/latest/userguide/foundation-models-reference.html)

#### Google-backed Providers

Required for the default OSS path:

- `GOOGLE_GEMINI_API_KEY`

Default auth mode:

- `GOOGLE_AUTH_METHOD=api_key`

Optional service-account mode:

- `GOOGLE_AUTH_METHOD=service_account`
- `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json`
- `GOOGLE_PROJECT_ID=<gcp-project-id>`
- `GOOGLE_LOCATION=us-central1`

This auth policy applies across Google completions, embeddings, images, videos, and voice.

##### Google auth modes per capability

The two `GOOGLE_AUTH_METHOD` values select different google-genai clients:
`api_key` creates the Gemini Developer client
(`genai.Client(api_key=...)`, generativelanguage endpoint) and
`service_account` creates the Vertex client
(`genai.Client(vertexai=True, project=..., location=...)`). Some capabilities
work in only one client:

| Google capability | `api_key` (Developer) | `service_account` (Vertex) | Notes |
| --- | :---: | :---: | --- |
| Completions (incl. streaming) | ✅ | ✅ | |
| Text embeddings | ✅ | ✅ | |
| Image generation (Imagen) | ✅ | ✅ | |
| Multimodal (media) embeddings | ✅ | ❌ | the SDK sends only text parts to the Vertex embedding endpoint, so the library raises `NotImplementedError` when media is attached in Vertex mode |
| Text-to-video with local download | ✅ | ❌ | downloaded via the Files API (`client.files.download`), which the SDK supports only in the Developer client; under Vertex set `download_outputs=False` to receive a remote `gs://` URI instead |
| `source_video` (video continuation) | ❌ | ✅ | the library raises `NotImplementedError` in `api_key` mode; this path requires Vertex |
| Voice (Gemini TTS / STT) | ⚠️ | ✅ | the library wires API keys into the Cloud TTS/STT clients, but Google may reject API keys that are not enabled for those APIs; `service_account` is the reliable mode |

Pick the mode for what you need: multimodal-media embeddings and
text-to-video-with-download require `api_key`, and `source_video` requires
`service_account`. Running both families means switching `GOOGLE_AUTH_METHOD`
between runs or constructing two clients. For a single call, `get_client(...,
use_api_key=True)` on the shared Google base forces the Developer client
without changing the environment.

#### Azure TTS

Required:

- `MICROSOFT_COGNITIVE_SERVICES_API_KEY`
- `MICROSOFT_COGNITIVE_SERVICES_REGION`

Optional:

- `MICROSOFT_COGNITIVE_SERVICES_ENDPOINT`
- `AI_VOICE_LANGUAGE`

#### ElevenLabs

Required:

- `ELEVEN_LABS_API_KEY`

## Lazy Loading and Imports

Prefer the stable interfaces and factories exported from the root package:

```python
from ai_api_unified import (
    AIFactory,
    AIVoiceFactory,
    AIBaseCompletions,
    AIBaseEmbeddings,
    AIBaseImages,
    AIBaseVideos,
    AIVoiceBase,
    AIEmbeddingsCapabilitiesBase,
    AIEmbeddingsMultimodalParams,
    AIIncludedMediaParamsBase,
    AiProviderCapabilityUnsupportedError,
    SupportedDataType,
)
```

Factory entry points:

- `AIFactory.get_ai_completions_client()`
- `AIFactory.get_ai_embedding_client()`
- `AIFactory.get_ai_images_client()`
- `AIFactory.get_ai_video_client()`
- `AIVoiceFactory.create()`

Concrete providers are no longer re-exported from package `__init__.py` modules. If you need a concrete class directly, import it from its implementation module, for example:

```python
from ai_api_unified.completions.ai_google_gemini_completions import (
    GoogleGeminiCompletions,
)
```

Typical factory failure modes:

- unsupported engine selector: `ValueError`
- selected provider extra is not installed: `AiProviderDependencyUnavailableError`
- provider load/runtime failure: `AiProviderRuntimeError`
- input modality unsupported by the configured embedding model: `AiProviderCapabilityUnsupportedError`
- retired model requested (or deprecated model under `AI_STRICT_DEPRECATIONS`): `AiProviderConfigurationError`

## Middleware

Middleware is configured by YAML referenced through `AI_MIDDLEWARE_CONFIG_PATH`.

### Observability

Observability middleware is built into the base package and does not require a separate extra.

Features:

- metadata-only input, output, and error events
- request-scoped correlation via `set_observability_context(...)`
- coverage for completions, embeddings, images, videos, and text-to-speech

Minimal config:

```yaml
middleware:
  - name: 'observability'
    enabled: true
```

#### Request-scoped context and tags

`set_observability_context` stores request-scoped correlation values in a
`contextvars.ContextVar`, so they apply to every library call made in that
context (including across `await` boundaries) without threading parameters
through application code. It accepts:

- `caller_id` — stable caller identifier; emitted as `originating_caller_id`
  on lifecycle events and `caller_id` on cost events
- `session_id` / `workflow_id` — emitted in event metadata when set
- `tags` — arbitrary string-to-string mapping (for example `run_id`,
  `node_id`, or a workflow name); each tag is emitted as a `tag_<name>` field
  on every event, including cost-topic events

The function returns a token; pass it to `reset_observability_context` to
restore the prior context (for example in a request-handler `finally` block).
Blank keys and values are dropped.

```python
from ai_api_unified.middleware import (
    set_observability_context,
    reset_observability_context,
)

token = set_observability_context(
    caller_id="workflow-service",
    tags={"run_id": run_id, "node_id": node_id, "workflow": workflow_name},
)
try:
    turn = client.send_conversation(system_prompt, messages, tools=tools)
finally:
    reset_observability_context(token)
```

Token usage is also available on every result object — `AITurnResult.usage`
per conversation turn and `AIStructuredOutputResult.usage` per structured call
— so billing attribution does not require parsing logs.

See [`docs/observability_middleware_example.yaml`](docs/observability_middleware_example.yaml) and [`docs/observability_middleware_design.md`](docs/observability_middleware_design.md).

#### Cost tracking (financial-ops)

Set `emit_cost: true` on the observability settings to attach a USD cost to each
call. Cost is computed from the provider-reported token counts and the model's
registry pricing (`capabilities.pricing`). The result is emitted as a structured
event on a dedicated cost topic logger (`ai_api_unified.observability.cost`),
separate from the observability event stream so handlers can route it
independently. Each event carries the pricing provenance (effective date,
source, confidence) so the cost is auditable. Unpriced models are skipped. This
is observe-only — it never affects program flow.

Prompt-cache reads are captured per provider (OpenAI, OpenAI Responses,
Anthropic, Bedrock, and Google Gemini) and billed at the model's cached-input
rate: the event carries `cached_input_tokens`, and the cached subset of
`input_tokens` is priced at the cached rate while the remainder bills at the
full input rate. Providers that report cache reads separately from the input
count (Anthropic, Bedrock) are normalized so `input_tokens` includes the cached
subset, keeping the cost split consistent across providers.

```yaml
middleware:
  - name: 'observability'
    enabled: true
    settings:
      emit_cost: true
      # emit_cost_topic: 'my.cost.logger'   # optional logger-name override
```

Cost enrichment fires whenever `emit_cost` is on, even when output events are
disabled by `direction`. See [`docs/finops_middleware_design.md`](docs/finops_middleware_design.md).

### PII Redaction

PII redaction is optional and requires one of:

- `middleware-pii-redaction`
- `middleware-pii-redaction-small`
- `middleware-pii-redaction-large`

Minimal config:

```yaml
middleware:
  - name: 'pii_redaction'
    enabled: true
    settings:
      direction: 'input_only'
```

Notes:

- `strict_mode: true` enables fail-closed behavior
- `balanced`, `high_accuracy`, and `low_memory` detection profiles are supported
- install a matching spaCy model separately, for example `poetry run python -m spacy download en_core_web_sm` for `balanced` or `poetry run python -m spacy download en_core_web_lg` for `high_accuracy`
- `middleware-pii-redaction-small` and `middleware-pii-redaction-large` are compatibility aliases for the same Python dependency set; the spaCy model is still installed as a separate build/runtime asset
- for no-egress images and Lambda-style deployments, install the spaCy model wheel into the build artifact instead of relying on runtime downloads
- recognizer customization is configured in YAML, not hard-coded in provider implementations

See [`docs/pii_redaction_design.md`](docs/pii_redaction_design.md) for the fuller contract and deployment tradeoffs.

## Structured Responses

Use `AIStructuredPrompt` together with `strict_schema_prompt(...)` when you want schema-validated structured output.

```python
from copy import deepcopy
from typing import Any

from ai_api_unified import (
    AIFactory,
    AIStructuredPrompt,
    StructuredResponseTokenLimitError,
)


class ContactExtraction(AIStructuredPrompt):
    name: str | None = None
    city: str | None = None

    @staticmethod
    def get_prompt() -> str:
        return "Extract the person's name and city from: Alice lives in Paris."

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]:
        schema = deepcopy(super().model_json_schema())
        schema["properties"] = {
            "name": {"type": "string"},
            "city": {"type": "string"},
        }
        schema["required"] = ["name", "city"]
        return schema


client = AIFactory.get_ai_completions_client()

try:
    result = client.strict_schema_prompt(
        prompt=ContactExtraction.get_prompt(),
        response_model=ContactExtraction,
        max_response_tokens=2048,
    )
    print(result.name, result.city)
except StructuredResponseTokenLimitError as exc:
    print(exc)
```

Key behavior:

- structured prompts are provider-agnostic at the call site
- the library validates the response against your output schema
- undersized or truncated structured responses raise `StructuredResponseTokenLimitError`

`AIStructuredPrompt.send_structured_prompt(...)` is also available when you want the prompt to live on the model instance itself.

## Testing

Regular test run:

```bash
poetry run pytest -m "not nonmock"
```

Live provider tests:

```bash
poetry run pytest -m nonmock -s -vv
```

Notes for live tests:

- install the matching provider extras first
- configure credentials in `.env`
- some tests may skip when a provider account lacks quota, a paid image tier, or an enabled cloud service

## Release and PyPI Publishing

The OSS repository publishes to public PyPI only.

Before publishing:

1. Bump the version in `pyproject.toml`.
2. Bump the version in `src/ai_api_unified/__version__.py`.
3. Bump the version in the `README.md` title heading (line 1).
4. Run `poetry run pytest tests/test_version_sync.py` to confirm all three agree.
5. Ensure the working tree is clean.
6. Ensure your PyPI token is configured for Poetry.

Publish with the checked-in script:

```bash
./publish.sh
```

The script:

- checks for uncommitted changes
- confirms the version
- removes old build artifacts
- builds the wheel and sdist locally
- fails before upload if built metadata contains direct URL requirements that PyPI rejects
- runs `poetry publish` only after metadata validation passes

After publishing, tag and push the release:

```bash
git tag v<version>
git push origin v<version>
```

## Troubleshooting

- `COMPLETIONS_ENGINE must be configured explicitly` or similar: set the required engine selector for that capability.
- `AiProviderDependencyUnavailableError`: install the extra for the selected provider.
- Google auth errors: the OSS default is `GOOGLE_AUTH_METHOD=api_key`. If you switch to `service_account`, make sure `GOOGLE_APPLICATION_CREDENTIALS` points to a valid local JSON credential file and set `GOOGLE_PROJECT_ID` and `GOOGLE_LOCATION` when required.
- Unexpected embeddings dimensions: leave `EMBEDDING_DIMENSIONS` unset unless you intentionally want a non-default size.
- Google image generation or TTS failures in live tests can reflect account/service state rather than library bugs, for example a disabled cloud API or missing paid-plan access.
- `AI_API_GEO_RESIDENCY=US` is a best-effort routing hint. Only providers that expose regional routing controls can honor it directly.

## License

This project is released under the MIT License.
