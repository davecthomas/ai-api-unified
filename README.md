# ai-api-unified 3.23.0

[![CI](https://github.com/davecthomas/ai-api-unified/actions/workflows/ci.yml/badge.svg)](https://github.com/davecthomas/ai-api-unified/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ai-api-unified.svg)](https://pypi.org/project/ai-api-unified/)
[![Python](https://img.shields.io/pypi/pyversions/ai-api-unified.svg)](https://pypi.org/project/ai-api-unified/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`ai-api-unified` is a unified Python library for AI completions, embeddings, image generation, video generation, and voice. Application code targets stable base interfaces and factory entry points while environment configuration selects the concrete providers at runtime.

Because every provider sits behind the same interface, the library can also
keep your application running when a provider cannot: configure a
[fallback chain](#model-fallback) and a request that fails because a model is
overloaded, rate limited, or out of quota is retried on the next model you
listed, on the same provider or a different one, with no change to the
calling code.

**Production status.** This library is in production use, serving live traffic
across several systems. It is published on PyPI on a regular release cadence,
follows [semantic versioning](https://semver.org/), and every release is gated
on the full mocked regression suite. Breaking changes land only in a major
version, and each release is recorded in [CHANGELOG.md](CHANGELOG.md).

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

## Supported models

More than 100 models from 10 model makers (OpenAI, Anthropic, Google,
Amazon, DeepSeek, Alibaba Qwen, Z.ai, Voyage AI, ElevenLabs, and Microsoft)
behind one interface, plus any server that speaks the OpenAI API. Switching
models is a configuration change: set the capability's engine and model in
`models.json` (for example `completions.engine` and `completions.model`).

<!-- supported-models:start -->
| Capability | Provider | Engine (`<section>.engine`) | Models |
| --- | --- | --- | --- |
| Completions | OpenAI | `openai`, `openai-responses` | `gpt-6-astra`, `gpt-6-sol`, `gpt-6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna` (default), `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.2`, `gpt-5.1-codex-max`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano` ⚠ until 2026-10-23, `o4-mini` ⚠ until 2026-10-23, `gpt-4o`, `gpt-4o-mini` |
| Completions | Anthropic | `claude` | `claude-fable-5-1`, `claude-opus-5-5`, `claude-fable-5`, `claude-opus-5` (default), `claude-sonnet-5`, `claude-opus-4-8`, `claude-opus-4-7`, `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5` |
| Completions | Google | `google-gemini` | `gemini-3.8-flash`, `gemini-3.7-flash` (default), `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3.1-pro-preview`, `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite` |
| Completions | Amazon Bedrock: Amazon Nova | `bedrock`, `nova` | `us.amazon.nova-2-lite-v1:0`, `amazon.nova-micro-v1:0`, `amazon.nova-lite-v1:0` (default), `amazon.nova-pro-v1:0`, `amazon.nova-premier-v1:0` |
| Completions | Amazon Bedrock: Anthropic Claude | `bedrock`, `anthropic` | `us.anthropic.claude-fable-5-1`, `us.anthropic.claude-opus-5-5`, `us.anthropic.claude-opus-5`, `us.anthropic.claude-sonnet-5`, `us.anthropic.claude-3-5-haiku-20241022-v1:0` |
| Completions | Amazon Bedrock: DeepSeek | `bedrock` | `deepseek.v3.2`, `us.deepseek.r1-v1:0` |
| Completions | Amazon Bedrock: Qwen (Alibaba) | `bedrock` | `qwen.qwen3-next-80b-a3b`, `qwen.qwen3-235b-a22b-2507-v1:0`, `qwen.qwen3-coder-next`, `qwen.qwen3-32b-v1:0` |
| Completions | Amazon Bedrock: GLM (Z.ai) | `bedrock` | `zai.glm-5`, `zai.glm-4.7`, `zai.glm-4.7-flash` |
| Completions | Any OpenAI-compatible server (vLLM, Ollama, LiteLLM, ...) | `openai-compatible` | any model the server hosts (`completions.model`) |
| Embeddings | OpenAI | `openai` | `text-embedding-3-small` (default), `text-embedding-3-large`, `text-embedding-ada-002` |
| Embeddings | Google | `google-gemini` | `gemini-embedding-001` (default), `gemini-embedding-2` (multimodal) |
| Embeddings | Voyage AI | `voyage` | `voyage-4-large`, `voyage-4`, `voyage-4-lite`, `voyage-code-4`, `voyage-3.5`, `voyage-3.5-lite`, `voyage-3-lite`, `voyage-3` (default), `voyage-3-large`, `voyage-code-3`, `voyage-finance-2`, `voyage-law-2` |
| Embeddings | Amazon Bedrock: Titan | `titan` | `amazon.titan-embed-text-v2:0` (default), `amazon.titan-embed-text-v1` |
| Images | OpenAI | `openai` | `gpt-image-2.5-flare`, `gpt-image-2.5-sunburst`, `gpt-image-2` (default), `gpt-image-1.5` ⚠ until 2026-12-01, `gpt-image-1-mini` ⚠ until 2026-12-01, `gpt-image-1` ⚠ until 2026-10-23 |
| Images | Google | `google-gemini` | `gemini-3.1-flash-image` (default), `gemini-3.1-flash-lite-image`, `gemini-3-pro-image`, `gemini-2.5-flash-image` ⚠ until 2026-10-02 |
| Images | Amazon Bedrock: Nova Canvas | `nova-canvas`, `bedrock`, `nova` | `amazon.nova-canvas-v1:0` (default) |
| Video | Google | `google-gemini` | `veo-3.1-generate-preview`, `veo-3.1-fast-generate-preview`, `veo-3.1-lite-generate-preview` (default) |
| Video | Amazon Bedrock: Nova Reel | `nova-reel`, `bedrock`, `nova` | `amazon.nova-reel-v1:1` (default) |
| Video | OpenAI | `openai` | none: `sora-2` and `sora-2-pro` retired 2026-09-24 |
| Text to speech | OpenAI | `openai` | `tts-1-hd` (default), `tts-1`, `gpt-4o-mini-tts` |
| Text to speech | Google | `google` | `gemini-2.5-pro-tts` (default), `gemini-2.5-flash-tts`, `gemini-3.1-flash-tts-preview` |
| Text to speech | ElevenLabs | `elevenlabs` | `eleven_multilingual_v2` (default), `eleven_v3`, `eleven_flash_v2_5` |
| Text to speech | Microsoft Azure | `azure` | `neural`, `neural_hd`, `openai_neural`, `openai_hd` voice families |
| Speech to text | OpenAI | `openai` | `gpt-transcribe` |
| Speech to text | Google | `google` | Google Cloud Speech-to-Text |
| Speech to text | ElevenLabs | `elevenlabs` | `scribe_v1` |
<!-- supported-models:end -->

(default) marks the model an engine uses when no model is configured. ⚠
marks a deprecated model that still works until the date shown. Bedrock
models run on AWS under your AWS account; AWS serves Qwen3 235B only in
us-east-2 and us-west-2. The list matches each provider's live model list as
of 2026-09-25 (Bedrock: the AWS model cards), and each model carries pricing
in the [cost registry](#model-pricing).

## Overview

Use this library when you want one consistent interface across multiple AI providers without binding application code to a single SDK. The library currently covers:

- text completions
- embeddings
- image generation
- video generation
- text-to-speech and selected speech-to-text flows

Across those capabilities the library gives you:

- one interface per capability, so swapping providers is a configuration change
- automatic [model fallback](#model-fallback) for completions, so an outage,
  rate limit, or exhausted quota on one model is served by the next
- cost tracking, prompt caching, PII redaction, and observability middleware
  that work the same way on every engine

The public entry points are the stable base interfaces and factories:

- `AIFactory.get_ai_completions_client()`
- `AIFactory.get_ai_embedding_client()`
- `AIFactory.get_ai_images_client()`
- `AIFactory.get_ai_video_client()`
- `AIFactory.get_ai_voice_client()` (or `AIVoiceFactory.create()`)

## Capabilities

| Capability  | Stable interface    | Engines                                                                                                                       | Required extra(s)                                    |
| ----------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| Completions | `AIBaseCompletions` | `openai`, `openai-responses`, `openai-compatible`, `claude`, `google-gemini`, Bedrock-routed aliases such as `nova`, `anthropic`, `llama`, `mistral`, `cohere`, `ai21`, `rerank` | `openai`, `anthropic`, `google_gemini`, `bedrock`    |
| Embeddings  | `AIBaseEmbeddings`  | `openai`, `titan`, `google-gemini`, `voyage`                                                                                  | `openai`, `bedrock`, `google_gemini`, `voyage`       |
| Classification | `AIBaseClassifier` | `openai-decisions`, and `openai`, `openai-responses`, `openai-compatible`, `claude`, `google-gemini`, `bedrock`, `anthropic` through structured output | `openai`, `anthropic`, `google_gemini`, `bedrock` |
| Images      | `AIBaseImages`      | `openai`, `google-gemini`, `nova-canvas` and Bedrock image aliases                                                            | `openai`, `google_gemini`, `bedrock`                 |
| Videos      | `AIBaseVideos`      | `openai`, `google-gemini`, `nova-reel` and Bedrock video aliases                                                              | `openai`, `google_gemini`, `bedrock`                 |
| Voice TTS   | `AIVoiceBase`       | `openai`, `google`, `azure`, `elevenlabs`                                                                                     | `openai`, `google_gemini`, `azure_tts`, `elevenlabs` |
| Voice STT   | `AIVoiceBase`       | provider-specific support such as Google and ElevenLabs                                                                       | `google_gemini`, `elevenlabs`                        |

Default model guidance in the checked-in OSS env files:

- Anthropic completions: `claude-opus-5`
- Google completions: `gemini-3.7-flash`
- Google embeddings: `gemini-embedding-001` (text-only) or `gemini-embedding-2` (multimodal)
- Google images: `gemini-3.1-flash-image`
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
poetry add 'ai-api-unified[google_gemini,voice]'   # Google voice/TTS/STT
poetry add 'ai-api-unified[openai,voice]'          # OpenAI voice
```

Audio dependencies (pydub and, on Python 3.13+, audioop-lts) ship in the
`voice` extra as of 2.16.0, so text-only installs such as
`ai-api-unified[anthropic]` carry no audio packages and importing the library
never triggers pydub's warnings. The `azure_tts` and `elevenlabs` extras
include the audio dependencies; Google and OpenAI voice pair their provider
extra with `voice`. Using a voice feature without the audio dependencies
raises `AiProviderDependencyUnavailableError` naming the extra to install.

The `google_gemini` extra states no protobuf range of its own as of 2.26.1.
Releases through 2.26.0 capped it below 5.0, which made the extra uninstallable
in any application already on protobuf 5 or later. protobuf is still required
here, but only transitively: `google-api-core`, `google-cloud-speech`,
`google-cloud-texttospeech` and `proto-plus` all depend on it, and they enforce
their own ranges. This library never imports protobuf, so any range it declared
could only conflict with theirs. Installs are verified against protobuf 4.25.8,
5.29.5, 6.33.6 and 7.36.1.

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
| `voyage`                         | Voyage AI embeddings (the `voyage` embeddings engine; `VOYAGE_API_KEY`) |
| `video_frames`                   | Optional frame extraction helpers backed by ImageIO + Pillow           |
| `azure_tts`                      | Azure Cognitive Services TTS (includes the `voice` audio deps)         |
| `elevenlabs`                     | ElevenLabs TTS and STT (includes the `voice` audio deps)               |
| `voice`                          | Shared audio processing (pydub; audioop-lts on Python 3.13+) for voice/TTS/STT; pair with a provider extra, e.g. `[google_gemini,voice]` or `[openai,voice]` |
| `middleware-pii-redaction`       | Presidio + spaCy + `usaddress`; install the required spaCy model separately |
| `middleware-pii-redaction-small` | Compatibility alias for PII redaction deps; pair with separate `en_core_web_sm` install |
| `middleware-pii-redaction-large` | Compatibility alias for PII redaction deps; pair with separate `en_core_web_lg` install |
| `similarity_score`               | NumPy-based similarity helpers                                         |
| `dev`                            | Optional dev dependencies from `[project.optional-dependencies]`       |

### Environment File and models.json

Two files: `.env` holds secrets only; `models.json` holds everything else
(model selections, provider settings, middleware). Copy
[`env_template`](env_template) to `.env` and fill in the keys for the
providers you use, and copy [`models_sample.json`](models_sample.json) to
`models.json` and keep the sections you need. The sample defaults to Google
with API-key auth (`providers.google.auth_method`):

```dotenv
GOOGLE_GEMINI_API_KEY=...
```

```json
{
  "completions": {"engine": "google-gemini", "model": "gemini-3.7-flash", "thinking": "medium"},
  "embeddings":  {"engine": "google-gemini", "model": "gemini-embedding-001"},
  "images":      {"engine": "google-gemini", "model": "gemini-3.1-flash-image"},
  "videos":      {"engine": "google-gemini", "model": "veo-3.1-lite-generate-preview"},
  "voice":       {"engine": "google", "tts_model": "gemini-2.5-pro-tts"}
}
```

Leave `embeddings.dimensions` at `0` unless you deliberately want a
provider-specific size. The library preserves provider defaults. A `.env`
from before `models.json` still works: each selection key it carries
overrides the file and logs a deprecation warning; `scripts/migrate2json.py`
moves them (see Configuration).

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

### Completions with the full result

`send_prompt_with_result` sends the same prompt and returns an
`AIPromptResult`: the text, plus which provider and model answered, the
router tier and how it was chosen, the thinking level, the tokens used, and
what the call cost. `send_prompt` is unchanged and still returns a string.

```python
result = client.send_prompt_with_result("Say hello in one short sentence.")
result.text                                # the response
result.provider, result.provider_engine    # "anthropic", "claude"
result.model_name                          # "claude-opus-5"
result.route_tier, result.route_method     # "hard", "classifier" (None off a router)
result.thinking                            # requested and applied level
result.usage.input_tokens, result.usage.output_tokens
result.total_usd                           # Decimal, classifier and failed attempts included
result.cost                                # the full AIRequestCostReport for this call
result.request_id, result.config_id        # correlate with logs and cost events
```

Every field describes this call alone, even inside a caller's
`request_cost_scope()` or when several calls run at once on one client.
`asend_prompt_with_result` is the async twin.

Conversation turns and structured results carry the same fields:
`AITurnResult` and `AIStructuredOutputResult` from `send_conversation`,
`send_structured_output`, and their async twins have `provider`,
`provider_engine`, `model_name`, `route_tier`, `route_method`, `model_pin`,
`thinking`, `usage`, and `cost` (the call's `AIRequestCostReport`, so
`turn.cost.total_usd` is what the turn cost).

### Completions with automatic fallback

Add one list and the same client keeps serving when the primary model
cannot. The chain below runs Claude first, then OpenAI, then Gemini:

```json
"completions": {
  "engine": "claude",
  "model": "claude-opus-5",
  "fallbacks": ["openai:gpt-5.6-luna", "google-gemini:gemini-3.7-flash"]
}
```

```python
from ai_api_unified import AIFactory

client = AIFactory.get_ai_completions_client()   # an AiFallbackCompletions
response: str = client.send_prompt("Say hello in one short sentence.")
print(response, "served by", client.last_route.label)
```

Nothing else changes: `send_prompt`, `send_conversation`,
`send_structured_output`, streaming, and the async variants all work as
before. See [Model fallback](#model-fallback) for what triggers a failover,
what does not, and how to see which model answered.

### Async everywhere

Every call that reaches a provider has an async form. Where the SDK has an async client
(prompts, structured output, conversations, embeddings, classifiers), the
`a`-prefixed method uses it. Everywhere else the async twin runs the
blocking call on a worker thread in the caller's context, so request cost
scopes, observability tags, pins, and fallback behave exactly as in the
sync call:

| Sync | Async |
| --- | --- |
| `send_prompt_streaming` | `asend_prompt_streaming` (an async iterator) |
| `count_tokens`, `strict_schema_prompt`, `preview_route` | `acount_tokens`, `astrict_schema_prompt`, `apreview_route` |
| `submit_batch`, `get_batch`, `get_batch_results`, `cancel_batch`, `run_batch` | `asubmit_batch`, `aget_batch`, `aget_batch_results`, `acancel_batch`, `arun_batch` |
| `generate_embeddings_multimodal` | `agenerate_embeddings_multimodal` |
| `generate_images`, `generate_image_files` | `agenerate_images`, `agenerate_image_files` |
| `generate_video`, `submit_video_generation`, `get_video_generation_job`, `wait_for_video_generation`, `download_video_result` | `agenerate_video`, `asubmit_video_generation`, `aget_video_generation_job`, `await_for_video_generation`, `adownload_video_result` |
| `text_to_voice`, `stream_audio`, `speech_to_text`, `speech_to_text_from_file`, `text_to_voice_with_emotion_prompt` | the same names with an `a` prefix |

```python
async for chunk in client.asend_prompt_streaming("Tell me a short story."):
    print(chunk, end="", flush=True)
```

`asend_prompt_streaming` runs the whole stream on one worker thread and
hands chunks to the event loop as they arrive; breaking out of the loop
closes the stream after its current chunk.

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
this via its `CountTokens` operation for Nova and Claude (not for the DeepSeek,
Qwen, and GLM models it hosts), and the native Anthropic API via its
`count_tokens` endpoint; other providers raise
`AiProviderCapabilityUnsupportedError`.

```python
client = AIFactory.get_ai_completions_client(completions_engine="claude")

if client.capabilities.supports_token_counting:
    print(client.count_tokens("How many tokens is this prompt?"))
```

### OpenAI Responses engine

OpenAI exposes two completions engines. The default `openai` engine uses Chat
Completions; the `openai-responses` engine (`completions.engine: "openai-responses"`)
uses the Responses API, OpenAI's successor to Chat Completions. Both implement
`send_prompt`, `strict_schema_prompt`, and
`send_prompt_streaming`. The Responses engine is text-only for now; use the
`openai` engine for image inputs.

### OpenAI-compatible servers

The `openai-compatible` engine drives any server that speaks the OpenAI Chat
Completions protocol (vLLM, Ollama, LiteLLM, hosted vendors with an
OpenAI-compatible endpoint) through the `openai` extra. It supports the same
surface as the `openai` engine: `send_prompt`, streaming,
`send_conversation` with tools, `send_structured_output`,
`strict_schema_prompt`, and the async variants.

```json
"completions": {"engine": "openai-compatible", "model": "Qwen/Qwen3-32B"},
"providers": {
  "openai_compatible": {
    "base_url": "http://localhost:8000/v1",
    "structured_output": "json_schema",
    "context_window": 131072
  }
}
```

```dotenv
OPENAI_COMPATIBLE_API_KEY=...   # optional for local servers
```

`base_url` is required and must be https:// unless it targets a loopback
host; `structured_output` (`json_schema` or `json_object`) and
`context_window` (a client-side context guard) are optional. Requests
send `max_tokens` rather than OpenAI's `max_completion_tokens`, which most
compatible servers do not accept. Set `structured_output`
to `json_object` for servers that guarantee valid JSON but do not enforce a
schema; the schema is then placed in the system prompt. The generic engine
has no pricing, reports no organization identity, and labels observability
events `openai-compatible`.

Vendor engines subclass `AiOpenAICompatibleCompletions` and set class
attributes (API key and base-URL settings, default endpoint, model
catalog, context windows, image and reasoning models, structured-output
mode, and the pricing-registry label) instead of reading them from the
environment.

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

```json
"completions": {"engine": "claude", "model": "claude-opus-5"}
```

```dotenv
ANTHROPIC_API_KEY=...
```

Models cataloged for the `claude` engine (alias model IDs):
`claude-fable-5-1`, `claude-opus-5-5`, `claude-fable-5`, `claude-opus-5`
(default), `claude-sonnet-5`, `claude-opus-4-8`, `claude-opus-4-7`,
`claude-opus-4-6`, `claude-sonnet-4-6`, and `claude-haiku-4-5`. Capabilities per model include the context window (1M
tokens except `claude-haiku-4-5` at 200K), streaming, provider-side token
counting, image inputs, and registry pricing. Structured output uses the Messages API JSON-schema response format,
so `strict_schema_prompt` works on every cataloged model. On
`claude-fable-5`, `claude-fable-5-1`, and `claude-opus-5-5`, whose thinking
is always on and counts against `max_tokens`, pass a `max_response_tokens`
well above the 2048 default so the budget covers thinking plus the JSON body.
`claude-fable-5-1` and `claude-opus-5-5` reject a forced tool choice, so
`send_conversation(..., tool_choice=...)` raises `ValueError` on those models
before any request is sent; leave `tool_choice` unset and name the tool in
the prompt. Image attachments are capped at Anthropic's 5MB
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
`openai`, `openai-responses`, `openai-compatible`, and `google-gemini`
engines. Bedrock-routed
engines map `system_prompt` and `max_response_tokens` but raise
`AiProviderCapabilityUnsupportedError` for `request_timeout_seconds` (boto3
has no per-call timeout). See the support matrix below.

#### Feature support by engine

| Feature | `claude` | `openai` | `openai-responses` | `openai-compatible` | `google-gemini` | Bedrock-routed |
|---|---|---|---|---|---|---|
| `send_prompt` extended params | yes | yes | yes | yes | yes | partial (no per-call timeout) |
| `send_structured_output` | yes | yes | yes | yes (`json_schema` or `json_object` mode) | yes | per-model: Claude 4.5 and Opus 4.6, DeepSeek V3.2, Qwen3, GLM |
| `send_conversation` tool loop | yes | yes | yes | yes | yes | Nova, Claude, DeepSeek V3.2, Qwen3, GLM (not DeepSeek R1) |
| Async variants (`asend_*`) | yes | yes | yes | yes | yes | no (boto3 has no official async client) |
| `retry_policy` / `AiProviderRequestError` | yes | yes | yes | yes | yes | yes (engine loop; SDK retries via `AWS_MAX_ATTEMPTS`) |

Unsupported combinations raise the typed `AiProviderCapabilityUnsupportedError`
and each engine's `client.capabilities` flags report support at runtime.

### Thinking levels

How much a model thinks before answering is one provider-neutral setting,
`AIThinkingLevel`, with six values: `off`, `minimal`, `low`, `medium`,
`high`, `max`. Configuration and application code use the level; each
engine applies it through its provider's own control, and no code or config
names a provider field. The default is `medium`.

```python
from ai_api_unified import AIFactory, AIThinkingLevel

client = AIFactory.get_ai_completions_client(thinking="high")   # or "thinking": "high" in models.json
client.thinking_level = AIThinkingLevel.LOW                      # change the default for later calls

text = client.send_prompt("...", other_params=params_with(thinking=AIThinkingLevel.MAX))
turn = client.send_conversation(system, messages, provider_options={"thinking_level": "off"})
result = client.send_structured_output(prompt, response_model=Summary,
                                       provider_options={"thinking_level": "low"})

turn.thinking.requested, turn.thinking.applied, turn.thinking.exact   # AIThinkingResolution
client.last_thinking                                                   # for send_prompt and friends
```

Precedence: the call (`other_params.thinking` on the text methods; the
reserved `thinking_level` key in `provider_options` on `send_conversation`
and `send_structured_output`, stripped before the options reach the
provider), then the client (`thinking_level`, set by the `thinking=`
factory argument), then `completions.thinking` in `models.json`, then
`medium`. The level has no `.env` key: it is a model selection, and those
live in the file. The factory applies one level to the primary,
its fallbacks, and every router tier, so a request that fails over keeps its
level.

`client.capabilities.thinking` lists the standard levels the model honors
exactly (`None` when the model has no thinking control). A requested level
outside that list resolves to the nearest one the model supports, and a tie
goes to the lower level, since lower thinking is cheaper and the caller
asked for no more. The library records both levels: on `result.thinking` and
`client.last_thinking` (`requested`, `applied`, `exact`, and a `reason`
such as `unsupported_level`, `no_thinking_control`, `forced_tool_choice`,
`assistant_prefill`, or `max_tokens_too_small`), on every completions
observability event (`thinking_requested`, `thinking_applied`,
`thinking_exact`), and in an `ai_thinking_event` log line when they differ.

How each engine applies a level:

| Engine | Models | `off` | `minimal` | `low` / `medium` / `high` / `max` |
|---|---|---|---|---|
| `claude`, Bedrock Claude | Fable 5 / 5.1, Opus 5.5 (always on) | nearest: `low` | nearest: `low` | `output_config.effort` |
| `claude`, Bedrock Claude | Opus 5, Haiku 5.5, Sonnet 5, Opus 4.6 to 4.8, Sonnet 4.6 | `thinking.type: disabled` | nearest: `off` | `thinking.type: adaptive` + `output_config.effort` |
| `claude`, Bedrock Claude | Sonnet 5.5 | `thinking.type: between_tools` | nearest: `off` | `thinking.type: adaptive` + `output_config.effort` |
| `claude`, Bedrock Claude | Haiku 4.5, Sonnet 4.5, Opus 4.5, Opus 4.1, Sonnet 4, Opus 4, 3.7 Sonnet | omit `thinking` | nearest: `off` | `thinking.budget_tokens` 1024 / 4096 / 16384 / 32768, capped at `max_tokens` less 1024 kept for the answer |
| `claude`, Bedrock Claude | Claude 3 Opus and Haiku, 3.5 Sonnet and Haiku | no control | no control | no control |
| `openai`, `openai-responses` | GPT-5.1 and later, GPT-6 | `reasoning_effort: none` (not on GPT-6 Astra) | nearest | `low` / `medium` / `high` / `xhigh` |
| `openai`, `openai-responses` | GPT-5, GPT-5 mini, nano, pro, codex, and dated snapshots | nearest: `minimal` | `minimal` | `low` / `medium` / `high` / nearest: `high` |
| `openai`, `openai-responses` | o-series | nearest: `low` | nearest: `low` | `low` / `medium` / `high` / nearest: `high` |
| `openai` | GPT-4.1, GPT-4o, uncataloged | no control | no control | no control |
| `openai-compatible` | `thinking_control: reasoning_effort` | nearest: `low` | nearest: `low` | `reasoning_effort` `low` / `medium` / `high` / nearest: `high` |
| `openai-compatible` | `thinking_control: enable_thinking` (vLLM, SGLang) | `enable_thinking: false` | nearest: `off` | `enable_thinking: true`, recorded as `medium` |
| `openai-compatible` | `thinking_control` unset or `none` | no control | no control | no control |
| `google-gemini` | Gemini 3 Flash and later | nearest: `minimal` | `thinking_level: MINIMAL` | `LOW` / `MEDIUM` / `HIGH` / nearest: `HIGH` |
| `google-gemini` | Gemini 3 Pro | nearest: `low` | nearest: `low` | `LOW` / nearest / `HIGH` / nearest: `HIGH` |
| `google-gemini` | Gemini 2.5 Flash, Flash Lite | `thinking_budget: 0` | nearest: `off` | `thinking_budget` 1024 / 8192 / 16384 / 24576 |
| `google-gemini` | Gemini 2.5 Pro | nearest: `low` | nearest: `low` | `thinking_budget` 1024 / 8192 / 16384 / 32768 |
| Bedrock | Nova 2 Lite, Nova 2 Pro | `reasoningConfig.type: disabled` | nearest: `off` | `maxReasoningEffort` `low` / `medium` / `high` / nearest: `high`; `high` drops temperature, topP, and topK |
| Bedrock | Nova 1, DeepSeek, Qwen, GLM | no control | no control | no control |

On Bedrock the Claude and Nova 2 fields travel under
`additionalModelRequestFields`. AWS documents no thinking control for the
DeepSeek, Qwen, and GLM models it hosts, so those record no control.

An OpenAI-compatible server's thinking control cannot be discovered, so
`providers.openai_compatible.thinking_control` declares it:
`reasoning_effort` for servers that accept the OpenAI field,
`enable_thinking` for vLLM and SGLang chat templates (Qwen3 and similar,
sent as `extra_body.chat_template_kwargs`), or `none` (the default).

A provider's own thinking field in `provider_options` (`thinking` or
`output_config.effort` on Claude, `reasoning_effort` on OpenAI chat,
`reasoning` on Responses, `thinking_config` on Gemini, and the thinking
fields under `additionalModelRequestFields` on Bedrock) is still accepted
and decides that call: the layer adds no thinking field of its own, records
the call with `applied` empty and reason `provider_option`, and logs one
WARNING per client and field naming the neutral `thinking_level` option.
Use the neutral option instead; the record then shows the level applied.

The router's classifier, and any structured-output classifier that builds
its own client, classifies with thinking `off`, since a classification is
one short label: set `classifier.thinking` in `models.json`, or pass
`thinking=` to `AiStructuredOutputClassifier`, to change it. An always-on
model applies its nearest level, which is recorded. A classifier handed an
existing client keeps that client's level unless `thinking=` is passed.

A batch request takes an optional `thinking` level per item
(`AIBatchRequestItem.thinking`, default the client's level). Each item gets
the same mapping as a single call, and the submitted `AIBatchJob.thinking`
records each request's resolution by `custom_id`.

Three
request shapes turn thinking off on models that allow it, with the reason
recorded: a forced `tool_choice` (Claude rejects the pair), an assistant
prefill (Bedrock's `strict_schema_prompt` on pre-4.6 Claude), and a
`max_tokens` with no room for the 1024-token minimum budget after 1024
tokens are kept for the answer. Bedrock's `send_prompt` and streaming
defaults of 256 tokens rise to 4096 on budget-era Claude models when the
caller sets no cap, so the configured level is applied.

The tables follow each provider's documentation as of October 2026 and the
live test (`poetry run pytest -m nonmock tests/test_thinking_levels_nonmock.py -q`)
exercises the configured models. When a model rejects a value the table
lists, the provider returns its own 400; correct the table entry.

### Structured output extraction (send_structured_output)

`send_structured_output` is the single-shot extraction call: prose in, a parsed
JSON object out. It accepts either an `AIStructuredPrompt` subclass
(`response_model`) or a raw JSON Schema (`response_schema`) — for example a
hand-written schema with `anyOf` variants per node type. Engines declare
support via `capabilities.supports_structured_output`. Provider mappings:
`claude` uses the Messages API JSON-schema response format (and streams and
accumulates internally above its non-streaming budget); `openai` and
`openai-responses` use the `json_schema` response format in schema-guided
mode; `openai-compatible` uses `json_schema`, or `json_object` with the
schema in the system prompt; `google-gemini` uses `response_json_schema`;
Bedrock uses Converse `outputConfig` on the models AWS supports (Claude 4.5
and Opus 4.6, DeepSeek V3.2, Qwen3, and GLM).

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
`openai-responses`, `openai-compatible`, `google-gemini`, and tool-capable
Bedrock models (Nova, Claude, DeepSeek V3.2, Qwen3, and GLM). On Gemini, tool calls carry no provider ids, so
`AIToolCall.id` is the function name.

### Async variants

Engines whose SDK has an async client expose `a`-prefixed variants with the
same signatures: `asend_prompt`, `asend_structured_output`, and
`asend_conversation`. Support is declared via `capabilities.supports_async`:
`claude` (lazy `AsyncAnthropic`), `openai`, `openai-responses`, and
`openai-compatible` (lazy `AsyncOpenAI`), and `google-gemini` (`client.aio`)
implement all three;
Bedrock does not (boto3 has no official async client). Gemini async calls
run a single attempt — the engine backoff loop is synchronous — so pair them
with caller-owned backoff on `AiProviderRequestError`. Sync methods are
unchanged.

```python
turn = await client.asend_conversation("system", messages, tools=tools)
```

### Response caching

Off unless configured. One entry in the `middleware` list turns it on:

```json
{"name": "response_cache", "enabled": true,
 "settings": {"max_entries": 10000,
              "operations": {"completions": {"ttl_seconds": 3600},
                             "embeddings": {"ttl_seconds": 86400},
                             "conversation": {"enabled": false}}}}
```

An identical request then returns the stored answer without reaching the
router, its classifier, or the provider. This is separate from the
provider-side prompt caching below, which still applies on a miss.

- **What is cached:** completions (`send_prompt` and
  `send_prompt_with_result`), structured output, conversations,
  embeddings, and classifications. Every operation is on inside an enabled
  cache; `operations.<name>.enabled: false` turns one off. Defaults: a day
  for embeddings and classifications, an hour for generated text;
  `ttl_seconds` at the top sets one TTL for all. Never cached: streaming,
  calls with media attachments, batches, voice.
- **The key** is the client (engine, model, thinking level, `config_id`)
  plus a hash of everything that determines the answer: the prompt or
  messages (surrounding whitespace trimmed), system prompt, tools, schema,
  token budget, and provider options, temperature included. Not in it:
  the caller, session, and tenant tags, request ids, prompt-cache hints,
  and the purpose hint. A stored answer was a valid answer to that exact
  request, so it is served again whatever the temperature was.
- **Per call:** `provider_options={"cache": "off"}` on conversations and
  structured output, `other_params.cache` on prompts, or
  `with cache_mode("off"):` around any call. `"on"` (the default) reads
  and writes, `"off"` neither reads nor writes (a call that wants a fresh
  answer each time), `"bypass"` skips the lookup and refreshes the entry.
  The option never reaches the provider.
- **What a hit reports:** every result envelope (`AIPromptResult`,
  `AITurnResult`, `AIStructuredOutputResult`, `AIClassificationResult`)
  carries `cache_hit` and, on a hit, `usd_saved` with the original call's
  cost; `cost.total_usd` is 0. The hit also emits one `ai_api_request_cost`
  event with `cache_hit: true` and `usd_saved`, under the caller's tags,
  and becomes `client.last_request_cost`.
- **The store** is in memory and per process by default: an LRU dict with
  a TTL per entry, `max_entries` deep, shipped with the library. Several
  processes each keep their own, so two workers can both miss once; the
  TTL bounds the waste. To share one cache, hand the library any object
  with `get(key) -> bytes | None`, `set(key, value, ttl_seconds)`, and
  `delete(key)`:

  ```python
  AIFactory.set_cache_store(MyRedisStore())   # None restores the default
  ```

  The library never imports a client for an external store, and a store
  that fails is a miss, never a failed call. Fallback cooldowns use the
  same store, so a shared store also shares a benched model across
  processes.
- **Looking at the cache:** it lives in the process that owns it, so the
  view is in-process. `response_cache_stats()` returns whether it is on,
  the store, the live entry count, and hits, misses, and `usd_saved`
  since the process started, in total and per operation.
  `response_cache_entries()` lists the live entries, most recently used
  first: key, operation, engine, model, stored time, seconds left, what
  one hit saves, and size, never the prompt or the answer.
  `response_cache_clear()` (or `clear("embeddings")`) removes them.
  Hits log one INFO line (`CACHE HIT`, with `ai_cache_event`, `usd_saved`,
  `caller_id`); misses log at DEBUG. `scripts/try_response_cache.py`
  shows all of this live against your configured provider.

### Prompt caching

Providers bill repeated prompt prefixes at a discounted cached-input rate. Some
cache every long prefix on their own; others cache only what the request marks.
`AIPromptCacheHint` asks for caching in one provider-neutral way: it marks the
system prompt and tool definitions as a stable prefix. Pass it on
`other_params.prompt_cache` (`send_prompt`, `asend_prompt`,
`send_prompt_streaming`, `strict_schema_prompt`), as `prompt_cache=` on
`send_conversation`, `send_structured_output`, and their async variants, or on
`AIBatchRequestItem.prompt_cache` for Anthropic batches.

```python
from ai_api_unified import AIPromptCacheHint, AIPromptCacheRetention

hint = AIPromptCacheHint(
    retention=AIPromptCacheRetention.EXTENDED,  # DEFAULT (about 5 min) or EXTENDED
    key="tenant-42",                            # optional; OpenAI cache routing
)
turn = client.send_conversation(
    LONG_STABLE_SYSTEM_PROMPT, messages, tools=tools, prompt_cache=hint
)
```

| Engine | Without a hint | With a hint |
| ------ | -------------- | ----------- |
| `claude` | No caching | `cache_control` breakpoint on the system block; conversations also cache the growing history. EXTENDED = 1-hour TTL |
| Bedrock (Claude, Nova) | Implicit, best effort | `cachePoint` after the system block. EXTENDED = 1-hour TTL on Claude 4.5 and later; Nova stays at 5 minutes |
| `openai`, `openai-responses` | Implicit on prefixes of 1,024+ tokens | Adds `prompt_cache_key`; EXTENDED sends `prompt_cache_retention="24h"` on models that offer it |
| `google-gemini` | Implicit on Gemini 2.5+; durable (up to 24 hours, no storage fee) on Gemini 3.x from 2026-10-15 | Ignored, since there is nothing to add |
| `openai-compatible`, other Bedrock models, and `claude` or `openai` behind a `base_url` gateway | Provider dependent | Ignored, since a gateway may reject the fields |

`capabilities.implicit_prompt_caching` and `capabilities.supports_prompt_cache_hint`
report both columns per model. Caching changes cost and latency but never the
response, so an engine that cannot honor the hint ignores it instead of raising.
The hint is not free on every provider: Anthropic and Bedrock bill cache
writes at 1.25x base input (2x for the 1-hour TTL), so it pays off only when
requests sharing the prefix repeat within the TTL. A hit needs a byte-identical
prefix, so keep timestamps and request IDs out of the system prompt. Cache
reads and writes show up in `AITurnResult.usage` and in the cost-tracking
events described under [Cost tracking](#cost-tracking-financial-ops).
Batch requests run in any order over minutes to hours, so batch cache hits are
best effort; EXTENDED retention usually suits them better than DEFAULT.

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
(`completions.retry_policy: "none"` in `models.json`), or per call
(`provider_options={"retry_policy": "none"}`; on Bedrock the per-call
override collapses that call's schedule). HTTP-level failures raise
`AiProviderRequestError`, whose `status_code` attribute lets caller backoff
classify 429/5xx/529 uniformly across engines; it is `None` when the failure
happened before a status was available (connection error or client-side
timeout).

The error also carries `fallback_reason`, an `AiFallbackReason` that says
whether a different model might serve the request. Each engine sets it from
the provider's own error codes, since a status alone cannot separate a rate
limit from an exhausted quota (OpenAI returns 429 for both, and Anthropic
reports a low credit balance or a reached spend cap as a 400):

| Reason | Meaning | Examples |
| ------ | ------- | -------- |
| `UNAVAILABLE` | The provider or model cannot serve right now | 5xx, Anthropic 529 overloaded, Bedrock `ServiceUnavailableException` / `ModelNotReadyException`, a failed connection |
| `RATE_LIMITED` | A transient 429; the engine's own backoff ran first | OpenAI `rate_limit_exceeded`, Bedrock `ThrottlingException`, Gemini per-minute quota |
| `QUOTA_EXHAUSTED` | Billing, credit, quota, or a spend cap is used up; retrying the same model is pointless | OpenAI `insufficient_quota` / `billing_hard_limit_reached`, Anthropic "credit balance is too low" or a reached workspace spend cap ("API usage limits"), Bedrock `ServiceQuotaExceededException`, Gemini plan or daily quota |
| `AUTH_REJECTED` | The provider rejected the credentials or the account; retrying the same provider is pointless | 401/403, Anthropic `authentication_error` / `permission_error` (including a disabled organization), Gemini "API key not valid" (a 400), Bedrock `AccessDeniedException` / `UnrecognizedClientException` / `ExpiredTokenException` |
| `MODEL_UNAVAILABLE` | The model id is unknown, retired, or not offered in the region | 404, Bedrock `ResourceNotFoundException` or an invalid model identifier |
| `TIMEOUT` | The caller's own deadline passed before the provider answered | `request_timeout_seconds` expired, botocore `read_timeout` on Bedrock |
| `None` | Another model would not help | Validation errors, missing local credentials, a refusal |

`is_transient` is True for `UNAVAILABLE` and `RATE_LIMITED`. Engines that run
their own retry schedule (Bedrock, Gemini, and the OpenAI `strict_schema_prompt`
loop) stop retrying as soon as an error classifies as `QUOTA_EXHAUSTED`,
`AUTH_REJECTED`, or `MODEL_UNAVAILABLE`, instead of sleeping through the schedule. Gemini reports
a per-minute limit and a daily quota with the same 429 text, so the split
reads the quota id in the message (`PerMinute` versus `PerDay`).

```python
from ai_api_unified import AiFallbackReason, AiProviderRequestError

try:
    turn = client.send_conversation("system", messages, tools=tools)
except AiProviderRequestError as error:
    if error.fallback_reason in (
        AiFallbackReason.UNAVAILABLE,
        AiFallbackReason.QUOTA_EXHAUSTED,
    ):
        turn = backup_client.send_conversation("system", messages, tools=tools)
    elif error.is_transient:
        backoff_and_retry()
    else:
        raise
```

You do not have to write that loop yourself: a [fallback chain](#model-fallback)
does it inside the client.

Embeddings, images, videos, and voice raise the same error. A provider
failure on any of them (an OpenAI, Google, Bedrock, Voyage, or ElevenLabs
SDK error, or an HTTP error from httpx) becomes `AiProviderRequestError`
with the same `status_code` and `fallback_reason` rules, and the SDK's own
exception as its `__cause__`. Errors the library raises about the request
itself (a malformed input, a provider response with no vectors) keep their
own types. `AiProviderRequestError` is a `RuntimeError`, so an existing
`except RuntimeError` still catches it.

### Batch completions (Anthropic)

The `claude` engine can process many prompts as one asynchronous batch through
Anthropic's Message Batches API. Batches run in the background (most finish well
under an hour, up to a 24-hour ceiling) at roughly half the per-token cost of
individual calls — use them for bulk work that can wait.

Batch support is capability-gated like streaming and token counting: check
`capabilities.supports_batch` before calling. Every cataloged `claude` model
supports it; other engines raise `AiProviderCapabilityUnsupportedError`.

Each request carries a `custom_id` you choose. Results come back in arbitrary
order, keyed by that `custom_id`.

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

For explicit control, drive the lifecycle
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

# Cache-aware. Pass the NON-CACHED remainder as input_tokens: of 1200 prompt
# tokens with 400 served from cache, 800 bill at the input rate and 400 at the
# cached rate. Cache writes are counted separately by the provider, so they add
# on top rather than coming out of the prompt count.
usd = client.compute_completion_cost(
    input_tokens=800,          # 1200 prompt tokens minus the 400 cache reads
    output_tokens=800,
    cached_input_tokens=400,
    cache_write_5m_tokens=2000,
    cache_write_1h_tokens=1000,
)
```

Note the difference from the cost event, where `input_tokens` *includes* the
cached subset because that is what providers report. This API takes the split
already applied.

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
date and replacement, then proceeds. Set `"strict_deprecations": true` in
`models.json` to escalate deprecated models to the same construction-time
error (useful in CI).

A retired model keeps whatever pricing it carried while it was active, so cost
enrichment can still price usage recorded before the withdrawal date. As of
2.22.0, `claude-opus-4-1` is retired (withdrawn 2026-08-05); use
`claude-opus-5`. As of 2.28.0, OpenAI `sora-2` / `sora-2-pro` (the Videos
API shut down 2026-09-24 with no replacement), `dall-e-2` / `dall-e-3`,
Google Imagen 4, and Veo 2.0 / 3.0 are retired. `o4-mini`, `gpt-4.1-nano`,
and `gpt-image-1` are deprecated (shutdown 2026-10-23), as are
`gpt-image-1-mini` and `gpt-image-1.5` (2026-12-01) and
`gemini-2.5-flash-image` (2026-10-02).

### Embeddings

Providers are interchangeable: the code below is identical for every
embeddings engine — only the engine name (and the vector dimensionality,
reported by `capabilities.default_dimensions`) differs.

```python
from ai_api_unified import AIFactory, AIBaseEmbeddings

client: AIBaseEmbeddings = AIFactory.get_ai_embedding_client()  # engine from env
result: dict[str, object] = client.generate_embeddings("hello world")
embedding = result.get("embedding")
print(len(embedding) if embedding else None)

# Same call, explicit engines — identical signatures and return shapes:
openai_client = AIFactory.get_ai_embedding_client(embedding_engine="openai")
voyage_client = AIFactory.get_ai_embedding_client(embedding_engine="voyage")
for client in (openai_client, voyage_client):
    result = client.generate_embeddings("hello world", input_type="document")
    print(client.model_name, result["dimensions"])
```

The `voyage` engine (extra: `voyage`, auth: `VOYAGE_API_KEY`) serves Voyage
AI's embeddings specialist models — `voyage-4-large`, `voyage-4`,
`voyage-4-lite`, `voyage-code-4`, `voyage-3.5`, `voyage-3.5-lite`, `voyage-3`
(default), `voyage-3-lite`, `voyage-3-large`, and the domain variants
`voyage-code-3`, `voyage-finance-2`, `voyage-law-2` — with registry pricing so cost events
work like completions. Anthropic recommends Voyage for embeddings since
Anthropic serves none.

- `input_type` ("query" | "document") is a provider-neutral retrieval hint:
  the `voyage` engine forwards it to the API; other engines ignore it.
- Batching: `generate_embeddings_batch(texts)` accepts any list length; the
  `voyage` engine chunks internally at the provider's 128-text request cap.
- Async: `agenerate_embeddings` / `agenerate_embeddings_batch` on engines
  whose capabilities declare `supports_async` (currently `voyage`); others
  raise `AiProviderCapabilityUnsupportedError`.
- The `voyage` engine honors `retry_policy="none"` (constructor or
  `completions.retry_policy`) and wraps provider failures in
  `AiProviderRequestError` carrying `status_code`, matching the completions
  clients.


### Classification

```python
from ai_api_unified import AIFactory, AIClassificationChoice

client = AIFactory.get_ai_classifier_client(classifier_engine="claude")
result = client.classify_choice(
    "Summarize this ticket in one line.",
    [AIClassificationChoice(value="easy"), AIClassificationChoice(value="hard")],
)
print(result.choice)  # "easy"
```

See the Classification section for the question types and the engines that
support them.
### Multimodal Embeddings

Every embeddings client exposes a `capabilities` descriptor stating which input
types the configured model supports. Text-only models raise
`AiProviderCapabilityUnsupportedError` from `generate_embeddings_multimodal`.
Google `gemini-embedding-2` embeds interleaved text, images, video, audio, and
PDFs into one vector space (set `embeddings.model` to `gemini-embedding-2`):

```python
from ai_api_unified import (
    AIEmbeddingsMultimodalParams,
    AIFactory,
    SupportedDataType,
)

client = AIFactory.get_ai_embedding_client()  # embeddings.model: gemini-embedding-2

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
require API-key auth (`providers.google.auth_method` set to `api_key`): the google-genai SDK sends
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

For explicit job control:

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

```json
"videos": {"engine": "google-gemini", "model": "veo-3.1-lite-generate-preview"}
```

```dotenv
GOOGLE_GEMINI_API_KEY=...
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

OpenAI shut down Sora 2 and the Videos API on 2026-09-24 with no
replacement. Both `sora-2` models are now retired, so constructing this
client raises `AiProviderConfigurationError`. The example below is kept
for reference only; use the `google-gemini` or `nova-reel` video engine.

Environment:

```json
"videos": {"engine": "openai", "model": "sora-2"}
```

```dotenv
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
from ai_api_unified import AIFactory, AIVoiceBase

voice: AIVoiceBase = AIFactory.get_ai_voice_client()
audio_bytes: bytes = voice.text_to_voice(
    text_to_convert="Hello from ai-api-unified",
    voice=voice.get_default_voice(),
    audio_format=voice.default_audio_format,
)

with open(f"out{voice.default_audio_format.file_extension}", "wb") as output_file:
    output_file.write(audio_bytes)
```

`text_to_voice` is keyword-only, and `voice` and `audio_format` are required;
`get_default_voice()` and `default_audio_format` supply the provider's own
defaults. Use `stream_audio(text)` for the streaming variant.

`AIFactory.get_ai_voice_client()` reads `voice.engine` from `models.json` and accepts an
explicit `voice_engine` override, matching the other capability constructors.
It delegates to `AIVoiceFactory.create()`, which remains supported and callable
with no arguments.

Calls are attributed to `providers.openai.user` on the OpenAI engine, falling back to the
`default_user` sentinel when it is unset — the same behavior the OpenAI
completions, embeddings, images, and videos clients already have. The other
voice engines expose no vendor user setting, so their calls carry no
originating caller id unless the application sets one through the
observability context.

The OpenAI voice client shares `AIOpenAIBase` with the other OpenAI-backed
capabilities, so it honors `completions.retry_policy` and `providers.openai.base_url`,
and accepts the same `retry_policy` and `base_url` constructor arguments:

```python
voice = AIFactory.get_ai_voice_client(
    voice_engine="openai",
    base_url="https://gateway.example.com/v1",
    retry_policy="none",
)
```

Only the `openai` voice engine accepts these; passing them to `google`,
`azure`, or `elevenlabs` raises `AiProviderCapabilityUnsupportedError` rather
than ignoring them.

A blank `completions.retry_policy` is treated as unconfigured and falls back
to `default` across every OpenAI-backed capability.

## Classification

A classifier answers one narrow question about a piece of text: which of
your options fits it, whether a condition holds, or where it sits on a
scale. It stands beside completions and embeddings as its own capability,
so you can use it for routing, triage, or moderation without writing a
prompt and parsing an answer. The model router (a later section) is one consumer.

### Quickstart

```python
from ai_api_unified import AIFactory, AIClassificationChoice

client = AIFactory.get_ai_classifier_client()   # classifier.engine from models.json
result = client.classify_choice(
    "My invoice shows two charges for one order.",
    [
        AIClassificationChoice(value="billing", description="Payments, invoices, refunds."),
        AIClassificationChoice(value="technical", description="Errors using the product."),
        AIClassificationChoice(value="other", description="Anything else."),
    ],
    instructions="Which team should handle this message?",
)
print(result.choice, result.confidence, result.usage.input_tokens)
```

Three question types share one result model, `AIClassificationResult`:

| Method | Question | Result fields |
| ------ | -------- | ------------- |
| `classify_choice(text, choices)` | Which option fits? | `choice`, `confidence`, `probabilities` |
| `classify_predicate(text, instructions=...)` | Does the condition hold? | `probability` |
| `classify_score(text, levels)` | Where on the scale, lowest level first? | `score` in [0, 1], `choice` (the level label) |

Every result also carries `refused` (the model declined), `usage`,
`token_source`, `provider_engine`, `model_name`, and `latency_ms`. Async
twins (`aclassify_choice` and so on) exist where `capabilities.supports_async`
is True. `capabilities` reports which question types the engine answers,
`max_choices`, `max_input_chars`, and `pricing`.

Before the call, the text goes through the PII middleware when it is
enabled, then is cut to `capabilities.max_input_chars` (6,000 by default)
keeping its head and tail: a label rarely needs more, and the provider
bills every input token.

### Models that can classify

| Engine | How it classifies | Models |
| ------ | ----------------- | ------ |
| `openai-decisions` | OpenAI's Decisions API (`POST /v1/decisions`). Native choice, predicate, and score answers with probabilities. About ten times faster than a chat completion. Bills input tokens only, at $0.10 per 1M. | `gpt-6-luna` (public beta) |
| `openai`, `openai-responses`, `openai-compatible` | Structured output (a one-field enum schema) | Every model the engine lists |
| `claude` | Structured output | Every model the engine lists |
| `google-gemini` | Structured output | Every model the engine lists |
| `bedrock`, `anthropic` | Structured output | Models whose capabilities include `supports_structured_output` (Claude, DeepSeek, Qwen, GLM); Nova cannot classify |

Through structured output, a predicate answer is a hard yes or no
(`probability` 1.0 or 0.0) and a score is the picked level's position on the
scale; engines with native classification report real probabilities.
Picking an engine without structured output fails at build with
`AiProviderCapabilityUnsupportedError`.

### Cost and metering

Every classification call runs inside the same observability lifecycle as
completions, so with observability on it emits the usual input, output,
error, and cost events, with `operation` set to `classify_choice`,
`classify_predicate`, or `classify_score`. The Decisions engine prices
under the registry key `decisions/gpt-6-luna`, since its rates differ from
the model's chat rates; when a response carries no `usage` block the engine
estimates input tokens from the request text and reports
`token_source="estimated"`. The completions engine a
structured-output classifier wraps meters the call, and that call carries
the tag `classifier_role` (`choice`, `predicate`, or `score`) so a dashboard can
split classifier spend from the engine's other traffic. `token_source` on
the result says where the counts came from: `provider`, `estimated`, or
`none`. A call that could not be metered logs a WARNING with
`ai_classifier_event=classifier_unmetered` on the `ai_api_unified.ai_base`
logger; it is never a silent zero.

### Configuration

| Setting | Meaning |
| ------- | ------- |
| `classifier.engine` (`models.json`) | Engine token. When unset, `openai-decisions` is used if `OPENAI_API_KEY` is set; otherwise the factory raises. |
| `classifier.model` | Model override; blank falls back to `completions.model` for structured-output engines, then the engine's default (`gpt-6-luna` for `openai-decisions`). |
| `AIFactory.get_ai_classifier_client(classifier_engine=, model_name=, base_url=)` | Arguments override the settings. |

Exports: `AIBaseClassifier`, `AIClassifierCapabilitiesBase`,
`AIClassificationChoice`, `AIClassificationLevel`, `AIClassificationResult`.

## Model fallback

A fallback chain keeps completions flowing when a model cannot serve a
request. You list the models to try, in order; the factory returns a client
with the same interface as a single engine; and a request that fails on one
candidate because that model is overloaded, rate limited, or out of quota is
retried on the next. The candidates can be on the same provider
(`claude-opus-5` then `claude-sonnet-5`) or on different ones
(`claude` then `openai` then `google-gemini`).

### Why it exists

Provider incidents, per-minute rate limits, and exhausted monthly quotas all
look the same to an application: a request that raises. Before this feature
you caught `AiProviderRequestError`, built a second client, and re-sent the
request by hand, in every code path. The chain moves that logic into the
client, and it uses the classification every engine already puts on the
error (`fallback_reason`, see
[Retry policy and typed request errors](#retry-policy-and-typed-request-errors))
so it acts only on failures another model could serve.

### Setting up a chain

From `models.json`, which is the usual way:

```json
"completions": {
  "engine": "claude",
  "model": "claude-opus-5",
  "fallbacks": ["openai:gpt-5.6-luna", "google-gemini:gemini-3.7-flash"],
  "fallback_on": "unavailable,rate_limited,quota_exhausted,auth_rejected"
}
```

`fallbacks` lists the `engine:model` pairs to try, in order, after the
primary fails; `fallback_on` names the failures that trigger a move (the
value above is the default).

Or in code, which overrides the settings:

```python
from ai_api_unified import AIFactory, AIFallbackCandidate, AiFallbackReason

client = AIFactory.get_ai_completions_client(
    completions_engine="claude",
    model_name="claude-opus-5",
    fallbacks=[
        ("openai", "gpt-5.6-luna"),
        AIFallbackCandidate(engine="google-gemini", model="gemini-3.7-flash"),
    ],
    fallback_on={AiFallbackReason.UNAVAILABLE, AiFallbackReason.QUOTA_EXHAUSTED},
)
```

Rules for the chain:

- The split in each `fallbacks` entry is on the first colon only, so Bedrock
  ids keep theirs: `bedrock:amazon.nova-lite-v1:0`. A pair with no model
  (`google-gemini`) uses that engine's default model.
- Each fallback engine needs its own credentials and optional extra, exactly
  as it would if it were the primary.
- The factory checks every fallback engine token against the provider
  registry when it builds the client, so a typo fails at startup, long
  before any outage. The [configured-model check](#configured-model-check)
  then probes each fallback model with its engine's credentials and prunes,
  with a WARNING, any that does not exist or cannot be reached (no key,
  rejected key, extra not installed).
- Passing `fallbacks=[]` in code turns the chain off even when the setting
  is present.

### What happens on a request

1. The request goes to the primary, unless the primary is resting after a
   recent failure (see [Cooldowns](#cooldowns)); it then starts at the
   first candidate that is not resting. The engine's own retry schedule
   runs first, so a rate limit fails over only after backoff runs out.
2. If the engine raises `AiProviderRequestError` with a `fallback_reason` in
   the configured set, the client logs a warning naming both candidates and
   the reason, benches the failed candidate, and sends the request to the
   next candidate.
3. The client builds a fallback the first time a request needs it. The
   configured-model check has already pruned fallbacks that are missing or
   unreachable, so a build failure here is rare (an unsupported option,
   for example). A key revoked after startup fails the request instead,
   as `AUTH_REJECTED` (see step 5). When a build fails, the client logs it
   once and skips that candidate for the rest of the process, so it never
   blocks the rest of the chain.
4. The client skips, with a warning, a fallback that lacks a capability the
   call needs (structured output, tool use, async, streaming).
5. After a `QUOTA_EXHAUSTED` or `AUTH_REJECTED` failure, the client skips
   the call's later candidates on the same account (same engine and
   `base_url`), with a warning, since they would fail the same way. Gemini
   quotas and Bedrock `ServiceQuotaExceededException` / `AccessDeniedException`
   apply to one model, so after those the client still tries the account's other models.
6. When every candidate has failed, the client raises the last failure.

| Reason | Fails over by default | Typical causes |
| ------ | --------------------- | -------------- |
| `UNAVAILABLE` | yes | 5xx, Anthropic 529 overloaded, Bedrock `ServiceUnavailableException`, a failed connection |
| `RATE_LIMITED` | yes | 429 after the engine's backoff is exhausted |
| `QUOTA_EXHAUSTED` | yes | OpenAI `insufficient_quota`, Anthropic low credit balance or workspace spend cap, Bedrock `ServiceQuotaExceededException`, a Gemini daily quota |
| `AUTH_REJECTED` | yes | 401/403, a revoked or invalid API key, a disabled organization, expired AWS credentials |
| `MODEL_UNAVAILABLE` | no | 404, an unknown or retired model id, a model not offered in the region |
| `TIMEOUT` | no | the call's `request_timeout_seconds` passed with no answer |
| `INVALID_OUTPUT` | no | a structured answer that is not valid JSON, not a JSON object, or fails the response model (`StructuredOutputValidationError`) |
| `None` | never | a safety refusal, request validation before any call |

`MODEL_UNAVAILABLE` is off by default because an unknown model id usually
means a configuration typo, and a working fallback would hide it. Turn it on
with `fallback_on` when you want an unknown or
retired model to fail over.

`TIMEOUT` is off by default because the deadline is yours, and a failover
adds the next candidate's wait on top of it. Turn it on to treat a provider
that hangs under load as an outage. It only helps with a short deadline:
set `request_timeout_seconds` on the call and `retry_policy="none"`, since
the Anthropic and OpenAI SDKs default to a 600-second timeout and retry a
timeout twice before the error reaches the fallback layer. The worst case
for one call is then the sum of the deadlines across the chain. Bedrock has
no per-call timeout, so its `TIMEOUT` comes from the botocore `read_timeout`
configured at client construction.

`QUOTA_EXHAUSTED` and `AUTH_REJECTED` are usually account-level failures:
every model on that provider's account fails the same way; the client skips
the account's other candidates (step 5), so a chain needs a candidate on a
different provider to recover from them. `AiProviderRequestError.account_wide`
reports which case a given error is. A rejected key is
often a configuration error too; the `failover` and `served_by_fallback` log
events (below) record it while traffic keeps flowing. Remove
`auth_rejected` from `fallback_on` to make it fail loudly instead.

`INVALID_OUTPUT` is off by default because the provider billed the bad answer, and
failing over buys a second billed answer on another model: a spending
decision. The error is `StructuredOutputValidationError`, a `ValueError` (so
existing handlers keep working) that carries the discarded answer's `usage`,
`raw_text`, and `finish_reason`. Turn the reason on to let the chain try the
next model, and the router escalate, on bad output; each discarded answer
then shows on the request's cost report as a wasted line item (see
"Per-request cost").

A safety refusal is never a trigger: retrying a
refused request on another model is a policy decision, not a reliability one,
and it belongs in your code if you want it.

### Cooldowns

Without a cooldown, every request starts at the primary. During an outage
each request then waits through the primary's retry backoff and sends one
more doomed request before failing over. A cooldown remembers the failure:

1. A request fails on the primary with a trigger reason and fails over.
   The primary now rests for a period that depends on the reason.
2. Requests during the rest skip the primary and start at the next
   candidate. No backoff, no wasted request.
3. When the rest ends, the primary is back in its place, and the next
   request tries it. If it serves, its cooldown is over
   (`candidate_recovered`); if it fails, it rests again.

| Reason | Default rest |
| ------ | ------------ |
| `UNAVAILABLE`, `RATE_LIMITED`, `TIMEOUT` | 30 seconds |
| `QUOTA_EXHAUSTED`, `MODEL_UNAVAILABLE` | 5 minutes |
| `AUTH_REJECTED` | 1 hour, or until the engine's key or base URL changes |
| `INVALID_OUTPUT` | none: a bad answer says nothing about the model's health |

Only reasons in `fallback_on` bench a candidate. An account-wide failure
(`QUOTA_EXHAUSTED` or `AUTH_REJECTED`, unless the engine marks it per-model)
benches every candidate on that account. Any candidate can rest, not only
the primary.

A resting candidate moves to the end of the order; it is not removed. If
every other candidate fails, the resting one is still tried, so a cooldown
never fails a request that would have been served without one. When every
candidate is resting, the chain's own order is used.

Cooldowns are per process and shared by every chain and router tier in it,
so a model benched by one chain is benched for all. A rotated key ends the
cooldowns on that engine: each bench records a one-way fingerprint of the
engine's credential settings (the same one the
[configured-model check](#configured-model-check) uses), and a bench taken
with other credentials is dropped. `AiCandidateCooldowns.reset()` ends every
cooldown at once.

Set the rest per reason, or turn cooldowns off, in `models.json`:

```json
"completions": {
  "fallbacks": ["claude:claude-sonnet-5"],
  "cooldown": {"enabled": true, "seconds": {"rate_limited": 10, "quota_exhausted": 600}}
}
```

`"cooldown": false` restores the earlier behavior: every request starts at
the primary. The factory takes the same values as `cooldown=`.

### Provider outage at startup

A primary whose provider is down when the process starts does not stop the
client. The [configured-model check](#configured-model-check) reports a
timeout, 5xx, or refused connection as `error` and keeps the model, and no
engine needs its provider to be up to be built. (The Gemini engine checks
its model while it is built; a 5xx, 429, timeout, or refused connection
there logs a WARNING and the build goes on.) The first request then fails
over and benches the primary, later requests start on the fallback, and
the primary is tried again each time its rest ends. The fallback is never
promoted for the life of the process.

A configuration error still stops the build: a missing or rejected key, a
missing extra, or a model the provider does not serve.

### Seeing which model answered

Every failover is logged, on the logger
`ai_api_unified.completions.ai_fallback_completions`, and each record carries
an `ai_fallback_event` field plus the candidates and reason involved, so a
log processor can count and alert on them without parsing the message:

| Level | `ai_fallback_event` | When |
| ----- | ------------------- | ---- |
| ERROR | `failover` | A candidate failed with a trigger reason and the request is moving to the next one. Fields: `operation`, `fallback_from`, `fallback_to`, `fallback_reason`, `status_code`. |
| WARNING | `served_by_fallback` | A fallback answered a request the primary could not. Once per request that moved. Fields: `operation`, `fallback_from`, `fallback_to`, `fallback_reason`. |
| ERROR | `chain_exhausted` | Every candidate failed; the last error is about to be raised. |
| ERROR | `candidate_unbuildable` | A fallback could not be constructed and is out of the chain for this process. |
| ERROR | `candidate_rejected_history` | On a family-bound turn, a sibling failed with no trigger reason, so it likely rejected the history. The failure that moved the call is about to be raised with this one as its cause. Fields: `operation`, `fallback_candidate`, `status_code`. |
| WARNING | `candidate_skipped` | A fallback lacks a capability the call needs, or its account already failed the call with `QUOTA_EXHAUSTED` or `AUTH_REJECTED`. |
| WARNING | `candidate_cooldown` | A candidate failed with a trigger reason and rests. Fields: `fallback_candidate`, `fallback_reason`, `cooldown_seconds`, `account_wide`. |
| INFO | `candidate_recovered` | A candidate that had rested served a request; its cooldown is over. |
| DEBUG | `candidate_resting` | A request is trying resting candidates last. Field: `fallback_resting`. |

The messages start with `FALLBACK`, `FALLBACK SERVED`, or `FALLBACK
EXHAUSTED`, so they stand out in plain log output too. A request the primary
serves logs nothing. In code:

```python
turn = client.send_conversation("system", messages, tools=tools)
turn.provider_engine, turn.model_name     # the candidate that served the turn

result = client.send_structured_output(prompt, response_model=Report)
result.provider_engine, result.model_name

text = client.send_prompt("...")
client.last_route                          # AIFallbackCandidate for the last call
client.last_route.label                    # "openai:gpt-5.6-luna"
```

Cost follows the route with no extra configuration: each engine emits its own
observability and cost events, so a call served by a fallback is billed to
that engine and model. `client.compute_completion_cost(...)` and
`price_per_1k_tokens()` price at the candidate that served the most recent
call. `client.capabilities` reports the primary's capabilities.

### Conversations, streaming, and batches

The engines shape their requests differently, and three consequences follow.

**Conversations fail over freely on the first turn, then only within a
family.** A history that holds engine-shaped entries (replayed
`raw_content`, tool results) cannot be sent to another engine family: Gemini
cannot read Claude's tool blocks. While the history is plain user,
assistant, and system text, a turn can fail over to any candidate. After
that the client reads which family shaped the history, sends the turn to
the first candidate of that family, and on a trigger failure tries the
family's later candidates in chain order. With the chain `claude:opus,
claude:sonnet, google-gemini:flash`, a 529 from Opus after a tool call goes
to Sonnet; Gemini is left out of that turn. The reason set and the
same-account skip apply as on any call. The next turn starts again at the
family's first candidate. Because the history decides the routing, one
client can serve conversations on different engines at the same time.

A family shares a request format. A sibling may still reject a block only
the model that produced it can read (a signed thinking block, a reasoning
item bound to one model).
When that happens the client raises the failure that moved the call, with
the sibling's rejection chained as its `__cause__`, and logs
`candidate_rejected_history`.

```python
messages = [{"role": "user", "content": "Summarize ticket VL-123."}]
turn = client.send_conversation("You are a support agent.", messages, tools=tools)
client.extend_messages_with_turn(messages, turn)          # shape of the engine that answered
for call in turn.tool_calls:
    messages.append(
        client.build_tool_result_message(
            tool_call_id=call.id,
            result=run_tool(call),
            messages=messages,                             # lets it match the history's shape
        )
    )
turn = client.send_conversation("You are a support agent.", messages, tools=tools)
```

Pass `messages=` to `build_tool_result_message` whenever one client serves
more than one conversation; without it the result takes the shape of the
engine that served the most recent call.

**Streams fail over only before the first chunk.** Once a candidate has
yielded output the caller already holds partial text, so a later error
propagates instead of restarting on another model.

**Batches and token counts always use the primary.** Batch jobs are
provider-specific, and token counts are model-specific.

### Per-call control

`provider_options={"fallback": "none"}` keeps one request on the primary,
for a call where a different model would be the wrong answer even during an
outage. The client removes the key before the options reach an engine, so
the same call works on a plain engine client too.

The prompt cache hint (`AIPromptCacheHint`) is provider-neutral and carries
across the chain.

### Configuration reference

| Setting | Meaning |
| ------- | ------- |
| `completions.fallbacks` (`models.json`) | Ordered `engine:model` pairs tried after the primary fails. Empty or absent means no chain. |
| `completions.fallback_on` | `AiFallbackReason` values (a list, or comma-separated) that trigger a move. Default `unavailable,rate_limited,quota_exhausted,auth_rejected`. |
| `fallbacks=` (factory argument) | Same as the setting, as `(engine, model)` tuples or `AIFallbackCandidate` objects; overrides the setting. `[]` turns the chain off. |
| `fallback_on=` (factory argument) | Same as the setting, as reasons or their values; overrides the setting. |
| `completions.cooldown` | `true` (default), `false`, or `{"enabled": ..., "seconds": {"<reason>": <seconds>}}`; see [Cooldowns](#cooldowns). Applies to the chain and to every router tier. |
| `cooldown=` (factory argument) | Same as the setting, or an `AIFallbackCooldownPolicy`; overrides the setting. |

Exports: `AiFallbackCompletions`, `AIFallbackCandidate`,
`DEFAULT_FALLBACK_REASONS`, `AiFallbackReason`, `AIFallbackCooldownPolicy`,
`AiCandidateCooldowns`.

### Testing a chain against live providers

A live test runs Claude as the primary with an unknown model id and OpenAI
as the fallback, so a real 404 drives a real second request. It also covers
a healthy primary that never builds its fallback and a first conversation
turn that fails over:

```bash
poetry run pytest -m nonmock tests/test_model_fallback_nonmock.py -q
```

It needs `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` in `.env` and skips without
them.

A second live test points a Gemini primary at a closed local port, so its
connection is refused the way a down provider's is, with a Claude fallback.
It proves the client starts, the first request is served by Claude and
benches Gemini, and the next request goes to Claude without trying Gemini:

```bash
poetry run pytest -m nonmock tests/test_fallback_cooldown_nonmock.py -q -s
```

It needs `GOOGLE_GEMINI_API_KEY` and `ANTHROPIC_API_KEY`.

## Model routing

A router sends each request to a tier of models that matches how demanding
the request is. A cheap classifier reads the latest user message, places it
in one of your tiers (easy, medium, hard, or names of your own), and the
request runs on that tier's model chain. A trivial question stops costing
flagship rates; a hard one still gets the strongest model.

### Why it exists

One configured model serves every prompt, whatever its difficulty. Routing
by complexity lowers cost and latency on the easy majority without giving
up quality on the hard minority, and it does so in the library, so an
application changes nothing but configuration.

### Setting up a router

Write the tiers as JSON, ascending capability. Each tier lists an ordered
`engine:model` chain: the first entry is the tier's primary and the rest
are its fallback chain. Tiers named `easy`, `medium`, and `hard` have
built-in descriptions; any other name needs a `description` the classifier
can read.

```json
{
  "tiers": [
    {"name": "easy",   "models": ["openai:gpt-4.1-mini", "google-gemini:gemini-2.5-flash-lite"]},
    {"name": "medium", "models": ["claude:claude-sonnet-5", "openai:gpt-5.6-luna"]},
    {"name": "hard",   "description": "Multi-step reasoning, long synthesis, cross-file code, or anything where a wrong answer is costly.",
                       "models": ["claude:claude-opus-5-5"]}
  ],
  "classifier": {"engine": "openai-decisions", "model": "gpt-6-luna", "timeout_seconds": 5},
  "default_tier": "medium",
  "on_exhausted": "escalate"
}
```

Put it in the `router` section of `models.json`, keep `completions.engine`
and `completions.model` as they are, and call the factory as usual:

```python
client = AIFactory.get_ai_completions_client()      # an AiRouterCompletions
print(client.send_prompt("What is 2 + 2?"))          # routed to the easy tier
print(client.last_tier, client.last_decision)
```

In code, pass `router=AIRouterConfig(...)` instead. The configured
completions LLM, with its `completions.fallbacks` chain, is the **default
route**: where calls go when no classifier is available, when the
classifier fails, or when `default_tier` is unset.

### The classifier

The router uses a classification engine (see Classification). Resolution
order: the router's own `classifier` entry, then the top-level
`classifier` section of `models.json`, then `openai-decisions` when
`OPENAI_API_KEY` is set. Set either to `none` to route without a classifier. The
configured-model check probes the classifier model with the tiers; if it
is missing or unreachable, or the classifier cannot be built, the router
logs `classifier_unavailable` once and degrades to the default route, and
never raises at startup.

The classifier reads the latest user message only, cut to the engine's
`max_input_chars`. The system prompt is left out: it is long, constant,
and carries no complexity signal. The tier descriptions carry the policy,
and `classifier.instructions` prepends your own guidance.

When the classifier is unsure, set `classifier.min_confidence` (0 to 1):
a pick whose confidence is below it goes one tier up (method
`low_confidence`). A wrong pick toward the stronger tier costs money; one
toward the weaker tier costs a worse answer, so an unsure call takes the
stronger tier. A classifier that reports no confidence (the
structured-output classifier) keeps its pick, and a pick already on the
top tier stays there. `openai-decisions` reports one.

The classifier's input and time are capped (`max_input_chars`,
`timeout_seconds`), and its cost lands on the request ledger as
`classifier_usd_cost`. At startup the router also compares prices: when
the classifier's input rate is at or above the cheapest tier primary's,
it logs `classifier_costly` once, since classifying then costs about as
much as sending every request to that tier. A per-call check is not
possible: the routed call's cost is only known after it returns.

### Thinking per tier

Each tier may set its own `thinking` level (see
[Thinking levels](#thinking-levels)); a tier without one uses the
router's level (`completions.thinking`, the factory's `thinking=`, or
`client.thinking_level`). A per-call level still wins.

```json
"tiers": [
  {"name": "easy", "models": ["claude:claude-haiku-4-5"], "thinking": "off"},
  {"name": "hard", "models": ["claude:claude-opus-5-5"],  "thinking": "high"}
]
```

### Follow-up turns

A plain conversation (no tool calls) is classified on each turn by
default (`followup: "reclassify"`): a cheap follow-up such as "summarize
that" may go to a cheaper tier, which is what routing is for. Two ways
keep a conversation from dropping:

- `provider_options={"route_floor": "medium"}` on a structured-output or
  conversation call: the classifier's pick (or the default route) is
  raised to at least that tier (method `floor`). A shared client serving
  many conversations passes each conversation's floor this way, for
  example the `client.last_tier` it recorded for that conversation.
- `"followup": "ratchet"` in the router config: a turn whose history
  already holds an assistant turn never goes below the tier that served
  this client's previous call. The library has no conversation id, so
  this is exact only with one client per conversation.

An override (`route`) and a tool-shaped history (method `pinned`) are
never moved by a floor.

### What happens on a request

1. A per-call `route` option or `for_tier()` pins the tier (method
   `override`).
2. A history that already holds engine-shaped entries (tool turns) skips
   the classifier and stays in a tier whose engines share that family,
   preferring the tier that served the previous turn (method `pinned`).
3. Otherwise the classifier names a tier (method `classifier`), one tier
   up when its confidence is below `min_confidence` (method
   `low_confidence`). No classifier, an error, a refusal, or an unknown
   tier name routes to `default_tier` or the default route (method
   `default`) and logs `classifier_failed`. A `route_floor`, or the
   follow-up ratchet, then raises the pick to the floor (method `floor`).
4. The call runs on the tier's chain, which fails over inside the tier as
   any chain does, cooldowns included: a tier model resting after a
   failure is tried last ([Cooldowns](#cooldowns)).
5. If the chain runs out of candidates on a trigger reason
   (`fallback_on`), the router escalates to the next tier up and logs
   `tier_exhausted`; with `on_exhausted: raise`, on a non-trigger error, or
   on the top tier, the error propagates. With `on_exhausted: any`, once
   no tier above can serve, the router moves down instead: the tiers
   below the pick, nearest first, each logged as `tier_exhausted`
   (`direction: down`) plus a `tier_descended` WARNING, since the answer
   may be weaker than the classifier asked for. The default stays
   `escalate`: a request the classifier called hard should fail visibly
   rather than be answered by a weaker model without anyone asking for
   that. Tier models that are missing or
   unreachable were pruned at startup by the configured-model check; if a
   tier's primary still cannot be built, the router logs
   `tier_unbuildable` once and skips the tier the same way.

### Seeing where a request went

Every call logs a `routed` record at INFO before it runs, on the logger
`ai_api_unified.completions.ai_router_completions`, with an
`ai_router_event` field so a log processor can count and alert without
parsing the message:

| Level | `ai_router_event` | When |
| ----- | ----------------- | ---- |
| INFO | `routed` | Every call. Fields: `route_id`, `tier`, `tier_index`, `method`, `operation`, `classifier_engine`, `classifier_model`, `confidence`, `probabilities`, `classifier_latency_ms`, `classifier_input_tokens`, `classifier_token_source`, `classifier_usd_cost`, `caller_id`, `session_id`, `workflow_id`. |
| WARNING | `classifier_unavailable` | Once at build: no classifier, with `reason` and `routed_to`. |
| WARNING | `classifier_failed` | The classifier raised, refused, or named no tier; the call went to `routed_to`. Fields: `error`, `fallback_reason`, `latency_ms`. |
| ERROR | `tier_exhausted` | A tier's chain ran out of candidates on a trigger reason; `escalate_to` names the next tier, `direction` is `up` or `down`. |
| WARNING | `tier_descended` | With `on_exhausted: any`, the call is moving below the classifier's pick. Fields: `tier`, `descend_to`, `fallback_reason`. |
| WARNING | `classifier_costly` | Once at build: the classifier's input rate is at or above the cheapest tier primary's. Fields: `classifier_model`, `classifier_input_per_1m`, `cheapest_tier`, `cheapest_tier_input_per_1m`. |
| ERROR | `tier_unbuildable` | A tier's primary could not be built; the tier is out for this process. |

The router also pushes `route_id`, `route_tier`, and `route_method` onto
the observability context around the downstream call, so the engine's own
input, output, and cost events carry them as `tag_route_tier` and so on:
cost by tier needs no extra work. The classifier call itself carries
`tag_router_role=classifier`.

### Per-call control

- `provider_options={"route": "hard"}` pins one structured-output or
  conversation call to a tier; `"default"` pins it to the default route.
  `provider_options={"route_floor": "medium"}` sets a minimum tier
  instead (see Follow-up turns), and `{"route_ceiling": "medium"}` a
  maximum: a pick above it is lowered to it (method `ceiling`), and a
  failed tier never escalates past it. The ceiling wins over the
  follow-up ratchet; a `route_floor` above it raises `ValueError`, and a
  `route` override is not moved. The router removes all three keys before
  options reach the engine.
- `client.for_tier("hard")` returns that tier's chain client, for
  `send_prompt` and streaming, which take no `provider_options`.
- `client.last_tier`, `client.last_decision`, and `client.last_route_client`
  report the most recent call.

### Route preview

`client.preview_route(...)` answers "where would this request go, and what
would it cost?" without sending it to the answering model:

```python
preview = client.preview_route(prompt, max_response_tokens=2000)
preview.tier, preview.method, preview.confidence   # "hard", "classifier", 0.96
preview.engine, preview.model, preview.thinking    # "claude", "claude-opus-5-5", medium
preview.input_tokens, preview.input_token_source   # 26, "estimated"
preview.usd_input, preview.usd_max                 # input cost; input plus a full 2000-token answer
preview.classifier_usd                             # what classifying cost
```

- A router picks the tier exactly as a call would: the classifier,
  `min_confidence`, `route`, `route_floor`, and `route_ceiling` in
  `provider_options`, the
  follow-up ratchet, and a pinned tool history. The classifier call is real
  and billed, and its cost is on the preview. Nothing is logged as routed.
- A fallback chain previews the candidate it would try first: the primary,
  or the first candidate that is not resting after a recent failure.
- A plain engine previews itself (method `direct`).
- Input tokens are estimated locally (characters / 4, over the system
  prompt, prompt, and message text). `bool_count_tokens=True` asks the
  provider's free token count instead where the engine supports it.
- `usd_max` is the input cost plus `max_response_tokens` (the engine's
  default when omitted) at the output rate, thinking included: the most
  the first attempt should cost. Fallbacks and escalations, if they
  happen, add to it. An unpriced model has no cost estimate.

### Pinning a model for one call

`model=` on `send_prompt_with_result` (and its async twin) names the model
for one call; on `send_conversation` and `send_structured_output` (and
their async twins) the reserved `provider_options["model"]` does the same:

```python
result = client.send_prompt_with_result(prompt, model="claude:claude-opus-5-5")
result.model_pin                           # "claude:claude-opus-5-5"
result.route_method                        # "model_pin"

turn = client.send_conversation(
    system_prompt, messages, provider_options={"model": "openai:gpt-5.4"}
)
```

The call skips routing (no classifier, no tier pick, and no `route`,
`route_floor`, or `route_ceiling` option) and keeps everything else: fallback, metering,
logging, and redaction. The `model` key is removed before the request
reaches the provider.

- Name the model as `engine:model`, or as a bare model id that the
  configuration already uses. Bedrock ids keep their colons
  (`bedrock:amazon.nova-pro-v1:0`).
- On a fallback chain the call starts on the pinned model, then tries the
  chain's own candidates in order. A model outside the chain is built on
  first use and kept for later pins; unpinned calls never use it.
- On a router the call runs on the chain of the tier that holds the pinned
  model, else on the default chain, and never escalates to another tier.
- On a single configured model, the pinned model falls back to the
  configured one.
- The models.json allow/deny policy applies; a denied model raises
  `ValueError`. A pinned model that cannot be built raises instead of
  moving silently to another model.

Operators can find every pinned call: it logs one `MODEL PIN` line
(`ai_route_event: "model_pin"`), and its events and request cost report
carry the tags `model_pin` (the engine:model) and `route_method`
(`model_pin`). A routed call's `ROUTED` line shows method `model_pin`.

### Configuration reference

| Setting | Meaning |
| ------- | ------- |
| `router` (`models.json`) | `tiers` (each with optional `description`, `fallback_on`, `thinking`), optional `classifier`, `default_tier`, `on_exhausted` (`escalate`, `any`, `raise`), `followup` (`reclassify`, `ratchet`), `fallback_on`. |
| `router.classifier` | `engine`, `model`, `base_url`, `timeout_seconds`, `instructions`, `min_confidence`. |
| `classifier` (`models.json`) | The classifier the router uses when its own `classifier` entry is absent, `min_confidence` included; `{"engine": "none"}` routes without one. |
| `AIFactory.get_ai_completions_client(router=AIRouterConfig(...))` | Code-first configuration; overrides the setting. |

Exports: `AiRouterCompletions`, `AIRouterConfig`, `AIRouterTier`,
`AIRouterClassifierConfig`, `load_router_config`.

## Configured-model check

Every model the completions client may call is checked before the client
is returned: the completions LLM, each fallback, each router tier
candidate, and the router's classifier. The check asks two questions:
does the provider serve this model, and can this process reach it with
the credentials the engine will use? A typo, a retired id, a model not
offered in your region, or a fallback whose key is missing then shows up
at startup, not in the middle of a failover.

### When it runs

- On every `AIFactory.get_ai_completions_client()` call, with or without
  a chain or router. A lone primary is checked too.
- Each model is probed at most once per process for a given engine,
  model id, base URL, and credentials. Later factory calls reuse the
  result and make no network call. A probe runs again only when one of
  those four changes: an edited `models.json` that names a new model or
  base URL, or a rotated key. Credentials enter the memo key as a
  one-way hash of the settings the probe reads (for example
  `ANTHROPIC_API_KEY` and `providers.anthropic.base_url`); the key itself
  is never stored. AWS credentials from an instance role or an SSO cache
  are not in the hash; set `AWS_PROFILE` or the key variables when you
  need a rotation to force a recheck, or call
  `AiModelChecker.reset_cache()`.
- A probe that failed for a transient reason (timeout, 5xx, dropped
  connection) is not cached, so the next factory call probes that model
  again.
- `models.json` itself is reloaded when its modification time changes.
- On `get_ai_embedding_client()`, `get_ai_images_client()`, and
  `get_ai_video_client()`, the configured model is checked the same way,
  with the probe of the same provider (OpenAI, Gemini, or Bedrock, which
  also serves Titan, Nova Canvas, and Nova Reel). That model is the only
  one the client has, so a missing model raises `ValueError` naming the
  setting (`embeddings.model`, `images.model`, `videos.model`) and an
  unreachable one raises `AiProviderConfigurationError`, in every mode but
  `off`. Voyage embeddings, an engine's default model (no model set), and
  voice are not probed.

### What it does

- One free metadata call per model, never a completion: OpenAI
  `models.retrieve`, Anthropic `models.retrieve`, Gemini `models.get` on
  the raw id, Bedrock `get_foundation_model` (or `get_inference_profile`
  for `us.`, `eu.`, `apac.`, and `global.` ids) in the configured AWS region, an
  OpenAI-compatible server's `/models` list, and the Decisions API's fixed
  model set. Calls run in parallel, so startup pays one round trip.
- Each model gets one status:

| Status | Meaning | Effect |
| ------ | ------- | ------ |
| `available` | The provider returned the model's metadata. | Kept. |
| `missing` | The provider has no such model (404, or an id outside a fixed catalog). | Pruned, WARNING. |
| `unreachable` | The engine cannot call the model: no key, a key rejected as invalid (401, Gemini `API_KEY_INVALID`, Bedrock `UnrecognizedClient` / `InvalidSignature` / `ExpiredToken`, no AWS credentials), no base URL for an OpenAI-compatible server, or the engine's extra not installed. | Pruned, WARNING. |
| `unverifiable` | The probe cannot ask, but nothing says the call would fail: the engine default model (no id configured), a permission denial (403, Bedrock `AccessDenied`) that may cover only the catalog call, Vertex auth, or a server without a model catalog. | Kept, INFO. |
| `error` | The probe failed for another reason (timeout, 5xx, network). | Kept, WARNING; not cached. |

- A pruned fallback leaves the chain: `configured model openai:gpt-nope
  is not available and will be ignored`, or `configured model
  claude:claude-haiku-4-5 cannot be reached and will be ignored`. A
  pruned tier candidate leaves its tier, and the router drops a tier left
  with no models. A request for a dropped tier (by the classifier, a
  `route` override, or `for_tier`) goes to the next **higher** surviving
  tier, never a lower one, and to the default route when nothing is
  above. A dropped `default_tier` moves the same way.
- The primary completions model cannot be pruned: it is the default
  route. A missing primary raises `ValueError`; an unreachable one raises
  `AiProviderConfigurationError` (the engine's own error comes first when
  its constructor already refuses, for example a missing extra).
- A missing or unreachable classifier model logs `classifier_unavailable`,
  and the router degrades to its default route.

Log records go to `ai_api_unified.util.ai_model_checker` with an
`ai_model_check_event` field: `model_missing` (WARNING),
`model_unreachable` (WARNING), `model_check_error` (WARNING),
`model_unverifiable` (INFO), `model_available` (DEBUG); the factory adds
`router_tier_pruned` and `router_disabled` (WARNING) on
`ai_api_unified.ai_factory`. `AIFactory.last_model_check_report()` returns
the results of the most recent check.

### Configuration

| Setting | Meaning |
| ------- | ------- |
| `model_check.mode` (`models.json`) | `prune` (default): drop missing and unreachable fallbacks and tier models with a warning. `strict`: raise on any of them. `off`: skip the check. `model_check.timeout_seconds` bounds each probe (default 10). |

### Startup requirements

The model check proves each model exists and can be reached. It does not
know what your callers need. Declare that in `completions.require`, and
the factory checks it once at build, without building any client:

```json
"completions": {
  "require": {
    "capabilities": ["structured_output", "tool_use", "streaming"],
    "second_provider": true,
    "mode": "warn"
  }
}
```

The check covers each **route**: the default chain (the primary plus its
fallbacks) and every router tier, after pruning.

- `capabilities`: each route's primary must support every listed
  capability: `structured_output`, `tool_use`, `streaming`, `async`,
  `batch`, `token_counting`, `prompt_cache_hint`, or
  `data_residency_constraint`. The primary is what counts: a call starts
  there, and a primary without the capability raises, so a fallback that
  has it does not help. Capabilities come from each engine's model table
  (OpenAI, Anthropic, Gemini, Bedrock); a model with no table entry (an
  OpenAI-compatible server, the engine default model) is logged at INFO
  and not counted as a failure.
- `second_provider`: each route must hold a model on a provider other than
  its primary's, so one provider's outage cannot take the route down.
  Claude on Bedrock counts as AWS, not Anthropic; each OpenAI-compatible
  server is its own provider.
- `mode`: `warn` (default) logs each unmet requirement as a WARNING with
  `ai_config_event=requirement_unmet` on
  `ai_api_unified.routing.config_checks`; `strict` raises `ValueError`
  at build.

The factory's `require=` takes the same values and overrides the file.

### Which configuration routed a call

Every completions client the factory builds records a `config_id`: a
12-character hash of the routing configuration it was built from (primary,
fallbacks, trigger reasons, router, cooldown, thinking level). The same
configuration gives the same id in every process and on every machine;
any change gives a new one. The hash is taken before the model check
prunes anything, so a model that is down at startup does not change it.
Add your own label with a top-level `config_version` in `models.json`
(a date, a release, a commit).

Both ride on every call:

- as the `config_id` and `config_version` observability tags, so every
  input, output, and cost event carries them (`tag_config_id`,
  `tag_config_version`), and so does the request cost report
  (`client.last_request_cost.tags`, the `ai_api_request_cost` event);
- as `config_id` and `config_version` fields on the router's `routed`
  record, and `config_id` on `tier_exhausted` and on the fallback chain's
  `failover`, `served_by_fallback`, and `chain_exhausted` records.

Tags you set on the observability context yourself win over these.
`client.config_id` and `client.config_version` read them in code.

Embeddings, images, videos, voice, and classifier clients from the factory
carry an identity too: a `config_id` hashed from their capability, engine,
and model (and embedding dimensions), plus the same `config_version` and
`service`. Their calls and request cost reports carry the same tags.

### Checking your own configuration

```bash
poetry run pytest -m nonmock tests/test_configured_models_nonmock.py -q -s --models-config ./models.json
```

prints every model the file names with its status and fails on any that
is missing or unreachable. To watch the pruning itself:

```bash
poetry run pytest -m nonmock tests/test_model_check_nonmock.py -q -s
```

reads `tests/fixtures/models_missing_test.json`, whose chain and easy tier
each list an id no provider serves, and proves both are reported missing
and pruned with the warning while the configuration still builds; with
`--models-config ./models.json` it does the same against your file,
injecting a bogus id in memory.

## Configuration

### models.json: one file for everything but secrets

Engines, models, the fallback chain, the router, the classifier, the
configured-model check, provider settings, the middleware, and the
allow/deny policy live in one file,
`models.json` (or `.yaml`). Copy `models_sample.json`, which sets every
section, and point `AI_MODELS_CONFIG` at it (or leave it unset: the library
picks up `./models.json` when it exists; a blank value turns the file off;
the value may also be inline JSON).

```json
{
  "allow": ["claude:*", "google-gemini:*", "openai-decisions:*"],
  "deny":  ["openai:gpt-4*", "bedrock"],
  "completions": {"engine": "google-gemini", "model": "gemini-2.5-flash", "thinking": "medium",
                  "fallbacks": ["claude:claude-sonnet-5"],
                  "fallback_on": "unavailable,rate_limited,quota_exhausted,auth_rejected"},
  "classifier":  {"engine": "openai-decisions", "model": "gpt-6-luna"},
  "router":      {"tiers": [{"name": "easy", "models": ["google-gemini:gemini-2.5-flash-lite"]},
                            {"name": "hard", "models": ["claude:claude-opus-5-5"]}],
                  "default_tier": "easy"},
  "model_check": {"mode": "prune", "timeout_seconds": 10},
  "embeddings":  {"engine": "google-gemini", "model": "gemini-embedding-001"},
  "images":      {"engine": "google-gemini", "model": "gemini-3.1-flash-image"},
  "videos":      {"engine": "google-gemini", "model": "veo-3.1-lite-generate-preview"},
  "voice":       {"engine": "google", "tts_model": "gemini-2.5-pro-tts", "language": "en-US"}
}
```

To convert an existing `.env`, run

```bash
poetry run python scripts/migrate2json.py            # writes ./models.json
poetry run python scripts/migrate2json.py --comment-env   # and comments the moved keys out of .env
```

It places each selection in its section, embeds a router file the old
`.env` pointed at, fills structural defaults from `models_sample.json`,
leaves the allow and deny lists empty, and never touches secrets.

The old selection keys are deprecated. One left in `.env` or the process
environment is still accepted and overrides the file, so a deployment does
not break, and the library logs one warning per key naming the
`models.json` field and the migrator command
(`ai_config_event=deprecated_selection_key`). Precedence, highest first: an
explicit factory argument, a deprecated environment key, the file, the
library default. Only secrets stay in `.env`: API keys, AWS credentials,
and `GOOGLE_APPLICATION_CREDENTIALS`. The AWS region may sit in either
place, since AWS hosts set `AWS_REGION` themselves; the environment wins
and draws no warning.

### Schemas and validation

Both configuration files have a published JSON Schema, generated from the
models the library loads them with, so the two cannot disagree:

| File | Schema |
| ---- | ------ |
| `models.json` | [`models.schema.json`](src/ai_api_unified/schemas/models.schema.json) |
| A separate middleware YAML, if you keep one | [`middleware.schema.json`](src/ai_api_unified/schemas/middleware.schema.json) |

Both ship in the package under `ai_api_unified/schemas/`. Point your
editor at one for completion and inline errors: add a `"$schema"` key at
the top of `models.json` (as `models_sample.json` does), or a
`# yaml-language-server: $schema=<url>` line at the top of the YAML. The
URLs are `https://raw.githubusercontent.com/davecthomas/ai-api-unified/main/src/ai_api_unified/schemas/models.schema.json`
and the same path ending in `middleware.schema.json`.

To check a file the way the library loads it (useful in CI):

```bash
poetry run python -m ai_api_unified.config_schema validate models.json middleware.yaml
```

It prints `valid` or `INVALID` with the errors for each file and exits 1
when any file fails. `print models` or `print middleware` writes a schema
to stdout. Where a loader accepts more than one spelling (a tier's
`models` or `candidates`, `fallback_on` as a list or a comma-separated
string), the schema accepts each one, so a file the library loads never
fails its schema.

### Allow and deny lists

`allow` and `deny` hold `engine:model` globs: `claude:*`, `*:gpt-4*`,
`openai-decisions:gpt-6-luna`; a bare engine token (`bedrock`) means
`bedrock:*`. Deny wins over allow, and an empty allow list allows every
model. The policy applies wherever a model enters: the completions LLM,
each fallback candidate, each router tier candidate, the router's
classifier, and the embeddings, image, and video selections.

- The factory prunes a denied fallback, tier, or classifier model the way
  it prunes a missing one (see Configured-model check) and warns
  `configured fallback model openai:gpt-4o is denied by policy pattern 'openai:gpt-4*' and will be ignored`
  (`ai_model_policy_event=model_denied`). A tier left with no models
  routes to the next higher surviving tier.
- A denied primary, or an explicit `model_name=` argument for a denied
  model, raises `ValueError` naming the pattern. Runtime never calls a
  denied model.

### Engine selectors

There is no implicit default provider. Set the engine for each capability
you use in `models.json`:

| `models.json` field | Valid values |
| ------------------- | ------------ |
| `completions.engine` | `openai`, `openai-responses`, `openai-compatible`, `claude`, `google-gemini`, `bedrock`, Bedrock-routed aliases such as `nova`, `anthropic`, `llama`, `mistral`, `cohere`, `ai21`, `rerank` |
| `embeddings.engine`  | `openai`, `titan`, `google-gemini`, `voyage` |
| `classifier.engine`  | `openai-decisions`, `openai`, `openai-responses`, `openai-compatible`, `claude`, `google-gemini`, `bedrock`, `anthropic` (see Classification) |
| `images.engine`      | `openai`, `google-gemini`, `nova-canvas`, `bedrock`, `nova` |
| `videos.engine`      | `openai`, `google-gemini`, `bedrock`, `nova`, `nova-reel` |
| `voice.engine`       | `openai`, `google`, `azure`, `elevenlabs` |

### Model selections

Every section of `models.json` is documented in
[`models_sample.json`](models_sample.json). The fields, by section:

| Section | Fields |
| ------- | ------ |
| `config_version` | A label for this configuration (top level); see [Which configuration routed a call](#which-configuration-routed-a-call) |
| `service` | The service this configuration belongs to (top level); carried as the `service` tag for cost attribution, see [Per-request cost](#per-request-cost) |
| `completions` | `engine`, `model`, `fallbacks` (see Model fallback), `fallback_on`, `retry_policy`, `thinking` (see Thinking levels), `cooldown` (see Cooldowns), `require` (see [Startup requirements](#startup-requirements)) |
| `classifier` | `engine`, `model`, `base_url`, `timeout_seconds`, `instructions`, `min_confidence` |
| `router` | `tiers`, `classifier`, `default_tier`, `on_exhausted`, `followup`, `fallback_on` (see Model routing) |
| `model_check` | `mode` (`prune`, `strict`, `off`), `timeout_seconds` (see Configured-model check) |
| `embeddings` | `engine`, `model` (`gemini-embedding-2` enables multimodal embeddings), `dimensions` (`0` keeps the provider default) |
| `images` | `engine`, `model` |
| `videos` | `engine`, `model`, `output_dir`, `poll_interval_seconds`, `timeout_seconds`, `bedrock_output_s3_uri` (Nova Reel, unless a request names its own) |
| `voice` | `engine`, `tts_model`, `language` |
| `geo_residency` | `US`, `USA`, or `United States` normalize to US routing where supported |
| `providers` | Per-provider settings that are not secrets; see [Provider settings](#provider-settings) |
| `middleware` | The PII redaction and observability middleware list; see [Middleware](#middleware) |
| `strict_deprecations` | `true` turns a deprecated-model warning into a startup error |

### Provider settings

| Field | Meaning |
| ----- | ------- |
| `providers.openai.base_url` | API base-URL override for the OpenAI engines (https required) |
| `providers.openai.user` | End-user id sent with each OpenAI request |
| `providers.anthropic.base_url` | API base-URL override for the `claude` engine (https required) |
| `providers.anthropic.admin_base_url` | Separate override for the Anthropic Admin API lookup; the admin key does not follow `base_url` |
| `providers.anthropic.user` | End-user id for Claude calls |
| `providers.google.auth_method` | `api_key` (default) or `service_account` (Vertex) |
| `providers.google.project_id`, `providers.google.location` | Vertex project and region, for `service_account` |
| `providers.google.base_url` | API base-URL override for `google-gemini` (https required) |
| `providers.aws.region` | Bedrock region, used when the environment sets no `AWS_REGION` (default `us-east-1`) |
| `providers.openai_compatible.base_url` | The server's `/v1` base URL (required for `openai-compatible`) |
| `providers.openai_compatible.structured_output` | `json_schema` (default) or `json_object` |
| `providers.openai_compatible.context_window` | Input token limit for the client-side context guard |
| `providers.openai_compatible.thinking_control` | `reasoning_effort`, `enable_thinking`, or `none` (default); see [Thinking levels](#thinking-levels) |
| `providers.azure_speech.region` | Azure speech region |

### What stays in `.env`

Secrets only: `OPENAI_API_KEY`, `OPENAI_COMPATIBLE_API_KEY`,
`ANTHROPIC_API_KEY`, `ANTHROPIC_ADMIN_KEY`, `GOOGLE_GEMINI_API_KEY`,
`GOOGLE_APPLICATION_CREDENTIALS`, `VOYAGE_API_KEY`, the AWS credentials,
`ELEVEN_LABS_API_KEY`, and `MICROSOFT_COGNITIVE_SERVICES_API_KEY`; plus
`AI_MODELS_CONFIG`, which names the file (a path to `models.json`/`.yaml`,
or inline JSON; unset uses `./models.json` when present; blank turns the
file off), and `AI_MODELS_ENV`, which picks an environment (below).
`AWS_REGION` may sit here too, since AWS hosts set it.

### Per-environment overrides

Any setting in the file can differ by environment. Put partial configs
under `environments`, and set `AI_MODELS_ENV` to the one to apply:

```json
"environments": {
  "prod": {
    "model_check": {"mode": "strict"},
    "completions": {"require": {"mode": "strict"}}
  },
  "dev": {"completions": {"model": "gemini-2.5-flash-lite", "fallbacks": []}}
}
```

The entry is merged over the rest of the file before it is validated:
objects merge key by key, and lists and values replace (a `fallbacks` or
`router.tiers` list in an environment is the whole list). An
`AI_MODELS_ENV` that names no entry stops the load and lists the defined
ones. Every client the factory builds carries the applied environment as
the `config_env` tag, beside `config_id`, on its events and request cost
reports. A `.env` or process-environment setting still overrides the
merged file, as it overrides the file alone.

### Cataloged Completions Models

The full list of cataloged models for every capability is in
[Supported models](#supported-models). Engine defaults sit one generation
behind the newest cataloged model. Gemini images generate through
`generate_content` because Imagen 4 is retired, and `person_generation`
on Gemini images applies only in Vertex AI mode; the Gemini Developer API
rejects it.

An uncataloged model name passes through to the provider on the OpenAI and
Claude engines (with a conservative default context window); the
`google-gemini` engine falls back to its default model and logs a warning.
Deprecated models warn once per process with a sunset date and replacement;
retired models fail fast (see the pricing registry).

### API Base URL Overrides

The `claude`, `openai`, `openai-responses`, `openai-compatible`, and
`google-gemini` engines accept a base-URL override, so provider traffic can
route through an LLM gateway, a corporate egress proxy, a recording proxy in
tests, or any OpenAI-compatible server:

| Engine | `models.json` field |
| --- | --- |
| `claude` | `providers.anthropic.base_url` |
| `openai`, `openai-responses` | `providers.openai.base_url` |
| `openai-compatible` | `providers.openai_compatible.base_url` (required) |
| `google-gemini` | `providers.google.base_url` |

```json
"providers": {
  "anthropic": {"base_url": "https://llm-gateway.internal/anthropic"},
  "openai":    {"base_url": "http://localhost:11434/v1"}
}
```

The first routes Claude traffic through a gateway; the second points the
`openai` engine at a local OpenAI-compatible server.

Per client, when one process needs several destinations:

```python
client = AIFactory.get_ai_completions_client(
    completions_engine="claude",
    base_url="https://llm-gateway.internal/anthropic",
)
```

Rules and rationale:

- **https is required.** An override redirects your API keys to that host, so
  plain `http://` is rejected with `AiProviderConfigurationError` unless the
  host is `localhost`, `127.0.0.1`, or `::1` (local gateways and test
  servers).
- **The SDKs' own variables never take effect unvalidated.** The OpenAI,
  Anthropic, and Google SDKs read their own `OPENAI_BASE_URL`,
  `ANTHROPIC_BASE_URL`, and `GOOGLE_GEMINI_BASE_URL` variables. The library
  passes the resolved value to the SDK explicitly, which keeps a stray native
  variable from taking effect without the https check. (On `google-gemini` with no override
  set, the SDK still resolves its own default, including those native
  variables, which this library does not validate.)
- **Precedence:** the `base_url` argument, then the `providers.<name>.base_url`
  field, then (OpenAI only) the deprecated `OPENAI_BASE_URL` and geo-residency
  routing, then the provider default.
- **Organization lookups follow the override**, so finops attribution does not
  bypass your gateway — with one deliberate exception: the Anthropic **Admin
  API key** grants org-wide read/write, far beyond the inference key, so it
  does not follow `providers.anthropic.base_url`. Routing inference through a
  gateway never silently hands that gateway an administration credential. Set
  `providers.anthropic.admin_base_url` to opt the admin lookup in as well.
- The same rules validate the deprecated `OPENAI_BASE_URL`, since other
  tooling commonly sets that name process-wide.
- Engines without SDK support (Bedrock-routed, `titan`, `voyage`, voice)
  raise `AiProviderCapabilityUnsupportedError` when passed `base_url`, rather
  than ignoring it silently.

### Provider Authentication

#### OpenAI

Required:

- `OPENAI_API_KEY`

Optional: `providers.openai.base_url`. Models and dimensions are selected in
`models.json`.

#### Anthropic (native Claude API)

Required for the `claude` completions engine:

- `ANTHROPIC_API_KEY`

The model is `completions.model` in `models.json` (default `claude-opus-5`).

Claude via Amazon Bedrock (the `anthropic` engine) does not use
`ANTHROPIC_API_KEY`; it authenticates with AWS credentials like the other
Bedrock engines below.

For current model IDs and pricing, use the Anthropic documentation:

- [Claude model overview](https://platform.claude.com/docs/en/about-claude/models/overview)
- [Claude API pricing](https://platform.claude.com/docs/en/pricing)

#### AWS Bedrock and Titan

Required:

- standard AWS credentials in environment or runtime IAM context
- a region: `AWS_REGION` in the environment, or `providers.aws.region` in
  `models.json` (default `us-east-1`)

Optional: `videos.bedrock_output_s3_uri` (Nova Reel). Models and dimensions
are selected in `models.json`.

If your AWS profile uses `aws login` (a `login_session` profile), install
`botocore[crt]` as well; botocore needs it to read those credentials.

##### DeepSeek, Qwen, and GLM on Bedrock

AWS hosts these models itself, so requests go to AWS under your AWS account
and never reach DeepSeek, Alibaba, or Z.ai.

```json
"completions": {"engine": "bedrock", "model": "deepseek.v3.2"}
```

```dotenv
AWS_PROFILE=<your profile>
```

| Model | Context | Tool calls and structured output | Notes |
| --- | ---: | --- | --- |
| `deepseek.v3.2` | 164K | yes | |
| `us.deepseek.r1-v1:0` | 128K | no | reasoning model; `strict_schema_prompt` extracts JSON from a plain reply |
| `qwen.qwen3-next-80b-a3b` | 256K | yes | reasoning |
| `qwen.qwen3-235b-a22b-2507-v1:0` | 256K | yes | us-east-2 and us-west-2 only |
| `qwen.qwen3-coder-next` | 256K | yes | coding |
| `qwen.qwen3-32b-v1:0` | 32K | yes | reasoning |
| `zai.glm-5` | 200K | yes | |
| `zai.glm-4.7` | 203K | yes | |
| `zai.glm-4.7-flash` | 203K | yes | lowest cost |

All are text-only, stream, and are priced in the registry. None supports
`count_tokens`. Enable each model under **Model access** in the Bedrock
console first. For a model outside your default region, set
`providers.aws.region` or pass `region=` to `AiBedrockCompletions`.

For current Bedrock model IDs, use the AWS documentation:

- [Supported foundation models in Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html)
- [Amazon Bedrock foundation model reference](https://docs.aws.amazon.com/bedrock/latest/userguide/foundation-models-reference.html)

#### Google-backed Providers

Required for the default OSS path:

- `GOOGLE_GEMINI_API_KEY`

Default auth mode: `providers.google.auth_method` set to `api_key`.

Optional service-account mode:

```json
"providers": {"google": {"auth_method": "service_account", "project_id": "<gcp-project-id>", "location": "us-central1"}}
```

```dotenv
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
```

This auth policy applies across Google completions, embeddings, images, videos, and voice.

##### Google auth modes per capability

The two `providers.google.auth_method` values select different google-genai clients:
`api_key` creates the Gemini Developer client
(`genai.Client(api_key=...)`, generativelanguage endpoint) and
`service_account` creates the Vertex client
(`genai.Client(vertexai=True, project=..., location=...)`). Some capabilities
work in only one client:

| Google capability | `api_key` (Developer) | `service_account` (Vertex) | Notes |
| --- | :---: | :---: | --- |
| Completions (incl. streaming) | ✅ | ✅ | |
| Text embeddings | ✅ | ✅ | |
| Image generation (Gemini image models) | ✅ | ✅ | `person_generation` is sent only in Vertex mode |
| Multimodal (media) embeddings | ✅ | ❌ | the SDK sends only text parts to the Vertex embedding endpoint, so the library raises `NotImplementedError` when media is attached in Vertex mode |
| Text-to-video with local download | ✅ | ❌ | downloaded via the Files API (`client.files.download`), which the SDK supports only in the Developer client; under Vertex set `download_outputs=False` to receive a remote `gs://` URI instead |
| `source_video` (video continuation) | ❌ | ✅ | the library raises `NotImplementedError` in `api_key` mode; this path requires Vertex |
| Voice (Gemini TTS / STT) | ⚠️ | ✅ | the library wires API keys into the Cloud TTS/STT clients. Google may reject a key not enabled for those APIs, so `service_account` is the reliable mode |

Pick the mode for what you need: multimodal-media embeddings and
text-to-video-with-download require `api_key`, and `source_video` requires
`service_account`. Running both families means switching `auth_method`
between runs or constructing two clients. For a single call, `get_client(...,
use_api_key=True)` on the shared Google base forces the Developer client
without changing the environment.

#### Azure TTS

Required:

- `MICROSOFT_COGNITIVE_SERVICES_API_KEY` in `.env`
- `providers.azure_speech.region` in `models.json`

Optional: `voice.language` in `models.json`.

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
- `AIFactory.get_ai_voice_client()` (or `AIVoiceFactory.create()`)

Concrete providers are not re-exported from package `__init__.py` modules. If you need a concrete class directly, import it from the module that defines it, for example:

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
- retired model requested (or deprecated model under `"strict_deprecations": true`): `AiProviderConfigurationError`

## Middleware

The `middleware` section of `models.json` configures middleware: a list of
entries, each with a `name` (`observability` or `pii_redaction`), `enabled`,
and `settings`. An entry that fails validation is skipped with a warning.
The [middleware schema](src/ai_api_unified/schemas/middleware.schema.json)
documents every setting, and `config_schema validate` checks the entries.

### Observability

Observability middleware is built into the base package and does not require a separate extra.

Features:

- metadata-only input, output, and error events
- request-scoped correlation via `set_observability_context(...)`
- coverage for completions, embeddings, images, videos, and text-to-speech

Minimal config:

```json
"middleware": [{"name": "observability", "enabled": true}]
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

Thinking tokens count as output, as every provider bills them. Gemini
reports them apart from the answer (`thoughts_token_count` beside
`candidates_token_count`), so its output count is the sum of the two; the
thinking share is also reported as `provider_reasoning_tokens` on the call's
events and `reasoning_tokens` on the request line item. Claude, OpenAI, and
Bedrock already include thinking in their output counts.

Prompt-cache reads are captured per provider (OpenAI, OpenAI Responses,
Anthropic, Bedrock, and Google Gemini) and billed at the model's cached-input
rate: the event carries `cached_input_tokens`, and the cached subset of
`input_tokens` is priced at the cached rate while the remainder bills at the
full input rate. Providers that report cache reads separately from the input
count (Anthropic, Bedrock) are normalized so `input_tokens` includes the cached
subset, keeping the cost split consistent across providers.

Prompt-cache **writes** are billed too. Priming a cache costs more than base
input, and the premium depends on how long the cache lives, so each rate is
stored per lifetime and each event carries its own count:

| Field | Meaning |
| --- | --- |
| `cache_write_5m_tokens` | Tokens written to a 5-minute-TTL cache (Anthropic: 1.25x base input) |
| `cache_write_1h_tokens` | Tokens written to a 1-hour-TTL cache (Anthropic: 2x base input) |

Unlike cache reads, writes are **not** a subset of `input_tokens` — the provider
reports them separately, so they add to the cost rather than re-slicing the
prompt. They are likewise excluded from the provider's own total token count, so
`usd_cost / provider_total_tokens` is not a valid effective rate on a
cache-priming call; include the cache-write counts in the denominator. Anthropic supplies the per-TTL split; Bedrock reports one
`cacheWriteInputTokens` that is attributed to the 5-minute tier, the only
lifetime it exposes. OpenAI charges nothing to write a cache, and Google bills
explicit-cache storage per hour rather than per written token, so both carry an
explicit zero write rate. The two states are recorded distinctly: a rate of
zero means the provider charges nothing to populate a cache, and writes bill as
free. An absent rate means the write is charged but not yet rated — it falls
back to the base input rate, under-reporting the premium rather than dropping
the cost. Five Bedrock entries are in that second state today: the hosted Claude
model and the four Amazon Nova models, all partner-priced with no
authoritative AWS cache-write rate sourced.

```json
"middleware": [
  {"name": "observability", "enabled": true,
   "settings": {"emit_cost": true}}
]
```

`emit_cost_topic` in `settings` overrides the cost logger's name.

Cost enrichment fires whenever `emit_cost` is on, even when output events are
disabled by `direction`. See [`docs/finops_middleware_design.md`](docs/finops_middleware_design.md).

Cost events also carry organization identity for billing attribution
(`org_id` / `org_name`), resolved per provider to what each platform
supports, cached per client, and failing open (cost events omit the fields
when identity is unavailable):

| Provider | `org_id` | `org_name` | How |
|---|---|---|---|
| `claude` (native Anthropic) | always | with `ANTHROPIC_ADMIN_KEY` | Admin API; else a free `count_tokens` response header |
| `openai` / `openai-responses` | always | always | account API (`/v1/me`) with the regular key; header-probe fallback |
| Bedrock-routed | always | with `iam:ListAccountAliases` | STS caller identity; IAM account alias |
| `google-gemini` | with `providers.google.project_id` | unsupported | configured GCP project id |
| `voyage`, `titan` | unsupported | unsupported | the platforms expose no caller identity |

Callers can also fetch the identity on demand from any client:

```python
info = client.get_org_info()   # AIProviderOrgInfoBase
print(info.org_id, info.org_name, info.source)
# source: "admin_api" (id + name) | "response_header" (id only) | "none"
```

Unlike enrichment, the explicit call raises `AiProviderRequestError` (with
`status_code`) when resolution fails — a rejected `ANTHROPIC_ADMIN_KEY`
surfaces as a 401. Providers subclass the returned model
(`AIProviderOrgInfoAnthropic`, `AIProviderOrgInfoOpenAI`, ...) for
provider-native fields, and `client.get_org_info_capability()` declares up
front what the configured engine can resolve:

```python
capability = client.get_org_info_capability()
capability.supports_org_id, capability.supports_org_name, capability.requirement
# e.g. on Bedrock: (True, True, "org_name is the AWS account alias and needs
#                    iam:ListAccountAliases; identity falls back to id-only without it")
```

#### Per-request cost

A request's total cost includes every call it made: classification, failed
attempts, fallbacks, and escalations. One routed structured-output request
can be four provider calls: the classifier, a 529 on the first tier, a
fallback on the same tier, and an escalation that answers. Each call still
emits its own `ai_api_call_cost` event; the request adds one total on top.

Every client keeps the report for the last request it ran as the outermost
call:

```python
result = client.send_structured_output(prompt, response_model=Summary)
report = client.last_request_cost          # AIRequestCostReport
report.total_usd                           # Decimal, every priced call
report.classifier_usd                      # the router's classifier call
report.wasted_usd                          # billed answers nobody used
report.billed_calls, report.failed_calls   # 3 and 1 in the example above
report.served_by                           # "claude:claude-opus-5"
for item in report.calls:                  # AIRequestCostLineItem, in call order
    item.role, item.route_tier, item.outcome, item.usd_cost, item.served
    item.pricing_source, item.pricing_effective_date   # the rates' URL and date
    item.reasoning_tokens                    # thinking share of output, when reported
report.service, report.tenant, report.purpose          # attribution, see below
```

Cost is attributable by service, tenant, purpose, difficulty level,
provider, and model:

| Dimension | Where it comes from |
| --- | --- |
| `service` | `service` at the top of `models.json`, stamped on every client the factory builds |
| `tenant` | the application's observability context: `set_observability_context(tags={"tenant": "acme"})` |
| `purpose` | a per-call hint, `provider_options={"purpose": "summarize"}`, removed before the request reaches the provider |
| difficulty level | `route_tier` on each line item and `served_by_tier` on the report |
| provider, model | `provider_engine` and `model` on each line item, `served_by` on the report |

All three tags also ride on every per-call event as `tag_service`,
`tag_tenant`, and `tag_purpose`, beside `tag_config_id`.

A failed attempt is a line item with no cost and no tokens; it still counts
as a call. A failed attempt the provider billed, a structured answer that
could not be parsed or validated (`StructuredOutputValidationError`) or a
truncated one (`StructuredResponseTokenLimitError`), keeps the usage the
exception carries, so the report prices it and counts it in `billed_calls`
and `wasted_usd`; its per-call cost event carries `outcome: "error"`. A model
with no registry pricing counts as billed but unpriced
(`unpriced_calls`), so `total_usd` understates the request by its cost.
The ledger records calls whether or not the observability middleware is on.

To roll several library calls into one logical request, such as a workflow
step, open the scope yourself. Nested scopes join the outer one, and the
clients inside keep no report of their own:

```python
from ai_api_unified import request_cost_scope

with request_cost_scope("summarize-contract") as scope:
    router.send_structured_output(prompt, response_model=Summary)
    router.send_prompt("One-line title for this summary.")
scope.report.total_usd      # both calls, every attempt behind them
```

When `emit_cost` is on, the outermost scope emits one `ai_api_request_cost`
event on the cost topic as it closes (normally or on an exception):
`request_id`, `operation`, `status`, `caller_id`, `session_id`,
`workflow_id`, `service`, `tenant`, `purpose`, `served_by`,
`served_by_tier`, `served_by_method`,
`total_usd`, `classifier_usd`, `wasted_usd`, `billed_calls`,
`failed_calls`, `unpriced_calls`, `elapsed_ms`, and `calls` (a compact list
with `call_id`, `role`, `model`, `outcome`, `input_tokens`, `output_tokens`,
`reasoning_tokens`, `usd_cost`, `pricing_source`, `pricing_effective_date`,
`served`). The
`request_id` is the router's `route_id` when routed, and every per-call
event under the request carries it as `tag_request_id`, so the per-call
events and the total join on one key.

### PII Redaction

PII redaction is optional and requires one of:

- `middleware-pii-redaction`
- `middleware-pii-redaction-small`
- `middleware-pii-redaction-large`

Minimal config:

```json
"middleware": [
  {"name": "pii_redaction", "enabled": true,
   "settings": {"direction": "input_only"}}
]
```

Notes:

- `strict_mode: true` enables fail-closed behavior
- `balanced`, `high_accuracy`, and `low_memory` detection profiles are supported
- install a matching spaCy model separately, for example `poetry run python -m spacy download en_core_web_sm` for `balanced` or `poetry run python -m spacy download en_core_web_lg` for `high_accuracy`
- `middleware-pii-redaction-small` and `middleware-pii-redaction-large` are compatibility aliases for the same Python dependency set; the spaCy model is still installed as a separate build/runtime asset
- for no-egress images and Lambda-style deployments, install the spaCy model wheel into the build artifact (runtime model downloads need egress)
- recognizer customization is configured in the entry's `settings`

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

On Bedrock, `strict_schema_prompt` picks a request each model accepts. By
default it pre-fills the reply with a JSON code fence and stops at the closing
fence. Claude 4.6 and later reject that pre-fill, so Opus 4.6 and the
open-weight models use native structured output, and the rest (Sonnet 4.6,
Claude 4.7 and later, DeepSeek R1) send a plain request and extract the JSON
from the reply. Claude 4.7 and later also get no `temperature`, which they
reject.

## Testing

Regular test run:

```bash
poetry run pytest -m "not nonmock"
```

Live provider tests:

```bash
poetry run pytest -m nonmock -s -vv
```

Tests never read your own `models.json`: the suite blanks `AI_MODELS_CONFIG`.
Live tests that need a models config read `tests/fixtures/models_test.json`,
whose hard tier lists a model its own deny list rejects so the policy tests
prove the deny list bites. To run them against any file:

```bash
poetry run pytest -m nonmock tests/test_model_policy_nonmock.py tests/test_configured_models_nonmock.py -q -s --models-config ./models.json
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

- An engine `must be configured explicitly` error: set that capability's `engine` in `models.json`.
- `AiProviderDependencyUnavailableError`: install the extra for the selected provider.
- Google auth errors: the OSS default is `providers.google.auth_method` set to `api_key`. If you switch to `service_account`, point `GOOGLE_APPLICATION_CREDENTIALS` at a valid local JSON credential file, and set `providers.google.project_id` and `providers.google.location`.
- Unexpected embeddings dimensions: leave `embeddings.dimensions` at `0` unless you intentionally want a non-default size.
- Google image generation or TTS failures in live tests often come from account or service state: a disabled cloud API, or missing paid-plan access. Check those before suspecting a library bug.
- `geo_residency: "US"` is a best-effort routing hint. Only providers that expose regional routing controls can honor it directly.

## License

This project is released under the MIT License.
