# ai-api-unified

`ai-api-unified` is a unified Python library for AI completions, embeddings, image generation, and voice. Application code targets stable base interfaces and factory entry points while concrete providers are selected at runtime from environment configuration.

This release adopts a registry-backed lazy-loading architecture:

- provider SDKs are optional extras, not base dependencies
- providers are resolved only when a factory selects them
- package `__init__` modules export stable interfaces only
- missing provider selectors are configuration errors, not implicit fallbacks

Install name: `ai-api-unified`  
Import path: `ai_api_unified`

## Breaking Changes in 2.x

- Provider SDKs, including OpenAI, are no longer included in the base install.
- `COMPLETIONS_ENGINE`, `EMBEDDING_ENGINE`, `IMAGE_ENGINE`, and `AI_VOICE_ENGINE` must be set explicitly when that capability is used.
- Root and subpackage `__init__.py` modules no longer re-export optional concrete provider classes.
- Provider dispatch now goes through the lazy provider registry and loader instead of eager import branching.
- Google-backed providers now default to `GOOGLE_AUTH_METHOD=api_key`. Service-account auth is still available, but it is explicit.

## Capabilities

| Capability  | Stable interface    | Engines                                                                                                                       | Required extra(s)                                    |
| ----------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| Completions | `AIBaseCompletions` | `openai`, `google-gemini`, Bedrock-routed aliases such as `nova`, `anthropic`, `llama`, `mistral`, `cohere`, `ai21`, `rerank` | `openai`, `google_gemini`, `bedrock`                 |
| Embeddings  | `AIBaseEmbeddings`  | `openai`, `titan`, `google-gemini`                                                                                            | `openai`, `bedrock`, `google_gemini`                 |
| Images      | `AIBaseImages`      | `openai`, `google-gemini`, `nova-canvas` and Bedrock image aliases                                                            | `openai`, `google_gemini`, `bedrock`                 |
| Voice TTS   | `AIVoiceBase`       | `openai`, `google`, `azure`, `elevenlabs`                                                                                     | `openai`, `google_gemini`, `azure_tts`, `elevenlabs` |
| Voice STT   | `AIVoiceBase`       | provider-specific support such as Google and ElevenLabs                                                                       | `google_gemini`, `elevenlabs`                        |

Default model guidance in the checked-in OSS env files:

- Google completions: `gemini-2.5-flash`
- Google embeddings: `gemini-embedding-001`
- Google images: `imagen-4.0-generate-001`
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
poetry add 'ai-api-unified[bedrock,google_gemini]'
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
poetry install --all-extras --with dev
```

### Optional Extras

| Extra                            | Installs                                                               |
| -------------------------------- | ---------------------------------------------------------------------- |
| `openai`                         | OpenAI completions, embeddings, images, and voice                      |
| `google_gemini`                  | Google Gemini completions, embeddings, images, and Google voice        |
| `bedrock`                        | AWS Bedrock completions, Titan embeddings, and Bedrock image providers |
| `azure_tts`                      | Azure Cognitive Services TTS                                           |
| `elevenlabs`                     | ElevenLabs TTS and STT                                                 |
| `middleware-pii-redaction`       | Presidio + `usaddress` without packaged spaCy model wheels             |
| `middleware-pii-redaction-small` | Presidio + `usaddress` + packaged `en_core_web_sm`                     |
| `middleware-pii-redaction-large` | Presidio + `usaddress` + packaged `en_core_web_lg`                     |
| `similarity_score`               | NumPy-based similarity helpers                                         |
| `dev`                            | Optional dev dependencies from `[project.optional-dependencies]`       |

### Environment File

Copy [`env_template`](env_template) to `.env` and fill in only the providers you use.

The OSS template now defaults to Google API-key auth:

```dotenv
COMPLETIONS_ENGINE=google-gemini
EMBEDDING_ENGINE=google-gemini
IMAGE_ENGINE=google-gemini
AI_VOICE_ENGINE=google

GOOGLE_GEMINI_API_KEY=...
GOOGLE_AUTH_METHOD=api_key

COMPLETIONS_MODEL_NAME=gemini-2.5-flash
EMBEDDING_MODEL_NAME=gemini-embedding-001
IMAGE_MODEL_NAME=imagen-4.0-generate-001
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

### Embeddings

```python
from ai_api_unified import AIFactory, AIBaseEmbeddings

client: AIBaseEmbeddings = AIFactory.get_ai_embedding_client()
result: dict[str, object] = client.generate_embeddings("hello world")
embedding = result.get("embedding")
print(len(embedding) if embedding else None)
```

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
| `COMPLETIONS_ENGINE` | `openai`, `google-gemini`, Bedrock-routed aliases such as `nova`, `anthropic`, `llama`, `mistral`, `cohere`, `ai21`, `rerank` |
| `EMBEDDING_ENGINE`   | `openai`, `titan`, `google-gemini`                                                                                            |
| `IMAGE_ENGINE`       | `openai`, `google-gemini`, `nova-canvas`, `bedrock`, `nova`                                                                   |
| `AI_VOICE_ENGINE`    | `openai`, `google`, `azure`, `elevenlabs`                                                                                     |

### Common Model Settings

| Environment variable        | Notes                                                                                       |
| --------------------------- | ------------------------------------------------------------------------------------------- |
| `COMPLETIONS_MODEL_NAME`    | Optional completions model override                                                         |
| `EMBEDDING_MODEL_NAME`      | Optional embeddings model override                                                          |
| `IMAGE_MODEL_NAME`          | Optional image model override                                                               |
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
- `EMBEDDING_DIMENSIONS`
- `AI_API_GEO_RESIDENCY`

#### AWS Bedrock and Titan

Required:

- `AWS_REGION`
- standard AWS credentials in environment or runtime IAM context

Common optional settings:

- `COMPLETIONS_MODEL_NAME`
- `EMBEDDING_MODEL_NAME`
- `IMAGE_MODEL_NAME`
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

This auth policy applies across Google completions, embeddings, images, and voice.

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
    AIVoiceBase,
)
```

Factory entry points:

- `AIFactory.get_ai_completions_client()`
- `AIFactory.get_ai_embedding_client()`
- `AIFactory.get_ai_images_client()`
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

## Middleware

Middleware is configured by YAML referenced through `AI_MIDDLEWARE_CONFIG_PATH`.

### Observability

Observability middleware is built into the base package and does not require a separate extra.

Features:

- metadata-only input, output, and error events
- request-scoped correlation via `set_observability_context(...)`
- coverage for completions, embeddings, images, and text-to-speech

Minimal config:

```yaml
middleware:
  - name: 'observability'
    enabled: true
```

See [`docs/observability_middleware_example.yaml`](docs/observability_middleware_example.yaml) and [`docs/observability_middleware_design.md`](docs/observability_middleware_design.md).

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
3. Ensure the working tree is clean.
4. Ensure your PyPI token is configured for Poetry.

Publish with the checked-in script:

```bash
./publish.sh
```

The script:

- checks for uncommitted changes
- confirms the version
- removes old build artifacts
- runs `poetry publish --build`

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
