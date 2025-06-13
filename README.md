# ai_api_unified · Unified Foundation-Model Client Library

> **Version:** 0.1.0 &nbsp;|&nbsp; **License:** MIT

`ai_api_unified` provides a single, typed interface for calling both completion-style
LLMs and text-embedding models across vendors (OpenAI, Amazon Bedrock/Titan, …).

## Prerequisites

- **Python 3.12.1** (only)  
  We strongly recommend using [pyenv](https://github.com/pyenv/pyenv) to install and pin **exactly** 3.12.1, so that compiled wheels (e.g. `tiktoken`) are available and no Rust toolchain is required.

## Installation

```bash
# from your internal Artifactory PyPI
pip install --index-url https://<org>.jfrog.io/artifactory/api/pypi/pypi-local/simple ai_api_unified
```

Supported Python ≥ 3.9 (< 4.0).

## Quick start

```python
from ai_api_unified import AIFactory

# Completions
client = AIFactory.get_ai_completions_client()           # auto-selects engine via .env
response = client.send_prompt("Say hello in German")
print(response)  # → "Hallo!"

# Embeddings
embedder = AIFactory.get_ai_embedding_client()
vector = embedder.generate_embeddings("vectorize me")
```

---

## Repository layout

```
src/ai_api_unified/              ← package source
└── ai_base.py           ← abstract interfaces
└── ai_factory.py        ← runtime factory
tests/                   ← pytest suite
.env_template            ← sample environment config
```

---

## Development

```bash
# create virtualenv & install runtime + dev dependencies
poetry install --with dev
pytest -q
```

Linting and formatting:

```bash
ruff check .
ruff format .
```

---

## Publishing to Artifactory (Poetry workflow)

```bash
poetry version 0.1.0        # bump when ready
poetry build                # builds wheel + sdist
poetry config repositories.ups-ai https://<org>.jfrog.io/artifactory/api/pypi/pypi-local
poetry publish -r ups-ai
```

---

## Roadmap

- Add more provider back-ends (Anthropic, Google).
- Provide async variants for high-throughput workloads.
