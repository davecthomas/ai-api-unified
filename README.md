# ai-api

This repository contains a Python package that aims to provide a unified
interface for working with different foundation model providers.  The
included code defines abstract base classes and a factory for creating
client instances for completion and embedding APIs.

The repository already includes example implementations for both
OpenAI and Amazon Bedrock/Titan.  Environment configuration is handled
through a small `EnvSettings` class powered by Pydantic.

## Repository layout

```
src/ai_api/        - package source code
src/ai_api/ai_base.py      - abstract interfaces
src/ai_api/ai_factory.py   - factory for selecting client implementations
```

## Installing

This project uses a standard `pyproject.toml` and can be installed in
editable mode while developing:

```bash
pip install -e .
```

Copy `env_template` to `.env` and fill in your credentials before running the examples.

Running the unit tests requires `pytest`:

```bash
pytest
```

## TODO / things to fix

1. Expand the example completion and embedding clients.
2. Provide real client implementations for each provider.
3. Extend environment configuration (`EnvSettings`) as needed.
4. Extend the test suite to cover factory selection and client behaviour.
5. Ensure the package metadata and versioning follow your JFrog
   Artifactory publishing requirements.
6. Fill in the license information in `LICENSE`.

Once these items are completed the project can be built and uploaded to
Artifactory using `pip wheel .` followed by your organisation's upload
process.
