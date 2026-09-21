# Contributing

Thanks for your interest in `ai-api-unified`. This library is in production
use, so the bar for a change is that it keeps working for the systems already
depending on it.

## Setup

```bash
poetry install --all-extras
```

Copy `env_template` to `.env` if you plan to run the live provider tests. The
mocked suite needs no credentials.

## Tests

The suite is large. During development, run only the areas your change
touches:

```bash
poetry run python scripts/run_impacted_tests.py
```

That maps your git diff to `area_*` pytest markers via `tests/area_map.py`.
You can also select by hand:

```bash
poetry run pytest -m "area_engine_openai and not nonmock"
```

Every new test file must be mapped in `tests/area_map.py`. Collection fails
with instructions if it is not.

Before opening a pull request, run the full mocked suite:

```bash
poetry run pytest -q -m "not nonmock"
```

Tests in `*_nonmock.py` call live provider APIs and cost money per run. They
execute only when credentials are present in `.env`, and CI never runs them.

## Lint and format

```bash
poetry run ruff check .
poetry run black .
```

CI runs both, plus the full mocked suite and the version-sync test, on Python
3.11, 3.12 and 3.13.

## Versioning

This library follows [semantic versioning](https://semver.org/). Any pull
request that changes shipped code under `src/` carries a version bump:

- **patch** — bug fixes, dependency updates, internal refactors with no API change
- **minor** — new features, new public API, new provider or capability support
- **major** — breaking changes: removed or renamed public API, changed
  signatures or return shapes, dropped Python versions

Docs-only, test-only, and CI-only pull requests do not bump the version.

The version lives in exactly three places and they move together:

1. `pyproject.toml` — `version` under `[project]`
2. `src/ai_api_unified/__version__.py`
3. `README.md` — the title on line 1

`tests/test_version_sync.py` fails when they disagree:

```bash
poetry run pytest tests/test_version_sync.py -q
```

Add a `CHANGELOG.md` entry for anything a consumer would want to gate on.

## Pull requests

- Branch from the updated remote primary branch; never commit to `main`.
- Keep a pull request to one concern.
- Say what you verified and what you could not.

## Releases

Releases are cut on `main` after merge: tag `v<version>` and push the tag.
Publishing to PyPI is a separate, explicit step via `./publish.sh`, which
re-runs the full mocked suite and refuses to publish on failure.
