# ai_api_unified — agent instructions

## Versioning policy

This library uses semantic versioning. Every PR that changes shipped code
(anything under `src/`) must carry a version bump chosen by semver rules:

- **patch** (x.y.Z) — bug fixes, dependency pin updates, internal refactors
  with no API change
- **minor** (x.Y.0) — new features, new public API surface, new provider or
  capability support, deprecations
- **major** (X.0.0) — breaking changes: removed or renamed public API,
  changed method signatures or return shapes, dropped Python versions

Docs-only, test-only, and CI-only PRs do not bump the version.

### The version lives in exactly three files — bump them together, always

1. `pyproject.toml` — `version = "X.Y.Z"` under `[project]`
2. `src/ai_api_unified/__version__.py` — `__version__: str = "X.Y.Z"`
3. `README.md` — the title heading on line 1 (`# ai-api-unified X.Y.Z`)

Never bump one without the others. The README title is the historically
missed one (it sat at 2.5.0 through the 2.6.0 release). Verify agreement
before committing:

```bash
poetry run pytest tests/test_version_sync.py -q
```

`tests/test_version_sync.py` fails whenever the three locations disagree, so
the mocked suite catches a partial bump automatically.

Do not add new version literals beyond these three (scripts, docs, badges).
README's release section describes the process without hardcoding a version,
and `publish.sh` reads the version from `pyproject.toml` — keep it that way.
If a new version location ever becomes unavoidable, list it here and in the
README release section.

### Release flow

- The version bump lands in the PR that ships the change (repo convention —
  see PR #17, #19). If several PRs merge before a release, the highest
  required bump level since the last tag wins.
- Releases are cut on `main` after merge: tag `v<version>` and push the tag.
  Publishing to PyPI is a separate, explicit step via `./publish.sh`.

## Working conventions

- Never commit directly to `main`; create a branch from the updated remote
  primary branch first.
- Run the mocked suite before any commit: `poetry run pytest -q -m "not nonmock"`
  (tests in `*_nonmock.py` files call live provider APIs when credentials are
  present in `.env`).
- Lint and format: `poetry run ruff check .` and `poetry run black .`.
