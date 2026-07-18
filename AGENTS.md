# Repository guidelines

- TEST SELECTION POLICY: during development run only impacted tests via
  `poetry run python scripts/run_impacted_tests.py` (areas map in
  `tests/area_map.py`; new test files must be added there). Run the FULL
  mocked regression suite (`poetry run pytest -q -m "not nonmock"`) before
  committing load-bearing changes and always before a release — `publish.sh`
  enforces the release gate.
- Keep code formatted with `black` using the default settings.
- Use US English for all documentation and comments.
- When adding examples or new modules, place them under `src/ai_api_unified/`.
- Tests should live in the `tests/` directory and avoid network calls by mocking where possible.
- All code should include type hints for every variable, parameter, and return value.
