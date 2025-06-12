# Repository guidelines

- Run `pytest` before committing changes to ensure the test suite passes.
- Keep code formatted with `black` using the default settings.
- Use US English for all documentation and comments.
- When adding examples or new modules, place them under `src/ai_api/`.
- Tests should live in the `tests/` directory and avoid network calls by mocking where possible.
- All code should include type hints for every variable, parameter, and return value.
