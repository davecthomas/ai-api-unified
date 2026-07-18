# Changelog

Notable changes per release, so consumers can gate on the package version.
Versions follow [semantic versioning](https://semver.org/); the authoritative
version lives in `pyproject.toml` (see the README release section).

## 2.15.0

The 2.14.0 capability-gated surface lands on every engine whose underlying
API supports it; the remaining gaps stay unimplemented and raise the typed
capability error.

- `openai` (Chat Completions) and `openai-responses`: full support —
  `send_conversation` tool loops (tools, forced `tool_choice`, strict
  functions), `send_structured_output` via the `json_schema` response format
  (schema-guided mode), async variants on a lazy `AsyncOpenAI`, extended
  `send_prompt` parameters, `retry_policy` (SDK `max_retries=0`), and
  status-coded `AiProviderRequestError`.
- `google-gemini`: full support — function-declaration tools with forced
  calling, raw-JSON-schema structured output via `response_json_schema`,
  async variants on `client.aio` (single attempt; pair with caller backoff),
  extended `send_prompt` parameters (per-request `http_options` timeout),
  `retry_policy` gating the engine backoff loop, and typed request errors.
  Gemini tool-call ids are the function name (the API carries no call ids).
- Bedrock-routed engines: partial per underlying API support —
  `send_conversation` via Converse `toolConfig` on Nova and Claude families,
  `send_structured_output` via Converse `outputConfig` only on models AWS
  lists (Claude 4.5+), `max_response_tokens` mapping, `retry_policy`
  collapsing the engine schedule, and status-coded errors from `ClientError`.
  Unimplemented (no underlying support): async variants (boto3 has no
  official async client) and per-call timeouts.
- New engine-agnostic replay helper `extend_messages_with_turn(messages,
  turn)` appends a model turn in each engine's wire shape, so one tool loop
  runs unchanged across engines (implemented on claude too).
- README gains a feature-support-by-engine matrix.

## 2.14.0

Engine-agnostic completions features for workflow-service call shapes, fully
implemented on the native `claude` engine and capability-gated elsewhere.
Feature support is declared on `client.capabilities`
(`supports_tool_use`, `supports_structured_output`, `supports_async`);
unsupported calls raise `AiProviderCapabilityUnsupportedError`.

- `send_structured_output` (and `asend_structured_output`): single-shot
  structured extraction with `system_prompt`, multi-turn `messages`, a raw
  JSON Schema `response_schema` alternative to pydantic `response_model`,
  `provider_options`, `request_timeout_seconds`, and `max_response_tokens` up
  to the context limit (the `claude` engine streams and accumulates large
  budgets internally). Results carry parsed `data`, token `usage`, and a
  normalized `finish_reason` (`complete | length | tool_use | refusal`) so
  callers distinguish truncation from refusal in code.
- `send_prompt` gains optional `system_prompt`, `max_response_tokens`, and
  `request_timeout_seconds`; omitting them leaves behavior unchanged.
- Tool-use conversations: `AITool`, `AIToolCall`, `AITokenUsage`,
  `AITurnResult`, `send_conversation` / `asend_conversation` (one turn per
  call; the caller owns the loop and executes tools), forced `tool_choice`,
  strict tools, replayable `raw_content`, and `build_tool_result_message`.
- Async variants (`asend_prompt`, `asend_structured_output`,
  `asend_conversation`) on engines whose SDK has an async client, starting
  with `claude` (lazy `AsyncAnthropic`).
- Retry policy: `retry_policy="none"` (constructor), `COMPLETIONS_RETRY_POLICY`
  (environment), or `provider_options={"retry_policy": "none"}` (per call)
  disables Anthropic SDK retries. HTTP failures raise
  `AiProviderRequestError` carrying `status_code` for uniform 429/5xx/529
  classification.
- Observability: `set_observability_context` accepts arbitrary string `tags`
  emitted as `tag_<name>` fields on every event, including cost-topic events.
  Token usage is available on every new result object without parsing logs.

Earlier releases predate this changelog; see git tags and the README feature
sections for their contents.
