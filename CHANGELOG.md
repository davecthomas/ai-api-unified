# Changelog

Notable changes per release, so consumers can gate on the package version.
Versions follow [semantic versioning](https://semver.org/); the authoritative
version lives in `pyproject.toml` (see the README release section).

## 2.20.0

- Per-engine API base-URL overrides for `claude`, `openai`,
  `openai-responses`, and `google-gemini`, so provider traffic can route
  through an LLM gateway, an egress proxy, a recording proxy, or any
  OpenAI-compatible server: `ANTHROPIC_BASE_URL_OVERRIDE`,
  `OPENAI_BASE_URL_OVERRIDE`, `GOOGLE_GEMINI_BASE_URL_OVERRIDE`, or a
  `base_url` argument on the factory and engine constructors for per-client
  routing.
- Overrides must be https unless they target `localhost`, `127.0.0.1`, or
  `::1`; anything else raises `AiProviderConfigurationError` before a
  credential leaves the process. The resolved value is passed to each SDK
  explicitly, so the SDKs' own `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` /
  `GOOGLE_GEMINI_BASE_URL` variables cannot take effect unvalidated.
- Organization-identity lookups (2.18.0/2.19.0) now derive from the resolved
  base URL, so finops attribution follows the gateway instead of calling the
  vendor directly.
- Engines whose SDK cannot honor an override (Bedrock-routed, `titan`,
  `voyage`, voice) raise `AiProviderCapabilityUnsupportedError` when passed
  `base_url` rather than ignoring it.

## 2.19.0

- Organization identity for finops attribution now covers every provider to
  what its platform supports: `openai`/`openai-responses` resolve org id and
  name from the account API (`/v1/me`, regular key) with a response-header
  fallback; Bedrock-routed engines resolve the AWS account id via STS and
  the account alias as org_name when `iam:ListAccountAliases` permits;
  `google-gemini` attributes by the configured `GOOGLE_PROJECT_ID` (the
  Developer API exposes no caller identity). `voyage` and `titan` report
  none.
- New `client.get_org_info_capability()` returns
  `AIProviderOrgInfoCapability` (`supports_org_id`, `supports_org_name`,
  `requirement`) so consumers can introspect what the configured engine can
  resolve before calling. New sources: `account_api`, `configuration`.
- Org-identity caching (success cache; enrichment-only negative cache with
  on-demand retry) moved into the shared base, one implementation for all
  providers.

## 2.18.0

- Organization-level finops attribution, v1 on the `claude` engine: cost
  events carry `org_id` and `org_name`. With `ANTHROPIC_ADMIN_KEY` set (an
  Admin API key), both fields resolve from the Admin API; without it, the
  org id alone is captured from one free `count_tokens` response header.
  Resolution runs once per client, is cached, and fails open — cost events
  omit the fields when identity is unavailable. The call-context model gains
  `provider_org_id` / `provider_org_name`, and engines implement one
  resolver hook to supply identity (other providers report none yet).
- Public `client.get_org_info()` on every client returns
  `AIProviderOrgInfoBase` (`org_id`, `org_name`, `source`:
  `admin_api | response_header | none`; providers subclass it, v1
  `AIProviderOrgInfoAnthropic`). Unlike fail-open cost enrichment, the
  explicit call raises `AiProviderRequestError` with `status_code` when
  resolution fails, and retries after a failed background attempt.

## 2.17.0

- New `voyage` embeddings engine (extra: `voyage`, auth: `VOYAGE_API_KEY`)
  serving Voyage AI's models: `voyage-3` (default), `voyage-3-lite`,
  `voyage-3-large`, `voyage-code-3`, `voyage-finance-2`, `voyage-law-2` —
  with per-model dimensions, input-token limits, and registry pricing so
  cost events work like completions. Identical public surface to the other
  embeddings engines (same signatures and `{"embedding", "text",
  "dimensions"}` return shape); a consumer swaps providers by changing only
  the engine name. Batch calls chunk internally at Voyage's 128-text cap.
- Provider-neutral `input_type` retrieval hint ("query" | "document") added
  to `generate_embeddings` / `generate_embeddings_batch`; the `voyage`
  engine forwards it, other engines accept and ignore it.
- Async embeddings variants `agenerate_embeddings` /
  `agenerate_embeddings_batch`, gated by the new
  `AIEmbeddingsCapabilitiesBase.supports_async` flag (currently `voyage`).
- The `voyage` engine honors `retry_policy="none"` and wraps provider
  failures in `AiProviderRequestError` with `status_code`, matching the
  completions clients. Missing SDK raises the typed dependency error naming
  the `voyage` extra.

## 2.16.0

- Audio dependencies (`pydub`; `audioop-lts` on Python 3.13+) moved out of
  the base install into the `voice` extra. Text-only installs such as
  `ai-api-unified[anthropic]` no longer pull audio packages, and importing
  the library or constructing completions clients never triggers pydub's
  import (or its SyntaxWarning/ffmpeg RuntimeWarning noise). The `azure_tts`
  and `elevenlabs` extras include the audio dependencies; Google and OpenAI
  voice consumers install `[<provider>,voice]`. Voice features without the
  audio dependencies raise `AiProviderDependencyUnavailableError` naming the
  extra. Migration: add `voice` to your extras if you use Google or OpenAI
  voice/TTS/STT.

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
