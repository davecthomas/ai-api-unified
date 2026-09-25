# Changelog

Notable changes per release, so consumers can gate on the package version.
Versions follow [semantic versioning](https://semver.org/); the authoritative
version lives in `pyproject.toml` (see the README release section).

## 2.29.0

### Added

- `openai-compatible` completions engine for any server that speaks the
  OpenAI Chat Completions protocol (vLLM, Ollama, LiteLLM, and hosted
  vendors with an OpenAI-compatible endpoint). It is configured with
  `OPENAI_COMPATIBLE_BASE_URL` (required), `COMPLETIONS_MODEL_NAME`
  (required), `OPENAI_COMPATIBLE_API_KEY` (optional),
  `OPENAI_COMPATIBLE_STRUCTURED_OUTPUT` (`json_schema` or `json_object`),
  and `OPENAI_COMPATIBLE_CONTEXT_WINDOW`, and it accepts the factory
  `base_url` argument.
- `AiOpenAICompatibleCompletions`, the base for vendor engines. A subclass
  sets class attributes for its API key and base-URL settings, default
  endpoint, model catalogue and context windows, image-input and reasoning
  models, structured-output mode, and pricing-registry label. The DeepSeek,
  Qwen, and Z.ai engines will build on it.

### Changed

- The compatible engine differs from `openai` where compatible servers
  differ: it sends `max_tokens` instead of `max_completion_tokens`,
  `strict_schema_prompt` uses `response_format` instead of the legacy
  `functions` parameter, `json_object` mode puts the schema in the system
  prompt, organization identity reports none, and observability events
  carry the vendor label instead of `openai`.
- Internal: the `openai` engine gained class attributes for the token field,
  registry label, and vendor display name, a `_build_capabilities` hook,
  and `AIOpenAIBase` gained `API_KEY_SETTING` and `_resolve_api_key`. The
  `openai` and `openai-responses` engines behave as before.

## 2.28.0

Model sweep, verified 2026-09-25 against the live models APIs of Anthropic,
OpenAI, and Google and against the AWS Bedrock model cards.

### Added

- Anthropic (`claude`): `claude-opus-5-5` ($4 / $20, cache reads $0.20) and
  `claude-fable-5-1` ($10 / $50, cache reads $0.25), both 1M context.
- OpenAI (`openai`, `openai-responses`): `gpt-6-astra`, `gpt-6-sol`,
  `gpt-6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, and `gpt-5.6-luna`, all with
  a 1.05M context window, reasoning capabilities, and a `context>272k`
  pricing tier (2x input, 1.5x output above 272K input tokens).
- OpenAI images: `gpt-image-2`, `gpt-image-2.5-flare`,
  `gpt-image-2.5-sunburst`, `gpt-image-1.5`, and `gpt-image-1-mini`.
- Google Gemini: `gemini-3.8-flash` and `gemini-3.7-flash`; Gemini images
  `gemini-3.1-flash-image`, `gemini-3.1-flash-lite-image`, and
  `gemini-3-pro-image`; Cloud TTS `gemini-3.1-flash-tts-preview`.
- Bedrock: `us.amazon.nova-2-lite-v1:0` and the Claude 5.x inference
  profiles `us.anthropic.claude-fable-5-1`, `us.anthropic.claude-opus-5-5`,
  `us.anthropic.claude-opus-5`, and `us.anthropic.claude-sonnet-5`.
- Voyage: `voyage-4-large`, `voyage-4`, `voyage-4-lite`, `voyage-code-4`,
  `voyage-3.5`, and `voyage-3.5-lite`, each accepting 256/512/1024/2048
  output dimensions.
- ElevenLabs: `eleven_v3` (now generally available) and `eleven_flash_v2_5`
  as selectable models.

### Changed

- Engine defaults move one generation behind the newest catalogued model:
  `claude` `claude-opus-4-8` -> `claude-opus-5`; `openai` `gpt-5.4-mini` ->
  `gpt-5.6-luna`; `google-gemini` `gemini-3.5-flash` -> `gemini-3.7-flash`.
  Embedding defaults are unchanged because vectors from different models
  cannot be compared.
- The Gemini images engine now generates through `generate_content` with the
  native Gemini image models, defaulting to `gemini-3.1-flash-image`. It
  previously called Imagen 4 through `generate_images`, and Imagen 4 was
  withdrawn from the Gemini API on 2026-08-17, so every call to the old
  default failed with a 404. Gemini image models return one image per
  request, so `num_images` issues one request per image. `person_generation`
  is sent only in Vertex AI mode because the Developer API rejects it.
- OpenAI images default to `gpt-image-2` (was `gpt-image-1`, which shuts
  down 2026-10-23).
- OpenAI speech-to-text uses `gpt-transcribe` instead of `whisper-1`, which
  shuts down 2027-02-26.
- `claude-sonnet-5` is priced at $2 / $10. The launch rate became the
  standard price after Anthropic cancelled the scheduled increase to $3 / $15.
- Bedrock Nova v1 context windows now match the model cards (128K Micro,
  300K Lite and Pro, 1M Premier, 200K for Claude 3.5 Haiku). The previous
  values were about 32 times too large, so the context guard never fired.
- `send_conversation(..., tool_choice=...)` raises `ValueError` before any
  request on `claude-opus-5-5` and `claude-fable-5-1` (native and Bedrock).
  Both models return a 400 for any forced tool choice.
- The image and video engines now apply the registry lifecycle policy at
  construction, as the completions and embeddings engines already did.

### Retired and deprecated

- Retired, so construction raises `AiProviderConfigurationError`: OpenAI
  `sora-2` and `sora-2-pro` (the Videos API shut down 2026-09-24 with no
  replacement), `dall-e-2`, `dall-e-3`; Google `imagen-4.0-*` (2026-08-17),
  `veo-3.0-generate-001`, `veo-3.0-fast-generate-001`,
  `veo-2.0-generate-001` (2026-06-30), and the `gemini-3-pro-image-preview`
  and `gemini-3.1-flash-image-preview` previews (2026-06-25). The OpenAI
  video engine remains registered so existing configurations get that
  explanation instead of an opaque 404.
- Deprecated, warning once per process: `o4-mini`, `gpt-4.1-nano`, and
  `gpt-image-1` (shutdown 2026-10-23); `gpt-image-1-mini` and
  `gpt-image-1.5` (2026-12-01); `gemini-2.5-flash-image` (2026-10-02).

## 2.27.0

### Removed

- `ai_api_unified.middleware.impl.middleware_extensibility_poc` is gone, along
  with its test. It was a 372-line feasibility spike for hard-wired SSN
  last-4 detection, dead since the PII redaction middleware shipped: nothing
  under `src/` imported it, no `__init__` exported it, and only its own test
  loaded it. Its own docstring said it was "not wired into the production
  middleware flow" while it shipped in every wheel. Production SSN redaction
  is unaffected and lives in the `middleware-pii-redaction` extra.

  This is a minor rather than a major bump: the module sat under
  `middleware/impl/`, outside the public surface the README documents, which
  is the stable base interfaces and the factories. Anyone importing it
  directly was reaching past that boundary into a module labelled a
  proof of concept.

### Changed


- Packaging metadata now states maturity and provenance: a
  `Development Status :: 5 - Production/Stable` classifier, audience and topic
  classifiers, and a `[project.urls]` table pointing at the repository, issues,
  and changelog. The PyPI page previously carried no status and no links.
- Added a GitHub Actions CI workflow. It runs the full mocked suite and the
  version-sync test on Python 3.11, 3.12 and 3.13, plus ruff and black. The
  repository had no automated checks before this.
- Added `CONTRIBUTING.md` and `SECURITY.md`.
- 18 tests in `test_google_gemini_nonmock.py` and `test_model_switch_nonmock.py`
  now carry the `nonmock` marker. They call live provider APIs, but nothing
  excluded them from `-m "not nonmock"` runs except an absent API key, so on
  a machine holding credentials they ran inside the mocked suite and billed
  real calls on every run. The 38 tests in `test_pii_redactor_nonmock.py` stay
  unmarked deliberately: Presidio runs locally, they reach no provider, and
  they belong in the mocked suite.
- `tests/conftest.py` no longer forces one developer's personal AWS SSO
  profile onto every run. It hardcoded that profile name and set
  `AWS_PROFILE` to it whenever the variable was unset, so the mocked suite
  raised `ProfileNotFound` on any machine without it, a fresh clone
  included. The profile is now used only when botocore can see it, with
  placeholder credentials as the fallback, and an explicitly chosen
  `AWS_PROFILE` still wins.
- Configured `per-file-ignores` for `E402` under `tests/`, where
  `pytest.importorskip` for an optional extra must precede the imports it
  guards. Those seven errors were the only thing standing between the existing
  lint command and a green CI job.
- Renamed `docs/middleware-extensibility-pattern-pii-poc.md` to drop `-poc`,
  marked it as delivered design history, and reworded the Titan
  `generate_embeddings_batch` docstring, which described the library itself as
  a POC. The PII redaction middleware it plans ships today as the
  `middleware-pii-redaction` extra.

Beyond the removal above, no functional change: no behavior of any engine,
factory, or middleware changed in this release.

## 2.26.1

- The `google_gemini` extra no longer states a protobuf range. It pinned
  `protobuf>=3.20.2,<5.0.0dev`, which made the extra uninstallable in any
  application already on protobuf 5 or later: pip reported
  `ResolutionImpossible`. protobuf is required here only transitively, by
  `google-api-core`, `google-cloud-speech`, `google-cloud-texttospeech` and
  `proto-plus`, each of which enforces its own range. This library imports
  protobuf nowhere, so any range it declared could only conflict with theirs.
- The `googleapis-common-protos` entry is gone for the same reason. It was
  never imported here, `google-api-core` and `grpcio-status` both require it,
  and the stated floor of `>=1.63.0` sat below the `>=1.69.2` that
  `google-api-core` actually requires.
- Installs are verified against protobuf 4.25.8, 5.29.5, 6.33.6 and 7.36.1.
- The lock moved across the protobuf and gRPC chain: protobuf 4.25.8 to 6.33.6
  and `grpcio-status` 1.62.3 to 1.84.0, which had been held 19 minor versions
  behind `grpcio` by the old ceiling, since 1.63 and later require protobuf
  5.26 or newer.

## 2.26.0

- Every completions engine now honors the `provider_options` contract in
  `AIBaseCompletions`: "engines ignore keys they do not understand." None of
  them did. Each forwarded the caller's keys to its SDK, which failed inside
  the caller's process before any request was sent — Gemini into
  `GenerateContentConfig`, which forbids extra fields; anthropic, openai, and
  openai-responses as keyword arguments to a `create()` that declares no
  `**kwargs`; bedrock into botocore's client-side parameter validator. That
  broke the cross-provider fallback `provider_options` exists to serve: a
  caller tuned for one engine could not keep its options when failing over.
- `_split_provider_options` now filters merge keys through a new
  `_known_provider_option_keys` hook before returning them. Dropped keys are
  logged at warning level, so an ignored option stays discoverable.
- Each engine derives its accepted keys from its SDK rather than a hardcoded
  list: Gemini from `GenerateContentConfig`'s pydantic fields and their
  camelCase aliases, anthropic and the two openai engines from the `create()`
  signature, bedrock from the `bedrock-runtime` Converse input shape. An SDK
  that adds an option keeps working without an edit here, the lesson of the
  2.25.2 block-list regression.
- An engine whose SDK cannot be introspected forwards every key unchanged,
  and logs why. Dropping a caller's option on a guess is worse than passing it
  through and letting the SDK speak.
- The signature walk and its per-class cache live once on `AIBaseCompletions`.
  An engine whose options are keyword arguments implements only
  `_sdk_option_method`, handing back the unbound SDK method and a log label.
  The cache is keyed on the owning class through `__dict__`, so
  `AiOpenAIResponsesCompletions`, which subclasses `AiOpenAICompletions`,
  resolves its own options rather than inheriting the parent's.
- The reserved `retry_policy` key still splits out ahead of the filter.
- Minor rather than patch: an option a caller passes today is now dropped and
  logged instead of raising, which is a behavior change on a public argument.

## 2.25.3

- The bedrock engine now reads the Converse `ContentBlock` members from the
  installed `bedrock-runtime` service model instead of a hardcoded list.
  2.25.2 froze the 12 members present in botocore 1.43.40; botocore 1.43.80
  added `toolAddition` and `toolRemoval`, so on a current boto3 the engine
  raised `ValueError` for content blocks Converse accepts — the same
  in-process rejection 2.25.2 set out to remove. The member set resolves once
  per process and falls back to the built-in list when the model cannot be
  read.
- `tests/test_multi_engine_conversation_api.py` asserted the hardcoded list
  equalled the service model, so the mocked suite went red on any boto3
  upgrade rather than only when the library was wrong. It now checks that the
  resolved set matches the installed model, that the fallback names only real
  members, that a member a newer SDK adds passes through, and that an
  unreadable model falls back rather than breaking the send path.

## 2.25.2

- The bedrock engine now accepts the same documented `{role, content}` messages
  shape 2.25.1 fixed on google-gemini. It forwarded the caller's list straight
  to the Converse API, which requires `content` as a list of blocks, so botocore
  rejected a bare string client-side before the request left the process. This
  is the same defect as the Gemini one, on a second engine.
- `_normalize_messages` on the bedrock engine wraps string content as
  `[{"text": ...}]`. Converse already names the assistant role `assistant`, so
  only the content wrapping differs. Content that is already a list of Converse
  blocks passes through untouched, keeping a history built from
  `extend_messages_with_turn` and `build_tool_result_message` valid, and
  anything else raises a `ValueError` naming both helpers. The block test is
  that every key is a Converse `ContentBlock` member, not that some key is: an
  Anthropic block is `{"type": "text", "text": ...}`, which carries `text` but
  is rejected by botocore on the unknown `type` key, so a looser test would
  forward another engine's history into the failure this fix removes.
- All five completions engines are now covered by
  `TestDocumentedShapeAcrossProviders`, which sends the documented shape
  through each engine and validates what reaches the SDK against that
  provider's own client-side validator where one ships — the google-genai
  pydantic request model and the botocore Converse parameter validator. An
  engine whose wire shape diverges from the documented contract fails there
  rather than in a caller's process.
- Audited and unchanged: anthropic, openai, and openai-responses all accept
  string content natively, so they keep the identity default.

## 2.25.1

- The google-gemini engine now accepts the `{role, content}` messages shape
  that `AIBaseCompletions` documents and calls provider-neutral. It previously
  forwarded the caller's list straight to google-genai, which requires
  `{role, parts: [...]}`, so the call died on a pydantic `ValidationError`
  inside the caller's process before reaching the network. Cross-provider
  fallback could not work: the same caller code could drive Claude but not
  Gemini.
- The translation lives in a new `_normalize_messages` hook on
  `AIBaseCompletions`, which returns the caller's list unchanged. Every
  conversation and structured-output surface routes through it —
  `send_conversation`, `asend_conversation`, `send_structured_output`, and
  `asend_structured_output` — so an engine whose wire shape differs overrides
  one method and all four surfaces follow. Engines whose shape is already
  `{role, content}` keep the default and are unaffected.
- Gemini's override maps `content` to `parts` and renames the `assistant` role
  to `model`. Entries already carrying `parts` pass through untouched, so a
  history mixing caller-written messages with `extend_messages_with_turn` and
  `build_tool_result_message` output stays valid. Everything else raises a
  `ValueError` naming both helpers: an entry with neither string `content` nor
  `parts` belongs to another engine, and forwarding it would reach google-genai
  and raise the same in-process `ValidationError` this fix exists to prevent.
- `asend_prompt` and `send_prompt` take a string and were never affected.

## 2.25.0

- `list_model_names` on the google-gemini completions engine now checks the
  static model catalogue against the provider's live `models.list` before
  answering. The catalogue differs per auth path (Gemini API vs Vertex),
  project, and region, so the hardcoded `GEMINI_MODEL_SPECS` list could name
  models the current credentials cannot call. Observed live: the Gemini API
  no longer lists the 2.0-family spec entries, and a Vertex project answered
  404 for a spec entry the static list presented as callable.
- How much that check verifies depends on the auth path, and the docstring
  now says so. The Gemini API publishes `supported_actions`, so entries that
  cannot `generateContent` are dropped. Vertex publishes none — the SDK's
  Vertex converter does not map the field — and its publisher catalogue is
  not scoped to `GOOGLE_LOCATION`, so there this is a name-presence check and
  a globally-listed model can still answer 404 in the configured region.
- The configured model is always listed, so `model_name` never goes missing
  from its own engine's list: the engine sends every request against
  `model_name`, so a list that omitted it would contradict the engine's own
  configuration.
- **Reading `list_model_names` on this engine can now make a blocking network
  call**, where every other engine returns a static literal. Callers should
  read it off the event loop, as the HTTP service already does. Both outcomes
  are cached per client instance and keyed on the configured model, so the
  cost is one round trip per TTL window (15 minutes on success, 1 minute
  after a failure) rather than one per read. A listing that succeeds but
  names none of the spec entries is a stable mismatch rather than a transient
  fault, so it takes the full window instead of re-querying every minute. A
  transient failure retries once, which absorbs a rate-limit blip without the
  multi-second sleep the completions retry budget would spend on a call that
  has an instant static fallback.
- When the listing call fails (offline, restricted credentials, mocked SDK),
  the property returns the full static list unchanged and logs the reason.
  That outcome is cached only for the short failure window, so a recovered
  provider is picked up quickly while a provider that cannot answer at all
  stops costing a round trip per read. A listing that succeeds but names none
  of the spec entries also serves the static list, under the full window as
  described above.
- **The Gemini 2.0 family is retired**: `gemini-2.0-flash`,
  `gemini-2.0-flash-001`, `gemini-2.0-flash-lite`, and
  `gemini-2.0-flash-lite-001` are removed from `GEMINI_MODEL_SPECS` and move
  from DEPRECATED to RETIRED in the pricing registry. Probed 2026-08-26:
  `models.list` no longer names any of them and `generateContent` answers 404
  for each, so their scheduled 2026-06-01 sunset has passed in fact. This
  matters most for the static list, which is what callers are served when the
  live catalogue cannot be reached — a dead entry there would be advertised
  as callable on the one path that cannot check it.
- Constructing a client on a retired model now fails instead of warning and
  quietly falling back to the default model, so a pinned
  `COMPLETIONS_MODEL_NAME=gemini-2.0-flash` stops billing a different model
  than the one requested. Constructing the class directly raises
  `AiProviderConfigurationError` naming the replacement
  (`Model 'gemini-2.0-flash' (google) is retired; withdrawn on 2026-06-01;
  use 'gemini-2.5-flash' instead.`). Going through
  `AIFactory.get_ai_completions_client`, that message is currently replaced
  with `Unsupported COMPLETIONS engine`, which is a pre-existing masking bug
  in `_translate_config_exception` that also hid the 1.5 retirements; it is
  tracked separately and not changed here, since it affects every engine and
  capability. The call still fails fast either way.
- No deprecated completions model remains catalogued, so the client-level
  lifecycle test now covers the retired branch, and the deprecated branch
  stays covered at the registry level in `test_model_pricing.py`.
- `AIGoogleBase.list_models` gains optional `required_action`,
  `bool_strip_resource_prefix`, and `bool_propagate_errors` parameters, so the
  completions engine reuses that pager instead of carrying a second copy.
  Default behavior is unchanged for existing callers.

## 2.24.0

- Prompt-cache **writes** are now priced and billed. Priming a cache costs more
  than base input, so cost events previously under-reported cache-heavy
  workloads (a call writing 20k tokens to cache under-reported by ~67%).
- `AITokenRates` gains `cache_write_5m_per_1m` and `cache_write_1h_per_1m`. The
  premium depends on cache lifetime, so rates are stored per TTL rather than
  blended. Populated for all 9 Anthropic models at the documented 1.25x (5m)
  and 2x (1h) of base input.
- `compute_token_cost()` accepts `cache_write_5m_tokens` and
  `cache_write_1h_tokens`. Cache writes add to the cost rather than being
  carved out of `input_tokens` — unlike cache reads, providers report them
  separately from the prompt count. An unset rate falls back to the base input
  rate rather than to free.
- Observability result summaries carry `provider_cache_write_5m_tokens` and
  `provider_cache_write_1h_tokens`, extracted from Anthropic's per-TTL
  `usage.cache_creation` split (falling back to the aggregate
  `cache_creation_input_tokens` as 5-minute) and Bedrock's
  `cacheWriteInputTokens`. Cost events emit `cache_write_5m_tokens` and
  `cache_write_1h_tokens`.
- A call that reports only cache writes is now costed instead of being skipped
  as no-usage.
- `compute_completion_cost()` (the public real-cost API) accepts the same
  cache-write arguments, and `AITokenUsage` on `AITurnResult` /
  `AIStructuredOutputResult` carries `cache_write_5m_tokens` /
  `cache_write_1h_tokens`, so result objects agree with the cost stream.
- An unknown future cache TTL tier reconciles against the aggregate write count
  and bills at the 5-minute rate instead of billing as free.
- Free-by-design and not-yet-rated are recorded distinctly. OpenAI and Google
  carry an explicit zero cache-write rate (OpenAI writes are free; Google bills
  explicit-cache storage per hour rather than per written token), so their
  writes bill as free. An absent rate means charged-but-unrated and falls back
  to the base input rate; the five Bedrock completions entries (hosted Claude
  plus the four Nova models) are in that state, pending an authoritative AWS
  cache-write rate.

## 2.23.0

- Fixed OpenAI text-to-speech: every `text_to_voice` and `stream_audio` call
  raised `AttributeError` because pydantic's MRO skipped
  `AIOpenAIBase.__init__`. The voice constructor now invokes that shared
  initializer, so the inherited surface (`async_client`, organization lookup,
  `OPENAI_USER` caller attribution) works.
- Fixed OpenAI `stream_audio` passing an unsupported `stream=True` argument;
  streaming now uses the SDK's `with_streaming_response` API.
- Fixed a latent `AttributeError` in OpenAI `speech_to_text` rate-limit
  backoff (`time.sleep` called on the `time` function).
- Added `AIFactory.get_ai_voice_client(voice_engine=None, base_url=None,
  retry_policy=None)`; `AIVoiceFactory.create()` gains the same optional
  arguments and stays callable with none. The `base_url` and `retry_policy`
  arguments apply to the `openai` engine and are rejected with
  `AiProviderCapabilityUnsupportedError` on engines that cannot honor them.
- Voice caller attribution now flows through
  `AIVoiceBase._resolve_legacy_caller_id()`: OpenAI attributes to
  `OPENAI_USER` (falling back to `default_user`), other voice engines stay
  unattributed unless the application sets an observability context.
- A present-but-blank `COMPLETIONS_RETRY_POLICY` or `OPENAI_USER` is now
  treated as unconfigured across all engines instead of raising or silently
  dropping attribution; explicit blank constructor arguments still raise.
- Fixed `get_default_voice()` dropping `language`, `locale`, `accent`, and
  `gender` from the returned selection, which mislabeled synthesis language
  on Azure and Google.

## 2.22.0

- `claude-opus-4-1` is now RETIRED (Anthropic withdrew it 2026-08-05) with
  `claude-opus-5` as its replacement. Requesting it raises
  `AiProviderConfigurationError` at construction instead of emitting a
  deprecation warning. Callers still on it must switch models.
- Retired registry entries keep the pricing they carried while active, so
  cost enrichment can still price usage recorded before the withdrawal
  date. Lifecycle enforcement is independent of whether rates are present.
- Lifecycle messages now read a sunset date as history for retired models
  ("withdrawn on <date>") and as a schedule for deprecated ones
  ("scheduled for withdrawal on <date>").

## 2.21.0

- Catalogue the latest models served by all three major completions
  providers (verified against each provider's live models API on
  2026-08-03), with registry pricing and capability entries:
  - Anthropic: `claude-opus-5` and `claude-sonnet-5` (both 1M context;
    Sonnet 5 priced at list rates, introductory pricing noted through
    2026-08-31). `claude-opus-4-1`'s recommended replacement is now
    `claude-opus-5`.
  - OpenAI: `gpt-5.5`, `gpt-5.4-mini`, `gpt-5.4-nano`, and `gpt-5.2` are now
    in the model list and context-window table (they were previously priced
    in the registry but not selectable).
  - Google Gemini: the 3.x generation — `gemini-3.6-flash`,
    `gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`,
    and `gemini-3.1-pro-preview` (tiered >200K pricing) — with a
    reasoning-capable Gemini 3 capabilities branch.
- Engine defaults move to one generation behind the newest catalogued
  model: OpenAI `gpt-4o-mini` -> `gpt-5.4-mini`, Gemini
  `gemini-2.5-flash` -> `gemini-3.5-flash` (default and unknown-model
  fallback). The Claude default stays `claude-opus-4-8`, already one
  generation behind `claude-opus-5`. `env_template` now lists per-engine
  model choices including the new generation.
- Fix: `AiOpenAICompletions()` constructed with no arguments previously
  used the literal `"4o-mini"` (an invalid model ID) and never consulted
  `COMPLETIONS_MODEL_NAME`; the signature default is now empty so the
  environment setting and the `gpt-5.4-mini` fallback apply.

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
  vendor directly. Exception: the Anthropic Admin API key grants org-wide
  read/write, so it does not follow `ANTHROPIC_BASE_URL_OVERRIDE`; set
  `ANTHROPIC_ADMIN_BASE_URL_OVERRIDE` to route that lookup too.
- The deprecated `OPENAI_BASE_URL` is now validated by the same https rules,
  closing a path where a process-wide value set by other tooling could send
  the API key to a plaintext host.
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
