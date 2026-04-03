# Observability Middleware Implementation Plan

## Summary

This plan breaks the observability middleware work into small phases that fit the
current repository structure. The goal is to land a logger-backed,
configuration-driven middleware component without changing runtime behavior
outside the intended feature.

## Current Implementation Status

As of branch `AIAPI-1-docs-release`, the implementation status is:

- Phase A through Phase G are implemented in the active PR stack
- Phase H is the remaining cleanup branch for docs, versioning, and final release follow-through
- the shipped runtime now covers completions, embeddings, image generation, and text-to-speech
- targeted raw-content provider logging cleanup has landed in the touched Gemini, Bedrock, and Google TTS paths

## Scope

In scope:

- add typed middleware configuration for `observability`
- add a logger-backed observability middleware component
- reuse the established middleware timing pattern
- instrument completions, embeddings, images, and TTS provider call paths
- remove existing raw AI content logging from provider paths touched by this
  rollout
- add tests and end-user documentation for the new middleware

Client-facing execution model:

- applications continue calling the existing public AI methods
- middleware execution stays internal to the provider path
- observability does not add a new client-facing method or explicit middleware call

Out of scope for the first implementation:

- speech-to-text coverage
- external tracing backends
- raw payload logging
- sampling, aggregation, or asynchronous log shipping
- centralizing or replacing all non-observability logging in the library

## Phase Plan

### Phase A: Shared Contracts and Configuration

Deliverables:

- add `ObservabilitySettingsModel`
- add `get_observability_settings()` to `MiddlewareConfig`
- add middleware constants for `observability`
- extract or generalize shared middleware timing helpers so the observability component can emit `middleware_execution_timing` without inheriting a text-only result contract
- add a no-op observability middleware implementation for the disabled case

Targeted files:

- `src/ai_api_unified/middleware/middleware.py`
- `src/ai_api_unified/middleware/middleware_config.py`
- `src/ai_api_unified/middleware/__init__.py`
- new observability middleware module under `src/ai_api_unified/middleware/`

Exit criteria:

- middleware config can parse and type-check `observability.settings`
- no-op behavior is the default when the component is missing or disabled
- a shared timing utility is available for both PII and observability middleware

### Phase B: Shared Metadata and Wrapping Helpers

Deliverables:

- add a lightweight call-context model
- add a lightweight result-summary model
- add a request-scoped observability context API backed by `contextvars`
- add optional `originating_caller_id` support for explicit application-supplied
  caller identifiers
- add helper methods that wrap provider calls with:
  - input event emission
  - provider elapsed timing
  - output event emission
  - error event emission
- add token-count source semantics

Targeted files:

- `src/ai_api_unified/ai_base.py`
- new observability helper module under `src/ai_api_unified/middleware/`
- `src/ai_api_unified/voice/ai_voice_base.py`

Design guardrails:

- do not widen `AiApiMiddleware.process_input/process_output`
- do not introduce payload copies or response serialization in the wrapper
- keep wrappers usable when provider classes are instantiated directly
- keep emission synchronous and logger-backed inside the middleware, with no middleware-managed async queue
- keep middleware invocation inside shared base classes and concrete provider
  methods rather than introducing a new client-facing observability call path

Exit criteria:

- the wrapper logic can be reused across capabilities
- logs include stable common fields and `call_id`
- wrapper failures do not change provider exception behavior
- the context API preserves the same explicit caller id across providers when the
  application opts in

### Phase C: Completions Integration

Deliverables:

- instrument `send_prompt`
- instrument `strict_schema_prompt`
- log post-transformation input metadata and pre-transformation output metadata
- capture provider usage fields when available

Targeted files:

- `src/ai_api_unified/completions/ai_openai_completions.py`
- `src/ai_api_unified/completions/ai_google_gemini_completions.py`
- `src/ai_api_unified/completions/ai_bedrock_completions.py`
- `src/ai_api_unified/ai_base.py`

Important sequencing:

- preserve current PII middleware behavior
- ensure observability ordering matches the design doc

Exit criteria:

- completions calls emit input and output events when enabled
- logs contain prompt size, token count metadata, response size, and elapsed time

### Phase D: Embeddings Integration

Deliverables:

- instrument single and batch embedding calls
- capture requested dimensions and returned dimensions
- capture item counts and total input sizes

Targeted files:

- `src/ai_api_unified/embeddings/ai_openai_embeddings.py`
- `src/ai_api_unified/embeddings/ai_google_gemini_embeddings.py`
- `src/ai_api_unified/embeddings/ai_titan_embeddings.py`
- `src/ai_api_unified/ai_base.py`

Exit criteria:

- embedding calls emit metadata without logging vectors
- batch and single-call paths both covered

### Phase E: Images Integration

Deliverables:

- instrument image generation calls
- log prompt size and requested image properties on input
- log image count and total byte count on output

Targeted files:

- `src/ai_api_unified/images/ai_openai_images.py`
- `src/ai_api_unified/images/ai_bedrock_images.py`
- `src/ai_api_unified/ai_base.py`

Exit criteria:

- no output-byte decoding is required to emit the default fields
- logs remain metadata-only

### Phase F: TTS Integration

Deliverables:

- instrument text-to-speech calls across current voice providers
- cover both standard generation and streaming paths where implemented
- log text size, voice metadata, output format, and audio byte count

Targeted files:

- `src/ai_api_unified/voice/ai_voice_openai.py`
- `src/ai_api_unified/voice/ai_voice_google.py`
- `src/ai_api_unified/voice/ai_voice_azure.py`
- `src/ai_api_unified/voice/ai_voice_elevenlabs.py`
- `src/ai_api_unified/voice/ai_voice_base.py`

Exit criteria:

- TTS calls emit input and output events when enabled
- logs do not decode audio to estimate duration in the default path

### Phase G: Targeted Raw-Content Logging Cleanup

Deliverables:

- remove existing provider-path log statements that emit raw AI content such as:
  - prompt text
  - prompt previews
  - completion text
  - structured response bodies
  - parsed AI payload fragments
- replace those log statements with metadata-only alternatives such as:
  - model and provider identifiers
  - operation name
  - exception type
  - validation summary
  - text and byte counts where useful
- preserve the existing operational purpose of those logs without changing what
  observability middleware itself emits

Targeted files:

- `src/ai_api_unified/completions/ai_google_gemini_completions.py`
- `src/ai_api_unified/completions/ai_bedrock_completions.py`
- `src/ai_api_unified/voice/ai_voice_google.py`
- other provider files only if implementation work in this rollout uncovers
  additional raw-content log statements in touched paths

Exit criteria:

- no touched provider path logs prompt text, response text, structured payload
  bodies, vectors, image bytes, audio bytes, or prompt previews
- this cleanup does not expand or alter the observability middleware event schema
- existing provider logs remain useful for debugging through metadata-only fields

### Phase H: Documentation and Release Follow-Through

Deliverables:

- update `README.md` with observability middleware usage
- update `env_template` if any new environment defaults are introduced
- add or update diagrams as needed
- bump package version because behavior and external configuration surface change

Targeted files:

- `README.md`
- `env_template`
- `pyproject.toml`
- `src/ai_api_unified/__version__.py`

Exit criteria:

- end-user docs explain enablement, fields, and safety defaults
- versioning is aligned with repository policy

## Suggested PR Stack

The implementation should land as a dependency-ordered PR stack rather than one
large branch. The default planning threshold for a healthy PR is a `Core Logic Δ`
of `2000`, but this stack should still prefer narrowly scoped, reviewable
changes even when a larger PR might technically fit under that threshold.

| Stack Seq | Status | Branch Name | Branch Purpose | Core Logic Δ |
| --- | --- | --- | --- | ---: |
| `1` | `Open PR` | `AIAPI-1-config-contracts` | Add observability config parsing, middleware constants, no-op behavior, and shared timing extraction. | `625` |
| `2` | `Open PR` | `AIAPI-1-shared-runtime` | Add call context, result summary, request-scoped observability context, and shared wrapper helpers. | `1452` |
| `3` | `Open PR` | `AIAPI-1-completions` | Instrument completions providers and lock middleware ordering relative to PII transforms. | `1015` |
| `4` | `Open PR` | `AIAPI-1-embeddings` | Instrument embeddings providers for single and batch flows. | `965` |
| `5` | `Open PR` | `AIAPI-1-images` | Instrument image generation providers and standardize image metadata summaries. | `494` |
| `6` | `Open PR` | `AIAPI-1-tts` | Instrument TTS providers, including standard and streaming output paths where implemented. | `676` |
| `7` | `Open PR` | `AIAPI-1-log-cleanup` | Remove raw AI content from touched provider logs without changing observability middleware semantics. | `97` |
| `8` | `Current` | `AIAPI-1-docs-release` | Finalize docs, versioning, and release follow-through after behavior is stable. |  |

Recommended stack rules:

- keep Phase A and Phase B separate from capability wiring so schema and helper
  contracts stabilize before provider integration begins
- land completions before all other capabilities because it already has the
  tightest coupling to existing middleware ordering
- keep embeddings, images, and TTS in separate PRs even if helper reuse makes
  them look mechanically similar
- keep raw-content logging cleanup separate from observability semantics so the
  middleware event contract stays easy to review
- keep tests in the same PR as the behavior they validate instead of deferring
  them to the final release PR

## Detailed Work Breakdown

### Middleware Configuration Work

- add canonical middleware name constant: `observability`
- add direction reuse rather than inventing new input-output semantics
- add capability allow-list parsing
- add token-count mode parsing
- add config tests covering disabled, empty, invalid, and normalized values

### Provider Metadata Normalization Work

- standardize provider vendor names
- standardize engine names used in logs
- define how `model_name` and `model_version` are populated when only one identifier exists
- define a consistent `operation` vocabulary for each public API method
- define the `originating_caller_id_source` vocabulary for explicit caller
  context and `none`
- define provider-specific caller transport fields and normalization rules

### Log Schema Work

- freeze event names before implementation starts
- freeze shared field names before capability-specific work begins
- distinguish provider token counts from estimated token counts
- keep key names consistent across capabilities wherever the same concept exists
- freeze the provider-resolution matrix so every shared field has an explicit population rule per provider

### Caller Identity Work

- add a request-scoped observability context helper so applications can set a raw
  caller identifier, session id, or workflow id once per request flow
- validate and normalize the explicit caller identifier into
  `originating_caller_id`
- preserve compatibility with existing provider-specific caller hints such as
  `OPENAI_USER`, but route them through the same normalized caller-id path only
  when intentionally configured by the application
- add provider-safe shortening rules for transports that restrict label values

### Mandatory Provider Wiring

OpenAI:

- completions: when explicit caller context is present, propagate it through
  `safety_identifier`
- embeddings: when explicit caller context is present, propagate it through `user`
- images: when explicit caller context is present, propagate it through `user`
- TTS: log locally only because the current speech request schema does not expose
  a caller field

Gemini API direct:

- completions: local caller-id logging only when explicit caller context is present
- embeddings: local caller-id logging only when explicit caller context is present

Vertex AI:

- completions: when explicit caller context is present and a provider-safe label
  can be produced without surprising mutation, propagate it through request
  `labels`
- embeddings: local caller-id logging only because the reviewed embed request
  schema does not expose labels

Bedrock:

- completions via Converse: when explicit caller context is present, propagate it
  through `requestMetadata`
- embeddings via InvokeModel: local caller-id logging only
- images via InvokeModel: local caller-id logging only

### Performance Work

- use logger level guards before constructing expensive metadata
- avoid image and audio decoding in the default path
- avoid embedding-vector inspection beyond already-materialized size data
- benchmark observability overhead separately from provider latency

### Existing Logging Cleanup Work

- treat raw AI content in existing provider logs as a separate cleanup concern,
  not as part of the observability middleware contract
- remove prompt previews and raw response bodies from touched provider logs
- preserve debugging value by logging validation summaries, exception types,
  counts, and provider/model metadata instead of payload content

## Risks and Mitigations

- Risk: expanding the existing text middleware interface creates a brittle abstraction.
  - Mitigation: add a sibling lifecycle contract and keep text middleware unchanged.

- Risk: token count expectations are interpreted as exact even when estimated.
  - Mitigation: log token count source explicitly.

- Risk: provider classes drift into inconsistent field naming.
  - Mitigation: centralize shared event construction and vocabulary.

- Risk: observability ordering changes what is logged relative to PII middleware.
  - Mitigation: lock ordering in unit tests before broad rollout.

- Risk: TTS and image instrumentation add hidden hot-path cost.
  - Mitigation: log only already-available scalar metadata by default.

## Suggested Delivery Sequence

- start with Phase A and Phase B in the same branch
- complete completions integration first because it already participates in middleware
- land embeddings and images next because their metadata extraction is straightforward
- land TTS last because voice providers have the widest implementation variance
- update docs and version only after the code path is stable

## Definition of Done

The implementation is complete when:

- `observability` is parsed from the middleware YAML profile
- disabled configuration adds near-zero overhead through a no-op component
- every in-scope AI capability emits input and output events when enabled
- middleware timing uses the existing metrics pattern
- provider elapsed time is logged on output and error events
- tests cover config normalization, event fields, ordering, and fail-open behavior
- README and versioning updates are included in the implementation branch
