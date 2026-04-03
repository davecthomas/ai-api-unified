# Observability Middleware Test Plan

## Summary

This test plan covers correctness, compatibility, and performance for the planned
observability middleware. The goal is to verify that the middleware emits the
right metadata with low overhead and without changing provider behavior.

## Test Objectives

- verify typed YAML parsing and normalization for `observability`
- verify input, output, and error events across every in-scope capability
- verify middleware timing events follow the established pattern
- verify observability remains metadata-only by default
- verify failures inside observability do not block AI calls
- verify ordering relative to transformation middleware, especially PII redaction
- verify overhead stays small enough for the feature to remain practical in the hot path

## Test Layers

### Unit Tests

Configuration tests:

- disabled when `settings` is missing
- disabled when `settings` is empty
- disabled when `enabled` is false
- normalized defaults for `direction`, `log_level`, `token_count_mode`, and capability allow-list
- invalid values fall back or disable according to the config model design

Context and summary builder tests:

- common fields are always present
- `call_id` is present on input, output, and error events
- `model_name` and `model_version` rules are consistent
- `originating_caller_id` and `originating_caller_id_source` are present only
  when explicit caller context is supplied
- the same explicit application caller id is preserved consistently across providers
- token count source is correctly labeled as `provider`, `estimated`, or `none`

Timing helper tests:

- `middleware_execution_timing` is emitted for observability input path
- `middleware_execution_timing` is emitted for observability output path
- elapsed time is a non-negative float

Fail-open tests:

- exceptions during log emission do not suppress the provider call result
- exceptions during output logging do not replace provider exceptions

### Capability Integration Tests

Completions:

- input event contains prompt metadata after input transformation
- output event contains output size metadata and provider elapsed time
- structured prompt path logs `response_mode=structured`
- media attachment metadata is emitted when present
- PII middleware ordering is preserved
- provider transport tests verify:
  - OpenAI completions send `safety_identifier`
  - Vertex completions send provider-safe `labels`
  - Bedrock Converse sends `requestMetadata`
  - Gemini direct completions do not attempt unsupported provider-side caller propagation

Embeddings:

- single embedding call emits correct input and output fields
- batch embedding call emits item counts and dimensions
- vectors are not present in logs
- provider transport tests verify:
  - OpenAI embeddings send `user`
  - Vertex and Gemini embed paths do not attempt unsupported provider-side caller propagation
  - Bedrock Titan embed paths do not attempt unsupported provider-side caller propagation

Images:

- image generation input event includes requested dimensions, count, format, quality, and background
- image generation output event includes image count and total output bytes
- negative prompt presence is logged when the provider surface supports it
- provider transport tests verify:
  - OpenAI images send `user`
  - Bedrock image generation does not attempt unsupported provider-side caller propagation

TTS:

- TTS input event includes text size, selected voice, locale, speaking rate, and format
- TTS output event includes audio byte count
- streaming path logs the final combined byte count
- provider transport tests verify no unsupported caller-propagation field is forced into OpenAI or Google TTS requests

### Error Path Tests

Completions:

- provider exception emits `ai_api_call_error`
- original exception still propagates

Embeddings:

- retryable and non-retryable failures still emit error metadata once per failed public call path

Images:

- invalid provider response shape still emits error metadata when the public method fails

TTS:

- vendor SDK failure still emits error metadata and preserves the original failure semantics

## Logging Assertions

Use `caplog` to validate:

- logger name
- event label
- required fields
- absence of raw payload content

Assertions should explicitly verify that logs do not contain:

- raw prompt text
- raw completion text
- embedding vectors
- base64 image payloads
- raw audio payloads
- raw API keys, bearer tokens, or unsalted credential hashes

## Recommended Test Files

Add focused tests rather than one large mixed file:

- `tests/test_observability_middleware_config.py`
- `tests/test_observability_middleware_completions.py`
- `tests/test_observability_middleware_embeddings.py`
- `tests/test_observability_middleware_images.py`
- `tests/test_observability_middleware_tts.py`
- `tests/test_observability_middleware_ordering.py`
- `tests/test_observability_middleware_performance.py`

This keeps failure isolation tight and mirrors the capability split in the design.

## Mocking Strategy

- mock provider SDK calls at the concrete provider boundary
- use fake responses that include and omit usage metadata to exercise both provider and estimated token count paths
- use fake logger handlers or `caplog` rather than real log sinks
- avoid network calls in all regular test suites
- avoid real credential material; use explicit fake caller ids for caller
  correlation tests

## Performance Tests

Add a manual-only or opt-in benchmark similar to the existing PII middleware
performance tests.

Benchmark goals:

- compare enabled vs disabled overhead for completions
- compare enabled vs disabled overhead for embeddings
- compare enabled vs disabled overhead for images
- compare enabled vs disabled overhead for TTS
- measure middleware execution time separately from provider latency

Suggested benchmark assertions:

- disabled mode overhead is negligible
- enabled mode adds only low single-digit milliseconds in mocked local scenarios
- no benchmark path decodes image or audio payloads by default

Benchmark gating recommendation:

- use an opt-in environment variable
- keep the benchmark out of default pytest runs

## Compatibility Tests

- direct provider instantiation should still emit logs when observability is enabled
- factory-created provider instances should emit the same logs
- completions with PII middleware enabled should preserve documented ordering
- completions without PII middleware enabled should still emit observability logs
- explicit caller-id handling should be deterministic for the same configured
  inputs and absent when no caller context is supplied

## Acceptance Criteria

The feature passes the test plan when:

- all in-scope capabilities emit input and output events with the expected fields
- error events are emitted without changing exception behavior
- middleware timing uses the established logger and event pattern
- raw payloads are absent from default logs
- ordering relative to transformation middleware is verified
- performance benchmarks show acceptable hot-path overhead

## Manual Verification Checklist

- enable `observability` in a middleware YAML profile
- run one completion call and confirm input and output events share the same `call_id`
- run one embedding call and confirm dimensions are logged but vectors are not
- run one image generation call and confirm requested image settings and output byte counts are logged
- run one TTS call and confirm voice metadata and audio byte counts are logged
- confirm `middleware_execution_timing` entries exist for observability input and output steps
