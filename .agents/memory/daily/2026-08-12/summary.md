# 2026-08-12 summary

## Snapshot

- Captured 1 memory event.
- Main work: Voice gained a factory constructor consistent with the other four capabilities, with AIVoiceFactory.create() now accepting an optional engine override and remaining supported as the delegate. Caller-id attribution moved from per-call-site arguments to a _resolve_legacy_caller_id() hook on AIVoiceBase that returns None by default and is overridden by the OpenAI provider to return its configured OPENAI_USER, so attribution is a provider property rather than something each call site must remember. The OpenAI voice provider now establishes its inherited AIOpenAIBase surface explicitly through PrivateAttr-backed properties, covering the env, api key, user, base URL, retry policy, and backoff schedule. A separate defect was fixed alongside: streaming text-to-speech called speech.create(stream=True), an argument no openai 2.x release accepts, and now goes through with_streaming_response. The OPENAI_USER setting key and its default were promoted to named constants shared with the embeddings provider.
- Top decision: Voice was the one capability callers could not reach through AIFactory; it required the separate AIVoiceFactory.create(), which took no engine argument and so offered no override path. Underneath that inconsistency sat a real defect: AIVoiceOpenAI inherits from both AIVoiceBase (a pydantic BaseModel) and AIOpenAIBase, and pydantic's BaseModel sits ahead of AIOpenAIBase in the MRO without chaining to super(). AIOpenAIBase.__init__ therefore never ran, so user, retry_policy, async_client, and the rest were never assigned. Observability attribution passed self.user explicitly at each call site to paper over this, which meant any provider that forgot the argument silently emitted calls with no originating caller id. The general lesson for future work in this package: a pydantic voice model that also inherits a vendor base class cannot rely on the vendor initializer running. ([2026-08-12 18:03:00 UTC by 2355287-davecthomas](events/2026-08-12T18-03-00Z--2355287-davecthomas--thread_bcd551a6-ba5b-4ac4-8fb6-6b3182463480--turn_c19e1adf4e.md))
- Blockers: Voice was the one capability callers could not reach through AIFactory; it required the separate AIVoiceFactory.create(), which took no engine argument and so offered no override path. Underneath that inconsistency sat a real defect: AIVoiceOpenAI inherits from both AIVoiceBase (a pydantic BaseModel) and AIOpenAIBase, and pydantic's BaseModel sits ahead of AIOpenAIBase in the MRO without chaining to super(). AIOpenAIBase.__init__ therefore never ran, so user, retry_policy, async_client, and the rest were never assigned. Observability attribution passed self.user explicitly at each call site to paper over this, which meant any provider that forgot the argument silently emitted calls with no originating caller id. The general lesson for future work in this package: a pydantic voice model that also inherits a vendor base class cannot rely on the vendor initializer running.

| Metric | Value |
|---|---|
| Memory events captured | 1 |
| Repo files changed | 1 |
| Decision candidates | 1 |
| Active blockers | 1 |

## Major work completed

- Voice gained a factory constructor consistent with the other four capabilities, with AIVoiceFactory.create() now accepting an optional engine override and remaining supported as the delegate. Caller-id attribution moved from per-call-site arguments to a _resolve_legacy_caller_id() hook on AIVoiceBase that returns None by default and is overridden by the OpenAI provider to return its configured OPENAI_USER, so attribution is a provider property rather than something each call site must remember. The OpenAI voice provider now establishes its inherited AIOpenAIBase surface explicitly through PrivateAttr-backed properties, covering the env, api key, user, base URL, retry policy, and backoff schedule. A separate defect was fixed alongside: streaming text-to-speech called speech.create(stream=True), an argument no openai 2.x release accepts, and now goes through with_streaming_response. The OPENAI_USER setting key and its default were promoted to named constants shared with the embeddings provider.

## Why this mattered

- Voice was the one capability callers could not reach through AIFactory; it required the separate AIVoiceFactory.create(), which took no engine argument and so offered no override path. Underneath that inconsistency sat a real defect: AIVoiceOpenAI inherits from both AIVoiceBase (a pydantic BaseModel) and AIOpenAIBase, and pydantic's BaseModel sits ahead of AIOpenAIBase in the MRO without chaining to super(). AIOpenAIBase.__init__ therefore never ran, so user, retry_policy, async_client, and the rest were never assigned. Observability attribution passed self.user explicitly at each call site to paper over this, which meant any provider that forgot the argument silently emitted calls with no originating caller id. The general lesson for future work in this package: a pydantic voice model that also inherits a vendor base class cannot rely on the vendor initializer running.

## Active blockers

- Voice was the one capability callers could not reach through AIFactory; it required the separate AIVoiceFactory.create(), which took no engine argument and so offered no override path. Underneath that inconsistency sat a real defect: AIVoiceOpenAI inherits from both AIVoiceBase (a pydantic BaseModel) and AIOpenAIBase, and pydantic's BaseModel sits ahead of AIOpenAIBase in the MRO without chaining to super(). AIOpenAIBase.__init__ therefore never ran, so user, retry_policy, async_client, and the rest were never assigned. Observability attribution passed self.user explicitly at each call site to paper over this, which meant any provider that forgot the argument silently emitted calls with no originating caller id. The general lesson for future work in this package: a pydantic voice model that also inherits a vendor base class cannot rely on the vendor initializer running.

## Decision candidates

- Voice was the one capability callers could not reach through AIFactory; it required the separate AIVoiceFactory.create(), which took no engine argument and so offered no override path. Underneath that inconsistency sat a real defect: AIVoiceOpenAI inherits from both AIVoiceBase (a pydantic BaseModel) and AIOpenAIBase, and pydantic's BaseModel sits ahead of AIOpenAIBase in the MRO without chaining to super(). AIOpenAIBase.__init__ therefore never ran, so user, retry_policy, async_client, and the rest were never assigned. Observability attribution passed self.user explicitly at each call site to paper over this, which meant any provider that forgot the argument silently emitted calls with no originating caller id. The general lesson for future work in this package: a pydantic voice model that also inherits a vendor base class cannot rely on the vendor initializer running. ([2026-08-12 18:03:00 UTC by 2355287-davecthomas](events/2026-08-12T18-03-00Z--2355287-davecthomas--thread_bcd551a6-ba5b-4ac4-8fb6-6b3182463480--turn_c19e1adf4e.md))

## Next likely steps

- Only the OpenAI voice engine resolves a vendor caller id; Google, Azure, and ElevenLabs return None from the hook, so their calls remain unattributed unless the observability context supplies a caller. Decide whether that gap should be closed per-vendor or documented as permanent.
- The same pydantic-plus-vendor-base MRO trap applies to any future voice provider that inherits a vendor base class; the AST guard test catches it, but the underlying inheritance shape is still the fragile part.
- Version 2.23.0 is bumped but unreleased on this branch; the full mocked suite is the release gate.

## Relevant event shards

- [2026-08-12 18:03:00 UTC by 2355287-davecthomas](events/2026-08-12T18-03-00Z--2355287-davecthomas--thread_bcd551a6-ba5b-4ac4-8fb6-6b3182463480--turn_c19e1adf4e.md)
