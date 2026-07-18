# 2026-07-18 summary

## Snapshot

- Captured 1 memory event.
- Main work: OpenAI Chat Completions and Responses engines gained full support: send_conversation tool loops with forced tool_choice and strict functions, json_schema structured output, async variants on a lazy AsyncOpenAI, extended send_prompt parameters, retry_policy, and status-coded AiProviderRequestError. Google Gemini gained the same full surface via function-declaration tools, response_json_schema, and client.aio async (tool-call ids are the function name since the API carries none). Bedrock-routed engines gained partial support per underlying API: Converse toolConfig on Nova/Claude, outputConfig structured output only on AWS-listed models, retry_policy collapsing the engine schedule; async and per-call timeouts stay unimplemented for lack of provider support. A new engine-agnostic extend_messages_with_turn helper lets one tool loop replay a model turn in each engine's wire shape. Version bumped to 2.15.0 across all three version locations per the three-file policy.
- Top decision: None.
- Blockers: Bedrock async variants and per-call timeouts remain open, blocked on upstream boto3/Converse support; Gemini async is single-attempt and relies on caller backoff, worth watching in practice.

| Metric | Value |
|---|---|
| Memory events captured | 1 |
| Repo files changed | 1 |
| Decision candidates | 0 |
| Active blockers | 1 |

## Major work completed

- OpenAI Chat Completions and Responses engines gained full support: send_conversation tool loops with forced tool_choice and strict functions, json_schema structured output, async variants on a lazy AsyncOpenAI, extended send_prompt parameters, retry_policy, and status-coded AiProviderRequestError. Google Gemini gained the same full surface via function-declaration tools, response_json_schema, and client.aio async (tool-call ids are the function name since the API carries none). Bedrock-routed engines gained partial support per underlying API: Converse toolConfig on Nova/Claude, outputConfig structured output only on AWS-listed models, retry_policy collapsing the engine schedule; async and per-call timeouts stay unimplemented for lack of provider support. A new engine-agnostic extend_messages_with_turn helper lets one tool loop replay a model turn in each engine's wire shape. Version bumped to 2.15.0 across all three version locations per the three-file policy.

## Why this mattered

- The library's value is one agent-feature contract that behaves identically across engines. 2.14.0 defined that capability-gated surface; this release makes it real everywhere, so callers can rely on either a working feature or a typed capability error instead of silent per-engine divergence. The remaining gaps are deliberate: they track missing underlying provider support (e.g. boto3 has no official async client), not unfinished work.

## Active blockers

- Bedrock async variants and per-call timeouts remain open, blocked on upstream boto3/Converse support; Gemini async is single-attempt and relies on caller backoff, worth watching in practice.

## Decision candidates

- None

## Next likely steps

- Bedrock async variants and per-call timeouts remain open, blocked on upstream boto3/Converse support; Gemini async is single-attempt and relies on caller backoff, worth watching in practice.

## Relevant event shards

- [2026-07-18 00:43:55 UTC by 2355287-davecthomas](events/2026-07-18T00-43-55Z--2355287-davecthomas--thread_4d5b72e2-795b-45f4-9ce9-29b3baabc94a--turn_blte0adwb.md)
