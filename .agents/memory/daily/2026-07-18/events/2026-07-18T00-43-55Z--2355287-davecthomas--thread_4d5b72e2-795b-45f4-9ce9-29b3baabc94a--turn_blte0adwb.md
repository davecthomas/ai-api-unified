---
agentmemory_version: "0.4.4"
timestamp: "2026-07-18T00:43:55Z"
author: "2355287-davecthomas"
branch: "feat/multi-engine-agent-features"
thread_id: "4d5b72e2-795b-45f4-9ce9-29b3baabc94a"
turn_id: "blte0adwb"
workstream_id: "thread-4d5b72e2-795b-45f4-9ce9-29b3baabc94a"
workstream_scope: "thread"
episode_id: "episode-main-b34c818a56"
episode_scope: "mixed"
checkpoint_goal: "Bring the capability-gated agent-feature surface introduced in 2.14.0 (tool loops, structured output, async variants, retry policy, typed errors) to full parity across every supported engine."
checkpoint_surface: "The completions layer across all provider engines \u2014 ai_base plus the Anthropic, OpenAI (Chat Completions and Responses), Google Gemini, and Bedrock completion classes and their shared capability-gating contract."
checkpoint_outcome: "Shipped release 2.15.0 implementing the gated surface on every engine whose underlying API supports it, leaving unsupported gaps raising the typed capability error."
decision_candidate: false
enriched: true
ai_generated: true
ai_model: "claude-fable-5[1m]"
ai_tool: "claude"
ai_surface: "claude-code"
ai_executor: "local-agent"
related_adrs:
files_touched:
  - "CHANGELOG.md"
  - "README.md"
  - "pyproject.toml"
  - "src/ai_api_unified/__version__.py"
  - "src/ai_api_unified/ai_anthropic_base.py"
  - "src/ai_api_unified/ai_base.py"
  - "src/ai_api_unified/ai_openai_base.py"
  - "src/ai_api_unified/completions/ai_anthropic_completions.py"
  - "src/ai_api_unified/completions/ai_bedrock_completions.py"
  - "src/ai_api_unified/completions/ai_google_gemini_capabilities.py"
  - "src/ai_api_unified/completions/ai_google_gemini_completions.py"
  - "src/ai_api_unified/completions/ai_openai_completions.py"
  - "src/ai_api_unified/completions/ai_openai_responses_completions.py"
  - "tests/test_google_gemini.py"
  - "tests/test_multi_engine_conversation_api.py"
design_docs_touched:
verification:
  - "git diff:  14 files changed, 3149 insertions(+), 133 deletions(-); ## 2.15.0; The 2.14.0 capability-gated surface lands on every engine whose underlying; API supports it; the remaining gaps stay unimplemented and raise the typed"
source_pending_shards:
  - ".agents/memory/pending/2026-07-18/2026-07-18T00-43-55Z--2355287-davecthomas--thread_4d5b72e2-795b-45f4-9ce9-29b3baabc94a--turn_blte0adwb.md"
---

## Why

- The library's value is one agent-feature contract that behaves identically across engines. 2.14.0 defined that capability-gated surface; this release makes it real everywhere, so callers can rely on either a working feature or a typed capability error instead of silent per-engine divergence. The remaining gaps are deliberate: they track missing underlying provider support (e.g. boto3 has no official async client), not unfinished work.

## What changed

- OpenAI Chat Completions and Responses engines gained full support: send_conversation tool loops with forced tool_choice and strict functions, json_schema structured output, async variants on a lazy AsyncOpenAI, extended send_prompt parameters, retry_policy, and status-coded AiProviderRequestError. Google Gemini gained the same full surface via function-declaration tools, response_json_schema, and client.aio async (tool-call ids are the function name since the API carries none). Bedrock-routed engines gained partial support per underlying API: Converse toolConfig on Nova/Claude, outputConfig structured output only on AWS-listed models, retry_policy collapsing the engine schedule; async and per-call timeouts stay unimplemented for lack of provider support. A new engine-agnostic extend_messages_with_turn helper lets one tool loop replay a model turn in each engine's wire shape. Version bumped to 2.15.0 across all three version locations per the three-file policy.

## Evidence

- The CHANGELOG 2.15.0 entry states the release contract explicitly: the 2.14.0 capability-gated surface lands on every engine whose underlying API supports it, and remaining gaps raise the typed capability error. The README now carries a feature-support-by-engine matrix that makes each engine's support level a documented public commitment. The multi-engine conversation API test suite exercises the same tool loop across engines via the new replay helper, and the Gemini test suite covers the function-declaration tool and structured-output paths, validating parity rather than per-engine one-offs.

## Next

- Bedrock async variants and per-call timeouts remain open, blocked on upstream boto3/Converse support; Gemini async is single-attempt and relies on caller backoff, worth watching in practice.
