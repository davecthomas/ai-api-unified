# ADR-0018 Expose caller-managed single-turn tool use and structured output on the completions abstraction, amending ADR-0004

Status: accepted
Date: 2026-07-17
Owners: dave-thomas
Must read: true
Supersedes: 
Superseded by: 
ai-generated: True
ai-tool: claude
ai-surface: claude-code
ai-executor: local-agent

Purpose: Put single-turn tool-use conversations and schema-constrained structured output on `AIBaseCompletions`, capability-gated per engine, amending ADR-0004's non-goal that kept tool calling off completions clients
Derived from: owner-specified feature requirements for a workflow-service consumer (2026-07-17)

## Context

- ADR-0004 placed tool calling in a separate `AIToolCallingBase` surface with a
  library-managed loop (`run_prompt_with_tools`), and recorded a hard non-goal:
  completions and tool calling must not merge into one client class. That
  surface was never implemented; it exists only as a specification.
- The first real consumer is a workflow service whose agent step needs the
  opposite ownership split: the caller executes tools (they are MCP calls on
  its side) and manages the loop bound; the library sends one turn at a time
  and returns the model's turn. A library-managed loop cannot execute
  caller-side tools without inverting control through callbacks.
- The same consumer needs single-shot structured extraction with a raw JSON
  Schema (hand-written `anyOf` variants), multi-turn correction replay, and a
  normalized finish reason to distinguish retryable truncation from refusal.
- The repo's completions abstraction already has the template-method plumbing
  (capability gates, observability wrapping, PII gating) these features need.

## Decision

- `AIBaseCompletions` gains capability-gated template methods:
  `send_conversation` (one turn per call, caller-owned loop),
  `send_structured_output`, `build_tool_result_message`, and async variants
  (`asend_prompt`, `asend_structured_output`, `asend_conversation`).
- Shared provider-neutral types live in `ai_base.py`: `AITool`, `AIToolCall`,
  `AITokenUsage`, `AITurnResult`, `AIStructuredOutputResult`, and the
  normalized `AIFinishReason` enum (`complete | length | tool_use | refusal`).
- Engines declare support via `AICompletionsCapabilitiesBase` flags
  (`supports_tool_use`, `supports_structured_output`, `supports_async`);
  unsupported calls raise the existing typed
  `AiProviderCapabilityUnsupportedError`. The native `claude` engine is the
  first full implementation.
- Engine-specific content blocks cross the boundary only as the opaque
  `AITurnResult.raw_content`, replayable verbatim as the next assistant turn;
  `build_tool_result_message` produces the engine's tool-result wire shape.
- This amends ADR-0004's non-goal (1): a caller-managed single-turn API on the
  completions client is now in scope. ADR-0004's remaining decisions stand —
  the per-vendor API locks for any future library-managed
  `run_prompt_with_tools` orchestrator, and the rejection of multiple
  high-level modes. If that orchestrator is ever built, it composes over
  `send_conversation` and reuses the engines' provider mappings.

## Consequences

- The workflow-service call shapes (structured extraction, plain generation,
  bounded tool loop) run against one client from `AIFactory`, selected by
  configuration, with token usage on every result object.
- Other engines adopt the surface incrementally behind capability flags; their
  callers get a typed error until support lands.
- ADR-0004's specification document remains the reference for a future
  orchestrated tool-run surface; new per-vendor mappings for single-turn
  conversations live in the engines themselves (Messages API `tools` /
  `tool_choice` / `output_config` for `claude`).
