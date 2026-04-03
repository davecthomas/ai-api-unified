---
adr: "0004"
title: Unified tool-calling abstraction with intentional per-vendor API choices
status: Accepted
date: "2026-04-03"
tags: [architecture, tool-calling, openai, gemini, bedrock]
must_read: true
supersedes: ~
superseded_by: ~
---

# ADR-0004: Unified tool-calling abstraction with intentional per-vendor API choices

## Status
Accepted

## Context
Adding tool-calling support across OpenAI, Gemini, Bedrock, and AgentCore + Strands required a consistent abstraction that preserves the library's existing vendor base-class pattern without collapsing completions and tool-calling into one class.

## Decision
Tool calling is exposed through `AIToolCallingBase` subclasses with shared types (`AIToolSpec`, `AIToolCall`, `AIToolResult`) and a single entrypoint `run_prompt_with_tools(...)`.

**Firm per-vendor API choices (require ADR to change):**
- **OpenAI:** Responses API only (`client.responses.create`); SDK >= 1.109.0 for `tool_choice.type="allowed_tools"`.
- **Gemini:** `client.models.generate_content` (google-genai) only.
- **Bedrock:** Converse API with `toolConfig` only.
- **AgentCore + Strands:** single tool-aware endpoint; internal tool orchestration is the agent's responsibility.

**Per-run control knobs (cross-vendor via `AIToolRunConfig`):**
- `allowed_tool_names: list[str] | None` — constrains active tools per invocation.
- `max_output_tokens: int | None` — maps to each vendor's output-length parameter.

**Hard non-goals:**
- Completions and tool calling must NOT be merged into one client class.
- No multiple high-level modes (simple vs. advanced flows).

## Consequences
- One consistent tool-definition and result surface for callers regardless of vendor.
- Per-vendor API lock-in is explicit and documented — unexpected vendor API drift surfaces as a decision point, not a silent behavior change.
- OpenAI custom tools require `parallel_tool_calls=false` when any `type="custom"` tool is present.

## Evidence
`docs/tool_calling_specification.md` (§ 1 Goals and Non-Goals, § 2 Vendor Capability Survey); commit `7748caf`.
