---
type: decision_candidate
decision_candidate: true
timestamp: "2026-04-03T00:00:04-07:00"
bootstrapped_at: "2026-04-03T21:00:00Z"
title: Unified tool-calling abstraction with intentional per-vendor API choices
tags: [architecture, tool-calling, openai, gemini, bedrock, design]
---

# Unified tool-calling abstraction with intentional per-vendor API choices

Tool calling is exposed through a single `AIToolCallingBase` subclass hierarchy with common types (`AIToolSpec`, `AIToolCall`, `AIToolResult`) and a single orchestration entrypoint `run_prompt_with_tools(...)`.

**Firm vendor API choices (do not deviate without an ADR):**
- **OpenAI:** always use the **Responses API** (`client.responses.create`), SDK >= 1.109.0 required for `tool_choice.type="allowed_tools"`.
- **Gemini:** always use `client.models.generate_content` (google-genai, REST: `models.generateContent`).
- **Bedrock:** always use **Converse API** with `toolConfig`.
- **AgentCore + Strands:** treat as a single tool-aware endpoint that orchestrates its own tools internally.

**Per-run control knobs (cross-vendor):**
- `allowed_tool_names: list[str] | None` in `AIToolRunConfig` — constrains which tools are active per run.
- `max_output_tokens: int | None` in `AIToolRunConfig` — maps to each vendor's output-length parameter.

**Hard non-goals:** completions and tool calling must NOT be merged into one client class. No multiple high-level modes (simple vs. advanced flows).

**Evidence:** `docs/tool_calling_specification.md` (§ 1.1 Goals, § 1.2 Non-Goals, § 2 Vendor Capability Survey); commit `7748caf`.
