# ADR-0024 Route unsure picks up, allow a floor per call, and descend only by opt-in

Status: accepted
Date: 2026-10-16
Owners: dave-thomas
Must read: true
Supersedes: 
Superseded by: 
ai-generated: True
ai-tool: claude
ai-surface: claude-code
ai-executor: local-agent

Purpose: Settle which way the router errs when it is unsure or out of tiers, and how a conversation keeps its level
Derived from: owner requirements audit (2026-10-16)

## Context

- The router shared one thinking level across tiers, logged the
  classifier's confidence without acting on it, reclassified every plain
  conversation turn, and on exhaustion could only escalate or raise.
- The library has no conversation id: one client may serve many
  conversations, so per-conversation state cannot live in the client.

## Decision

- **Thinking is a tier property.** `AIRouterTier.thinking`, falling back
  to the router's level; per-call levels still win (ADR-0021).
- **Unsure goes up, one tier.** Below `classifier.min_confidence` the pick
  moves one tier up. A wrong pick toward the stronger tier costs money; a
  wrong pick toward the weaker one costs answer quality, and the second is
  the one a caller cannot see. One tier, not the top, keeps the cost
  bounded. A classifier without a confidence keeps its pick. Off by
  default.
- **A floor is per call; the ratchet is per client.** `route_floor` is a
  reserved provider option, the exact mechanism for shared clients. The
  `followup: "ratchet"` setting uses the client's last tier and is
  documented as exact only with one client per conversation. Neither
  moves an override or a tool-shaped history. Reclassifying stays the
  default, since a cheap follow-up going to a cheap tier is the point of
  routing.
- **Descending is opt-in.** `on_exhausted: "any"` tries tiers above the
  pick first, then below it, nearest first, and logs every descent at
  WARNING. The default stays `escalate`, so a request the classifier
  called hard never gets a weaker model's answer without the operator
  asking for that.
- **Classifier cost is checked at startup, not per call.** A routed call's
  cost is known only after it returns, so the router compares input rates
  once at build and warns when the classifier is at or above the
  cheapest tier; the per-request ledger already records the classifier's
  spend.

## Consequences

- Every option is off by default, so existing configurations route as
  before.
- `tier_exhausted` gains a `direction` field; dashboards that assumed
  upward escalation can now tell a descent apart.

## Related code paths

- src/ai_api_unified/routing/router_config.py
- src/ai_api_unified/completions/ai_router_completions.py
- src/ai_api_unified/ai_models_config.py
