# Harness Engine Design

## Summary

This document describes a `harness` completions engine. It lets a router
tier candidate name a whole coding run instead of a single model call. The
candidate form is `harness:<profile>/<inner-engine>:<inner-model>`, for
example `harness:pi-lean/claude:claude-sonnet-5`.

The router, fallback chain, cooldowns, model pin, per-call floors, and the
request cost report treat a run the same way they treat any other
completions call. The engine itself holds no state between calls: it reads
its run identity from the call, asks two remote servers for what it needs,
blocks until the run ends, and returns a result with a cost report.

This engine exists to serve the meta-harness, a control plane that routes
coding tasks to a harness-and-model combination and runs the harness in a
sandbox. The meta-harness is the caller in every diagram below. Its own
design document carries the whole-system sequences (task API, scheduler,
gateway, console); this document stops at the library's boundary.

The library never knows what program runs inside the sandbox. The profile
name is an opaque string the engine passes along. The sandbox program (pi,
or anything else) is owned by the meta-harness and speaks a small Run API.

## What this library adds

| Change | Where |
|---|---|
| `harness` completions engine: registry entry, model-string parser, probe that delegates to the inner model's probe and checks `pool_url`, capability table that delegates to the inner model's, pool and Run API client | `completions/ai_harness_completions.py`, `ai_provider_registry.py`, `util/ai_model_probes.py` |
| `harness_run_started` observability event at lease time (run id, sandbox, candidate, tier) | the engine |
| Reported-cost entry in the request cost ledger, so an engine can record dollars a gateway stated rather than tokens times a price | `middleware/request_cost.py` |
| `AiFallbackReason.RUN_FAILED`, with no cooldown period | `ai_provider_exceptions.py`, `completions/ai_fallback_cooldown.py` |
| The fallback chain places the candidate index in the request-scoped observability context | `completions/ai_fallback_completions.py` |
| `providers.harness` with `pool_url` and `gateway_url`; credentials stay in the environment | `ai_models_config.py`, `schemas/` |

Later, optional: a cancel event on completions calls; a streaming form of
`send_conversation` with typed events.

## Parties in the diagrams

| Party | Owned by | Role |
|---|---|---|
| Caller | the calling system | In the meta-harness this is the dispatcher, run from a run workflow: it builds the client with `AIFactory`, renders the task spec as the latest user message, and passes `harness_call_id` in `provider_options`. The library does not care who calls it; any Python program with harness candidates in its `models.json` is a caller |
| Router | this library | `AiRouterCompletions`: classifies the prompt, picks a tier, runs the tier's chain |
| Chain | this library | `AiFallbackCompletions` for one tier: tries candidates in order, applies cooldowns, sums cost across tries |
| Engine | this library | `AiHarnessCompletions`: one blocking call per run |
| Pool | the calling system | Pool manager: leases sandboxes by run id |
| Sandbox | the calling system | The harness server inside a leased sandbox: `POST /run`, `GET /events`, `GET /usage`, cancel |
| Gateway | ai-api-unified-http | Where the sandbox program sends its model calls; outside these diagrams except where noted |

Run id is `harness_call_id` plus the candidate index, so a retried call
derives the same id and finds its run.

## Scenario 1: a routed run that succeeds

```mermaid
sequenceDiagram
    autonumber
    participant C as Caller
    participant R as Router
    participant K as Classifier
    participant F as Chain (tier)
    participant E as Engine
    participant P as Pool
    participant S as Sandbox

    C->>R: send_conversation(system, messages, provider_options={harness_call_id})
    R->>K: classify(latest user message)
    K-->>R: tier "standard", confidence 0.82
    Note over R: emits routed(route_tier, route_method, config_id)
    R->>F: send_conversation(...) on the "standard" chain
    F->>E: candidate 0: harness:pi-lean/claude:claude-sonnet-5
    Note over F,E: candidate index 0 placed in request context
    E->>E: run_id = harness_call_id + "/0"
    E->>P: GET lease(run_id)
    P-->>E: none
    E->>P: POST lease(run_id, image="pi-lean", resources)
    P-->>E: sandbox address, lease TTL
    Note over E: emits harness_run_started(run_id, sandbox, candidate, tier)
    E->>E: mint run token (run id, tags, pin=claude:claude-sonnet-5, cap, expiry)
    E->>S: POST /run (prompt, profile, limits, token)
    S-->>E: 202 run accepted
    E->>S: GET /events (SSE)
    S-->>E: ... run_finished(end_reason=pr_opened)
    E->>S: GET /usage
    S-->>E: usd_total, turns, tokens
    E->>P: release lease(run_id)
    E-->>F: AITurnResult(text=PR URL, finish=complete, cost=reported usd)
    Note over F: ledger: one reported-cost entry, source=gateway, run_id
    F-->>R: result
    R-->>C: result (route_tier=standard, model_name=harness:pi-lean/..., cost)
```

## Scenario 2: the first candidate has no sandbox; the second runs

```mermaid
sequenceDiagram
    autonumber
    participant F as Chain (tier)
    participant E as Engine
    participant P as Pool
    participant S as Sandbox
    participant CD as Cooldown registry

    F->>E: candidate 0: harness:pi-lean/claude:claude-sonnet-5
    E->>P: GET lease(call/0) -> none; POST lease(call/0, "pi-lean")
    P-->>E: 503 no capacity within ready_timeout
    E-->>F: raise AiProviderRequestError(reason=UNAVAILABLE)
    F->>CD: bench candidate 0 for 30 s (reason UNAVAILABLE)
    Note over F: UNAVAILABLE is in fallback_on: move on
    F->>E: candidate 1: harness:pi-careful/openai:gpt-5.6-luna
    E->>P: lease(call/1, "pi-careful")
    P-->>E: sandbox
    E->>S: POST /run ... GET /events ... run_finished
    E-->>F: result with reported cost
    Note over F: request total = cost of try 0 (zero) + try 1
    F-->>F: return result; provider_engine/model name the candidate that served
```

## Scenario 3: the run fails on the task; the tier is exhausted; escalate

```mermaid
sequenceDiagram
    autonumber
    participant R as Router
    participant F1 as Chain (standard)
    participant F2 as Chain (complex)
    participant E as Engine
    participant S as Sandbox
    participant CD as Cooldown registry

    R->>F1: send on "standard"
    F1->>E: candidate 0
    E->>S: run ... run_finished(end_reason=ERROR)
    E-->>F1: raise (reason=RUN_FAILED), cost of the failed run attached
    Note over F1,CD: RUN_FAILED benches nothing: one hard task says nothing about the candidate's health
    F1->>E: candidate 1
    E->>S: run ... run_finished(end_reason=turn_limit)
    E-->>F1: raise (reason=RUN_FAILED)
    F1-->>R: tier exhausted (both tries' cost on the ledger)
    Note over R: on_exhausted = escalate: next tier up, never down
    Note over R: emits tier_exhausted(direction=up)
    R->>F2: send on "complex"
    F2->>E: candidate 0: harness:pi-careful/claude:claude-opus-5-5
    E->>S: run ... run_finished(pr_opened)
    E-->>F2: result
    F2-->>R: result
    Note over R: request cost report sums all three runs
```

## Scenario 4: a follow-up pinned to the original candidate

```mermaid
sequenceDiagram
    autonumber
    participant C as Caller
    participant R as Router
    participant F as Chain (tier)
    participant E as Engine
    participant P as Pool
    participant S as Sandbox

    Note over C: with model_pin_scope("harness:pi-lean/claude:claude-sonnet-5")
    C->>R: send_conversation(messages incl. restored session, provider_options={harness_call_id: new id})
    Note over R: pinned: skip classifier, ignore route/route_floor, never escalate
    R->>F: send on the tier that holds the pinned candidate
    F->>E: the pinned candidate only
    E->>P: lease(new_id/0, "pi-lean")
    P-->>E: sandbox
    E->>S: POST /run (prompt = follow-up, session = restored history, token)
    S-->>E: ... run_finished
    E-->>F: result (route_method=model_pin, model_pin=harness:pi-lean/...)
    F-->>R: result
    R-->>C: result
```

## Scenario 5: the caller's process dies mid-run and retries

```mermaid
sequenceDiagram
    autonumber
    participant C as Caller
    participant F as Chain (tier)
    participant E as Engine
    participant P as Pool
    participant S as Sandbox

    C->>F: call (harness_call_id = X)
    F->>E: candidate 0
    E->>P: lease(X/0) -> sandbox A
    E->>S: POST /run; GET /events ...
    Note over C,E: caller process exits; sandbox A keeps running
    C->>F: same call again (harness_call_id = X)
    F->>E: candidate 0
    E->>P: GET lease(X/0)
    P-->>E: existing lease: sandbox A, run in progress
    Note over E: re-attach, no second run, no second lease
    E->>S: GET /events (from cursor 0)
    S-->>E: replayed events ... run_finished
    E->>S: GET /usage
    E-->>F: result with the run's full reported cost
```

## Scenario 6: the lease TTL passes; the engine cancels and raises TIMEOUT

```mermaid
sequenceDiagram
    autonumber
    participant F as Chain (tier)
    participant E as Engine
    participant S as Sandbox
    participant P as Pool

    F->>E: candidate 0, request_timeout_seconds = lease TTL
    E->>S: POST /run; GET /events
    Note over E: deadline passes with no run_finished
    E->>S: POST /cancel
    S-->>E: run_canceled, partial usage
    E->>P: release lease
    E-->>F: raise (reason=TIMEOUT), partial cost attached
    Note over F: TIMEOUT in fallback_on -> next candidate; otherwise the call fails with the cost so far
```

## Scenario 7: startup check of a harness candidate

```mermaid
sequenceDiagram
    autonumber
    participant Fa as AIFactory
    participant MC as Model checker
    participant HP as harness probe
    participant IP as inner engine probe
    participant P as Pool

    Fa->>MC: check_all(every configured candidate)
    MC->>HP: harness:pi-lean/claude:claude-sonnet-5
    HP->>HP: parse: profile=pi-lean, inner=claude:claude-sonnet-5
    HP->>IP: probe claude model claude-sonnet-5
    IP-->>HP: available
    HP->>P: GET /healthz (pool_url)
    P-->>HP: ok
    HP-->>MC: available
    Note over MC: an unknown inner model or an unreachable pool is reported here, not at the first run
```

## What the engine does not do

- It never calls a model. The classifier call happens in the router before
  the engine is chosen; the sandbox program's model calls go to the gateway.
- It never streams progress through the router. Progress is read from the
  sandbox's event stream by the calling system.
- It never holds a run between calls. Everything it needs arrives with the
  call or is asked of the pool.
