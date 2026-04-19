# 2026-04-19 summary

## Snapshot

- Captured 1 memory event.
- Main work: The six raise ValueError statements in _validate_google_video_properties were converted to f-strings that interpolate self.<field>!r into the message, mirroring the fps form ("received fps={self.fps!r}"). Existing test regexes for "must be ..." were loosened to [Mm]ust patterns because several messages now start the allowed-set clause mid-sentence after the interpolated value. A new positive-contract test, test_google_video_properties_rejection_message_includes_offending_value, uses resolution='8k' and matches r"resolution='8k'" to lock in that the offending value must appear in the raised message going forward. Net change is +27/-10 across the provider module and its test file; no behavior change beyond error-message content and the matching regex relaxations.
- Top decision: None.
- Blockers: None.

| Metric | Value |
|---|---|
| Memory events captured | 1 |
| Repo files changed | 1 |
| Decision candidates | 0 |
| Active blockers | 0 |

## Major work completed

- The six raise ValueError statements in _validate_google_video_properties were converted to f-strings that interpolate self.<field>!r into the message, mirroring the fps form ("received fps={self.fps!r}"). Existing test regexes for "must be ..." were loosened to [Mm]ust patterns because several messages now start the allowed-set clause mid-sentence after the interpolated value. A new positive-contract test, test_google_video_properties_rejection_message_includes_offending_value, uses resolution='8k' and matches r"resolution='8k'" to lock in that the offending value must appear in the raised message going forward. Net change is +27/-10 across the provider module and its test file; no behavior change beyond error-message content and the matching regex relaxations.

## Why this mattered

- The prior fps-rejection fix introduced the pattern of naming the offending value in the error message, but the other five rejection paths still produced generic 'must be X' messages that left callers to re-read their config to guess what they had actually submitted. That asymmetry undermined the fail-fast Properties boundary: a caller who mistyped resolution='8k' or person_generation='allow_all' got a message describing the allowed set without echoing the rejected value, which is exactly the debuggability gap the Properties-layer validation is supposed to close (per the 2026-04-18 daily summary and ADR-0008's stance that providers own their capability surface). Aligning all six messages with the fps form preserves the contract the fps fix already made and keeps Veo-specific capability knowledge inside the provider module instead of leaking to remote errors or opaque local silence.

## Active blockers

- None

## Decision candidates

- None

## Next likely steps

- Propagate the include-offending-value pattern to other video providers' Properties validators as part of the 2026-04-18 follow-up audit, so debuggability is uniform across vendors.
- Consider lifting the 'rejection message names the offending value' rule into a shared contract (base-class helper or shared test fixture) rather than per-provider regex assertions.
- Land this change through the normal PR flow so the new regression test gates future validator-message regressions.

## Relevant event shards

- [2026-04-19 22:37:21 UTC by 2355287-davecthomas](events/2026-04-19T22-37-21Z--2355287-davecthomas--thread_0407f408-a895-496a-9531-9bcfa6c94600--turn_40d0ff8d1f.md)
