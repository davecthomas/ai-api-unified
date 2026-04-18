# 2026-04-18 summary

## Snapshot

- Captured 1 memory event.
- Main work: AIGoogleGeminiVideoProperties._validate_google_video_properties now raises ValueError with a message naming the unsupported 'fps' parameter when it is set. The config_kwargs builder in AIGoogleGeminiVideos no longer forwards fps into the generate_videos config; the previous forwarding block is replaced with a comment explaining that validation upstream prevents this case. Tests removed fps=24 from the 'forwards supported generate_videos config fields' scenario (which is no longer a valid property set) and added test_google_video_properties_reject_fps to pin down the rejection behavior via pytest.raises.
- Top decision: None.
- Blockers: None.

| Metric | Value |
|---|---|
| Memory events captured | 1 |
| Repo files changed | 1 |
| Decision candidates | 0 |
| Active blockers | 0 |

## Major work completed

- AIGoogleGeminiVideoProperties._validate_google_video_properties now raises ValueError with a message naming the unsupported 'fps' parameter when it is set. The config_kwargs builder in AIGoogleGeminiVideos no longer forwards fps into the generate_videos config; the previous forwarding block is replaced with a comment explaining that validation upstream prevents this case. Tests removed fps=24 from the 'forwards supported generate_videos config fields' scenario (which is no longer a valid property set) and added test_google_video_properties_reject_fps to pin down the rejection behavior via pytest.raises.

## Why this mattered

- Veo does not accept fps; the previous code forwarded it into the generate_videos config, which either triggered a remote rejection or was silently dropped — a leaky abstraction that hid a provider capability gap from the caller. Validating at the Properties layer gives callers a clear, actionable error at the right boundary and keeps provider-specific capability handling owned by the provider code, consistent with ADR-0008's stance on explicit video-engine selection and provider-owned defaults.

## Active blockers

- None

## Decision candidates

- None

## Next likely steps

- Audit the other video providers for analogous silently-accepted-but-unsupported parameters and apply the same fail-fast validation pattern.
- Decide whether AIBaseVideoProperties should grow an explicit per-provider 'unsupported fields' declaration mechanism, or whether per-provider model_validators remain the right surface.
- Land this fix through the normal PR flow so the new rejection test gates future regressions.

## Relevant event shards

- [2026-04-18 21:56:37 UTC by 2355287-davecthomas](events/2026-04-18T21-56-37Z--2355287-davecthomas--thread_0407f408-a895-496a-9531-9bcfa6c94600--turn_7ef0d1ee04.md)
