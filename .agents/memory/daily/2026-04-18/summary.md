# 2026-04-18 summary

## Snapshot

- Captured 2 memory events.
- Main work: AIGoogleGeminiVideoProperties._validate_google_video_properties now raises ValueError with a message naming the unsupported 'fps' parameter when it is set. The config_kwargs builder in AIGoogleGeminiVideos no longer forwards fps into the generate_videos config; the previous forwarding block is replaced with a comment explaining that validation upstream prevents this case. Tests removed fps=24 from the 'forwards supported generate_videos config fields' scenario (which is no longer a valid property set) and added test_google_video_properties_reject_fps to pin down the rejection behavior via pytest.raises.
- Top decision: None.
- Blockers: None.

| Metric | Value |
|---|---|
| Memory events captured | 2 |
| Repo files changed | 2 |
| Decision candidates | 0 |
| Active blockers | 0 |

## Major work completed

- AIGoogleGeminiVideoProperties._validate_google_video_properties now raises ValueError with a message naming the unsupported 'fps' parameter when it is set. The config_kwargs builder in AIGoogleGeminiVideos no longer forwards fps into the generate_videos config; the previous forwarding block is replaced with a comment explaining that validation upstream prevents this case. Tests removed fps=24 from the 'forwards supported generate_videos config fields' scenario (which is no longer a valid property set) and added test_google_video_properties_reject_fps to pin down the rejection behavior via pytest.raises.
- AIGoogleGeminiVideoProperties gained four optional fields (negative_prompt, enhance_prompt, output_gcs_uri, compression_quality) with upstream validation: compression_quality is normalized to uppercase and checked against the SDK enum set {OPTIMIZED, LOSSLESS}; output_gcs_uri must start with gs://. The _ALLOWED_RESOLUTIONS set lost '4k' and _ALLOWED_PERSON_GENERATION lost 'allow_all', with matching error messages updated. The config_kwargs builder in AIGoogleGeminiVideos now forwards each new field into the generate_videos config, converting compression_quality through types.VideoCompressionQuality. This is a net +100/-4 change across the provider and its test module, continuing the same fail-fast-at-the-Properties-boundary pattern introduced for fps.

## Why this mattered

- Veo does not accept fps; the previous code forwarded it into the generate_videos config, which either triggered a remote rejection or was silently dropped — a leaky abstraction that hid a provider capability gap from the caller. Validating at the Properties layer gives callers a clear, actionable error at the right boundary and keeps provider-specific capability handling owned by the provider code, consistent with ADR-0008's stance on explicit video-engine selection and provider-owned defaults.
- Veo's real capability surface is narrower than earlier code implied: '4k' and 'allow_all' were accepted client-side but rejected or silently dropped by the remote API, and four useful parameters (negative_prompt, enhance_prompt, output_gcs_uri, compression_quality) were not exposed at all. Leaving these gaps in place would keep the same class of leaky abstraction the fps fix addressed — callers getting either opaque remote errors or silent behavior loss. Validating at the Properties layer keeps provider-specific capability knowledge owned by the provider module per ADR-0008, gives callers actionable errors at the right boundary, and prepares a regression-gated path to expose the rest of Veo's supported surface.

## Active blockers

- None

## Decision candidates

- None

## Next likely steps

- Audit the other video providers for analogous silently-accepted-but-unsupported parameters and apply the same fail-fast validation pattern.
- Decide whether AIBaseVideoProperties should grow an explicit per-provider 'unsupported fields' declaration mechanism, or whether per-provider model_validators remain the right surface.
- Land this fix through the normal PR flow so the new rejection test gates future regressions.
- Land this change through the normal PR flow on issue/14 so the five new tests gate future regressions.
- Run the same audit on the other video providers (per the 2026-04-18 summary's follow-up item) and mirror the pattern.
- Consider whether the recurring per-provider 'unsupported / rejected fields' shape warrants lifting into a shared declaration on AIBaseVideoProperties rather than repeating model_validator blocks per provider — candidate discussion, not yet a decision.

## Relevant event shards

- [2026-04-18 21:56:37 UTC by 2355287-davecthomas](events/2026-04-18T21-56-37Z--2355287-davecthomas--thread_0407f408-a895-496a-9531-9bcfa6c94600--turn_7ef0d1ee04.md)
- [2026-04-18 23:52:58 UTC by 2355287-davecthomas](events/2026-04-18T23-52-58Z--2355287-davecthomas--thread_0407f408-a895-496a-9531-9bcfa6c94600--turn_c66a9c9687.md)
