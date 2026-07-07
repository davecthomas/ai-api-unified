# Model pricing research and API recommendation

Compiled 2026-07-07 from official provider pricing pages. This is a review
deliverable for the pricing redesign that precedes the planned financial-ops
observability middleware. Rates are current as of the compile date; every rate
carries a confidence level and source below.

## The core finding: the current cost model cannot support finops

Today each model exposes a single blended `price_per_1k_tokens` float, and
`calculate_cost(num_tokens)` multiplies that one rate by a token count. Every
row of the research below breaks that model:

1. **Input and output are priced separately, and asymmetrically.** Output runs
   4–8× input (gpt-5.5: $5 in / $30 out; gemini-2.5-pro: $1.25 / $10). A single
   blended rate times total tokens is wrong for any real input/output mix.
2. **Cached input is a first-class rate now**, roughly 10× cheaper than fresh
   input (gpt-5.4: $2.50 vs $0.25). The current model has no slot for it.
3. **Non-text modalities do not price per token.** Images bill per image, video
   per second, TTS per character. The code already works around this — the
   Gemini table carries comments like "converted from $0.0003125 per 1k
   characters," which is a different unit forced into the token field.
4. **Some models have context-length tiers.** gemini-2.5-pro doubles above 200K
   input tokens ($1.25→$2.50 in, $10→$15 out). One float cannot express a tier.
5. **No provenance.** There is no effective date or source on any rate, so cost
   figures are not auditable — a hard requirement for a finops layer.
6. **The current numbers are stale and partly invented.** The OpenAI blended
   values are midpoints (gpt-5 "0.0100" = a guess between $1.25 in and $10 out),
   and several listed models are retired (see Model health below).

## Normalized pricing tables

There is no single cross-modal unit — a token and a rendered second are not
convertible. The honest normalization is a **canonical unit per modality**:
text and embeddings per 1M tokens, character-billed speech per 1M characters,
images per image, video per second, transcription per audio minute. Token
models split into input / cached-input / output.

### Text completions — USD per 1M tokens

| Provider | Model | Input | Cached input | Output | Conf. |
|---|---|---:|---:|---:|---|
| OpenAI | gpt-5.5 | 5.00 | 0.50 | 30.00 | high |
| OpenAI | gpt-5.4 | 2.50 | 0.25 | 15.00 | high |
| OpenAI | gpt-5.4-mini | 0.75 | 0.075 | 4.50 | high |
| OpenAI | gpt-5.4-nano | 0.20 | 0.02 | 1.25 | high |
| OpenAI | gpt-5.2 | 1.75 | 0.175 | 14.00 | high |
| OpenAI | gpt-5.1-codex-max | 1.25 | 0.125 | 10.00 | high |
| OpenAI | gpt-5 | 1.25 | 0.125 | 10.00 | high |
| OpenAI | gpt-5-mini | 0.25 | 0.025 | 2.00 | high |
| OpenAI | gpt-5-nano | 0.05 | 0.005 | 0.40 | high |
| OpenAI | gpt-4.1 | 2.00 | 0.50 | 8.00 | high |
| OpenAI | gpt-4.1-mini | 0.40 | 0.10 | 1.60 | high |
| OpenAI | gpt-4.1-nano | 0.10 | 0.025 | 0.40 | high |
| OpenAI | o4-mini | 1.10 | 0.275 | 4.40 | high |
| OpenAI | gpt-4o | 2.50 | 1.25 | 10.00 | high |
| OpenAI | gpt-4o-mini | 0.15 | 0.075 | 0.60 | high |
| Google | gemini-2.5-pro (≤200K) | 1.25 | 0.13 | 10.00 | high |
| Google | gemini-2.5-pro (>200K) | 2.50 | 0.25 | 15.00 | high |
| Google | gemini-2.5-flash | 0.30 | 0.03 | 2.50 | high |
| Google | gemini-2.5-flash-lite | 0.10 | 0.01 | 0.40 | high |
| Google | gemini-2.0-flash ⚠ | 0.10 | n/a | 0.40 | med |
| Google | gemini-2.0-flash-lite ⚠ | 0.075 | n/a | 0.30 | med |
| Bedrock | amazon.nova-micro | 0.035 | n/a | 0.14 | high |
| Bedrock | amazon.nova-lite | 0.06 | n/a | 0.24 | high |
| Bedrock | amazon.nova-pro | 0.80 | n/a | 3.20 | high |
| Bedrock | amazon.nova-premier | 2.50 | n/a | 12.50 | high |
| Bedrock | claude-3-5-haiku | 0.80 | n/a | 4.00 | high |
| Anthropic | claude-fable-5 | 10.00 | 1.00 | 50.00 | high |
| Anthropic | claude-opus-4-8 | 5.00 | 0.50 | 25.00 | high |
| Anthropic | claude-opus-4-7 | 5.00 | 0.50 | 25.00 | high |
| Anthropic | claude-opus-4-6 | 5.00 | 0.50 | 25.00 | high |
| Anthropic | claude-sonnet-4-6 | 3.00 | 0.30 | 15.00 | high |
| Anthropic | claude-haiku-4-5 | 1.00 | 0.10 | 5.00 | high |
| Anthropic | claude-opus-4-1 ⚠ | 15.00 | 1.50 | 75.00 | high |

Notes: Anthropic rates are the native-API list (added 2026-07 with the `claude`
engine); the cached-input column is the documented 0.1x prompt-cache read rate,
and 5-minute cache writes bill 1.25x input (not modeled). Anthropic lifecycle
entries also cover claude-opus-4-1 (deprecated, retires 2026-08-05, replace
with claude-opus-4-8) and the retired claude-3-7-sonnet, claude-3-5-haiku, and
claude-3-opus snapshots. Gemini audio input is priced higher than text (2.5-flash audio input
$1.00/1M). Gemini rates shown are the Gemini Developer API list; Vertex differs
for the 2.0 family ($0.15 in / $0.60 out). Bedrock on-demand; cross-region
inference profiles add ~10%, batch ≈50% off.

### Embeddings — USD per 1M tokens (input only)

| Provider | Model | Input | Conf. |
|---|---|---:|---|
| OpenAI | text-embedding-3-small | 0.02 | high |
| OpenAI | text-embedding-3-large | 0.13 | high |
| OpenAI | text-embedding-ada-002 | 0.10 | high |
| Google | gemini-embedding-001 | 0.15 (batch 0.075) | high |
| Google | gemini-embedding-2 (text) | 0.20 (batch 0.10) | high |
| Bedrock | amazon.titan-embed-text-v2 | 0.02 | med |
| Bedrock | amazon.titan-embed-text-v1 | 0.10 | med |

Note: gemini-embedding-2 is multimodal and prices non-text input separately —
image $0.45/1M, audio $6.50/1M, video $12.00/1M. This is another case the
single-rate model cannot hold.

### Images — USD per image

| Provider | Model | Rate | Conf. |
|---|---|---|---|
| OpenAI | gpt-image-1 | low 0.011 / medium 0.042 / high 0.167 (1024²); more for larger sizes | high |
| Google | imagen-4.0-generate-001 ⚠ | fast 0.02 / standard 0.04 / ultra 0.06 | high |
| Bedrock | amazon.nova-canvas-v1:0 | 0.04 std / 0.06 premium (≤1024²); 0.06 / 0.08 (≤2048²) | med |

gpt-image-1 also has token rates for image input/output ($5 text in, $10 image
in, $40 image out per 1M) when used in the multimodal token path.

### Video — USD per second of output

| Provider | Model | Rate | Conf. |
|---|---|---|---|
| OpenAI | sora-2 | 0.10 (720p) | high |
| OpenAI | sora-2-pro | 0.30 (720p) / 0.50 / 0.70 (1080p) | high |
| Google | veo-3.1 | 0.40 (720/1080p) / 0.60 (4K) | high |
| Google | veo-3.1-lite | 0.05 (720p) / 0.08 (1080p) | med |
| Bedrock | amazon.nova-reel-v1:0 | 0.08 (720p) | med |

### Voice — TTS and STT

| Provider | Model | Modality | Rate | Unit | Conf. |
|---|---|---|---|---|---|
| OpenAI | tts-1 | TTS | 15.00 | per 1M chars | high |
| OpenAI | tts-1-hd | TTS | 30.00 | per 1M chars | high |
| OpenAI | gpt-4o-mini-tts | TTS | 0.60 in / 12.00 out | per 1M tokens | high |
| Google | gemini-2.5-flash-tts | TTS | 0.50 in / 10.00 out | per 1M tokens | high |
| Google | gemini-2.5-pro-tts | TTS | 1.00 in / 20.00 out | per 1M tokens | high |
| Azure | neural (standard) | TTS | 16.00 | per 1M chars | med |
| Azure | neural HD | TTS | 22.00 | per 1M chars | med |
| ElevenLabs | Multilingual v2/v3 | TTS | 100.00 | per 1M chars (API PAYG) | high |
| ElevenLabs | Flash/Turbo v2.5 | TTS | 50.00 | per 1M chars (API PAYG) | high |
| OpenAI | whisper-1 | STT | 0.006 | per minute | high |
| OpenAI | gpt-4o-transcribe | STT | 2.50 in / 10.00 out | per 1M tokens (≈$0.006/min) | high |
| Azure | standard STT | STT | 1.00 | per audio hour (≈$0.017/min) | med |
| ElevenLabs | Scribe v1/v2 | STT | 0.22 | per audio hour (≈$0.0037/min) | high |

TTS billing units are not uniform: tts-1 and Azure/ElevenLabs bill per
character, while gpt-4o-mini-tts and the Gemini TTS models bill per token. A
finops layer must track the billing unit per model, not assume characters.

ElevenLabs caveat: the per-character rates are API pay-as-you-go. Under a
subscription plan you prepay bundled credits, so the effective cost per
character depends on how fully the tier is consumed. Treat the API rate as an
upper-bound reference.

## Model health — several listed models are dead or dying

The current code advertises models that no longer serve requests. This should
be fixed alongside the pricing work.

| Model | Status | Action |
|---|---|---|
| gemini-1.5-pro-002 | RETIRED — returns 404 | Remove from GEMINI_MODEL_SPECS / list_model_names |
| gemini-1.5-flash-002 | RETIRED — returns 404 | Remove |
| gemini-2.0-flash / -lite (+ -001) | DEPRECATED — shutdown date (2026-06-01) has passed | Verify availability; likely remove |
| imagen-4.0-generate-001 | DEPRECATED — shutdown 2026-08-17 | Plan migration to Gemini 3.x image models |
| gpt-5.4, gpt-5.1-codex-max | Available, not yet in tables | Add (the deferred #4 work — now with high-confidence official rates) |

## API recommendation

Replace the single float with a structured, per-modality, sourced pricing
descriptor, stored separately from the provider classes.

```python
from datetime import date
from decimal import Decimal
from typing import Literal

class AITokenRates(BaseModel):
    input_per_1m: Decimal
    output_per_1m: Decimal | None = None       # None for embeddings
    cached_input_per_1m: Decimal | None = None

class AIPricingTier(BaseModel):
    label: str                                  # "context>200k", "quality:high", "1080p"
    rates: AITokenRates | None = None
    per_unit_usd: Decimal | None = None

class AIModelPricing(BaseModel):
    currency: str = "USD"
    effective_date: date
    source: str                                 # official pricing URL
    confidence: Literal["high", "medium", "low"]
    unit: Literal["token", "character", "image", "second", "minute"]
    token_rates: AITokenRates | None = None      # unit == token/character
    per_image_usd: Decimal | None = None         # unit == image
    per_second_usd: Decimal | None = None        # unit == second
    per_minute_usd: Decimal | None = None        # unit == minute
    tiers: list[AIPricingTier] | None = None     # context or quality/size tiers
    notes: str | None = None
```

Design points:

- **Decimal, not float.** These are money; float rounding is unacceptable in a
  cost ledger.
- **Store pricing outside the model classes.** Model classes describe behavior;
  a rate table is data that changes on the provider's schedule. Put it in a
  dedicated versioned pricing registry keyed by `(provider, model)`, with
  `effective_date` and `source` per entry so cost is auditable. This is the
  single most important change for finops.
- **Expose it through capabilities.** Add `pricing: AIModelPricing | None` to
  the existing capabilities descriptors, resolved by the same `for_model()`
  path already in place. Callers read `client.capabilities.pricing`.
- **Compute cost from real usage, not a blend.** Replace
  `calculate_cost(num_tokens)` with a function that takes the provider-reported
  input and output token counts and applies the split rates. The observability
  runtime already captures these (`provider_prompt_tokens`,
  `provider_completion_tokens`), so the finops middleware can hook `after_call`,
  multiply actual usage by real rates, and emit an auditable cost event.

Keep the old `price_per_1k_tokens` / `calculate_cost` as thin deprecated
shims (blended = input+output midpoint) for one release so callers migrate
without a break.

## Finops connection (next, per backlog)

The planned financial-ops observability middleware sits on the existing
observability lifecycle: on `after_call` it reads the real token counts from the
result summary, looks up `AIModelPricing` for the model, and emits a cost event
carrying `{model, input_tokens, output_tokens, cached_tokens, usd_cost,
effective_date, source}`. The pricing descriptor above is exactly the input that
layer needs; building it first is why this work comes before the middleware.

## Sources

- OpenAI: developers.openai.com/api/docs/pricing and per-model `/models/<name>` pages
- Google: ai.google.dev/gemini-api/docs/pricing, cloud.google.com/vertex-ai/generative-ai/pricing, ai.google.dev/gemini-api/docs/deprecations
- AWS: aws.amazon.com/bedrock/pricing, aws.amazon.com/nova/pricing (tables are JS-rendered; AWS figures cross-checked via search, med confidence on embeddings/images/video)
- Anthropic: platform.claude.com/docs/en/about-claude/models/overview and platform.claude.com/docs/en/pricing (high confidence)
- Azure: azure.microsoft.com/en-us/pricing/details/cognitive-services/speech-services (page timed out; search-derived, med confidence)
- ElevenLabs: elevenlabs.io/pricing/api (high confidence)
