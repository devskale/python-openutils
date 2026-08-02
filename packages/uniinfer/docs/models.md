# Models Catalog

Generated model catalog with rich metadata for all supported providers.

## Architecture

```
uniinfer/models/
  models.json            # Generated catalog (DO NOT edit manually)
  type_overrides.json    # Curated model type assignments (edit this to fix types)
  _model_history.json    # first_seen/last_seen tracking (auto-managed)
  _speed_results.json    # Speed test results (auto-managed by --speedtest CLI)
scripts/
  generate_models.py     # Regenerates models.json from live APIs + models.dev
```

## No Hardcoded Model Lists

When a provider's API key is missing or the API call fails, `list_models()` **must return `[]`** — never fall back to a hardcoded list. Hardcoded lists go stale.

```python
# ❌ Wrong
if api_key is None:
    return [ModelInfo(id=m) for m in ["Meta-Llama-3.1-8B-Instruct", "sambastudio-7b"]]

# ✅ Correct
if api_key is None:
    return []
```

## Type Assignment

Three-layer priority:

| Layer | Source | Priority |
|-------|--------|----------|
| `type_overrides.json` | Curated — always wins | **Highest** |
| `ModelInfo.derive_type()` | Audio modalities + name patterns | Medium |
| Provider factory kind | Default from registration | Lowest |

### derive_type() rules

- **stt**: `modalities.output == ["text"]` + input is only audio, or ID contains `whisper`
- **tts**: `modalities.output == ["audio"]` + input is only text, or ID contains `kokoro`/`piper-`
- **chat**: everything else (default)

### type_overrides.json

```json
{
  "models": {
    "e5-mistral-7b": "embed",
    "kokoro": "tts",
    "whisper-large": "stt",
    "dall-e-3": "image"
  }
}
```

Matches by bare model ID (no provider prefix). Add entries as you discover wrong types.

## Embedding Dimensions

Sources for the `dimensions` field:
- **Ollama**: `POST /api/show` returns `model_info.{arch}.embedding_length` — free, no embed call
- **models.dev**: `limit.output` mapped to dimensions during merge
- **Other providers**: from `/v1/models` response or models.dev merge

## Provider Metadata Richness

| Provider | Context | Max Output | Modalities | Capabilities | Cost | Deprecation |
|----------|:-------:|:----------:|:----------:|:------------:|:----:|:-----------:|
| **Mistral** | ✅ | ✅ | ✅ | ✅ reasoning, vision, tools, audio, ocr | — | ✅ dates + replacements |
| **Anthropic** | ✅ | ✅ | ✅ | ✅ thinking, vision, pdf, code_exec, tools | — | — |
| **OpenRouter** | ✅ | ✅ | ✅ | ✅ tools, reasoning, structured_outputs | ✅ | — |
| **Gemini** | ✅ | ✅ | — | ✅ thinking | — | — |
| **Moonshot** | ✅ | — | ✅ | ✅ vision | — | — |
| **SambaNova** | ✅ | ✅ | — | — | ✅ | — |
| **AI21** | ✅ | ✅ | — | — | ✅ | — |
| **Arli** | ✅ (14/130) | — | ✅ | ✅ reasoning, vision | — | — |
| **Pollinations** | — | — | ✅ | ✅ reasoning, vision, tools | — | — |
| **Groq** | ✅ | — | — | — | — | — |
| **Cohere** | ✅ | — | — | — | — | — |
| **Ollama** | — | — | — | — | — | ✅ dimensions via `/api/show` |
| OpenAI, NGC, StepFun, Upstage, InternLM, Chutes, Cloudflare, HuggingFace, TU | bare (id only) | | | | | |

## Metadata reliability & access truth

Catalog metadata is **enriched, not verified**. Three caveats (generalized from the
Arli case, web-grounded) — treat the catalog as a hint, not ground truth:

### 1. Advertised context/output ≠ served reality

Providers report the **model's native maximum**, which the served endpoint often
caps lower (quantization, fewer GPUs, per-key/trial limits). The catalog carries
the advertised figure → `context_window` / `max_output` are **upper bounds, not
guarantees**.

> **Arli / Qwen3.5-27B-Derestricted** — Arli's page and the HF model card both
> state *“Context Length: 262,144 natively, extensible to 1,010,000”* (256k, on
> 8-GPU tensor-parallel). Arli **advertises 262,144**, so the catalog reports
> 262,144 — but Arli *serves* it quantized on fewer GPUs, and trial/limited keys
> hit a lower effective cap. The advertised 256k overstates what a real request
> gets. Fix known cases via `model_overrides.json` (e.g. set the real served ctx).

### 2. `access` (free/paid) is a *cost/naming heuristic*, not key reachability

`ModelInfo.access` is derived from **cost** (`cost.input == 0` → free) or a
provider **naming heuristic** — it describes *pricing*, not whether **your key**
can reach the model. The two can be **inverted**:

> **Arli** — `access="paid" if id.startswith("(TRIAL)") else "free"` (see
> `providers/arli.py`). So the `(TRIAL) …`-prefixed models are tagged `paid`, the
> bare ones `free` — yet the bare `free` models are the dead/crippled trials
> (`Fastest` has `context_window: 0`), and the `(TRIAL)`-prefixed (tagged `paid`)
> are what a trial key actually reaches. A “free” tag here ≈ *unusable*; a “paid”
> tag ≈ *trial-accessible*. **Do not filter “free” to mean “my key works.”**
>
> **Kilo** — cost can be **actively wrong**, not just heuristic. `google/lyria-3-*`
> reports `pricing.prompt='0'` + `pricing.completion='0'` yet serves **paid**
> (“Add credits to continue”). The gateway's explicit **`isFree` flag** is the
> only reliable signal — all 11 `isFree=true` models serve free (verified by
> actual completions; `cohere/north-mini-code:free` rate-limits but is free), and
> Kilo's `access` is now derived from `isFree`, **not** cost. The `:free` id
> suffix is redundant within the gateway (all such ids are `isFree=true`).
> **The public site (kilo.ai/models) is stale** — it still lists expired-free
> models (`tencent/hy3:free`, `inclusionai/ling-2.6-flash:free`,
> `google/gemma-4-26b-a4b-it:free`, `nex-agi/nex-n2-pro:free`, `baidu/cobuddy:free`)
> whose free periods ended; the gateway rejects them with *“free period ended /
> transitioned to paid.”* **Trust the gateway, not the site.**

### 3. Trial/free-tier reachability is an *irregular named subset*, not “all free-cost models”

Providers expose trial-accessible models under a **naming convention** — not by
cost:
- **Arli**: `(TRIAL) …` prefix
- **OpenCode/Zen**: `-free` / `big-pickle` / `mimo-v2.5-free` suffixes
- **Kilo / OpenRouter**: `:free` suffix (redundant within the Kilo gateway —
  coincides with `isFree=true`; the public site's `:free`/`-free` listings are
  stale, see above)

So `access == "free"` won’t reliably list “what my trial key reaches.” Declare a
key’s real reachable set with the **instance `access.models` selective list**
(see [provider-instances-design.md §Key management](provider-instances-design.md));
filter-by-`access` is a coarse hint, not a routing key.

### Takeaways
- `context_window` / `max_output`: upper bounds — verify empirically, override
  (`model_overrides.json`) where you know the served figure.
- `access`: cost-flavored hint, **never** a reachability guarantee.
- Trial reachability: an irregular named subset → model it per-instance
  (`access.models`), not via a global free/paid filter.

## Generation Pipeline

```bash
uv run python3 scripts/generate_models.py
```

1. Calls `list_models()` on all installed providers
2. Applies type overrides → derive_type() fallback
3. Merges enrichment from models.dev (context, cost, modalities, capabilities, dimensions)
4. Probes Ollama embed models via `/api/show`
5. Tracks `first_seen`/`last_seen` via `_model_history.json`

### models.dev Mapping

| uniinfer | models.dev |
|----------|-----------|
| openai | openai |
| anthropic | anthropic |
| gemini | google |
| mistral | mistral |
| groq | groq |
| cohere | cohere |
| openrouter | openrouter |
| ollama | ollama-cloud |
| chutes | chutes |
| cloudflare | cloudflare-ai-gateway |
| sambanova | nova |
| ngc | nvidia |
| moonshot | moonshotai |
| upstage | upstage |
| stepfun | stepfun |
| huggingface | huggingface |
| zai, zai-code | zai |

Unmapped: arli, pollinations, internlm, tu, ai21.

Matching: exact ID first, then fuzzy (strip date suffixes, `@cf/` prefixes). Live API always wins over models.dev.

## Model Lifecycle

| State | Condition | In models.json? |
|-------|-----------|----------------|
| `fresh` | `last_seen` == today | ✅ |
| `stale` | `last_seen` < today, < 90 days | ✅ (with `days_since_seen`) |
| `pruned` | `last_seen` > 90 days | ❌ kept in history only |

Deprecation sources: Mistral API (`deprecation` date + replacement), disappeared models.
