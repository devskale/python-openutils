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
