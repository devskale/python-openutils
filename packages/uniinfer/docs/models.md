# Models Catalog

UniInfer maintains a model catalog with rich metadata for all supported providers.

## Architecture

```
uniinfer/models/
  models.json            # Generated catalog (DO NOT edit manually)
  type_overrides.json    # Curated model type assignments (edit this)
  _model_history.json    # first_seen tracking (auto-managed by generator)
scripts/
  generate_models.py     # Regenerates models.json from live APIs + models.dev
  explore_models_dev.py  # Explore models.dev mapping and data
  _models_dev_cache.json # Cached models.dev API response
```

## Model Types

Each model has a `type` field: `chat | embed | tts | stt | image`.

Type assignment follows a three-layer priority:

| Layer | Source | Priority |
|-------|--------|----------|
| `type_overrides.json` | Curated DB — always wins | **Highest** |
| `ModelInfo.derive_type()` | Audio modalities + name patterns (whisper, kokoro/piper) | Medium |
| Provider factory kind | Default from provider registration (chat/embed/tts/stt) | Lowest |

### derive_type() rules (conservative — only catches unambiguous cases)

- **stt**: `modalities.output == ["text"]` and `modalities.input` contains only `audio`
- **tts**: `modalities.output == ["audio"]` and `modalities.input` contains only `text`
- **stt**: model ID contains `whisper`
- **tts**: model ID contains `kokoro` or `piper-`
- **chat**: everything else (default)

### type_overrides.json

Curated type assignments that override all other sources. Add entries here for any model that needs a non-default type. Format:

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

Matches by bare model ID (no provider prefix). Add entries as you discover models with wrong types — this is the knowledge base that grows with the project.

## ModelInfo Dataclass

All `list_models()` methods return `list[ModelInfo]` (not `list[str]`).

```python
@dataclass
class ModelInfo:
    id: str                        # Unique model ID within provider
    name: str | None               # Display name
    type: str                      # "chat" | "embed" | "tts" | "stt" | "image"
    status: str                    # "active" | "deprecated" | "alpha" | "beta"
    context_window: int | None     # Max context tokens
    max_output: int | None         # Max output tokens
    dimensions: int | None         # Embedding vector dimensions (embed models only)
    modalities: dict | None        # {"input": ["text","image"], "output": ["text"]}
    capabilities: dict | None      # {"reasoning": true, "vision": true, "tool_call": true}
    cost: dict | None              # {"input": 2.5, "output": 10.0} per 1M tokens USD
    owned_by: str | None           # Provider or organisation
    created: int | None            # Unix timestamp
    first_seen: str | None         # "YYYY-MM-DD" — first appearance in our catalog
    deprecation_date: str | None   # ISO date from provider API (e.g. "2026-05-31T12:00:00Z")
    deprecation_replacement: str | None  # Suggested replacement model ID
    raw: dict | None               # Full raw API response
```

Backward compat: `str(m)` returns `m.id`, `m == "model-id"` works, hashable.

### Embedding-Specific Metadata

Embed models have a `dimensions` field (vector size). Sources:
- **TU**: probed live via `POST /v1/embeddings` during generation
- **models.dev**: `limit.output` mapped to dimensions during merge
- **Other providers**: not available from `/v1/models`, would need per-provider probing

## Provider Metadata Richness

| Provider | Context | Max Output | Modalities | Capabilities | Cost | Dimensions |
|----------|---------|------------|------------|--------------|------|------------|
| **Anthropic** | ✅ | ✅ | ✅ | ✅ thinking, vision, pdf, code_exec, tools | — | — |
| **Mistral** | ✅ | ✅ | ✅ | ✅ reasoning, vision, tools, audio, ocr + deprecation dates | — | — |
| **Gemini** | ✅ | ✅ | — | ✅ thinking | — | — |
| **OpenRouter** | ✅ | ✅ | ✅ | ✅ tools, reasoning, structured_outputs | ✅ | — |
| **Moonshot** | ✅ | — | ✅ | ✅ vision | — | — |
| **Groq** | ✅ | — | — | — | — | — |
| **Cohere** | ✅ | — | — | — | — | — |
| **Arli** | ✅ (14/130) | — | ✅ | ✅ reasoning, vision | — | — |
| **SambaNova** | ✅ | ✅ | — | — | ✅ | — |
| **AI21** | ✅ | ✅ | — | — | ✅ | — |
| **Pollinations** | — | — | ✅ | ✅ reasoning, vision, tools | — | — |
| **TU** | — | — | — | — | — | ✅ (probed live) |
| OpenAI, NGC, Stepfun, Upstage, InternLM, Chutes, Cloudflare, Ollama, BigModel, HuggingFace | bare (id only) | | | | | |

## Generating models.json

```bash
cd packages/uniinfer
uv run python3 scripts/generate_models.py
```

Pipeline:
1. Calls `list_models()` on all installed providers
2. Applies type overrides from `type_overrides.json`, then `derive_type()` fallback
3. Merges enrichment from models.dev (context, cost, modalities, capabilities, dimensions, release_date, knowledge_cutoff)
4. Probes TU embed models for dimensions (live `POST /v1/embeddings`)
5. Tracks `first_seen` via `_model_history.json`, reports new and disappeared models

### models.dev Merge

19 providers map to models.dev for enrichment. Matching strategy:
- **Exact ID** first (e.g. `gpt-4o` → `gpt-4o`)
- **Fuzzy** second: strips date suffixes (`gpt-5.4-nano-2026-03-17` → `gpt-5.4-nano`) and `@cf/` prefixes (`@cf/baai/bge-m3` → `baai/bge-m3`)
- **Zero values skipped**: `limit.context: 0` is ignored (e.g. image models)
- **Priority**: live API always wins over models.dev

Provider mapping (uniinfer → models.dev):

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
| minimax | minimax |
| upstage | upstage |
| stepfun | stepfun |
| moonshot | moonshotai |
| huggingface | huggingface |
| zai, zai-code | zai |
| sambanova | nova |
| ngc | nvidia |

Unmapped: arli, pollinations, internlm, tu, ai21.

## Date Tracking

### first_seen

Every model gets a `first_seen` date (`YYYY-MM-DD`). Tracked in `_model_history.json` — a flat `{provider/model_id: date}` dict. New models get today's date on first appearance. The file persists across generations.

### Deprecation

Two sources of deprecation info:
- **Provider API**: Mistral exposes `deprecation` date + `deprecation_replacement_model` in `/v1/models`
- **Disappeared models**: models present in `_model_history.json` but missing from current generation are reported as deprecated and removed from history

## CLI Commands

```bash
# List models first seen in the last 7 days (default)
uniinfer --new-models
uniinfer --new-models 30    # last 30 days

# List all deprecated models with deprecation date and replacement
uniinfer --deprecated-models
```

## Proxy Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /v1/models` | All models from cache (auto-refresh if stale) |
| `GET /v1/models/{provider}` | Live provider models (updates cache) |
| `GET /v1/models/new?days=7` | Models first seen in last N days |
| `GET /v1/models/deprecated` | Deprecated models with deprecation info |

`models.txt` is deprecated — `models.json` is the single source of truth.
