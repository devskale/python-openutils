# Models Catalog

UniInfer maintains a model catalog with rich metadata for all supported providers.

## Architecture

```
uniinfer/models/
  models.json          # Generated catalog (DO NOT edit manually)
  type_overrides.json  # Curated model type assignments (edit this)
scripts/
  generate_models.py   # Regenerates models.json from live APIs
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
    modalities: dict | None        # {"input": ["text","image"], "output": ["text"]}
    capabilities: dict | None      # {"reasoning": true, "vision": true, "tool_call": true}
    cost: dict | None              # {"input": 2.5, "output": 10.0} per 1M tokens USD
    owned_by: str | None           # Provider or organisation
    created: int | None            # Unix timestamp
    raw: dict | None               # Full raw API response
```

Backward compat: `str(m)` returns `m.id`, `m == "model-id"` works, hashable.

## Provider Metadata Richness

| Provider | Context | Max Output | Modalities | Capabilities | Cost |
|----------|---------|------------|------------|--------------|------|
| **Anthropic** | ✅ | ✅ | ✅ | ✅ thinking, vision, pdf, code_exec, tools | — |
| **Mistral** | ✅ | ✅ | ✅ | ✅ reasoning, vision, tools, audio, ocr | — |
| **Gemini** | ✅ | ✅ | — | ✅ thinking | — |
| **OpenRouter** | ✅ | ✅ | ✅ | ✅ tools, reasoning, structured_outputs | ✅ |
| **Moonshot** | ✅ | — | ✅ | ✅ vision | — |
| **Groq** | ✅ | — | — | — | — |
| **Cohere** | ✅ | — | — | — | — |
| **Arli** | ✅ (14/130) | — | ✅ | ✅ reasoning, vision | — |
| **SambaNova** | ✅ | ✅ | — | — | ✅ |
| **AI21** | ✅ | ✅ | — | — | ✅ |
| **Pollinations** | — | — | ✅ | ✅ reasoning, vision, tools | — |
| OpenAI, NGC, Stepfun, Upstage, InternLM, TU, Chutes, Cloudflare, Ollama, BigModel, HuggingFace | bare (id only) |

## Generating models.json

```bash
cd packages/uniinfer
uv run python3 scripts/generate_models.py
```

Calls `list_models()` on all installed providers, applies type overrides + derive_type, writes `uniinfer/models/models.json`.

## Proxy Integration

The proxy (`/v1/models`) serves from `models.json` with auto-refresh:

- `GET /v1/models` → reads models.json cache, regenerates if stale (>24h via `REFETCHTIME` env)
- `GET /v1/models/{provider}` → live API call, then updates models.json cache for that provider
- Both return rich metadata (context_window, capabilities, modalities, cost, type)

`models.txt` is deprecated — `models.json` is the single source of truth.

## Provider ↔ models.dev Mapping

19 of 24 uniinfer providers map to models.dev for enrichment (cost, release_date, knowledge_cutoff). Unmapped: arli, pollinations, internlm, tu, ai21.

See `scripts/explore_models_dev.py` and `.pi/handoffs/2026-04-17-uniinfer-models-json-library.md` for the full mapping table and enrichment design.
