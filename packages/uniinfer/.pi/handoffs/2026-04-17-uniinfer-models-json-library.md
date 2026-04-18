# Design models.json library for uniinfer with provider model catalog enriched from models.dev

## Slug
`uniinfer-models-json-library`

## 1. Primary Request and Intent
The user wants to create a `models.json` library for the **uniinfer** package that catalogs past and active models across all supported providers. Key requirements:
- Maintain a static `models.json` with past and active models of all supported providers
- Include **model type** classification (LLM, embed, VLM, TTS, STT) where possible
- Include **context window size** where available
- Use [models.dev](https://github.com/anomalyco/models.dev) as a primary data source

The session resulted in a **complete design proposal** but **no implementation was started**. The user has not yet confirmed the approach or answered 5 open questions.

## 2. Key Technical Concepts
- **uniinfer**: Unified LLM inference Python package with 23 chat providers, 2 embedding providers, 2 TTS, 1 STT
- **models.dev**: Open-source AI model database (1.75 MB API JSON, 111 providers, 700+ models) with rich metadata
- **models.dev API**: `https://models.dev/api.json` — returns `{provider_id: {id, name, models: {model_id: {...}}}}`
- **Current `list_models()` pattern**: Each provider has a classmethod that hits the live API, returns `list[str]` — no metadata
- **Current proxy caching**: `models_registry.py` caches to flat `models.txt` with zero enrichment
- **Provider base classes**: `OpenAICompatibleChatProvider`, `AnthropicCompatibleProvider` — many providers inherit from these
- **Type derivation heuristic**: From models.dev `modalities` + `family` fields (embed→"embed", audio output→"tts", image input→"vlm"/"llm" with vision)

## 3. Files and Code Sections

### `uniinfer/core.py`
- **Why important**: Defines all base classes (`ChatProvider`, `EmbeddingProvider`, `TTSProvider`, `STTProvider`) and data types
- `list_models()` classmethod returns `list[str]` — the new models.json is additive, no breaking change needed
- Key classes: `ChatCompletionRequest`, `ChatCompletionResponse`, `ChatMessage`, `EmbeddingRequest`, `EmbeddingResponse`, `TTSRequest`, `TTSResponse`, `STTRequest`, `STTResponse`

### `uniinfer/__init__.py`
- **Why important**: Registers all providers in `ProviderFactory` and `EmbeddingProviderFactory`
- 23 registered chat providers, 2 embedding providers
- Full provider list: mistral, anthropic, minimax, openai, ollama, openrouter, arli, internlm, stepfun, sambanova, upstage, ngc, cloudflare, chutes, pollinations, zai, zai-code, tu + optional: huggingface, cohere, moonshot, groq, ai21, gemini

### `uniinfer/providers/__init__.py`
- **Why important**: All provider imports and `__all__` exports
- Optional providers guarded by try/except (gemini, tu, huggingface, cohere, moonshot, groq, ai21)

### `uniinfer/proxy_services/models_registry.py`
- **Why important**: The existing model caching mechanism that would be replaced/enhanced
- Currently writes flat `models.txt` (parsed with regex), has `PREDEFINED_MODELS` fallback
- Uses subprocess to call `uniinfer -l --list-models` for refresh
- Has staleness check via file mtime + configurable `REFETCHTIME` env var

### `uniinfer/providers/openai_compatible.py`
- **Why important**: Base class for many providers (openai, mistral, openrouter, arli, internlm, stepfun, sambanova, upstage, ngc, cloudflare, chutes, pollinations, minimax)
- Does NOT have its own `list_models()` — subclasses implement individually
- Key constants: `BASE_URL`, `PROVIDER_ID`, `ERROR_PROVIDER_NAME`, `DEFAULT_MODEL`

### `uniinfer/providers/anthropic_compatible.py`
- **Why important**: Base for anthropic, minimax providers
- Has `list_models()` that uses `anthropic` SDK `client.models.list()`
- Requires API key, uses credgoo as fallback

### models.dev API JSON structure (downloaded to `/tmp/models_dev_api.json` — may be stale)
- **Why important**: The primary enrichment data source
- Structure:
```json
{
  "provider_id": {
    "id": "...", "env": [...], "npm": "...", "api": "...", "name": "...", "doc": "...",
    "models": {
      "model_id": {
        "id": "gpt-5.1-codex-max",
        "name": "GPT-5.1 Codex Max",
        "family": "gpt-codex",
        "attachment": true,
        "reasoning": true,
        "tool_call": true,
        "structured_output": true,
        "temperature": false,
        "knowledge": "2024-09-30",
        "release_date": "2025-11-13",
        "last_updated": "2025-11-13",
        "modalities": { "input": ["text", "image"], "output": ["text"] },
        "open_weights": false,
        "cost": { "input": 1.25, "output": 10, "cache_read": 0.125 },
        "limit": { "context": 400000, "input": 272000, "output": 128000 },
        "status": "deprecated"
      }
    }
  }
}
```

### Provider Mapping Table (uniinfer ↔ models.dev)
| uniinfer ID | models.dev provider | Models | Status |
|---|---|---|---|
| `openai` | `openai` | 51 | ✅ Full |
| `anthropic` | `anthropic` | 23 | ✅ Full |
| `gemini` | `google` | 37 | ✅ (rename) |
| `mistral` | `mistral` | 27 | ✅ Full |
| `groq` | `groq` | 27 | ✅ Full |
| `cohere` | `cohere` | 12 | ✅ Full |
| `openrouter` | `openrouter` | 171 | ✅ Full |
| `ollama` | `ollama-cloud` | 36 | ⚠️ Partial |
| `chutes` | `chutes` | 69 | ✅ Full |
| `cloudflare` | `cloudflare-workers-ai` | 7 | ✅ Full |
| `minimax` | `minimax` | 6 | ✅ Full |
| `upstage` | `upstage` | 3 | ✅ Full |
| `stepfun` | `stepfun` | 4 | ✅ Full |
| `moonshot` | `moonshotai` | 6 | ✅ (rename) |
| `huggingface` | `huggingface` | 22 | ✅ Full |
| `zai`/`zai-code` | `zai` | 13 | ✅ (rename) |
| `sambanova` | `nova` | 2 | ⚠️ Only 2 |
| `ngc` | `nvidia` | 76 | ⚠️ Different IDs |
| `arli` | — | — | ❌ Not in models.dev |
| `pollinations` | — | — | ❌ Not in models.dev |
| `internlm` | — | — | ❌ Not in models.dev |
| `tu` | — | — | ❌ Internal TU Wien |
| `ai21` | — | — | ❌ Not in models.dev |

## 4. Problem Solving
- **Fetched and parsed models.dev API** — had to use curl with retry due to large response (1.75 MB). The JSON was initially truncated with a timeout; resolved by saving to file first.
- **Derived type classification heuristic**: `family` contains "embed" → embed; `modalities.output` only `["audio"]` → TTS; `modalities.input` contains `["audio"]` with output `["text"]` → STT; `image` in input → VLM/LLM with vision flag; default → LLM.
- **Identified the status field** in models.dev (alpha/beta/deprecated) for tracking past vs active models.

## 5. Pending Tasks
**No implementation has been started.** The proposal was delivered but the user has not yet:
1. Confirmed the approach
2. Answered the 5 open questions:
   - Q1: Ship full JSON with package, or fetch on demand?
   - Q2: Manual curation for arli/pollinations/internlm/tu/ai21, or best-effort live API?
   - Q3: Independent versioning for models.json, or pinned to uniinfer releases?
   - Q4: VLM as separate type, or `"llm"` with `capabilities.vision: true`?
   - Q5: Store cross-provider model IDs (e.g., `google/gemma-3-27b-it`) for lookup?

## 6. Current Work
The session concluded with a **complete design proposal** for the models.json library. The proposal includes:
- **Proposed `models.json` schema** with type, status, context_window, capabilities, cost, modalities
- **Generator script design** (`scripts/generate_models.py`) — fetches models.dev, maps providers, derives types
- **Runtime API** (`uniinfer/models/`) — `ModelInfo` dataclass + `get_all_models()`, `get_provider_models()`, `get_model_info()`, `find_models()`
- **File layout**: `uniinfer/models/{__init__.py, models.json, overrides.json, loader.py, types.py}`
- **Integration points**: models_registry.py, CLI, strategies

### Proposed models.json schema:
```jsonc
{
  "_meta": {
    "version": "1.0.0",
    "generated": "2026-04-17T...",
    "source": "models.dev + provider APIs + manual",
    "uniinfer_version": "0.3.x"
  },
  "providers": {
    "openai": {
      "name": "OpenAI",
      "models_dev_id": "openai",
      "models": {
        "gpt-4o": {
          "name": "GPT-4o",
          "type": "llm",
          "status": "active",
          "context_window": 128000,
          "max_output": 16384,
          "modalities": { "input": ["text","image"], "output": ["text"] },
          "capabilities": { "reasoning": false, "tool_call": true, "vision": true, "structured_output": true },
          "cost": { "input": 2.5, "output": 10.0 },
          "release_date": "2024-05-13",
          "knowledge_cutoff": "2023-10",
          "open_weights": false
        }
      }
    }
  }
}
```

### Proposed runtime API:
```python
@dataclass
class ModelInfo:
    id: str
    provider: str
    name: str
    type: str           # "llm" | "embed" | "vlm" | "tts" | "stt"
    status: str         # "active" | "deprecated" | "alpha" | "beta"
    context_window: int | None
    max_output: int | None
    modalities: dict
    capabilities: dict
    cost: dict | None
    release_date: str | None

def get_all_models(status="active") -> dict[str, list[ModelInfo]]: ...
def get_provider_models(provider: str, status="active") -> list[ModelInfo]: ...
def get_model_info(provider: str, model: str) -> ModelInfo | None: ...
def find_models(type: str = None, min_context: int = None, capability: str = None) -> list[ModelInfo]: ...
```

### Proposed file layout:
```
uniinfer/
  models/
    __init__.py          # re-exports ModelInfo, get_*
    models.json          # the big catalogue (generated)
    overrides.json       # manual additions/corrections
    loader.py            # load models.json, merge overrides
    types.py             # ModelInfo dataclass
scripts/
  generate_models.py     # generator script
```

## 7. Optional Next Step
**Wait for user confirmation** on the design and answers to the 5 open questions before implementing. Once confirmed, the implementation order would be:
1. Create `uniinfer/models/types.py` with `ModelInfo` dataclass
2. Create `scripts/generate_models.py` to build the JSON from models.dev + live APIs
3. Run the generator to produce `uniinfer/models/models.json`
4. Create `uniinfer/models/loader.py` for runtime access
5. Create `uniinfer/models/__init__.py` with public API
6. Integrate with `models_registry.py` and CLI

## Key Paths
- Project root: `/Users/johannwaldherr/code/kontext.one/python-openutils/packages/uniinfer`
- Cached models.dev API: `/tmp/models_dev_api.json` (may be stale, re-fetch from `https://models.dev/api.json`)
