# Catalog Endpoint (`/v1/catalog`)

A **public** (no-auth) endpoint that serves the raw nested `models.json` catalog
regenerated daily at 04:00 UTC by `uniioai-models-refresh.timer`.

Unlike `/v1/models` (which flattens into an OpenAI-style list and merges
overrides), `/v1/catalog` returns the catalog in its native nested shape so
consumers can download the full file or just the providers they care about.

## Usage

```
GET /v1/catalog
GET /v1/catalog?providers=openai
GET /v1/catalog?providers=openai,gemini,groq
GET /v1/catalog?providers=openai&download=1
```

### Query params

| Param | Default | Description |
|-------|---------|-------------|
| `providers` | _(all)_ | Comma-separated provider IDs to include |
| `download` | `false` | If `true`, sets `Content-Disposition: attachment` for direct download |

## Response

Native nested catalog:

```json
{
  "_meta": {
    "version": "1.0.0",
    "generated": "2026-06-24T04:00:52+00:00",
    "source": "live provider APIs + models.dev",
    "total_models": 993,
    "total_providers": 20,
    "filtered": false
  },
  "providers": {
    "openai": {
      "provider_class": "OpenAIProvider",
      "kind": "chat",
      "models": [
        { "id": "gpt-4o", "context_window": 128000, "cost": { ... } }
      ]
    }
  }
}
```

When `providers` is set, `total_models`/`total_providers` are recomputed for
the filtered subset and `filtered: true`.

## Examples

Download a single provider's catalog:

```bash
curl -sOJ "https://localhost:8123/v1/catalog?providers=openai&download=1"
# → models-openai.json
```

Fetch multiple providers inlined:

```bash
curl -s "https://localhost:8123/v1/catalog?providers=openai,gemini"
```

Fetch from Python:

```python
import requests
catalog = requests.get(
    "https://localhost:8123/v1/catalog",
    params={"providers": "openai,gemini"},
).json()
for pid, pdata in catalog["providers"].items():
    print(pid, len(pdata["models"]))
```

## Webdemo

In the Models browser (`/webdemo`), the **⬇ JSON** button next to the provider
dropdown downloads the catalog filtered by the current provider selection.

## Untracked by design

`models.json`, `_stale_models.json`, and `_model_history.json` are runtime
caches — they are gitignored and regenerated daily on the server, never
committed. `/v1/catalog` is the canonical way to fetch them over the web.
