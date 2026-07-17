# UniInfer Proxy Architecture (Brief)

This document describes the current `proxy_app` layout and where to add changes.

## Naming (read this first — `uniioai_proxy` ≠ the module anymore)

`uniioai` is the **product**; `proxy_app` is the **module**. The app module was
renamed (`uniioai_proxy.py` → `proxy_app.py`); the user-facing command and log
channel deliberately keep the product name.

| What | Name | Notes |
|------|------|-------|
| Product / service | **UniIOAI proxy** | the OpenAI-compatible proxy product |
| App module | `uniinfer/proxy_app.py` | FastAPI app + `main()` entry point. **Renamed from `uniioai_proxy.py`** (commit `344b68f`). |
| Console command | `uniioai-proxy` | `[project.scripts]` binary → `uniinfer.proxy_app:main` (name unchanged) |
| uvicorn target | `uniinfer.proxy_app:app` | for `uvicorn` / `--reload` |
| Logger channel | `"uniioai_proxy"` | stable log channel (intentionally kept) |
| Log file | `logs/uniioai_proxy.log` | stable log file name (intentionally kept) |

`uniioai_proxy.py` / `uniinfer.uniioai_proxy` **no longer exist** — don't import
or reference them. Tests import `from uniinfer.proxy_app import app`.

## Entry Point

- `uniinfer/proxy_app.py`
  - FastAPI app bootstrap
  - middleware (logging, request size, CORS, rate limiter wiring)
  - shared helpers (e.g. `parse_provider_model`)
  - router registration

Keep this file thin. Prefer adding endpoint logic in routers/services.

## Routers

- `uniinfer/proxy_routers/models.py`
  - `/v1/models`
  - `/v1/system/update-models`
  - `/v1/providers`
  - `/v1/models/{provider}`
  - embedding model/provider listing
  - `/v1/system/info`

- `uniinfer/proxy_routers/media.py`
  - image model listing + image generation
  - TTS (`/v1/audio/speech`)
  - STT (`/v1/audio/transcriptions`)

- `uniinfer/proxy_routers/chat.py`
  - chat completions (`/v1/chat/completions`)
  - embeddings (`/v1/embeddings`)

## Services

- `uniinfer/proxy_services/models_registry.py`
  - `models.txt` refresh/staleness parsing logic

- `uniinfer/proxy_services/streaming.py`
  - SSE streaming shaping
  - `thinking -> reasoning_content` normalization
  - strict/non-strict compatibility helpers

## Schemas

- `uniinfer/proxy_schemas/chat.py`
  - request/response schemas for chat + embeddings
  - streaming chunk models

## Change Guidelines

1. **Add new endpoint** in a router module, not `proxy_app.py`.
2. **Move reusable logic** into `proxy_services/*`.
3. **Put request/response models** into `proxy_schemas/*`.
4. Keep `proxy_app.py` focused on app wiring.
5. Run before commit:
   - `uv run ruff check .`
   - `uv run python -m py_compile uniinfer/proxy_app.py`
   - smoke test changed endpoints via curl/webdemo.

## Compatibility Notes

- Current default is non-strict OpenAI mode for TU/vLLM thinking visibility.
- Proxy emits/normalizes `reasoning_content` for Pi compatibility.
- Preserve this behavior unless explicitly changed.
