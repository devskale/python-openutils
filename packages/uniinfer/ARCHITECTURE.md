# UniInfer Proxy Architecture (Brief)

This document describes the current `uniioai_proxy` layout and where to add changes.

## Entry Point

- `uniinfer/uniioai_proxy.py`
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

1. **Add new endpoint** in a router module, not `uniioai_proxy.py`.
2. **Move reusable logic** into `proxy_services/*`.
3. **Put request/response models** into `proxy_schemas/*`.
4. Keep `uniioai_proxy.py` focused on app wiring.
5. Run before commit:
   - `uv run ruff check .`
   - `uv run python -m py_compile uniinfer/uniioai_proxy.py`
   - smoke test changed endpoints via curl/webdemo.

## Compatibility Notes

- Current default is non-strict OpenAI mode for TU/vLLM thinking visibility.
- Proxy emits/normalizes `reasoning_content` for Pi compatibility.
- Preserve this behavior unless explicitly changed.
