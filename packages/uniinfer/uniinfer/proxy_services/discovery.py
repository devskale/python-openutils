"""Progressive API discovery for agents.

Builds the small, secret-free contract an agent needs to walk from a bare
endpoint to providers and then to model metadata. Provider discovery reads the
enabled instance registry plus the local catalog — it never performs an
upstream call.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from uniinfer.config.instances import get_instances
from uniinfer.proxy_services.models_registry import Catalog

# Model entries already contain no credentials, but a projection should still be
# explicit so future internal fields do not silently become part of the public
# slim response.
# Navigation core retained even in slim projections, but clients may name it
# explicitly without a confusing "unknown field" error.
MODEL_CORE_FIELDS = frozenset({"id", "object", "provider"})

MODEL_FIELDS = (
    frozenset(
        {
            "access",
            "capabilities",
            "context_window",
            "cost",
            "days_since_seen",
            "deprecation_date",
            "deprecation_replacement",
            "dimensions",
            "first_seen",
            "freshness",
            "knowledge_cutoff",
            "last_seen",
            "max_output",
            "modalities",
            "name",
            "owned_by",
            "provider",
            "release_date",
            "speed",
            "status",
            "type",
        }
    )
    | MODEL_CORE_FIELDS
)

_DISCOVERY_HEADERS = {
    "Link": ('</openapi.json>; rel="service-desc", </guide.md>; rel="help", '
             '</llms.txt>; rel="describedby"'),
    "Cache-Control": "public, max-age=300",
}


def discovery_headers() -> dict[str, str]:
    """Standard links and cache guidance for discovery responses."""
    return dict(_DISCOVERY_HEADERS)


def discovery_manifest(version: str) -> dict[str, Any]:
    """Return the first machine-readable entry point under ``/v1``."""
    return {
        "service": "uniinfer",
        "protocol": "openai-compatible",
        "version": version,
        "api": "/v1",
        "model_id_format": "provider@model",
        "model_id_format_note": (
            "strict: 'alias@model' with non-empty sides; bare model names "
            "without '@' are rejected (400). The alias addresses a provider "
            "instance — built-in ('mistral', 'groq', …) or fleet alias ('hab')."
        ),
        "auth": {
            "type": "bearer",
            "header": "Authorization",
            "token_shape": "operator-issued combined token",
            "required_for": "keyed providers and live instance models",
            "websocket_note": (
                "WebSockets accept the bearer as ?token= query param "
                "(browsers cannot set WS headers) or as Authorization header"
            ),
            "x_provider_key": (
                "Optional X-Provider-Key header overrides the upstream bearer "
                "for ANY provider — no meta-bearer needed for own-key-only "
                "requests; the call then rides the caller's own upstream quota"
            ),
        },
        "links": {
            "self": "/v1",
            "providers": "/v1/providers",
            "models": "/v1/models",
            "live_instance_models": "/v1/models/{provider}",
            "chat_completions": "/v1/chat/completions",
            "responses": "/v1/responses",
            "embeddings": "/v1/embeddings",
            "images": "/v1/images/generations",
            "routing_preview": "/v1/routing/auto",
            "model_health": "/v1/health/models",
            "openapi": "/openapi.json",
            "guide": "/guide.md",
            "contracts": "/contracts.md",
            "llms_txt": "/llms.txt",
            "realtime_stt": "/v1/audio/transcriptions/stream",
            "health": "/health",
        },
        "next": [
            {"rel": "providers", "href": "/v1/providers"},
            {"rel": "service-desc", "href": "/openapi.json"},
        ],
    }


def llms_txt_index() -> str:
    """Markdown index for agents, per the llms.txt v2 proposal (llmstxt.org).

    Stays small enough for one context window: terse endpoint map plus the
    realtime-STT WebSocket protocol inline (OpenAPI cannot describe WS, so
    this is the machine-readable home of that contract). Detail lives behind
    the links and is fetched on demand.
    """
    return """# uniinfer

> OpenAI-compatible multi-provider LLM gateway: chat, embeddings, images, TTS/STT and realtime streaming STT over one bearer token. Model ids are `provider@model`.

## Model ids (canonical form)

Every endpoint that takes a `model` expects the strict form `alias@model`:

- `alias` — provider instance: built-in (`mistral`, `groq`, `openrouter`, …) or fleet alias (`hab` = dgx gateway).
- `model` — the upstream model id, e.g. `mistral@mistral-medium-latest`, `hab@nemotron-3.5-asr-streaming-0.6b`.
- Bare names without `@` are rejected with 400. `@` splits once, so model ids may contain `@` themselves — the first `@` binds the alias.

## API

- [GET /v1](/v1) — machine-readable discovery manifest (JSON, no auth)
- [GET /v1/providers](/v1/providers) — enabled provider instances (no auth)
- [GET /v1/models](/v1/models) — model catalog (no auth); `/v1/models/{provider}` live listing (auth)
- [POST /v1/chat/completions](/v1/chat/completions) — chat, stream + tools (auth)
- [POST /v1/responses](/v1/responses) — OpenAI Responses dialect, translated onto EVERY chat provider; stateless v1 (input/instructions/tools/stream) (auth)
- [POST /v1/embeddings](/v1/embeddings) — embeddings (auth)
- [POST /v1/images/generations](/v1/images/generations) — image gen (auth)
- [POST /v1/audio/speech](/v1/audio/speech) — TTS (auth)
- [POST /v1/audio/transcriptions](/v1/audio/transcriptions) — batch STT, multipart upload (auth)
- [GET /openapi.json](/openapi.json) — OpenAPI 3 schema (HTTP endpoints only)

Auth everywhere: `Authorization: Bearer <token>`, operator-issued — unlocks every provider with pooled keys. Operator machines fetch the current gateway token with `credgoo uniinfer` (refresh with `--no-cache` when a cached one 401s; `~/.config/api_keys/api_keys.json` is only credgoo's cache). `/v1/models` and other no-auth routes validate nothing — a 200 there proves no token. Per-request override: `X-Provider-Key: <your-key>` replaces the pooled bearer for ANY provider; an own-key-only request needs no meta-bearer at all.

## Realtime STT (WebSocket — not in OpenAPI)

`WS /v1/audio/transcriptions/stream?model={provider@asr-model}&language={code}`

- Auth: bearer as `?token=` query param (browsers cannot set WS headers) **or** `Authorization` header.
- Client → server: binary frames, raw int16 PCM 16 kHz mono. Empty binary frame = end-of-audio.
- Server → client: JSON `{"type":"partial","partial":true,"text":…}` while streaming, `{"type":"final","partial":false,"text":…}` after EOF.
- Close codes: 4400 bad request · 4401 auth · 4404 unknown instance · 4502 upstream unreachable.
- Tip: ~300 ms of lead-in silence improves first-word recognition.

## Docs

- [Full integration guide (markdown)](/guide.md) — SDK usage, all endpoints, realtime STT client example
- [Agentic contracts](/contracts.md) — machine-stable promises: discovery, identity, auth, surfaces, reliability, versioning
- [Discovery manifest](/v1) — JSON entry point
"""


def provider_discovery(version: str) -> dict[str, Any]:
    """List enabled provider instances from local registry and cached catalog.

    Custom fleet aliases are included so agents can discover them, but
    infrastructure fields (``base_url``, ``credgoo_service``) are deliberately
    omitted. Model counts are catalog-only: an empty count means that no cached
    listing is available yet, not that the provider has no models.
    """
    catalog = Catalog().read_nested().get("providers", {})
    data: list[dict[str, Any]] = []

    for alias, spec in sorted(get_instances().items()):
        if not spec.enabled:
            continue
        cached_models = catalog.get(alias, {}).get("models", [])
        item: dict[str, Any] = {
            "id": alias,
            "object": "provider",
            "kind": "builtin" if spec.is_builtin else "custom",
            "requires_auth": spec.requires_api_key,
            "model_count": len(cached_models),
            "default_model": spec.default_model,
            "models_url": f"/v1/models?provider={quote(alias, safe='')}",
            "live_models_url": f"/v1/models/{quote(alias, safe='')}",
        }
        if not spec.is_builtin:
            item["underlying_provider"] = spec.provider
        data.append(item)

    return {
        "object": "list",
        "data": data,
        "total": len(data),
        "version": version,
        "links": {
            "models": "/v1/models",
            "openapi": "/openapi.json",
            "guide": "/guide.md",
            "contracts": "/contracts.md",
        },
    }


def project_model_fields(
    models: list[dict[str, Any]], fields: str | None
) -> list[dict[str, Any]]:
    """Project model entries to an explicit, agent-friendly field subset.

    ``id``, ``object`` and ``provider`` are retained as the navigation core even
    in slim responses. Unknown fields fail with a helpful error instead of
    silently returning an empty projection.
    """
    if not fields:
        return models

    requested = {value.strip() for value in fields.split(",") if value.strip()}
    unknown = sorted(requested - MODEL_FIELDS)
    if unknown:
        allowed = ", ".join(sorted(MODEL_FIELDS))
        raise ValueError(
            f"unknown model fields: {', '.join(unknown)}; allowed: {allowed}"
        )

    selected = requested | {"id", "object", "provider"}
    projected: list[dict[str, Any]] = []
    for model in models:
        projected.append(
            {key: value for key, value in model.items() if key in selected}
        )
    return projected
