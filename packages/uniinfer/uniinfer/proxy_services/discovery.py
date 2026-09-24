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
    "Link": '</openapi.json>; rel="service-desc", </guide.md>; rel="help"',
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
        "auth": {
            "type": "bearer",
            "header": "Authorization",
            "token_shape": "operator-issued combined token",
            "required_for": "keyed providers and live instance models",
        },
        "links": {
            "self": "/v1",
            "providers": "/v1/providers",
            "models": "/v1/models",
            "live_instance_models": "/v1/models/{provider}",
            "chat_completions": "/v1/chat/completions",
            "embeddings": "/v1/embeddings",
            "images": "/v1/images/generations",
            "openapi": "/openapi.json",
            "guide": "/guide.md",
            "health": "/health",
        },
        "next": [
            {"rel": "providers", "href": "/v1/providers"},
            {"rel": "service-desc", "href": "/openapi.json"},
        ],
    }


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
