import asyncio
import logging
import os

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from uniinfer.auth import (
    get_optional_proxy_token,
    validate_proxy_token,
    verify_provider_access,
)
from uniinfer.config.instances import alias_serve_decision, resolve_instance
from uniinfer.core import ModelInfo
from uniinfer.errors import AuthenticationError
from uniinfer.provider_access import (
    list_embedding_models_for_provider,
    list_embedding_providers,
    list_models_for_provider,
)
from uniinfer.proxy_services.discovery import (
    discovery_headers,
    discovery_manifest,
    project_model_fields,
    provider_discovery,
)
from uniinfer.proxy_services.models_registry import (
    Catalog,
    ensure_fresh_models_file,
    models_refresh_status,
    trigger_models_refresh,
)

_logger = logging.getLogger("uniioai_proxy")
_ALIAS_REFRESH_TTL_DEFAULT = 300.0
_alias_inflight: set[str] = set()
_alias_bg_tasks: set[asyncio.Task] = set()


def _alias_refresh_ttl() -> float:
    try:
        return float(
            os.getenv("UNIINFER_ALIAS_REFRESH_TTL", _ALIAS_REFRESH_TTL_DEFAULT)
        )
    except (TypeError, ValueError):
        return _ALIAS_REFRESH_TTL_DEFAULT


async def _bg_refresh_alias(alias: str, token: str | None) -> None:
    try:
        await run_in_threadpool(list_models_for_provider, alias, token)
    except Exception as e:
        _logger.warning("background alias refresh failed for %s: %s", alias, e)
    finally:
        _alias_inflight.discard(alias)


def _cached_alias_response(provider_name: str, fields: str | None = None) -> dict:
    entry = Catalog().read_nested(provider_name)["providers"].get(provider_name, {})
    data = [dict(m, object="model") for m in entry.get("models", [])]
    try:
        # full pipeline (overrides + access filter) — cached aliases used to
        # filter but skip overrides, dropping e.g. arli's served-ctx cap.
        data = Catalog().resolve_for_instance(provider_name, data)
    except Exception:
        pass
    for model in data:
        model.setdefault("provider", provider_name)
    try:
        data = project_model_fields(data, fields)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"object": "list", "data": data}


def _model_info_to_dict(m) -> dict:
    """Convert a ModelInfo or plain string to an OpenAI-compatible model dict."""
    if isinstance(m, ModelInfo):
        d = {"id": m.id, "object": "model"}
        if m.owned_by:
            d["owned_by"] = m.owned_by
        else:
            d["owned_by"] = "skaledev"
        if m.context_window:
            d["context_window"] = m.context_window
        if m.max_output:
            d["max_output"] = m.max_output
        if m.type and m.type != "chat":
            d["type"] = m.type
        if m.capabilities:
            d["capabilities"] = m.capabilities
        if m.modalities:
            d["modalities"] = m.modalities
        if m.cost:
            d["cost"] = m.cost
        if m.access:
            d["access"] = m.access
        return d
    return {"id": str(m), "object": "model", "owned_by": "skaledev"}


def create_models_router(version: str) -> APIRouter:
    router = APIRouter()

    @router.get("/v1")
    async def discover_api():
        """Return the secret-free entry point for progressive API discovery."""
        return JSONResponse(discovery_manifest(version), headers=discovery_headers())

    @router.get("/v1/models")
    async def list_models(
        show_all: bool = Query(
            False,
            alias="all",
            description="Include all models (default: only free/granted).",
        ),
        provider: str | None = Query(
            None, description="Filter by provider (e.g. 'openai')."
        ),
        type: str | None = Query(
            None, description="Filter by type (chat, image, tts, stt, embed)."
        ),
        fields: str | None = Query(
            None,
            description="Comma-separated fields to project; id/object/provider are always included.",
        ),
    ):
        """List models. Defaults to accessible (free/granted) only — ~300 models
        instead of ~1500. Use ?all=true for everything, ?provider=X / ?type=X
        to narrow further. Catalog is mtime-cached (instant after first call).
        Virtual routing models (auto@free) are appended — clients discover
        them like concrete models."""
        await ensure_fresh_models_file()  # stale (>REFETCHTIME, default 24h) → background regen
        models = Catalog().list_resolved()
        from uniinfer.proxy_services.router import virtual_model_entries

        models = models + virtual_model_entries()
        if not show_all:
            models = [m for m in models if m.get("access") in ("free", "granted", "")]
        if provider:
            models = [m for m in models if m.get("provider") == provider]
        if type:
            models = [m for m in models if m.get("type") == type]
        try:
            models = project_model_fields(models, fields)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {
            "object": "list",
            "data": models,
            "version": version,
        }

    @router.get("/v1/providers")
    async def discover_providers():
        """List enabled provider instances without credentials or upstream calls."""
        return JSONResponse(provider_discovery(version), headers=discovery_headers())

    @router.get("/v1/system/version")
    async def get_version():
        return {"version": version}

    @router.post("/v1/system/update-models")
    async def update_models():
        """Trigger a full catalog gather without freezing or killing the proxy.

        systemd hosts delegate to the refresh oneshot service (own cgroup,
        202 + status polling); everywhere else the in-process threadpool
        subprocess runs synchronously (legacy 200). Never blocks the event
        loop long enough to starve the WATCHDOG pings — that exact freeze got
        the proxy SIGABRT-killed by systemd before (issue: watchdog-kill).
        """
        try:
            result = await trigger_models_refresh()
            code = 202 if result.get("via") == "systemd" else 200
            return JSONResponse(result, status_code=code)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update models: {e}")

    @router.get("/v1/system/update-models")
    async def get_update_models_status():
        """Gather status: running (service or in-process) + catalog age."""
        try:
            return await models_refresh_status()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read status: {e}")

    @router.get("/v1/catalog")
    async def get_catalog(
        providers: str | None = Query(
            None,
            description="Comma-separated provider IDs to include (e.g. 'openai,gemini'). Omit for all.",
        ),
        download: bool = Query(
            False,
            description="If true, send as attachment (Content-Disposition) for direct download.",
        ),
    ):
        """Return the raw nested models.json catalog, optionally filtered by provider(s).

        Public endpoint (no auth) so the catalog regenerated daily at 04:00 UTC
        can be fetched and downloaded via web. Returns the native nested shape:
        {"_meta": {...}, "providers": {pid: {provider_class, kind, models}}}.
        """
        await ensure_fresh_models_file()
        catalog = Catalog().read_nested(providers)
        import json as _json

        body = _json.dumps(catalog, indent=2, ensure_ascii=False)
        headers = {}
        if download:
            if providers:
                slug = "-".join(p.strip() for p in providers.split(",") if p.strip())
                filename = f"models-{slug}.json"
            else:
                filename = "models.json"
            headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        return Response(content=body, media_type="application/json", headers=headers)

    @router.get("/v1/models/deprecated")
    async def list_deprecated_models():
        """List deprecated models with deprecation info."""
        await ensure_fresh_models_file()
        models = Catalog().list_resolved()
        deprecated = [
            m
            for m in models
            if m.get("status") == "deprecated" or m.get("deprecation_date")
        ]
        return {
            "object": "list",
            "data": deprecated,
            "total": len(deprecated),
        }

    @router.get("/v1/models/new")
    async def list_new_models(days: int = 7):
        """List models first seen in the last N days."""
        from datetime import datetime, timedelta, timezone

        await ensure_fresh_models_file()
        models = Catalog().list_resolved()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
            "%Y-%m-%d"
        )
        new = [m for m in models if m.get("first_seen", "") >= cutoff]
        return {
            "object": "list",
            "data": new,
            "total": len(new),
            "since": cutoff,
            "days": days,
        }

    @router.get("/v1/models/stale")
    async def list_stale_models(days: int = 90):
        """List models not seen from their provider within N days.

        These are models tracked in _model_history.json but absent from
        the last generate_models.py run. Models gone > PRUNE_DAYS are
        auto-pruned from the catalog.
        """
        await ensure_fresh_models_file()
        stale = Catalog().read_stale_models()
        filtered = (
            [s for s in stale if s.get("days_missing", 0) >= days] if days else stale
        )
        return {
            "object": "list",
            "data": filtered,
            "total": len(filtered),
            "prune_after_days": 90,
        }

    @router.get("/v1/models/{provider_name}")
    async def dynamic_list_models(
        provider_name: str,
        fields: str | None = Query(
            None,
            description="Comma-separated fields to project; id/object/provider are always included.",
        ),
        api_bearer_token: str = Depends(validate_proxy_token),
    ):
        try:
            spec = resolve_instance(provider_name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        try:
            verify_provider_access(api_bearer_token, provider_name)
        except HTTPException as e:
            raise e
        try:
            # Custom alias: serve cached within TTL, stale-while-revalidate beyond it
            # (background refresh, never blocks). First-ever hit (no cache) fetches sync.
            if not spec.is_builtin:
                age = Catalog().provider_age_seconds(provider_name)
                decision = alias_serve_decision(
                    age, _alias_refresh_ttl(), provider_name in _alias_inflight
                )
                if decision == "serve_cached":
                    return _cached_alias_response(provider_name, fields)
                if decision == "serve_cached_and_refresh":
                    _alias_inflight.add(provider_name)
                    task = asyncio.create_task(
                        _bg_refresh_alias(provider_name, api_bearer_token)
                    )
                    _alias_bg_tasks.add(task)
                    task.add_done_callback(_alias_bg_tasks.discard)
                    return _cached_alias_response(provider_name, fields)
                # fetch_sync -> fall through to the live fetch below
            raw_models = await run_in_threadpool(
                list_models_for_provider, provider_name, api_bearer_token
            )
            if provider_name == "zai" and "glm-4.5-flash" not in [
                str(m) for m in raw_models
            ]:
                raw_models.append("glm-4.5-flash")
            data = [_model_info_to_dict(m) for m in raw_models]
            for model in data:
                model.setdefault("provider", provider_name)
            try:
                data = project_model_fields(data, fields)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            return {
                "object": "list",
                "data": data,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except AuthenticationError as e:
            raise HTTPException(status_code=401, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/v1/embedding/providers")
    async def get_embedding_providers():
        try:
            providers = list_embedding_providers()
            return {"object": "list", "data": providers}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/v1/embedding/models/{provider_name}")
    async def dynamic_list_embedding_models(
        provider_name: str,
        api_bearer_token: str | None = Depends(get_optional_proxy_token),
    ):
        try:
            if provider_name != "ollama" and not api_bearer_token:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required for this provider",
                )
            verify_provider_access(api_bearer_token, provider_name)

            raw_models = list_embedding_models_for_provider(
                provider_name,
                "" if provider_name == "ollama" else (api_bearer_token or ""),
            )
            return {
                "object": "list",
                "data": [_model_info_to_dict(m) for m in raw_models],
            }
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except AuthenticationError as e:
            raise HTTPException(status_code=401, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/v1/models/overrides")
    async def get_model_overrides(
        api_bearer_token: str = Depends(validate_proxy_token),
    ):
        """Return all model overrides."""
        return Catalog().read_overrides()

    @router.put("/v1/models/overrides/{model_id:path}")
    async def put_model_override(
        model_id: str, body: dict, api_bearer_token: str = Depends(validate_proxy_token)
    ):
        """Save a model override. Fields: type, context_window, max_output, dimensions, cost, capabilities, name, weight, pin, exclude."""
        allowed = {
            "type",
            "context_window",
            "max_output",
            "dimensions",
            "cost",
            "capabilities",
            "name",
            "modalities",
            "weight",
            "pin",
            "exclude",
        }
        override = {k: v for k, v in body.items() if k in allowed and v is not None}
        if not override:
            raise HTTPException(status_code=400, detail="No valid fields to update")
        Catalog().save_override(model_id, override)
        return {"status": "ok", "model": model_id, "fields": list(override.keys())}

    @router.delete("/v1/models/overrides/{model_id:path}")
    async def del_model_override(
        model_id: str, api_bearer_token: str = Depends(validate_proxy_token)
    ):
        """Delete a model override."""
        deleted = Catalog().delete_override(model_id)
        return {"status": "ok", "deleted": deleted}

    @router.post("/v1/models/deprecate/{model_id:path}")
    async def deprecate_model(
        model_id: str, body: dict, api_bearer_token: str = Depends(validate_proxy_token)
    ):
        """Mark a model deprecated (manual). Body: {"reason": "..."}.

        Writes status=deprecated + deprecation_date/reason/source=manual via
        the override seam: the model leaves active listings and the
        auto-router candidates; /v1/models/deprecated shows it with reason.
        """
        reason = str(body.get("reason") or "").strip()
        if not reason:
            raise HTTPException(status_code=400, detail="'reason' is required")
        entry = Catalog().deprecate_model(model_id, reason, source="manual")
        return {"status": "ok", "deprecated": model_id, "entry": entry}

    @router.post("/v1/models/undeprecate/{model_id:path}")
    async def undeprecate_model(
        model_id: str, api_bearer_token: str = Depends(validate_proxy_token)
    ):
        """Remove the deprecation mark (restore active)."""
        changed = Catalog().undeprecate_model(model_id)
        return {"status": "ok", "undeprecated": model_id, "changed": changed}

    @router.get("/v1/system/info")
    async def get_system_info():
        return {"version": version}

    return router
