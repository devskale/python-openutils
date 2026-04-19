from fastapi import APIRouter, Depends, HTTPException

from uniinfer.auth import get_optional_proxy_token, validate_proxy_token
from uniinfer.uniioai import (
    list_embedding_models_for_provider,
    list_embedding_providers,
    list_models_for_provider,
    list_providers,
)
from uniinfer.errors import AuthenticationError
from uniinfer.proxy_services.models_registry import (
    ensure_fresh_models_file,
    refresh_models_file,
    list_all_models_from_factories,
    update_provider_in_cache,
    save_model_override,
    load_model_overrides,
    delete_model_override,
)
from uniinfer.core import ModelInfo
import dataclasses


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
        return d
    return {"id": str(m), "object": "model", "owned_by": "skaledev"}


def create_models_router(version: str) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/models")
    async def list_models():
        await ensure_fresh_models_file()
        models = list_all_models_from_factories()
        return {
            "object": "list",
            "data": models,
        }

    @router.post("/v1/system/update-models")
    async def update_models():
        try:
            return await refresh_models_file()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update models: {e}")

    @router.get("/v1/models/deprecated")
    async def list_deprecated_models():
        """List deprecated models with deprecation info."""
        await ensure_fresh_models_file()
        models = list_all_models_from_factories()
        deprecated = [
            m for m in models
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
        from datetime import datetime, timezone, timedelta
        await ensure_fresh_models_file()
        models = list_all_models_from_factories()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        new = [m for m in models if m.get("first_seen", "") >= cutoff]
        return {
            "object": "list",
            "data": new,
            "total": len(new),
            "since": cutoff,
            "days": days,
        }

    @router.get("/v1/providers")
    async def get_providers(api_bearer_token: str = Depends(validate_proxy_token)):
        try:
            providers = list_providers()
            return {"object": "list", "data": providers}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/v1/models/{provider_name}")
    async def dynamic_list_models(provider_name: str, api_bearer_token: str = Depends(validate_proxy_token)):
        try:
            raw_models = list_models_for_provider(provider_name, api_bearer_token)
            if provider_name == "zai" and "glm-4.5-flash" not in [str(m) for m in raw_models]:
                raw_models.append("glm-4.5-flash")
            return {
                "object": "list",
                "data": [
                    _model_info_to_dict(m) for m in raw_models
                ],
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

            raw_models = list_embedding_models_for_provider(
                provider_name,
                "" if provider_name == "ollama" else (api_bearer_token or ""),
            )
            return {
                "object": "list",
                "data": [
                    _model_info_to_dict(m) for m in raw_models
                ],
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except AuthenticationError as e:
            raise HTTPException(status_code=401, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/v1/models/overrides")
    async def get_model_overrides(api_bearer_token: str = Depends(validate_proxy_token)):
        """Return all model overrides."""
        return load_model_overrides()

    @router.put("/v1/models/overrides/{model_id:path}")
    async def put_model_override(model_id: str, body: dict, api_bearer_token: str = Depends(validate_proxy_token)):
        """Save a model override. Fields: type, context_window, max_output, dimensions, cost, capabilities, name."""
        allowed = {"type", "context_window", "max_output", "dimensions", "cost", "capabilities", "name", "modalities"}
        override = {k: v for k, v in body.items() if k in allowed and v is not None}
        if not override:
            raise HTTPException(status_code=400, detail="No valid fields to update")
        save_model_override(model_id, override)
        return {"status": "ok", "model": model_id, "fields": list(override.keys())}

    @router.delete("/v1/models/overrides/{model_id:path}")
    async def del_model_override(model_id: str, api_bearer_token: str = Depends(validate_proxy_token)):
        """Delete a model override."""
        deleted = delete_model_override(model_id)
        return {"status": "ok", "deleted": deleted}

    @router.get("/v1/system/info")
    async def get_system_info():
        return {"version": version}

    return router
