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
    parse_models_file,
    refresh_models_file,
)


def create_models_router(version: str) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/models")
    async def list_models():
        await ensure_fresh_models_file()
        models = parse_models_file()
        return {
            "object": "list",
            "data": [
                {"id": model_id, "object": "model", "owned_by": "skaledev"}
                for model_id in models
            ],
        }

    @router.post("/v1/system/update-models")
    async def update_models():
        try:
            return await refresh_models_file()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update models: {e}")

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
            if provider_name == "zai" and "glm-4.5-flash" not in raw_models:
                raw_models.append("glm-4.5-flash")
            return {
                "object": "list",
                "data": [
                    {"id": m, "object": "model", "owned_by": "skaledev"}
                    for m in raw_models
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
                    {"id": m, "object": "model", "owned_by": "skaledev"}
                    for m in raw_models
                ],
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except AuthenticationError as e:
            raise HTTPException(status_code=401, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/v1/system/info")
    async def get_system_info():
        return {"version": version}

    return router
