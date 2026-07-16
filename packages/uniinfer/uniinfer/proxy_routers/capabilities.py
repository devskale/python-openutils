"""Capability-test router: probe what a model can do + exercise each feature.

``GET /v1/system/capabilities?model=ollama@qwen3.5:0.8b[&probes=csv][&perf=true]``

Returns the capability matrix (profile + per-probe ``pass|fail|skip|error`` with
evidence). Mirrors the chat router's credential resolution (Ollama uses credgoo
``ollama`` + the configured ``base_url``; other providers use bearer access).
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from uniinfer.auth import get_optional_proxy_token, verify_provider_access
from uniinfer.capabilities import Target, run_capabilities

logger = logging.getLogger("uniioai_proxy")


def create_capabilities_router(*, parse_provider_model, provider_configs) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/system/capabilities")
    async def capabilities(
        model: str = Query(..., description="provider@model, e.g. ollama@qwen3.5:0.8b"),
        probes: str | None = Query(None, description="comma-separated probe subset"),
        perf: bool = Query(False, description="also run perf probes"),
        api_bearer_token: str | None = Depends(get_optional_proxy_token),
    ):
        try:
            provider_name, _ = parse_provider_model(model)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Mirror the chat router's per-provider credential resolution.
        if provider_name == "ollama":
            from credgoo import get_api_key as _get_credgoo_key

            try:
                provider_api_key = _get_credgoo_key("ollama")
            except Exception:  # noqa: BLE001
                provider_api_key = None
            base_url = provider_configs.get("ollama", {}).get("extra_params", {}).get("base_url")
        else:
            try:
                provider_api_key = verify_provider_access(api_bearer_token, provider_name)
            except Exception as e:
                raise HTTPException(status_code=401, detail=str(e))
            base_url = None

        probe_list = [p.strip() for p in probes.split(",") if p.strip()] if probes else None
        target = Target(provider_model=model, api_key=provider_api_key, base_url=base_url)
        try:
            report = await run_capabilities(target, probes=probe_list, perf=perf)
            return JSONResponse(content=report.as_dict())
        except Exception as e:  # noqa: BLE001
            logger.exception("capabilities failed for %s: %s", model, e)
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    return router
