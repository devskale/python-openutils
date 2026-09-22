"""Systemone decision-reads: generischer Pass-Through für Instanz-Aliase.

``POST /v1/systemone`` mit ``{"model": "alias@model", "state": …, "questions": …}``
leitet an ``{instance.base_url}/systemone`` weiter und relayt die Antwort.

Der Alias entscheidet das Ziel — kein Provider-Hardcode: funktioniert für
Custom-Aliase auf vLLM-PR-#57250-Endpunkten (DiffusionGemma, …) genauso wie
für built-ins mit Systemone-Dialekt (opencode jev). Das ``alias@``-Präfix wird
vom weitergeleiteten ``model``-Feld abgestreift (Upstreams kennen nur bare IDs).

Auth folgt dem Chat-Muster: keyless Instanzen (``requires_api_key: false``)
kommen ohne Proxy-Token aus; alles andere verlangt Provider-Zugang.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from uniinfer.auth import get_optional_proxy_token, verify_provider_access
from uniinfer.config.instances import instance_requires_api_key, resolve_instance
from uniinfer.provider_access import _resolve_credgoo_service


def create_systemone_router(
    parse_provider_model=None,
) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/systemone")
    async def systemone(
        request: Request,
        api_bearer_token: str | None = Depends(get_optional_proxy_token),
    ):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "body must be JSON"}, status_code=400)

        model = body.get("model")
        if not model:
            return JSONResponse(
                {"error": "field 'model' ('alias@model') is required"},
                status_code=400)

        if parse_provider_model is not None:
            alias, bare_model = parse_provider_model(model)
        else:
            alias, _, bare_model = model.partition("@")

        try:
            spec = resolve_instance(alias)
        except Exception as e:
            return JSONResponse({"error": f"unknown instance '{alias}': {e}"},
                                status_code=404)

        # Auth: keyless Instanzen (public vLLM) brauchen keinen Proxy-Key;
        # alles andere verlangt Provider-Zugang wie im Chat-Flow.
        if instance_requires_api_key(alias):
            try:
                verify_provider_access(api_bearer_token, alias)
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=401)
        else:
            api_key = None
            try:
                from credgoo import get_api_key as _gk
                api_key = _gk(_resolve_credgoo_service(alias))
            except Exception:
                api_key = None

        base = (spec.base_url or "").rstrip("/")
        if not base:
            return JSONResponse(
                {"error": f"instance '{alias}' has no base_url for systemone"},
                status_code=400)

        forward = {k: v for k, v in body.items() if k != "model"}
        forward["model"] = bare_model  # bare id — Upstreams kennen kein alias@

        headers = {"Content-Type": "application/json"}
        if not instance_requires_api_key(alias):
            try:
                from credgoo import get_api_key as _gk
                key = _gk(_resolve_credgoo_service(alias))
            except Exception:
                key = None
            if key:
                headers["Authorization"] = f"Bearer {key}"

        from uniinfer.core import _shared_async_client
        url = f"{base}/systemone"
        try:
            client = _shared_async_client()
            resp = await client.post(url, json=forward, headers=headers, timeout=60)
        except httpx.TimeoutException:
            return JSONResponse({"error": f"upstream timeout at {url}"}, status_code=504)
        except Exception as e:
            return JSONResponse({"error": f"upstream error: {e}"}, status_code=502)

        try:
            payload = resp.json()
        except Exception:
            payload = {"raw": resp.text[:2000]}
        return JSONResponse(payload, status_code=resp.status_code)

    return router
