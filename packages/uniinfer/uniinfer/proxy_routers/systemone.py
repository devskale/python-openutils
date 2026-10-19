"""Systemone decision-reads: generischer Pass-Through für Instanz-Aliase.

``POST /v1/systemone`` mit ``{"model": "alias@model", "state": …, "questions": …}``
leitet an ``{instance.base_url}/systemone`` weiter und relayt die Antwort.

Der Alias entscheidet das Ziel — kein Provider-Hardcode: funktioniert für
Custom-Aliase auf vLLM-PR-#57250-Endpunkten (DiffusionGemma, …) genauso wie
für built-ins mit Systemone-Dialekt (opencode jev). Das ``alias@``-Präfix wird
vom weitergeleiteten ``model``-Feld abgestreift (Upstreams kennen nur bare IDs).
Instanzen MIT ``base_url`` werden durchgereicht (natives ``/systemone``);
Instanzen OHNE (built-ins) gehen durch den Chat-Wrap: Payload als letzte
user-Nachricht über den Chat-Pfad des Providers, Antwort-Content als JSON
zurück — eine Tür für alle Decision-Modelle, egal welcher Wire-Dialekt.

Auth folgt dem Chat-Muster: keyless Instanzen (``requires_api_key: false``)
kommen ohne Proxy-Token aus; alles andere verlangt Provider-Zugang — und der
dabei aufgelöste Key wird als Upstream-Bearer weitergereicht (Gateway-Auth am
Ziel, z.B. dgx model-proxy hinter einem Tunnel).
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
        # alles andere verlangt Provider-Zugang wie im Chat-Flow. Der dabei
        # aufgelöste Key wird als Upstream-Bearer weitergereicht (Gateway-Auth
        # am Ziel, z.B. dgx model-proxy hinter einem Tunnel) statt verworfen.
        upstream_key = None
        if instance_requires_api_key(alias):
            try:
                upstream_key = verify_provider_access(api_bearer_token, alias)
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=401)
        else:
            try:
                from credgoo import get_api_key as _gk
                upstream_key = _gk(_resolve_credgoo_service(alias))
            except Exception:
                upstream_key = None

        base = (spec.base_url or "").rstrip("/")
        if not base:
            return await _chat_wrap(alias, bare_model, body, upstream_key)

        forward = {k: v for k, v in body.items() if k != "model"}
        forward["model"] = bare_model  # bare id — Upstreams kennen kein alias@

        headers = {"Content-Type": "application/json"}
        if upstream_key:
            headers["Authorization"] = f"Bearer {upstream_key}"

        from uniinfer.core import _shared_async_client
        # Systemone-Ziel: Default /systemone-Suffix; Overlay kann pro Instanz
        # überschreiben ("" = base_url AS-IS, z.B. CF Workers AI /ai/run/…).
        so_path = getattr(spec, "systemone_path", None)
        if so_path is None:
            so_path = "/systemone"
        url = f"{base}{so_path}"
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

    async def _chat_wrap(alias, bare_model, body, api_key):
        """Fallback für Instanzen ohne natives /systemone (built-ins wie
        opencode/pollinations): {state, questions} als letzte user-Nachricht
        durch den Chat-Pfad des Providers (Target), Antwort-Content als JSON
        zurück im Systemone-Shape. Content = {answers: …} ODER direkt das
        Answers-Objekt (Jev-Dialekt)."""
        from uniinfer.completion import Target

        wrap_payload = {"state": body.get("state"), "questions": body.get("questions")}
        try:
            target = Target(f"{alias}@{bare_model}", api_key)
            resp = await target.acomplete(
                [{"role": "user", "content": json.dumps(wrap_payload)}])
        except Exception as e:
            return JSONResponse(
                {"error": f"chat-wrap upstream error: {e}"}, status_code=502)

        content = getattr(getattr(resp, "message", None), "content", None)
        if not content or not isinstance(content, str):
            return JSONResponse(
                {"error": "chat-wrap: upstream returned no content"},
                status_code=502)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return JSONResponse(
                {"error": "chat-wrap: upstream content is not JSON "
                          "(model speaks no Systemone dialect)",
                 "content": content[:200]}, status_code=502)

        answers = parsed.get("answers") if isinstance(parsed, dict) and "answers" in parsed else parsed
        out = {"model": f"{alias}@{bare_model}", "answers": answers}
        usage = getattr(resp, "usage", None)
        if isinstance(usage, dict) and usage:
            out["usage"] = usage
        return JSONResponse(out)

    return router
