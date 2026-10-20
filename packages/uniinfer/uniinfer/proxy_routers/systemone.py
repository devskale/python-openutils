"""Systemone: eine Tür für System-One-Modelle — decision UND extraction.

``POST /v1/systemone`` leitet an Instanz-Aliase weiter, dialekt-agnostisch:

- **decision** (``{"model", "state", "questions"}`` → ``{answers}``):
  jev, clef, d3-mini (dgx), …
- **extraction** (``{"model", "state", "fields"}`` → typisierte Feldwerte
  mit Confidence): clairic, … — Forward-Pass, keine Output-Tokens.

Zwei deklarative Achsen, kein Provider-Hardcode:

1. **Dialekt** ist eine Modell-Eigenschaft: ``type_overrides.json`` typed
   Modelle als ``decision``/``extraction`` (Katalog-Typ ist Autorität); der
   Payload-Shape (``questions`` vs. ``fields``) ist Fallback für Ungetypte
   und wird bei Typ-Konflikt mit klarer 400 abgelehnt.

2. **Transport** ist eine Instanz-Eigenschaft: Instanzen MIT ``base_url``
   werden nativ durchgereicht (Body 1:1, beide Dialekte); Instanzen OHNE
   (built-ins) gehen durch den Chat-Wrap, der NUR decision bedienen kann —
   Extraction-Modelle generieren keine Output-Tokens, Chat-Wrap ist dafür
   physikalisch unmöglich → 400 statt kryptischem 502.

``systemone_path``-Suffix pro Instanz: ``None`` → ``/systemone`` (native
vLLM/jev), ``""`` → base_url AS-IS, ``"/{model}"`` → Modell im Pfad (eine
Instanz pro Endpoint, z.B. CF Workers AI ``/ai/run`` für ALLE CF-Modelle).

Auth folgt dem Chat-Muster: keyless Instanzen kommen ohne Proxy-Token aus;
alles andere verlangt Provider-Zugang — der aufgelöste Key wird als
Upstream-Bearer weitergereicht.
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


# ── Dialekt-Layer: Modell-Eigenschaft, deklarativ aus dem Katalog ────────

_SYSTEMONE_TYPES = {"decision", "extraction"}
_type_overrides_cache: tuple[float, dict] | None = None


def _catalog_systemone_type(bare_model: str) -> str | None:
    """``decision``/``extraction`` aus type_overrides.json (mtime-gecacht).

    Gleiche kuratierte Quelle wie der Katalog (``/v1/models``-Typ-Auflösung):
    Neu Modelle werden deklarativ getyped, kein Router-/Provider-Code nötig.
    """
    global _type_overrides_cache
    import os
    import time
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "models", "type_overrides.json")
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    if _type_overrides_cache is not None and _type_overrides_cache[0] == mtime:
        doc = _type_overrides_cache[1]
    else:
        try:
            import json as _json
            with open(path) as f:
                doc = _json.load(f).get("models", {})
            _type_overrides_cache = (mtime, doc)
        except (OSError, ValueError):
            return None
    t = doc.get(bare_model)
    return t if t in _SYSTEMONE_TYPES else None


def _dialect_from_body(body: dict) -> str | None:
    """``extraction`` wenn ``fields`` am Root, ``decision`` wenn ``questions``."""
    if isinstance(body.get("fields"), (dict, list)):
        return "extraction"
    if isinstance(body.get("questions"), dict):
        return "decision"
    return None


def _resolve_dialect(bare_model: str, body: dict) -> tuple[str | None, JSONResponse | None]:
    """Dialekt-Auflösung: Katalog-Typ ist Autorität, Payload-Shape Fallback.

    Rückgabe (Dialekt | None, Fehler-Response | None).
    """
    payload_dialect = _dialect_from_body(body)
    if payload_dialect is None:
        return None, JSONResponse(
            {"error": "systemone body must contain 'questions' (decision) "
                      "or 'fields' (extraction)"}, status_code=400)
    catalog_type = _catalog_systemone_type(bare_model)
    if catalog_type is not None and catalog_type != payload_dialect:
        return None, JSONResponse(
            {"error": f"model '{bare_model}' is typed '{catalog_type}' in the "
                      f"catalog; body carries a {payload_dialect} payload "
                      f"({'fields' if payload_dialect == 'extraction' else 'questions'})"},
            status_code=400)
    return payload_dialect, None


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

        dialect, dialect_err = _resolve_dialect(bare_model, body)
        if dialect_err is not None:
            return dialect_err

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
            if dialect != "decision":
                return JSONResponse(
                    {"error": f"extraction models need a native instance "
                              f"(base_url); chat-wrap only serves decision — "
                              f"extraction generates no output tokens"},
                    status_code=400)
            return await _chat_wrap(alias, bare_model, body, upstream_key)

        forward = {k: v for k, v in body.items() if k != "model"}
        forward["model"] = bare_model  # bare id — Upstreams kennen kein alias@

        headers = {"Content-Type": "application/json"}
        if upstream_key:
            headers["Authorization"] = f"Bearer {upstream_key}"

        from uniinfer.core import _shared_async_client
        # Systemone-Ziel: Default /systemone-Suffix; Overlay kann pro Instanz
        # überschreiben ("" = base_url AS-IS, "/{model}" = Modell im Pfad —
        # eine Instanz pro Endpoint, z.B. CF Workers AI /ai/run).
        so_path = getattr(spec, "systemone_path", None)
        if so_path is None:
            so_path = "/systemone"
        if "{model}" in so_path:
            so_path = so_path.replace("{model}", bare_model.lstrip("/"))
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
        # CF Workers AI wickelt System-One-Antworten in {success, result: …} —
        # für Consumer-Einheitlichkeit heben: flacher {answers, usage}-Shape
        # wie native vLLM/jev-Endpoints.
        if (isinstance(payload, dict) and payload.get("success") is True
                and isinstance(payload.get("result"), dict)):
            payload = payload["result"]
        return JSONResponse(payload, status_code=resp.status_code)

    async def _chat_wrap(alias, bare_model, body, api_key):
        """Fallback für Instanzen ohne natives /systemone (built-ins wie
        opencode/pollinations): {state, questions} als letzte user-Nachricht
        durch den Chat-Pfad des Providers (Target), Antwort-Content als JSON
        zurück im Systemone-Shape. Content = {answers: …} ODER direkt das
        Answers-Objekt (Jev-Dialekt). Nur für Decision-Modelle — der Router
        lehnt Extraction-Dialekt vor diesem Wrap ab."""
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
