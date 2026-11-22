"""Smoke-test router: live health checks against provider models.

Reads models from the cached catalog (no extra API calls, respects rate
limits), picks a few per provider, and fires a minimal completion at each to
verify end-to-end reachability. Thinking models get a generous token budget so
their reasoning doesn't eat the whole response.
"""
import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from uniinfer.auth import get_optional_proxy_token, verify_provider_access
from uniinfer.errors import UniInferError
from uniinfer.proxy_services.models_registry import ensure_fresh_models_file, Catalog
from uniinfer.completion import Target

logger = logging.getLogger("uniioai_proxy")

# TU is the primary backend; these are sensible defaults for a quick check.
DEFAULT_PROVIDERS = "tu"
# Reasonable budget for reasoning models — thinking can consume many tokens
# before the visible answer, so don't starve it.
# Thinking models (Qwen3.x / GLM-5.x / Claude extended thinking) need
# max_tokens ≫ 1–2k: reasoning consumes the budget before the visible answer.
# A too-low cap makes them look broken (empty / truncated output).
DEFAULT_MAX_TOKENS = 4096
DEFAULT_PER_PROVIDER = 3
DEFAULT_PROMPT = "Say hello."
# Per-model smoke timeout (seconds). Reasoning models can take a while.
PER_MODEL_TIMEOUT = 60.0


async def _smoke_surface_one(
    *,
    base_url: str,
    token: str | None,
    model: str,
    surface: str,
    stream: bool,
    prompt: str,
) -> dict[str, Any]:
    """Smoke one model through the HTTP SURFACE (chat or responses), not the
    library seam — validates the full proxy pipeline incl. SSE shaping.

    Streams are decoded through ``uniinfer.sse`` (the canonical seam); for
    responses we verify the ``response.created … response.completed``
    envelope, for chat the presence of content deltas + ``[DONE]``.
    Never raises — returns an error dict.
    """
    import httpx
    from uniinfer.sse import DONE, SSEDecoder

    started = time.monotonic()
    if surface not in ("chat", "responses"):
        return _err("bad_request", f"unknown surface '{surface}' (chat|responses)", started)
    url = f"{base_url.rstrip('/')}/v1/{'chat/completions' if surface == 'chat' else 'responses'}"
    if surface == "chat":
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
    else:
        payload = {"model": model, "input": prompt}
    if stream:
        payload["stream"] = True
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        events: list[str] = []
        text_parts: list[str] = []
        saw_done = False
        async with httpx.AsyncClient(timeout=PER_MODEL_TIMEOUT) as client:
            if not stream:
                r = await client.post(url, json=payload, headers=headers)
                elapsed = round((time.monotonic() - started) * 1000)
                if r.status_code != 200:
                    return _err("http_error", r.text[:300], started, status_code=r.status_code)
                body = r.json()
                if surface == "chat":
                    preview = (body.get("choices") or [{}])[0].get("message", {}).get("content")
                else:
                    out = (body.get("output") or [{}])[0]
                    preview = (out.get("content") or [{}])[0].get("text") if out.get("type") == "message" else out.get("type")
                return {
                    "ok": True,
                    "status": "ok",
                    "surface": surface,
                    "stream": False,
                    "latency_ms": elapsed,
                    "content_preview": (preview[:80] if isinstance(preview, str) else None),
                }

            async with client.stream("POST", url, json=payload, headers=headers) as r:
                elapsed = round((time.monotonic() - started) * 1000)
                if r.status_code != 200:
                    detail = (await r.aread())[:300].decode("utf-8", "replace")
                    return _err("http_error", detail, started, status_code=r.status_code)
                decoder = SSEDecoder()
                buf: list[str] = []
                async for chunk in r.aiter_text():
                    buf.extend(chunk.splitlines())
                    while buf:
                        line = buf.pop(0)
                        ev = decoder.decode(line.rstrip("\r") if line.endswith("\r") else line)
                        if ev is None:
                            continue
                        events.append(ev.event or "")
                        data = ev.data or ""
                        if data == DONE:
                            saw_done = True
                        elif surface == "chat":
                            try:
                                d = json.loads(data)
                                c = ((d.get("choices") or [{}])[0].get("delta") or {}).get("content")
                                if c:
                                    text_parts.append(c)
                                if "error" in d:
                                    return _err("stream_error", str(d["error"])[:200], started)
                            except ValueError:
                                pass
                        else:
                            if ev.event == "response.output_text.delta":
                                try:
                                    text_parts.append(json.loads(data).get("delta", ""))
                                except ValueError:
                                    pass
                            elif ev.event == "response.failed":
                                try:
                                    err = json.loads(data).get("response", {}).get("error")
                                    return _err("stream_error", str(err)[:200], started)
                                except ValueError:
                                    pass
        elapsed = round((time.monotonic() - started) * 1000)
        ok = bool(text_parts) or bool(events)
        if surface == "chat" and not saw_done:
            return _err("stream_error", "stream ended without [DONE]", started)
        if surface == "responses" and events and events[-1] != "response.completed":
            return _err("stream_error", f"last event: {events[-1]}", started)
        return {
            "ok": ok,
            "status": "ok" if ok else "empty_stream",
            "surface": surface,
            "stream": True,
            "latency_ms": elapsed,
            "content_preview": ("".join(text_parts)[:80] or None),
            "events": len(events),
            "event_first": events[0] if events else None,
            "event_last": events[-1] if events else None,
        }
    except asyncio.TimeoutError:
        return _err("timeout", f"no response within {PER_MODEL_TIMEOUT:.0f}s", started)
    except Exception as e:  # noqa: BLE001
        return _err("http_error", f"{type(e).__name__}: {str(e)[:200]}", started)


def create_smoke_router() -> APIRouter:
    router = APIRouter()

    @router.post("/v1/system/smoke")
    async def smoke(
        request: Request,
        providers: str = Query(
            DEFAULT_PROVIDERS,
            description="Comma-separated provider IDs to smoke (default: tu).",
        ),
        per_provider: int = Query(
            DEFAULT_PER_PROVIDER,
            ge=1,
            le=10,
            description="Number of models to smoke per provider.",
        ),
        models: str = Query(
            None,
            description="Comma-separated bare model ids to smoke for the (single) "
            "provider — replaces the per_provider catalog sample. Use for "
            "modality-correct targeting; ids must exist in the provider's catalog.",
        ),
        surfaces: str = Query(
            "chat",
            description="Which HTTP surfaces to smoke: 'chat', 'responses', or "
            "'chat,responses'. 'chat' keeps the legacy library-seam behavior; "
            "anything else goes through the real HTTP pipeline.",
        ),
        stream: bool = Query(
            False,
            description="Also smoke the SSE stream variant of each surface "
            "(verifies event envelopes, not just non-stream responses).",
        ),
        max_tokens: int = Query(
            DEFAULT_MAX_TOKENS,
            ge=16,
            le=8192,
            description="max_tokens for each smoke. Must be ≫1–2k for thinking models.",
        ),
        prompt: str = Query(DEFAULT_PROMPT, description="Prompt to send."),
        api_bearer_token: str | None = Depends(get_optional_proxy_token),
    ):
        """Smoke-test a handful of models across the given providers.

        Sequential (not parallel) so per-provider throttles/rate limits are
        respected. Returns a per-provider, per-model report.
        """
        await ensure_fresh_models_file()
        catalog = Catalog().read_nested(providers)
        provider_catalog = catalog.get("providers", {})

        requested = [p.strip() for p in providers.split(",") if p.strip()]
        # Only smoke providers present in the catalog.
        targets = [p for p in requested if p in provider_catalog]
        missing = [p for p in requested if p not in provider_catalog]

        report: dict[str, Any] = {
            "providers": {},
            "skipped_missing_providers": missing,
            "params": {
                "per_provider": per_provider,
                "max_tokens": max_tokens,
                "prompt": prompt,
            },
        }

        for provider_name in targets:
            catalog_models = [
                m["id"]
                for m in provider_catalog[provider_name].get("models", [])
                if m.get("id")
            ]
            if models:
                wanted = [m.strip() for m in models.split(",") if m.strip()]
                unknown = [m for m in wanted if m not in catalog_models]
                selected = [m for m in wanted if m in catalog_models]
                provider_report: dict[str, Any] = {
                    "models": {}, "selected": selected,
                }
                if unknown:
                    provider_report["unknown_models"] = unknown
            else:
                selected = catalog_models[:per_provider]
                provider_report = {"models": {}, "selected": selected}
            surface_list = [s.strip() for s in surfaces.split(",") if s.strip()]
            legacy = surface_list == ["chat"] and not stream
            for model_id in selected:
                key = f"{provider_name}@{model_id}"
                if legacy:
                    provider_report["models"][key] = await _smoke_one(
                        provider_name=provider_name,
                        model_id=model_id,
                        provider_model=key,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        api_bearer_token=api_bearer_token,
                    )
                else:
                    base = str(request.base_url).rstrip("/")
                    entry: dict[str, Any] = {}
                    for surface in surface_list:
                        entry[surface] = await _smoke_surface_one(
                            base_url=base,
                            token=api_bearer_token,
                            model=key,
                            surface=surface,
                            stream=False,
                            prompt=prompt,
                        )
                        if stream:
                            entry[f"{surface}:stream"] = await _smoke_surface_one(
                                base_url=base,
                                token=api_bearer_token,
                                model=key,
                                surface=surface,
                                stream=True,
                                prompt=prompt,
                            )
                    ok_all = all(
                        v.get("ok") for v in entry.values() if isinstance(v, dict)
                    )
                    entry["ok"] = ok_all
                    provider_report["models"][key] = entry

            ok = sum(1 for r in provider_report["models"].values() if r["ok"])
            provider_report["summary"] = f"{ok}/{len(selected)} ok"
            report["providers"][provider_name] = provider_report

        total_ok = sum(
            1
            for p in report["providers"].values()
            for r in p["models"].values()
            if r["ok"]
        )
        total = sum(len(p["models"]) for p in report["providers"].values())
        report["summary"] = f"{total_ok}/{total} models ok"
        return JSONResponse(content=report)

    return router


async def _smoke_one(
    *,
    provider_name: str,
    model_id: str,
    provider_model: str,
    prompt: str,
    max_tokens: int,
    api_bearer_token: str | None,
) -> dict[str, Any]:
    """Smoke a single provider@model. Never raises — returns an error dict."""
    started = time.monotonic()
    try:
        provider_api_key = verify_provider_access(api_bearer_token, provider_name)
        # Disable reasoning for a fast, deterministic smoke. reasoning_effort="none"
        # is the cross-provider "off" intent (no-op on non-reasoning backends).
        target = Target(provider_model, provider_api_key)
        completion = await asyncio.wait_for(
            target.acomplete(
                [{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=max_tokens,
                reasoning_effort="none",
            ),
            timeout=PER_MODEL_TIMEOUT,
        )
        msg = getattr(completion, "message", None)
        content = getattr(msg, "content", None) if msg else None
        usage = getattr(completion, "usage", None)
        finish = getattr(completion, "finish_reason", None)
        elapsed = round((time.monotonic() - started) * 1000)
        return {
            "ok": True,
            "status": "ok",
            "latency_ms": elapsed,
            "finish_reason": finish,
            "content_preview": (content[:80] if isinstance(content, str) else None),
            "usage": usage if isinstance(usage, dict) else None,
        }
    except asyncio.TimeoutError:
        return _err("timeout", f"no response within {PER_MODEL_TIMEOUT:.0f}s", started)
    except UniInferError as e:
        return _err("provider_error", str(e)[:300], started, status_code=getattr(e, "status_code", None))
    except HTTPException as e:
        return _err("http_error", str(e.detail)[:300], started, status_code=e.status_code)
    except Exception as e:  # noqa: BLE001
        logger.exception("Smoke failure for %s: %s", provider_model, e)
        return _err("error", f"{type(e).__name__}: {e}"[:300], started)


def _err(status: str, message: str, started: float, status_code: int | None = None) -> dict[str, Any]:
    return {
        "ok": False,
        "status": status,
        "error": message,
        "status_code": status_code,
        "latency_ms": round((time.monotonic() - started) * 1000),
    }
