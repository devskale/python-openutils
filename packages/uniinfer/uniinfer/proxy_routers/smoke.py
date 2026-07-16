"""Smoke-test router: live health checks against provider models.

Reads models from the cached catalog (no extra API calls, respects rate
limits), picks a few per provider, and fires a minimal completion at each to
verify end-to-end reachability. Thinking models get a generous token budget so
their reasoning doesn't eat the whole response.
"""
import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from uniinfer.auth import get_optional_proxy_token, verify_provider_access
from uniinfer.errors import UniInferError
from uniinfer.proxy_services.models_registry import ensure_fresh_models_file, load_catalog
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


def create_smoke_router() -> APIRouter:
    router = APIRouter()

    @router.post("/v1/system/smoke")
    async def smoke(
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
        catalog = load_catalog(providers)
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
            models = [
                m["id"]
                for m in provider_catalog[provider_name].get("models", [])
                if m.get("id")
            ][:per_provider]
            provider_report: dict[str, Any] = {"models": {}, "selected": models}

            for model_id in models:
                key = f"{provider_name}@{model_id}"
                provider_report["models"][key] = await _smoke_one(
                    provider_name=provider_name,
                    model_id=model_id,
                    provider_model=key,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    api_bearer_token=api_bearer_token,
                )

            ok = sum(1 for r in provider_report["models"].values() if r["ok"])
            provider_report["summary"] = f"{ok}/{len(models)} ok"
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
