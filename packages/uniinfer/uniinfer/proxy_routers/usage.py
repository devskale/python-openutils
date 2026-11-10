"""Usage router: per-key quota attribution for statuslines and scripts.

GET /v1/usage — read-only reporting on the shared upstream TU (Aqueduct)
quota: the caller's weighted usage per window, the proxy aggregate, the
documented limits and the reset times (top of minute / top of hour /
midnight UTC = 01:00 Vienna in winter). Same bearer auth as the chat API;
unknown or revoked tokens get 401. Reports tu@ models only — other
providers draw on their own quotas, which are not tracked here.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from uniinfer.auth import _allowed_token_hashes, _token_metadata, validate_proxy_token
from uniinfer.config.providers import get_provider_limits, get_request_multipliers
from uniinfer.proxy_services.key_usage import (
    bucket_end,
    bucket_start,
    get_key_usage,
    token_hash,
)

_SOURCE_KEY = "https://datalab.tuwien.ac.at/blog/aqueduct-rate-limits/"
_UPSTREAM_NOTE = (
    "Aqueduct limits bind the SHARED upstream TU key (all proxy clients "
    "funnel through it); 'used' is this key's attribution, 'proxy_total' "
    "the aggregate. Day resets midnight UTC (01:00 Vienna in winter)."
)


def _display_key(name: str) -> str:
    """Truncated key name for the response — never a secret."""
    return name if len(name) <= 8 else name[:8] + "…"


def create_usage_router() -> APIRouter:
    router = APIRouter()

    @router.get("/v1/usage")
    async def usage(
        api_bearer_token: str = Depends(validate_proxy_token),
    ) -> JSONResponse:
        th = token_hash(api_bearer_token)
        allowed = _allowed_token_hashes()
        if allowed is not None and th not in allowed:
            raise HTTPException(status_code=401, detail="Unknown or revoked token.")
        meta: dict[str, Any] = _token_metadata().get("tokens", {}).get(th, {})
        collector = get_key_usage()
        multipliers = get_request_multipliers("tu")
        limits_cfg = get_provider_limits("tu")
        window_limits = {
            "minute": "requests_per_minute",
            "hour": "requests_per_hour",
            "day": "requests_per_day",
        }
        now = time.time()
        view = collector.usage_view(th, multipliers, now=now)
        windows = {}
        for kind, lim_key in window_limits.items():
            start = bucket_start(kind, int(now))
            windows[kind] = {
                "used": view["windows"][kind]["used"],
                "limit": limits_cfg.get(lim_key),
                "proxy_total": collector.proxy_total(kind, multipliers, now=now),
                "reset_at": datetime.fromtimestamp(
                    bucket_end(kind, start), timezone.utc
                ).isoformat().replace("+00:00", "Z"),
            }
        payload = {
            "key": _display_key(str(meta.get("name") or "unnamed")),
            "windows": windows,
            "models": view["models"],
            "upstream": {
                "provider": "tu",
                "multiplier_note": (
                    "Aqueduct 'Limits 2x' models count 0.5 requests "
                    "(qwen-3.8-27b and its aliases vision/ocr)"
                ),
                "note": _UPSTREAM_NOTE,
                "source": _SOURCE_KEY,
            },
        }
        return JSONResponse(content=payload)

    return router
