"""Stats router: usage dashboard (JSON + HTML). Public, no auth."""
from __future__ import annotations
import os
from collections import defaultdict

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from uniinfer.config.providers import get_all_provider_limits
from uniinfer.proxy_services.stats import get_stats
from uniinfer.ratelimit import all_rate_limiter_status

_DASHBOARD_FILE = os.path.join(os.path.dirname(__file__), "..", "proxy_services", "stats_dashboard.html")
_LIMITS_DASHBOARD_FILE = os.path.join(os.path.dirname(__file__), "..", "proxy_services", "provider_limits_dashboard.html")


def _usage_by_provider(per_model: list[dict]) -> dict[str, dict[str, int]]:
    """Aggregate stats per_model entries (keyed 'provider@model') by provider."""
    agg: dict[str, dict[str, int]] = defaultdict(lambda: {"requests": 0, "tokens": 0})
    for row in per_model:
        m = row.get("model", "")
        if "@" not in m:
            continue
        prov = m.split("@", 1)[0]
        agg[prov]["requests"] += int(row.get("requests", 0) or 0)
        agg[prov]["tokens"] += int(row.get("total_tokens", 0) or 0)
    return agg


def _util_entry(limits: dict, metric: str, used: float, *, approx: bool = False, note: str | None = None) -> dict | None:
    """One utilization row: used/limit/pct for a limit metric, or None if absent."""
    if metric not in limits:
        return None
    limit = limits[metric]
    pct = round(100 * used / limit, 1) if limit else 0.0
    entry = {"metric": metric, "limit": limit, "used": round(used, 1), "pct": pct}
    if approx:
        entry["approximate"] = True
    if note:
        entry["note"] = note
    return entry


def create_stats_router() -> APIRouter:
    router = APIRouter()

    @router.get("/v1/system/stats")
    async def stats_json():
        """Aggregated usage stats: last 24h (hourly) + last 7d (daily) + per-model totals."""
        return JSONResponse(get_stats().get())

    @router.get("/v1/system/stats.html")
    async def stats_html():
        """Lightweight vanilla-JS dashboard (no build step)."""
        path = os.path.abspath(_DASHBOARD_FILE)
        if os.path.exists(path):
            return FileResponse(path, media_type="text/html")
        return HTMLResponse("<h1>stats_dashboard.html not found</h1>", status_code=404)

    @router.get("/v1/system/rate-limits")
    async def rate_limits_json():
        """Live adaptive rate-limit state per provider/model.

        Returns the learned requests/minute estimate, ceiling, active cooldown,
        current 429 streak and last re-challenge time for every (provider, model)
        the proxy has tracked. Public (read-only observability).
        """
        return JSONResponse(all_rate_limiter_status())

    @router.get("/v1/system/provider-limits")
    async def provider_limits_json():
        """Documented free-tier limits joined with live usage + utilization %.

        Public (read-only). Usage comes from the in-memory stats windows (24h /
        7d), so minute and month figures are approximations: RPM/TPM are
        24h-averages, monthly figures are 7d-projected (x30/7). Metrics without a
        matching usage window (concurrent, Neurons, unlimited) are listed in
        ``limits`` without a utilization row.
        """
        stats = get_stats().get()
        usage_24h = _usage_by_provider(stats.get("last_24h", {}).get("per_model", []))
        usage_7d = _usage_by_provider(stats.get("last_7d", {}).get("per_model", []))
        out: dict[str, dict] = {}
        for pid, lim in get_all_provider_limits().items():
            u24 = usage_24h.get(pid, {"requests": 0, "tokens": 0})
            u7 = usage_7d.get(pid, {"requests": 0, "tokens": 0})
            util = [
                e for e in (
                    _util_entry(lim, "requests_per_day", u24["requests"]),
                    _util_entry(lim, "tokens_per_day", u24["tokens"]),
                    _util_entry(lim, "requests_per_minute", u24["requests"] / 60, approx=True, note="24h-average"),
                    _util_entry(lim, "tokens_per_minute", u24["tokens"] / 60, approx=True, note="24h-average"),
                    _util_entry(lim, "requests_per_month", u7["requests"] * 30 / 7, approx=True, note="7d-projected"),
                    _util_entry(lim, "tokens_per_month", u7["tokens"] * 30 / 7, approx=True, note="7d-projected"),
                ) if e
            ]
            out[pid] = {
                "limits": lim,
                "usage": {
                    "requests_24h": u24["requests"], "tokens_24h": u24["tokens"],
                    "requests_7d": u7["requests"], "tokens_7d": u7["tokens"],
                },
                "utilization": util,
            }
        return JSONResponse({"generated_at": stats.get("generated_at"), "providers": out})

    @router.get("/v1/system/provider-limits.html")
    async def provider_limits_html():
        """Provider limits + live utilization dashboard (vanilla JS, no build)."""
        path = os.path.abspath(_LIMITS_DASHBOARD_FILE)
        if os.path.exists(path):
            return FileResponse(path, media_type="text/html")
        return HTMLResponse("<h1>provider_limits_dashboard.html not found</h1>", status_code=404)

    return router
