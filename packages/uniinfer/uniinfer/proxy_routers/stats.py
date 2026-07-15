"""Stats router: usage dashboard (JSON + HTML). Public, no auth."""
from __future__ import annotations
import os

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from uniinfer.proxy_services.stats import get_stats
from uniinfer.ratelimit import all_rate_limiter_status

_DASHBOARD_FILE = os.path.join(os.path.dirname(__file__), "..", "proxy_services", "stats_dashboard.html")


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

    return router
