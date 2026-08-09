"""System endpoints — webdemo, perf dashboard, capabilities, integration guide.

Split from proxy_app.py for locality: these are static-file-serving + JSON-data
endpoints, distinct from the app factory, health probe, and API routers.
"""
from __future__ import annotations

import json
import os
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

logger = logging.getLogger("uniioai_proxy")

_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WEBDEMO_DIR = os.path.join(_SCRIPT_DIR, "examples", "webdemo")
_SPEED_RESULTS = os.path.join(_SCRIPT_DIR, "models", "_speed_results.json")
_PROBE_RESULTS = os.path.join(_SCRIPT_DIR, "models", "_probe_results.json")


def create_system_router(version: str) -> APIRouter:
    router = APIRouter()

    @router.get("/webdemo", include_in_schema=False)
    async def get_web_demo():
        """Serve webdemo HTML with cache-bust."""
        html_file_path = os.path.join(_WEBDEMO_DIR, "webdemo.html")
        if not os.path.exists(html_file_path):
            raise HTTPException(status_code=404, detail="webdemo.html not found")
        with open(html_file_path, encoding="utf-8") as f:
            html = f.read()
        build = str(int(max(
            os.path.getmtime(os.path.join(_WEBDEMO_DIR, fn))
            for fn in os.listdir(_WEBDEMO_DIR)
            if os.path.isfile(os.path.join(_WEBDEMO_DIR, fn))
        )))
        return HTMLResponse(html.replace("__BUILD__", build))

    @router.get("/perf", include_in_schema=False)
    async def get_perf_dashboard():
        html_file_path = os.path.join(_WEBDEMO_DIR, "perf.html")
        if not os.path.exists(html_file_path):
            raise HTTPException(status_code=404, detail="perf.html not found")
        return FileResponse(html_file_path)

    @router.get("/perf/results", include_in_schema=False)
    async def get_perf_results():
        if not os.path.exists(_SPEED_RESULTS):
            return {}
        try:
            with open(_SPEED_RESULTS, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    @router.post("/perf/results", include_in_schema=False)
    async def save_perf_result(request: Request):
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")
        key = body.get("key")
        result = body.get("result")
        if not key or not isinstance(result, dict):
            raise HTTPException(status_code=400, detail="Body must contain 'key' and 'result'")
        existing = {}
        if os.path.exists(_SPEED_RESULTS):
            try:
                with open(_SPEED_RESULTS, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing[key] = result
        os.makedirs(os.path.dirname(_SPEED_RESULTS), exist_ok=True)
        with open(_SPEED_RESULTS, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        return {"ok": True, "saved": key}

    @router.get("/capabilities", include_in_schema=False)
    async def get_capabilities_dashboard():
        html_file_path = os.path.join(_WEBDEMO_DIR, "capabilities.html")
        if not os.path.exists(html_file_path):
            raise HTTPException(status_code=404, detail="capabilities.html not found")
        return FileResponse(html_file_path)

    @router.get("/capabilities/results", include_in_schema=False)
    async def get_capabilities_results():
        if not os.path.exists(_PROBE_RESULTS):
            return {}
        try:
            with open(_PROBE_RESULTS, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    @router.get("/guide", include_in_schema=False)
    async def get_integration_guide():
        html_file_path = os.path.join(_WEBDEMO_DIR, "guide.html")
        if not os.path.exists(html_file_path):
            raise HTTPException(status_code=404, detail="guide.html not found")
        return FileResponse(html_file_path)

    @router.get("/guide.md", include_in_schema=False)
    async def get_integration_guide_md():
        md_file_path = os.path.join(_SCRIPT_DIR, "..", "docs", "integration.md")
        if not os.path.exists(md_file_path):
            raise HTTPException(status_code=404, detail="integration.md not found")
        return FileResponse(md_file_path, media_type="text/markdown")

    @router.get("/", include_in_schema=False)
    async def root():
        """Serve the unified web app."""
        html = os.path.join(_WEBDEMO_DIR, "webdemo.html")
        if not os.path.exists(html):
            raise HTTPException(status_code=404, detail="webdemo.html not found")
        return FileResponse(html)

    return router
