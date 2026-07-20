"""Proxy router for web tools: search and fetch-url."""
from __future__ import annotations

import logging
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request

logger = logging.getLogger(__name__)

PUBLIC_SEARXNG_INSTANCES = [
    "https://searx.be",
    "https://search.bus-hit.me",
    "https://search.rowie.at",
    "https://searx.fmac.xyz",
]

JINA_READER_URL = "https://r.jina.ai"


def _get_searxng_config() -> Optional[dict]:
    try:
        import io
        import contextlib
        from credgoo.credgoo import get_api_key
        with contextlib.redirect_stdout(io.StringIO()):
            creds = get_api_key("searx")
        if creds and "@" in creds:
            parts = creds.split("@")
            url = parts[0]
            user = parts[1] if len(parts) > 1 else ""
            pw = parts[2] if len(parts) > 2 else ""
            return {"url": url, "auth": (user, pw) if user else None}
    except Exception:
        pass
    return None


def _get_duck_token() -> Optional[str]:
    try:
        import io
        import contextlib
        from credgoo.credgoo import get_api_key
        with contextlib.redirect_stdout(io.StringIO()):
            return get_api_key("WEB_SEARCH_BEARER")
    except Exception:
        pass
    return None


def create_tools_router() -> APIRouter:
    router = APIRouter(prefix="/v1/tools", tags=["tools"])

    @router.get("/search")
    async def web_search(
        request: Request,
        q: str = Query(..., description="Search query"),
        max_results: int = Query(10, ge=1, le=30),
    ):
        query = q.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query parameter 'q' is required")

        last_err = None

        # 1. Try private SearXNG
        searx_config = _get_searxng_config()
        if searx_config:
            try:
                params = {"q": query, "format": "json"}
                headers = {"User-Agent": "UniInfer/1.0"}
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(
                        f"{searx_config['url']}/search",
                        params=params,
                        headers=headers,
                        auth=searx_config.get("auth"),
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    results = [
                        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
                        for r in data.get("results", [])[:max_results]
                    ]
                    return {
                        "query": query,
                        "results": results,
                        "suggestions": data.get("suggestions", []),
                        "number_of_results": data.get("number_of_results", len(results)),
                    }
            except Exception as e:
                last_err = str(e)
                logger.debug("Private SearXNG failed: %s", e)

        # 2. Try Duck API
        duck_token = _get_duck_token()
        if duck_token:
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(
                        "https://amd1.mooo.com/api/duck/search",
                        params={"query": query},
                        headers={"Authorization": f"Bearer {duck_token}", "User-Agent": "UniInfer/1.0"},
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    results = [
                        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
                        for r in data.get("results", [])[:max_results]
                    ]
                    return {"query": query, "results": results, "suggestions": [], "number_of_results": len(results)}
            except Exception as e:
                last_err = str(e)
                logger.debug("Duck API failed: %s", e)

        # 3. Public SearXNG fallback
        for instance in PUBLIC_SEARXNG_INSTANCES:
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(
                        f"{instance}/search",
                        params={"q": query, "format": "json", "engines": "google,bing,duckduckgo"},
                        headers={"User-Agent": "UniInfer/1.0"},
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    results = [
                        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
                        for r in data.get("results", [])[:max_results]
                    ]
                    return {
                        "query": query,
                        "results": results,
                        "suggestions": data.get("suggestions", []),
                        "number_of_results": data.get("number_of_results", len(results)),
                    }
            except Exception as e:
                last_err = str(e)
                logger.debug("Public SearXNG %s failed: %s", instance, e)

        raise HTTPException(status_code=502, detail=f"All search backends failed: {last_err}")

    @router.get("/fetch")
    async def fetch_url(
        request: Request,
        url: str = Query(..., description="URL to fetch"),
    ):
        target = url.strip()
        if not target:
            raise HTTPException(status_code=400, detail="URL parameter 'url' is required")

        if not target.startswith(("http://", "https://")):
            target = "https://" + target

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(
                    f"{JINA_READER_URL}/{target}",
                    headers={
                        "Accept": "text/plain",
                        "User-Agent": "UniInfer/1.0",
                    },
                )
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"Jina Reader returned HTTP {resp.status_code} for {target}",
                )
            text = resp.text
            if len(text) > 50000:
                text = text[:50000] + "\n\n... [truncated]"
            return {"url": target, "content": text, "length": len(text)}
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail=f"Timeout fetching {target}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Fetch failed: {e}")

    return router
