"""Shared proxy config + HTTP client for the uniinfer testsuite.

uniinfer's whole point is provider-agnosticism: every testsuite script reads the
SAME proxy config and exercises ANY provider through the proxy. Config precedence
(env wins; .env loaded as a fallback):

    PROXY_URL   base URL                  (default https://localhost:8123)
    PROXY_AUTH  bearer token              (also read as PROXY_KEY; open providers need none)

If PROXY_URL is unset, it is built from PROXYHOST[/PROXY_PORT][/PROXY_SCHEME].
All HTTP uses httpx with verify=False so self-signed HTTPS proxies work.
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Fail-fast liveness gate: how long to wait on a chat/readiness probe before
# declaring a model "down". Kept SHORT (single-digit seconds) so reachability
# is known within seconds, not minutes. Override with DOWN_TIMEOUT (seconds).
LIVENESS_TIMEOUT = float(os.getenv("DOWN_TIMEOUT", "8"))


def proxy_base_url() -> str:
    """Proxy base URL without a trailing slash."""
    url = os.getenv("PROXY_URL")
    if url:
        return url.rstrip("/")
    scheme = os.getenv("PROXY_SCHEME", "https")
    host = os.getenv("PROXYHOST", "localhost")
    port = os.getenv("PROXY_PORT", "8123")
    return f"{scheme}://{host}:{port}"


def proxy_auth() -> str:
    """Bearer token (PROXY_AUTH, else PROXY_KEY). Empty for open providers."""
    return os.getenv("PROXY_AUTH") or os.getenv("PROXY_KEY") or ""


def make_client(**kw) -> httpx.Client:
    """httpx.Client that tolerates self-signed HTTPS proxies."""
    kw.setdefault("verify", False)
    kw.setdefault("timeout", 180)
    return httpx.Client(**kw)


def auth_header() -> dict:
    """Authorization bearer header, or {} when no token is configured."""
    auth = proxy_auth()
    return {"Authorization": f"Bearer {auth}"} if auth else {}


# Shared result glyphs (pass / fail / skip / error).
MARKS = {"pass": "✅", "fail": "❌", "skip": "⏭️ ", "error": "💥"}


def mark(status: str) -> str:
    return MARKS.get(status, "❓")
