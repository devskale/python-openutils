"""Upstream quota truth from response headers (read-only, in-memory).

Providers that answer with x-ratelimit-* headers (Groq) expose their real
windows on EVERY response — limit/remaining per (key, model), resets as Go
durations. This store captures that truth at zero extra cost (no probing
calls) and /v1/usage surfaces it next to the proxy-counted fallback.

Dimensions are provider-specific: Groq's requests window is per DAY (RPD,
org-level — keys share it, they do NOT stack), tokens per MINUTE (TPM).
Nothing persists: remaining ages with each response, and the next response
refreshes it — a snapshot would be a lie by the time it's read.
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
from typing import Any, Mapping

_LOCK = threading.Lock()
# key_hash -> model -> {"requests"/"tokens": {...}, "ts": epoch}
_STORE: dict[str, dict[str, dict[str, Any]]] = {}

_DURATION_TOKEN = re.compile(r"(\d+(?:\.\d+)?)(h|ms|m|s)")  # ms vor m (sonst frisst m den ms-Prefix)

_DIMS = ("requests", "tokens")


def _parse_go_duration(value: str | None) -> float | None:
    """Seconds from a Go duration string ('1m26.4s', '547ms', '2m59.56s')."""
    if not value:
        return None
    total = 0.0
    matched = False
    for amount, unit in _DURATION_TOKEN.findall(value):
        matched = True
        total += float(amount) * {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}[unit]
    return total if matched else None


def note_from_headers(api_key: str | None, model: str, headers: Mapping[str, str]) -> None:
    """Capture x-ratelimit-* truth for (api_key, model) — silent no-op without
    rate-limit headers; must never break a request."""
    try:
        raw = {str(k).lower(): str(v) for k, v in headers.items()}
        if not any(f"x-ratelimit-limit-{d}" in raw for d in _DIMS):
            return
        entry: dict[str, Any] = {"ts": time.time()}
        for dim in _DIMS:
            limit = raw.get(f"x-ratelimit-limit-{dim}")
            remaining = raw.get(f"x-ratelimit-remaining-{dim}")
            reset = _parse_go_duration(raw.get(f"x-ratelimit-reset-{dim}"))
            if limit is None or remaining is None:
                continue
            entry[dim] = {
                "limit": int(limit),
                "remaining": int(remaining),
                "reset_s": reset,
            }
        if len(entry) <= 1:
            return
        kh = hashlib.sha256((api_key or "").encode()).hexdigest()
        with _LOCK:
            _STORE.setdefault(kh, {})[model] = entry
    except Exception:  # noqa: BLE001 — telemetry must never break the call
        pass


def view(key_hash: str) -> dict[str, dict[str, Any]]:
    """Per-model quota truth for one upstream key hash (copy)."""
    with _LOCK:
        return {m: dict(e) for m, e in _STORE.get(key_hash, {}).items()}


def reset() -> None:
    """Drop all captured truth (test helper)."""
    with _LOCK:
        _STORE.clear()
