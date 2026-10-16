"""Auto-routing for virtual ``auto@<selector>`` models (v0: ``auto@free``).

Resolves a virtual model id to a concrete ``provider@model`` target at request
time, in the chat router, BEFORE ``Target`` dispatch — the rest of the pipeline
(stats, reliability, response shaping) then sees the concrete model.

Candidate set: active free models from the catalog (``models.json`` via
``Catalog().list_resolved()``), filtered by:

1. hard excludes — operator ``exclude: true`` override
2. degradation — models flagged degraded by the ReliabilityMonitor
3. provider quota pressure — provider utilization vs its documented free-tier
   limits (from ``provider_limits.json`` + live stats), skip when >= 80%

Ordering (v0, deterministic, no scoring): pinned models first
(operator ``pin: true``), then operator ``weight`` (default 1.0, higher first),
then catalog order as the stable tiebreak. The full ordered candidate list is
returned so the caller can fall through on failure (non-streaming path).

Routing metadata (chosen id, reason, candidate count) is returned with the
resolution for logging/auditing.
"""

from __future__ import annotations

import logging
import threading

from uniinfer.config.providers import get_provider_limits
from uniinfer.proxy_services.models_registry import Catalog

logger = logging.getLogger("uniioai_proxy")

DEFAULT_SELECTOR = "free"
_QUOTA_SKIP_PCT = 80.0
_SELECTOR_ALIASES = {"", "free", "auto"}
_ROUTER_OVERRIDES_PATH = "model_overrides.json"


def _router_overrides_path() -> str:
    import os

    return os.path.join(os.getcwd(), _ROUTER_OVERRIDES_PATH)


class _RoutingPrefs:
    """Operator routing preferences from model_overrides.json (weight/pin/exclude).

    Read-through with a short mtime-keyed cache: the overrides file is small
    and read once per routing decision; the cache keeps repeated auto@free
    requests off the disk. Thread-safe.
    """

    _cache: dict | None = None
    _cache_mtime: float | None = None
    _lock = threading.Lock()

    @classmethod
    def load(cls) -> dict:
        import os

        try:
            path = _router_overrides_path()
            mtime = os.path.getmtime(path) if os.path.exists(path) else None
        except OSError:
            mtime = None
        with cls._lock:
            if cls._cache is not None and cls._cache_mtime == mtime:
                return cls._cache
        prefs: dict[str, dict] = {}
        try:
            with open(_router_overrides_path()) as f:
                import json

                doc = json.load(f)
            models = doc.get("models", {}) if isinstance(doc, dict) else {}
            for mid, ov in models.items():
                if not isinstance(ov, dict):
                    continue
                if any(k in ov for k in ("weight", "pin", "exclude")):
                    prefs[mid] = {
                        "weight": float(ov.get("weight", 1.0)),
                        "pin": bool(ov.get("pin", False)),
                        "exclude": bool(ov.get("exclude", False)),
                    }
        except (OSError, ValueError) as e:
            logger.warning(
                "routing prefs load failed (%s): %s", _ROUTER_OVERRIDES_PATH, e
            )
        with cls._lock:
            cls._cache = prefs
            cls._cache_mtime = mtime
        return prefs


def _degraded_models() -> set[str]:
    try:
        from uniinfer.proxy_services.reliability import get_reliability

        snap = get_reliability().snapshot()
        return set(snap.get("degraded_models") or [])
    except Exception as e:  # noqa: BLE001
        logger.warning("reliability snapshot failed during routing: %s", e)
        return set()


def _provider_utilization_pct(provider: str) -> float | None:
    """Live provider utilization vs documented free-tier limits, in percent.

    Uses requests/day from the 7d stats window vs the provider's documented
    free-tier request limit. Returns None when no limit is documented (unknown
    pressure — treated as no penalty).
    """
    limits = get_provider_limits(provider)
    req_limit = limits.get("requests_per_day")
    if not req_limit:
        return None
    try:
        from uniinfer.proxy_services.stats import get_stats

        per_model = get_stats().get()["last_7d"].get("per_model", [])
    except Exception as e:  # noqa: BLE001
        logger.warning("stats unavailable during routing: %s", e)
        return None
    used = 0
    for row in per_model:
        m = row.get("model", "")
        if m.split("@", 1)[0] == provider:
            used += int(row.get("requests", 0) or 0)
    # 7d window vs a per-day limit: normalize used to a daily average.
    used_daily = used / 7.0
    return 100.0 * used_daily / req_limit


def _candidates(selector: str) -> list[dict]:
    if selector not in _SELECTOR_ALIASES:
        return []
    out = []
    for m in Catalog().list_resolved():
        if m.get("access") != "free" or m.get("status") != "active":
            continue
        if m.get("type") != "chat":
            continue
        out.append(m)
    return out


def resolve_auto(provider_model: str) -> tuple[str, dict] | None:
    """Resolve ``auto@<selector>`` to a concrete ``provider@model``.

    Returns (resolved_id, meta) with meta = {selector, candidates, skipped,
    chosen, reason}, or None if the input is not an auto model (pass-through)
    or no candidate survives the filters (caller decides the error).
    """
    if "@" not in provider_model:
        return None
    provider, selector = provider_model.split("@", 1)
    if provider != "auto":
        return None

    if selector not in _SELECTOR_ALIASES:
        raise ValueError(f"Unknown auto selector '{selector}'. Supported: auto@free.")

    prefs = _RoutingPrefs.load()
    degraded = _degraded_models()

    ranked: list[tuple[float, int, dict]] = []
    skipped: dict[str, list[str]] = {"excluded": [], "degraded": [], "quota": []}
    order = 0
    for m in _candidates(selector):
        mid = m["id"]
        bare = mid.split("@", 1)[1]
        p = prefs.get(bare) or prefs.get(mid) or {}
        if p.get("exclude"):
            skipped["excluded"].append(mid)
            continue
        if mid in degraded or bare in degraded:
            skipped["degraded"].append(mid)
            continue
        provider_id = m.get("provider") or mid.split("@", 1)[0]
        util = _provider_utilization_pct(provider_id)
        if util is not None and util >= _QUOTA_SKIP_PCT:
            skipped["quota"].append(mid)
            continue
        ranked.append((1.0 if p.get("pin") else 0.0, p.get("weight", 1.0), order, m))
        order += 1

    if not ranked:
        return None

    ranked.sort(key=lambda t: (-t[0], -t[1], t[2]))
    chosen = ranked[0][3]
    reason = "pin" if ranked[0][0] else "weight"
    meta = {
        "selector": selector,
        "candidates": len(ranked),
        "skipped": {k: len(v) for k, v in skipped.items()},
        "chosen": chosen["id"],
        "reason": reason,
        "fallbacks": [t[3]["id"] for t in ranked[1:]],
    }
    logger.info(
        "[auto-route] %s -> %s (%s; %d candidates, skipped: %s)",
        provider_model,
        chosen["id"],
        reason,
        len(ranked),
        meta["skipped"],
    )
    return chosen["id"], meta


def resolve_auto_candidates(provider_model: str) -> list[str]:
    """Full ordered candidate list for an auto model (fallback chain), or []."""
    resolved = resolve_auto(provider_model)
    if resolved is None:
        return []
    mid, meta = resolved
    return [mid] + meta.get("fallbacks", [])
