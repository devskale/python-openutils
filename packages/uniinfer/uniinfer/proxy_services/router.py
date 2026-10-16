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
import time
from typing import ClassVar

from uniinfer.config.providers import get_provider_limits
from uniinfer.proxy_services.models_registry import Catalog

logger = logging.getLogger("uniioai_proxy")

DEFAULT_SELECTOR = "free"
_QUOTA_SKIP_PCT = 80.0
_SELECTOR_ALIASES = {"", "free", "auto"}
_ROUTER_OVERRIDES_PATH = "model_overrides.json"

# Quota cooldown: a model whose upstream answers 429/402 (tokens/quota spent)
# is parked for this long before the router considers it again. Upstream
# Retry-After (when sent) wins if it is longer.
_QUOTA_COOLDOWN_S = 15 * 60
_QUOTA_COOLDOWN_MAX_S = 6 * 3600

# Dead-upstream cooldown: "Model is unavailable" / 404-not-found class. These
# are not quota (retrying in 15 min will not help) — the model is gone from
# the upstream. Parked long, and after _DEAD_PARKS consecutive long parks the
# model is auto-deprecated via the override seam (operator can undeprecate).
_DEAD_COOLDOWN_S = 60 * 60
_DEAD_PARKS_TO_DEPRECATE = 3

# Substrings marking an upstream answer as "model gone" (not quota, not flaky).
_DEAD_MARKERS = ("model is unavailable", "model not found", "does not exist")


class _QuotaCooldown:
    """Parked models: upstream said tokens/quota are spent (429 with
    retry_after, or 402-style quota errors) — do not route here until the
    cooldown expires. In-memory (a restart re-arms on the first real 429,
    same philosophy as the ReliabilityMonitor). Thread-safe.
    """

    _parked: ClassVar[dict[str, float]] = {}
    _dead_parks: ClassVar[dict[str, int]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def park(cls, model: str, seconds: float) -> None:
        until = time.monotonic() + max(0.0, seconds)
        with cls._lock:
            cls._parked[model] = until
        logger.warning(
            "[auto-route] parking %s for %.0fs (quota spent)", model, seconds
        )

    @classmethod
    def release(cls, model: str) -> None:
        with cls._lock:
            cls._parked.pop(model, None)

    @classmethod
    def parked(cls) -> set[str]:
        now = time.monotonic()
        with cls._lock:
            expired = [m for m, until in cls._parked.items() if until <= now]
            for m in expired:
                del cls._parked[m]
            return set(cls._parked)

    @classmethod
    def snapshot(cls) -> dict[str, float]:
        """Remaining park time per model, in seconds (for /health + tests)."""
        now = time.monotonic()
        with cls._lock:
            return {m: max(0.0, until - now) for m, until in cls._parked.items()}


def note_quota_hit(provider_model: str, retry_after: float | None = None) -> None:
    """A routed model answered quota-spent: park it for the cooldown.

    Called by the chat router when an auto-routed request fails with 429/402.
    An upstream Retry-After (seconds) extends the park if it is longer than
    the default; it never shortens it below the default (a Retry-After of 1s
    on a daily quota is lying).
    """
    seconds = _QUOTA_COOLDOWN_S
    if retry_after and retry_after > seconds:
        seconds = min(retry_after, _QUOTA_COOLDOWN_MAX_S)
    _QuotaCooldown.park(provider_model, seconds)


def is_dead_upstream(provider_model: str, exc: Exception) -> bool:
    """True if the failure means the model is GONE upstream (not quota/flaky).

    Matches the "Model is unavailable" 400/404 class observed on dead free
    models (step-5-preview-free on both kilo and opencode, 2026-10-16).
    """
    body = getattr(exc, "response_body", None) or ""
    msg = f"{exc} {body}".lower()
    status = getattr(exc, "status_code", None)
    if status == 404:
        return True
    return any(marker in msg for marker in _DEAD_MARKERS)


def note_dead_upstream(provider_model: str, exc: Exception) -> None:
    """A routed model answered dead-upstream: park it long, deprecate after
    _DEAD_PARKS_TO_DEPRECATE consecutive parks (auto-deprecation).

    Auto-deprecation writes status=deprecated through the override seam, so
    the model leaves the candidate set until an operator undeprecates it (or
    the upstream revives it and the catalog refresh re-lists it active — the
    override still wins, so undeprecate is the deliberate path back).
    """
    _QuotaCooldown.park(provider_model, _DEAD_COOLDOWN_S)
    with _QuotaCooldown._lock:
        _QuotaCooldown._dead_parks[provider_model] = (
            _QuotaCooldown._dead_parks.get(provider_model, 0) + 1
        )
        parks = _QuotaCooldown._dead_parks[provider_model]
    if parks >= _DEAD_PARKS_TO_DEPRECATE:
        try:
            Catalog().deprecate_model(
                provider_model,
                reason=f"auto: upstream answered '{str(exc)[:120]}' {parks}x",
                source="auto",
            )
            logger.warning(
                "[auto-route] auto-deprecated %s after %d dead-upstream parks",
                provider_model,
                parks,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("auto-deprecation of %s failed: %s", provider_model, e)
    else:
        logger.warning(
            "[auto-route] dead-upstream %s (%d/%d parks before deprecation)",
            provider_model,
            parks,
            _DEAD_PARKS_TO_DEPRECATE,
        )


def clear_dead_parks(provider_model: str) -> None:
    """A model served successfully again: forget its dead-park streak."""
    with _QuotaCooldown._lock:
        _QuotaCooldown._dead_parks.pop(provider_model, None)


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


def _deprecated_models() -> set[str]:
    """Bare model ids with a status=deprecated override (excluded from routing)."""
    try:
        overrides = Catalog().read_overrides().get("models", {})
        return {
            mid
            for mid, ov in overrides.items()
            if isinstance(ov, dict) and ov.get("status") == "deprecated"
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("overrides read failed during routing: %s", e)
        return set()


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
    parked = _QuotaCooldown.parked()
    deprecated = _deprecated_models()

    ranked: list[tuple[float, int, dict]] = []
    skipped: dict[str, list[str]] = {
        "excluded": [],
        "degraded": [],
        "quota": [],
        "parked": [],
    }
    order = 0
    for m in _candidates(selector):
        mid = m["id"]
        bare = mid.split("@", 1)[1]
        p = prefs.get(bare) or prefs.get(mid) or {}
        if p.get("exclude"):
            skipped["excluded"].append(mid)
            continue
        if bare in deprecated or mid in deprecated:
            skipped["excluded"].append(mid)
            continue
        if mid in degraded or bare in degraded:
            skipped["degraded"].append(mid)
            continue
        if mid in parked or bare in parked:
            skipped["parked"].append(mid)
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
