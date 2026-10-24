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

Ordering (v0): pinned models first (operator ``pin: true``), then personally
preferred models (per-caller ``prefer``, routing_preferences.json), then
operator ``weight`` (default 1.0, higher first), then catalog order as the
stable tiebreak. The full ordered candidate list is returned so the caller
can fall through on failure (non-streaming path).

Exploration (optional, env): with no pin/prefer deciding, the plain tier can
EXPLORE instead of always serving the same head — least-served first (models
with the fewest observed requests, random tiebreak) or uniform random,
bounded to a pool of K least-served candidates. SMART randomness: the
exploration target is HELD for a window (``UNIINFER_AUTO_EXPLORE_HOLD_S``)
so consecutive requests reuse the same model — provider prompt caches stay
warm (cache thrash is the failure mode of naive per-request rotation) and
the health index gets coherent samples; the hold releases on expiry, park,
or filter drop. The arc: explore → measure (health index fills) → exploit
(v1 score ordering supersedes least-served naturally).

Personal preferences (``avoid`` filter, ``prefer`` boost) resolve per caller:
the request bearer's sha256 maps to a token NAME via the auth-token metadata
sidecar, and that name's section in routing_preferences.json merges over the
file's ``default`` section. Direct (non-auto) calls are never affected —
preferences steer auto-routing only, like routing_exclude.

Routing metadata (chosen id, reason, candidate count) is returned with the
resolution for logging/auditing.
"""

from __future__ import annotations

import logging
import random
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
_DEAD_MARKERS = (
    "model is unavailable",
    "model not found",
    "does not exist",
    # Stream-only upstreams answering non-streaming requests: not flaky, not
    # quota — this model will 400 the same way every time. Dead for ROUTING
    # (park + auto-exclude); direct streaming calls keep working.
    "stream:true is required",
    "stream:false is not supported",
)


# ── exploration (optional): rotate the free pool, cache-friendly ────────

def _explore_cfg() -> dict:
    """UNIINFER_AUTO_EXPLORE=off|on|least-served|random (+_POOL, +_HOLD_S).

    off is the default (deterministic v0 behavior); on/least-served orders
    the plain tier by fewest observed requests (random tiebreak), random
    shuffles it. POOL bounds the exploration zone to the K least-served
    candidates; HOLD_S is the sticky window per exploration target — the
    cache-friendliness knob (per-request rotation thrashes provider prompt
    caches; a held target warms them).
    """
    import os

    raw = (os.getenv("UNIINFER_AUTO_EXPLORE", "") or "").strip().lower()
    if raw == "random":
        mode = "random"
    elif raw in {"on", "1", "true", "yes", "least-served"}:
        mode = "least-served"
    else:
        mode = "off"

    def _int_env(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, "") or default)
        except ValueError:
            return default

    def _float_env(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, "") or default)
        except ValueError:
            return default

    return {
        "mode": mode,
        "pool": max(1, _int_env("UNIINFER_AUTO_EXPLORE_POOL", 50)),
        "hold_s": max(0.0, _float_env("UNIINFER_AUTO_EXPLORE_HOLD_S", 900.0)),
    }


class _ExploreHold:
    """The current exploration target per selector, held for a window.

    Holding (instead of per-request rotation) keeps provider prompt caches
    warm and gives the health index coherent per-model samples. A hold
    invalidates itself: the resolver only honors it while the model is still
    in the filtered candidate list (parked/degraded/excluded/avoided drops
    it, and the next resolve picks fresh). In-memory; a restart re-arms from
    the least-served counters.
    """

    _state: ClassVar[dict[str, tuple[str, float]]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def get(cls, selector: str) -> tuple[str, float] | None:
        now = time.monotonic()
        with cls._lock:
            held = cls._state.get(selector)
            if held and held[1] > now:
                return held
            if held:
                del cls._state[selector]
            return None

    @classmethod
    def set(cls, selector: str, model_id: str, hold_s: float) -> None:
        with cls._lock:
            cls._state[selector] = (model_id, time.monotonic() + max(0.0, hold_s))

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._state.clear()


def _observed_counts() -> dict[str, int]:
    """Requests per model over the trailing 7d (the exploration signal)."""
    try:
        from uniinfer.proxy_services.stats import get_stats

        return {
            mid: c.get("req", 0)
            for mid, c in get_stats().merged_per_model(24 * 7).items()
        }
    except Exception as e:  # noqa: BLE001 — explore degrades to ties, never fails
        logger.debug("stats unavailable for exploration: %s", e)
        return {}


def _explore_order(
    selector: str, order: list[dict], cfg: dict, *, set_hold: bool = True
) -> tuple[list[dict], dict]:
    """Reorder the plain tier for exploration; returns (order, explore_meta).

    A live hold wins (sticky target); otherwise the zone head is picked per
    mode — least-served (n asc, random tiebreak) or random shuffle — bounded
    to the K-entry pool zone, and (optionally) held. ``set_hold=False`` for
    read-only callers (preview, fallback-chain construction): they follow
    the hold but never replace it.
    """
    ids = {m["id"] for m in order}
    held = _ExploreHold.get(selector)
    if held and held[0] in ids:
        head = next(m for m in order if m["id"] == held[0])
        meta = {"mode": cfg["mode"], "pool": cfg["pool"], "held_s": round(held[1] - time.monotonic(), 1)}
        return [head] + [m for m in order if m["id"] != held[0]], meta

    counts = _observed_counts()
    # Zone selection is ALWAYS least-served (the pool = the K least-observed
    # candidates — random mode randomizes WITHIN that zone, not across the
    # whole catalog, or heavily-served models would re-enter the draw).
    ordered = sorted(order, key=lambda m: (counts.get(m["id"], 0), random.random()))
    if cfg["mode"] == "random":
        zone = ordered[: cfg["pool"]]
        random.shuffle(zone)
        ordered = zone + ordered[cfg["pool"] :]
    if set_hold and cfg["hold_s"] > 0 and ordered:
        _ExploreHold.set(selector, ordered[0]["id"], cfg["hold_s"])
    meta = {"mode": cfg["mode"], "pool": cfg["pool"], "hold_s": cfg["hold_s"]}
    return ordered, meta


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

    @classmethod
    def dead_parks_snapshot(cls) -> dict[str, int]:
        """Consecutive dead-upstream parks per model (for the routing preview)."""
        with cls._lock:
            return dict(cls._dead_parks)


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
    """A routed model answered dead-upstream: park it long, exclude from
    ROUTING after _DEAD_PARKS consecutive parks.

    Auto-exclusion writes a ROUTING-scoped exclude (``routing_exclude``) —
    NOT a catalog deprecation: auto@free is virtual, and excluding a model
    from it must not touch the model's catalog status or block direct calls.
    The operator can lift the exclusion (delete the field or via the
    overrides API); catalog deprecate/undeprecate stay separate, deliberate
    catalog operations.
    """
    _QuotaCooldown.park(provider_model, _DEAD_COOLDOWN_S)
    with _QuotaCooldown._lock:
        _QuotaCooldown._dead_parks[provider_model] = (
            _QuotaCooldown._dead_parks.get(provider_model, 0) + 1
        )
        parks = _QuotaCooldown._dead_parks[provider_model]
    if parks >= _DEAD_PARKS_TO_DEPRECATE:
        try:
            _set_routing_exclude(
                provider_model,
                reason=f"auto: upstream answered '{str(exc)[:120]}' {parks}x",
            )
            logger.warning(
                "[auto-route] auto-excluded %s from routing after %d "
                "dead-upstream parks",
                provider_model,
                parks,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("auto-exclusion of %s failed: %s", provider_model, e)
    else:
        logger.warning(
            "[auto-route] dead-upstream %s (%d/%d parks before exclusion)",
            provider_model,
            parks,
            _DEAD_PARKS_TO_DEPRECATE,
        )


def clear_dead_parks(provider_model: str) -> None:
    """A model served successfully again: forget its dead-park streak."""
    with _QuotaCooldown._lock:
        _QuotaCooldown._dead_parks.pop(provider_model, None)


def _set_routing_exclude(
    provider_model: str, *, reason: str, source: str = "auto"
) -> None:
    """Write a ROUTING-scoped exclude for a model into model_overrides.json.

    Sets ``routing_exclude: true`` + reason/source on the model's override
    entry — scoped to auto-routing only: the catalog status stays untouched,
    direct calls keep working. Uses the Catalog's locked writer (same file,
    same serialization as every other override write) instead of a private
    write path that could race the CRUD endpoints.
    """
    bare = provider_model.split("@", 1)[1] if "@" in provider_model else provider_model
    Catalog().save_override(
        bare,
        {
            "routing_exclude": True,
            "routing_exclude_reason": reason,
            "routing_exclude_source": source,
        },
    )
    _RoutingPrefs.invalidate()


def _router_overrides_path() -> str:
    import os

    return os.path.join(os.getcwd(), _ROUTER_OVERRIDES_PATH)


class _RoutingPrefs:
    """Operator routing preferences from model_overrides.json
    (weight/pin/exclude/routing_exclude).

    Read-through with a short mtime-keyed cache: the overrides file is small
    and read once per routing decision; the cache keeps repeated auto@free
    requests off the disk. Thread-safe.
    """

    _cache: ClassVar[dict | None] = None
    _cache_mtime: ClassVar[float | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def invalidate(cls) -> None:
        with cls._lock:
            cls._cache = None
            cls._cache_mtime = None

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
                if any(
                    k in ov for k in ("weight", "pin", "exclude", "routing_exclude")
                ):
                    prefs[mid] = {
                        "weight": float(ov.get("weight", 1.0)),
                        "pin": bool(ov.get("pin", False)),
                        "exclude": bool(ov.get("exclude", False))
                        or bool(ov.get("routing_exclude", False)),
                    }
        except (OSError, ValueError) as e:
            logger.warning(
                "routing prefs load failed (%s): %s", _ROUTER_OVERRIDES_PATH, e
            )
        with cls._lock:
            cls._cache = prefs
            cls._cache_mtime = mtime
        return prefs


def _routing_prefs_path() -> str:
    import os
    from pathlib import Path

    explicit = os.getenv("UNIINFER_ROUTING_PREFS_FILE", "").strip()
    if explicit:
        return explicit
    return str(Path.home() / ".config" / "uniinfer" / "routing_preferences.json")


class _TokenNames:
    """sha256(token) → stable NAME, from the auth-token metadata sidecar.

    Routing preferences are keyed by name (human-editable); the hash→name
    mapping rides on the existing token registry sidecar next to the
    allowlist. Mtime-cached read — same pattern as the other pref caches.
    """

    _cache: ClassVar[tuple[str, float | None, dict[str, str]] | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def invalidate(cls) -> None:
        with cls._lock:
            cls._cache = None

    @classmethod
    def get(cls) -> dict[str, str]:
        import json
        import os
        from pathlib import Path

        from uniinfer.proxy_services.token_registry import metadata_path_for

        allow = os.getenv("UNIINFER_AUTH_TOKENS_FILE", "").strip()
        path = str(metadata_path_for(Path(allow))) if allow else ""
        mtime: float | None = None
        if path:
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = None
        with cls._lock:
            if (
                cls._cache is not None
                and cls._cache[0] == path
                and cls._cache[1] == mtime
            ):
                return cls._cache[2]
        names: dict[str, str] = {}
        if path and mtime is not None:
            try:
                with open(path, encoding="utf-8") as f:
                    doc = json.load(f)
                for h, rec in (doc.get("tokens") or {}).items():
                    name = rec.get("name") if isinstance(rec, dict) else None
                    if name:
                        names[str(h).lower()] = str(name)
            except (OSError, ValueError) as e:
                logger.warning("token metadata read for routing prefs failed: %s", e)
        with cls._lock:
            cls._cache = (path, mtime, names)
        return names


def _normalize_pref_section(sec: object) -> dict | None:
    """Normalize one prefer/avoid section to {providers, models} sets.

    None when the section carries nothing (absent prefs must behave exactly
    like no preferences at all). Providers lowercase; models verbatim — model
    ids are case-sensitive (`groq@moonshotai/kimi-k2-instruct`).
    """
    if not isinstance(sec, dict):
        return None

    def _sets(sub: object) -> tuple[set[str], set[str]]:
        sub = sub if isinstance(sub, dict) else {}
        provs = {
            str(x).strip().lower()
            for x in (sub.get("providers") or [])
            if str(x).strip()
        }
        models = {str(x).strip() for x in (sub.get("models") or []) if str(x).strip()}
        return provs, models

    prefer_p, prefer_m = _sets(sec.get("prefer"))
    avoid_p, avoid_m = _sets(sec.get("avoid"))
    out = {
        "prefer": {"providers": prefer_p, "models": prefer_m},
        "avoid": {"providers": avoid_p, "models": avoid_m},
    }
    if not (prefer_p or prefer_m or avoid_p or avoid_m):
        return None
    return out


def _merge_pref_sections(base: dict | None, overlay: dict | None) -> dict | None:
    """Union-merge two normalized sections (a token section can only ADD to
    the defaults — removing a default entry is a deliberate file edit, not a
    per-token override)."""
    if base is None:
        return overlay
    if overlay is None:
        return base
    return {
        key: {
            "providers": base[key]["providers"] | overlay[key]["providers"],
            "models": base[key]["models"] | overlay[key]["models"],
        }
        for key in ("prefer", "avoid")
    }


class _PersonalPrefs:
    """Caller-scoped routing preferences from routing_preferences.json.

    Optional file (absent → nobody has preferences): a ``default`` section
    plus per-token sections keyed by token NAME. Read-through with an
    mtime-keyed cache, same pattern as _RoutingPrefs — an edit takes effect
    on the next request, no restart. Thread-safe.
    """

    _cache: ClassVar[dict | None] = None
    _cache_mtime: ClassVar[float | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def invalidate(cls) -> None:
        with cls._lock:
            cls._cache = None
            cls._cache_mtime = None

    @classmethod
    def load(cls) -> dict:
        import json
        import os

        path = _routing_prefs_path()
        try:
            mtime = os.path.getmtime(path) if os.path.exists(path) else None
        except OSError:
            mtime = None
        with cls._lock:
            if cls._cache is not None and cls._cache_mtime == mtime:
                return cls._cache
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning("routing preferences load failed (%s): %s", path, e)
            doc = {}
        doc = doc if isinstance(doc, dict) else {}
        normalized = {
            "default": _normalize_pref_section(doc.get("default")),
            "tokens": {
                str(name): _normalize_pref_section(sec)
                for name, sec in (doc.get("tokens") or {}).items()
                if isinstance(sec, dict)
            },
        }
        _warn_unknown_model_entries(normalized)
        with cls._lock:
            cls._cache = normalized
            cls._cache_mtime = mtime
        return normalized


def _warn_unknown_model_entries(normalized: dict) -> None:
    """Typo protection: warn (once per file revision) about prefer/avoid model
    entries that match no catalog model — a silent typo would look like the
    preference being ignored."""
    entries: set[str] = set()
    for sec in [normalized["default"], *normalized["tokens"].values()]:
        if sec:
            entries |= sec["prefer"]["models"] | sec["avoid"]["models"]
    if not entries:
        return
    try:
        ids = {m.get("id", "") for m in Catalog().list_resolved()}
        bares = {i.split("@", 1)[1] for i in ids if "@" in i}
    except Exception as e:  # noqa: BLE001 — a warning must never break loading
        logger.debug("catalog check for routing prefs skipped: %s", e)
        return
    unknown = sorted(entries - ids - bares)
    if unknown:
        logger.warning(
            "[routing-prefs] entries match no catalog model (typo?): %s",
            ", ".join(unknown),
        )


def effective_prefs(token_hash: str | None) -> dict | None:
    """Merged prefer/avoid for one caller: token section over default.

    ``token_hash`` is the sha256 of the bearer; its NAME is resolved via the
    auth-token metadata sidecar. Anonymous callers and unknown/legacy tokens
    get the ``default`` section. Returns None when nothing is configured.
    """
    doc = _PersonalPrefs.load()
    section = doc["default"]
    if token_hash:
        name = _TokenNames.get().get(token_hash)
        if name:
            section = _merge_pref_sections(section, doc["tokens"].get(name))
    return section


def _prefs_hit(sub: dict, mid: str, bare: str, provider_id: str) -> bool:
    """Does one prefer/avoid {providers, models} entry match a candidate?
    Providers compare case-insensitively (entries are normalized lowercase,
    catalog ids are mixed-case); model entries match bare and full
    ``provider@model`` form, verbatim — model ids are case-sensitive.
    """
    return (
        provider_id.lower() in sub["providers"]
        or mid in sub["models"]
        or bare in sub["models"]
    )


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


def _rank_candidates(
    selector: str, personal: dict | None = None
) -> tuple[list[tuple[int, float, int, dict]], dict[str, list[str]]]:
    """Filter + order the candidate pool for a selector.

    Returns ``(ranked, skipped)``: ranked is the [(tier, weight, order, model)]
    list in catalog order (resolve_auto sorts it) with tier 2=operator pin,
    1=personal prefer, 0=plain; skipped the exclusion breakdown by reason
    (incl. the personal ``avoided`` filter). Split from resolve_auto so the
    routing preview can report WHY the pool is empty when every candidate is
    filtered away. ``personal`` is the caller's effective prefer/avoid section
    (None = no preferences configured).
    """
    prefs = _RoutingPrefs.load()
    degraded = _degraded_models()
    parked = _QuotaCooldown.parked()
    deprecated = _deprecated_models()

    ranked: list[tuple[int, float, int, dict]] = []
    skipped: dict[str, list[str]] = {
        "excluded": [],
        "avoided": [],
        "degraded": [],
        "quota": [],
        "parked": [],
    }
    order = 0
    for m in _candidates(selector):
        mid = m["id"]
        bare = mid.split("@", 1)[1]
        provider_id = m.get("provider") or mid.split("@", 1)[0]
        p = prefs.get(bare) or prefs.get(mid) or {}
        if p.get("exclude"):
            skipped["excluded"].append(mid)
            continue
        if bare in deprecated or mid in deprecated:
            skipped["excluded"].append(mid)
            continue
        if personal and _prefs_hit(personal["avoid"], mid, bare, provider_id):
            skipped["avoided"].append(mid)
            continue
        if mid in degraded or bare in degraded:
            skipped["degraded"].append(mid)
            continue
        if mid in parked or bare in parked:
            skipped["parked"].append(mid)
            continue
        util = _provider_utilization_pct(provider_id)
        if util is not None and util >= _QUOTA_SKIP_PCT:
            skipped["quota"].append(mid)
            continue
        tier = 2 if p.get("pin") else (
            1 if personal and _prefs_hit(personal["prefer"], mid, bare, provider_id) else 0
        )
        ranked.append((tier, p.get("weight", 1.0), order, m))
        order += 1
    return ranked, skipped


def resolve_auto(
    provider_model: str,
    *,
    token_hash: str | None = None,
    log_decision: bool = True,
    apply_hold: bool = True,
) -> tuple[str, dict] | None:
    """Resolve ``auto@<selector>`` to a concrete ``provider@model``.

    Returns (resolved_id, meta) with meta = {selector, candidates, skipped,
    chosen, reason, explore?}, or None if the input is not an auto model
    (pass-through) or no candidate survives the filters (caller decides the
    error).

    ``token_hash`` (sha256 of the request bearer) applies that caller's
    personal prefer/avoid from routing_preferences.json — None or an unknown
    token falls back to the file's ``default`` section.

    ``log_decision=False`` suppresses the ``[auto-route]`` journal line — for
    callers that resolve without serving traffic (routing preview, fallback-
    chain construction after an already-logged decision): the journal's
    routing lines must stay attributable to real request decisions.

    ``apply_hold=False``: follow (but never set) the exploration hold — the
    read-only callers' side-effect-free view.
    """
    if "@" not in provider_model:
        return None
    provider, selector = provider_model.split("@", 1)
    if provider != "auto":
        return None

    if selector not in _SELECTOR_ALIASES:
        raise ValueError(f"Unknown auto selector '{selector}'. Supported: auto@free.")

    personal = effective_prefs(token_hash)
    ranked, skipped = _rank_candidates(selector, personal)
    if not ranked:
        return None

    ranked.sort(key=lambda t: (-t[0], -t[1], t[2]))
    order = [t[3] for t in ranked]
    reason = {2: "pin", 1: "prefer", 0: "weight"}[ranked[0][0]]
    explore_meta = None
    if ranked[0][0] == 0:
        ecfg = _explore_cfg()
        if ecfg["mode"] != "off":
            order, explore_meta = _explore_order(
                selector, order, ecfg, set_hold=apply_hold
            )
            reason = "explore"
    chosen = order[0]
    meta = {
        "selector": selector,
        "candidates": len(ranked),
        "skipped": {k: len(v) for k, v in skipped.items()},
        "chosen": chosen["id"],
        "reason": reason,
        "fallbacks": [m["id"] for m in order[1:]],
    }
    if explore_meta is not None:
        meta["explore"] = explore_meta
    if log_decision:
        logger.info(
            "[auto-route] %s -> %s (%s; %d candidates, skipped: %s)",
            provider_model,
            chosen["id"],
            reason,
            len(ranked),
            meta["skipped"],
        )
    return chosen["id"], meta


def resolve_auto_candidates(
    provider_model: str, *, token_hash: str | None = None
) -> list[str]:
    """Full ordered candidate list for an auto model (fallback chain), or [].

    Quiet + caller-scoped: the chat router calls this right after its own
    (logged) resolve with the same bearer — a second identical journal line
    per request would double-count decisions, and the chain must respect the
    same personal preferences as the head choice.
    """
    resolved = resolve_auto(
        provider_model, token_hash=token_hash, log_decision=False, apply_hold=False
    )
    if resolved is None:
        return []
    mid, meta = resolved
    return [mid] + meta.get("fallbacks", [])


def routing_preview(
    provider_model: str, *, token_hash: str | None = None
) -> dict:
    """Dry-run routing state for a virtual auto model — no upstream call.

    The read-only view behind GET /v1/routing/auto: what the resolver WOULD
    choose right now (chosen, reason, fallback chain) plus the filter
    breakdown (skipped), the active cooldowns (parked, dead-park streaks),
    and the caller's applied preferences. Unlike a real request this never
    fails on an empty candidate set — an unusable router is reported
    (``resolved: null`` + why), not raised. Raises ValueError for unknown
    selectors (the endpoint maps it to 400).
    """
    personal = effective_prefs(token_hash)
    resolved = resolve_auto(
        provider_model,
        token_hash=token_hash,
        log_decision=False,
        apply_hold=False,
    )
    preview = {
        "model": provider_model,
        "resolved": resolved[0] if resolved else None,
        "reason": resolved[1].get("reason") if resolved else None,
        "candidates": resolved[1].get("candidates", 0) if resolved else 0,
        "skipped": resolved[1].get("skipped", {}) if resolved else {},
        "fallbacks": resolved[1].get("fallbacks", []) if resolved else [],
        "explore": (resolved[1].get("explore") if resolved else None)
        or _explore_cfg()["mode"],
        "parked": _QuotaCooldown.snapshot(),
        "dead_parks": _QuotaCooldown.dead_parks_snapshot(),
        "preferences": {
            "caller": _TokenNames.get().get(token_hash) if token_hash else None,
            "active": personal is not None,
        },
    }
    if resolved is None and "@" in provider_model:
        provider, selector = provider_model.split("@", 1)
        if provider == "auto" and selector in _SELECTOR_ALIASES:
            _, skipped = _rank_candidates(selector, personal)
            preview["skipped"] = {k: len(v) for k, v in skipped.items()}
    preview["preferences"]["avoided"] = int(
        (preview.get("skipped") or {}).get("avoided", 0) or 0
    )
    return preview


def virtual_model_entries() -> list[dict]:
    """Synthetic catalog entries for the routable virtual models (auto@free).

    Injected into /v1/models so clients discover the virtual id like any
    concrete model. Keeps the selector list derived from the same source the
    resolver uses — one place defines what auto@<name> exists.
    """
    entries = []
    for name in sorted(_SELECTOR_ALIASES - {"", "auto"}):
        entries.append(
            {
                "id": f"auto@{name}",
                "object": "model",
                "owned_by": "skaledev",
                "provider": "auto",
                "type": "chat",
                "access": "free",
                "name": f"Auto-Routing: best available {name} model",
                "description": (
                    "Virtual model — resolved at request time to a concrete "
                    "free provider@model (pin/weight/exclude via overrides, "
                    "fallback chain on quota/dead upstream)."
                ),
            }
        )
    return entries
