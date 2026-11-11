"""Per-key usage counters for the /v1/usage endpoint (read-only reporting).

Tracks raw request counts per (token-hash, upstream-key-hash, provider@model)
in three calendar-aligned windows — minute / hour / day, all UTC: the exact
reset boundaries Aqueduct uses (top of minute, top of hour, midnight UTC).
The upstream-key dimension attributes traffic to the actual quota holder:
proxy pool keys (default `tu@`, aliases like `tu-klark@`) and caller-supplied
keys (X-Provider-Key) are counted separately under their sha256.

Limit multipliers (Aqueduct "Limits 2x" models count 0.5 requests) are
applied at READ time from provider_limits.json, so config changes apply
retroactively to the raw counts.

Read-only reporting, deliberately NOT a limiter — the adaptive AIMD limiter
was deleted as dead infrastructure in 0.8.38. Failover/cooldown (slice B)
may READ these counters but never enforces here.

Only the current bucket per (key, window) is kept: used/limit/reset_at
needs no history. State snapshots to logs/key_usage.json (version 2; v1
snapshots from the pre-key-dimension era are dropped — at most one day of
day-window data) and is reloaded on startup so counters survive restarts,
mirroring stats.py.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("uniioai_proxy")

_SNAPSHOT_EVERY = 25           # persist after this many notes
_SNAPSHOT_PATH = os.path.join("logs", "key_usage.json")
_MAX_KEYS = 512
_WINDOW_KINDS = ("minute", "hour", "day")
_MINUTE = 60
_HOUR = 3600
_DAY = 86400


def bucket_start(kind: str, now: int) -> int:
    """Start timestamp of the window of ``kind`` containing ``now`` (UTC)."""
    if kind == "minute":
        return now - (now % _MINUTE)
    if kind == "hour":
        return now - (now % _HOUR)
    midnight = datetime.fromtimestamp(now, timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return int(midnight.timestamp())


def bucket_end(kind: str, start: int) -> int:
    """Exclusive end (= reset_at) of the window starting at ``start``."""
    if kind == "minute":
        return start + _MINUTE
    if kind == "hour":
        return start + _HOUR
    return start + _DAY


def token_hash(token: str) -> str:
    """sha256 hex of the bearer token — the collector never stores raw tokens."""
    return hashlib.sha256(token.encode()).hexdigest()


def upstream_key_hash(upstream_key: str | None) -> str:
    """sha256 hex of an upstream API key — identity of the quota holder."""
    return hashlib.sha256((upstream_key or "").encode()).hexdigest()


def weight_of(provider_model: str, multipliers: dict[str, float]) -> float:
    """Request weight for a provider@model id (provider-scoped multipliers).

    ``multipliers`` is ONE provider's map (bare model name -> weight, from
    provider_limits.json request_multipliers); unknown models weigh 1.0. Callers
    scope the view to one provider's models — the collector itself stays
    provider-agnostic (no provider hardcode).
    """
    bare = provider_model.split("@", 1)[1] if "@" in provider_model else provider_model
    return float(multipliers.get(bare, 1.0))


def entity_model(provider_model: str) -> str:
    """Normalize an alias-prefixed model to its ENTITY provider.

    ``tu-klark@qwen-3.8-27b`` counts toward the ``tu`` entity; custom aliases
    over generic classes (``novita-2`` → openai-compat) group via the overlay's
    ``entity`` field (fallback: spec.provider — for builtins class == entity).
    Unknown aliases pass through unchanged; resolution failures never break
    counting.
    """
    if "@" not in provider_model:
        return provider_model
    alias, rest = provider_model.split("@", 1)
    try:
        from uniinfer.config.instances import resolve_instance

        spec = resolve_instance(alias)
        entity = getattr(spec, "entity", None) or spec.provider or alias
        return f"{entity}@{rest}" if entity != alias else provider_model
    except Exception:  # noqa: BLE001 — counting must never break on config issues
        return provider_model


class KeyUsageCollector:
    """Singleton usage counter: (caller × upstream-key × model), UTC windows."""

    _instance: "KeyUsageCollector | None" = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        # token_hash -> kind -> bucket_ts -> key_hash -> {provider@model: raw_count}
        self._cur: dict[str, dict[str, dict[int, dict[str, dict[str, int]]]]] = {}
        self._last_note: dict[str, int] = {}
        self._notes_since_snapshot = 0
        self._file_lock = threading.Lock()
        self._load()

    # --- public API ---

    def note(
        self,
        token: str | None,
        provider_model: str,
        upstream_key: str | None = None,
        *,
        now: float | None = None,
    ) -> None:
        """Count one upstream-bound request for a bearer token + provider@model.

        ``upstream_key`` is the RESOLVED upstream API key material (pool key or
        caller-supplied) — hashed immediately, never stored. None/missing token
        or model is not attributable usage (no-op).
        """
        if not token or not provider_model:
            return
        ts = int(now if now is not None else time.time())
        th = token_hash(token)
        kh = upstream_key_hash(upstream_key)
        provider_model = entity_model(provider_model)
        with self._lock:
            if th not in self._cur:
                self._evict_if_full(ts)
            per_kind = self._cur.setdefault(th, {})
            for kind in _WINDOW_KINDS:
                start = bucket_start(kind, ts)
                buckets = per_kind.setdefault(kind, {})
                for stale in [b for b in buckets if b != start]:
                    del buckets[stale]
                models = buckets.setdefault(start, {}).setdefault(kh, {})
                models[provider_model] = models.get(provider_model, 0) + 1
            self._last_note[th] = ts
            self._notes_since_snapshot += 1
            if self._notes_since_snapshot >= _SNAPSHOT_EVERY:
                self._notes_since_snapshot = 0
                self.snapshot()

    def usage_view(
        self,
        th: str,
        multipliers: dict[str, float] | None = None,
        *,
        provider: str | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Weighted windows + per-model breakdown (day window) for one token hash.

        Attribution across ALL upstream keys the caller used (pool + own).
        ``provider`` scopes the view to one provider's models (None = all
        providers; weights then apply per-model via ``multipliers``).
        """
        multipliers = multipliers or {}
        prefix = f"{provider}@" if provider else None
        ts = int(now if now is not None else time.time())
        with self._lock:
            per_kind = self._cur.get(th, {})
            windows: dict[str, dict[str, Any]] = {}
            day_models: dict[str, int] = {}
            for kind in _WINDOW_KINDS:
                used = 0.0
                for models in per_kind.get(kind, {}).get(bucket_start(kind, ts), {}).values():
                    for m, n in models.items():
                        if prefix and not m.startswith(prefix):
                            continue
                        used += n * weight_of(m, multipliers)
                        if kind == "day":
                            day_models[m] = day_models.get(m, 0) + n
                windows[kind] = {"used": round(used, 3)}
            out_models = [
                {
                    "model": m,
                    "requests": n,
                    "multiplier": weight_of(m, multipliers),
                    "weighted": round(n * weight_of(m, multipliers), 3),
                }
                for m, n in sorted(day_models.items())
            ]
        return {"windows": windows, "models": out_models}

    def provider_pool_view(
        self,
        provider: str,
        multipliers: dict[str, float] | None = None,
        *,
        now: float | None = None,
    ) -> dict[str, dict[str, float]]:
        """Per-upstream-key weighted usage (proxy-wide, all callers) for a provider.

        Returns {key_hash: {"minute": x, "hour": y, "day": z}} covering ONLY the
        given provider's models. Caller-supplied keys (hashes unknown to the
        pool registry) appear here too — the router filters by label map.
        """
        multipliers = multipliers or {}
        ts = int(now if now is not None else time.time())
        prefix = f"{provider}@"
        out: dict[str, dict[str, float]] = {}
        with self._lock:
            for per_kind in self._cur.values():
                for kind in _WINDOW_KINDS:
                    models_by_key = per_kind.get(kind, {}).get(bucket_start(kind, ts), {})
                    for kh, models in models_by_key.items():
                        used = sum(
                            n * weight_of(m, multipliers)
                            for m, n in models.items()
                            if m.startswith(prefix)
                        )
                        if used:
                            out.setdefault(kh, {"minute": 0.0, "hour": 0.0, "day": 0.0})
                            out[kh][kind] += used
        return {kh: {k: round(v, 3) for k, v in w.items()} for kh, w in out.items()}

    def active_providers(self) -> set[str]:
        """Providers with any counted traffic in the current day window."""
        ts = int(time.time())
        providers: set[str] = set()
        with self._lock:
            for per_kind in self._cur.values():
                for models in per_kind.get("day", {}).get(bucket_start("day", ts), {}).values():
                    for m in models:
                        providers.add(m.split("@", 1)[0])
        return providers

    def snapshot(self) -> None:
        """Persist current state to logs/key_usage.json."""
        with self._file_lock:
            try:
                os.makedirs(os.path.dirname(_SNAPSHOT_PATH), exist_ok=True)
                tmp = _SNAPSHOT_PATH + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(self._serialize(), f)
                os.replace(tmp, _SNAPSHOT_PATH)
            except Exception as e:  # noqa: BLE001
                logger.warning("key_usage snapshot failed: %s", e)

    # --- internals ---

    def _evict_if_full(self, ts: int) -> None:
        """Cap tracked keys; evict the least-recently-noted one."""
        if len(self._cur) < _MAX_KEYS:
            return
        oldest = min(self._cur, key=lambda k: self._last_note.get(k, 0))
        del self._cur[oldest]
        self._last_note.pop(oldest, None)

    def _serialize(self) -> dict[str, Any]:
        keys = {
            th: {kind: {str(ts): {kh: dict(models) for kh, models in by_key.items()}
                        for ts, by_key in buckets.items()}
                 for kind, buckets in per_kind.items()}
            for th, per_kind in self._cur.items()
        }
        return {"version": 2, "keys": keys, "last_note": dict(self._last_note)}

    def _load(self) -> None:
        try:
            with open(_SNAPSHOT_PATH) as f:
                data = json.load(f)
            version = int(data.get("version", 1))
            if version != 2:
                logger.info("key_usage snapshot v%s dropped (pre-key-dimension era)", version)
                return
            now = int(time.time())
            for th, per_kind in data.get("keys", {}).items():
                restored: dict[str, dict[int, dict[str, dict[str, int]]]] = {}
                for kind, buckets in per_kind.items():
                    if kind not in _WINDOW_KINDS:
                        continue
                    current = bucket_start(kind, now)
                    by_key = buckets.get(str(current))
                    if by_key:
                        restored[kind] = {
                            current: {kh: {m: int(n) for m, n in models.items()}
                                      for kh, models in by_key.items()}
                        }
                if restored:
                    self._cur[th] = restored
            self._last_note = {
                th: ts for th, ts in data.get("last_note", {}).items() if th in self._cur
            }
        except FileNotFoundError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.warning("key_usage load failed (starting fresh): %s", e)


def get_key_usage() -> KeyUsageCollector:
    """Get the shared KeyUsageCollector singleton."""
    return KeyUsageCollector()
