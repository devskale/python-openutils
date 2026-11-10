"""Per-key usage counters for the /v1/usage endpoint (read-only reporting).

Tracks raw request counts per (token-hash, provider@model) in three
calendar-aligned windows — minute / hour / day, all UTC: the exact reset
boundaries Aqueduct uses (top of minute, top of hour, midnight UTC).
Limit multipliers (Aqueduct "Limits 2x" models count 0.5 requests) are
applied at READ time from provider_limits.json, so config changes apply
retroactively to the raw counts.

Read-only reporting, deliberately NOT a limiter — the adaptive AIMD limiter
was deleted as dead infrastructure in 0.8.38. Aqueduct's limits bind the
SHARED upstream TU key (all proxy clients funnel through it), so these
counters attribute per-proxy-key usage inside that aggregate quota; the
/v1/usage endpoint reports both the caller's share and the proxy total.

Only the current bucket per (key, window) is kept: used/limit/reset_at
needs no history. State snapshots to logs/key_usage.json (tiny file) and is
reloaded on startup so counters survive restarts, mirroring stats.py.
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


def weight_of(provider_model: str, multipliers: dict[str, float]) -> float:
    """Multiplier for a provider@model id; non-tu models weigh 0 (own quota domain)."""
    if not provider_model.startswith("tu@"):
        return 0.0
    bare = provider_model.split("@", 1)[1]
    return float(multipliers.get(bare, 1.0))


class KeyUsageCollector:
    """Singleton per-key usage counter (minute/hour/day windows, UTC)."""

    _instance: "KeyUsageCollector | None" = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        # token_hash -> kind -> bucket_ts -> {provider@model: raw_count}
        self._cur: dict[str, dict[str, dict[int, dict[str, int]]]] = {}
        self._last_note: dict[str, int] = {}
        self._notes_since_snapshot = 0
        self._file_lock = threading.Lock()
        self._load()

    # --- public API ---

    def note(self, token: str | None, provider_model: str, *, now: float | None = None) -> None:
        """Count one upstream-bound request for a bearer token + provider@model.

        No-op for missing token/model — keyless or malformed calls are not
        attributable usage.
        """
        if not token or not provider_model:
            return
        ts = int(now if now is not None else time.time())
        th = token_hash(token)
        with self._lock:
            if th not in self._cur:
                self._evict_if_full(ts)
            per_kind = self._cur.setdefault(th, {})
            for kind in _WINDOW_KINDS:
                start = bucket_start(kind, ts)
                buckets = per_kind.setdefault(kind, {})
                for stale in [b for b in buckets if b != start]:
                    del buckets[stale]
                buckets.setdefault(start, {})
                buckets[start][provider_model] = buckets[start].get(provider_model, 0) + 1
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
        now: float | None = None,
    ) -> dict[str, Any]:
        """Weighted windows + per-model breakdown (day window) for one token hash.

        ``used`` weights tu@ models by ``multipliers`` (default 1.0 per model);
        non-tu models weigh 0 — they draw on their own providers' quotas, not
        on the shared Aqueduct key. ``models`` lists tu@ entries only.
        """
        multipliers = multipliers or {}
        ts = int(now if now is not None else time.time())
        with self._lock:
            per_kind = self._cur.get(th, {})
            windows: dict[str, dict[str, Any]] = {}
            for kind in _WINDOW_KINDS:
                models = per_kind.get(kind, {}).get(bucket_start(kind, ts), {})
                used = sum(n * weight_of(m, multipliers) for m, n in models.items())
                windows[kind] = {"used": round(used, 3)}
            day_models = per_kind.get("day", {}).get(bucket_start("day", ts), {})
            out_models = [
                {
                    "model": m,
                    "requests": n,
                    "multiplier": weight_of(m, multipliers),
                    "weighted": round(n * weight_of(m, multipliers), 3),
                }
                for m, n in sorted(day_models.items())
                if m.startswith("tu@")
            ]
        return {"windows": windows, "models": out_models}

    def proxy_total(
        self,
        kind: str,
        multipliers: dict[str, float] | None = None,
        *,
        now: float | None = None,
    ) -> float:
        """Weighted tu@ usage across ALL keys in the current window (shared-quota view)."""
        multipliers = multipliers or {}
        ts = int(now if now is not None else time.time())
        with self._lock:
            total = 0.0
            for per_kind in self._cur.values():
                models = per_kind.get(kind, {}).get(bucket_start(kind, ts), {})
                total += sum(n * weight_of(m, multipliers) for m, n in models.items())
        return round(total, 3)

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
            th: {kind: {str(ts): dict(models) for ts, models in buckets.items()}
                 for kind, buckets in per_kind.items()}
            for th, per_kind in self._cur.items()
        }
        return {"version": 1, "keys": keys, "last_note": dict(self._last_note)}

    def _load(self) -> None:
        try:
            with open(_SNAPSHOT_PATH) as f:
                data = json.load(f)
            now = int(time.time())
            for th, per_kind in data.get("keys", {}).items():
                restored: dict[str, dict[int, dict[str, int]]] = {}
                for kind, buckets in per_kind.items():
                    if kind not in _WINDOW_KINDS:
                        continue
                    current = bucket_start(kind, now)
                    models = buckets.get(str(current))
                    if models:
                        restored[kind] = {current: {m: int(n) for m, n in models.items()}}
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
