"""Lightweight in-memory usage statistics for the uniioai proxy.

Tracks per-model request/error/token/latency metrics in two rolling windows:
- hourly buckets for the last 24h (granular)
- daily buckets for the last 7d (coarser)

Footprint is tiny (a few KB): buckets are pruned as they age out. State is
snapshotted to logs/stats.json periodically and on shutdown, and reloaded on
startup so counters survive restarts.

Designed for memory-constrained hosts — no DB, no background thread, no large
in-memory structures.
"""
from __future__ import annotations
import json
import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("uniioai_proxy")

_HOUR = 3600
_DAY = 86400
_HOURLY_KEEP = 24 * _HOUR      # 24h of hourly buckets
_DAILY_KEEP = 7 * _DAY         # 7d of daily buckets
_SNAPSHOT_EVERY = 50           # persist after this many records
_SNAPSHOT_PATH = os.path.join("logs", "stats.json")


def _empty_counters() -> dict[str, int]:
    return {"req": 0, "errors": 0, "prompt": 0, "completion": 0, "total": 0, "latency_sum": 0, "latency_n": 0}


class StatsCollector:
    """Singleton stats collector with rolling hourly (24h) and daily (7d) windows."""

    _instance: "StatsCollector | None" = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        self._hourly: dict[int, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(_empty_counters))
        self._daily: dict[int, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(_empty_counters))
        # Status-code counts per window: ts -> {"200": n, "400": n, ...}
        self._status_hourly: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._status_daily: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._status_total: dict[str, int] = defaultdict(int)
        # Per-model all-time totals (kept small — one entry per distinct model).
        self._totals: dict[str, dict[str, int]] = defaultdict(_empty_counters)
        self._since = time.time()
        self._records_since_snapshot = 0
        self._file_lock = threading.Lock()
        self._load()

    # --- public API ---

    def record(
        self,
        provider_model: str,
        *,
        status: int | None,
        latency_ms: float | None,
        usage: dict[str, Any] | None,
    ) -> None:
        """Record one completed request."""
        if not provider_model:
            return
        now = int(time.time())
        prompt = int((usage or {}).get("prompt_tokens") or 0)
        completion = int((usage or {}).get("completion_tokens") or 0)
        total = int((usage or {}).get("total_tokens") or (prompt + completion))
        is_error = 1 if (status is not None and status >= 400) else 0
        lat = int(latency_ms or 0)

        hour = now - (now % _HOUR)
        day = now - (now % _DAY)

        code = str(status) if status is not None else "unknown"
        self._status_hourly[hour][code] += 1
        self._status_daily[day][code] += 1
        self._status_total[code] += 1

        for store, key in ((self._hourly, hour), (self._daily, day)):
            c = store[key][provider_model]
            c["req"] += 1
            c["errors"] += is_error
            c["prompt"] += prompt
            c["completion"] += completion
            c["total"] += total
            if lat:
                c["latency_sum"] += lat
                c["latency_n"] += 1

        t = self._totals[provider_model]
        t["req"] += 1
        t["errors"] += is_error
        t["prompt"] += prompt
        t["completion"] += completion
        t["total"] += total
        if lat:
            t["latency_sum"] += lat
            t["latency_n"] += 1

        self._records_since_snapshot += 1
        if self._records_since_snapshot >= _SNAPSHOT_EVERY:
            self._records_since_snapshot = 0
            self.snapshot()

    def snapshot(self) -> None:
        """Persist current state to logs/stats.json."""
        with self._file_lock:
            try:
                os.makedirs(os.path.dirname(_SNAPSHOT_PATH), exist_ok=True)
                tmp = _SNAPSHOT_PATH + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(self._serialize(), f)
                os.replace(tmp, _SNAPSHOT_PATH)
            except Exception as e:  # noqa: BLE001
                logger.warning("stats snapshot failed: %s", e)

    def get(self) -> dict[str, Any]:
        """Return aggregated stats for the dashboard/API."""
        now = int(time.time())
        self._prune(now)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "since": datetime.fromtimestamp(self._since, timezone.utc).isoformat(),
            "last_24h": self._aggregate(self._hourly, self._status_hourly),
            "last_7d": self._aggregate(self._daily, self._status_daily),
            "per_model_total": self._per_model(self._totals),
            "status_total": dict(self._status_total),
        }

    # --- internals ---

    def _prune(self, now: int) -> None:
        cutoff_h = now - _HOURLY_KEEP
        cutoff_d = now - _DAILY_KEEP
        for k in [k for k in self._hourly if k < cutoff_h]:
            del self._hourly[k]
        for k in [k for k in self._daily if k < cutoff_d]:
            del self._daily[k]
        for k in [k for k in self._status_hourly if k < cutoff_h]:
            del self._status_hourly[k]
        for k in [k for k in self._status_daily if k < cutoff_d]:
            del self._status_daily[k]

    @staticmethod
    def _aggregate(buckets: dict[int, dict[str, dict[str, int]]], status_buckets: dict[int, dict[str, int]]) -> dict[str, Any]:
        """Collapse time-buckets into a per-model summary + a time series + status codes."""
        per_model: dict[str, dict[str, int]] = defaultdict(_empty_counters)
        series: list[dict[str, Any]] = []
        status_totals: dict[str, int] = defaultdict(int)
        for ts in sorted(buckets):
            models = buckets[ts]
            bucket_req = bucket_err = bucket_prompt = bucket_completion = bucket_total = 0
            for model, c in models.items():
                agg = per_model[model]
                for k in ("req", "errors", "prompt", "completion", "total", "latency_sum", "latency_n"):
                    agg[k] += c[k]
                bucket_req += c["req"]
                bucket_err += c["errors"]
                bucket_prompt += c["prompt"]
                bucket_completion += c["completion"]
                bucket_total += c["total"]
            for code, n in status_buckets.get(ts, {}).items():
                status_totals[code] += n
            series.append({
                "ts": ts,
                "req": bucket_req,
                "errors": bucket_err,
                "prompt_tokens": bucket_prompt,
                "completion_tokens": bucket_completion,
                "total_tokens": bucket_total,
            })
        return {"series": series, "per_model": StatsCollector._per_model(per_model), "status_codes": dict(status_totals)}

    @staticmethod
    def _per_model(src: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
        out = []
        for model, c in src.items():
            lat_n = c["latency_n"] or 1
            out.append({
                "model": model,
                "requests": c["req"],
                "errors": c["errors"],
                "error_rate": round(c["errors"] / c["req"], 4) if c["req"] else 0.0,
                "prompt_tokens": c["prompt"],
                "completion_tokens": c["completion"],
                "total_tokens": c["total"],
                "avg_latency_ms": round(c["latency_sum"] / lat_n) if c["latency_n"] else 0,
            })
        out.sort(key=lambda r: r["requests"], reverse=True)
        return out

    def _serialize(self) -> dict[str, Any]:
        return {
            "since": self._since,
            "hourly": {str(k): {m: dict(v) for m, v in models.items()} for k, models in self._hourly.items()},
            "daily": {str(k): {m: dict(v) for m, v in models.items()} for k, models in self._daily.items()},
            "status_hourly": {str(k): dict(v) for k, v in self._status_hourly.items()},
            "status_daily": {str(k): dict(v) for k, v in self._status_daily.items()},
            "status_total": dict(self._status_total),
            "totals": {m: dict(v) for m, v in self._totals.items()},
        }

    def _load(self) -> None:
        try:
            with open(_SNAPSHOT_PATH) as f:
                data = json.load(f)
            self._since = data.get("since", time.time())
            for k, models in data.get("hourly", {}).items():
                self._hourly[int(k)] = {m: dict(v) for m, v in models.items()}
            for k, models in data.get("daily", {}).items():
                self._daily[int(k)] = {m: dict(v) for m, v in models.items()}
            for k, v in data.get("status_hourly", {}).items():
                self._status_hourly[int(k)] = dict(v)
            for k, v in data.get("status_daily", {}).items():
                self._status_daily[int(k)] = dict(v)
            for m, v in data.get("totals", {}).items():
                self._totals[m] = dict(v)
            self._status_total.update(data.get("status_total", {}))
        except FileNotFoundError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.warning("stats load failed (starting fresh): %s", e)


def get_stats() -> StatsCollector:
    """Get the shared StatsCollector singleton."""
    return StatsCollector()
