"""Upstream unreliability pattern detection + notification for the uniioai proxy.

Hooks into ``StatsCollector.record`` (every completed request outcome — success,
upstream error, client disconnect — passes there exactly once) and flags a
``provider@model`` as *degraded* when its recent pattern matches the failure
forensics from the TU wedge incidents (2026-09-29/30):

* ``wedge_burst`` — >=N hard upstream failures (HTTP >= 500, the wedge/timeout
  class; 429 rate limits are NOT unreliability) within T minutes. Yesterday's
  incident windows: ~10 events per 45-70 min at 1-8 req/min traffic.
* ``slow_open``   — TTFT crawling toward the open-timeout boundary on a
  sustained fraction of recent streams (the "90s-crawl" mode observed while
  a replica was wedging: requests succeed but only just).
* ``error_rate``  — >=x% hard failures over >=M requests within the last hour
  (the sustained-degradation mode of 2026-09-30: ~7% over hours).

On a healthy -> degraded transition the monitor emits ONE notification through
every configured channel, plus ALWAYS a log WARNING (zero-config baseline:
visible in the journal and in /health). While degraded, re-alerts are
cooldown-suppressed; a matching ``recovered`` notice fires once the model
serves ``RECOVERY_CLEAN`` consecutive clean, fast requests again.

Channels (env, on the proxy host):

* ``UNIINFER_RELIABILITY_WEBHOOK`` — JSON POST target (Slack/Discord/generic).
* ``UNIINFER_RELIABILITY_NTFY``    — ntfy topic name (posts to ntfy.sh) or a
  full https URL for a self-hosted server; plain-text push.

State is in-memory by design: detection re-arms within minutes of real traffic
after a restart, and nothing here should ever outvote the request path —
delivery runs on a daemon thread, failures are logged, never raised.
``UNIINFER_RELIABILITY_DISABLED=1`` turns detection off entirely.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("uniioai_proxy")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, "")))
    except (TypeError, ValueError):
        return default


# --- tunables (env-overridable; defaults from the 2026-09-29/30 forensics) ----
def _cfg() -> dict[str, Any]:
    return {
        "wedge_min": _env_int("UNIINFER_RELIABILITY_WEDGE_MIN", 3),
        "wedge_window_s": _env_float("UNIINFER_RELIABILITY_WEDGE_WINDOW_S", 1800.0),
        "slow_ttft_s": _env_float("UNIINFER_RELIABILITY_SLOW_TTFT_S", 60.0),
        "slow_min": _env_int("UNIINFER_RELIABILITY_SLOW_MIN", 3),
        "slow_window_n": _env_int("UNIINFER_RELIABILITY_SLOW_WINDOW_N", 20),
        "error_rate": _env_float("UNIINFER_RELIABILITY_ERROR_RATE", 0.10),
        "error_min_n": _env_int("UNIINFER_RELIABILITY_ERROR_MIN_N", 20),
        "error_window_s": _env_float("UNIINFER_RELIABILITY_ERROR_WINDOW_S", 3600.0),
        "recovery_clean": _env_int("UNIINFER_RELIABILITY_RECOVERY_CLEAN", 10),
        "cooldown_s": _env_float("UNIINFER_RELIABILITY_COOLDOWN_S", 1800.0),
        "keep_s": _env_float("UNIINFER_RELIABILITY_KEEP_S", 7200.0),
        "disabled": os.getenv("UNIINFER_RELIABILITY_DISABLED", "").lower() in {"1", "true", "yes"},
        "webhook": os.getenv("UNIINFER_RELIABILITY_WEBHOOK", "").strip() or None,
        "ntfy": os.getenv("UNIINFER_RELIABILITY_NTFY", "").strip() or None,
    }


def _channels(cfg: dict[str, Any]) -> list[str]:
    out = []
    if cfg["webhook"]:
        out.append("webhook")
    if cfg["ntfy"]:
        out.append("ntfy")
    return out


class ReliabilityMonitor:
    """Singleton rolling-window reliability watcher (see module docstring)."""

    _instance: "ReliabilityMonitor | None" = None
    _singleton_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._init()
                    cls._instance = inst
        return cls._instance

    def _init(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Clear all live state (also the test/config-reload entry point)."""
        self._events: dict[str, deque] = {}          # model -> [(ts, status, ttft_ms)]
        self._state: dict[str, dict[str, Any]] = {}  # model -> {state, since, reason, last_alert}
        self._recovery_streak: dict[str, int] = {}
        self._history: deque = deque(maxlen=100)      # recent emitted events

    # --- ingestion -------------------------------------------------------------

    def note(self, provider_model: str, *, status: int | None, latency_ms: float | None, ttft_ms: float | None) -> None:
        """Record one request outcome and re-evaluate the model's state."""
        if not provider_model or status is None:
            return
        cfg = _cfg()
        if cfg["disabled"]:
            return
        now = time.time()
        with self._lock:
            dq = self._events.setdefault(provider_model, deque(maxlen=512))
            dq.append((now, int(status), ttft_ms))
            self._prune(provider_model, dq, now, cfg)
            self._evaluate(provider_model, dq, now, cfg)

    def _prune(self, model: str, dq: deque, now: float, cfg: dict[str, Any]) -> None:
        keep = cfg["keep_s"]
        while dq and now - dq[0][0] > keep:
            dq.popleft()

    # --- rules -----------------------------------------------------------------

    def _evaluate(self, model: str, dq: deque, now: float, cfg: dict[str, Any]) -> None:
        events = list(dq)

        # Recovery first: a model serving N consecutive clean, fast requests is
        # healthy NOW — ancient 5xx still inside the failure windows must not
        # pin it degraded (windows age out on wall-clock, recovery on behavior).
        clean = 0
        for ts, code, ttft in reversed(events):
            if code < 400 and (ttft is None or ttft < cfg["slow_ttft_s"] * 1000):
                clean += 1
            else:
                break
        self._recovery_streak[model] = clean

        st = self._state.setdefault(model, {"state": "ok", "since": now, "reason": None, "last_alert": 0.0})
        if st["state"] == "degraded" and clean >= cfg["recovery_clean"]:
            st.update(state="ok", since=now, reason=None)
            self._emit(
                model, "recovered", "recovery",
                f"{clean} consecutive clean requests (status <400, TTFT < {cfg['slow_ttft_s']:.0f}s)",
                events, now,
            )
            return

        hard = [(ts, code) for ts, code, _ in events if code >= 500]
        reason = None
        detail = ""

        wedge_window = [ts for ts, _ in hard if now - ts <= cfg["wedge_window_s"]]
        if len(wedge_window) >= cfg["wedge_min"]:
            span_min = (wedge_window[-1] - wedge_window[0]) / 60.0
            reason = "wedge_burst"
            detail = (
                f"{len(wedge_window)} hard upstream failures (5xx) within "
                f"{cfg['wedge_window_s'] / 60.0:.0f} min (span {span_min:.0f} min) — "
                f"wedge-class outage window"
            )

        if reason is None:
            streamed = [(ts, ttft) for ts, code, ttft in events if ttft is not None]
            recent = streamed[-cfg["slow_window_n"]:]
            slow = [1 for _, ttft in recent if ttft >= cfg["slow_ttft_s"] * 1000]
            if len(slow) >= cfg["slow_min"]:
                reason = "slow_open"
                detail = (
                    f"{len(slow)}/{len(recent)} recent streams took >= {cfg['slow_ttft_s']:.0f}s "
                    f"to first token — crawl mode toward the open-timeout boundary"
                )

        if reason is None:
            window = [(ts, code) for ts, code in hard if now - ts <= cfg["error_window_s"]]
            n_window = [1 for ts, _, _ in events if now - ts <= cfg["error_window_s"]]
            if len(n_window) >= cfg["error_min_n"] and len(window) / len(n_window) >= cfg["error_rate"]:
                reason = "error_rate"
                pct = round(100.0 * len(window) / len(n_window), 1)
                detail = (
                    f"{len(window)}/{len(n_window)} requests failed with 5xx "
                    f"({pct}%) within {cfg['error_window_s'] / 3600.0:.0f}h — sustained degradation"
                )

        if reason is not None:
            if st["state"] != "degraded" or now - st["last_alert"] >= cfg["cooldown_s"]:
                was = st["state"]
                st.update(state="degraded", since=now, reason=reason, last_alert=now)
                self._emit(model, "degraded" if was != "degraded" else "still_degraded", reason, detail, events, now)

    # --- notification ----------------------------------------------------------

    def _emit(self, model: str, event: str, reason: str, detail: str, events: list, now: float) -> None:
        n = len(events)
        errs = sum(1 for _, code, _ in events if code >= 500)
        rate429 = sum(1 for _, code, _ in events if code == 429)
        payload = {
            "event": event,
            "model": model,
            "reason": reason,
            "detail": detail,
            "context": {
                "recent_requests": n,
                "hard_failures": errs,
                "rate_limited_429": rate429,
            },
            "host": socket.gethostname(),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        self._history.append(payload)
        logger.warning("[reliability] %s %s (%s): %s", event, model, reason, detail)
        cfg = _cfg()
        channels = _channels(cfg)
        if not channels:
            return
        threading.Thread(target=_deliver, args=(payload, cfg), daemon=True, name="reliability-notify").start()

    # --- introspection ----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Live state for /health — one glance, no log diving."""
        cfg = _cfg()
        with self._lock:
            models = {}
            for model, st in self._state.items():
                dq = self._events.get(model)
                recent = list(dq) if dq else []
                models[model] = {
                    "state": st["state"],
                    "since": datetime.fromtimestamp(st["since"], timezone.utc).isoformat(),
                    "reason": st["reason"],
                    "recovery_streak": self._recovery_streak.get(model, 0),
                    "recent_requests": len(recent),
                    "recent_hard_failures": sum(1 for _, code, _ in recent if code >= 500),
                }
            degraded = [m for m, s in models.items() if s["state"] == "degraded"]
            return {
                "enabled": not cfg["disabled"],
                "channels": _channels(cfg) or ["log"],
                "degraded_models": degraded,
                "models": models,
                "last_events": list(self._history)[-5:],
            }

    def send_test_notification(self) -> dict[str, Any]:
        """Operator probe (POST /debug/reliability/test): exercise every
        configured channel without waiting for an incident."""
        cfg = _cfg()
        channels = _channels(cfg)
        payload = {
            "event": "test",
            "model": "n/a",
            "reason": "operator_probe",
            "detail": "reliability notification channel test",
            "host": socket.gethostname(),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        results: dict[str, str] = {}
        if not channels:
            results["log"] = "ok (no channel configured — log-only mode)"
        else:
            for name, ok, err in _deliver_sync(payload, cfg):
                results[name] = "ok" if ok else f"failed: {err}"
        return {"channels": results, "event": payload}


def _deliver(payload: dict[str, Any], cfg: dict[str, Any]) -> None:
    for _name, ok, err in _deliver_sync(payload, cfg):
        if not ok:
            logger.warning("[reliability] %s delivery failed: %s", _name, err)


def _deliver_sync(payload: dict[str, Any], cfg: dict[str, Any]) -> list[tuple[str, bool, str]]:
    """Post the event to every configured channel. Returns (name, ok, error)."""
    import httpx

    results: list[tuple[str, bool, str]] = []
    if cfg["webhook"]:
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.post(cfg["webhook"], json=payload)
            results.append(("webhook", r.status_code < 500, f"HTTP {r.status_code}"))
        except Exception as e:  # noqa: BLE001
            results.append(("webhook", False, str(e)[:200]))
    if cfg["ntfy"]:
        url = cfg["ntfy"] if cfg["ntfy"].startswith("http") else f"https://ntfy.sh/{cfg['ntfy']}"
        title = f"uniinfer {payload['event']}: {payload['model']}"
        tag = {"degraded": "warning", "still_degraded": "warning", "recovered": "white_check_mark"}.get(
            str(payload.get("event")), "loudspeaker"
        )
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.post(
                    url,
                    content=f"{payload['reason']}: {payload['detail']}".encode(),
                    headers={
                        "Title": title,
                        "Tags": tag,
                        "Priority": "high" if payload.get("event") in {"degraded", "still_degraded"} else "default",
                    },
                )
            results.append(("ntfy", r.status_code < 500, f"HTTP {r.status_code}"))
        except Exception as e:  # noqa: BLE001
            results.append(("ntfy", False, str(e)[:200]))
    return results


def get_reliability() -> ReliabilityMonitor:
    """Get the shared ReliabilityMonitor singleton."""
    return ReliabilityMonitor()
