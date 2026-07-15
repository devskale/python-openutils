"""Self-tuning per-provider / per-model request-rate limiter.

Backend "requests per minute" limits are rarely what providers advertise.
TU/Aqueduct, for example, returns ``Rate limit exceeded. Request limit
(25/min).`` while the real effective ceiling (especially for heavy models
such as ``glm-5.2-744b-preview``) is closer to 5/min, and it can be raised
to 100 or 1000/min in a future upgrade without any notice.

This module implements an **AIMD** (Additive-Increase / Multiplicative-Decrease)
controller -- the same idea behind TCP congestion control -- so a provider
learns and respects its real limit automatically:

* **Enforcement**: a sliding 60s window caps how many requests we send for a
  given ``(provider, model)`` key, with an even temporal spacing derived from
  the current estimate.
* **Multiplicative decrease on HTTP 429**: when a 429 slips through, the real
  limit is inferred from the size of the burst that just tripped it, and the
  estimate is halved (or dropped to the observed-safe count). An exponential
  cooldown is applied so we do not immediately re-hammer the backend.
* **Additive increase on success**: after a sustained 429-free period the
  estimate is nudged upward toward the ceiling. This keeps us near the limit
  without the endless sawtooth churn of blind probing.
* **Daily re-challenge**: at least every 24h the ceiling is restored and the
  estimate is deliberately probed higher. If the provider has been upgraded we
  climb to the new limit; if not, the next 429 simply halves us again. This is
  how a silent 25 -> 100 -> 1000/min upgrade is discovered instead of missed.
* **Persistence**: learned estimates survive process restarts (best-effort,
  never fatal) so we do not re-learn from scratch on every deploy.

The limiter is safe to share across many concurrent coroutines: all state
mutations are guarded by a lock and the (possibly long) sleep happens outside
the lock. It is also deterministic and unit-testable via an injectable clock.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


class _KeyState:
    """Mutable per-(provider, model) rate-limit state."""

    __slots__ = (
        "rpm",
        "ceiling",
        "window",
        "consecutive_429",
        "last_429",
        "last_success",
        "last_rechallenge",
        "cooldown_until",
    )

    def __init__(self, rpm: float, ceiling: float) -> None:
        self.rpm = float(rpm)
        self.ceiling = float(ceiling)
        self.window: deque[datetime] = deque()
        self.consecutive_429 = 0
        self.last_429: Optional[datetime] = None
        self.last_success: Optional[datetime] = None
        self.last_rechallenge = datetime.min
        self.cooldown_until: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rpm": self.rpm,
            "ceiling": self.ceiling,
            "consecutive_429": self.consecutive_429,
            "last_429": self.last_429.isoformat() if self.last_429 else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_rechallenge": self.last_rechallenge.isoformat() if self.last_rechallenge != datetime.min else None,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], rpm: float, ceiling: float) -> "_KeyState":
        def _num(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        st = cls(_num(data.get("rpm", rpm), rpm), _num(data.get("ceiling", ceiling), ceiling))
        raw_lr = data.get("last_rechallenge")
        if raw_lr:
            try:
                st.last_rechallenge = datetime.fromisoformat(raw_lr)
            except (TypeError, ValueError):
                st.last_rechallenge = datetime.min
        return st


class AdaptiveRateLimiter:
    """AIMD-based, self-tuning rate limiter keyed per model within a provider."""

    def __init__(
        self,
        provider_id: str,
        *,
        default_rpm: float = 25.0,
        min_rpm: Optional[float] = None,
        ceiling_rpm: Optional[float] = None,
        window_seconds: Optional[float] = None,
        rechallenge_hours: Optional[float] = None,
        persist_path: Optional[str] = None,
        now_provider: Optional[Callable[[], datetime]] = None,
        on_rechallenge: Optional[Callable[[str, "_KeyState"], None]] = None,
        async_sleep: Optional[Callable[[float], "Any"]] = None,
        sync_sleep: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.provider_id = provider_id
        self.default_rpm = float(os.getenv("UNIINFER_RATE_LIMIT_DEFAULT", str(default_rpm)) or default_rpm)
        self.min_rpm = float(min_rpm if min_rpm is not None else _env_float("UNIINFER_RATE_LIMIT_MIN", 0.5))
        self.ceiling_rpm = float(ceiling_rpm if ceiling_rpm is not None else _env_float("UNIINFER_RATE_LIMIT_CEILING", 1000.0))
        self.window_seconds = float(window_seconds if window_seconds is not None else _env_float("UNIINFER_RATE_LIMIT_WINDOW", 60.0))
        self.rechallenge_interval = timedelta(hours=float(rechallenge_hours if rechallenge_hours is not None else _env_int("UNIINFER_RATE_LIMIT_RECHALLENGE_HOURS", 24)))
        self._window_delta = timedelta(seconds=self.window_seconds)
        self._now = now_provider or datetime.now
        self._async_sleep = async_sleep or asyncio.sleep
        self._sync_sleep = sync_sleep or time.sleep
        self._on_rechallenge = on_rechallenge

        self.additive = 1.0
        self.multiplicative = 0.5
        self.stable_threshold = timedelta(minutes=5)
        self.cooldown_base = timedelta(seconds=5)
        self.cooldown_max = timedelta(seconds=120)
        self.rechallenge_step_abs = 5.0
        self.rechallenge_step_rel = 0.5

        self._states: dict[str, _KeyState] = {}
        self._lock = threading.RLock()
        self._save_counter = 0

        persist = persist_path if persist_path is not None else os.getenv("UNIINFER_RATE_LIMIT_PERSIST", "_rate_limits.json")
        self._persist_path = persist or None
        self._load()

    def _state_for(self, model: str) -> _KeyState:
        st = self._states.get(model)
        if st is None:
            persisted = self._persisted.get(model) if self._persisted else None
            if persisted:
                st = _KeyState.from_dict(persisted, self.default_rpm, self.ceiling_rpm)
            else:
                st = _KeyState(self.default_rpm, self.ceiling_rpm)
            if st.last_rechallenge == datetime.min:
                st.last_rechallenge = self._now()
            self._states[model] = st
        return st

    def _prune(self, st: _KeyState, now: datetime) -> None:
        cutoff = now - self._window_delta
        while st.window and st.window[0] < cutoff:
            st.window.popleft()

    def _compute_wait(self, st: _KeyState, now: datetime) -> float:
        wait = 0.0
        if st.cooldown_until is not None and st.cooldown_until > now:
            wait = max(wait, (st.cooldown_until - now).total_seconds())
        rpm = st.rpm
        max_in_window = max(1, int(math.floor(rpm))) if rpm >= 1.0 else 1
        if len(st.window) >= max_in_window:
            oldest = st.window[0]
            expire = (oldest + self._window_delta) - now
            wait = max(wait, expire.total_seconds())
        if st.window:
            spacing = self.window_seconds / max(rpm, self.min_rpm)
            spacing = min(spacing, self.window_seconds)
            last = st.window[-1]
            elapsed = (now - last).total_seconds()
            if elapsed < spacing:
                wait = max(wait, spacing - elapsed)
        return max(0.0, wait)

    def _maybe_rechallenge(self, st: _KeyState, now: datetime) -> bool:
        if now - st.last_rechallenge < self.rechallenge_interval:
            return False
        st.last_rechallenge = now
        st.ceiling = self.ceiling_rpm
        st.consecutive_429 = 0
        st.cooldown_until = None
        probe = st.rpm + max(self.rechallenge_step_abs, st.rpm * self.rechallenge_step_rel)
        st.rpm = min(self.ceiling_rpm, probe)
        if self._on_rechallenge is not None:
            try:
                self._on_rechallenge(self.provider_id, st)
            except Exception:
                logger.exception("rate-limit rechallenge callback failed")
        logger.info(
            "[ratelimit:%s] daily re-challenge: probing higher limit (rpm=%.2f, ceiling=%.2f)",
            self.provider_id,
            st.rpm,
            st.ceiling,
        )
        self._save()
        return True

    async def acquire(self, model: str = "") -> dict[str, Any]:
        """Wait (if needed) then record an attempt for ``model``.

        Returns a small info dict with the waited duration and the current
        estimated rate. The recorded attempt is counted as an in-window request
        so a subsequent 429 can be attributed to the burst that caused it.
        """
        now = self._now()
        st = self._state_for(model)
        rechallenged = self._maybe_rechallenge(st, now)
        with self._lock:
            self._prune(st, now)
            wait = self._compute_wait(st, now)
        if wait > 0:
            await self._async_sleep(wait)
        with self._lock:
            self._prune(st, self._now())
            st.window.append(self._now())
        return {"waited": wait, "rpm": st.rpm, "rechallenged": rechallenged}

    def acquire_blocking(self, model: str = "") -> dict[str, Any]:
        """Synchronous twin of :meth:`acquire` (for non-async providers)."""
        now = self._now()
        st = self._state_for(model)
        rechallenged = self._maybe_rechallenge(st, now)
        with self._lock:
            self._prune(st, now)
            wait = self._compute_wait(st, now)
        if wait > 0:
            self._sync_sleep(wait)
        with self._lock:
            self._prune(self._now())
            st.window.append(self._now())
        return {"waited": wait, "rpm": st.rpm, "rechallenged": rechallenged}

    def on_success(self, model: str = "", now: Optional[datetime] = None) -> None:
        """Record a successful (non-429) response and nudge the estimate up."""
        now = now or self._now()
        with self._lock:
            st = self._state_for(model)
            st.consecutive_429 = 0
            st.last_success = now
            if st.cooldown_until is not None and st.cooldown_until <= now:
                st.cooldown_until = None
            if now - (st.last_429 or datetime.min) > self.stable_threshold:
                st.rpm = min(st.ceiling, st.rpm + self.additive)
            self._save_throttled()

    def on_429(self, model: str = "", retry_after_s: Optional[float] = None, now: Optional[datetime] = None) -> float:
        """React to an HTTP 429: lower the estimate and return a backoff (seconds)."""
        now = now or self._now()
        with self._lock:
            st = self._state_for(model)
            self._prune(st, now)
            burst = len(st.window)
            if burst >= 2:
                observed_safe = max(0.0, burst - 1)
                candidate = observed_safe * self.multiplicative
            else:
                candidate = st.rpm * self.multiplicative
            st.rpm = max(self.min_rpm, min(st.rpm, candidate))
            st.consecutive_429 += 1
            st.last_429 = now
            backoff = min(
                self.cooldown_max,
                self.cooldown_base * (2 ** (st.consecutive_429 - 1)),
            )
            if retry_after_s is not None:
                try:
                    backoff = max(backoff, float(retry_after_s))
                except (TypeError, ValueError):
                    pass
            st.cooldown_until = now + backoff
            self._save()
            return backoff.total_seconds()

    def reset(self, model: Optional[str] = None) -> None:
        """Forget learned state (tests / manual recovery)."""
        with self._lock:
            if model is None:
                self._states.clear()
            else:
                self._states.pop(model, None)
            self._save()

    def status(self) -> dict[str, Any]:
        """Snapshot of current per-model estimates (observability / debugging)."""
        with self._lock:
            return {
                model: {
                    "rpm": round(st.rpm, 3),
                    "ceiling": round(st.ceiling, 3),
                    "consecutive_429": st.consecutive_429,
                    "in_window": len(st.window),
                    "last_429": st.last_429.isoformat() if st.last_429 else None,
                    "last_rechallenge": st.last_rechallenge.isoformat() if st.last_rechallenge != datetime.min else None,
                    "cooldown_until": st.cooldown_until.isoformat() if st.cooldown_until else None,
                }
                for model, st in self._states.items()
            }

    def _save_throttled(self) -> None:
        self._save_counter += 1
        if self._save_counter % 25 == 0:
            self._save()

    def _load(self) -> None:
        self._persisted: dict[str, dict[str, Any]] = {}
        if not self._persist_path:
            return
        try:
            with open(self._persist_path) as f:
                data = json.load(f)
            providers = data.get("providers", {})
            self._persisted = providers.get(self.provider_id, {})
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self._persisted = {}

    def _save(self) -> None:
        if not self._persist_path:
            return
        try:
            try:
                with open(self._persist_path) as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                data = {}
            providers = data.setdefault("providers", {})
            snapshot: dict[str, dict[str, Any]] = {}
            with self._lock:
                for model, st in self._states.items():
                    snapshot[model] = st.to_dict()
            providers[self.provider_id] = snapshot
            tmp = self._persist_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._persist_path)
        except OSError:
            pass


_LIMITERS: dict[str, AdaptiveRateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def get_rate_limiter(provider_id: str, **kwargs: Any) -> AdaptiveRateLimiter:
    """Return the shared :class:`AdaptiveRateLimiter` for a provider (created once)."""
    limiter = _LIMITERS.get(provider_id)
    if limiter is None:
        with _LIMITERS_LOCK:
            limiter = _LIMITERS.get(provider_id)
            if limiter is None:
                limiter = AdaptiveRateLimiter(provider_id, **kwargs)
                _LIMITERS[provider_id] = limiter
    return limiter


def all_rate_limiter_status() -> dict[str, dict[str, Any]]:
    """Snapshot of every provider limiter's per-model state.

    For observability (e.g. the ``/v1/system/rate-limits`` endpoint): maps each
    provider id to its per-model learned rate, ceiling, cooldown, 429 streak and
    last re-challenge time.
    """
    with _LIMITERS_LOCK:
        items = list(_LIMITERS.items())
    return {pid: limiter.status() for pid, limiter in items}
