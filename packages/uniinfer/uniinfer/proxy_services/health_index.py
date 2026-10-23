"""Per-model usability index (0..1) — the "how usable is this model" number.

Grounded in how the SOTA gateways do it (research 2026-10-24):

* **Apdex** (apdex.org): one index from three zones — satisfied / tolerating
  (half credit) / frustrated. Applied here to TTFT: satisfied < 5 s,
  tolerating < 20 s (= 4T), frustrated above (Aqueduct crawl-mode forensics).
* **Laplace smoothing** (the low-traffic fix): (successes + 1) / (n + 2) — a
  model with 1/1 successes scores 0.67, not 1.0; a model with no traffic
  scores the neutral 0.5 prior, never a fake 0 or 1.
* **429 ≠ broken** (LiteLLM/OpenRouter consensus): rate limits count HALF a
  failure; hard errors (5xx, empty-stream 502) count full.
* **Confidence tier** (OpenRouter's "insufficient data" + SRE multi-window):
  hot (>= 20 req in 1h) trust 1.0, warm (>= 20 in 7d) 0.9, cold 0.5 with an
  ``insufficient_data`` flag — one sortable value, the reason stays visible.
* **ReliabilityMonitor** (the existing incident layer): a degraded model's
  score is floored at half — incident state must outvote smooth counters.

Read-only telemetry (v0): dashboards, /v1/health/models, the /v1/models
health block, the routing preview. Router ordering by score is the separate
v1 decision (docs/routing.md) — this module never mutates routing.

All inputs come from StatsCollector's persisted, restart-surviving counters.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("uniioai_proxy")

_LAPLACE_ALPHA = 1.0
_TRUST_HOT_N = 20
_TRUST_HOT_FACTOR = 1.0
_TRUST_WARM_FACTOR = 0.9
_TRUST_COLD_FACTOR = 0.5
_DEGRADED_FACTOR = 0.5

_VERDICTS = ((0.95, "A"), (0.85, "B"), (0.70, "C"), (0.0, "D"))


def _verdict(score: float) -> str:
    for threshold, label in _VERDICTS:
        if score >= threshold:
            return label
    return "D"


def _availability(c: dict) -> float:
    """Laplace-smoothed success rate; 429 costs half a failure."""
    n = c["req"]
    hard = c["errors"] - c["soft_errors"]
    soft = c["soft_errors"]
    successes = n - hard - 0.5 * soft
    return (successes + _LAPLACE_ALPHA) / (n + 2.0 * _LAPLACE_ALPHA)


def _latency(c: dict) -> float:
    """Apdex over streamed requests (TTFT zones, Laplace-smoothed).

    Non-streamed traffic has no TTFT — the factor is neutral 1.0 then (a
    chat-only-nonstream model is not punished for missing stream data).
    """
    streams = c["ttft_n"]
    if not streams:
        return 1.0
    zones = c["ttft_ok"] + 0.5 * c["ttft_slow"]
    return (zones + _LAPLACE_ALPHA) / (streams + 2.0 * _LAPLACE_ALPHA)


def _degraded_models() -> set[str]:
    try:
        from uniinfer.proxy_services.reliability import get_reliability

        return set(get_reliability().snapshot().get("degraded_models") or [])
    except Exception as e:  # noqa: BLE001 — the index must never raise
        logger.debug("reliability snapshot failed in health index: %s", e)
        return set()


def _stats():
    from uniinfer.proxy_services.stats import get_stats

    return get_stats()


def _compute(mid: str, windows: dict[str, dict], degraded: set[str]) -> dict:
    """Score one model from its 1h/24h/7d counter windows."""
    n = {w: c["req"] for w, c in windows.items() if c}
    base_window = max(n, key=lambda w: n[w]) if n else None
    base = windows.get(base_window) if base_window else None
    availability = _availability(base) if base else 0.5
    latency = _latency(base) if base else 1.0

    n_1h = windows.get("1h", {}).get("req", 0)
    n_7d = windows.get("7d", {}).get("req", 0)
    if n_1h >= _TRUST_HOT_N:
        trust, tier = _TRUST_HOT_FACTOR, "hot"
    elif n_7d >= _TRUST_HOT_N:
        trust, tier = _TRUST_WARM_FACTOR, "warm"
    else:
        trust, tier = _TRUST_COLD_FACTOR, "cold"

    score = availability * latency * trust
    insufficient = tier == "cold"
    is_degraded = mid in degraded
    if is_degraded:
        score = min(score, _DEGRADED_FACTOR)

    return {
        "model": mid,
        "score": round(score, 3),
        "verdict": _verdict(score),
        "availability": round(availability, 3),
        "latency": round(latency, 3),
        "trust": trust,
        "trust_tier": tier,
        "insufficient_data": insufficient,
        "degraded": is_degraded,
        "n_observed": n.get(base_window, 0) if base_window else 0,
        "window": base_window,
        "n_1h": n_1h,
        "n_7d": n_7d,
    }


def health_snapshot() -> list[dict]:
    """All models seen in stats, scored and ranked (score desc, then n)."""
    stats = _stats()
    windows = {
        "1h": stats.merged_per_model(1),
        "24h": stats.merged_per_model(24),
        "7d": stats.merged_per_model(24 * 7),
    }
    degraded = _degraded_models()
    models = set(windows["7d"]) | set(windows["1h"])
    rows = [
        _compute(
            mid,
            {w: windows[w].get(mid, {}) for w in ("1h", "24h", "7d")},
            degraded,
        )
        for mid in models
    ]
    rows.sort(key=lambda r: (-r["score"], -r["n_observed"], r["model"]))
    return rows


def health_map() -> dict[str, dict]:
    """Compact {model_id: {score, verdict}} for embedding into /v1/models."""
    return {r["model"]: {"score": r["score"], "verdict": r["verdict"]} for r in health_snapshot()}
