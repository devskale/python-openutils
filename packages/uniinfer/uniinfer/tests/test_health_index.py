"""Health index — the per-model usability formula, component by component.

Hermetic: StatsCollector isolated via tmp snapshot + singleton reset (same
pattern as test_stats_ttft.py). Covers: Laplace smoothing (1/1 is not 100%),
429-half-weighting, Apdex TTFT zones, trust tiers, the degraded floor,
verdict buckets, zero-traffic neutrality, and the persisted-counter restore
(schema drift: old snapshots without the new keys load as zeros).
"""

import pytest

from uniinfer.proxy_services import stats as stats_mod
from uniinfer.proxy_services import health_index as hi
from uniinfer.proxy_services.stats import StatsCollector


@pytest.fixture
def collector(monkeypatch, tmp_path):
    monkeypatch.setattr(stats_mod, "_SNAPSHOT_PATH", str(tmp_path / "stats.json"))
    StatsCollector._instance = None
    yield StatsCollector()
    StatsCollector._instance = None


@pytest.fixture(autouse=True)
def _no_reliability(monkeypatch):
    monkeypatch.setattr(hi, "_degraded_models", lambda: set())


def _seed(collector, model, outcomes, ttfts=None):
    """outcomes: list of status codes; ttfts: parallel list (None = no stream)."""
    ttfts = ttfts or [None] * len(outcomes)
    for status, ttft in zip(outcomes, ttfts):
        collector.record(
            model, status=status, latency_ms=100.0, usage=None,
            ttft_ms=(float(ttft) if ttft is not None else None),
        )


def _row(collector, model):
    snap = {r["model"]: r for r in hi.health_snapshot()}
    return snap[model]


# ── availability: Laplace + 429 half-weight ────────────────────────────

def test_perfect_record_is_not_100_percent(collector):
    _seed(collector, "p@m", [200])
    r = _row(collector, "p@m")
    # (1 + 1) / (1 + 2) — ein Einzelsieg ist kein Beleg
    assert r["availability"] == pytest.approx(2 / 3, abs=0.01)
    assert r["insufficient_data"] is True


def test_many_successes_converge_to_one(collector):
    _seed(collector, "p@m", [200] * 50)
    # (50+1)/(50+2) — Laplace konvergiert mit 1/(n+2), nicht instantly
    # (Feld ist auf 3 Dezimalen gerundet)
    assert _row(collector, "p@m")["availability"] == pytest.approx(51 / 52, abs=0.001)


def test_429_costs_half_a_failure(collector):
    _seed(collector, "soft@m", [200, 429])
    _seed(collector, "hard@m", [200, 500])
    soft = _row(collector, "soft@m")["availability"]
    hard = _row(collector, "hard@m")["availability"]
    assert soft > hard
    # soft: (1.5+1)/4 = 0.625, hard: (1+1)/4 = 0.5
    assert soft == pytest.approx(0.625, abs=0.01)
    assert hard == pytest.approx(0.5, abs=0.01)


def test_zero_traffic_is_neutral_not_zero(collector):
    _seed(collector, "other@m", [200])
    snap = hi.health_snapshot()
    assert "never-called@m" not in {r["model"] for r in snap}


# ── latency: Apdex zones over TTFT ─────────────────────────────────────

def test_ttft_zones_weighted_apdex_style(collector):
    # 1 satisfied (<5s), 1 tolerating (5-20s), 1 frustrated (>20s)
    _seed(collector, "p@m", [200, 200, 200], ttfts=[1000, 10000, 30000])
    r = _row(collector, "p@m")
    # (1 + 0.5 + 1) / (3 + 2)
    assert r["latency"] == pytest.approx(2.5 / 5, abs=0.01)


def test_nonstream_only_is_latency_neutral(collector):
    _seed(collector, "p@m", [200] * 10)  # keine TTFTs
    assert _row(collector, "p@m")["latency"] == 1.0


# ── trust tiers + verdicts ─────────────────────────────────────────────

def test_trust_tiers_hot_warm_cold(collector):
    _seed(collector, "hot@m", [200] * 25)
    _seed(collector, "cold@m", [200] * 3)
    hot = _row(collector, "hot@m")
    cold = _row(collector, "cold@m")
    assert hot["trust_tier"] == "hot" and hot["trust"] == 1.0
    assert cold["trust_tier"] == "cold" and cold["trust"] == 0.5
    assert cold["insufficient_data"] is True and hot["insufficient_data"] is False


def test_confidence_orders_unknown_below_known_good_and_known_bad(collector):
    # Der Sinn der Confidence: 3 saubere Requests sind UNBEKANNT (rankt unter
    # beiden bekannten), 25 saubere sind bewiesen gut, 25 mit 7 Hard-Fehlern
    # ist bewiesen mittelmässig — known-good > known-bad > unknown.
    _seed(collector, "unknown@m", [200] * 3)
    _seed(collector, "known-good@m", [200] * 25)
    _seed(collector, "known-bad@m", [200] * 18 + [500] * 7)
    u = _row(collector, "unknown@m")
    g = _row(collector, "known-good@m")
    b = _row(collector, "known-bad@m")
    assert g["score"] > b["score"] > u["score"]
    assert u["insufficient_data"] is True and g["insufficient_data"] is False


def test_verdict_buckets(collector):
    assert hi._verdict(0.97) == "A"
    assert hi._verdict(0.95) == "A"
    assert hi._verdict(0.9) == "B"
    assert hi._verdict(0.8) == "C"
    assert hi._verdict(0.5) == "D"


# ── degraded floor + window choice ─────────────────────────────────────

def test_degraded_model_is_floored(collector, monkeypatch):
    _seed(collector, "p@m", [200] * 25)
    monkeypatch.setattr(hi, "_degraded_models", lambda: {"p@m"})
    r = _row(collector, "p@m")
    assert r["degraded"] is True
    assert r["score"] <= hi._DEGRADED_FACTOR


def test_base_window_is_the_richest(collector):
    _seed(collector, "p@m", [200] * 25)
    r = _row(collector, "p@m")
    assert r["window"] == "1h"  # alles gerade passiert
    assert r["n_observed"] == 25


def test_old_snapshot_without_new_keys_restores_as_zeros(monkeypatch, tmp_path):
    """Schema drift: ein persistiertes stats.json ohne soft_errors/ttft_ok
    laedt mit Nullen — der Index rechnet weiter, statt zu crashen."""
    import json

    snap = tmp_path / "stats.json"
    old_counter = {"req": 5, "errors": 1, "prompt": 0, "completion": 0, "total": 0,
                   "latency_sum": 100, "latency_n": 5, "ttft_sum": 0, "ttft_n": 0}
    hour = 0  # längst gepruned — nur der Restore-Pfad interessiert
    snap.write_text(json.dumps({
        "since": 0,
        "hourly": {str(hour): {"p@m": old_counter}},
        "daily": {}, "status_hourly": {}, "status_daily": {},
        "status_total": {}, "totals": {"p@m": old_counter},
    }))
    monkeypatch.setattr(stats_mod, "_SNAPSHOT_PATH", str(snap))
    StatsCollector._instance = None
    try:
        c = StatsCollector()
        # Stunde 0 ist ausserhalb des Fensters — aber der Collector muss
        # geladen sein ohne KeyError; ein frischer Record funktioniert:
        c.record("p@m", status=200, latency_ms=10.0, usage=None, ttft_ms=1000.0)
        row = {r["model"]: r for r in hi.health_snapshot()}["p@m"]
        assert row["availability"] > 0
    finally:
        StatsCollector._instance = None
