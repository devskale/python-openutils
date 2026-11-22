"""Exploration for auto@free — smart, cache-friendly rotation.

Hermetic (catalog/overrides/stats/reliability faked, same seams as
test_auto_router.py + test_routing_prefs.py). Covers: least-served ordering,
the sticky hold (cache warmth — consecutive resolves reuse the target), hold
release on expiry + counter movement (rotation through the pool), random
mode, pool bounding, pin/prefer outranking exploration, the read-only
no-side-effect view (preview/candidates never set a hold), and the
stream-only-400 dead-upstream markers that make exploring safe.
"""

import json

import pytest

from uniinfer.proxy_services import router as routing
from uniinfer.proxy_services import stats as stats_mod
from uniinfer.proxy_services.stats import StatsCollector


def _fake_catalog_models():
    return [
        {"id": "provA@alpha", "provider": "provA", "type": "chat",
         "access": "free", "status": "active"},
        {"id": "provA@beta", "provider": "provA", "type": "chat",
         "access": "free", "status": "active"},
        {"id": "provB@gamma", "provider": "provB", "type": "chat",
         "access": "free", "status": "active"},
        {"id": "provB@delta", "provider": "provB", "type": "chat",
         "access": "free", "status": "active"},
    ]


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch, tmp_path):
    routing._ExploreHold.clear()
    routing._QuotaCooldown._parked = {}
    monkeypatch.setattr(
        routing, "Catalog",
        lambda: type("C", (), {"list_resolved": staticmethod(_fake_catalog_models)})(),
    )
    monkeypatch.setattr(routing, "_degraded_models", lambda: set())
    monkeypatch.setattr(routing, "_provider_utilization_pct", lambda p: None)
    monkeypatch.delenv("UNIINFER_AUTO_EXPLORE", raising=False)
    monkeypatch.delenv("UNIINFER_AUTO_EXPLORE_POOL", raising=False)
    monkeypatch.delenv("UNIINFER_AUTO_EXPLORE_HOLD_S", raising=False)
    prefs = tmp_path / "routing_preferences.json"
    monkeypatch.setattr(routing, "_routing_prefs_path", lambda: str(prefs))
    ov = tmp_path / "model_overrides.json"
    ov.write_text(json.dumps({"models": {}}))
    monkeypatch.setattr(routing, "_router_overrides_path", lambda: str(ov))
    monkeypatch.setattr(stats_mod, "_SNAPSHOT_PATH", str(tmp_path / "stats.json"))
    StatsCollector._instance = None
    yield
    StatsCollector._instance = None
    routing._ExploreHold.clear()
    routing._QuotaCooldown._parked = {}


def _observe(model: str, n: int):
    for _ in range(n):
        StatsCollector().record(model, status=200, latency_ms=50.0, usage=None)


def _resolve():
    return routing.resolve_auto("auto@free")


# ── off (default): behavior unchanged ──────────────────────────────────

def test_explore_off_keeps_deterministic_order():
    _observe("provA@alpha", 5)
    mid, meta = _resolve()
    assert mid == "provA@alpha"  # Katalogordnung, kein Umsortieren
    assert meta["reason"] == "weight"
    assert "explore" not in meta


# ── least-served: coverage with random tiebreak ────────────────────────

def test_least_served_picks_unexplored_first(monkeypatch):
    monkeypatch.setenv("UNIINFER_AUTO_EXPLORE", "1")
    monkeypatch.setenv("UNIINFER_AUTO_EXPLORE_HOLD_S", "0")
    _observe("provA@alpha", 10)
    _observe("provA@beta", 5)
    mid, meta = _resolve()
    assert mid in ("provB@gamma", "provB@delta")  # beide n=0
    assert meta["reason"] == "explore"
    assert meta["explore"]["mode"] == "least-served"


def test_hold_makes_consecutive_resolves_sticky(monkeypatch):
    monkeypatch.setenv("UNIINFER_AUTO_EXPLORE", "on")
    monkeypatch.setenv("UNIINFER_AUTO_EXPLORE_HOLD_S", "600")
    first, meta = _resolve()
    for _ in range(5):
        again, m2 = _resolve()
        assert again == first  # Cache bleibt warm: Ziel wird gehalten
        assert m2["explore"]["held_s"] > 0


def test_rotation_after_hold_expiry_and_traffic(monkeypatch):
    monkeypatch.setenv("UNIINFER_AUTO_EXPLORE", "1")
    monkeypatch.setenv("UNIINFER_AUTO_EXPLORE_HOLD_S", "0.05")
    first, _ = _resolve()
    _observe(first, 10)  # das Ziel hat jetzt Samples
    import time as _t
    _t.sleep(0.1)  # Hold abgelaufen
    second, _ = _resolve()
    assert second != first  # naechstes least-served Ziel
    assert second in ("provA@alpha", "provA@beta", "provB@gamma", "provB@delta")


def test_held_target_invalidated_by_park(monkeypatch):
    monkeypatch.setenv("UNIINFER_AUTO_EXPLORE", "1")
    monkeypatch.setenv("UNIINFER_AUTO_EXPLORE_HOLD_S", "600")
    first, _ = _resolve()
    routing._QuotaCooldown.park(first, 900)
    second, _ = _resolve()
    assert second != first  # geparktes Ziel wird sofort ersetzt


# ── random mode + pool bounding ────────────────────────────────────────

def test_random_mode_rotates_and_stays_in_catalog(monkeypatch):
    monkeypatch.setenv("UNIINFER_AUTO_EXPLORE", "random")
    monkeypatch.setenv("UNIINFER_AUTO_EXPLORE_HOLD_S", "0")
    picks = {_resolve()[0] for _ in range(30)}
    assert len(picks) > 1  # Zufall arbeitet
    assert picks <= {m["id"] for m in _fake_catalog_models()}


def test_pool_bounds_exploration_zone(monkeypatch):
    # Pool=2: nur die 2 least-served stehen zur Debatte — die 10x
    # beobachteten sind aus dem Spiel, egal wie oft gezogen wird.
    monkeypatch.setenv("UNIINFER_AUTO_EXPLORE", "random")
    monkeypatch.setenv("UNIINFER_AUTO_EXPLORE_POOL", "2")
    monkeypatch.setenv("UNIINFER_AUTO_EXPLORE_HOLD_S", "0")
    _observe("provA@alpha", 10)
    _observe("provA@beta", 9)
    picks = {_resolve()[0] for _ in range(30)}
    assert picks <= {"provB@gamma", "provB@delta"}


# ── conscious decisions outrank exploration ────────────────────────────

def test_pin_beats_exploration(monkeypatch, tmp_path):
    monkeypatch.setenv("UNIINFER_AUTO_EXPLORE", "1")
    ov = tmp_path / "model_overrides.json"
    ov.write_text(json.dumps({"models": {"delta": {"pin": True}}}))
    monkeypatch.setattr(routing, "_router_overrides_path", lambda: str(ov))
    routing._RoutingPrefs.invalidate()
    mid, meta = _resolve()
    assert mid == "provB@delta"
    assert meta["reason"] == "pin"


def test_personal_prefer_beats_exploration(monkeypatch, tmp_path):
    monkeypatch.setenv("UNIINFER_AUTO_EXPLORE", "1")
    prefs = tmp_path / "routing_preferences.json"
    prefs.write_text(json.dumps({"default": {"prefer": {"models": ["gamma"]}}}))
    routing._PersonalPrefs.invalidate()
    mid, meta = _resolve()
    assert mid == "provB@gamma"
    assert meta["reason"] == "prefer"


# ── read-only callers never set a hold ─────────────────────────────────

def test_preview_and_candidates_do_not_set_hold(monkeypatch):
    monkeypatch.setenv("UNIINFER_AUTO_EXPLORE", "random")
    monkeypatch.setenv("UNIINFER_AUTO_EXPLORE_HOLD_S", "600")
    from uniinfer.proxy_services.router import (
        resolve_auto_candidates,
        routing_preview,
    )

    for _ in range(5):
        resolve_auto_candidates("auto@free")
        routing_preview("auto@free")
    assert routing._ExploreHold.get("free") is None  # kein Side-Effect
    # und der erste echte Resolve setzt ihn:
    _resolve()
    assert routing._ExploreHold.get("free") is not None


def test_preview_surfaces_explore_state(monkeypatch):
    monkeypatch.setenv("UNIINFER_AUTO_EXPLORE", "1")
    monkeypatch.setenv("UNIINFER_AUTO_EXPLORE_HOLD_S", "600")
    _resolve()  # Hold setzen
    preview = routing.routing_preview("auto@free")
    assert preview["reason"] == "explore"
    assert preview["explore"]["held_s"] > 0


# ── stream-only 400s are dead-upstream (exploration safety) ────────────

def test_stream_only_400_classifies_dead_upstream():
    from uniinfer.errors import ProviderError
    from uniinfer.proxy_services.router import is_dead_upstream

    e = ProviderError(
        '400 Bad Request: stream:true is required for this model. '
        'stream:false is not supported.',
        status_code=400,
        response_body='{"error":{"message":"stream:true is required for this model."}}',
    )
    assert is_dead_upstream("provA@alpha", e) is True
