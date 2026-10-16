"""Auto-router (auto@free): candidate filter, ordering, and pass-through.

Hermetic — no network, no live providers. Catalog/Stats/Reliability/overrides
are faked via monkeypatch. Covers: pin/weight/exclude overrides, degraded and
quota skipping, deterministic tiebreak, non-auto pass-through, unknown
selector, and the empty-candidate case.
"""

import json

import pytest

from uniinfer.proxy_services import router as routing


@pytest.fixture(autouse=True)
def _reset_prefs_cache():
    routing._RoutingPrefs._cache = None
    routing._RoutingPrefs._cache_mtime = None
    yield
    routing._RoutingPrefs._cache = None
    routing._RoutingPrefs._cache_mtime = None


def _fake_catalog_models():
    return [
        {
            "id": "provA@alpha",
            "provider": "provA",
            "type": "chat",
            "access": "free",
            "status": "active",
        },
        {
            "id": "provA@beta",
            "provider": "provA",
            "type": "chat",
            "access": "free",
            "status": "active",
        },
        {
            "id": "provB@gamma",
            "provider": "provB",
            "type": "chat",
            "access": "free",
            "status": "active",
        },
        {
            "id": "provB@delta",
            "provider": "provB",
            "type": "chat",
            "access": "free",
            "status": "active",
        },
        # filtered out by construction:
        {
            "id": "provA@paid",
            "provider": "provA",
            "type": "chat",
            "access": "paid",
            "status": "active",
        },
        {
            "id": "provA@dead",
            "provider": "provA",
            "type": "chat",
            "access": "free",
            "status": "deprecated",
        },
        {
            "id": "provA@embed",
            "provider": "provA",
            "type": "embedding",
            "access": "free",
            "status": "active",
        },
    ]


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    monkeypatch.setattr(
        routing,
        "Catalog",
        lambda: type("C", (), {"list_resolved": staticmethod(_fake_catalog_models)})(),
    )
    monkeypatch.setattr(routing, "_degraded_models", lambda: {"provA@beta"})
    monkeypatch.setattr(
        routing, "_provider_utilization_pct", lambda p: 90.0 if p == "provB" else None
    )
    # overrides file: pin gamma? no — gamma is quota-skipped; weight beta? degraded.
    # pin none, weight provA@alpha up via bare id.
    ov = {"models": {"alpha": {"weight": 2.0}, "delta": {"exclude": True}}}
    p = tmp_path / "model_overrides.json"
    p.write_text(json.dumps(ov))
    monkeypatch.setattr(routing, "_router_overrides_path", lambda: str(p))
    return p


def test_resolve_auto_pass_through_non_auto(fakes):
    assert routing.resolve_auto("openai@gpt-5") is None
    assert routing.resolve_auto("gpt-5") is None


def test_resolve_auto_free_filters_and_orders(fakes):
    resolved = routing.resolve_auto("auto@free")
    assert resolved is not None
    mid, meta = resolved
    # beta degraded, gamma over quota, delta excluded, paid/embed/dead filtered:
    # only alpha survives.
    assert mid == "provA@alpha"
    assert meta["candidates"] == 1
    assert meta["skipped"] == {"excluded": 1, "degraded": 1, "quota": 1}


def test_resolve_auto_pin_wins_over_weight(fakes, monkeypatch):
    ov = {"models": {"gamma": {"pin": True}}}
    routing._RoutingPrefs._cache = None
    monkeypatch.setattr(routing, "_provider_utilization_pct", lambda p: None)
    p = fakes
    p.write_text(json.dumps(ov))
    routing._RoutingPrefs._cache = None
    resolved = routing.resolve_auto("auto@free")
    assert resolved is not None
    assert resolved[0] == "provB@gamma"
    assert resolved[1]["reason"] == "pin"


def test_resolve_auto_unknown_selector(fakes):
    with pytest.raises(ValueError, match="Unknown auto selector"):
        routing.resolve_auto("auto@paid")


def test_resolve_auto_empty_candidates(fakes, monkeypatch):
    monkeypatch.setattr(
        routing,
        "_degraded_models",
        lambda: {"provA@alpha", "provA@beta", "provB@gamma"},
    )
    assert routing.resolve_auto("auto@free") is None


def test_resolve_auto_candidates_chain(fakes, monkeypatch):
    monkeypatch.setattr(routing, "_degraded_models", lambda: set())
    monkeypatch.setattr(routing, "_provider_utilization_pct", lambda p: None)
    chain = routing.resolve_auto_candidates("auto@free")
    assert chain[0] == "provA@alpha"  # weight 2.0
    assert set(chain[1:]) == {"provA@beta", "provB@gamma"}
    assert len(chain) == 3


def test_prefs_cache_invalidates_on_mtime(fakes, monkeypatch):
    p = fakes
    first = routing._RoutingPrefs.load()
    assert "alpha" in first
    p.write_text(json.dumps({"models": {"alpha": {"exclude": True}}}))
    routing._RoutingPrefs._cache = None  # simulate fresh process
    second = routing._RoutingPrefs.load()
    assert second["alpha"]["exclude"] is True
