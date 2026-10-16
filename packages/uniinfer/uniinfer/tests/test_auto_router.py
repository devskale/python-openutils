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
    assert meta["skipped"] == {"excluded": 1, "degraded": 1, "quota": 1, "parked": 0}


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


# --- Quota-Cooldown: Modell parken, bis die Zeit um ist ----------------------


@pytest.fixture(autouse=True)
def _reset_cooldown():
    routing._QuotaCooldown._parked = {}
    yield
    routing._QuotaCooldown._parked = {}


def test_quota_hit_parks_model(fakes):
    routing.note_quota_hit("provA@alpha")
    assert "provA@alpha" in routing._QuotaCooldown.parked()
    # ...und wird aus der Kandidatenliste gefiltert
    resolved = routing.resolve_auto("auto@free")
    assert resolved is None  # alpha war der einzige Kandidat


def test_quota_hit_retry_after_extends_park(fakes):
    routing.note_quota_hit("provA@alpha", retry_after=3600.0)
    snap = routing._QuotaCooldown.snapshot()
    assert 3000 < snap["provA@alpha"] <= 3600


def test_quota_hit_retry_after_never_shortens(fakes):
    routing.note_quota_hit("provA@alpha", retry_after=5.0)
    snap = routing._QuotaCooldown.snapshot()
    assert snap["provA@alpha"] > 600  # Default-Cooldown bleibt


def test_park_expires(fakes, monkeypatch):
    import time as _time

    routing.note_quota_hit("provA@alpha")
    assert "provA@alpha" in routing._QuotaCooldown.parked()
    # Zeit 20 Minuten vorspulen
    real_monotonic = _time.monotonic
    monkeypatch.setattr(routing.time, "monotonic", lambda: real_monotonic() + 1200)
    assert "provA@alpha" not in routing._QuotaCooldown.parked()
    resolved = routing.resolve_auto("auto@free")
    assert resolved is not None and resolved[0] == "provA@alpha"


# --- Deprecation: manuell + auto ---------------------------------------------


@pytest.fixture
def catalog_tmp(monkeypatch, tmp_path):
    """Catalog mit echtem overrides-file (deprecate schreibt ja hinein)."""
    import json as _json

    ov_path = tmp_path / "model_overrides.json"
    ov_path.write_text(_json.dumps({"models": {}}))
    monkeypatch.setattr(routing, "_router_overrides_path", lambda: str(ov_path))
    routing._RoutingPrefs._cache = None

    class _Cat:
        def __init__(self):
            self._ov = ov_path

        def list_resolved(self):
            return _fake_catalog_models()

        def read_overrides(self):
            return _json.loads(self._ov.read_text())

        def deprecate_model(self, mid, reason, *, source="manual"):
            doc = _json.loads(self._ov.read_text())
            entry = {
                "status": "deprecated",
                "deprecation_date": "2026-10-16",
                "deprecation_reason": reason,
                "deprecation_source": source,
            }
            doc["models"].setdefault(mid.split("@", 1)[-1], {}).update(entry)
            self._ov.write_text(_json.dumps(doc))
            return entry

    monkeypatch.setattr(routing, "Catalog", _Cat)
    monkeypatch.setattr(routing, "_degraded_models", lambda: {"provA@beta"})
    monkeypatch.setattr(
        routing, "_provider_utilization_pct", lambda p: 90.0 if p == "provB" else None
    )
    return _Cat


def test_manual_deprecated_model_is_filtered(catalog_tmp, monkeypatch):
    catalog_tmp().deprecate_model("provA@alpha", "upstream gone", source="manual")
    resolved = routing.resolve_auto("auto@free")
    # alpha deprecated, beta degraded, gamma quota, delta excluded -> leer
    assert resolved is None


def test_dead_upstream_detection():
    from uniinfer.errors import InvalidRequestError
    from uniinfer.errors import ProviderError as PE

    e = InvalidRequestError(
        'OpenCode API error: 400 - {"message":"Upstream request failed: Model is unavailable."}',
        status_code=400,
        response_body='{"message":"Upstream request failed: Model is unavailable."}',
    )
    assert routing.is_dead_upstream("provA@alpha", e)
    e404 = PE("Kilo 404 not found", status_code=404, response_body="")
    assert routing.is_dead_upstream("provA@alpha", e404)
    # quota/flaky NICHT als dead klassifizieren
    from uniinfer.errors import RateLimitError

    e429 = RateLimitError("rate limit exceeded", status_code=429, response_body="")
    assert not routing.is_dead_upstream("provA@alpha", e429)
    e500 = PE("boom 500", status_code=500, response_body="internal error")
    assert not routing.is_dead_upstream("provA@alpha", e500)


def test_auto_deprecation_after_repeated_dead_parks(catalog_tmp, monkeypatch):
    from uniinfer.errors import InvalidRequestError

    e = InvalidRequestError(
        "Model is unavailable", 400, '{"message":"Model is unavailable"}'
    )
    for _ in range(routing._DEAD_PARKS_TO_DEPRECATE):
        routing.note_dead_upstream("provA@alpha", e)
    # ...und das deprecated-override ist gesetzt und filtert das Modell
    ov = catalog_tmp().read_overrides()["models"]
    assert ov["alpha"]["status"] == "deprecated"
    assert ov["alpha"]["deprecation_source"] == "auto"
    resolved = routing.resolve_auto("auto@free")
    assert resolved is None
