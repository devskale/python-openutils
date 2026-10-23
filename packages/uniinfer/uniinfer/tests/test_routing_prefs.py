"""Personal routing preferences (routing_preferences.json): filter + boost.

Hermetic — no network, no live providers. Catalog/overrides/reliability faked
like test_auto_router.py; the prefs file + token-metadata sidecar are tmp
files. Covers: avoid as a hard filter (own skip reason), prefer as a tier
between operator pin and weight, model-level avoid beating provider-level
prefer, token-name resolution + union merge over defaults, and bare/full
model-id matching.
"""

import hashlib
import json

import pytest

from uniinfer.proxy_services import router as routing


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
    """Hermetic prefs: tmp file path + clean caches + no env leakage."""
    prefs_file = tmp_path / "routing_preferences.json"
    monkeypatch.setattr(routing, "_routing_prefs_path", lambda: str(prefs_file))
    monkeypatch.delenv("UNIINFER_AUTH_TOKENS_FILE", raising=False)
    _clear_caches()
    yield prefs_file
    _clear_caches()


def _clear_caches():
    routing._RoutingPrefs._cache = None
    routing._RoutingPrefs._cache_mtime = None
    routing._PersonalPrefs._cache = None
    routing._PersonalPrefs._cache_mtime = None
    routing._TokenNames._cache = None


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    monkeypatch.setattr(
        routing,
        "Catalog",
        lambda: type("C", (), {"list_resolved": staticmethod(_fake_catalog_models)})(),
    )
    monkeypatch.setattr(routing, "_degraded_models", lambda: set())
    monkeypatch.setattr(routing, "_provider_utilization_pct", lambda p: None)
    ov = tmp_path / "model_overrides.json"
    ov.write_text(json.dumps({"models": {}}))
    monkeypatch.setattr(routing, "_router_overrides_path", lambda: str(ov))
    return tmp_path / "routing_preferences.json"


def _write_prefs(path, doc):
    path.write_text(json.dumps(doc), encoding="utf-8")
    routing._PersonalPrefs.invalidate()


def _write_token_meta(monkeypatch, tmp_path, names):
    """Allowlist + metadata sidecar mapping sha256(name-keyed tokens) → name.

    ``names`` maps an arbitrary bearer string to its registry name.
    """
    allow = tmp_path / "auth_tokens.allow"
    meta = tmp_path / "auth_tokens.meta.json"
    tokens = {}
    hashes = []
    for bearer, name in names.items():
        h = hashlib.sha256(bearer.encode()).hexdigest()
        hashes.append(h)
        tokens[h] = {"name": name}
    allow.write_text("\n".join(hashes) + "\n")
    meta.write_text(json.dumps({"version": 1, "tokens": tokens}))
    monkeypatch.setenv("UNIINFER_AUTH_TOKENS_FILE", str(allow))
    routing._TokenNames.invalidate()


# ── avoid: hard personal filter with its own skip reason ───────────────

def test_avoid_provider_filters_and_reports(fakes):
    _write_prefs(fakes, {"default": {"avoid": {"providers": ["prova"]}}})
    mid, meta = routing.resolve_auto("auto@free")
    assert mid.startswith("provB@")
    assert meta["skipped"]["avoided"] == 2


def test_avoid_model_full_and_bare_id_match(fakes):
    _write_prefs(fakes, {"default": {"avoid": {"models": ["provA@alpha", "gamma"]}}})
    mid, meta = routing.resolve_auto("auto@free")
    assert mid in ("provA@beta", "provB@delta")
    assert meta["skipped"]["avoided"] == 2


def test_model_level_avoid_beats_provider_level_prefer(fakes):
    _write_prefs(
        fakes,
        {"default": {"prefer": {"providers": ["provA"]},
                     "avoid": {"models": ["alpha", "beta"]}}},
    )
    mid, _ = routing.resolve_auto("auto@free")
    assert mid.startswith("provB@")  # alle provA-Modelle personally vermieden


# ── prefer: tier between operator pin and weight ───────────────────────

def test_prefer_outranks_catalog_order(fakes):
    _write_prefs(fakes, {"default": {"prefer": {"models": ["provB@delta"]}}})
    mid, meta = routing.resolve_auto("auto@free")
    assert mid == "provB@delta"
    assert meta["reason"] == "prefer"
    assert meta["fallbacks"][0] == "provA@alpha"  # Rest in Katalogordnung


def test_operator_pin_still_beats_prefer(fakes, monkeypatch, tmp_path):
    ov = tmp_path / "model_overrides.json"
    ov.write_text(json.dumps({"models": {"alpha": {"pin": True}}}))
    monkeypatch.setattr(routing, "_router_overrides_path", lambda: str(ov))
    routing._RoutingPrefs.invalidate()
    _write_prefs(fakes, {"default": {"prefer": {"models": ["provB@delta"]}}})
    mid, meta = routing.resolve_auto("auto@free")
    assert mid == "provA@alpha"
    assert meta["reason"] == "pin"


def test_preferred_provider_boosts_all_its_models(fakes):
    _write_prefs(fakes, {"default": {"prefer": {"providers": ["provb"]}}})
    mid, _ = routing.resolve_auto("auto@free")
    assert mid.startswith("provB@")


# ── per-caller resolution via token name ───────────────────────────────

def test_token_section_applies_for_that_bearer_only(fakes, monkeypatch, tmp_path):
    _write_token_meta(monkeypatch, tmp_path, {"bearer-alice@xyz": "alice"})
    _write_prefs(
        fakes,
        {"tokens": {"alice": {"prefer": {"models": ["provB@delta"]}}}},
    )
    h = hashlib.sha256(b"bearer-alice@xyz").hexdigest()
    mid, meta = routing.resolve_auto("auto@free", token_hash=h)
    assert mid == "provB@delta"
    assert meta["reason"] == "prefer"
    # ohne Token (anonym): Default-Sektion → Katalogordnung
    anon_mid, _ = routing.resolve_auto("auto@free")
    assert anon_mid == "provA@alpha"


def test_token_section_unions_over_default(fakes, monkeypatch, tmp_path):
    _write_token_meta(monkeypatch, tmp_path, {"bearer-bob@xyz": "bob"})
    _write_prefs(
        fakes,
        {"default": {"avoid": {"providers": ["prova"]}},
         "tokens": {"bob": {"avoid": {"models": ["gamma"]}}}},
    )
    h = hashlib.sha256(b"bearer-bob@xyz").hexdigest()
    mid, meta = routing.resolve_auto("auto@free", token_hash=h)
    assert mid == "provB@delta"  # provA (Default-avoid) + gamma (Bobs avoid)
    assert meta["skipped"]["avoided"] == 3


def test_unknown_bearer_hash_falls_back_to_default(fakes, monkeypatch, tmp_path):
    _write_token_meta(monkeypatch, tmp_path, {"bearer-carol@xyz": "carol"})
    _write_prefs(fakes, {"default": {"prefer": {"models": ["provB@gamma"]}}})
    stranger = hashlib.sha256(b"stranger@token").hexdigest()
    mid, _ = routing.resolve_auto("auto@free", token_hash=stranger)
    assert mid == "provB@gamma"  # Default gilt; stranger hat keine Sektion


def test_everything_avoided_leaves_empty_pool(fakes):
    _write_prefs(fakes, {"default": {"avoid": {"providers": ["prova", "provb"]}}})
    assert routing.resolve_auto("auto@free") is None


def test_absent_prefs_file_behaves_like_no_preferences(fakes):
    mid, meta = routing.resolve_auto("auto@free")
    assert mid == "provA@alpha"
    assert meta["skipped"]["avoided"] == 0
    assert routing.effective_prefs(None) is None
