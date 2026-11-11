"""Per-key usage collector + /v1/usage endpoint (read-only quota attribution).

Hermetic: KeyUsageCollector isolated via tmp snapshot + singleton reset
(same pattern as test_stats_ttft.py), fake time via explicit now= params
(and a patched time.time for the snapshot-reload roundtrip). Router tests
patch the pool-label resolution and provider configs directly — no env
files, no live config, no credgoo calls.

Covers: UTC window floors (minute/hour/midnight), provider-scoped multiplier
weighting (2x-Limits models count 0.5, incl. aliases), upstream-key
attribution (pool keys vs caller-supplied X-Provider-Key keys counted
separately under their hash), per-key isolation, window rollover pruning,
snapshot v2 roundtrip (v1 dropped), key-cap eviction, and the endpoint
contract: 401 without/unknown bearer, providers/pool/mine/worst shape,
legacy windows preserved, config-driven provider list.
"""

import hashlib
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from uniinfer.proxy_services import key_usage as ku
from uniinfer.proxy_services.key_usage import (
    KeyUsageCollector,
    bucket_end,
    bucket_start,
    get_key_usage,
    upstream_key_hash,
    weight_of,
)

MULTS = {"qwen-3.8-27b": 0.5, "vision": 0.5, "ocr": 0.5}

# 2026-11-11 13:34:30 UTC — mid-window, ±30s clearance from every boundary.
NOW = int(datetime(2026, 11, 11, 13, 34, 30, tzinfo=timezone.utc).timestamp())

POOL_KEY = "sk-pool-a"
KLARK_KEY = "sk-klark-b"
OWN_KEY = "sk-caller-own-c"
POOL_H = upstream_key_hash(POOL_KEY)
KLARK_H = upstream_key_hash(KLARK_KEY)
OWN_H = upstream_key_hash(OWN_KEY)


@pytest.fixture
def collector(monkeypatch, tmp_path):
    monkeypatch.setattr(ku, "_SNAPSHOT_PATH", str(tmp_path / "key_usage.json"))
    KeyUsageCollector._instance = None
    yield get_key_usage()
    KeyUsageCollector._instance = None


# --- window geometry ---

def test_minute_and_hour_floors():
    assert bucket_start("minute", NOW) == int(datetime(2026, 11, 11, 13, 34, tzinfo=timezone.utc).timestamp())
    assert bucket_start("hour", NOW) == int(datetime(2026, 11, 11, 13, 0, tzinfo=timezone.utc).timestamp())


def test_day_floor_is_midnight_utc():
    start = bucket_start("day", NOW)
    assert datetime.fromtimestamp(start, timezone.utc).strftime("%H:%M:%S") == "00:00:00"


def test_reset_at_is_next_boundary():
    assert bucket_end("minute", bucket_start("minute", NOW)) - NOW == 30
    assert bucket_end("day", bucket_start("day", NOW)) - NOW == 10 * 3600 + 25 * 60 + 30


# --- weighting (provider-scoped config, no provider hardcode) ---

def test_weight_of_applies_multipliers_and_aliases():
    assert weight_of("tu@qwen-3.8-27b", MULTS) == 0.5
    assert weight_of("tu@vision", MULTS) == 0.5
    assert weight_of("tu@deepseek-v4-vision-exp-284b", MULTS) == 1.0
    assert weight_of("groq@llama-9", {}) == 1.0


# --- collector: 3-dim counting + attribution ---

def test_usage_view_scopes_to_provider_and_mixes_keys(collector):
    th = hashlib.sha256(b"tok1").hexdigest()
    collector.note("tok1", "tu@qwen-3.8-27b", POOL_KEY, now=NOW)
    collector.note("tok1", "tu@qwen-3.8-27b", POOL_KEY, now=NOW)
    collector.note("tok1", "tu@qwen-3.8-27b", OWN_KEY, now=NOW)
    collector.note("tok1", "groq@llama-9", OWN_KEY, now=NOW)

    tu = collector.usage_view(th, MULTS, provider="tu", now=NOW)
    assert tu["windows"]["minute"]["used"] == 1.5  # 2x0.5 pool + 0.5 own
    groq = collector.usage_view(th, {}, provider="groq", now=NOW)
    assert groq["windows"]["minute"]["used"] == 1.0
    allv = collector.usage_view(th, MULTS, now=NOW)
    assert allv["windows"]["minute"]["used"] == 2.5  # tu 1.5 + groq 1.0 (weight 1.0)


def test_pool_view_separates_upstream_keys(collector):
    collector.note("tok1", "tu@qwen-3.8-27b", POOL_KEY, now=NOW)
    collector.note("tok2", "tu@glm-5.3-744b-experimental", KLARK_KEY, now=NOW)
    collector.note("tok1", "tu@qwen-3.8-27b", OWN_KEY, now=NOW)

    pool = collector.provider_pool_view("tu", MULTS, now=NOW)
    assert pool[POOL_H]["minute"] == 0.5
    assert pool[KLARK_H]["minute"] == 1.0
    assert pool[OWN_H]["minute"] == 0.5  # in raw view; the router filters by label map


def test_keys_are_isolated(collector):
    th1 = hashlib.sha256(b"tok1").hexdigest()
    collector.note("tok1", "tu@qwen-3.8-27b", POOL_KEY, now=NOW)
    collector.note("tok2", "tu@qwen-3.8-27b", POOL_KEY, now=NOW)
    collector.note("tok2", "tu@qwen-3.8-27b", POOL_KEY, now=NOW)
    assert collector.usage_view(th1, MULTS, provider="tu", now=NOW)["windows"]["day"]["used"] == 0.5
    assert collector.usage_view(
        hashlib.sha256(b"tok2").hexdigest(), MULTS, provider="tu", now=NOW
    )["windows"]["day"]["used"] == 1.0


def test_rollover_drops_stale_buckets_and_higher_windows_accumulate(collector):
    th = hashlib.sha256(b"tok1").hexdigest()
    collector.note("tok1", "tu@qwen-3.8-27b", POOL_KEY, now=NOW)
    next_minute = NOW + 120
    collector.note("tok1", "tu@qwen-3.8-27b", POOL_KEY, now=next_minute)
    view = collector.usage_view(th, MULTS, provider="tu", now=next_minute)
    assert view["windows"]["minute"]["used"] == 0.5
    assert view["windows"]["hour"]["used"] == 1.0


def test_day_rollover_resets_everything(collector):
    th = hashlib.sha256(b"tok1").hexdigest()
    collector.note("tok1", "tu@qwen-3.8-27b", POOL_KEY, now=NOW)
    next_day = NOW + 13 * 3600
    collector.note("tok1", "tu@qwen-3.8-27b", POOL_KEY, now=next_day)
    view = collector.usage_view(th, MULTS, provider="tu", now=next_day)
    assert view["windows"]["day"]["used"] == 0.5


def test_note_noop_without_token_or_model(collector):
    th = hashlib.sha256(b"tok1").hexdigest()
    collector.note(None, "tu@qwen-3.8-27b", POOL_KEY, now=NOW)
    collector.note("tok1", "", POOL_KEY, now=NOW)
    assert collector.usage_view(th, MULTS, provider="tu", now=NOW)["windows"]["minute"]["used"] == 0.0


def test_snapshot_v2_roundtrip_and_v1_dropped(collector, monkeypatch, tmp_path):
    th = hashlib.sha256(b"tok1").hexdigest()
    collector.note("tok1", "tu@qwen-3.8-27b", POOL_KEY, now=NOW)
    collector.snapshot()

    KeyUsageCollector._instance = None
    with patch.object(ku.time, "time", return_value=float(NOW + 5)):
        reloaded = get_key_usage()
    view = reloaded.usage_view(th, MULTS, provider="tu", now=NOW + 5)
    assert view["windows"]["minute"]["used"] == 0.5

    # v1 snapshot (pre-key-dimension era) is dropped, not crashed on
    tmp_path.joinpath("key_usage_v1.json").write_text('{"keys": {"x": {}}}')
    monkeypatch.setattr(ku, "_SNAPSHOT_PATH", str(tmp_path / "key_usage_v1.json"))
    KeyUsageCollector._instance = None
    fresh = get_key_usage()
    assert fresh.usage_view("x", MULTS, now=NOW)["windows"]["day"]["used"] == 0.0


def test_key_cap_evicts_least_recently_noted(collector, monkeypatch):
    monkeypatch.setattr(ku, "_MAX_KEYS", 2)
    collector.note("tok1", "tu@x", POOL_KEY, now=NOW)
    collector.note("tok2", "tu@x", POOL_KEY, now=NOW)
    collector.note("tok1", "tu@x", POOL_KEY, now=NOW + 10)
    collector.note("tok3", "tu@x", POOL_KEY, now=NOW + 20)  # evicts tok2
    assert collector.usage_view(hashlib.sha256(b"tok2").hexdigest(), {}, now=NOW + 20)["windows"]["minute"]["used"] == 0.0
    assert collector.usage_view(hashlib.sha256(b"tok1").hexdigest(), {}, now=NOW + 20)["windows"]["minute"]["used"] == 2.0


# --- /v1/usage endpoint ---

@pytest.fixture
def usage_client(monkeypatch, tmp_path):
    import uniinfer.proxy_routers.usage as usage_mod
    from uniinfer.proxy_routers.usage import create_usage_router

    monkeypatch.setattr(ku, "_SNAPSHOT_PATH", str(tmp_path / "key_usage.json"))
    KeyUsageCollector._instance = None
    collector = get_key_usage()
    th = hashlib.sha256(b"timo-token@x").hexdigest()
    other = hashlib.sha256(b"other-key@y").hexdigest()

    # timo: 2x qwen on the DEFAULT pool key, 1x glm on tu-klark (other caller)
    collector.note("timo-token@x", "tu@qwen-3.8-27b", POOL_KEY, now=NOW)
    collector.note("timo-token@x", "tu@qwen-3.8-27b", POOL_KEY, now=NOW)
    collector.note("timo-token@x", "tu@qwen-3.8-27b", OWN_KEY, now=NOW)
    collector.note("other-key@y", "tu@glm-5.3-744b-experimental", KLARK_KEY, now=NOW)
    collector.note("other-key@y", "groq@llama-9", "sk-groq-pool", now=NOW)

    monkeypatch.setattr(
        usage_mod, "_allowed_token_hashes",
        lambda: frozenset([th, other]),
    )
    monkeypatch.setattr(
        usage_mod, "_token_metadata",
        lambda: {"tokens": {th: {"name": "timo-service"}, other: {"name": "other"}}},
    )
    monkeypatch.setattr(
        usage_mod, "_pool_labels",
        lambda prov: {POOL_H: "tu", KLARK_H: "tu-klark"}.copy() if prov == "tu"
        else ({upstream_key_hash("sk-groq-pool"): "groq"}.copy() if prov == "groq" else {}),
    )
    monkeypatch.setattr(usage_mod, "get_all_provider_limits", lambda: {
        "tu": {"requests_per_minute": 50, "requests_per_hour": 1500,
               "requests_per_day": 15000, "source": "https://aqueduct"},
        "groq": {"requests_per_day": 14400},
    })
    monkeypatch.setattr(usage_mod, "get_request_multipliers", lambda pid: MULTS if pid == "tu" else {})
    monkeypatch.setattr(usage_mod, "get_window_limits", lambda pid: {
        "tu": {"minute": 50, "hour": 1500, "day": 15000},
        "groq": {"day": 14400},
    }.get(pid, {}))
    monkeypatch.setattr(usage_mod.time, "time", lambda: float(NOW))

    app = FastAPI()
    app.include_router(create_usage_router())
    yield TestClient(app)
    KeyUsageCollector._instance = None


def test_usage_without_token_401(usage_client):
    assert usage_client.get("/v1/usage").status_code == 401


def test_usage_unknown_token_401(usage_client):
    r = usage_client.get("/v1/usage", headers={"Authorization": "Bearer nope@nope"})
    assert r.status_code == 401


def test_usage_providers_pool_mine_shape(usage_client):
    r = usage_client.get("/v1/usage", headers={"Authorization": "Bearer timo-token@x"})
    assert r.status_code == 200
    body = r.json()
    assert body["key"] == "timo-ser…"

    tu = body["providers"]["tu"]
    pool = tu["pool"]
    by_label = {k["key"]: k for k in pool["keys"]}
    assert by_label["tu"]["used"]["minute"] == 1.0      # timo's 2x qwen x 0.5 on pool key
    assert by_label["tu-klark"]["used"]["minute"] == 1.0  # other caller's glm on klark key
    # Per-key truth, no pool aggregates
    assert set(pool.keys()) == {"keys"}
    assert by_label["tu"]["limits"] == {"minute": 50, "hour": 1500, "day": 15000}
    # mine: caller attribution across pool + own keys, tu-scoped, weighted
    assert tu["mine"]["minute"] == 1.5
    assert "2x" in tu["multiplier_note"]

    groq = body["providers"]["groq"]
    gk = groq["pool"]["keys"][0]
    assert gk["key"] == "groq"
    assert gk["used"]["day"] == 1.0
    assert gk["limits"]["day"] == 14400
    assert groq["mine"]["minute"] == 0.0            # other caller's traffic, not timo's


def test_usage_legacy_windows_preserved(usage_client):
    r = usage_client.get("/v1/usage", headers={"Authorization": "Bearer timo-token@x"})
    w = r.json()["windows"]
    assert w["minute"]["used"] == 1.5               # tu-weighted caller attribution
    assert w["minute"]["limit"] == 50
    assert w["minute"]["reset_at"].endswith("Z")
    assert r.json()["upstream"]["primary_provider"] == "tu"


def test_usage_primary_provider_env(monkeypatch, usage_client):
    monkeypatch.setenv("UNIINFER_USAGE_PRIMARY_PROVIDER", "groq")
    r = usage_client.get("/v1/usage", headers={"Authorization": "Bearer other-key@y"})
    w = r.json()["windows"]
    assert w["day"]["used"] == 1.0                  # groq-scoped legacy view
    assert r.json()["upstream"]["primary_provider"] == "groq"


# --- entity normalization: alias models count toward their entity provider ---

def test_alias_model_counts_toward_entity(collector, monkeypatch):
    """tu-klark@model (overlay alias) is counted under the tu entity."""
    from uniinfer.config import instances as instances_mod

    class _Spec:
        provider = "tu"

    monkeypatch.setattr(instances_mod, "resolve_instance", lambda a: _Spec() if a == "tu-klark" else (_ for _ in ()).throw(ValueError(a)))
    th = hashlib.sha256(b"tok1").hexdigest()
    collector.note("tok1", "tu-klark@qwen-3.8-27b", KLARK_KEY, now=NOW)

    view = collector.usage_view(th, MULTS, provider="tu", now=NOW)
    assert view["windows"]["minute"]["used"] == 0.5
    pool = collector.provider_pool_view("tu", MULTS, now=NOW)
    assert pool[KLARK_H]["minute"] == 0.5


def test_unknown_alias_passes_through_unnormalized(collector, monkeypatch):
    from uniinfer.config import instances as instances_mod

    monkeypatch.setattr(instances_mod, "resolve_instance", lambda a: (_ for _ in ()).throw(ValueError(a)))
    collector.note("tok1", "mystery@model", POOL_KEY, now=NOW)
    pool = collector.provider_pool_view("mystery", {}, now=NOW)
    assert pool[POOL_H]["minute"] == 1.0


# --- entity field: custom aliases over generic classes group by entity ---

class _EntitySpec:
    """Overlay spec stub: novita/novita-2 are openai-compat customs with entity."""
    def __init__(self, entity, provider):
        self.entity = entity
        self.provider = provider


def _patch_entities(monkeypatch, mapping):
    from uniinfer.config import instances as instances_mod
    monkeypatch.setattr(
        instances_mod, "resolve_instance",
        lambda a: mapping[a] if a in mapping else (_ for _ in ()).throw(ValueError(a)),
    )


def test_entity_field_groups_custom_aliases(collector, monkeypatch):
    _patch_entities(monkeypatch, {
        "novita": _EntitySpec("novita", "openai-compat"),
        "novita-2": _EntitySpec("novita", "openai-compat"),
        "hab": _EntitySpec(None, "openai-compat"),  # no entity → class fallback
    })
    th = hashlib.sha256(b"tok1").hexdigest()
    kh1 = upstream_key_hash("sk-novita-1")
    kh2 = upstream_key_hash("sk-novita-2")
    collector.note("tok1", "novita@llama-x", "sk-novita-1", now=NOW)
    collector.note("tok2", "novita-2@llama-x", "sk-novita-2", now=NOW)
    collector.note("tok1", "hab@hab-model", "sk-hab", now=NOW)

    # both novita aliases count toward the novita entity (NOT openai-compat)
    view = collector.usage_view(th, {}, provider="novita", now=NOW)
    assert view["windows"]["minute"]["used"] == 1.0
    pool = collector.provider_pool_view("novita", {}, now=NOW)
    assert pool[kh1]["minute"] == 1.0 and pool[kh2]["minute"] == 1.0
    # hab falls back to its class entity
    hab_pool = collector.provider_pool_view("openai-compat", {}, now=NOW)
    assert hab_pool[upstream_key_hash("sk-hab")]["minute"] == 1.0


def test_entity_model_normalizes_alias_chain(monkeypatch):
    _patch_entities(monkeypatch, {
        "novita-2": _EntitySpec("novita", "openai-compat"),
        "tu-klark": _EntitySpec(None, "tu"),
    })
    from uniinfer.proxy_services.key_usage import entity_model
    assert entity_model("novita-2@llama-x") == "novita@llama-x"
    assert entity_model("tu-klark@qwen-3.8-27b") == "tu@qwen-3.8-27b"
    assert entity_model("unknown@x") == "unknown@x"


def test_usage_per_key_limits_in_payload(usage_client):
    """Each pool key carries its OWN limits (per-key truth for the statusline)."""
    r = usage_client.get("/v1/usage", headers={"Authorization": "Bearer timo-token@x"})
    tu = r.json()["providers"]["tu"]["pool"]
    for entry in tu["keys"]:
        assert entry["limits"] == {"minute": 50, "hour": 1500, "day": 15000}
    assert "limits" not in tu and "used" not in tu and "worst" not in tu  # lean: no aggregates
