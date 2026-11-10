"""Per-key usage collector + /v1/usage endpoint (read-only quota attribution).

Hermetic: KeyUsageCollector isolated via tmp snapshot + singleton reset
(same pattern as test_stats_ttft.py), fake time via explicit now= params
(and a patched time.time for the snapshot-reload roundtrip). Router tests
patch the auth metadata lookups directly — no env files, no live config.

Covers: UTC window floors (minute/hour/midnight), Aqueduct multiplier
weighting (2x-Limits models count 0.5, aliases included, non-tu weighs 0),
per-key isolation, window rollover pruning, snapshot roundtrip, key-cap
eviction, and the endpoint contract: 401 without/unknown bearer, weighted
windows + limits + reset_at + proxy_total + per-model breakdown.
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
    weight_of,
)

MULTS = {"qwen-3.8-27b": 0.5, "vision": 0.5, "ocr": 0.5}

# 2026-11-11 13:34:30 UTC — mid-window, ±30s clearance from every boundary.
NOW = int(datetime(2026, 11, 11, 13, 34, 30, tzinfo=timezone.utc).timestamp())


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
    assert datetime.fromtimestamp(start, timezone.utc).strftime("%Y-%m-%d") == "2026-11-11"


def test_reset_at_is_next_boundary():
    assert bucket_end("minute", bucket_start("minute", NOW)) - NOW == 30
    assert bucket_end("hour", bucket_start("hour", NOW)) - NOW == 25 * 60 + 30
    assert bucket_end("day", bucket_start("day", NOW)) - NOW == 10 * 3600 + 25 * 60 + 30


# --- weighting ---

def test_weight_of_applies_multipliers_and_aliases():
    assert weight_of("tu@qwen-3.8-27b", MULTS) == 0.5
    assert weight_of("tu@vision", MULTS) == 0.5
    assert weight_of("tu@ocr", MULTS) == 0.5
    assert weight_of("tu@deepseek-v4-vision-exp-284b", MULTS) == 1.0


def test_weight_of_non_tu_is_zero():
    assert weight_of("openrouter@foo", MULTS) == 0.0
    assert weight_of("openrouter@foo", {}) == 0.0


# --- collector behavior ---

def test_note_counts_and_weights_per_key(collector):
    th = hashlib.sha256(b"tok1").hexdigest()
    collector.note("tok1", "tu@qwen-3.8-27b", now=NOW)
    collector.note("tok1", "tu@qwen-3.8-27b", now=NOW)
    collector.note("tok1", "tu@vision", now=NOW)
    collector.note("tok1", "tu@deepseek-v4-vision-exp-284b", now=NOW)
    collector.note("tok1", "openrouter@foo", now=NOW)

    view = collector.usage_view(th, MULTS, now=NOW)
    assert view["windows"]["minute"]["used"] == 2.5  # 2*0.5 + 0.5 + 1.0
    models = {m["model"]: m for m in view["models"]}
    assert models["tu@qwen-3.8-27b"]["requests"] == 2
    assert models["tu@qwen-3.8-27b"]["weighted"] == 1.0
    assert "openrouter@foo" not in models  # non-tu excluded from breakdown


def test_keys_are_isolated(collector):
    th1 = hashlib.sha256(b"tok1").hexdigest()
    collector.note("tok1", "tu@qwen-3.8-27b", now=NOW)
    collector.note("tok2", "tu@qwen-3.8-27b", now=NOW)
    collector.note("tok2", "tu@qwen-3.8-27b", now=NOW)
    assert collector.usage_view(th1, MULTS, now=NOW)["windows"]["day"]["used"] == 0.5
    assert collector.usage_view(hashlib.sha256(b"tok2").hexdigest(), MULTS, now=NOW)["windows"]["day"]["used"] == 1.0


def test_rollover_drops_stale_buckets_and_higher_windows_accumulate(collector):
    th = hashlib.sha256(b"tok1").hexdigest()
    collector.note("tok1", "tu@qwen-3.8-27b", now=NOW)
    next_minute = NOW + 120  # minute + hour bucket roll, day stays
    collector.note("tok1", "tu@qwen-3.8-27b", now=next_minute)

    view = collector.usage_view(th, MULTS, now=next_minute)
    assert view["windows"]["minute"]["used"] == 0.5
    assert view["windows"]["hour"]["used"] == 1.0
    assert view["windows"]["day"]["used"] == 1.0


def test_day_rollover_resets_everything(collector):
    th = hashlib.sha256(b"tok1").hexdigest()
    collector.note("tok1", "tu@qwen-3.8-27b", now=NOW)
    next_day = NOW + 13 * 3600  # past midnight UTC
    collector.note("tok1", "tu@vision", now=next_day)
    view = collector.usage_view(th, MULTS, now=next_day)
    assert view["windows"]["minute"]["used"] == 0.5
    assert view["windows"]["hour"]["used"] == 0.5
    assert view["windows"]["day"]["used"] == 0.5


def test_proxy_total_sums_all_keys(collector):
    collector.note("tok1", "tu@qwen-3.8-27b", now=NOW)
    collector.note("tok2", "tu@qwen-3.8-27b", now=NOW)
    collector.note("tok3", "openrouter@foo", now=NOW)
    assert collector.proxy_total("minute", MULTS, now=NOW) == 1.0
    assert collector.proxy_total("day", MULTS, now=NOW) == 1.0


def test_note_noop_without_token_or_model(collector):
    th = hashlib.sha256(b"tok1").hexdigest()
    collector.note(None, "tu@qwen-3.8-27b", now=NOW)
    collector.note("tok1", "", now=NOW)
    assert collector.usage_view(th, MULTS, now=NOW)["windows"]["minute"]["used"] == 0.0


def test_snapshot_roundtrip_preserves_current_buckets_only(collector, monkeypatch, tmp_path):
    th = hashlib.sha256(b"tok1").hexdigest()
    collector.note("tok1", "tu@qwen-3.8-27b", now=NOW)
    collector.snapshot()

    KeyUsageCollector._instance = None
    with patch.object(ku.time, "time", return_value=float(NOW + 5)):
        reloaded = get_key_usage()  # same minute/hour/day → all buckets survive
    view = reloaded.usage_view(th, MULTS, now=NOW + 5)
    assert view["windows"]["minute"]["used"] == 0.5
    assert view["windows"]["hour"]["used"] == 0.5
    assert view["windows"]["day"]["used"] == 0.5


def test_reload_drops_stale_minute_buckets(collector, monkeypatch, tmp_path):
    th = hashlib.sha256(b"tok1").hexdigest()
    collector.note("tok1", "tu@qwen-3.8-27b", now=NOW)
    collector.snapshot()

    KeyUsageCollector._instance = None
    with patch.object(ku.time, "time", return_value=float(NOW + 120)):
        reloaded = get_key_usage()  # next minute: minute bucket stale, hour/day current
    view = reloaded.usage_view(th, MULTS, now=NOW + 120)
    assert view["windows"]["minute"]["used"] == 0.0
    assert view["windows"]["hour"]["used"] == 0.5


def test_key_cap_evicts_least_recently_noted(collector, monkeypatch):
    monkeypatch.setattr(ku, "_MAX_KEYS", 2)
    collector.note("tok1", "tu@qwen-3.8-27b", now=NOW)
    collector.note("tok2", "tu@qwen-3.8-27b", now=NOW)
    collector.note("tok1", "tu@qwen-3.8-27b", now=NOW + 10)  # refresh tok1
    collector.note("tok3", "tu@qwen-3.8-27b", now=NOW + 20)  # evicts tok2
    assert collector.usage_view(hashlib.sha256(b"tok2").hexdigest(), MULTS, now=NOW + 20)["windows"]["minute"]["used"] == 0.0
    assert collector.usage_view(hashlib.sha256(b"tok1").hexdigest(), MULTS, now=NOW + 20)["windows"]["minute"]["used"] == 1.0
    assert collector.usage_view(hashlib.sha256(b"tok3").hexdigest(), MULTS, now=NOW + 20)["windows"]["minute"]["used"] == 0.5


# --- /v1/usage endpoint ---

@pytest.fixture
def usage_client(monkeypatch, tmp_path):
    import uniinfer.proxy_routers.usage as usage_mod
    from uniinfer.proxy_routers.usage import create_usage_router

    monkeypatch.setattr(ku, "_SNAPSHOT_PATH", str(tmp_path / "key_usage.json"))
    KeyUsageCollector._instance = None
    collector = get_key_usage()
    th = hashlib.sha256(b"timo-token@x").hexdigest()
    collector.note("timo-token@x", "tu@qwen-3.8-27b", now=NOW)
    collector.note("timo-token@x", "tu@qwen-3.8-27b", now=NOW)
    collector.note("other-key@y", "tu@glm-5.3-744b-experimental", now=NOW)

    monkeypatch.setattr(
        usage_mod, "_allowed_token_hashes",
        lambda: frozenset([th, hashlib.sha256(b"other-key@y").hexdigest()]),
    )
    monkeypatch.setattr(
        usage_mod, "_token_metadata",
        lambda: {"tokens": {th: {"name": "timo-service"}, hashlib.sha256(b"other-key@y").hexdigest(): {"name": "other"}}},
    )
    monkeypatch.setattr(usage_mod, "get_provider_limits", lambda pid: {
        "requests_per_minute": 50, "requests_per_hour": 1500, "requests_per_day": 15000,
    } if pid == "tu" else {})
    monkeypatch.setattr(usage_mod, "get_request_multipliers", lambda pid: MULTS if pid == "tu" else {})
    # Freeze the endpoint's clock inside the fixture's window (the collector
    # notes above use fake NOW; the route must read the same buckets).
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


def test_usage_shape_and_attribution(usage_client):
    r = usage_client.get("/v1/usage", headers={"Authorization": "Bearer timo-token@x"})
    assert r.status_code == 200
    body = r.json()
    assert body["key"] == "timo-ser…"  # truncated name, no secret
    w = body["windows"]
    assert w["minute"]["used"] == 1.0  # 2 x qwen x 0.5
    assert w["minute"]["limit"] == 50
    assert w["hour"]["limit"] == 1500
    assert w["day"]["limit"] == 15000
    assert w["minute"]["proxy_total"] == pytest.approx(2.0)  # timo 1.0 + other glm 1.0
    reset = datetime.fromisoformat(w["minute"]["reset_at"].replace("Z", "+00:00"))
    assert reset.tzinfo is not None
    models = {m["model"]: m for m in body["models"]}
    assert models["tu@qwen-3.8-27b"]["requests"] == 2
    assert models["tu@qwen-3.8-27b"]["weighted"] == 1.0
    up = body["upstream"]
    assert up["provider"] == "tu"
    assert "SHARED" in up["note"]


def test_usage_limits_absent_when_not_documented(usage_client, monkeypatch):
    import uniinfer.proxy_routers.usage as usage_mod
    monkeypatch.setattr(usage_mod, "get_provider_limits", lambda pid: {})
    r = usage_client.get("/v1/usage", headers={"Authorization": "Bearer timo-token@x"})
    assert r.json()["windows"]["minute"]["limit"] is None
