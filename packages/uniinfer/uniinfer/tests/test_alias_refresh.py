"""Slice 3 (item d): per-alias stale-while-revalidate.

/v1/models/{alias} for a custom alias serves the cached catalog entry
immediately when fresh; when stale it serves the stale cache AND schedules a
background refresh (non-blocking); on the first ever hit (no cache) it fetches
synchronously. Driven by a per-alias last_refreshed timestamp.
"""
import pytest

from uniinfer.config.instances import alias_serve_decision


# --------------------------------------------------------------------------- #
# alias_serve_decision — the pure coordinator (age, ttl, inflight) -> action
# --------------------------------------------------------------------------- #
def test_no_cache_fetches_sync():
    assert alias_serve_decision(age=None, ttl=300, inflight=False) == "fetch_sync"


def test_fresh_serves_cached():
    assert alias_serve_decision(age=10, ttl=300, inflight=False) == "serve_cached"


def test_stale_schedules_background_refresh():
    assert alias_serve_decision(age=400, ttl=300, inflight=False) == "serve_cached_and_refresh"


def test_stale_but_inflight_just_serves_cached():
    # a refresh is already running -> don't stack another
    assert alias_serve_decision(age=400, ttl=300, inflight=True) == "serve_cached"


# --------------------------------------------------------------------------- #
# Catalog: upsert stamps last_refreshed; provider_age_seconds reads it
# --------------------------------------------------------------------------- #
def test_upsert_stamps_last_refreshed(tmp_path):
    from uniinfer.core import ModelInfo
    from uniinfer.proxy_services.models_registry import Catalog

    cat = Catalog(path=str(tmp_path / "models.json"))
    cat.upsert_provider("vllm-local", [ModelInfo(id="m1", owned_by="x")])
    age = cat.provider_age_seconds("vllm-local")
    assert age is not None
    assert age < 5  # just stamped


def test_age_none_when_no_entry(tmp_path):
    from uniinfer.proxy_services.models_registry import Catalog

    cat = Catalog(path=str(tmp_path / "models.json"))
    assert cat.provider_age_seconds("never-cached") is None


def test_age_grows_for_old_entry(tmp_path):
    from datetime import datetime, timezone, timedelta
    from uniinfer.proxy_services.models_registry import Catalog
    import json

    p = tmp_path / "models.json"
    old = (datetime.now(timezone.utc) - timedelta(seconds=3600)).isoformat()
    p.write_text(json.dumps({"providers": {"vllm-local": {"last_refreshed": old, "models": [{"id": "m"}]}}}))
    age = Catalog(path=str(p)).provider_age_seconds("vllm-local")
    assert age is not None and age > 3500
