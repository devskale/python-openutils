"""Hermetische Tests für die daily model-gather routine (generate_models.py).

Verhalten entsteht aus Kombination — kein echter Netzwerk-Call, kein echter
Provider. Jede Testform entspricht einem historischen Verstoß:

- Cache war write-once-forever (prod: 4 Monate altes models.dev-Enrichment)
- Korrupter/truncated Cache hätte jeden künftigen Run gekillt (unkapterter
  json.load im Hot Path von main())
- Transient leerer Provider flog für 24h aus dem Listing (Listing-Flap)
- _stale_models.json wurde non-atomar geschrieben
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import generate_models as gm


# ------------------------------------------------------------------ #
# fetch_models_dev: TTL + stale-beats-none + corrupt-cache robustness
# ------------------------------------------------------------------ #
def _write_cache(tmp_path, payload, mtime=None):
    cache = tmp_path / "_models_dev_cache.json"
    cache.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    if mtime is not None:
        os.utime(cache, (mtime, mtime))
    return cache


def test_cache_fresh_is_used_without_refetch(tmp_path, monkeypatch):
    cache = _write_cache(tmp_path, {"openai": {"models": {"gpt-x": {}}}},
                         mtime=time.time())
    monkeypatch.setattr(gm, "MODELS_DEV_CACHE", cache)
    calls = []
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(AssertionError("refetch!")),
    )
    data = gm.fetch_models_dev()
    assert data == {"openai": {"models": {"gpt-x": {}}}}
    assert calls == []


def test_cache_stale_triggers_refetch_and_atomic_rewrite(tmp_path, monkeypatch):
    cache = _write_cache(tmp_path, {"old": True}, mtime=time.time() - 30 * 86400)
    monkeypatch.setattr(gm, "MODELS_DEV_CACHE", cache)

    class Resp:
        def __enter(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"openai": {"models": {"fresh": {}}}}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Resp())
    monkeypatch.setattr("urllib.request.Request", lambda *a, **k: None)

    data = gm.fetch_models_dev()
    assert data == {"openai": {"models": {"fresh": {}}}}
    # atomic rewrite: no .tmp leftover, file replaced
    assert json.loads(cache.read_text())["openai"]["models"] == {"fresh": {}}
    assert not (tmp_path / "_models_dev_cache.json.tmp").exists()
    assert cache.stat().st_mtime > time.time() - 60


def test_refetch_failure_serves_stale_cache(tmp_path, monkeypatch):
    cache = _write_cache(tmp_path, {"old": True}, mtime=time.time() - 30 * 86400)
    monkeypatch.setattr(gm, "MODELS_DEV_CACHE", cache)

    def boom(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    monkeypatch.setattr("urllib.request.Request", lambda *a, **k: None)
    data = gm.fetch_models_dev()
    assert data == {"old": True}


def test_corrupt_cache_refetches_or_degrades_to_empty(tmp_path, monkeypatch):
    _write_cache(tmp_path, '{"truncated": ', mtime=time.time())
    monkeypatch.setattr(gm, "MODELS_DEV_CACHE", tmp_path / "_models_dev_cache.json")

    def boom(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    monkeypatch.setattr("urllib.request.Request", lambda *a, **k: None)
    assert gm.fetch_models_dev() == {}  # kein Crash — Run läuft ohne Enrichment weiter


# ------------------------------------------------------------------ #
# carry_forward_empty_providers: Listing-Flap-Schutz
# ------------------------------------------------------------------ #
def test_empty_provider_carried_forward():
    existing = {
        "groq": {"models": [{"id": "old-groq-1"}]},
        "openai": {"models": [{"id": "gpt-1"}]},
    }
    result = {"openai": {"models": [{"id": "gpt-1"}]}}  # groq transient leer
    carried = gm.carry_forward_empty_providers(result, existing, {"groq"})
    assert carried == 1
    assert result["groq"]["models"] == [{"id": "old-groq-1"}]


def test_provider_with_real_zero_models_not_carried():
    existing = {"dead": {"models": [{"id": "x"}]}}
    result = {}
    # 'dead' lieferte erfolgreich LEER (nicht in empty_providers — der Fetch
    # war ok): kein Carry, Modell-Tod decayt über last_seen/PRUNE_DAYS.
    assert gm.carry_forward_empty_providers(result, existing, set()) == 0
    assert "dead" not in result


def test_carry_never_overrides_fresh_result():
    existing = {"groq": {"models": [{"id": "stale"}]}}
    result = {"groq": {"models": [{"id": "fresh"}]}}
    gm.carry_forward_empty_providers(result, existing, {"groq"})
    assert result["groq"]["models"] == [{"id": "fresh"}]
