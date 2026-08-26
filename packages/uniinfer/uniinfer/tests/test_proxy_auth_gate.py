"""
Proxy auth gate: bare (non-'@') bearer tokens are rejected for keyed providers.

Live finding 2026-08-26: keyed providers (tu) are served from the process-wide
POOLED client whose Authorization header carries the SERVER's own credentials —
any junk bearer got served on our quota once the pool was warm. Consumers must
present a credgoo combined token ('bearer@encryption') that resolves via
credgoo.
"""
import pytest
from fastapi import HTTPException

from uniinfer.auth import verify_provider_access


@pytest.fixture
def patched(monkeypatch):
    calls = {"resolved": None, "resolution_calls": 0}

    def fake_get_key(token, provider):
        calls["resolution_calls"] += 1
        if token:
            calls["resolved"] = f"tu-key-of({token.split('@')[0]})"
        return calls["resolved"]

    monkeypatch.setattr("uniinfer.auth.get_provider_api_key", fake_get_key)
    monkeypatch.setattr("uniinfer.auth.instance_requires_api_key", lambda p: p != "ollama")
    return calls


class TestBareTokenGate:
    def test_bare_token_rejected(self, patched):
        with pytest.raises(HTTPException) as e:
            verify_provider_access("totally-wrong-bearer", "tu")
        assert e.value.status_code == 401
        assert "credgoo combined token" in str(e.value.detail)
        assert patched["resolution_calls"] == 0  # Gate stoppt VOR credgoo-Auflösung

    def test_combo_token_resolves(self, patched):
        key = verify_provider_access("bearer-part@encryption-part", "tu")
        assert key == "tu-key-of(bearer-part)"

    def test_missing_token_still_rejected(self, patched):
        with pytest.raises(HTTPException) as e:
            verify_provider_access("", "tu")
        assert e.value.status_code == 401

    def test_keyless_provider_accepts_bare_or_empty(self, patched):
        # ollama & Co: requires_api_key=False → Gate aus, Pass-through ok
        assert verify_provider_access(None, "ollama") is None
        assert verify_provider_access("anything", "ollama") is not None
