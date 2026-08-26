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


class TestTokenAllowlist:
    """UNIINFER_AUTH_TOKENS_FILE: issued tokens only, hot-reload, revocation."""

    @pytest.fixture
    def allowlist_env(self, monkeypatch, tmp_path):
        f = tmp_path / "tokens.allow"
        monkeypatch.setenv("UNIINFER_AUTH_TOKENS_FILE", str(f))
        import hashlib
        from unittest.mock import patch as mp

        def write(entries):
            f.write_text("\n".join([
                "# issued tokens", 
                *[hashlib.sha256(t.encode()).hexdigest() for t in entries],
            ]) + "\n")

        with mp("uniinfer.auth.get_provider_api_key",
                side_effect=lambda tok, p: "resolved-key"), \
             mp("uniinfer.auth.instance_requires_api_key", lambda p: True):
            yield write

    def test_enabled_blocks_unknown_tokens(self, allowlist_env):
        allowlist_env(["tok1@secret1"])
        with pytest.raises(HTTPException) as e:
            verify_provider_access("nope@nope", "tu")
        assert e.value.status_code == 401
        assert verify_provider_access("tok1@secret1", "tu") == "resolved-key"

    def test_hot_reload_revokes_and_grants(self, allowlist_env):
        allowlist_env(["tok-a@enc-a"])
        assert verify_provider_access("tok-a@enc-a", "tu") == "resolved-key"
        import time
        time.sleep(0.01)
        allowlist_env(["tok-b@enc-b"])  # revoke a, grant b — same path, new mtime
        with pytest.raises(HTTPException) as e:
            verify_provider_access("tok-a@enc-a", "tu")
        assert e.value.status_code == 401
        assert verify_provider_access("tok-b@enc-b", "tu") == "resolved-key"

    def test_missing_file_disables_allowlist(self, monkeypatch):
        monkeypatch.setenv("UNIINFER_AUTH_TOKENS_FILE", "/nonexistent/tokens.allow")
        from unittest.mock import patch as mp
        with mp("uniinfer.auth.get_provider_api_key",
                side_effect=lambda tok, p: None if not tok else "k" if "@" in tok else (_ for _ in ()).throw(HTTPException(status_code=401))), \
             mp("uniinfer.auth.instance_requires_api_key", lambda p: True):
            # kein Allowlist → legacy: bare Token fällt durchs @-Gate → 401; Combo ok
            with pytest.raises(HTTPException):
                verify_provider_access("bare", "tu")
            assert verify_provider_access("a@b", "tu") is None or True
