"""Tests for operator token minting and gateway-enforced token metadata."""

import hashlib
import importlib.util
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException

from uniinfer.auth import clear_token_caches, verify_provider_access
from uniinfer.proxy_services import token_registry as registry


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _cli_main(argv):
    path = Path(__file__).resolve().parents[2] / "scripts" / "unii-token.py"
    spec = importlib.util.spec_from_file_location("unii_token_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main(argv)


@pytest.fixture
def registry_env(tmp_path, monkeypatch):
    allow = tmp_path / "auth_tokens.allow"
    metadata = tmp_path / "auth_tokens.meta.json"
    monkeypatch.setenv("UNIINFER_AUTH_TOKENS_FILE", str(allow))
    monkeypatch.setenv("UNIINFER_AUTH_TOKENS_META_FILE", str(metadata))
    clear_token_caches()
    yield allow, metadata
    clear_token_caches()


@pytest.fixture
def auth_env(monkeypatch):
    monkeypatch.setattr(
        "uniinfer.auth.get_provider_api_key", lambda token, provider: "upstream-key"
    )
    monkeypatch.setattr(
        "uniinfer.auth.instance_requires_api_key", lambda provider: True
    )


def test_mint_stores_hash_and_metadata_once(registry_env):
    allow, metadata = registry_env
    now = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
    token, record = registry.mint_token(
        "habit",
        "30d",
        ["tu", "pollinations"],
        now=now,
    )

    assert token.startswith("u") and token.count("@") == 1
    assert record.allowlisted is True
    assert record.sha256 == _hash(token)
    assert allow.read_text().count(record.sha256) == 1
    assert allow.stat().st_mode & 0o777 == 0o600
    assert metadata.stat().st_mode & 0o777 == 0o600

    document = json.loads(metadata.read_text())
    stored = document["tokens"][record.sha256]
    assert stored["name"] == "habit"
    assert stored["providers"] == ["tu", "pollinations"]
    assert stored["expires_at"] == "2026-10-24T12:00:00Z"
    assert token not in metadata.read_text()


def test_mint_rejects_duplicate_active_name(registry_env):
    registry.mint_token("agent", "30d")
    with pytest.raises(registry.TokenRegistryError, match="already exists"):
        registry.mint_token("agent", "30d")


def test_list_reports_allowlisted_state(registry_env):
    allow, _ = registry_env
    registry.mint_token("active", "30d")
    revoked = registry.revoke_token(name="active")
    registry.mint_token("kept", "never")

    records = {record.name: record for record in registry.list_tokens()}
    assert records["kept"].allowlisted is True
    assert records["kept"].expires_at is None
    assert revoked[0].name not in records
    assert "kept@" not in allow.read_text()


def test_revoke_updates_allowlist_and_metadata(registry_env):
    allow, metadata = registry_env
    token, record = registry.mint_token("temporary", "30d")

    revoked = registry.revoke_token(name="temporary")

    assert [item.sha256 for item in revoked] == [record.sha256]
    assert record.sha256 not in allow.read_text()
    assert record.sha256 not in json.loads(metadata.read_text())["tokens"]
    with pytest.raises(registry.TokenRegistryError):
        registry.revoke_token(name="temporary")
    with pytest.raises(HTTPException) as revoked_error:
        verify_provider_access(token, "tu")
    assert revoked_error.value.status_code == 401
    assert "Unknown or revoked token" in revoked_error.value.detail


def test_auth_enforces_expiry_and_provider_scope(registry_env, auth_env):
    allow, metadata = registry_env
    future = datetime.now(timezone.utc) + timedelta(days=1)
    past = datetime.now(timezone.utc) - timedelta(seconds=1)
    live_token, live = registry.mint_token("live", "30d", providers=["tu"], now=future)
    expired_token, expired = registry.mint_token("expired", "30d", now=past)

    document = json.loads(metadata.read_text())
    document["tokens"][expired.sha256]["expires_at"] = past.isoformat().replace(
        "+00:00", "Z"
    )
    metadata.write_text(json.dumps(document))
    clear_token_caches()

    assert verify_provider_access(live_token, "tu") == "upstream-key"
    with pytest.raises(HTTPException) as scope_error:
        verify_provider_access(live_token, "groq")
    assert scope_error.value.status_code == 401
    assert "not authorized for this provider" in scope_error.value.detail

    with pytest.raises(HTTPException) as expiry_error:
        verify_provider_access(expired_token, "tu")
    assert expiry_error.value.status_code == 401
    assert "Token expired" in expiry_error.value.detail
    assert expired.name  # record remains meaningful to tests


def test_auth_hot_reloads_expiry(registry_env, auth_env):
    allow, metadata = registry_env
    deadline = datetime.now(timezone.utc) + timedelta(seconds=2)
    token, record = registry.mint_token("fleet", "30d", now=deadline)

    assert verify_provider_access(token, "tu") == "upstream-key"

    document = json.loads(metadata.read_text())
    document["tokens"][record.sha256]["expires_at"] = (
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    metadata.write_text(json.dumps(document))
    os.utime(metadata, (time.time() + 0.01, time.time() + 0.01))
    clear_token_caches()

    with pytest.raises(HTTPException) as error:
        verify_provider_access(token, "tu")
    assert error.value.status_code == 401
    assert "Token expired" in error.value.detail


def test_auth_fails_closed_on_unreadable_metadata(registry_env, auth_env):
    allow, metadata = registry_env
    token, record = registry.mint_token("broken-meta", "30d")
    metadata.write_text("{ broken json")
    os.utime(metadata, (time.time() + 0.01, time.time() + 0.01))
    clear_token_caches()

    with pytest.raises(HTTPException) as error:
        verify_provider_access(token, "tu")
    assert error.value.status_code == 401
    assert "metadata unavailable" in error.value.detail


def test_cli_mint_prints_token_once(registry_env, capsys, monkeypatch):
    allow, metadata = registry_env
    exit_code = _cli_main(["mint", "--name", "cli-agent", "--ttl", "7d"])
    assert exit_code == 0
    captured = capsys.readouterr()
    token = captured.out.strip()
    assert token.startswith("u") and token.count("@") == 1
    assert token not in captured.err
    assert _hash(token) in allow.read_text()
    assert token not in metadata.read_text()


def test_cli_rejects_invalid_ttl(registry_env, capsys):
    assert _cli_main(["mint", "--name", "bad", "--ttl", "soon"]) == 2
    assert "invalid TTL" in capsys.readouterr().err
