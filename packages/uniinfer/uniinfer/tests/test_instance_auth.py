"""Slice 3 (item 1): proxy auth via instance.requires_api_key.

Replaces the hardcoded ``provider_name == 'ollama'`` keyless special-case with
the instance's ``requires_api_key`` flag, so ANY keyless endpoint (local vLLM,
LM Studio, localhost Ollama) is auth-optional through the proxy — not just ollama.
"""
import json

import pytest

from uniinfer.config.instances import (
    clear_instances_cache,
    instance_requires_api_key,
)
from uniinfer.provider_access import get_provider_api_key
from uniinfer.auth import verify_provider_access


@pytest.fixture
def instances_env(tmp_path, monkeypatch):
    clear_instances_cache()
    path = tmp_path / "provider_instances.json"
    monkeypatch.setenv("UNIINFER_INSTANCES_FILE", str(path))
    monkeypatch.delenv("CREDGOO_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("CREDGOO_ENCRYPTION_KEY", raising=False)
    yield path
    clear_instances_cache()


# --------------------------------------------------------------------------- #
# instance_requires_api_key — the one flag that replaces the ollama special-case
# --------------------------------------------------------------------------- #
def test_builtin_ollama_is_keyless(instances_env):
    assert instance_requires_api_key("ollama") is False


def test_builtin_mistral_requires_key(instances_env):
    assert instance_requires_api_key("mistral") is True


def test_custom_keyless_alias(instances_env):
    instances_env.write_text(
        json.dumps({"vllm-local": {"provider": "openai-compat", "base_url": "http://x/v1", "requires_api_key": False}})
    )
    assert instance_requires_api_key("vllm-local") is False


# --------------------------------------------------------------------------- #
# get_provider_api_key — keyless providers don't demand a token
# --------------------------------------------------------------------------- #
def test_keyless_no_token_returns_none(instances_env):
    # ollama, no bearer, no credgoo env -> None, no raise
    assert get_provider_api_key(None, "ollama") is None


def test_required_no_token_raises(instances_env):
    with pytest.raises(ValueError):
        get_provider_api_key(None, "mistral")


def test_keyless_with_credgoo_token_resolves(instances_env, monkeypatch):
    # A keyless provider still picks up its credgoo key when a combo token is given
    # (e.g. amp1 ollama needs a bearer to access the server itself).
    captured = {}

    def fake_get(service, encryption_key=None, bearer_token=None):
        captured["service"] = service
        return f"key-for-{service}"

    monkeypatch.setattr("uniinfer.provider_access.get_api_key", fake_get)
    out = get_provider_api_key("bearer@enc", "ollama")
    assert out == "key-for-ollama"
    assert captured["service"] == "ollama"


# --------------------------------------------------------------------------- #
# verify_provider_access — HTTP seam honours requires_api_key
# --------------------------------------------------------------------------- #
def test_verify_keyless_none_key_is_ok(instances_env):
    assert verify_provider_access(None, "ollama") is None  # no 401


def test_verify_required_no_token_is_401(instances_env):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        verify_provider_access(None, "mistral")
    assert exc.value.status_code == 401
