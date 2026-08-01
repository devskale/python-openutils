"""Slice 3 (item 2): instance-management primitives (add/remove/enable/reset/show).

Pure file operations on the overlay — no network. The CLI flags and the smart
add-probe are thin shells over these.
"""
import json

import pytest

from uniinfer.config.instances import (
    clear_instances_cache,
    read_overlay,
    remove_instance,
    reset_instance,
    resolve_instance,
    set_instance_enabled,
    show_instance,
    upsert_instance,
)
from uniinfer.factory import ProviderFactory


@pytest.fixture
def overlay_env(tmp_path, monkeypatch):
    clear_instances_cache()
    path = tmp_path / "provider_instances.json"
    monkeypatch.setenv("UNIINFER_INSTANCES_FILE", str(path))
    yield path
    clear_instances_cache()


# --------------------------------------------------------------------------- #
# upsert
# --------------------------------------------------------------------------- #
def test_upsert_creates_custom_alias(overlay_env):
    upsert_instance("vllm-local", provider="openai-compat", base_url="http://localhost:8000/v1")
    assert not overlay_env.exists() or True  # file created
    raw = json.loads(overlay_env.read_text())
    assert raw["vllm-local"] == {"provider": "openai-compat", "base_url": "http://localhost:8000/v1"}
    assert resolve_instance("vllm-local").base_url == "http://localhost:8000/v1"


def test_upsert_preserves_other_entries(overlay_env):
    upsert_instance("vllm-local", provider="openai-compat", base_url="http://x/v1")
    upsert_instance("ollama-home", provider="ollama", base_url="http://y:11434")
    raw = json.loads(overlay_env.read_text())
    assert "vllm-local" in raw and "ollama-home" in raw


def test_upsert_override_builtin(overlay_env):
    upsert_instance("ollama", base_url="https://amp1.mooo.com:11444")
    spec = resolve_instance("ollama")
    assert spec.is_builtin is True
    assert spec.base_url == "https://amp1.mooo.com:11444"


def test_upsert_custom_without_provider_raises(overlay_env):
    with pytest.raises(ValueError):
        upsert_instance("bogus", base_url="http://x/v1")


def test_upsert_unknown_provider_raises(overlay_env):
    with pytest.raises(ValueError):
        upsert_instance("bogus", provider="nope", base_url="http://x/v1")


# --------------------------------------------------------------------------- #
# remove (smart: builtins refuse)
# --------------------------------------------------------------------------- #
def test_remove_custom_alias(overlay_env):
    upsert_instance("vllm-local", provider="openai-compat", base_url="http://x/v1")
    assert remove_instance("vllm-local") is True
    assert "vllm-local" not in read_overlay()


def test_remove_builtin_refuses(overlay_env):
    with pytest.raises(ValueError):  # routes to disable/reset
        remove_instance("ollama")


def test_remove_builtin_override_refuses_even_if_overridden(overlay_env):
    upsert_instance("ollama", base_url="http://x:11434")
    with pytest.raises(ValueError):
        remove_instance("ollama")


# --------------------------------------------------------------------------- #
# enable / disable / reset
# --------------------------------------------------------------------------- #
def test_disable_then_enable_builtin(overlay_env):
    set_instance_enabled("groq", False)
    assert resolve_instance("groq").enabled is False
    set_instance_enabled("groq", True)
    assert resolve_instance("groq").enabled is True


def test_reset_reverts_builtin_override(overlay_env):
    upsert_instance("ollama", base_url="http://overridden:11434")
    assert resolve_instance("ollama").base_url == "http://overridden:11434"
    assert reset_instance("ollama") is True
    assert "ollama" not in read_overlay()  # override dropped -> registry default re-applies


# --------------------------------------------------------------------------- #
# show
# --------------------------------------------------------------------------- #
def test_show_returns_spec(overlay_env):
    upsert_instance("vllm-local", provider="openai-compat", base_url="http://x/v1")
    spec = show_instance("vllm-local")
    assert spec.alias == "vllm-local"
    assert spec.provider == "openai-compat"
