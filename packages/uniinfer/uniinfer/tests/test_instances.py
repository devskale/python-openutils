"""Unit tests for the provider-instances overlay (L1 core seam).

The instances file is an overlay on the built-in registry: `enabled` flags +
per-instance overrides + custom aliases. `load_instances()` owns the merge;
`resolve_instance()` is the one alias -> InstanceSpec seam used by completion,
embeddings, the models router, and the CLI.

No network. Uses pytest's tmp_path for a fake config file.
"""
import json
import os
from pathlib import Path

import pytest

from uniinfer.config.instances import (
    InstanceSpec,
    clear_instances_cache,
    get_instances,
    load_instances,
    resolve_instance,
)
from uniinfer.factory import ProviderFactory


def _write(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj) if isinstance(obj, str) is False else obj)
    return path


# --------------------------------------------------------------------------- #
# load_instances — no file = built-ins only
# --------------------------------------------------------------------------- #
def test_no_file_returns_all_builtins_enabled(tmp_path):
    merged = load_instances(path=str(tmp_path / "does-not-exist.json"))

    registered = set(ProviderFactory.list_providers())
    assert set(merged) == registered
    # a known built-in
    spec = merged["ollama"]
    assert spec.is_builtin is True
    assert spec.enabled is True
    assert spec.provider == "ollama"  # built-in: provider key == alias


def test_custom_alias_uses_underlying_class_defaults(tmp_path):
    f = _write(
        tmp_path / "provider_instances.json",
        {"vllm-local": {"provider": "openai-compat", "base_url": "http://localhost:8000/v1"}},
    )
    merged = load_instances(path=str(f))

    assert "vllm-local" in merged
    spec = merged["vllm-local"]
    assert spec.is_builtin is False
    assert spec.provider == "openai-compat"
    assert spec.base_url == "http://localhost:8000/v1"
    assert spec.enabled is True  # default
    assert spec.requires_api_key is True  # inherited from OpenAICompatProvider class default


def test_custom_alias_overrides_requires_api_key(tmp_path):
    f = _write(
        tmp_path / "provider_instances.json",
        {"vllm-local": {"provider": "openai-compat", "base_url": "http://x/v1", "requires_api_key": False}},
    )
    spec = load_instances(path=str(f))["vllm-local"]
    assert spec.requires_api_key is False


def test_override_builtin_base_url_keeps_is_builtin(tmp_path):
    f = _write(
        tmp_path / "provider_instances.json",
        {"ollama": {"base_url": "https://amp1.mooo.com:11444"}},
    )
    merged = load_instances(path=str(f))
    spec = merged["ollama"]
    assert spec.is_builtin is True  # still a built-in, just overridden
    assert spec.base_url == "https://amp1.mooo.com:11444"
    assert spec.provider == "ollama"


def test_disable_builtin_via_enabled_flag(tmp_path):
    f = _write(tmp_path / "provider_instances.json", {"groq": {"enabled": False}})
    assert load_instances(path=str(f))["groq"].enabled is False


def test_custom_alias_credgoo_service_defaults_to_alias(tmp_path):
    # No credgoo_service declared -> custom instance defaults to its own alias name.
    f = _write(
        tmp_path / "provider_instances.json",
        {"vllm-prod": {"provider": "openai-compat", "base_url": "http://x/v1"}},
    )
    assert load_instances(path=str(f))["vllm-prod"].credgoo_service == "vllm-prod"


# --------------------------------------------------------------------------- #
# load_instances — error handling (fail fast at load)
# --------------------------------------------------------------------------- #
def test_unregistered_provider_ref_raises(tmp_path):
    f = _write(
        tmp_path / "provider_instances.json",
        {"bogus": {"provider": "not-a-real-provider", "base_url": "http://x/v1"}},
    )
    with pytest.raises(ValueError):
        load_instances(path=str(f))


def test_custom_alias_missing_provider_field_raises(tmp_path):
    f = _write(tmp_path / "provider_instances.json", {"bogus": {"base_url": "http://x/v1"}})
    with pytest.raises(ValueError):
        load_instances(path=str(f))


def test_malformed_json_raises(tmp_path):
    f = tmp_path / "provider_instances.json"
    f.write_text("{ not valid json ")
    with pytest.raises(ValueError):
        load_instances(path=str(f))


# --------------------------------------------------------------------------- #
# resolve_instance — the one alias -> spec seam
# --------------------------------------------------------------------------- #
def test_resolve_instance_returns_spec(tmp_path):
    f = _write(
        tmp_path / "provider_instances.json",
        {"vllm-local": {"provider": "openai-compat", "base_url": "http://localhost:8000/v1"}},
    )
    instances = load_instances(path=str(f))
    spec = resolve_instance("vllm-local", instances=instances)
    assert isinstance(spec, InstanceSpec)
    assert spec.base_url == "http://localhost:8000/v1"


def test_resolve_instance_unknown_alias_raises(tmp_path):
    instances = load_instances(path=str(tmp_path / "nope.json"))
    with pytest.raises(ValueError):
        resolve_instance("never-registered", instances=instances)


# --------------------------------------------------------------------------- #
# get_instances — mtime-cached loader + graceful-degrade (Q12)
# --------------------------------------------------------------------------- #
def test_get_instances_caches_by_mtime(tmp_path):
    clear_instances_cache()
    f = _write(tmp_path / "provider_instances.json", {"vllm-local": {"provider": "openai-compat", "base_url": "http://x/v1"}})
    os.utime(f, (1000, 1000))
    first = get_instances(path=str(f))
    second = get_instances(path=str(f))  # same mtime -> cache hit, same object
    assert first is second


def test_get_instances_graceful_degrade_keeps_last_good(tmp_path, caplog):
    clear_instances_cache()
    f = _write(tmp_path / "provider_instances.json", {"vllm-local": {"provider": "openai-compat", "base_url": "http://x/v1"}})
    os.utime(f, (1000, 1000))
    good = get_instances(path=str(f))
    assert "vllm-local" in good

    # Corrupt the file at a newer mtime -> reload fails -> keep last-good, no raise.
    f.write_text("{ broken json")
    os.utime(f, (2000, 2000))
    with caplog.at_level("WARNING"):
        again = get_instances(path=str(f))
    assert again is good  # same cached object served
    assert "vllm-local" in again


def test_get_instances_raises_on_first_bad_load(tmp_path):
    clear_instances_cache()
    f = tmp_path / "provider_instances.json"
    f.write_text("{ broken json")
    os.utime(f, (1000, 1000))
    with pytest.raises(ValueError):
        get_instances(path=str(f))  # nothing cached yet -> fail at boot
