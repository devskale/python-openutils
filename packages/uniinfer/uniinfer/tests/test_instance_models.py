"""Slice 3 (item 3): per-instance model listing — /v1/models/{alias}.

list_models_for_provider resolves the alias via resolve_instance and lists
through the underlying class with the instance's base_url, so a custom fleet
member (vllm-local) exposes its models just like a built-in.
"""
import json

import pytest

from uniinfer.config.instances import clear_instances_cache
from uniinfer.core import ChatProvider, ModelInfo
from uniinfer.factory import ProviderFactory
from uniinfer.provider_access import list_models_for_provider


class _FakeListProvider(ChatProvider):
    BASE_URL = ""
    PROVIDER_ID = "fakecompat"
    REQUIRES_API_KEY = False
    _last_base_url = None

    @classmethod
    def list_models(cls, api_key=None, base_url=None, **kwargs):
        cls._last_base_url = base_url
        return [ModelInfo(id="fake-1", owned_by="fakecompat"), ModelInfo(id="fake-2", owned_by="fakecompat")]


@pytest.fixture
def fake_provider():
    ProviderFactory.register_provider("fakecompat", _FakeListProvider)
    yield "fakecompat"
    ProviderFactory._providers.pop("fakecompat", None)


@pytest.fixture
def instances_env(tmp_path, monkeypatch):
    clear_instances_cache()
    path = tmp_path / "provider_instances.json"
    monkeypatch.setenv("UNIINFER_INSTANCES_FILE", str(path))
    # Avoid catalog/disk side effects in the test.
    monkeypatch.setattr("uniinfer.provider_access.update_models", lambda *a, **k: None)
    monkeypatch.setattr(
        "uniinfer.proxy_services.models_registry.Catalog.upsert_provider",
        lambda self, *a, **k: None,
    )
    yield path
    clear_instances_cache()


def test_list_models_custom_alias_uses_instance_base_url(fake_provider, instances_env):
    instances_env.write_text(
        json.dumps({"myfake": {"provider": "fakecompat", "base_url": "http://example.test/v1", "requires_api_key": False}})
    )
    models = list_models_for_provider("myfake", None)
    assert [str(m) for m in models] == ["fake-1", "fake-2"]
    assert _FakeListProvider._last_base_url == "http://example.test/v1"


def test_list_models_unknown_alias_raises(instances_env):
    with pytest.raises(ValueError):
        list_models_for_provider("never-registered", None)


# --------------------------------------------------------------------------- #
# merge_custom_aliases — daily regenerate must not wipe fleet entries
# --------------------------------------------------------------------------- #
def test_merge_custom_aliases_preserves_custom_entry():
    from uniinfer.config.instances import InstanceSpec, merge_custom_aliases

    builtins = {"ollama": {"models": []}, "mistral": {"models": []}}
    existing = {
        "vllm-local": {"models": [{"id": "x"}]},  # a custom fleet member
        "ollama": {"models": ["STALE"]},           # stale built-in copy
    }
    aliases = {
        "vllm-local": InstanceSpec(alias="vllm-local", provider="openai-compat", is_builtin=False),
        "ollama": InstanceSpec(alias="ollama", provider="ollama", is_builtin=True),
    }
    merged = merge_custom_aliases(builtins, existing, aliases)
    assert "vllm-local" in merged              # custom alias preserved
    assert merged["ollama"] == builtins["ollama"]  # built-in NOT overwritten by stale existing


def test_merge_custom_aliases_skips_unknown():
    from uniinfer.config.instances import InstanceSpec, merge_custom_aliases

    # an existing catalog key that is NOT a declared instance is dropped (no orphan)
    builtins = {"ollama": {"models": []}}
    existing = {"orphan": {"models": []}}
    merged = merge_custom_aliases(builtins, existing, {})
    assert "orphan" not in merged
