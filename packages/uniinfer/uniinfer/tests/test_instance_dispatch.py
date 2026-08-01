"""Slice 2: instance dispatch wiring — Target routes via resolve_instance.

A custom alias declared in the overlay must instantiate its *underlying* class
with the instance's base_url, and credgoo-service resolution must consult the
instance before the class attr. No network: a fake provider records its
construction args.
"""
import json
import os

import pytest

from uniinfer.completion import Target
from uniinfer.core import ChatProvider
from uniinfer.factory import ProviderFactory
from uniinfer.config.instances import clear_instances_cache
from uniinfer.provider_access import _resolve_credgoo_service


class _FakeProvider(ChatProvider):
    """Records construction args so tests can assert routing."""

    BASE_URL = ""
    PROVIDER_ID = "fakecompat"
    DEFAULT_MODEL = "fake-model"

    def __init__(self, api_key=None, base_url=None, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        self.base_url = base_url
        self.api_key = api_key


@pytest.fixture
def fake_provider():
    ProviderFactory.register_provider("fakecompat", _FakeProvider)
    yield "fakecompat"
    ProviderFactory._providers.pop("fakecompat", None)


@pytest.fixture
def instances_env(tmp_path, monkeypatch):
    """Point UNIINFER_INSTANCES_FILE at a tmp overlay and return its path."""
    clear_instances_cache()
    path = tmp_path / "provider_instances.json"
    monkeypatch.setenv("UNIINFER_INSTANCES_FILE", str(path))
    yield path
    clear_instances_cache()


def test_target_routes_custom_alias_to_base_url(fake_provider, instances_env):
    instances_env.write_text(
        json.dumps({"myfake": {"provider": "fakecompat", "base_url": "http://example.test/v1"}})
    )
    target = Target("myfake@some-model", record_access=False)
    assert isinstance(target.provider, _FakeProvider)
    assert target.provider.base_url == "http://example.test/v1"


def test_target_explicit_base_url_overrides_instance(fake_provider, instances_env):
    instances_env.write_text(
        json.dumps({"myfake": {"provider": "fakecompat", "base_url": "http://from-file/v1"}})
    )
    target = Target("myfake@m", base_url="http://explicit/v1", record_access=False)
    assert target.provider.base_url == "http://explicit/v1"  # caller wins


def test_target_builtin_still_works(fake_provider, instances_env):
    # No file entry for the built-in mistral -> routes via the registry as before.
    target = Target("mistral@mistral-tiny-latest", api_key="k", record_access=False)
    assert target.provider_name == "mistral"


# --------------------------------------------------------------------------- #
# credgoo service resolution — instance wins, then class attr
# --------------------------------------------------------------------------- #
def test_resolve_credgoo_service_uses_instance_field(instances_env):
    instances_env.write_text(
        json.dumps({"vllm-prod": {"provider": "openai-compat", "base_url": "http://x/v1", "credgoo_service": "shared-key"}})
    )
    assert _resolve_credgoo_service("vllm-prod") == "shared-key"


def test_resolve_credgoo_service_custom_default_is_alias(instances_env):
    instances_env.write_text(
        json.dumps({"vllm-local": {"provider": "openai-compat", "base_url": "http://x/v1"}})
    )
    assert _resolve_credgoo_service("vllm-local") == "vllm-local"


def test_resolve_credgoo_service_builtin_uses_class_attr(instances_env):
    # zai-code's class declares CREDGOO_SERVICE = "zai-code"; no instance override.
    assert _resolve_credgoo_service("zai-code") == "zai-code"
