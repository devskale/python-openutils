"""Tests for progressive agent discovery endpoints."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from uniinfer.config.instances import InstanceSpec
from uniinfer.core import ModelInfo
from uniinfer.proxy_app import app
from uniinfer.proxy_services import discovery
from uniinfer.proxy_services.models_registry import Catalog


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


def _catalog() -> dict[str, Any]:
    return {
        "providers": {
            "alpha": {
                "models": [
                    {"id": "one"},
                    {"id": "two"},
                ]
            },
            "beta": {"models": [{"id": "three"}]},
        }
    }


def test_v1_manifest_is_public_and_links_discovery(client):
    response = client.get("/v1")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "uniinfer"
    assert body["protocol"] == "openai-compatible"
    assert body["model_id_format"] == "provider@model"
    assert body["links"]["providers"] == "/v1/providers"
    assert body["links"]["models"] == "/v1/models"
    assert body["links"]["openapi"] == "/openapi.json"
    assert "base_url" not in body
    assert 'rel="service-desc"' in response.headers["Link"]


def test_root_negotiates_json_discovery(client):
    response = client.get("/", headers={"Accept": "application/json"})
    assert response.status_code == 200
    assert response.json()["links"]["providers"] == "/v1/providers"

    html = client.get("/", headers={"Accept": "text/html"})
    assert html.status_code == 200
    assert "text/html" in html.headers["content-type"]


def test_provider_discovery_lists_enabled_instances_without_infra(client, monkeypatch):
    specs = {
        "alpha": InstanceSpec(
            alias="alpha",
            provider="alpha",
            requires_api_key=True,
            default_model="one",
            is_builtin=True,
        ),
        "zenfg": InstanceSpec(
            alias="zenfg",
            provider="opencode",
            requires_api_key=True,
            default_model="three",
            is_builtin=False,
        ),
        "disabled": InstanceSpec(
            alias="disabled",
            provider="alpha",
            enabled=False,
            is_builtin=True,
        ),
    }
    monkeypatch.setattr(discovery, "get_instances", lambda: specs)
    monkeypatch.setattr(
        Catalog,
        "read_nested",
        lambda self, provider_filter=None: _catalog(),
    )

    response = client.get("/v1/providers")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert [item["id"] for item in body["data"]] == ["alpha", "zenfg"]

    alpha = body["data"][0]
    assert alpha["kind"] == "builtin"
    assert alpha["model_count"] == 2
    assert alpha["models_url"] == "/v1/models?provider=alpha"
    assert alpha["live_models_url"] == "/v1/models/alpha"

    custom = body["data"][1]
    assert custom["kind"] == "custom"
    assert custom["underlying_provider"] == "opencode"
    assert "base_url" not in custom
    assert "credgoo_service" not in custom
    assert "credgoo" not in response.text.lower()


def test_models_field_projection_keeps_navigation_core(client, monkeypatch):
    async def no_refresh():
        return None

    monkeypatch.setattr(
        "uniinfer.proxy_routers.models.ensure_fresh_models_file", no_refresh
    )
    monkeypatch.setattr(
        Catalog,
        "list_resolved",
        lambda self: [
            {
                "id": "alpha@one",
                "object": "model",
                "provider": "alpha",
                "type": "chat",
                "access": "free",
                "context_window": 128_000,
                "cost": {"input": 0},
                "secret_internal_value": "must-not-leak",
            }
        ],
    )

    response = client.get("/v1/models?provider=alpha&fields=id,context_window")
    assert response.status_code == 200
    model = response.json()["data"][0]
    assert model == {
        "id": "alpha@one",
        "object": "model",
        "provider": "alpha",
        "context_window": 128_000,
    }


def test_instance_models_support_field_projection(client, monkeypatch):
    spec = InstanceSpec(
        alias="slim-fleet", provider="opencode", is_builtin=False, requires_api_key=True
    )

    def fetch_models(alias, token):
        return [ModelInfo(id="three")]

    monkeypatch.setattr(
        "uniinfer.proxy_routers.models.resolve_instance", lambda alias: spec
    )
    monkeypatch.setattr(
        "uniinfer.proxy_routers.models.verify_provider_access",
        lambda token, provider: "operator-key",
    )
    monkeypatch.setattr(
        "uniinfer.proxy_routers.models.list_models_for_provider", fetch_models
    )

    response = client.get(
        "/v1/models/slim-fleet?fields=id,type",
        headers={"Authorization": "Bearer operator-token"},
    )
    assert response.status_code == 200
    assert response.json()["data"] == [
        {"id": "three", "object": "model", "provider": "slim-fleet"}
    ]


def test_models_field_projection_rejects_unknown_fields(client, monkeypatch):
    async def no_refresh():
        return None

    monkeypatch.setattr(
        "uniinfer.proxy_routers.models.ensure_fresh_models_file", no_refresh
    )
    response = client.get("/v1/models?fields=id,not_a_field")
    assert response.status_code == 400
    assert "unknown model fields" in response.json()["detail"]
