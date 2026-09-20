"""Images routes must pass the SAME auth gate as chat (allowlist + combo +
bare-key rejection), failing closed — issue: images-route-bypasses-auth-gate."""

import hashlib
import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import uniinfer.auth as auth_mod
import uniinfer.proxy_routers.images as images_mod
from uniinfer.proxy_routers.images import create_images_router

COMBO = "deadbeef@cafebabe"  # bearer@encryption shaped test grant
COMBO_HASH = hashlib.sha256(COMBO.encode()).hexdigest()


@pytest.fixture
def client(tmp_path, monkeypatch):
    # allowlist with exactly one issued combo
    allow = tmp_path / "allow"
    allow.write_text(COMBO_HASH + "\n")
    monkeypatch.setenv("UNIINFER_AUTH_TOKENS_FILE", str(allow))
    # keyed provider semantics like the live gateway
    monkeypatch.setattr(auth_mod, "instance_requires_api_key", lambda p: True)
    monkeypatch.setattr(
        auth_mod, "get_provider_api_key",
        lambda token, provider: "upstream-key" if (token and token == COMBO) else (_ for _ in ()).throw(ValueError("no key")),
    )

    def parse(model):  # "tu@img" -> ("tu", "img")
        p, _, rest = model.partition("@")
        if not p or not rest:
            raise ValueError("Invalid model format. Expected 'provider@modelname'")
        return p, rest

    class FakeTarget:
        def __init__(self, model, api_key=None):
            assert api_key == "upstream-key", "route must forward the RESOLVED key"

        async def agenerate(self, prompt, n=1, size="512x512", client=None):
            item = MagicMock()
            item.to_dict = lambda: {"url": "https://x/img.png"}
            return [item]

    monkeypatch.setattr(images_mod, "ImageTarget", FakeTarget)

    app = FastAPI()
    app.include_router(create_images_router(parse))
    return TestClient(app)


def test_generations_without_token_401(client):
    r = client.post("/v1/images/generations", json={"model": "tu@img", "prompt": "x"})
    assert r.status_code == 401


def test_generations_bare_key_rejected(client):
    r = client.post("/v1/images/generations", json={"model": "tu@img", "prompt": "x"},
                    headers={"Authorization": "Bearer bare-upstream-key"})
    assert r.status_code == 401  # allowlist fires first: unknown token


def test_generations_unknown_token_401(client):
    r = client.post("/v1/images/generations", json={"model": "tu@img", "prompt": "x"},
                    headers={"Authorization": "Bearer someone-else@nope"})
    assert r.status_code == 401
    assert "Unknown or revoked" in r.json()["detail"]


def test_generations_allowlisted_combo_200(client):
    r = client.post("/v1/images/generations", json={"model": "tu@img", "prompt": "x"},
                    headers={"Authorization": f"Bearer {COMBO}"})
    assert r.status_code == 200
    assert "img.png" in r.text


def test_models_list_bad_token_401(client):
    r = client.get("/v1/image/models/pollinations", headers={"Authorization": "Bearer someone-else@nope"})
    assert r.status_code == 401


def test_models_list_no_token_stays_public(client):
    """The public catalog stays listable without credentials (no quota usage)."""
    r = client.get("/v1/image/models/nonexistent-provider")
    assert r.status_code == 200
    assert r.json()["data"] == []
