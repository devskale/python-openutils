"""Systemone-Pass-Through: generisch über Instanz-Aliase, kein Modell-Hardcode.

Fälle:
- dgemma-ähnlicher keyless Custom-Alias → forwarded an base_url/systemone,
  alias@-Präfix gestript, Antwort relayed
- unbekannter Alias → 404
- fehlendes model-Feld → 400
- requires_api_key-Instanz ohne Token → 401 (Gate wie im Chat)
"""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from uniinfer.config.instances import InstanceSpec
from uniinfer.proxy_routers.systemone import create_systemone_router


def _parse(model: str):
    alias, _, rest = model.partition("@")
    return alias, rest


def _app(spec, require_token=False):
    app = FastAPI()
    app.include_router(create_systemone_router(parse_provider_model=_parse))
    return TestClient(app)


def _spec(alias, keyless=True):
    return InstanceSpec(alias=alias, provider="openai-compat", is_builtin=False,
                        base_url=f"https://{alias}.example/v1",
                        requires_api_key=not keyless)


def test_forward_strips_prefix_and_relays(monkeypatch):
    spec = _spec("dgemma")
    monkeypatch.setattr("uniinfer.proxy_routers.systemone.resolve_instance",
                        lambda a: spec)
    monkeypatch.setattr("uniinfer.proxy_routers.systemone.instance_requires_api_key",
                        lambda a: False)

    captured = {}

    class Resp:
        status_code = 200
        def json(self):
            return {"answers": {"x": {"type": "noul", "noul": 0.9}}}

    async def fake_post(self, url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return Resp()

    import httpx
    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.setattr("uniinfer.core._shared_async_client",
                        lambda: httpx.AsyncClient())

    c = _app(spec)
    r = c.post("/v1/systemone", json={
        "model": "dgemma@google/diffusiongemma-26B-A4B-it",
        "state": {"note": "log line"},
        "questions": {"alarm": {"type": "noul", "text": "Alarm?"}},
    })
    assert r.status_code == 200
    assert r.json()["answers"]["x"]["noul"] == 0.9
    assert captured["url"] == "https://dgemma.example/v1/systemone"
    assert captured["json"]["model"] == "google/diffusiongemma-26B-A4B-it"  # prefix weg
    assert "state" in captured["json"] and "questions" in captured["json"]


def test_unknown_alias_404(monkeypatch):
    monkeypatch.setattr("uniinfer.proxy_routers.systemone.resolve_instance",
                        lambda a: (_ for _ in ()).throw(ValueError("nope")))
    c = _app(None)
    r = c.post("/v1/systemone", json={"model": "nope@m", "state": {}, "questions": {}})
    assert r.status_code == 404


def test_missing_model_400():
    c = _app(None)
    r = c.post("/v1/systemone", json={"state": {}, "questions": {}})
    assert r.status_code == 400
    assert "model" in r.json()["error"]


def test_api_key_instance_requires_token(monkeypatch):
    spec = _spec("gated", keyless=False)
    monkeypatch.setattr("uniinfer.proxy_routers.systemone.resolve_instance",
                        lambda a: spec)
    monkeypatch.setattr("uniinfer.proxy_routers.systemone.instance_requires_api_key",
                        lambda a: True)
    monkeypatch.setattr("uniinfer.proxy_routers.systemone.verify_provider_access",
                        lambda token, alias: (_ for _ in ()).throw(
                            PermissionError("no access")))
    c = _app(spec)
    r = c.post("/v1/systemone", json={"model": "gated@m", "state": {}, "questions": {}})
    assert r.status_code == 401
