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


def test_keyed_instance_forwards_upstream_bearer(monkeypatch):
    """Keyed-Instanz: der von verify_provider_access aufgelöste Key wandert
    als Upstream-Bearer in den Forward (Gateway-Auth am Ziel, z.B. dgx
    model-proxy hinter einem Tunnel) — nicht nur als Caller-Gate."""
    spec = _spec("d3mini", keyless=False)
    monkeypatch.setattr("uniinfer.proxy_routers.systemone.resolve_instance",
                        lambda a: spec)
    monkeypatch.setattr("uniinfer.proxy_routers.systemone.instance_requires_api_key",
                        lambda a: True)
    monkeypatch.setattr("uniinfer.proxy_routers.systemone.verify_provider_access",
                        lambda token, alias: "sk-dgxp-upstream-token")

    captured = {}

    class Resp:
        status_code = 200
        def json(self):
            return {"answers": {"team": {"type": "choice", "choice": "billing"}}}

    async def fake_post(self, url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return Resp()

    import httpx
    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.setattr("uniinfer.core._shared_async_client",
                        lambda: httpx.AsyncClient())

    c = _app(spec)
    r = c.post("/v1/systemone",
               headers={"authorization": "Bearer caller-combo-token"},
               json={"model": "d3mini@vllm-sr/d3-mini", "state": {}, "questions": {}})
    assert r.status_code == 200
    assert captured["headers"]["Authorization"] == "Bearer sk-dgxp-upstream-token"


def _wrap_setup(monkeypatch, content, usage=None):
    """Instanz ohne base_url (built-in) + gefaktes Target für den Chat-Pfad."""
    spec = _spec("opencode", keyless=False)
    spec = InstanceSpec(alias="opencode", provider="opencode", is_builtin=True,
                        base_url=None, requires_api_key=True)
    monkeypatch.setattr("uniinfer.proxy_routers.systemone.resolve_instance",
                        lambda a: spec)
    monkeypatch.setattr("uniinfer.proxy_routers.systemone.instance_requires_api_key",
                        lambda a: True)
    monkeypatch.setattr("uniinfer.proxy_routers.systemone.verify_provider_access",
                        lambda token, alias: "sk-upstream")

    captured = {}

    class FakeTarget:
        def __init__(self, provider_model, api_key=None, **kw):
            captured["provider_model"] = provider_model
            captured["api_key"] = api_key

        async def acomplete(self, messages, **kw):
            captured["messages"] = messages
            from types import SimpleNamespace
            msg = SimpleNamespace(content=content)
            return SimpleNamespace(message=msg, usage=usage or {})

    monkeypatch.setattr("uniinfer.completion.Target", FakeTarget)
    return captured


def test_chat_wrap_fallback_answers_object(monkeypatch):
    """Built-in ohne base_url: Chat-Wrap, Content = direktes Answers-Objekt."""
    import json as _json
    content = _json.dumps({"team": {"type": "choice", "choice": "billing",
                                    "probabilities": {"billing": 0.9, "tech": 0.1}}})
    captured = _wrap_setup(monkeypatch, content)
    c = _app(None)
    r = c.post("/v1/systemone",
               headers={"authorization": "Bearer caller"},
               json={"model": "opencode@jev-1.13-free",
                     "state": {"s": "x"}, "questions": {"team": {}}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answers"]["team"]["choice"] == "billing"
    assert body["model"] == "opencode@jev-1.13-free"
    # Wrap-Payload: letzte user-Nachricht = JSON mit state+questions
    sent = _json.loads(captured["messages"][0]["content"])
    assert sent["state"] == {"s": "x"} and "team" in sent["questions"]
    assert captured["provider_model"] == "opencode@jev-1.13-free"
    assert captured["api_key"] == "sk-upstream"


def test_chat_wrap_fallback_answers_key(monkeypatch):
    """Content = {"answers": {...}} → inneres Objekt wird gehoben."""
    import json as _json
    content = _json.dumps({"answers": {"ok": {"type": "noul", "noul": 0.8}},
                           "usage": {"input_tokens": 5}})
    _wrap_setup(monkeypatch, content)
    c = _app(None)
    r = c.post("/v1/systemone",
               headers={"authorization": "Bearer caller"},
               json={"model": "opencode@jev-1.13", "state": {}, "questions": {}})
    assert r.status_code == 200
    assert r.json()["answers"]["ok"]["noul"] == 0.8


def test_chat_wrap_non_json_content_502(monkeypatch):
    _wrap_setup(monkeypatch, "Ich bin kein JSON, nur Text.")
    c = _app(None)
    r = c.post("/v1/systemone",
               headers={"authorization": "Bearer caller"},
               json={"model": "opencode@jev-1.13", "state": {}, "questions": {}})
    assert r.status_code == 502
    assert "Systemone dialect" in r.json()["error"]
