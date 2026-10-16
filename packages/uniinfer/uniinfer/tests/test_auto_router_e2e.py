"""Auto-routing end-to-end through the real chat router (hermetic).

Spins up the FastAPI TestClient against the real app wiring with a fake free
provider: `auto@free` must resolve to the fake provider's model, the response
must carry the concrete model + x-routed-model header, and stats must record
the concrete model.
"""

import json

import pytest
from fastapi.testclient import TestClient

from uniinfer.config.instances import InstanceSpec
from uniinfer.core import ChatCompletionResponse, ChatMessage, ChatProvider, ModelInfo
from uniinfer.factory import ProviderFactory
from uniinfer.proxy_services import router as routing


class _FakeFreeProvider(ChatProvider):
    PROVIDER_ID = "fakefreeprov"
    REQUIRES_API_KEY = False
    BASE_URL = ""

    def complete(self, request, **kwargs):
        return ChatCompletionResponse(
            message=ChatMessage(role="assistant", content="hello from fake free"),
            model="fake-free-1",
            provider="fakefreeprov",
            usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            raw_response={},
        )

    async def acomplete(self, request, **kwargs):
        return self.complete(request, **kwargs)

    def stream_complete(self, request, **kwargs):
        yield ChatCompletionResponse(
            message={"role": "assistant", "content": "stream"},
            model="fake-free-1",
            provider="fakefreeprov",
        )

    async def astream_complete(self, request, **kwargs):
        for chunk in self.stream_complete(request, **kwargs):
            yield chunk

    @classmethod
    def list_models(cls, api_key=None, base_url=None, **kwargs):
        return [ModelInfo(id="fake-free-1", owned_by="fakefreeprov")]


@pytest.fixture
def app_client(monkeypatch, tmp_path):
    ProviderFactory.register_provider("fakefreeprov", _FakeFreeProvider)
    routing._RoutingPrefs._cache = None
    routing._RoutingPrefs._cache_mtime = None

    def _fake_catalog():
        return [
            {
                "id": "fakefreeprov@fake-free-1",
                "provider": "fakefreeprov",
                "type": "chat",
                "access": "free",
                "status": "active",
            }
        ]

    monkeypatch.setattr(
        routing,
        "Catalog",
        lambda: type("C", (), {"list_resolved": staticmethod(_fake_catalog)})(),
    )
    monkeypatch.setattr(routing, "_degraded_models", lambda: set())
    monkeypatch.setattr(routing, "_provider_utilization_pct", lambda p: None)
    ov = tmp_path / "model_overrides.json"
    ov.write_text(json.dumps({"models": {}}))
    monkeypatch.setattr(routing, "_router_overrides_path", lambda: str(ov))
    monkeypatch.setattr(
        "uniinfer.proxy_routers.chat.verify_provider_access", lambda t, p: "fake-key"
    )
    monkeypatch.setattr(
        "uniinfer.completion.resolve_instance",
        lambda alias, **k: InstanceSpec(
            alias=alias, provider=alias, requires_api_key=False, is_builtin=False
        ),
    )

    from uniinfer.proxy_app import app

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client

    ProviderFactory._providers.pop("fakefreeprov", None)
    routing._RoutingPrefs._cache = None
    routing._RoutingPrefs._cache_mtime = None


def _post_chat(client, model, **extra):
    return client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test@fake"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Say hi"}],
            **extra,
        },
    )


def test_auto_free_resolves_to_concrete_model(app_client):
    r = _post_chat(app_client, "auto@free")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "fakefreeprov@fake-free-1"
    assert r.headers.get("x-routed-model") == "fakefreeprov@fake-free-1"
    assert body["choices"][0]["message"]["content"] == "hello from fake free"


def test_auto_free_no_candidates_503(app_client, monkeypatch):
    monkeypatch.setattr(
        routing, "_degraded_models", lambda: {"fakefreeprov@fake-free-1"}
    )
    r = _post_chat(app_client, "auto@free")
    assert r.status_code == 503
    assert "no available free model" in r.json()["detail"]


def test_auto_unknown_selector_400(app_client):
    r = _post_chat(app_client, "auto@paid")
    assert r.status_code == 400
