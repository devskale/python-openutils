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
from uniinfer.errors import InvalidRequestError, RateLimitError
from uniinfer.factory import ProviderFactory
from uniinfer.proxy_services import router as routing


class _FakeFreeProvider(ChatProvider):
    PROVIDER_ID = "fakefreeprov"
    REQUIRES_API_KEY = False
    BASE_URL = ""
    fail_first_n = 0
    dead_first = False

    def complete(self, request, **kwargs):
        if _FakeFreeProvider.dead_first and request.model == "fake-free-1":
            raise InvalidRequestError(
                'OpenCode API error: 400 - {"message":'
                '"Upstream request failed: Model is unavailable."}',
                status_code=400,
                response_body='{"message":"Upstream request failed: Model is unavailable."}',
            )
        if _FakeFreeProvider.fail_first_n > 0:
            _FakeFreeProvider.fail_first_n -= 1
            raise RateLimitError(
                "fake quota spent",
                status_code=429,
                response_body="{}",
                retry_after=900.0,
            )
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
    _clear_personal_caches()
    prefs_file = tmp_path / "routing_preferences.json"
    monkeypatch.setattr(routing, "_routing_prefs_path", lambda: str(prefs_file))
    monkeypatch.delenv("UNIINFER_AUTH_TOKENS_FILE", raising=False)

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
    _clear_personal_caches()


def _clear_personal_caches():
    routing._PersonalPrefs._cache = None
    routing._PersonalPrefs._cache_mtime = None
    routing._TokenNames._cache = None


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


def test_routing_preview_endpoint(app_client):
    """GET /v1/routing/auto: trockene Resolver-Sicht — gewaehltes Ziel, keine
    Upstream-Kosten, inkl. Cooldown-Snapshots."""
    r = app_client.get("/v1/routing/auto")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "routing"
    assert body["model"] == "auto@free"
    assert body["resolved"] == "fakefreeprov@fake-free-1"
    assert body["reason"] in ("pin", "weight")
    assert body["candidates"] == 1
    assert isinstance(body["parked"], dict)
    assert isinstance(body["dead_parks"], dict)


def test_routing_preview_reports_unusable_router(app_client, monkeypatch):
    """Keine ueberlebenden Kandidaten: Preview bleibt 200 und berichtet
    resolved=null + Skipped-Breakdown (ein Request wuerde 503 geben)."""
    monkeypatch.setattr(
        routing, "_degraded_models", lambda: {"fakefreeprov@fake-free-1"}
    )
    r = app_client.get("/v1/routing/auto")
    assert r.status_code == 200
    body = r.json()
    assert body["resolved"] is None
    assert body["candidates"] == 0
    assert body["skipped"]["degraded"] == 1


def test_routing_preview_non_auto_and_unknown_selector_400(app_client):
    r = app_client.get("/v1/routing/auto", params={"model": "mistral@x"})
    assert r.status_code == 400
    r = app_client.get("/v1/routing/auto", params={"model": "auto@paid"})
    assert r.status_code == 400


def test_auto_free_stream_carries_routed_header(app_client):
    """Auch der Streaming-Pfad traegt x-routed-model — davor war der Resolve
    im SSE-Fall nur aus chunk.model lesbar (Header existierte nur non-stream)."""
    r = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test@fake"},
        json={
            "model": "auto@free",
            "messages": [{"role": "user", "content": "Say hi"}],
            "stream": True,
        },
    )
    assert r.status_code == 200, r.text
    assert "text/event-stream" in r.headers.get("content-type", "")
    assert r.headers.get("x-routed-model") == "fakefreeprov@fake-free-1"
    assert "fakefreeprov@fake-free-1" in r.text


def test_personal_prefs_route_per_caller(app_client, monkeypatch):
    """Der gleiche auto@free-Call routet je nach Bearer anders: alice (via
    Token-Metadaten-Sidecar identifiziert) preferiert fake-free-2, anonym
    bleibt es bei der Katalogordnung (fake-free-1)."""
    import hashlib

    def _two_catalog_models():
        return [
            {
                "id": "fakefreeprov@fake-free-1",
                "provider": "fakefreeprov",
                "type": "chat",
                "access": "free",
                "status": "active",
            },
            {
                "id": "fakefreeprov@fake-free-2",
                "provider": "fakefreeprov",
                "type": "chat",
                "access": "free",
                "status": "active",
            },
        ]

    monkeypatch.setattr(
        routing,
        "Catalog",
        lambda: type("C", (), {"list_resolved": staticmethod(_two_catalog_models)})(),
    )
    prefs_path = routing._routing_prefs_path()
    from pathlib import Path

    prefs_path = Path(prefs_path)
    prefs_path.write_text(
        json.dumps(
            {"tokens": {"alice": {"prefer": {"models": ["fake-free-2"]}}}}
        )
    )
    routing._PersonalPrefs.invalidate()
    h = hashlib.sha256(b"test@fake").hexdigest()
    allow = prefs_path.parent / "auth_tokens.allow"
    meta = prefs_path.parent / "auth_tokens.meta.json"
    allow.write_text(h + "\n")
    meta.write_text(json.dumps({"version": 1, "tokens": {h: {"name": "alice"}}}))
    monkeypatch.setenv("UNIINFER_AUTH_TOKENS_FILE", str(allow))
    routing._TokenNames.invalidate()

    r = _post_chat(app_client, "auto@free")
    assert r.status_code == 200, r.text
    assert r.json()["model"] == "fakefreeprov@fake-free-2"
    assert r.headers.get("x-routed-model") == "fakefreeprov@fake-free-2"

    # Preview: mit Alices Bearer wird ihre Sicht sichtbar, anonym die Default-
    # Sicht (Katalogordnung)
    r = app_client.get(
        "/v1/routing/auto", headers={"Authorization": "Bearer test@fake"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["preferences"]["caller"] == "alice"
    assert body["preferences"]["active"] is True
    assert body["resolved"] == "fakefreeprov@fake-free-2"

    r_anon = app_client.get("/v1/routing/auto")
    assert r_anon.json()["preferences"] == {
        "caller": None,
        "active": False,
        "avoided": 0,
    }
    assert r_anon.json()["resolved"] == "fakefreeprov@fake-free-1"


def test_auto_free_listed_in_v1_models(app_client):
    """auto@free taucht in /v1/models auf — Clients entdecken das virtuelle
    Modell wie jedes konkrete."""
    r = app_client.get("/v1/models")
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert "auto@free" in ids
    entry = next(m for m in r.json()["data"] if m["id"] == "auto@free")
    assert entry["provider"] == "auto"
    assert entry["type"] == "chat"
    # type-filter chat laesst es drin, embedding-filter wirft es raus
    r_chat = app_client.get("/v1/models", params={"type": "chat"})
    assert "auto@free" in [m["id"] for m in r_chat.json()["data"]]
    r_embed = app_client.get("/v1/models", params={"type": "embedding"})
    assert "auto@free" not in [m["id"] for m in r_embed.json()["data"]]


def test_auto_quota_hit_falls_through_to_next_model(app_client, monkeypatch):
    """Kernszenario: Tokens des gewaehlten Modells sind verbraucht (429) —
    der Router parkt es und bedient den Request vom naechsten Kandidaten."""
    from uniinfer.proxy_services import router as routing

    def _two_catalog_models():
        return [
            {
                "id": "fakefreeprov@fake-free-1",
                "provider": "fakefreeprov",
                "type": "chat",
                "access": "free",
                "status": "active",
            },
            {
                "id": "fakefreeprov@fake-free-2",
                "provider": "fakefreeprov",
                "type": "chat",
                "access": "free",
                "status": "active",
            },
        ]

    monkeypatch.setattr(
        routing,
        "Catalog",
        lambda: type("C", (), {"list_resolved": staticmethod(_two_catalog_models)})(),
    )
    routing._QuotaCooldown._parked = {}
    _FakeFreeProvider.fail_first_n = 1  # fake-free-1 antwortet 429
    try:
        r = _post_chat(app_client, "auto@free")
        assert r.status_code == 200, r.text
        body = r.json()
        # durchgefallen auf kandidat 2
        assert body["model"] == "fakefreeprov@fake-free-2"
        assert r.headers.get("x-routed-model") == "fakefreeprov@fake-free-2"
        # und kandidat 1 ist geparkt
        assert "fakefreeprov@fake-free-1" in routing._QuotaCooldown.parked()
    finally:
        _FakeFreeProvider.fail_first_n = 0
        routing._QuotaCooldown._parked = {}


def test_auto_all_candidates_quota_hit_503(app_client, monkeypatch):
    """Alle Kandidaten verbraucht: sauberer 503/429 statt Endlos-Loop."""
    from uniinfer.proxy_services import router as routing

    routing._QuotaCooldown._parked = {}
    _FakeFreeProvider.fail_first_n = 99  # alles 429t
    try:
        r = _post_chat(app_client, "auto@free")
        assert r.status_code in (429, 503)
    finally:
        _FakeFreeProvider.fail_first_n = 0
        routing._QuotaCooldown._parked = {}


def test_auto_dead_upstream_falls_through_and_parks(app_client, monkeypatch):
    """Dead-upstream ("Model is unavailable"): naechster Kandidat bedient, das
    tote Modell wird lang geparkt (nicht nur 15 min)."""
    from uniinfer.proxy_services import router as routing

    def _two_catalog_models():
        return [
            {
                "id": "fakefreeprov@fake-free-1",
                "provider": "fakefreeprov",
                "type": "chat",
                "access": "free",
                "status": "active",
            },
            {
                "id": "fakefreeprov@fake-free-2",
                "provider": "fakefreeprov",
                "type": "chat",
                "access": "free",
                "status": "active",
            },
        ]

    monkeypatch.setattr(
        routing,
        "Catalog",
        lambda: type("C", (), {"list_resolved": staticmethod(_two_catalog_models)})(),
    )
    routing._QuotaCooldown._parked = {}
    routing._QuotaCooldown._dead_parks = {}
    _FakeFreeProvider.dead_first = True
    try:
        r = _post_chat(app_client, "auto@free")
        assert r.status_code == 200, r.text
        assert r.json()["model"] == "fakefreeprov@fake-free-2"
        snap = routing._QuotaCooldown.snapshot()
        assert "fakefreeprov@fake-free-1" in snap
        assert snap["fakefreeprov@fake-free-1"] > 3000  # ~1h, nicht 15 min
    finally:
        _FakeFreeProvider.dead_first = False
        routing._QuotaCooldown._parked = {}
        routing._QuotaCooldown._dead_parks = {}


def test_health_models_endpoint_and_listing_block(app_client, monkeypatch, tmp_path):
    """Health-Telemetrie E2E: nach echten Requests (fake provider) liefert
    /v1/health/models die gerankte Index-Zeile, und /v1/models traegt den
    kompakten health-Block am Katalog-Eintrag."""
    from uniinfer.proxy_services import stats as stats_mod
    from uniinfer.proxy_services.stats import StatsCollector

    monkeypatch.setattr(stats_mod, "_SNAPSHOT_PATH", str(tmp_path / "stats.json"))
    StatsCollector._instance = None
    try:
        # /v1/models listet aus dem models_router-eigenen Catalog-Import — für
        # den listing-Teil auf den Fake-Katalog zeigen (die Routing-Fixture
        # patcht nur das routing-Modul).
        import uniinfer.proxy_routers.models as models_router

        def _one_fake_model():
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
            models_router,
            "Catalog",
            lambda: type(
                "C", (), {"list_resolved": staticmethod(_one_fake_model)}
            )(),
        )
        for _ in range(3):
            r = _post_chat(app_client, "fakefreeprov@fake-free-1")
            assert r.status_code == 200, r.text

        r = app_client.get("/v1/health/models")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "list"
        row = next(x for x in body["data"] if x["model"] == "fakefreeprov@fake-free-1")
        assert row["n_observed"] == 3
        assert row["trust_tier"] == "cold"  # 3 < 20
        assert row["insufficient_data"] is True
        assert 0 < row["score"] < 1

        r = app_client.get("/v1/models")
        assert r.status_code == 200
        entry = next(
            m for m in r.json()["data"] if m["id"] == "fakefreeprov@fake-free-1"
        )
        assert entry["health"]["verdict"] in ("A", "B", "C", "D")
        assert isinstance(entry["health"]["score"], float)
    finally:
        StatsCollector._instance = None


def test_explore_e2e_hold_visible_in_header_and_preview(app_client, monkeypatch):
    """Explore-Mode E2E: env an -> auto@free exploriert (least-served), der
    Hold haelt aufeinanderfolgende Requests auf demselben Ziel (Cache warm),
    Preview zeigt den Explore-Zustand."""
    monkeypatch.setenv("UNIINFER_AUTO_EXPLORE", "1")
    monkeypatch.setenv("UNIINFER_AUTO_EXPLORE_HOLD_S", "600")
    routing._ExploreHold.clear()
    r1 = _post_chat(app_client, "auto@free")
    assert r1.status_code == 200, r1.text
    first = r1.headers.get("x-routed-model")
    assert first  # irgendein free-Kandidat der Fake-Liste
    r2 = _post_chat(app_client, "auto@free")
    assert r2.headers.get("x-routed-model") == first  # sticky hold
    preview = app_client.get("/v1/routing/auto").json()
    assert preview["reason"] == "explore"
    assert preview["explore"]["held_s"] > 0
    assert preview["resolved"] == first
