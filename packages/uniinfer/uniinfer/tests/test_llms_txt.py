"""Agent-discovery surfaces: /llms.txt, manifest links, Link headers.

llms.txt follows the llmstxt.org v2 proposal (H1 + blockquote + link
sections); the realtime-STT WS contract lives inline because OpenAPI cannot
describe WebSockets.
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from uniinfer.proxy_app import app

    return TestClient(app, raise_server_exceptions=False)


def test_llms_txt_served_as_markdown(client):
    r = client.get("/llms.txt")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/markdown")
    body = r.text
    assert body.startswith("# uniinfer\n")
    assert "> " in body.splitlines()[2]


def test_llms_txt_contains_ws_stt_contract(client):
    body = client.get("/llms.txt").text
    assert "/v1/audio/transcriptions/stream" in body
    assert "int16 PCM 16 kHz mono" in body
    assert "?token=" in body
    for code in ("4400", "4401", "4404", "4502"):
        assert code in body


def test_llms_txt_explains_model_id_format(client):
    body = client.get("/llms.txt").text
    assert "## Model ids" in body
    assert "alias@model" in body
    assert "rejected with 400" in body
    assert "first `@` binds the alias" in body


def test_llms_txt_documents_own_key_override(client):
    body = client.get("/llms.txt").text
    assert "X-Provider-Key" in body
    assert "no meta-bearer" in body


def test_manifest_model_id_note_and_own_key(client):
    manifest = client.get("/v1").json()
    assert "strict" in manifest["model_id_format_note"]
    assert "X-Provider-Key" in manifest["auth"]["x_provider_key"]


def test_providers_list_has_no_own_key_flag(client):
    """Own-key is universal since 2026-11-21 — a per-provider flag would be
    constant True and pure noise; it was removed from the listing."""
    data = client.get("/v1/providers").json()["data"]
    assert data
    for p in data:
        assert "accepts_own_key" not in p


def test_manifest_lists_realtime_stt_and_ws_auth_note(client):
    manifest = client.get("/v1").json()
    assert manifest["links"]["realtime_stt"] == "/v1/audio/transcriptions/stream"
    assert manifest["links"]["llms_txt"] == "/llms.txt"
    assert "?token=" in manifest["auth"]["websocket_note"]


def test_discovery_link_header_has_describedby(client):
    r = client.get("/v1")
    link = r.headers.get("link", "")
    assert 'rel="describedby"' in link
    assert "/llms.txt" in link


def test_contracts_md_served_and_linked(client):
    r = client.get("/contracts.md")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/markdown")
    assert "Agentic Contracts" in r.text
    assert "4401" in r.text  # WS close codes contract present
    body = client.get("/llms.txt").text
    assert "/contracts.md" in body
    manifest = client.get("/v1").json()
    assert manifest["links"]["contracts"] == "/contracts.md"
