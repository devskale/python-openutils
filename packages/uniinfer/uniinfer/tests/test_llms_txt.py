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
