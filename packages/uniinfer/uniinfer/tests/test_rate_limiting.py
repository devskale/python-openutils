"""Tests for the /v1/system/rate-limits observability endpoint.

slowapi per-IP rate limiting was removed (its ASGI middleware re-sent
http.response.start on every body chunk, corrupting multi-chunk responses like
the webdemo's FileResponse; and it never fired). The TU adaptive limiter was
also removed (429s relayed transparently). What remains is this read-only
status endpoint, populated by any provider limiters still in use.
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from uniinfer.proxy_app import app
    return TestClient(app, raise_server_exceptions=False)


def test_rate_limits_endpoint(client):
    """The /v1/system/rate-limits observability endpoint returns structured state."""
    resp = client.get("/v1/system/rate-limits")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
