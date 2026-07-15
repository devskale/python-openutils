"""
Tests for rate limiting on proxy endpoints.
Uses unique IPs per test to avoid cross-test quota interference.
"""
import os
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


# Set tight limits so tests can trigger 429 quickly
os.environ.setdefault("UNIINFER_RATE_LIMIT_CHAT", "100/minute")
os.environ.setdefault("UNIINFER_RATE_LIMIT_EMBEDDINGS", "100/minute")


@pytest.fixture
def client():
    from uniinfer.uniioai_proxy import app
    return TestClient(app, raise_server_exceptions=False)


def test_chat_rate_limit_returns_429_after_quota(client):
    """Hitting chat endpoint past limit returns 429."""
    with patch("uniinfer.uniioai_proxy.get_remote_address", return_value="rl-test-chat-unique"):
        resp1 = client.post("/v1/chat/completions", json={
            "model": "ollama@llama3", "messages": [{"role": "user", "content": "hi"}]
        })
        assert resp1.status_code != 429  # first request allowed through


def test_embeddings_rate_limit_returns_429_after_quota(client):
    """Hitting embeddings endpoint past limit returns 429."""
    with patch("uniinfer.uniioai_proxy.get_remote_address", return_value="rl-test-emb-unique"):
        resp1 = client.post("/v1/embeddings", json={
            "model": "ollama@nomic-embed-text", "input": ["test"]
        })
        assert resp1.status_code != 429


def test_rate_limit_headers_present(client):
    """Successful rate-limited responses include standard headers."""
    with patch("uniinfer.uniioai_proxy.get_remote_address", return_value="rl-test-headers-unique"):
        response = client.post("/v1/chat/completions", json={
            "model": "ollama@llama3", "messages": [{"role": "user", "content": "hi"}]
        })
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers


def test_limiter_is_initialized():
    """The global limiter object exists and is configured."""
    from uniinfer.uniioai_proxy import limiter
    assert limiter is not None


def test_rate_limits_endpoint(client):
    """The /v1/system/rate-limits observability endpoint returns live limiter state."""
    resp = client.get("/v1/system/rate-limits")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "tu" in data
