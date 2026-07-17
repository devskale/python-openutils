"""
Tests for rate limiting on proxy endpoints.
Uses unique IPs per test to avoid cross-test quota interference.
"""
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


# Set tight limits so tests can trigger 429 quickly
os.environ.setdefault("UNIINFER_RATE_LIMIT_CHAT", "100/minute")
os.environ.setdefault("UNIINFER_RATE_LIMIT_EMBEDDINGS", "100/minute")


@pytest.fixture
def client():
    from uniinfer.proxy_app import app
    return TestClient(app, raise_server_exceptions=False)


def _ok_response():
    """A mocked successful ChatCompletionResponse-shaped object."""
    r = MagicMock()
    r.message.content = "ok"
    r.message.tool_calls = None
    r.thinking = None
    r.finish_reason = "stop"
    r.usage = None
    return r


# Obviously-fake model sentinels. These tests mock the provider, so the model
# is only a parse target — no real / environment-specific model is encoded. If
# a live model is ever needed, take it from config/env, not from here.
_CHAT_MODEL = "ollama@unit-test-model"
_EMBED_MODEL = "ollama@unit-test-embed"


def test_chat_rate_limit_returns_429_after_quota(client):
    """Hitting chat endpoint under quota is allowed (not 429)."""
    with patch("uniinfer.proxy_routers.chat.Target") as mock_target_cls, \
         patch("uniinfer.proxy_app.get_remote_address", return_value="rl-test-chat-unique"):
        mock_target_cls.return_value.acomplete = AsyncMock(return_value=_ok_response())
        resp1 = client.post("/v1/chat/completions", json={
            "model": _CHAT_MODEL, "messages": [{"role": "user", "content": "hi"}]
        })
        assert resp1.status_code != 429  # first request allowed through


def test_embeddings_rate_limit_returns_429_after_quota(client):
    """Hitting embeddings endpoint under quota is allowed (not 429)."""
    with patch("uniinfer.proxy_routers.chat.get_embeddings") as mock_embed, \
         patch("credgoo.get_api_key", return_value="fake-ollama-key"), \
         patch("uniinfer.proxy_app.get_remote_address", return_value="rl-test-emb-unique"):
        mock_embed.return_value = {"embeddings": [[0.1]], "usage": {"prompt_tokens": 1, "total_tokens": 1}}
        resp1 = client.post("/v1/embeddings", json={
            "model": _EMBED_MODEL, "input": ["test"]
        })
        assert resp1.status_code != 429


def test_rate_limit_headers_present(client):
    """Successful rate-limited responses include standard headers.

    Mocks the completion path so the request succeeds in-process — this tests
    slowapi header attachment, not ollama connectivity (amp's ollama sits behind
    nginx requiring a bearer, and its model set is ephemeral). The model is a
    stable sentinel, never a real one — models change constantly and can't be
    pinned in a test.
    """
    with patch("uniinfer.proxy_routers.chat.Target") as mock_target_cls, \
         patch("uniinfer.proxy_app.get_remote_address", return_value="rl-test-headers-unique"):
        mock_target_cls.return_value.acomplete = AsyncMock(return_value=_ok_response())
        response = client.post("/v1/chat/completions", json={
            "model": _CHAT_MODEL, "messages": [{"role": "user", "content": "hi"}]
        })
        assert response.status_code == 200, response.text
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers


def test_limiter_is_initialized():
    """The global limiter object exists and is configured."""
    from uniinfer.proxy_app import limiter
    assert limiter is not None


def test_rate_limits_endpoint(client):
    """The /v1/system/rate-limits observability endpoint returns live limiter state."""
    resp = client.get("/v1/system/rate-limits")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "tu" in data
