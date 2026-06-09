"""
Tests for proxy security: auth, rate limiting, embeddings.
"""
import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from uniinfer.uniioai_proxy import app
    return TestClient(app, raise_server_exceptions=False)


def test_chat_completions_requires_auth(client):
    response = client.post("/v1/chat/completions", json={
        "model": "openai@gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hello"}]
    })
    assert response.status_code == 401


def test_chat_completions_success(client):
    with patch("uniinfer.proxy_routers.chat.verify_provider_access", return_value="fake-key"), \
         patch("uniinfer.proxy_routers.chat.aget_completion") as mock_aget:
        mock_response = MagicMock()
        mock_response.message.content = "Hello!"
        mock_response.message.tool_calls = None
        mock_response.thinking = None
        mock_response.finish_reason = "stop"
        mock_response.usage = {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        mock_aget.return_value = mock_response

        response = client.post("/v1/chat/completions", json={
            "model": "openai@gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "hello"}]
        })

    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "openai@gpt-3.5-turbo"
    assert data["usage"]["total_tokens"] == 7


def test_rate_limiter_exists():
    from uniinfer.uniioai_proxy import limiter
    assert limiter is not None


def test_embeddings_requires_auth(client):
    response = client.post("/v1/embeddings", json={
        "model": "openai@text-embedding-3-small",
        "input": ["hello"]
    })
    assert response.status_code == 401


def test_embeddings_ollama_no_auth(client):
    with patch("uniinfer.proxy_routers.chat.get_embeddings") as mock_get:
        mock_get.return_value = {
            "embeddings": [[0.1, 0.2]],
            "usage": {"total_tokens": 1}
        }
        response = client.post("/v1/embeddings", json={
            "model": "ollama@mxbai-embed-large",
            "input": ["hello"]
        })
    assert response.status_code == 200
    assert response.json()["model"] == "ollama@mxbai-embed-large"
