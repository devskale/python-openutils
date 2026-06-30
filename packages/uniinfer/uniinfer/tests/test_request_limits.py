"""
Tests for request validation: size limits, message count, model format.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def _mock_completion():
    mock = MagicMock()
    mock.message.content = "Hello"
    mock.message.tool_calls = None
    mock.thinking = None
    mock.finish_reason = "stop"
    return mock


@pytest.fixture
def client():
    from uniinfer.uniioai_proxy import app
    return TestClient(app, raise_server_exceptions=False)


def test_request_size_limit(client):
    large = "x" * (10 * 1024 * 1024 + 100)
    response = client.post("/v1/chat/completions", json={
        "model": "openai@gpt-3.5-turbo",
        "messages": [{"role": "user", "content": large}]
    })
    assert response.status_code == 413


def test_too_many_messages(client):
    from uniinfer.proxy_schemas.chat import ChatCompletionRequestInput
    over = ChatCompletionRequestInput.MAX_MESSAGES + 1
    response = client.post("/v1/chat/completions", json={
        "model": "openai@gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "m"}] * over
    })
    assert response.status_code == 422
    assert "Too many messages" in str(response.json())


def test_invalid_model_format(client):
    response = client.post("/v1/chat/completions", json={
        "model": "no_at_sign",
        "messages": [{"role": "user", "content": "hi"}]
    })
    assert response.status_code in [422, 400]
    assert "Invalid model format" in str(response.json())
