
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from uniinfer.uniioai_proxy import app

client = TestClient(app)

# Mock the authentication to bypass token check
@pytest.fixture
def mock_auth():
    # Patch verify_provider_access which is used in chat_completions
    with patch("uniinfer.uniioai_proxy.verify_provider_access", return_value="mock-api-key"), \
         patch("uniinfer.uniioai_proxy.validate_proxy_token", return_value="mock-token"), \
         patch("uniinfer.uniioai_proxy.get_optional_proxy_token", return_value="mock-token"):
        yield

# Mock the chat completion to avoid actual API calls
@pytest.fixture
def mock_chat_completion():
    # Mock run_in_threadpool which wraps get_completion
    with patch("uniinfer.uniioai_proxy.run_in_threadpool") as mock_run:
        mock_run.return_value = "Mock response content" # Return string content
        yield mock_run

def test_request_size_limit_middleware(mock_auth):
    # Test request with large content-length
    # Create a large payload > 10MB
    large_content = "x" * (10 * 1024 * 1024 + 100)
    payload = {
        "model": "openai@gpt-3.5-turbo",
        "messages": [{"role": "user", "content": large_content}]
    }
    # We rely on TestClient to calculate Content-Length
    headers = {"Authorization": "Bearer mock-token"}
    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 413
    assert response.json() == {"detail": "Request too large"}

def test_valid_request_size(mock_auth, mock_chat_completion):
    # Test valid request size
    payload = {
        "model": "openai@gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}]
    }
    headers = {"Authorization": "Bearer mock-token"}
    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 200

def test_too_many_messages(mock_auth):
    # Test with 501 messages
    messages = [{"role": "user", "content": "msg"} for _ in range(501)]
    payload = {
        "model": "openai@gpt-3.5-turbo",
        "messages": messages
    }
    headers = {"Authorization": "Bearer mock-token"}
    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 422
    data = response.json()
    assert "Too many messages" in str(data)

def test_valid_message_count(mock_auth, mock_chat_completion):
    # Test with 500 messages (boundary condition)
    messages = [{"role": "user", "content": "msg"} for _ in range(500)]
    payload = {
        "model": "openai@gpt-3.5-turbo",
        "messages": messages
    }
    headers = {"Authorization": "Bearer mock-token"}
    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 200

def test_invalid_model_format(mock_auth):
    payload = {
        "model": "invalid_format",
        "messages": [{"role": "user", "content": "Hello"}]
    }
    headers = {"Authorization": "Bearer mock-token"}
    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    # 422 is expected from Pydantic, but if it passes to endpoint it might be 400
    assert response.status_code in [422, 400] 
    data = response.json()
    # Check for either error message
    assert "Invalid model format" in str(data) or "Invalid model format" in str(data.get("detail", ""))

def test_valid_model_format(mock_auth, mock_chat_completion):
    payload = {
        "model": "provider@model",
        "messages": [{"role": "user", "content": "Hello"}]
    }
    headers = {"Authorization": "Bearer mock-token"}
    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 200


