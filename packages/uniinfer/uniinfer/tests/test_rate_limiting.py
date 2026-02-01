import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

# Set environment variables BEFORE importing the app to ensure defaults are overridden if they were read at module level
# (Though our implementation reads them at request time, it's good practice)
os.environ["UNIINFER_RATE_LIMIT_CHAT"] = "1/minute"
os.environ["UNIINFER_RATE_LIMIT_EMBEDDINGS"] = "2/minute"

from uniinfer.uniioai_proxy import app

client = TestClient(app)

def test_chat_rate_limiting():
    # Use a unique IP for this test to avoid interference
    client_ip = "127.0.0.1"
    headers = {"X-Forwarded-For": client_ip} # slowapi's get_remote_address might look at this or we rely on client host
    
    # TestClient sets client.host to "testclient" by default or similar.
    # We can pass client param to request? No, TestClient constructor takes base_url etc.
    # We can mock get_remote_address.
    
    with patch("uniinfer.uniioai_proxy.get_remote_address", return_value="1.2.3.4"):
        # First request should succeed (or fail with 401, but not 429)
        # We don't provide auth, so we expect 401, but rate limit is checked BEFORE auth?
        # Usually rate limit is checked first.
        # Let's hit an endpoint.
        
        # Request 1: Should be allowed (1/minute)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "ollama@llama3",
                "messages": [{"role": "user", "content": "hello"}]
            }
        )
        # We expect 200 or 401 (if auth failed) or 500.
        # Ideally we want to pass auth to make sure we hit the limit logic cleanly.
        # But rate limit should trigger regardless of auth result if it's a global limit per IP.
        # Wait, if 401 is returned, does it count towards rate limit?
        # Typically yes.
        assert response.status_code != 429
        
        # Request 2: Should fail (Limit is 1/minute)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "ollama@llama3",
                "messages": [{"role": "user", "content": "hello"}]
            }
        )
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.text

def test_embeddings_rate_limiting():
    with patch("uniinfer.uniioai_proxy.get_remote_address", return_value="5.6.7.8"):
        # Limit is 2/minute
        
        # Req 1
        response = client.post(
            "/v1/embeddings",
            json={"model": "ollama@nomic-embed-text", "input": ["test"]}
        )
        assert response.status_code != 429
        
        # Req 2
        response = client.post(
            "/v1/embeddings",
            json={"model": "ollama@nomic-embed-text", "input": ["test"]}
        )
        assert response.status_code != 429
        
        # Req 3: Should fail
        response = client.post(
            "/v1/embeddings",
            json={"model": "ollama@nomic-embed-text", "input": ["test"]}
        )
        assert response.status_code == 429

def test_rate_limit_headers():
    with patch("uniinfer.uniioai_proxy.get_remote_address", return_value="9.9.9.9"):
        # We set headers_enabled=True in the app
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "ollama@llama3",
                "messages": [{"role": "user", "content": "hello"}]
            }
        )
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers
