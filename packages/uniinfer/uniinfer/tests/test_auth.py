import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from uniinfer.auth import validate_proxy_token, get_optional_proxy_token, verify_provider_access
from uniinfer.errors import AuthenticationError
from unittest.mock import patch

# Create a dummy FastAPI app for testing dependencies
app = FastAPI()

@app.get("/test-required")
def route_required(token: str = Depends(validate_proxy_token)):
    return {"token": token}

@app.get("/test-optional")
def route_optional(token: str = Depends(get_optional_proxy_token)):
    return {"token": token}

client = TestClient(app)

def test_validate_proxy_token_success():
    """Test validate_proxy_token with a valid bearer token."""
    response = client.get("/test-required", headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    assert response.json() == {"token": "test-token"}

def test_validate_proxy_token_missing():
    """Test validate_proxy_token with missing authorization."""
    response = client.get("/test-required")
    assert response.status_code == 401
    assert "Authentication required" in response.json()["detail"]

def test_get_optional_proxy_token_success():
    """Test get_optional_proxy_token with a valid bearer token."""
    response = client.get("/test-optional", headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    assert response.json() == {"token": "test-token"}

def test_get_optional_proxy_token_missing():
    """Test get_optional_proxy_token with missing authorization."""
    response = client.get("/test-optional")
    assert response.status_code == 200
    assert response.json() == {"token": None}

@patch("uniinfer.auth.get_provider_api_key")
def test_verify_provider_access_success(mock_get_key):
    """Test verify_provider_access returns the key on success."""
    mock_get_key.return_value = "real-api-key"
    result = verify_provider_access("test-token", "openai")
    assert result == "real-api-key"
    mock_get_key.assert_called_once_with("test-token", "openai")

@patch("uniinfer.auth.get_provider_api_key")
def test_verify_provider_access_ollama(mock_get_key):
    """Test verify_provider_access for ollama (key can be None)."""
    mock_get_key.return_value = None
    result = verify_provider_access("test-token", "ollama")
    assert result is None

@patch("uniinfer.auth.get_provider_api_key")
def test_verify_provider_access_failure(mock_get_key):
    """Test verify_provider_access raises 401 on failure."""
    from fastapi import HTTPException
    mock_get_key.side_effect = AuthenticationError("Invalid key")
    
    with pytest.raises(HTTPException) as excinfo:
        verify_provider_access("bad-token", "openai")
    assert excinfo.value.status_code == 401
    assert "Invalid key" in excinfo.value.detail
