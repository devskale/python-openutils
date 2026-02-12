"""
Live tests for the uniioai-proxy.
Assumes the proxy is running on localhost:8000.
"""
import pytest
import httpx
import json

BASE_URL = "http://localhost:8123/v1"
HEADERS = {"Authorization": "Bearer test23@test34"}

@pytest.mark.asyncio
async def test_live_proxy_models():
    """Test that the proxy returns a list of models."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/models", headers=HEADERS, timeout=5.0)
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert len(data["data"]) > 0
        except httpx.ConnectError:
            pytest.skip("Proxy is not running on localhost:8000")

@pytest.mark.asyncio
async def test_live_proxy_chat_completion():
    """Test a live chat completion via the proxy."""
    async with httpx.AsyncClient() as client:
        payload = {
            "model": "pollinations@openai",
            "messages": [{"role": "user", "content": "Say 'PROXY_TEST_SUCCESS'"}],
            "stream": False
        }
        try:
            response = await client.post(
                f"{BASE_URL}/chat/completions",
                headers=HEADERS,
                json=payload,
                timeout=30.0
            )
            assert response.status_code == 200
            data = response.json()
            assert "choices" in data
            content = data["choices"][0]["message"]["content"]
            assert "PROXY_TEST_SUCCESS" in content
        except httpx.ConnectError:
            pytest.skip("Proxy is not running on localhost:8000")

@pytest.mark.asyncio
async def test_live_proxy_chat_streaming():
    """Test live chat streaming via the proxy."""
    async with httpx.AsyncClient() as client:
        payload = {
            "model": "pollinations@openai",
            "messages": [{"role": "user", "content": "Count from 1 to 3"}],
            "stream": True
        }
        try:
            async with client.stream(
                "POST",
                f"{BASE_URL}/chat/completions",
                headers=HEADERS,
                json=payload,
                timeout=30.0
            ) as response:
                assert response.status_code == 200
                
                full_content = ""
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        data = json.loads(data_str)
                        if data["choices"][0]["delta"].get("content"):
                            full_content += data["choices"][0]["delta"]["content"]
                
                assert len(full_content) > 0
        except httpx.ConnectError:
            pytest.skip("Proxy is not running on localhost:8000")
