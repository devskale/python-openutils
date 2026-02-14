"""
Simple tests for uniioai_proxy async functionality.
"""
import pytest
from unittest.mock import patch


def test_aget_completion_imports():
    """Test that async completion function imports correctly."""
    from uniinfer.uniioai import aget_completion
    assert callable(aget_completion)


def test_astream_completion_imports():
    """Test that async streaming function imports correctly."""
    from uniinfer.uniioai import astream_completion
    assert callable(astream_completion)


def test_format_chunk_to_openai_imports():
    """Test that format_chunk function imports correctly."""
    from uniinfer.uniioai import format_chunk_to_openai
    assert callable(format_chunk_to_openai)


def test_format_chunk_to_openai_basic():
    """Test basic format_chunk_to_openai functionality."""
    from uniinfer.uniioai import format_chunk_to_openai
    from uniinfer.core import ChatCompletionResponse, ChatMessage

    response = ChatCompletionResponse(
        message=ChatMessage(role="assistant", content="Hello world!"),
        provider="openai",
        model="openai@gpt-3.5-turbo",
        usage={"total_tokens": 10},
        raw_response={}
    )

    result = format_chunk_to_openai(response, "openai@gpt-3.5-turbo")

    assert "id" in result
    assert "object" in result
    assert result["object"] == "chat.completion.chunk"
    assert "model" in result
    assert result["model"] == "openai@gpt-3.5-turbo"
    assert "choices" in result
    assert len(result["choices"]) == 1
    assert "delta" in result["choices"][0]


def test_format_chunk_to_openai_with_finish_reason():
    """Test format_chunk with finish reason."""
    from uniinfer.uniioai import format_chunk_to_openai
    from uniinfer.core import ChatCompletionResponse, ChatMessage

    response = ChatCompletionResponse(
        message=ChatMessage(role="assistant", content="Test"),
        provider="openai",
        model="openai@gpt-3.5-turbo",
        usage={"total_tokens": 5},
        raw_response={},
        finish_reason="stop"
    )

    result = format_chunk_to_openai(response, "openai@gpt-3.5-turbo")

    assert "choices" in result
    assert "finish_reason" in result["choices"][0]
    assert result["choices"][0]["finish_reason"] == "stop"


def test_format_chunk_to_openai_with_tool_calls():
    """Test format_chunk with tool calls."""
    from uniinfer.uniioai import format_chunk_to_openai
    from uniinfer.core import ChatCompletionResponse, ChatMessage

    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "test_func",
                "arguments": '{"arg": "value"}'
            }
        }
    ]

    response = ChatCompletionResponse(
        message=ChatMessage(
            role="assistant",
            content=None,
            tool_calls=tool_calls
        ),
        provider="openai",
        model="openai@gpt-3.5-turbo",
        usage={"total_tokens": 15},
        raw_response={}
    )

    result = format_chunk_to_openai(response, "openai@gpt-3.5-turbo")

    assert "choices" in result
    assert "delta" in result["choices"][0]
    assert "tool_calls" in result["choices"][0]["delta"]
    assert len(result["choices"][0]["delta"]["tool_calls"]) == 1


def test_astream_response_generator_imports():
    """Test that async stream response generator imports correctly."""
    from uniinfer.uniioai_proxy import astream_response_generator
    assert callable(astream_response_generator)


def test_proxy_imports():
    """Test that proxy can be imported."""
    from uniinfer.uniioai_proxy import app
    assert app is not None


def test_proxy_has_async_endpoints():
    """Test that proxy has async support in imports."""

    # Check that async functions are available in uniioai
    from uniinfer.uniioai import aget_completion, astream_completion

    assert callable(aget_completion)
    assert callable(astream_completion)


def test_proxy_rate_limit_helper():
    """Test rate limit helper function."""
    from uniinfer.uniioai_proxy import get_chat_rate_limit

    limit = get_chat_rate_limit()
    assert limit is not None
    assert isinstance(limit, str)


def test_proxy_models_helper():
    """Test models list helper function."""
    from uniinfer.uniioai import list_models_for_provider

    assert callable(list_models_for_provider)


def test_get_refetch_interval_seconds_default(monkeypatch):
    """Test default REFETCHTIME interval."""
    from uniinfer.uniioai_proxy import get_refetch_interval_seconds
    monkeypatch.delenv("REFETCHTIME", raising=False)
    assert get_refetch_interval_seconds() == 24 * 3600


def test_get_refetch_interval_seconds_invalid(monkeypatch):
    """Test invalid REFETCHTIME interval fallback."""
    from uniinfer.uniioai_proxy import get_refetch_interval_seconds
    monkeypatch.setenv("REFETCHTIME", "abc")
    assert get_refetch_interval_seconds() == 24 * 3600


@pytest.mark.asyncio
async def test_ensure_fresh_models_file_refreshes_when_stale(monkeypatch):
    """Test stale models trigger refresh."""
    from uniinfer import uniioai_proxy as proxy
    calls = {"refresh": 0}

    monkeypatch.setattr(proxy, "models_file_is_stale", lambda: True)

    async def fake_refresh():
        calls["refresh"] += 1
        return {"status": "success"}

    monkeypatch.setattr(proxy, "refresh_models_file", fake_refresh)
    await proxy.ensure_fresh_models_file()
    assert calls["refresh"] == 1


@pytest.mark.asyncio
async def test_ensure_fresh_models_file_skips_when_fresh(monkeypatch):
    """Test fresh models do not trigger refresh."""
    from uniinfer import uniioai_proxy as proxy
    calls = {"refresh": 0}

    monkeypatch.setattr(proxy, "models_file_is_stale", lambda: False)

    async def fake_refresh():
        calls["refresh"] += 1
        return {"status": "success"}

    monkeypatch.setattr(proxy, "refresh_models_file", fake_refresh)
    await proxy.ensure_fresh_models_file()
    assert calls["refresh"] == 0


def test_chat_message_input_validation():
    """Test ChatMessageInput validation."""
    from pydantic import ValidationError
    from uniinfer.uniioai_proxy import ChatMessageInput

    # Valid message
    msg = ChatMessageInput(
        role="user",
        content="Hello"
    )
    assert msg.role == "user"
    assert msg.content == "Hello"

    # Missing required fields should raise error
    with pytest.raises(ValidationError):
        ChatMessageInput(
            content="Test"
        )


def test_chat_completion_request_input_validation():
    """Test ChatCompletionRequestInput validation."""
    from pydantic import ValidationError
    from uniinfer.uniioai_proxy import ChatCompletionRequestInput, ChatMessageInput

    # Valid request
    req = ChatCompletionRequestInput(
        model="openai@gpt-3.5-turbo",
        messages=[ChatMessageInput(role="user", content="Hello")],
        temperature=0.7
    )
    assert req.model == "openai@gpt-3.5-turbo"
    assert req.temperature == 0.7

    # Invalid model format should raise error
    with pytest.raises(ValidationError):
        ChatCompletionRequestInput(
            model="invalid-model",
            messages=[ChatMessageInput(role="user", content="Test")]
        )


class TestProxyAsyncIntegration:
    """Integration tests for proxy async functionality."""

    @patch('uniinfer.uniioai_proxy.verify_provider_access', return_value='test-key')
    @patch('uniinfer.uniioai.ProviderFactory.get_provider')
    def test_async_completes_non_streaming_integration(self, mock_get_provider, _mock_verify_provider_access):
        """Test async non-streaming integration."""
        from uniinfer.uniioai_proxy import app
        from uniinfer.core import ChatCompletionResponse, ChatMessage
        from fastapi.testclient import TestClient

        # Create mock provider
        mock_provider = mock_get_provider.return_value
        mock_response = ChatCompletionResponse(
            message=ChatMessage(role="assistant", content="Hello!"),
            provider="openai",
            model="openai@gpt-3.5-turbo",
            usage={"total_tokens": 10},
            raw_response={}
        )

        async def mock_acomplete(*args, **kwargs):
            return mock_response

        mock_provider.acomplete = mock_acomplete

        # Create test client
        client = TestClient(app)

        # Make request
        response = client.post("/v1/chat/completions", json={
            "model": "openai@gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Test"}],
            "stream": False
        }, headers={"Authorization": "Bearer test-token"})

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Hello!"

    @patch('uniinfer.uniioai_proxy.verify_provider_access', return_value='test-key')
    @patch('uniinfer.uniioai.ProviderFactory.get_provider')
    def test_async_completes_streaming_integration(self, mock_get_provider, _mock_verify_provider_access):
        """Test async streaming integration."""
        from uniinfer.uniioai_proxy import app
        from uniinfer.core import ChatCompletionResponse, ChatMessage
        from fastapi.testclient import TestClient

        # Create mock provider
        mock_provider = mock_get_provider.return_value
        mock_response = ChatCompletionResponse(
            message=ChatMessage(role="assistant", content="Hello"),
            provider="openai",
            model="openai@gpt-3.5-turbo",
            usage={"total_tokens": 10},
            raw_response={}
        )

        async def mock_astream(*args, **kwargs):
            yield mock_response

        mock_provider.astream_complete = mock_astream

        # Create test client
        client = TestClient(app)

        # Make streaming request
        response = client.post("/v1/chat/completions", json={
            "model": "openai@gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Test"}],
            "stream": True
        }, headers={"Authorization": "Bearer test-token"})

        assert response.status_code == 200

        # Check streaming response
        content = response.content
        assert b"data: " in content
        assert b"[DONE]" in content


class TestProxyMiddleware:
    """Test proxy middleware functionality."""

    def test_request_size_middleware_exists(self):
        """Test that request size middleware exists."""
        from uniinfer.uniioai_proxy import limit_request_size

        assert callable(limit_request_size)

    def test_log_requests_middleware_exists(self):
        """Test that request logging middleware exists."""
        from uniinfer.uniioai_proxy import log_requests

        assert callable(log_requests)

    def test_cors_middleware_configured(self):
        """Test that CORS middleware is configured."""
        from uniinfer.uniioai_proxy import app

        # Check if CORS middleware is added
        has_cors = any(
            middleware.cls.__name__ == "CORSMiddleware"
            for middleware in app.user_middleware
        )
        assert has_cors


class TestProxyEndpoints:
    """Test proxy endpoints."""

    def test_chat_completions_endpoint_exists(self):
        """Test that chat completions endpoint exists."""
        from uniinfer.uniioai_proxy import chat_completions

        assert hasattr(chat_completions, "__call__")

    def test_models_endpoint_exists(self):
        """Test that models endpoint exists."""
        from uniinfer.uniioai_proxy import list_models

        assert hasattr(list_models, "__call__")

    def test_system_info_endpoint_exists(self):
        """Test that system info endpoint exists."""
        from uniinfer.uniioai_proxy import get_system_info

        assert hasattr(get_system_info, "__call__")

    def test_webdemo_endpoint_exists(self):
        """Test that web demo endpoint exists."""
        from uniinfer.uniioai_proxy import get_web_demo

        assert hasattr(get_web_demo, "__call__")
