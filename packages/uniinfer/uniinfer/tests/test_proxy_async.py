"""
Tests for proxy async functionality, middleware, and endpoint existence.
"""
import pytest
from unittest.mock import patch, MagicMock


# --- Import smoke tests ---

def test_completion_target_imports():
    from uniinfer.completion import Target
    assert callable(Target)


def test_parse_provider_model_imports():
    from uniinfer.completion import parse_provider_model
    assert callable(parse_provider_model)


def test_format_chunk_to_openai_basic():
    from uniinfer.proxy_services.streaming import format_chunk_to_openai
    from uniinfer.core import ChatCompletionResponse, ChatMessage

    response = ChatCompletionResponse(
        message=ChatMessage(role="assistant", content="Hello world!"),
        provider="openai",
        model="openai@gpt-3.5-turbo",
        usage={"total_tokens": 10},
        raw_response={}
    )

    result = format_chunk_to_openai(response, "openai@gpt-3.5-turbo")
    assert result["object"] == "chat.completion.chunk"
    assert result["model"] == "openai@gpt-3.5-turbo"


def test_format_chunk_to_openai_with_finish_reason():
    from uniinfer.proxy_services.streaming import format_chunk_to_openai
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
    assert result["choices"][0]["finish_reason"] == "stop"


def test_format_chunk_to_openai_with_tool_calls():
    from uniinfer.proxy_services.streaming import format_chunk_to_openai
    from uniinfer.core import ChatCompletionResponse, ChatMessage

    tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "test_func", "arguments": '{"arg": "value"}'}}]
    response = ChatCompletionResponse(
        message=ChatMessage(role="assistant", content=None, tool_calls=tool_calls),
        provider="openai",
        model="openai@gpt-3.5-turbo",
        usage={"total_tokens": 15},
        raw_response={}
    )

    result = format_chunk_to_openai(response, "openai@gpt-3.5-turbo")
    assert "tool_calls" in result["choices"][0]["delta"]


def test_streaming_generator_imports():
    from uniinfer.proxy_services.streaming import astream_response_generator
    assert callable(astream_response_generator)


def test_proxy_app_imports():
    from uniinfer.uniioai_proxy import app
    assert app is not None


def test_rate_limit_helper():
    from uniinfer.uniioai_proxy import get_chat_rate_limit
    limit = get_chat_rate_limit()
    assert isinstance(limit, str)


def test_models_helper():
    from uniinfer.uniioai import list_models_for_provider
    assert callable(list_models_for_provider)


# --- Schema validation ---

def test_chat_message_input_validation():
    from pydantic import ValidationError
    from uniinfer.proxy_schemas.chat import ChatMessageInput

    msg = ChatMessageInput(role="user", content="Hello")
    assert msg.role == "user"

    with pytest.raises(ValidationError):
        ChatMessageInput(content="Test")


def test_chat_completion_request_input_validation():
    from pydantic import ValidationError
    from uniinfer.proxy_schemas.chat import ChatCompletionRequestInput, ChatMessageInput

    req = ChatCompletionRequestInput(
        model="openai@gpt-3.5-turbo",
        messages=[ChatMessageInput(role="user", content="Hello")],
        temperature=0.7
    )
    assert req.model == "openai@gpt-3.5-turbo"

    with pytest.raises(ValidationError):
        ChatCompletionRequestInput(
            model="invalid-model",
            messages=[ChatMessageInput(role="user", content="Test")]
        )


# --- Middleware ---

class TestProxyMiddleware:

    def test_request_size_middleware_exists(self):
        from uniinfer.uniioai_proxy import limit_request_size
        assert callable(limit_request_size)

    def test_log_requests_middleware_exists(self):
        from uniinfer.uniioai_proxy import log_requests
        assert callable(log_requests)

    def test_cors_middleware_configured(self):
        from uniinfer.uniioai_proxy import app
        has_cors = any(
            middleware.cls.__name__ == "CORSMiddleware"
            for middleware in app.user_middleware
        )
        assert has_cors


# --- Endpoint existence (via app routes) ---

class TestProxyEndpoints:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from uniinfer.uniioai_proxy import app
        return TestClient(app, raise_server_exceptions=False)

    def test_chat_completions_endpoint_exists(self, client):
        response = client.post("/v1/chat/completions", json={
            "model": "openai@gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "test"}]
        })
        assert response.status_code == 401

    def test_models_endpoint_exists(self, client):
        response = client.get("/v1/models")
        assert response.status_code == 200

    def test_root_endpoint_exists(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_webdemo_endpoint_exists(self, client):
        from uniinfer.uniioai_proxy import get_web_demo
        assert hasattr(get_web_demo, "__call__")
