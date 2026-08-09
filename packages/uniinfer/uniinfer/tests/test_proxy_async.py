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

    result = format_chunk_to_openai(response, "openai@gpt-3.5-turbo", "chatcmpl-test")
    assert result["id"] == "chatcmpl-test"
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

    result = format_chunk_to_openai(response, "openai@gpt-3.5-turbo", "chatcmpl-test")
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

    result = format_chunk_to_openai(response, "openai@gpt-3.5-turbo", "chatcmpl-test")
    assert "tool_calls" in result["choices"][0]["delta"]


def test_streaming_generator_imports():
    from uniinfer.proxy_services.streaming import astream_response_generator
    assert callable(astream_response_generator)


def test_proxy_app_imports():
    from uniinfer.proxy_app import app
    assert app is not None


def test_models_helper():
    from uniinfer.provider_access import list_models_for_provider
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

    def test_lean_http_middleware_configured(self):
        """The pure-ASGI request-logging + size-limit middleware is in the stack.

        Replaces the former @app.middleware('http') middlewares (which were
        BaseHTTPMiddleware — leaks under streaming/SSE, Starlette #1012).
        """
        from uniinfer.proxy_app import app, LeanHTTPMiddleware
        has = any(m.cls is LeanHTTPMiddleware for m in app.user_middleware)
        assert has, "LeanHTTPMiddleware missing from the ASGI stack"

    def test_no_base_http_middleware_in_stack(self):
        """Regression guard: no BaseHTTPMiddleware subclass may be in the stack
        (it leaks RSS under concurrent SSE). Covers slowapi too."""
        from uniinfer.proxy_app import app
        from starlette.middleware.base import BaseHTTPMiddleware
        for m in app.user_middleware:
            assert not (isinstance(m.cls, type) and issubclass(m.cls, BaseHTTPMiddleware)), \
                f"{m.cls.__name__} is BaseHTTPMiddleware (leaks under SSE)"

    def test_cors_middleware_configured(self):
        from uniinfer.proxy_app import app
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
        from uniinfer.proxy_app import app
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
        response = client.get("/webdemo")
        assert response.status_code == 200
