"""Regression: non-streaming chat must not 400 on nested usage details.

Mistral (and OpenAI reasoning models) return ``prompt_tokens_details`` /
``completion_tokens_details`` as OBJECTS. The proxy schemas previously typed
``usage: dict[str, int]``, which rejected those objects and made the proxy 400
its own successful upstream result on non-streaming /v1/chat/completions.
"""
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


def test_schema_accepts_nested_usage_details():
    """NonStreamingChatCompletion must accept OpenAI-spec nested usage."""
    from uniinfer.proxy_schemas.chat import (
        ChatMessageOutput,
        NonStreamingChatCompletion,
        NonStreamingChoice,
    )

    r = NonStreamingChatCompletion(
        model="mistral@mistral-tiny-latest",
        choices=[NonStreamingChoice(message=ChatMessageOutput(role="assistant", content="OK"))],
        usage={
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    )
    # modelled + round-trips through model_dump
    assert r.usage.prompt_tokens_details.cached_tokens == 0
    assert r.usage.completion_tokens_details.reasoning_tokens == 0
    dumped = r.model_dump()
    assert dumped["usage"]["prompt_tokens_details"]["cached_tokens"] == 0


def test_schema_embeddings_accepts_nested_usage():
    from uniinfer.proxy_schemas.chat import EmbeddingData, EmbeddingResponse

    r = EmbeddingResponse(
        data=[EmbeddingData(embedding=[0.1, 0.2], index=0)],
        model="mistral@mistral-embed",
        usage={"prompt_tokens": 3, "total_tokens": 3, "prompt_tokens_details": {"cached_tokens": 1}},
    )
    assert r.usage.prompt_tokens_details.cached_tokens == 1


def test_router_nonstream_nested_usage_returns_200():
    """The full non-stream path: a Mistral-style nested-usage response must
    serialize to 200, not be rejected by Pydantic validation."""
    from fastapi.testclient import TestClient

    from uniinfer.proxy_app import app  # renamed from uniioai_proxy (candidate 4)

    mock_response = MagicMock()
    mock_response.message.content = "OK"
    mock_response.message.tool_calls = None
    mock_response.thinking = None
    mock_response.finish_reason = "stop"
    mock_response.usage = {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 7,
        "prompt_tokens_details": {"cached_tokens": 0},
    }

    with patch("uniinfer.proxy_routers.chat.verify_provider_access", return_value="k"), \
         patch("uniinfer.proxy_routers.chat.Target") as mt:
        mt.return_value.acomplete = AsyncMock(return_value=mock_response)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "mistral@mistral-tiny-latest",
                "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                "max_tokens": 16,
                "stream": False,
            },
        )
    assert resp.status_code == 200, resp.text
    usage = resp.json()["usage"]
    assert usage["prompt_tokens"] == 5
    assert usage["prompt_tokens_details"]["cached_tokens"] == 0


async def _fake_stream_with_usage(*args, **kwargs):
    """Mimic a provider that yields content chunks then a terminal usage chunk.

    Matches the vLLM/OpenAI streaming shape: usage arrives in a final chunk
    with choices:[] (or carried on the finish_reason chunk).
    """
    from uniinfer.core import ChatCompletionResponse, ChatMessage

    # Content chunk (no usage yet)
    yield ChatCompletionResponse(
        message=ChatMessage(role="assistant", content="OK"),
        provider="tu",
        model="tu@glm-test",
        usage={},
        raw_response={},
        finish_reason=None,
        thinking=None,
    )
    # Terminal usage-only chunk (choices:[] upstream)
    yield ChatCompletionResponse(
        message=ChatMessage(role="assistant", content=None),
        provider="tu",
        model="tu@glm-test",
        usage={"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
        raw_response={"usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11}},
        finish_reason=None,
        thinking=None,
    )


def test_router_stream_emits_terminal_usage_chunk():
    """Streaming must emit a terminal choices:[]+usage chunk when the client
    requested stream_options.include_usage and the backend supplied usage.

    Regression for the bug where streaming /v1/chat/completions never emitted
    usage, breaking context-% and token stats for streaming consumers.
    """
    from fastapi.testclient import TestClient

    from uniinfer.proxy_app import app

    mock_target = MagicMock()
    mock_target.provider_model = "tu@glm-test"
    mock_target.astream_complete = _fake_stream_with_usage

    with patch("uniinfer.proxy_routers.chat.verify_provider_access", return_value="k"), \
         patch("uniinfer.proxy_routers.chat.Target", return_value=mock_target):
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "tu@glm-test",
                "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                "max_tokens": 16,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )

    assert resp.status_code == 200, resp.text
    body = resp.text
    # A terminal usage chunk with choices:[] must be present.
    assert '"choices":[]' in body.replace(" ", ""), "no empty-choices usage chunk emitted"
    # And the usage values must be carried.
    assert '"prompt_tokens":9' in body.replace(" ", "")
    assert '"total_tokens":11' in body.replace(" ", "")


def test_router_stream_omits_usage_when_not_requested():
    """When include_usage is not requested, no terminal usage chunk is emitted."""
    from fastapi.testclient import TestClient

    from uniinfer.proxy_app import app

    mock_target = MagicMock()
    mock_target.provider_model = "tu@glm-test"
    mock_target.astream_complete = _fake_stream_with_usage

    with patch("uniinfer.proxy_routers.chat.verify_provider_access", return_value="k"), \
         patch("uniinfer.proxy_routers.chat.Target", return_value=mock_target):
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "tu@glm-test",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
                "stream": True,
            },
        )

    assert resp.status_code == 200, resp.text
    body = resp.text.replace(" ", "")
    assert '"choices":[]' not in body, "usage chunk emitted without include_usage"
