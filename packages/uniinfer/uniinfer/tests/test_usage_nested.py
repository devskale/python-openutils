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
