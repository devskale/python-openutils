"""
TU provider: empty-content-as-success guard (silent rate-limit shadow).

TU Aqueduct throttles >25 req/min with 200 + empty content instead of a proper
429. Empty is only legitimate when thinking consumed the whole budget
(reasoning present) or the answer is a tool call. See issue
uniinfer-tu-empty-content-as-success.
"""
import json
from unittest.mock import MagicMock, AsyncMock, patch

import httpx
import pytest

from uniinfer import ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from uniinfer.errors import RateLimitError
from uniinfer.providers.tu import TUProvider


def _http_response(payload: dict, text: str | None = None) -> MagicMock:
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200
    mock.json.return_value = payload
    mock.text = text if text is not None else json.dumps(payload)
    mock.headers = {"content-type": "application/json"}
    return mock


def _mock_client(response: MagicMock) -> AsyncMock:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post.return_value = response
    client.is_closed = False
    return client


class TestTUEmptyContentAcomplete:
    """acomplete: 200 + empty content is a rate-limit shadow unless explained."""

    @pytest.fixture
    def provider(self):
        return TUProvider(api_key="test-key")

    def _request(self):
        return ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
        )

    @pytest.mark.parametrize("empty_content", [None, ""], ids=["null", "empty-string"])
    @pytest.mark.asyncio
    async def test_empty_content_without_reasoning_maps_to_429(self, provider, empty_content):
        """200 + content:null/"" and NO reasoning → RateLimitError with 429."""
        payload = {
            "choices": [
                {"message": {"role": "assistant", "content": empty_content}, "finish_reason": "stop"}
            ],
            "model": "test-model",
            "usage": {"total_tokens": 10},
        }
        provider._async_client = _mock_client(_http_response(payload))

        with pytest.raises(RateLimitError) as exc_info:
            await provider.acomplete(self._request())
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_empty_content_with_reasoning_is_success(self, provider):
        """200 + content:null WITH reasoning_content → legit thinking-empty, success."""
        payload = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": None, "reasoning_content": "thinking hard"},
                    "finish_reason": "stop",
                }
            ],
            "model": "test-model",
            "usage": {"total_tokens": 10},
        }
        provider._async_client = _mock_client(_http_response(payload))

        response = await provider.acomplete(self._request())

        assert isinstance(response, ChatCompletionResponse)
        assert response.message.content is None
        assert response.thinking == "thinking hard"

    @pytest.mark.asyncio
    async def test_empty_content_with_tool_calls_is_success(self, provider):
        """200 + content:null WITH tool_calls → tool answer, success."""
        tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        payload = {
            "choices": [
                {"message": {"role": "assistant", "content": None, "tool_calls": tool_calls}, "finish_reason": "tool_calls"}
            ],
            "model": "test-model",
            "usage": {"total_tokens": 10},
        }
        provider._async_client = _mock_client(_http_response(payload))

        response = await provider.acomplete(self._request())

        assert isinstance(response, ChatCompletionResponse)
        assert response.message.tool_calls == tool_calls
        assert response.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_normal_content_still_success(self, provider):
        """200 with real content → unchanged success (regression guard)."""
        payload = {
            "choices": [
                {"message": {"role": "assistant", "content": "Hi there!"}, "finish_reason": "stop"}
            ],
            "model": "test-model",
            "usage": {"total_tokens": 10},
        }
        provider._async_client = _mock_client(_http_response(payload))

        response = await provider.acomplete(self._request())

        assert response.message.content == "Hi there!"


class TestTUEmptyContentStream:
    """astream_complete: a completed stream with zero payload is an error, not success."""

    @pytest.fixture
    def provider(self):
        return TUProvider(api_key="test-key")

    def _request(self):
        return ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
        )

    def _stream_client(self, lines: list[str]) -> AsyncMock:
        async def mock_aiter_lines():
            for line in lines:
                yield line

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.stream.return_value.__aenter__.return_value = mock_response
        return mock_client

    async def _collect(self, provider, mock_client):
        with patch.object(provider, "_get_async_client", return_value=mock_client):
            return [chunk async for chunk in provider.astream_complete(self._request())]

    @pytest.mark.asyncio
    async def test_finish_reason_only_stream_yields_error_marker(self, provider):
        """Stream completes (finish_reason + [DONE]) but never carries payload → error marker."""
        lines = [
            'data: {"choices": [{"delta": {}, "finish_reason": "stop"}], "model": "test-model"}',
            'data: [DONE]',
        ]
        chunks = await self._collect(provider, self._stream_client(lines))

        assert chunks, "stream must yield at least the error marker"
        assert chunks[-1].finish_reason == "error"
        assert "rate-limit shadow" in chunks[-1].raw_response.get("error", "")

    @pytest.mark.asyncio
    async def test_reasoning_only_stream_is_success(self, provider):
        """Stream with only reasoning deltas (thinking-empty) stays a success."""
        lines = [
            'data: {"choices": [{"delta": {"role": "assistant", "reasoning_content": "hm"}}], "model": "test-model"}',
            'data: {"choices": [{"delta": {}, "finish_reason": "stop"}], "model": "test-model"}',
            'data: [DONE]',
        ]
        chunks = await self._collect(provider, self._stream_client(lines))

        assert all(c.finish_reason != "error" for c in chunks)
        assert chunks[0].thinking == "hm"

    @pytest.mark.asyncio
    async def test_content_stream_still_success(self, provider):
        """Normal content stream → unchanged success (regression guard)."""
        lines = [
            'data: {"choices": [{"delta": {"role": "assistant", "content": "Hi"}}], "model": "test-model"}',
            'data: {"choices": [{"delta": {"content": " there!"}}], "model": "test-model"}',
            'data: [DONE]',
        ]
        chunks = await self._collect(provider, self._stream_client(lines))

        assert "".join(c.message.content for c in chunks) == "Hi there!"
        assert all(c.finish_reason != "error" for c in chunks)
