"""
TU provider: pooled-client eviction on read timeout (wedged-backend hardening).

A wedged TU backend accepts the request but never answers. The process-wide
pooled h2 connection pinned to it turns every request into a full read-timeout
hang, and a retry on the same client repeats it. On TimeoutException the pool
must be evicted so the retry (and all concurrent requests via the pool re-sync)
get a fresh connection / fresh load-balancer routing.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from uniinfer import ChatCompletionRequest, ChatMessage
from uniinfer.errors import ProviderError
from uniinfer.providers.tu import TUProvider, _TU_CLIENT_CACHE


@pytest.fixture(autouse=True)
def clean_pool():
    """Isolate the process-wide client cache per test."""
    _TU_CLIENT_CACHE.clear()
    yield
    _TU_CLIENT_CACHE.clear()


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    """Skip the retry backoff delays."""
    monkeypatch.setattr("uniinfer.providers.tu.asyncio.sleep", AsyncMock())


def _good_response(content: str = "ok") -> MagicMock:
    payload = {
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "model": "m",
        "usage": {"total_tokens": 2},
    }
    m = MagicMock(spec=httpx.Response)
    m.status_code = 200
    m.json.return_value = payload
    m.text = json.dumps(payload)
    return m


def _pool_client(post_return=None, post_side_effect=None) -> AsyncMock:
    c = AsyncMock(spec=httpx.AsyncClient)
    c.is_closed = False
    if post_side_effect is not None:
        c.post.side_effect = post_side_effect
    else:
        c.post.return_value = post_return
    return c


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content="hi")])


class TestTUPoolEvictionAcomplete:
    """Non-streaming: read timeout evicts the pool; retry lands on fresh client."""

    @pytest.mark.asyncio
    async def test_transport_error_retries_on_fresh_client(self, monkeypatch):
        """Wedged hangs often end as TransportError (LB kills the conn) — evict too."""
        provider = TUProvider(api_key="k")
        mock_a = _pool_client(post_side_effect=httpx.ReadError("connection killed by LB"))
        mock_b = _pool_client(post_return=_good_response("recovered"))
        _TU_CLIENT_CACHE[provider.base_url] = mock_a
        monkeypatch.setattr(TUProvider, "_new_async_client", lambda self: mock_b)

        response = await provider.acomplete(_request())

        assert response.message.content == "recovered"
        assert _TU_CLIENT_CACHE[provider.base_url] is mock_b



    @pytest.mark.asyncio
    async def test_read_timeout_retries_on_fresh_client(self, monkeypatch):
        provider = TUProvider(api_key="k")
        mock_a = _pool_client(post_side_effect=httpx.ReadTimeout("wedged backend"))
        mock_b = _pool_client(post_return=_good_response())
        _TU_CLIENT_CACHE[provider.base_url] = mock_a
        monkeypatch.setattr(TUProvider, "_new_async_client", lambda self: mock_b)

        response = await provider.acomplete(_request())

        assert response.message.content == "ok"
        assert mock_a.post.await_count == 1
        assert mock_b.post.await_count == 1
        assert _TU_CLIENT_CACHE[provider.base_url] is mock_b

    @pytest.mark.asyncio
    async def test_injected_client_is_not_evicted(self):
        provider = TUProvider(api_key="k")
        mock_injected = _pool_client(post_side_effect=httpx.ReadTimeout("wedged"))
        mock_pooled = _pool_client(post_return=_good_response("pooled"))
        _TU_CLIENT_CACHE[provider.base_url] = mock_pooled
        provider._async_client = mock_injected  # caller-injected (owns)

        with pytest.raises(ProviderError):
            await provider.acomplete(_request())

        # All retries stayed on the injected client; the pool was untouched.
        assert mock_injected.post.await_count == 5
        assert mock_pooled.post.await_count == 0
        assert _TU_CLIENT_CACHE[provider.base_url] is mock_pooled


class TestTUPoolResync:
    """_get_async_client re-syncs instances to an evicted replacement."""

    @pytest.mark.asyncio
    async def test_stale_instance_resyncs_to_replacement(self, monkeypatch):
        p1, p2 = TUProvider(api_key="k"), TUProvider(api_key="k")
        mock_a = _pool_client()
        mock_b = _pool_client()
        _TU_CLIENT_CACHE[p1.base_url] = mock_a

        assert await p1._get_async_client() is mock_a
        assert await p2._get_async_client() is mock_a

        monkeypatch.setattr(TUProvider, "_new_async_client", lambda self: mock_b)
        p1._replace_pooled_client()

        # The evicting instance binds the replacement…
        assert await p1._get_async_client() is mock_b
        # …and the other long-lived instance re-syncs instead of serving on the corpse.
        assert await p2._get_async_client() is mock_b
        assert _TU_CLIENT_CACHE[p1.base_url] is mock_b

    @pytest.mark.asyncio
    async def test_injected_client_is_not_resynced_away(self):
        provider = TUProvider(api_key="k")
        mock_injected = _pool_client()
        mock_pooled = _pool_client()
        _TU_CLIENT_CACHE[provider.base_url] = mock_pooled
        provider._async_client = mock_injected  # injected: caller owns lifecycle

        assert await provider._get_async_client() is mock_injected


class TestTUPoolEvictionStream:
    """Streaming: open-time read timeout evicts the pool; retry opens on fresh client."""

    @pytest.mark.asyncio
    async def test_stream_open_timeout_retries_on_fresh_client(self, monkeypatch):
        provider = TUProvider(api_key="k")
        mock_a = AsyncMock(spec=httpx.AsyncClient)
        mock_a.is_closed = False
        mock_a.stream = MagicMock(side_effect=httpx.ReadTimeout("wedged backend"))

        lines = [
            'data: {"choices": [{"delta": {"role": "assistant", "content": "ok"}}], "model": "m"}',
            'data: [DONE]',
        ]

        async def aiter_lines():
            for line in lines:
                yield line

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = aiter_lines

        mock_b = AsyncMock(spec=httpx.AsyncClient)
        mock_b.is_closed = False
        mock_b.stream = MagicMock()
        mock_b.stream.return_value.__aenter__.return_value = mock_response

        _TU_CLIENT_CACHE[provider.base_url] = mock_a
        monkeypatch.setattr(TUProvider, "_new_async_client", lambda self: mock_b)

        chunks = [c async for c in provider.astream_complete(_request())]

        assert mock_a.stream.call_count == 1
        assert mock_b.stream.call_count == 1
        assert _TU_CLIENT_CACHE[provider.base_url] is mock_b
        assert chunks and chunks[0].message.content == "ok"
