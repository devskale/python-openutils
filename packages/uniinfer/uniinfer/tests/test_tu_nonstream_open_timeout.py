"""
TU provider: non-streaming open-timeout guard (wedged-replica fast-fail).

A wedged TU replica accepts a non-streaming POST but answers only after 45–600s
(production observation, 2026-09-29) while a fresh-routing attempt returns in
<1s. The provider must bound the wait with TU_NONSTREAM_OPEN_TIMEOUT instead
of the client's 300s read budget so the existing evict+retry machinery (see
test_tu_pool_eviction.py) can heal it. Eviction/retry itself is covered there;
this file pins the timeout contract and the error budget it reports.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from uniinfer import ChatCompletionRequest, ChatMessage
from uniinfer.errors import ProviderError
from uniinfer.providers.tu import (
    TU_NONSTREAM_OPEN_TIMEOUT,
    TUProvider,
    _TU_CLIENT_CACHE,
)


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


def _pool_client(post_side_effect=None) -> AsyncMock:
    c = AsyncMock(spec=httpx.AsyncClient)
    c.is_closed = False
    if post_side_effect is not None:
        c.post.side_effect = post_side_effect
    return c


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content="hi")])


class TestNonstreamOpenTimeout:
    """Non-streaming POSTs carry a bounded read timeout + honest error budget."""

    @pytest.mark.asyncio
    async def test_post_bounded_by_open_timeout(self):
        """The POST carries a per-request read timeout of TU_NONSTREAM_OPEN_TIMEOUT
        instead of the pooled client's 300s default — the precondition for the
        evict+retry path to fire at all on a wedged replica."""
        provider = TUProvider(api_key="k")
        mock = _pool_client(post_side_effect=[_good_response()])
        _TU_CLIENT_CACHE[provider.base_url] = (mock, asyncio.get_running_loop())

        await provider.acomplete(_request())

        timeout = mock.post.call_args.kwargs["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.read == TU_NONSTREAM_OPEN_TIMEOUT
        assert timeout.connect == 30.0

    @pytest.mark.asyncio
    async def test_exhausted_retries_report_nonstream_budget(self, monkeypatch):
        """Every attempt wedged → ProviderError names the non-streaming budget
        (not the stream-open one) so operators tune the right knob."""
        provider = TUProvider(api_key="k")
        monkeypatch.setenv("TU_TRANSPORT_RETRIES", "0")
        mock = _pool_client(post_side_effect=httpx.ReadTimeout("wedged replica"))
        _TU_CLIENT_CACHE[provider.base_url] = (mock, asyncio.get_running_loop())

        with pytest.raises(ProviderError) as exc:
            await provider.acomplete(_request())

        assert f"within {TU_NONSTREAM_OPEN_TIMEOUT:.0f}s" in str(exc.value)
        assert "wedged TU replica" in str(exc.value)

    @pytest.mark.asyncio
    async def test_timeout_evicts_pool_even_without_retry(self, monkeypatch):
        """Lean relay (TU_TRANSPORT_RETRIES=0): the request still fails fast,
        but the poisoned pooled client is evicted — pool hygiene, not a retry.
        The caller's own next attempt then lands on fresh routing."""
        from uniinfer.providers.tu import _TU_TELEMETRY

        provider = TUProvider(api_key="k")
        monkeypatch.setenv("TU_TRANSPORT_RETRIES", "0")
        mock_a = _pool_client(post_side_effect=httpx.ReadTimeout("wedged replica"))
        mock_b = AsyncMock(spec=httpx.AsyncClient)
        mock_b.is_closed = False
        _TU_CLIENT_CACHE[provider.base_url] = (mock_a, asyncio.get_running_loop())
        monkeypatch.setattr(TUProvider, "_new_async_client", lambda self: mock_b)
        _TU_TELEMETRY.evictions = 0
        _TU_TELEMETRY.transport_retries = 0

        with pytest.raises(ProviderError):
            await provider.acomplete(_request())

        assert mock_a.post.await_count == 1
        assert _TU_TELEMETRY.evictions == 1
        assert _TU_TELEMETRY.transport_retries == 0
        assert _TU_CLIENT_CACHE[provider.base_url][0] is mock_b
