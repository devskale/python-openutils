"""
TU provider: idle-gap stream staleness + wedge telemetry.

A wedged TU backend accepts a stream but never sends data. The provider must
(1) fail that case fast — bounded by TU_STREAM_GAP_TIMEOUT instead of the 300s
httpx read budget — (2) replay the whole stream on a fresh connection while no
chunk has reached the caller (safe: nothing to duplicate), and (3) expose how
much is wedged/hanging via TUTelemetry.snapshot() so /health can report it.
"""
import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from uniinfer import ChatCompletionRequest, ChatMessage
from uniinfer.errors import ProviderError
from uniinfer.providers.tu import (
    TUProvider,
    TUTelemetry,
    _TU_CLIENT_CACHE,
    _TU_TELEMETRY,
)


@pytest.fixture(autouse=True)
def clean_state():
    """Isolate the process-wide pool + telemetry counters per test."""
    _TU_CLIENT_CACHE.clear()
    for k in ("evictions", "transport_retries", "open_stalls", "body_stalls",
              "stall_retries", "rate_limits"):
        setattr(_TU_TELEMETRY, k, 0)
    _TU_TELEMETRY.active_streams.clear()
    yield
    _TU_CLIENT_CACHE.clear()


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    monkeypatch.setattr("uniinfer.providers.tu.asyncio.sleep", AsyncMock())


@pytest.fixture(autouse=True)
def tiny_gap(monkeypatch):
    """Drive the idle-gap logic without waiting 60s."""
    monkeypatch.setattr("uniinfer.providers.tu.TU_STREAM_GAP_TIMEOUT", 0.05)


def _request(model: str = "deepseek-v4-flash-284b") -> ChatCompletionRequest:
    return ChatCompletionRequest(model=model, messages=[ChatMessage(role="user", content="hi")])


def _stream_response(*lines) -> AsyncMock:
    """AsyncMock httpx response whose aiter_lines yields the given lines."""
    async def gen():
        for ln in lines:
            yield ln
    r = AsyncMock()
    r.status_code = 200
    r.aiter_lines = gen
    return r


def _cm(response) -> object:
    """A context manager whose __aenter__ returns the given httpx response."""
    class CM:
        def __init__(self, resp):
            self._resp = resp
        async def __aenter__(self):
            return self._resp
        async def __aexit__(self, *args):
            return None
    return CM(response)


def _client(*responses) -> AsyncMock:
    """AsyncMock client whose stream() returns each response in turn — a retry
    consumes a fresh :meth:`stream` call, like a fresh-connection replay."""
    c = AsyncMock(spec=httpx.AsyncClient)
    c.is_closed = False
    c.stream.side_effect = [_cm(r) for r in responses]
    return c


def test_snapshot_reports_stuck_streams():
    """A stream idle past the gap shows as in_flight + stuck in the snapshot."""
    tel = TUTelemetry()
    sid = tel.register_stream("m")
    tel.touch(sid)
    snap = tel.snapshot(stalled_after_s=60.0)
    assert snap["in_flight"] == 1
    assert snap["stuck_streams"] == 0
    # simulate a wedged stream: last activity in the past
    tel.active_streams[sid]["last"] -= 999
    snap2 = tel.snapshot(stalled_after_s=60.0)
    assert snap2["stuck_streams"] == 1
    assert snap2["streams"][0]["idle_s"] >= 999


@pytest.mark.asyncio
async def test_stream_stall_before_first_chunk_replays_fresh(monkeypatch):
    """A stream that wedges before any chunk is replayed on a fresh connection;
    telemetry records the stall + retry + eviction. No data reaches the caller
    from the first (wedged) attempt, so the replay is duplication-free."""
    provider = TUProvider(api_key="k")

    async def wedged_gen():
        await asyncio.Event().wait()  # block forever — the 0.05s gap fires first
        yield 'data: {"choices": [{"delta": {"role": "assistant", "content": "x"}}], "model": "m"}\n'

    wedged = AsyncMock()
    wedged.status_code = 200
    wedged.aiter_lines = wedged_gen

    ok = _stream_response(
        'data: {"choices": [{"delta": {"role": "assistant", "content": "recovered"}}], "model": "m"}\n',
        'data: {"choices": [{"delta": {"finish_reason": "stop"}}], "model": "m"}\n',
        "data: [DONE]\n",
    )

    mock_a = _client(wedged)
    mock_b = _client(ok)
    _TU_CLIENT_CACHE[provider.base_url] = mock_a
    monkeypatch.setattr(TUProvider, "_new_async_client", lambda self: mock_b)

    chunks = []
    async for ch in provider.astream_complete(_request("m")):
        chunks.append(ch)

    assert [ch.message.content for ch in chunks if ch.message.content] == ["recovered"]
    assert _TU_TELEMETRY.body_stalls == 1
    assert _TU_TELEMETRY.stall_retries == 1
    assert _TU_TELEMETRY.evictions == 1  # pooled client evicted before the replay
    assert _TU_CLIENT_CACHE[provider.base_url] is mock_b


@pytest.mark.asyncio
async def test_stream_stall_after_chunk_is_a_hard_error(monkeypatch):
    """Once data has reached the caller it can't be replayed — a stall there
    surfaces as ProviderError (no silent partial-duplicate) and clears the
    wedged connection for concurrent requesters."""
    provider = TUProvider(api_key="k")

    async def wedged_gen():
        yield 'data: {"choices": [{"delta": {"role": "assistant", "content": "partial"}}], "model": "m"}\n'
        await asyncio.Event().wait()  # stall on the SECOND read — gap fires

    wedged = AsyncMock()
    wedged.status_code = 200
    wedged.aiter_lines = wedged_gen

    mock_a = _client(wedged)
    _TU_CLIENT_CACHE[provider.base_url] = mock_a
    monkeypatch.setattr(TUProvider, "_new_async_client", lambda self: _client())

    with pytest.raises(ProviderError):
        async for _ch in provider.astream_complete(_request("m")):
            pass

    assert _TU_TELEMETRY.body_stalls == 1
    assert _TU_TELEMETRY.stall_retries == 0  # no replay once data was emitted
    assert _TU_TELEMETRY.evictions == 1


@pytest.mark.asyncio
async def test_stream_stall_out_of_retries_is_a_hard_error(monkeypatch):
    """Even pre-first-chunk, an exhausted retry budget is an error, not a hang."""
    provider = TUProvider(api_key="k")

    async def wedged_gen():
        await asyncio.Event().wait()  # never emits — wedged from the start
        yield ''  # unreachable; keeps this an async generator (not a coroutine)

    wedged = AsyncMock()
    wedged.status_code = 200
    wedged.aiter_lines = wedged_gen

    mock_a = _client(wedged)
    _TU_CLIENT_CACHE[provider.base_url] = mock_a
    monkeypatch.setattr(TUProvider, "_new_async_client", lambda self: _client(wedged))

    with pytest.raises(ProviderError):
        async for _ch in provider.astream_complete(_request("m")):
            pass

    assert _TU_TELEMETRY.body_stalls > 0
    assert _TU_TELEMETRY.stall_retries <= 4


@pytest.mark.asyncio
async def test_clear_wedge_state_drops_pool_and_streams(monkeypatch):
    """clear_wedge_state() closes every pooled client and empties the active-
    stream registry — the operator action behind /debug/wedge/clear."""
    from uniinfer.providers.tu import clear_wedge_state

    provider = TUProvider(api_key="k")
    mock_a = _client()
    mock_b = _client()
    _TU_CLIENT_CACHE[provider.base_url] = mock_a
    # a second TU base_url (e.g. staging) as an independent pooled client
    _TU_CLIENT_CACHE["https://other.example/v1"] = mock_b

    sid = _TU_TELEMETRY.register_stream("deepseek-v4-flash-284b")
    _TU_TELEMETRY.touch(sid)
    _TU_TELEMETRY.active_streams[sid]["last"] -= 999  # stuck

    result = await clear_wedge_state()

    assert result["closed_clients"] == 2
    assert result["cleared_streams"] == 1
    assert mock_a.aclose.await_count == 1
    assert mock_b.aclose.await_count == 1
    assert _TU_CLIENT_CACHE == {}  # pool dropped → next request mints fresh
    assert _TU_TELEMETRY.active_streams == {}
    assert _TU_TELEMETRY.snapshot(60.0)["in_flight"] == 0
