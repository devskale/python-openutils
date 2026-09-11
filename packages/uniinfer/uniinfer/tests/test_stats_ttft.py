"""TTFT telemetry: stats collector records time-to-first-token per model and
exposes averages; the streaming path measures the first UPSTREAM chunk."""

import json
import pytest

import uniinfer.proxy_services.stats as stats_mod
from uniinfer.proxy_services.stats import StatsCollector


@pytest.fixture
def fresh_stats(tmp_path, monkeypatch):
    """Isolated collector: tmp snapshot file, singleton reset."""
    monkeypatch.setattr(stats_mod, "_SNAPSHOT_PATH", str(tmp_path / "stats.json"))
    StatsCollector._instance = None
    yield stats_mod.get_stats()
    StatsCollector._instance = None


def test_record_ttft_updates_counters(fresh_stats):
    fresh_stats.record("tu@m", status=200, latency_ms=5000, usage=None, ttft_ms=1234)
    hour_bucket = next(iter(fresh_stats._hourly.values()))["tu@m"]
    assert hour_bucket["ttft_sum"] == 1234
    assert hour_bucket["ttft_n"] == 1
    assert fresh_stats._totals["tu@m"]["ttft_n"] == 1


def test_record_without_ttft_leaves_ttft_zero(fresh_stats):
    fresh_stats.record("tu@m", status=200, latency_ms=500, usage=None)
    hour_bucket = next(iter(fresh_stats._hourly.values()))["tu@m"]
    assert hour_bucket["ttft_n"] == 0
    assert hour_bucket["ttft_sum"] == 0


def test_avg_ttft_in_per_model(fresh_stats):
    fresh_stats.record("tu@m", status=200, latency_ms=1000, usage=None, ttft_ms=1000)
    fresh_stats.record("tu@m", status=200, latency_ms=2000, usage=None, ttft_ms=3000)
    entry = fresh_stats.get()["last_24h"]["per_model"][0]
    assert entry["avg_ttft_ms"] == 2000
    assert entry["avg_latency_ms"] == 1500


def test_merged_counters_schema_drift_safe():
    """Old snapshots (pre-TTFT) load without KeyError and get zeroed ttft keys."""
    c = stats_mod._merged_counters({"req": 5, "errors": 1})
    assert c["req"] == 5
    assert c["ttft_sum"] == 0
    assert c["ttft_n"] == 0


def test_snapshot_roundtrip_keeps_ttft(fresh_stats):
    fresh_stats.record("tu@m", status=200, latency_ms=100, usage=None, ttft_ms=777)
    fresh_stats.snapshot()
    data = json.load(open(stats_mod._SNAPSHOT_PATH))
    hour_key = next(iter(data["hourly"]))
    assert data["hourly"][hour_key]["tu@m"]["ttft_sum"] == 777
    StatsCollector._instance = None
    again = stats_mod.get_stats()
    assert again._totals["tu@m"]["ttft_n"] == 1


def test_stream_records_ttft(fresh_stats):
    """The streaming generator measures TTFT at the first upstream chunk."""
    import asyncio
    from unittest.mock import MagicMock
    from uniinfer import ChatMessage
    from uniinfer.core import ChatCompletionResponse
    from uniinfer.proxy_services.streaming import astream_response_generator

    def _chunk(text):
        return ChatCompletionResponse(
            message=ChatMessage(role="assistant", content=text),
            provider="tu", model="testmodel", usage={}, raw_response=None,
        )

    async def fake_stream(*a, **kw):
        await asyncio.sleep(0.005)  # measurable TTFT (>0.5ms rounding floor)
        yield _chunk("hello")
        yield _chunk(" world")

    target = MagicMock()
    target.provider_model = "tu@testmodel"
    target.astream_complete = MagicMock(side_effect=lambda *a, **k: fake_stream(*a, **k))

    async def aclose():
        return None

    target.aclose = aclose

    async def consume():
        gen = astream_response_generator(target=target, messages=[{"role": "user", "content": "hi"}], temp=0.5, max_tok=10)
        out = []
        async for item in gen:
            out.append(item)
        return out

    out = asyncio.run(consume())
    assert any("[DONE]" in s for s in out)
    entry = fresh_stats._totals["tu@testmodel"]
    assert entry["req"] == 1
    assert entry["ttft_n"] == 1
    assert entry["ttft_sum"] > 0


def test_stream_prime_timeout_records_no_ttft(fresh_stats):
    """Upstream never yields a first chunk → no TTFT recorded (not a TTFT sample)."""
    import asyncio
    from unittest.mock import MagicMock
    from uniinfer.proxy_services.streaming import astream_response_generator

    async def never(*a, **kw):
        await asyncio.sleep(3600)
        yield None

    async def aclose():
        return None

    target = MagicMock()
    target.provider_model = "tu@wedged"
    target.astream_complete = MagicMock(return_value=never())
    target.aclose = aclose

    async def consume():
        gen = astream_response_generator(
            target=target, messages=[{"role": "user", "content": "hi"}], temp=0.5, max_tok=10,
        )
        # short-circuit: pull only a few SSE items, then close the generator
        it = gen.__aiter__()
        for _ in range(3):
            try:
                await it.__anext__()
            except StopAsyncIteration:
                break
        await it.aclose()

    import os
    os.environ["UNIINFER_STREAM_HEARTBEAT"] = "0.2"
    os.environ["UNIINFER_STREAM_PRIME_KEEPALIVE"] = "0.1"
    try:
        asyncio.run(consume())
    finally:
        os.environ.pop("UNIINFER_STREAM_HEARTBEAT", None)
        os.environ.pop("UNIINFER_STREAM_PRIME_KEEPALIVE", None)

    entry = fresh_stats._totals.get("tu@wedged", {})
    assert entry.get("ttft_n", 0) == 0
    assert entry.get("req", 0) >= 1
