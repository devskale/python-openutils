"""RC#5: Stream-Priming-Timeout tarnt sich als 200 — regression tests.

The incident (E_BERUFL_006 @ KONSTANT, 2026-08-17): TU vLLM needed >120s to
the FIRST token. The proxy's priming ``wait_for`` raised ``asyncio.TimeoutError``,
which was stored as ``_prime_error`` and re-raised AFTER the SSE 200 was
committed → generic ``internal_server_error`` error-chunk mid-200-stream
(``code: null``) → status-code-based client retries see 200 OK → no retry
anywhere → agentos silently degrades the doc to ``nicht_beurteilbar``.

Contract under test (fix design A, uniinfer proxy):
  1. A first-token (priming) timeout retries the upstream ONCE — the model is
     usually just thinking long. The client is kept alive during the second
     priming window via SSE comment heartbeats.
  2. A second priming timeout fails LOUD and RECOGNIZABLE: an error chunk with
     ``type: "stream_timeout"`` and ``code: 504`` (not the generic
     ``internal_server_error``/``code: null`` disguise).
  3. An upstream open-time RateLimitError still raises BEFORE the SSE 200 is
     committed (real HTTP 429 transport — unchanged).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from uniinfer.core import ChatCompletionResponse, ChatMessage
from uniinfer.errors import RateLimitError
from uniinfer.proxy_services.streaming import astream_response_generator

HANG_S = 30.0  # "forever" relative to the tiny test heartbeat


def _raw_chunk(content: str | None = None, finish: str | None = None) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        message=ChatMessage(role="assistant", content=content),
        provider="tu",
        model="tu/test-model",
        usage={},
        raw_response={},
        finish_reason=finish,
    )


class _CountingGen:
    """Wraps the fake upstream generator to count early aclose() calls."""

    def __init__(self, g, target):
        self._g, self._t = g, target

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._g.__anext__()

    async def aclose(self):
        self._t.closed_iters += 1
        await self._g.aclose()


class _FakeTarget:
    """Upstream stand-in. ``behaviors`` is a list of call scripts, one per
    astream_complete call: 'hang' (never first token), 'ok' (content + stop),
    or an exception instance to raise on first pull."""

    def __init__(self, behaviors: list[Any]):
        self.provider_model = "tu/test-model"
        self.behaviors = behaviors
        self.calls = 0
        self.closed_iters = 0

    def astream_complete(self, messages, **kw):
        behavior = self.behaviors[min(self.calls, len(self.behaviors) - 1)]
        self.calls += 1
        outer = self

        async def _gen():
            if isinstance(behavior, BaseException):
                raise behavior
            if behavior == "hang":
                await asyncio.sleep(HANG_S)
                yield _raw_chunk(content="too late")
                return
            if isinstance(behavior, str) and behavior.startswith("slow:"):
                await asyncio.sleep(float(behavior[5:]))
            yield _raw_chunk(content="hello ")
            yield _raw_chunk(content="world")
            yield _raw_chunk(finish="stop")

        return _CountingGen(_gen(), outer)

    async def aclose(self):
        pass


def _fast_heartbeats(monkeypatch) -> None:
    """Shrink the priming window + keepalive to test speed."""
    monkeypatch.setenv("UNIINFER_STREAM_HEARTBEAT", "0.3")
    monkeypatch.setenv("UNIINFER_STREAM_PRIME_KEEPALIVE", "0.05")


def _run(target) -> list[str]:
    async def drive() -> list[str]:
        gen = astream_response_generator(
            target=target,
            messages=[{"role": "user", "content": "hi"}],
            temp=0.0,
            max_tok=64,
        )
        return [ev async for ev in gen]

    return asyncio.run(drive())


def _data_events(raw: list[str]) -> list[dict]:
    return [
        json.loads(ev.removeprefix("data: ").strip())
        for ev in raw
        if ev.startswith("data: {")
    ]


def _error_chunk(raw: list[str]) -> dict | None:
    for d in _data_events(raw):
        if isinstance(d, dict) and "error" in d:
            return d["error"]
    return None


def _content(raw: list[str]) -> str:
    parts = []
    for d in _data_events(raw):
        for ch in d.get("choices", []):
            c = (ch.get("delta") or {}).get("content")
            if c:
                parts.append(c)
    return "".join(parts)


def test_prime_timeout_retries_once_and_recovers(monkeypatch):
    """Timeout on attempt 1 → ONE upstream retry → stream completes normally."""
    _fast_heartbeats(monkeypatch)
    target = _FakeTarget(["hang", "ok"])
    raw = _run(target)
    assert target.calls == 2, "priming timeout must retry the upstream exactly once"
    assert _content(raw) == "hello world"
    assert raw[-1] == "data: [DONE]\n\n"
    assert _error_chunk(raw) is None
    # the wedged first iterator was closed (no leaked upstream stream)
    assert target.closed_iters == 1


def test_prime_timeout_keeps_client_alive_with_comments(monkeypatch):
    """During the second priming window the client gets SSE comment heartbeats
    (``: keep-alive``) so its read timeout can't fire on the silent retry."""
    _fast_heartbeats(monkeypatch)
    # attempt 2 is slow (0.15s ≈ 3 keepalive periods) but eventually succeeds
    target = _FakeTarget(["hang", "slow:0.15"])
    raw = _run(target)
    comments = [ev for ev in raw if ev.startswith(":")]
    assert comments, "expected ': keep-alive' comments during the retry window"
    assert _content(raw) == "hello world"  # and the stream still completes
    assert _error_chunk(raw) is None


def test_prime_timeout_twice_is_loud_stream_timeout_504(monkeypatch):
    """Both priming attempts time out → recognizable stream_timeout/504 chunk
    (NOT the generic internal_server_error / code:null disguise)."""
    _fast_heartbeats(monkeypatch)
    target = _FakeTarget(["hang", "hang"])
    raw = _run(target)
    assert target.calls == 2
    err = _error_chunk(raw)
    assert err is not None, "expected an error chunk"
    assert err["type"] == "stream_timeout"
    assert err["code"] == 504
    assert "timeout" in err["message"].lower()
    assert raw[-1] == "data: [DONE]\n\n"


def test_prime_ratelimit_still_raises_before_commit(monkeypatch):
    """Open-time upstream 429 must still surface as a real RateLimitError
    raised from the generator (route turns it into HTTP 429) — no SSE bytes."""
    _fast_heartbeats(monkeypatch)
    target = _FakeTarget([RateLimitError("upstream quota exceeded")])

    async def first_pull():
        gen = astream_response_generator(
            target=target,
            messages=[{"role": "user", "content": "hi"}],
            temp=0.0,
            max_tok=64,
        )
        return await gen.__anext__()

    with pytest.raises(RateLimitError):
        asyncio.run(first_pull())


def test_prime_ratelimit_on_retry_is_chunk_not_silent(monkeypatch):
    """RateLimit raised only on the retried attempt (200 already committed) →
    error chunk carries the rate-limit shape, not the generic disguise."""
    _fast_heartbeats(monkeypatch)
    target = _FakeTarget(["hang", RateLimitError("upstream quota exceeded")])
    raw = _run(target)
    err = _error_chunk(raw)
    assert err is not None
    assert err["type"] in ("RateLimitError", "rate_limit")
    assert err["code"] == 429


def test_normal_stream_unaffected(monkeypatch):
    """No timeout anywhere → byte-identical behavior to before the fix."""
    _fast_heartbeats(monkeypatch)
    target = _FakeTarget(["ok"])
    raw = _run(target)
    assert target.calls == 1
    assert _content(raw) == "hello world"
    assert _error_chunk(raw) is None
    assert not [ev for ev in raw if ev.startswith(":")], "no comments without a retry window"
