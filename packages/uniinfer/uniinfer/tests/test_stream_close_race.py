"""aclose() race on upstream stream iterators — regression tests.

The incident (amd uniioai-proxy journal, 2026-08-19 09:24): repeated

    asyncio - ERROR - Task exception was never retrieved
    future: <Task finished coro=<<async_generator_athrow ...>()>
            exception=RuntimeError('aclose(): asynchronous generator is already running')>

deepseek-v4-flash regularly exceeds the first-token window, so its streams
landed in the RC#5 priming-retry loop, which polls a background
``_pull = ensure_future(__anext__())`` while yielding ``: keep-alive``
comments. A client disconnect at that yield threw GeneratorExit into the SSE
generator, which unwound WITHOUT cancelling or awaiting ``_pull`` — the
upstream generator stayed "running", and its GC finalizer's aclose() then
crashed with the RuntimeError above. Two streams dying at once produced the
paired journal entries.

Contract under test:
  1. A disconnect during the priming keep-alive window cancels + awaits the
     in-flight pull (no leaked tasks) and closes the upstream iterator.
  2. A close landing while the generator is still unwinding (concurrent
     close) is absorbed, not raised.
  3. A close interrupted by outer cancellation still completes in the
     background instead of being abandoned half-run.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from uniinfer.core import ChatCompletionResponse, ChatMessage
from uniinfer.proxy_services.streaming import (
    _close_upstream_stream,
    astream_response_generator,
)

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
    """Wraps the fake upstream generator to count aclose() calls."""

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
    """Upstream stand-in. ``behaviors``: one script per astream_complete call,
    where 'hang' never produces a first token in test time."""

    def __init__(self, behaviors: list[Any]):
        self.provider_model = "tu/test-model"
        self.behaviors = behaviors
        self.calls = 0
        self.closed_iters = 0

    def astream_complete(self, messages, **kw):
        behavior = self.behaviors[min(self.calls, len(self.behaviors) - 1)]
        self.calls += 1

        async def _gen():
            if behavior == "hang":
                await asyncio.sleep(HANG_S)
                yield _raw_chunk(content="too late")
                return
            yield _raw_chunk(content="hello ")
            yield _raw_chunk(content="world")
            yield _raw_chunk(finish="stop")

        return _CountingGen(_gen(), self)

    async def aclose(self):
        pass


def _fast_heartbeats(monkeypatch) -> None:
    monkeypatch.setenv("UNIINFER_STREAM_HEARTBEAT", "0.3")
    monkeypatch.setenv("UNIINFER_STREAM_PRIME_KEEPALIVE", "0.05")


async def _collect_events(gen) -> list[str]:
    return [ev async for ev in gen]


def test_disconnect_during_prime_keepalive_leaks_nothing(monkeypatch):
    """Client disconnect at the keep-alive yield: the in-flight ``_pull`` is
    cancelled + awaited (old code leaked it for HANG_S), both upstream
    iterators are closed, and gen.aclose() itself completes cleanly."""
    _fast_heartbeats(monkeypatch)

    async def scenario() -> list[str]:
        target = _FakeTarget(["hang", "hang"])
        gen = astream_response_generator(
            target=target,
            messages=[{"role": "user", "content": "hi"}],
            temp=0.0,
            max_tok=64,
        )
        events: list[str] = []
        # 1) welcome chunk (priming timed out on attempt 1)
        events.append(await gen.__anext__())
        assert events[0].startswith("data: {")
        # 2) drive into the keep-alive window of the retry, then disconnect
        while True:
            ev = await gen.__anext__()
            events.append(ev)
            if ev.startswith(":"):
                break
        await gen.aclose()  # the client disconnect
        # let any leaked background task surface before auditing
        await asyncio.sleep(0.2)
        leaked = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        assert leaked == [], f"leaked tasks after disconnect: {leaked}"
        assert target.closed_iters == 2, "both upstream iterators must be closed"
        return events

    events = asyncio.run(scenario())
    assert any(ev.startswith(":") for ev in events), "expected keep-alive comments"
    assert not any("hello" in ev for ev in events), "stream must not continue after disconnect"


def test_close_upstream_stream_absorbs_concurrent_close():
    """A second close landing while the first is still unwinding the
    generator's finally must be absorbed (RuntimeError), not raised — the
    winner finishes the job exactly once."""

    async def scenario():
        closed: list[bool] = []

        async def _gen():
            try:
                yield 1
            finally:
                await asyncio.sleep(0.05)
                closed.append(True)

        g = _gen().__aiter__()
        await g.__anext__()  # park it suspended at the yield
        winners = [asyncio.ensure_future(_close_upstream_stream(g)) for _ in range(3)]
        await asyncio.gather(*winners)  # must not raise RuntimeError
        assert closed == [True], "generator must unwind its finally exactly once"

    asyncio.run(scenario())


def test_close_upstream_stream_survives_outer_cancellation():
    """A close interrupted by outer cancellation must still complete in the
    background (an abandoned half-run athrow is what leaves the generator
    'running' for the GC finalizer to crash on)."""

    async def scenario():
        closed: list[bool] = []

        async def _gen():
            try:
                yield 1
            finally:
                await asyncio.sleep(0.1)
                closed.append(True)

        g = _gen().__aiter__()
        await g.__anext__()
        t = asyncio.ensure_future(_close_upstream_stream(g))
        await asyncio.sleep(0.02)  # close is inside the generator's finally now
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.2)  # background close finishes
        assert closed == [True], "shielded close must complete despite cancellation"
        leaked = [tk for tk in asyncio.all_tasks() if tk is not asyncio.current_task()]
        assert leaked == [], f"leaked tasks: {leaked}"

    asyncio.run(scenario())


def test_close_upstream_stream_none_is_noop():
    """Defensive: closing a missing/None iterator is a silent no-op."""

    async def scenario():
        await _close_upstream_stream(None)

    asyncio.run(scenario())


def test_stream_completes_normally_after_fix(monkeypatch):
    """The happy path is byte-identical: content, finish, [DONE]."""
    _fast_heartbeats(monkeypatch)
    target = _FakeTarget(["ok"])

    async def scenario() -> list[str]:
        gen = astream_response_generator(
            target=target,
            messages=[{"role": "user", "content": "hi"}],
            temp=0.0,
            max_tok=64,
        )
        return await _collect_events(gen)

    events = asyncio.run(scenario())
    content = []
    for ev in events:
        if ev.startswith("data: {"):
            d = json.loads(ev.removeprefix("data: ").strip())
            for ch in d.get("choices", []):
                c = (ch.get("delta") or {}).get("content")
                if c:
                    content.append(c)
    assert "".join(content) == "hello world"
    assert events[-1] == "data: [DONE]\n\n"
    assert target.calls == 1
