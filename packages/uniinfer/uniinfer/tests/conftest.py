"""Shared test harness — the deep module behind the whole suite.

One client shape, one live-server factory, spec-correct SSE builders, and the
dispatch test seam (use_plan/fake_plan). Local fixtures may override where a
test genuinely needs a different shape; duplication is the smell, not the
override.
"""
from __future__ import annotations

import asyncio
import socket
import threading
import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


# ── HTTP surface ────────────────────────────────────────────────────────


@pytest.fixture()
def client():
    """The one TestClient shape: app as-shipped, exceptions surfaced as
    status codes (router tests assert HTTP behavior, not tracebacks)."""
    from uniinfer.proxy_app import app

    return TestClient(app, raise_server_exceptions=False)


# ── Live server harness (real sockets, real SSE) ────────────────────────


@pytest.fixture()
def live_server():
    """Factory: spin ANY FastAPI app on a free port in a daemon uvicorn
    thread. Yields the base URL; shuts down via should_exit."""
    import uvicorn

    def _start(app):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        )
        threading.Thread(target=server.run, daemon=True).start()
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        assert server.started
        return f"http://127.0.0.1:{port}", server

    started: list = []

    def start(app):
        url, server = _start(app)
        started.append(server)
        return url

    yield start

    for server in started:
        server.should_exit = True


# ── SSE spec-correct mock data ──────────────────────────────────────────


def sse_lines(*payloads: str, done: bool = False) -> list[str]:
    """SSE lines per spec: every event terminated by a blank line.

    The blank separator is load-bearing — the canonical decoder (uniinfer.sse)
    dispatches on it. Hand-built mocks without it parse as ONE multi-line
    event (deploy-protect incident 2026-11-21).
    """
    lines: list[str] = []
    for p in payloads:
        lines.append(f"data: {p}")
        lines.append("")
    if done:
        lines += ["data: [DONE]", ""]
    return lines


@pytest.fixture()
def sse():
    return sse_lines


# ── Dispatch test seam ──────────────────────────────────────────────────


class FakeTarget:
    """Minimal Target stand-in for router-level tests: scripted non-stream
    response + a closable handle. Streaming stays with real providers."""

    def __init__(self, content="ok", usage=None, finish="stop", error=None):
        self.provider_model = "fake@fake"
        self.provider_name = "fake"
        self._content = content
        self._usage = usage or {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        self._finish = finish
        self._error = error
        self.closed = False

    async def acomplete(self, messages, **kw):
        if self._error:
            raise self._error
        m = MagicMock()
        m.content = self._content
        m.tool_calls = None
        r = MagicMock()
        r.message = m
        r.usage = self._usage
        r.finish_reason = self._finish
        r.thinking = None
        r.reasoning_details = None
        return r

    async def aclose(self):
        self.closed = True


def fake_plan(target=None, model="tu@d1-3b", auto_chain=None, routed=None):
    """A DispatchPlan with a FakeTarget — the object tests swap in via
    use_plan instead of patching dispatch internals (auth/Target attrs)."""
    from uniinfer.proxy_services.completion_dispatch import DispatchPlan

    return DispatchPlan(
        provider_name=model.split("@", 1)[0],
        provider_model=model,
        target=target or FakeTarget(),
        provider_api_key="fake-key",
        auto_chain=auto_chain,
        routed_model=routed,
    )


@pytest.fixture()
def use_plan(monkeypatch):
    """The dispatch seam: replace the WHOLE resolution (one stable name)
    with a plan-carrying fake. Router tests stay on the HTTP interface;
    the core's internals stay free to move."""

    def _use(plan):
        import uniinfer.proxy_services.completion_dispatch as cd

        monkeypatch.setattr(cd, "verify_provider_access", lambda *a, **k: "fake-key")
        monkeypatch.setattr(cd, "Target", lambda *a, **k: plan.target)

    return _use


# module-scoped variant for expensive apps (echo servers reused across tests)
@pytest.fixture(scope="module")
def live_server_mod():
    import uvicorn

    def _start(app):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        )
        threading.Thread(target=server.run, daemon=True).start()
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        assert server.started
        return f"http://127.0.0.1:{port}"

    return _start
