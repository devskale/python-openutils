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

from uniinfer.core import ChatCompletionResponse, ChatMessage, ChatProvider


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


# ── Scripted chat provider (candidate C: two adapters made the seam real) ──


class ScriptedChatProvider(ChatProvider):
    """Declarative fake for router/dispatch tests: script the behavior, not
    the class. Covers the two recurring shapes (fail-429-first, empty-first)
    plus the plain echo; genuinely new behaviors stay local subclasses.

        ScriptedChatProvider("prov", script=["429", "empty", "ok"])

    Script entries: "ok" (content), "empty" (content=None + stop),
    "429" (RateLimitError), "dead" (ProviderError unavailable). Past the
    script, the last entry repeats (or "ok").
    """

    PROVIDER_ID = "scripted"
    REQUIRES_API_KEY = False
    BASE_URL = ""
    # CLASS state: the factory re-instantiates per Target, so per-instance
    # counters would reset every request (the classic fake-provider trap).
    _STATE: dict[str, dict] = {}

    def __init__(self, provider_id: str, script: list[str] | None = None):
        self.PROVIDER_ID = provider_id
        ScriptedChatProvider._STATE.setdefault(
            provider_id, {"i": 0, "calls": 0, "script": list(script or ["ok"])}
        )

    @classmethod
    def reset(cls, provider_id: str | None = None, script: list[str] | None = None) -> None:
        """(Re)arm the script — tests call this in setup instead of mutating
        class attributes."""
        if provider_id is None:
            cls._STATE.clear()
        elif script is not None:
            cls._STATE[provider_id] = {"i": 0, "calls": 0, "script": list(script)}
        else:
            cls._STATE.pop(provider_id, None)

    def _next(self, request):
        st = ScriptedChatProvider._STATE[self.PROVIDER_ID]
        st["calls"] += 1
        idx = min(st["i"], len(st["script"]) - 1)
        st["i"] += 1
        op = st["script"][idx]
        model = getattr(request, "model", "fake-1")
        if op == "429":
            from uniinfer.errors import RateLimitError

            raise RateLimitError("scripted quota spent", status_code=429,
                                 response_body="{}", retry_after=900.0)
        if op == "dead":
            from uniinfer.errors import ProviderError

            raise ProviderError(
                'scripted dead: {"message":"Upstream request failed: '
                'Model is unavailable."}',
                status_code=400,
                response_body='{"message":"Model is unavailable."}',
            )
        content = None if op == "empty" else op
        finish = "stop" if op == "empty" else None
        return ChatCompletionResponse(
            message=ChatMessage(role="assistant", content=content),
            provider=self.PROVIDER_ID, model=model,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            raw_response={}, finish_reason=finish,
        )

    def complete(self, request, **kwargs):
        return self._next(request)

    async def acomplete(self, request, **kwargs):
        return self._next(request)

    def stream_complete(self, request, **kwargs):
        resp = self._next(request)
        if resp.message.content is not None:
            yield resp
        yield ChatCompletionResponse(
            message=ChatMessage(role="assistant", content=None),
            provider=self.PROVIDER_ID, model=resp.model, usage={},
            raw_response={}, finish_reason="stop",
        )

    async def astream_complete(self, request, **kwargs):
        for chunk in self.stream_complete(request, **kwargs):
            yield chunk
