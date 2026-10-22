"""Empty-stream + first-token-stall guard (uniinfer-proxy-leerstrom-erkennung).

Incident (2026-10-22, ``dgx-vllm-leerstrom-unter-last`` / edge issues): vLLM
under load ends SSE streams with HTTP 200 and ZERO content. The relay
forwarded that as a clean empty "success"; clients burned retries on a masked
provider failure (5x ``unparseable response ('')``). Contract under test:

  (a) empty 200 stream   → 1 internal retry; healed → transparent continuation
                            (one [DONE], content present, no error chunk);
                            exhausted → error chunk ``empty_stream``/502.
  (b) stalled first token→ abort + retry; exhausted before ANY yield → real
                            ``ProviderError(502)`` (router: true HTTP 502).
  (c) healthy stream     → byte-identical passthrough, single [DONE].

Every detection emits an ``ALARM {json}`` line in the robotni-alarms schema
(error_type ``empty_stream`` / ``first_token_stall``).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import pytest
from fastapi.testclient import TestClient

from uniinfer import ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from uniinfer.core import ChatProvider, ModelInfo
from uniinfer.errors import ProviderError, RateLimitError
from uniinfer.factory import ProviderFactory
from uniinfer.proxy_services.stream_guard import (
    _PassObserver,
    astream_guarded,
    emit_guard_alarm,
    nonstream_masked_empty,
)
from uniinfer.proxy_services.streaming import astream_response_generator

HANG_S = 30.0  # "forever" relative to the tiny test budgets


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
    """Upstream stand-in. ``behaviors``: one script per astream_complete call —
    'ok' (content + stop), 'empty' (finish-only, the 200-empty shape), 'none'
    (zero chunks), 'hang' (never a first token), or an exception instance."""

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
            if behavior == "empty":
                yield _raw_chunk(finish="stop")
                return
            if behavior == "none":
                return
            yield _raw_chunk(content="hello ")
            yield _raw_chunk(content="world")
            yield _raw_chunk(finish="stop")

        return _CountingGen(_gen(), outer)

    async def aclose(self):
        pass


def _normalize(events: list[str]) -> list[str]:
    """Erase per-generator randomness (completion id UUID, created second)."""
    out = []
    for ev in events:
        ev = re.sub(r"chatcmpl-[0-9a-f-]+", "chatcmpl-X", ev)
        ev = re.sub(r'"created": \d+', '"created": 0', ev)
        out.append(ev)
    return out


def _drive(target) -> list[str]:
    async def run() -> list[str]:
        gen = astream_guarded(
            target=target,
            messages=[{"role": "user", "content": "hi"}],
            temp=0.0,
            max_tok=64,
        )
        return [ev async for ev in gen]

    return asyncio.run(run())


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


def _alarms(caplog) -> list[dict]:
    out = []
    for rec in caplog.records:
        msg = rec.getMessage()
        if msg.startswith("ALARM "):
            out.append(json.loads(msg[6:]))
    return out


# --- observer (pure unit) -----------------------------------------------------


def test_observer_classifies_events():
    ob = _PassObserver()
    ob.observe(": keep-alive\n\n")                      # comment — ignored
    ob.observe('data: {"delta": {"role": "assistant"}}\n\n')  # welcome — not payload
    assert not ob.has_payload
    ob.observe('data: {"choices": [{"delta": {"content": "hi"}}]}\n\n')
    assert ob.has_payload and ob.content_events == 1
    ob2 = _PassObserver()
    ob2.observe('data: {"choices": [{"delta": {"reasoning_content": "hm"}}]}\n\n')
    assert ob2.has_payload                       # thinking-only answers are legit
    ob3 = _PassObserver()
    ob3.observe('data: {"error": {"message": "x", "type": "t", "code": 504}}\n\n')
    assert ob3.saw_error and not ob3.has_payload
    ob4 = _PassObserver()
    ob4.observe('data: {"choices": [], "usage": {"total_tokens": 1}}\n\n')
    assert not ob4.has_payload                   # usage-only terminal chunk


def test_alarm_schema_matches_robotni_contract(caplog):
    with caplog.at_level(logging.WARNING, logger="uniioai_proxy"):
        emit_guard_alarm("empty_stream", "tu@qwen-3.6-35b", "EMPTY-STREAM: test")
    (alarm,) = _alarms(caplog)
    assert alarm["provider"] == "tu"             # provider prefix split off
    assert alarm["model"] == "tu@qwen-3.6-35b"
    assert alarm["error_type"] == "empty_stream"
    assert alarm["severity"] == "alarm"
    assert alarm["package"] == "uniinfer"
    assert isinstance(alarm["ts"], float)


def test_nonstream_masked_empty_predicate():
    def _resp(**kw) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            message=ChatMessage(role="assistant", content=kw.get("content")),
            provider="tu", model="m", usage={}, raw_response={},
            thinking=kw.get("thinking"),
            reasoning_details=kw.get("reasoning_details"),
        )

    assert nonstream_masked_empty(_resp(content=None)) is True
    assert nonstream_masked_empty(_resp(content="")) is True
    assert nonstream_masked_empty(_resp(content="answer")) is False
    assert nonstream_masked_empty(_resp(content=None, thinking="thought")) is False
    assert nonstream_masked_empty(_resp(content=None, reasoning_details=[{"a": 1}])) is False
    tool_resp = ChatCompletionResponse(
        message=ChatMessage(
            role="assistant", content=None,
            tool_calls=[{"id": "c1", "type": "function", "function": {"name": "f"}}],
        ),
        provider="tu", model="m", usage={}, raw_response={},
    )
    assert nonstream_masked_empty(tool_resp) is False


# --- guard over the real relay (hermetic, fake upstream) -----------------------


def test_healthy_stream_passthrough_byte_identical():
    """Guard ON: a healthy stream is relayed byte-for-byte (regression guard)."""
    guarded = _drive(_FakeTarget(["ok"]))
    bare_target = _FakeTarget(["ok"])

    async def bare() -> list[str]:
        gen = astream_response_generator(
            target=bare_target, messages=[{"role": "user", "content": "hi"}],
            temp=0.0, max_tok=64,
        )
        return [ev async for ev in gen]

    assert _normalize(guarded) == _normalize(asyncio.run(bare()))


def test_empty_stream_retried_and_healed(caplog):
    """(a) empty pass → 1 retry → healed: transparent continuation for the client."""
    target = _FakeTarget(["empty", "ok"])
    with caplog.at_level(logging.WARNING, logger="uniioai_proxy"):
        raw = _drive(target)
    assert target.calls == 2, "empty verdict must retry the upstream once"
    assert _content(raw) == "hello world"
    assert raw.count("data: [DONE]\n\n") == 1, "each pass holds its [DONE] — one terminator"
    assert _error_chunk(raw) is None
    (alarm,) = _alarms(caplog)
    assert alarm["error_type"] == "empty_stream"


def test_zero_chunk_stream_is_also_empty(caplog):
    """Upstream yields literally nothing (immediate StopAsyncIteration) — same verdict."""
    target = _FakeTarget(["none", "ok"])
    with caplog.at_level(logging.WARNING, logger="uniioai_proxy"):
        raw = _drive(target)
    assert target.calls == 2
    assert _content(raw) == "hello world"
    assert _error_chunk(raw) is None


def test_empty_stream_exhausted_yields_error_chunk_502(caplog):
    """(a) budget spent → recognizable empty_stream/502 chunk, never a clean
    empty success; the SSE 200 was already committed (welcome seen)."""
    target = _FakeTarget(["empty", "empty"])
    with caplog.at_level(logging.WARNING, logger="uniioai_proxy"):
        raw = _drive(target)
    assert target.calls == 2
    err = _error_chunk(raw)
    assert err is not None
    assert err["type"] == "empty_stream"
    assert err["code"] == 502
    assert "no content" in err["message"].lower()
    assert raw[-1] == "data: [DONE]\n\n"
    assert len(_alarms(caplog)) == 2, "one alarm per failed attempt"


def test_first_token_stall_aborts_and_retries(monkeypatch, caplog):
    """(b) no contentful event within the stall budget → pass aborted (client-
    disconnect-equivalent close) → retry heals."""
    monkeypatch.setenv("UNIINFER_STREAM_HEARTBEAT", "5")     # relay priming stays quiet
    monkeypatch.setenv("UNIINFER_FIRST_TOKEN_STALL_S", "0.15")
    target = _FakeTarget(["hang", "ok"])
    with caplog.at_level(logging.WARNING, logger="uniioai_proxy"):
        raw = _drive(target)
    assert target.calls == 2
    assert _content(raw) == "hello world"
    assert _error_chunk(raw) is None
    (alarm,) = _alarms(caplog)
    assert alarm["error_type"] == "first_token_stall"
    assert target.closed_iters == 2, "both relay passes closed their upstream"


def test_stall_exhausted_before_anything_yields_real_502(monkeypatch, caplog):
    """(b) budget spent while NOTHING reached the client → ProviderError(502)
    — the router answers with a true HTTP 502 before committing the SSE 200,
    so backup chains route around the dead upstream."""
    monkeypatch.setenv("UNIINFER_STREAM_HEARTBEAT", "5")
    monkeypatch.setenv("UNIINFER_FIRST_TOKEN_STALL_S", "0.15")

    async def first_pull():
        gen = astream_guarded(
            target=_FakeTarget(["hang", "hang"]),
            messages=[{"role": "user", "content": "hi"}],
            temp=0.0, max_tok=64,
        )
        return await gen.__anext__()

    with caplog.at_level(logging.WARNING, logger="uniioai_proxy"):
        with pytest.raises(ProviderError) as exc_info:
            asyncio.run(first_pull())
    assert exc_info.value.status_code == 502
    assert any(a["error_type"] == "first_token_stall" for a in _alarms(caplog))


def test_open_ratelimit_still_propagates_unwrapped():
    """Open-time upstream 429 passes through the guard untouched (router: 429)."""

    async def first_pull():
        gen = astream_guarded(
            target=_FakeTarget([RateLimitError("upstream quota exceeded")]),
            messages=[{"role": "user", "content": "hi"}],
            temp=0.0, max_tok=64,
        )
        return await gen.__anext__()

    with pytest.raises(RateLimitError):
        asyncio.run(first_pull())


def test_detect_disabled_passthrough(monkeypatch, caplog):
    """UNIINFER_EMPTY_STREAM_DETECT=0: the empty pass goes through as before
    (one call, no retry, no alarm) — the operator's escape hatch."""
    monkeypatch.setenv("UNIINFER_EMPTY_STREAM_DETECT", "0")
    target = _FakeTarget(["empty"])
    with caplog.at_level(logging.WARNING, logger="uniioai_proxy"):
        raw = _drive(target)
    assert target.calls == 1
    assert _error_chunk(raw) is None
    assert _alarms(caplog) == []


# --- router e2e (real app wiring, fake provider) --------------------------------


class _FakeGuardProvider(ChatProvider):
    PROVIDER_ID = "fakeguard"
    REQUIRES_API_KEY = False
    BASE_URL = ""
    calls = 0
    always_empty = False

    @staticmethod
    def _response(content):
        return ChatCompletionResponse(
            message=ChatMessage(role="assistant", content=content),
            provider="fakeguard", model="fake-1",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            raw_response={},
        )

    def complete(self, request: ChatCompletionRequest, **kwargs):
        _FakeGuardProvider.calls += 1
        if _FakeGuardProvider.always_empty or _FakeGuardProvider.calls == 1:
            return self._response(None)
        return self._response("healed")

    async def acomplete(self, request, **kwargs):
        return self.complete(request, **kwargs)

    def stream_complete(self, request, **kwargs):
        _FakeGuardProvider.calls += 1
        if _FakeGuardProvider.always_empty or _FakeGuardProvider.calls == 1:
            return
        yield self._response("stream-healed")
        yield ChatCompletionResponse(
            message=ChatMessage(role="assistant", content=None),
            provider="fakeguard", model="fake-1", usage={}, raw_response={},
            finish_reason="stop",
        )

    async def astream_complete(self, request, **kwargs):
        for chunk in self.stream_complete(request, **kwargs):
            yield chunk

    @classmethod
    def list_models(cls, api_key=None, base_url=None, **kwargs):
        return [ModelInfo(id="fake-1", owned_by="fakeguard")]


@pytest.fixture
def app_client(monkeypatch):
    ProviderFactory.register_provider("fakeguard", _FakeGuardProvider)
    _FakeGuardProvider.calls = 0
    _FakeGuardProvider.always_empty = False
    monkeypatch.setattr(
        "uniinfer.proxy_routers.chat.verify_provider_access", lambda t, p: "fake-key"
    )
    from uniinfer.config.instances import InstanceSpec

    monkeypatch.setattr(
        "uniinfer.completion.resolve_instance",
        lambda alias, **k: InstanceSpec(
            alias=alias, provider=alias, requires_api_key=False, is_builtin=False
        ),
    )
    from uniinfer.proxy_app import app

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
    ProviderFactory._providers.pop("fakeguard", None)


def _post(client, *, stream: bool):
    return client.post(
        "/v1/chat/completions",
        json={
            "model": "fakeguard@fake-1",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": stream,
        },
    )


def test_router_nonstream_empty_retries_then_answers(app_client):
    """Non-stream: empty completion retried once; the client gets real content."""
    r = _post(app_client, stream=False)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "healed"
    assert _FakeGuardProvider.calls == 2


def test_router_nonstream_empty_exhausted_is_real_502(app_client):
    """Non-stream, budget spent: a TRUE HTTP 502 (no SSE committed) with a
    clear detail — the llminvoke backup chain can finally see the failure."""
    _FakeGuardProvider.always_empty = True
    r = _post(app_client, stream=False)
    assert r.status_code == 502
    assert "empty completion" in r.json()["detail"].lower()
    assert _FakeGuardProvider.calls == 2, "default budget: initial + 1 retry"


def test_router_stream_empty_heals_over_http(app_client):
    """Streaming through the FULL stack (router prime + guard + relay): the
    empty pass heals transparently — content arrives, exactly one [DONE]."""
    with app_client.stream("POST", "/v1/chat/completions", json={
        "model": "fakeguard@fake-1",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }) as r:
        assert r.status_code == 200
        lines = [ln for ln in r.iter_lines() if ln]
    content = "".join(
        (json.loads(ln[6:]).get("choices") or [{}])[0].get("delta", {}).get("content", "")
        for ln in lines
        if ln.startswith("data: {") and "error" not in ln
    )
    assert content == "stream-healed"
    assert lines.count("data: [DONE]") == 1
    assert not any('"error"' in ln for ln in lines)
