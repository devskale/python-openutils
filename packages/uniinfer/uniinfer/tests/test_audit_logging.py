"""Audit-logging seam: every provider streaming through
OpenAICompatibleChatProvider.astream_complete must emit a STREAM-OPEN INFO line
and capture upstream non-200s to the provider raw log.

Regression for the 2026-10-05 invisible-fail incident: the first logging fix
hooked only the non-streaming path (_chat_acomplete) while pi — like every
OpenAI-compatible client — streams; the audit line never fired and opencode
403/400 fails left zero journal trace.
"""
import json
import logging

import pytest

from uniinfer.core import ChatCompletionRequest, ChatMessage
from uniinfer.errors import UniInferError
from uniinfer.providers.opencode import OpenCodeProvider

_SSE_OK = [
    'data: {"model":"space-bunny-free","choices":[{"index":0,"delta":{"role":"assistant"}}]}',
    'data: {"model":"space-bunny-free","choices":[{"index":0,"delta":{"content":"ok"}}]}',
    'data: {"model":"space-bunny-free","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
    "data: [DONE]",
]
_ERR_BODY = '{"error":{"type":"FreeTierError","message":"nope"}}'


class _SSEResponse:
    def __init__(self, lines, status_code):
        self._lines = lines
        self.status_code = status_code

    async def aread(self):
        return _ERR_BODY.encode()

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _StreamCtx:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        return False


class _FakeStreamClient:
    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code

    def stream(self, method, url, **kwargs):
        return _StreamCtx(_SSEResponse(self._lines, self.status_code))


def _awaitable(value):
    async def get():
        return value
    return get


def _req():
    return ChatCompletionRequest(model="space-bunny-free",
                                 messages=[ChatMessage(role="user", content="hi")])


def _provider(monkeypatch, client):
    p = OpenCodeProvider(api_key="test")
    monkeypatch.setattr(p, "_get_async_client", _awaitable(client))
    monkeypatch.setattr("uniinfer.providers.opencode._docs_tables", lambda: {})
    return p


@pytest.mark.asyncio
async def test_stream_open_audited_on_streaming_path(monkeypatch, caplog):
    p = _provider(monkeypatch, _FakeStreamClient(_SSE_OK))
    with caplog.at_level(logging.INFO, logger="uniinfer.logging_utils"):
        chunks = [c async for c in p.astream_complete(_req())]
    assert any(c.message.content == "ok" for c in chunks)
    assert "[opencode] STREAM-OPEN" in caplog.text, (
        "streaming path must emit the audit line (2026-10-05: first fix hung "
        "on _chat_acomplete and never fired for streaming clients)")


@pytest.mark.asyncio
async def test_upstream_error_captured_to_raw_log(monkeypatch, caplog, tmp_path):
    monkeypatch.setenv("UNIINFER_LOG_DIR", str(tmp_path))
    p = _provider(monkeypatch, _FakeStreamClient([], status_code=403))
    with caplog.at_level(logging.INFO, logger="uniinfer.logging_utils"):
        with pytest.raises(UniInferError):
            async for _ in p.astream_complete(_req()):
                pass
    raw = tmp_path / "opencode_raw_chat.log"
    assert raw.exists(), "upstream non-200 must be captured to the provider raw log"
    event = json.loads(raw.read_text().splitlines()[-1])
    assert event["provider"] == "opencode"
    assert event["data"]["status_code"] == 403
    assert "FreeTierError" in event["data"]["body"]
