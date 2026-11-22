"""Hermetic tests for the surface-smoke engine (chat + responses, stream + non-stream).

Spins a minimal echo backend on a uvicorn thread — the smoke code under test
parses REAL SSE via uniinfer.sse, no proxy internals patched.
"""
import asyncio
import socket
import threading
import time

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from uniinfer.proxy_routers.smoke import _smoke_surface_one

app = FastAPI()


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    if body.get("stream"):
        async def gen():
            for part in ("ok", "ay"):
                yield f'data: {{"choices": [{{"index": 0, "delta": {{"content": "{part}"}}}}]}}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")
    return JSONResponse(
        {"choices": [{"index": 0, "message": {"role": "assistant", "content": "okay"}}]}
    )


@app.post("/v1/responses")
async def responses(request: Request):
    body = await request.json()
    if body.get("stream"):
        async def gen():
            yield 'event: response.created\ndata: {"type": "response.created", "response": {}}\n\n'
            yield 'event: response.output_text.delta\ndata: {"type": "response.output_text.delta", "delta": "ok"}\n\n'
            yield 'event: response.completed\ndata: {"type": "response.completed", "response": {"output": []}}\n\n'

        return StreamingResponse(gen(), media_type="text/event-stream")
    return JSONResponse(
        {
            "object": "response",
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "okay"}]}
            ],
        }
    )


@pytest.fixture(scope="module")
def base_url():
    import uvicorn

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
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True


def run(coro):
    return asyncio.run(coro)


def _probe(base_url, **kw):
    defaults = dict(base_url=base_url, token="t", model="x@y", prompt="hi", stream=False)
    defaults.update(kw)
    return run(_smoke_surface_one(**defaults))


def test_chat_nonstream(base_url):
    r = _probe(base_url, surface="chat")
    assert r["ok"] and r["content_preview"] == "okay"


def test_chat_stream(base_url):
    r = _probe(base_url, surface="chat", stream=True)
    assert r["ok"]
    assert r["content_preview"] == "okay"  # "ok"+"ay" deltas joined
    assert r["events"] >= 3


def test_responses_nonstream(base_url):
    r = _probe(base_url, surface="responses")
    assert r["ok"] and r["content_preview"] == "okay"


def test_responses_stream_envelope(base_url):
    r = _probe(base_url, surface="responses", stream=True)
    assert r["ok"]
    assert r["event_first"] == "response.created"
    assert r["event_last"] == "response.completed"
    assert r["content_preview"] == "ok"


def test_unknown_surface_rejected(base_url):
    r = _probe(base_url, surface="nope")
    assert not r["ok"] and r["status"] == "bad_request"
