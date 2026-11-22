"""Hermetic tests for the WS realtime-STT passthrough route.

Fake upstream speaks the sttts protocol (binary PCM in → JSON partial/final
out). No dgx, no credgoo: auth + instance resolution are patched; only the
piping logic runs for real.
"""
import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

WS_PATH = "/v1/audio/transcriptions/stream"


class FakeUpstream:
    """sttts-shaped WS server: echoes partials per chunk, final on empty frame."""

    def __init__(self):
        self.received: list[bytes] = []
        self.headers: dict[str, str] = {}
        self.connected = asyncio.Event()
        self.language = None

    async def handler(self, websocket):
        self.headers = {k: v for k, v in websocket.request.headers.items()}
        qs = websocket.request.path.split("?", 1)
        self.query = qs[1] if len(qs) > 1 else ""
        self.connected.set()
        async for msg in websocket:
            if isinstance(msg, bytes):
                if not msg:
                    await websocket.send(json.dumps({"type": "final", "partial": False, "text": "final text"}))
                    await websocket.close()
                    return
                self.received.append(msg)
                await websocket.send(json.dumps({"type": "partial", "partial": True, "text": f"chunk {len(msg)}"}))



def _patch_instance(monkeypatch, base_url="http://127.0.0.1:1"):
    from uniinfer.config import instances as inst_mod

    class _Spec:
        alias = "fakestt"
        provider = "openai-compat"

    _Spec.base_url = base_url
    monkeypatch.setattr(inst_mod, "resolve_instance", lambda alias: _Spec())


def _patch_auth(monkeypatch, key="up-key-123"):
    from uniinfer import auth as auth_mod

    monkeypatch.setattr(auth_mod, "verify_provider_access", lambda token, provider, own_key=None: key)


def test_missing_model_closes_4400(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(WS_PATH) as ws:
            ws.receive_json()
    assert exc.value.code == 4400


def test_unknown_instance_closes_4404(client, monkeypatch):
    from starlette.websockets import WebSocketDisconnect
    from uniinfer.config import instances as inst_mod

    def boom(alias):
        raise ValueError("unknown")

    monkeypatch.setattr(inst_mod, "resolve_instance", boom)
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f"{WS_PATH}?model=nope@x&token=t@t") as ws:
            ws.receive_json()
    assert exc.value.code == 4404


def test_happy_path_piping(monkeypatch):
    import socket
    import threading

    import uvicorn
    import websockets
    from websockets import serve

    from uniinfer.proxy_app import app as proxy_app

    def free_port() -> int:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    upstream_port = free_port()
    proxy_port = free_port()
    upstream = FakeUpstream()

    async def run_upstream(stop: threading.Event):
        async with serve(upstream.handler, "127.0.0.1", upstream_port):
            while not stop.is_set():
                await asyncio.sleep(0.05)

    up_stop = threading.Event()
    threading.Thread(
        target=lambda: asyncio.run(run_upstream(up_stop)), daemon=True
    ).start()

    server = uvicorn.Server(
        uvicorn.Config(proxy_app, host="127.0.0.1", port=proxy_port, log_level="error")
    )
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started

    _patch_instance(monkeypatch, base_url=f"ws://127.0.0.1:{upstream_port}")
    _patch_auth(monkeypatch)

    async def scenario():
        async with websockets.connect(
            f"ws://127.0.0.1:{proxy_port}{WS_PATH}"
            "?model=fakestt@nemotron&token=t@t&language=de"
        ) as client_ws:
            await client_ws.send(b"\x01\x02\x03\x04")
            partial = json.loads(await client_ws.recv())
            assert partial == {"type": "partial", "partial": True, "text": "chunk 4"}
            await client_ws.send(b"")
            final = json.loads(await client_ws.recv())
            assert final == {"type": "final", "partial": False, "text": "final text"}

    asyncio.run(asyncio.wait_for(scenario(), timeout=10))
    assert upstream.received == [b"\x01\x02\x03\x04"]
    auth_header = next(
        (v for k, v in upstream.headers.items() if k.lower() == "authorization"), None
    )
    assert auth_header == "Bearer up-key-123"
    assert "language=de" in upstream.query

    up_stop.set()


def test_auth_fail_closes_4401(client, monkeypatch):
    from starlette.websockets import WebSocketDisconnect

    _patch_instance(monkeypatch)
    from uniinfer import auth as auth_mod

    def deny(token, provider, own_key=None):
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="no")

    monkeypatch.setattr(auth_mod, "verify_provider_access", deny)
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f"{WS_PATH}?model=fakestt@m&token=bad") as ws:
            ws.receive_json()
    assert exc.value.code == 4401


def test_scheme_translation_http_to_ws(client, monkeypatch):
    """Real hab-style base_url (http://) must translate to ws:// for the upstream."""
    import websockets as ws_mod
    from starlette.websockets import WebSocketDisconnect

    captured = {}

    class _Up:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def close(self):
            pass

    async def fake_connect(url, **kwargs):
        captured["url"] = url
        return _Up()

    monkeypatch.setattr(ws_mod, "connect", fake_connect)
    _patch_instance(monkeypatch, base_url="http://[232:bc06::1]:3001/v1")
    _patch_auth(monkeypatch)

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"{WS_PATH}?model=fakestt@m&token=t@t") as ws:
            ws.receive_json()
    assert captured["url"].startswith("ws://[232:bc06::1]:3001/v1/audio/transcriptions/stream")


def test_token_via_authorization_header(client, monkeypatch):
    """Non-browser clients may send the bearer as header instead of ?token=."""
    from starlette.websockets import WebSocketDisconnect

    seen = {}

    class _Up:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def close(self):
            pass

    async def fake_connect(url, **kwargs):
        seen["url"] = url
        return _Up()

    import websockets as ws_mod

    def check(token, provider, own_key=None):
        seen["token"] = token
        return "up-key"

    monkeypatch.setattr(ws_mod, "connect", fake_connect)
    from uniinfer import auth as auth_mod

    monkeypatch.setattr(auth_mod, "verify_provider_access", check)
    _patch_instance(monkeypatch)

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            WS_PATH + "?model=fakestt@m",
            headers={"Authorization": "Bearer hdr-token@x"},
        ) as ws:
            ws.receive_json()
    assert seen["token"] == "hdr-token@x"


def test_ws_forwards_x_provider_key(client, monkeypatch):
    """X-Provider-Key on the WS handshake must reach verify_provider_access."""
    from starlette.websockets import WebSocketDisconnect

    seen = {}

    class _Up:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def close(self):
            pass

    async def fake_connect(url, **kwargs):
        seen["url"] = url
        return _Up()

    import websockets as ws_mod
    from uniinfer import auth as auth_mod

    def check(token, provider, own_key=None):
        seen["own_key"] = own_key
        return own_key or "pooled"

    monkeypatch.setattr(ws_mod, "connect", fake_connect)
    monkeypatch.setattr(auth_mod, "verify_provider_access", check)
    _patch_instance(monkeypatch)

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            WS_PATH + "?model=fakestt@m&token=t@t",
            headers={"X-Provider-Key": "my-own-upstream-key"},
        ) as ws:
            ws.receive_json()
    assert seen["own_key"] == "my-own-upstream-key"


def test_upstream_unreachable_closes_4502(client, monkeypatch):
    from starlette.websockets import WebSocketDisconnect

    _patch_instance(monkeypatch, base_url="ws://127.0.0.1:1")
    _patch_auth(monkeypatch)
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f"{WS_PATH}?model=fakestt@m&token=t@t") as ws:
            ws.receive_json()
    assert exc.value.code == 4502
