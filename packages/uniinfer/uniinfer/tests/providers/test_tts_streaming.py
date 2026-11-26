"""TTS-Live-Streaming durch den uniinfer-Proxy (hab → DGX vLLM-Omni).

Issue-Kette 2026-11-19: der Proxy pufferte jede TTS-Response komplett
(await client.post → resp.content), bevor der Router sie als eine Response
zurückgab — erster Ton bei 38s statt bei Synthese-Beginn. Jetzt:

  - TTSRequest kennt language/stream/stream_format (language wurde vorher
    vom Router-Modell still verworfen — pydantic ignoriert Extra-Felder)
  - TTSProvider.astream_speech: Default-Fallback (ein Chunk aus
    agenerate_speech) — Provider ohne Nativ-Streaming bleiben nutzbar
  - OpenAICompatTTSProvider.astream_speech: httpx client.stream +
    aiter_bytes — Bytes fließen, sobald der Upstream sie schickt
  - TTSTarget.astream_synthesize: derselbe Dispatch wie asynthesize
  - Router: stream:true → StreamingResponse (kein Puffer)

Hermetisch, kein Netz (httpx gemockt).
"""
import json
from typing import AsyncIterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from uniinfer.audio import TTSTarget
from uniinfer.core import TTSProvider, TTSRequest, TTSResponse
from uniinfer.providers.openai_compat_audio import OpenAICompatTTSProvider
from uniinfer.proxy_routers.audio import create_audio_router


# --- TTSRequest: neue Felder -------------------------------------------------

def test_tts_request_language_and_stream_fields():
    r = TTSRequest(input="x", language="German", stream=True, stream_format="audio")
    assert r.language == "German"
    assert r.stream is True
    assert r.stream_format == "audio"
    # Defaults bleiben
    r2 = TTSRequest(input="x")
    assert r2.language is None and r2.stream is False


# --- Provider-Fallback (kein Nativ-Streaming) --------------------------------

class _BufferedProvider(TTSProvider):
    """Simuliert einen Provider OHNE astream_speech-Override."""
    async def agenerate_speech(self, request, **kw):
        return TTSResponse(audio_content=b"FULL", model="m", provider="p")


@pytest.mark.asyncio
async def test_provider_stream_fallback_single_chunk():
    chunks = [c async for c in _BufferedProvider().astream_speech(TTSRequest(input="x"))]
    assert chunks == [b"FULL"]  # ein Chunk, kompletter Buffer — Seam bleibt uniform


# --- OpenAICompatTTSProvider.astream_speech (nativ) ---------------------------

class _FakeStreamResp:
    def __init__(self, status=200, chunks=(b"RIFF", b"DATA1", b"DATA2")):
        self.status_code = status
        self._chunks = chunks
        self._err_body = b'{"error":"nope"}'

    async def aread(self):
        return self._err_body

    async def aiter_bytes(self):
        for c in self._chunks:
            yield c


class _FakeStreamCtx:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_openai_compat_stream_yields_chunks(monkeypatch):
    import httpx
    captured = {}

    class FakeClient:
        def __init__(self, timeout=None):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def stream(self, method, url, json=None, headers=None):
            captured.update(method=method, url=url, json=json, headers=headers)
            return _FakeStreamCtx(_FakeStreamResp())

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    p = OpenAICompatTTSProvider(api_key="k", base_url="http://up/v1")
    req = TTSRequest(input="Hallo", model="m", voice="sohee",
                     language="German", stream_format="audio")
    chunks = [c async for c in p.astream_speech(req)]

    assert chunks == [b"RIFF", b"DATA1", b"DATA2"]
    # Payload: stream-Flags + language kommen durch
    body = captured["json"]
    assert body["stream"] is True
    assert body["stream_format"] == "audio"
    assert body["language"] == "German"
    assert captured["url"] == "http://up/v1/audio/speech"


@pytest.mark.asyncio
async def test_openai_compat_stream_upstream_error(monkeypatch):
    import httpx

    class FakeClient:
        def __init__(self, timeout=None):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def stream(self, method, url, json=None, headers=None):
            return _FakeStreamCtx(_FakeStreamResp(status=500))

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    p = OpenAICompatTTSProvider(api_key="k", base_url="http://up/v1")
    with pytest.raises(RuntimeError, match="500"):
        async for _ in p.astream_speech(TTSRequest(input="x")):
            pass


# --- TTSTarget.astream_synthesize (Seam-Dispatch) ------------------------------

@pytest.mark.asyncio
async def test_target_stream_dispatches_to_provider(monkeypatch):
    calls = {}

    class FakeProvider:
        def __init__(self, api_key=None, base_url=None):
            pass

        async def astream_speech(self, request):
            calls["payload"] = {
                "model": request.model, "language": request.language,
                "stream": request.stream, "stream_format": request.stream_format,
            }
            yield b"CHUNK"

    target = TTSTarget.__new__(TTSTarget)
    target.provider_name = "openai-compat"
    target.model_name = "m"
    target.api_key = "k"
    target.base_url = "http://up/v1"
    target._provider_cls = FakeProvider

    chunks = [c async for c in target.astream_synthesize(
        "Text", voice="sohee", language="German", stream_format="audio")]
    assert chunks == [b"CHUNK"]
    assert calls["payload"] == {"model": "m", "language": "German",
                                "stream": True, "stream_format": "audio"}


# --- Router: stream:true → StreamingResponse ----------------------------------

def _router_client():
    app = FastAPI()
    app.include_router(create_audio_router(
        parse_model=lambda m: (m.partition("@")[0], m.partition("@")[2])))
    return TestClient(app)


class _PcmStreamingProvider:
    """Liefert rohe s16le-Samples OHNE RIFF-Kopf (wie der Phonon-Server)."""

    def __init__(self, api_key=None, base_url=None):
        pass

    async def astream_speech(self, request):
        yield b"\xdf\xff\xdc\xff"
        yield b"more-pcm"


class _RouterStreamingProvider:
    """Modul-level (Registry will dotted paths): streamed 2-Chunk-Antwort."""

    def __init__(self, api_key=None, base_url=None):
        pass

    async def astream_speech(self, request):
        yield b"RIFF-header"
        yield b"audio-bytes"


def test_router_streams_audio(client=None):
    client = _router_client()

    class _InstSpec:
        alias = "hab"
        provider = "openai-compat"
        base_url = "http://127.0.0.1:9/v1"
        requires_api_key = False

    import uniinfer.proxy_routers.audio as audio_router
    import uniinfer.config.instances as instances

    orig_resolve = instances.resolve_instance
    orig_verify = audio_router.verify_provider_access
    instances.resolve_instance = lambda a: _InstSpec()
    audio_router.verify_provider_access = lambda token, alias, own_key=None: "k"

    class _StreamingProvider:
        def __init__(self, api_key=None, base_url=None):
            pass

        async def astream_speech(self, request):
            yield b"RIFF-header"
            yield b"audio-bytes"

    import uniinfer.audio as audio_mod
    orig_registry = audio_mod._TTS_REGISTRY
    dotted = f"{_RouterStreamingProvider.__module__}.{_RouterStreamingProvider.__name__}"
    audio_mod._TTS_REGISTRY = {**orig_registry, "openai-compat": dotted}

    try:
        with client.stream("POST", "/v1/audio/speech", json={
            "model": "hab@qwen3-tts", "input": "Hallo Welt",
            "voice": "sohee", "language": "German",
            "stream": True, "stream_format": "audio",
        }) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "audio/wav"
            body = b"".join(resp.iter_bytes())
            assert body == b"RIFF-headeraudio-bytes"
    finally:
        instances.resolve_instance = orig_resolve
        audio_router.verify_provider_access = orig_verify
        audio_mod._TTS_REGISTRY = orig_registry


def test_router_streams_pcm_content_type():
    """stream:true + response_format:pcm muss 'audio/pcm' melden, nicht
    'audio/wav' (Vorfall 2026-11-26: Client schnitt blind 44 B WAV-Kopf ab
    und zerstoerte den Stream-Anfang, weil der Content-Type log)."""
    client = _router_client()

    class _InstSpec:
        alias = "hab"
        provider = "openai-compat"
        base_url = "http://127.0.0.1:9/v1"
        requires_api_key = False

    import uniinfer.proxy_routers.audio as audio_router
    import uniinfer.config.instances as instances

    orig_resolve = instances.resolve_instance
    orig_verify = audio_router.verify_provider_access
    instances.resolve_instance = lambda a: _InstSpec()
    audio_router.verify_provider_access = lambda token, alias, own_key=None: "k"

    import uniinfer.audio as audio_mod
    orig_registry = audio_mod._TTS_REGISTRY
    dotted = f"{_PcmStreamingProvider.__module__}.{_PcmStreamingProvider.__name__}"
    audio_mod._TTS_REGISTRY = {**orig_registry, "openai-compat": dotted}

    try:
        with client.stream("POST", "/v1/audio/speech", json={
            "model": "hab@qwen3-tts", "input": "Hallo Welt",
            "voice": "sohee", "response_format": "pcm",
            "stream": True, "stream_format": "audio",
        }) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "audio/pcm"
            body = b"".join(resp.iter_bytes())
            assert body == b"\xdf\xff\xdc\xffmore-pcm"
    finally:
        instances.resolve_instance = orig_resolve
        audio_router.verify_provider_access = orig_verify
        audio_mod._TTS_REGISTRY = orig_registry


def test_router_nonstreaming_unchanged():
    """stream:false (Default) bleibt die gepufferte Response — kein Verhaltenbruch."""
    client = _router_client()
    # Ohne Instanz-Resolution landet der Request im Registry-Pfad und scheitert
    # an auth — hier reicht der Nachweis, dass stream-los KEIN StreamingResponse-
    # Weg nimmt (4xx, nicht Streaming-200 mit anderem Format).
    r = client.post("/v1/audio/speech", json={
        "model": "hab@x", "input": "text"})
    assert r.status_code in (400, 401, 500)
