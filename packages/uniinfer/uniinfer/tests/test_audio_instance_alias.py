"""Audio-Router mit Instanz-Aliasen: hab@modell → openai-compat Target.

Custom instances (provider_instances.json, z.B. der dgx-Gateway `hab`) dienen
OpenAI-geformte /v1/audio/*-Endpoints hinter ihrem base_url. Der Router löst
den Alias auf, reicht base_url + aufgelösten Key ans Target; ohne Instanz gilt
weiterhin der Registry-Pfad (tu, openai, openaitts). Hermetisch, kein Netz.
"""
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from uniinfer.config.instances import InstanceSpec
from uniinfer.proxy_routers.audio import create_audio_router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(create_audio_router(
        parse_model=lambda m: (m.partition("@")[0], m.partition("@")[2])))
    return TestClient(app)


def _hab_spec():
    return InstanceSpec(alias="hab", provider="openai-compat", is_builtin=False,
                        base_url="http://127.0.0.1:18020/v1",
                        credgoo_service="dgxp", requires_api_key=True)


def _patch(monkeypatch, captured, stt_body=None, tts_content=b"RIFF"):
    monkeypatch.setattr("uniinfer.proxy_routers.audio.verify_provider_access",
                        lambda token, alias: "sk-hab-upstream")
    monkeypatch.setattr(
        "uniinfer.config.instances.resolve_instance",
        lambda a: _hab_spec() if a == "hab" else (_ for _ in ()).throw(ValueError(a)))

    class Resp:
        def __init__(self, status=200, content=b"", headers=None, js=None):
            self.status_code = status
            self.content = content if isinstance(content, bytes) else content.encode()
            self.headers = headers or {}
            self._js = js
            self.text = content if isinstance(content, str) else content.decode("latin-1")
        def json(self):
            return self._js

    async def fake_post(self, url, json=None, files=None, data=None, headers=None, **kw):
        captured["url"] = url
        captured["json"] = json
        captured["files"] = files
        captured["data"] = data
        captured["headers"] = headers
        if url.endswith("/audio/speech"):
            return Resp(content=tts_content,
                        headers={"Content-Type": "audio/wav"})
        return Resp(js=stt_body or {"text": "hall", "language": "auto",
                                    "duration_sec": 1.0})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)


class TestTTSAlias:
    def test_hab_instance_tts(self, client, monkeypatch):
        captured = {}
        _patch(monkeypatch, captured)
        r = client.post("/v1/audio/speech", json={
            "model": "hab@/models/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "input": "Guten Tag", "voice": "ono_anna"})
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "audio/wav"
        assert captured["url"] == "http://127.0.0.1:18020/v1/audio/speech"
        assert captured["json"]["model"] == "/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        assert captured["json"]["voice"] == "ono_anna"
        assert captured["headers"]["Authorization"] == "Bearer sk-hab-upstream"


class TestSTTAlias:
    def test_hab_instance_stt(self, client, monkeypatch):
        captured = {}
        _patch(monkeypatch, captured)
        r = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", b"RIFFxxxx", "audio/wav")},
            data={"model": "hab@nemotron-3.5-asr-streaming-0.6b"})
        assert r.status_code == 200, r.text
        assert r.json()["text"] == "hall"
        assert captured["url"] == "http://127.0.0.1:18020/v1/audio/transcriptions"
        assert captured["data"]["model"] == "nemotron-3.5-asr-streaming-0.6b"
        assert captured["headers"]["Authorization"] == "Bearer sk-hab-upstream"

    def test_alias_loescht_prefix_im_model_field(self, client, monkeypatch):
        """model-Form-Feld ist 'hab@x' — upstream bekommt NUR das bare 'x'."""
        captured = {}
        _patch(monkeypatch, captured)
        client.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", b"RIFF", "audio/wav")},
                    data={"model": "hab@whisper-large"})
        assert captured["data"]["model"] == "whisper-large"


class TestLegacyPfad:
    def test_keine_instanz_registry_pfad(self, client, monkeypatch):
        """Nicht-Instanz-Aliase (z.B. tu@...) laufen wie bisher über die
        Registry — resolve_instance wirft, Target-Bau entscheidet."""
        monkeypatch.setattr(
            "uniinfer.config.instances.resolve_instance",
            lambda a: (_ for _ in ()).throw(ValueError(a)))
        from uniinfer.proxy_routers.audio import verify_provider_access
        monkeypatch.setattr("uniinfer.proxy_routers.audio.verify_provider_access",
                            lambda token, alias: "sk-tu")
        r = client.post("/v1/audio/speech", json={
            "model": "unknownprov@does-not-exist", "input": "x"})
        # TTSTarget wirft ValueError (TTS not supported) → 400, kein 500
        assert r.status_code == 400
