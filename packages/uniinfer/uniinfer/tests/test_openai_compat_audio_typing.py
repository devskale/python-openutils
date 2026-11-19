"""OpenAICompatProvider.list_models + ModelInfo.derive_type — Audio-Klassifizierung.

Issue uniinfer-hab-audio-models-mislabeled-as-chat: der DGX model-proxy (hab)
listet den sttts-Stack (Nemotron ASR, Supertonic-3 TTS, Qwen3-TTS) in
/v1/models mit owned_by "sttts" — der generische openai-compat Provider
warf den Diskriminator away (owned_by := "openai-compat") und ließ type auf
dem Default "chat", womit Audio-Modelle als Chat-Kandidaten durchrutschten.

Hier geprüft: owned_by bleibt Upstream-Wahrheit, derive_type klassifiziert
die Fleet-Patterns (-asr/-, -tts/-, supertonic) und die bekannten Familien
(whisper, kokoro, piper). Hermetisch, kein Netz (httpx gemockt).
"""
from types import SimpleNamespace

from uniinfer.core import ModelInfo
from uniinfer.providers.openai_compat import OpenAICompatProvider


def _mock_httpx(monkeypatch, payload, captured=None):
    def fake_get(url, headers=None, timeout=None):
        if captured is not None:
            captured.append({"url": url, "headers": headers})
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: payload,
        )

    monkeypatch.setattr("uniinfer.providers.openai_compat.httpx.get", fake_get)


# --- derive_type: Fleet-Audio-Patterns --------------------------------------

def test_derive_type_asr_patterns():
    assert ModelInfo(id="nemotron-3.5-asr-streaming-0.6b").derive_type() == "stt"
    assert ModelInfo(id="step-asr-1.1").derive_type() == "stt"
    assert ModelInfo(id="stepaudio-2.5-asr").derive_type() == "stt"
    assert ModelInfo(id="parakeet-asr-en").derive_type() == "stt"


def test_derive_type_tts_patterns():
    assert ModelInfo(id="supertonic-3").derive_type() == "tts"
    assert ModelInfo(id="qwen3-tts-customvoice").derive_type() == "tts"
    assert ModelInfo(id="voxtral-4b-tts").derive_type() == "tts"
    assert ModelInfo(id="tts-1").derive_type() == "tts"
    assert ModelInfo(id="kokoro-82m").derive_type() == "tts"
    assert ModelInfo(id="piper-thorsten").derive_type() == "tts"


def test_derive_type_whisper_and_modalities():
    assert ModelInfo(id="whisper-large-v3").derive_type() == "stt"
    assert ModelInfo(
        id="x", modalities={"input": ["text"], "output": ["audio"]}
    ).derive_type() == "tts"
    assert ModelInfo(
        id="x", modalities={"input": ["audio"], "output": ["text"]}
    ).derive_type() == "stt"


def test_derive_type_chat_stays_chat():
    # Kein Audio-Pattern — darf nicht anschlagen ( keine False Positives auf Chat-Ids)
    for mid in ("qwen3.6-35b", "LiquidAI/d1-2B", "gpt-5.6-luna",
                "claude-fable-5", "deepseek-v4.1-flash", "glm-5.3-744b"):
        assert ModelInfo(id=mid).derive_type() == "chat", mid


# --- OpenAICompatProvider.list_models ---------------------------------------

def test_list_models_preserves_upstream_owned_by(monkeypatch):
    payload = {"data": [
        {"id": "qwen3.6-35b", "owned_by": "vllm"},
        {"id": "supertonic-3", "owned_by": "sttts"},
        {"id": "LiquidAI/d1-2B"},
    ]}
    _mock_httpx(monkeypatch, payload)
    models = OpenAICompatProvider.list_models(
        api_key="k", base_url="http://127.0.0.1:3001/v1")
    by_id = {m.id: m for m in models}
    assert by_id["qwen3.6-35b"].owned_by == "vllm"
    assert by_id["supertonic-3"].owned_by == "sttts"
    # fehlt upstream → Klassen-Fallback
    assert by_id["LiquidAI/d1-2B"].owned_by == "openai-compat"


def test_list_models_derives_audio_type(monkeypatch):
    """DGX-/v1/models-Shape: audio entries kommen als chat-ähnliche dicts und
    müssen als stt/tts klassifiziert werden, nicht als chat durchrutschen."""
    payload = {"data": [
        {"id": "qwen3.6-35b", "owned_by": "vllm"},
        {"id": "nemotron-3.5-asr-streaming-0.6b", "owned_by": "sttts"},
        {"id": "supertonic-3", "owned_by": "sttts"},
        {"id": "qwen3-tts-customvoice", "owned_by": "vllm"},
    ]}
    _mock_httpx(monkeypatch, payload)
    models = OpenAICompatProvider.list_models(
        api_key="k", base_url="http://127.0.0.1:3001/v1")
    by_id = {m.id: m for m in models}
    assert by_id["qwen3.6-35b"].type == "chat"
    assert by_id["nemotron-3.5-asr-streaming-0.6b"].type == "stt"
    assert by_id["supertonic-3"].type == "tts"
    assert by_id["qwen3-tts-customvoice"].type == "tts"


def test_list_models_bare_string_entries(monkeypatch):
    """Legacy-Endpoints listen bare Strings — Klassen-Fallback bleibt."""
    payload = {"models": ["m1", "m2"]}
    _mock_httpx(monkeypatch, payload)
    models = OpenAICompatProvider.list_models(
        api_key=None, base_url="http://x/v1")
    assert [m.id for m in models] == ["m1", "m2"]
    assert all(m.owned_by == "openai-compat" for m in models)
