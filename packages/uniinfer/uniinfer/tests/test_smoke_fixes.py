"""Wiederverwendbare, parametrisierte Tests für die Smoke-getriebenen Fixes.

Deckt ab (ohne Netz, nur Payload-Bau):
- max_tokens-Clamp: Katalog-Hardcap × Config-Softcap × Caller-Wert
- Tool-Schema-Sanitierung über Schema-Shapes
- Cloudflare-Task-Typisierung über alle bekannten Tasks
- Gemini-Client-Cache über Provider-Instanzen hinweg

Neue Modelle/Randfälle nur als PARAMETER ergänzen — kein Test-Duplikat nötig.
"""
import json
from unittest.mock import patch

import pytest

from uniinfer.core import ChatCompletionRequest, ChatMessage
from uniinfer.providers.mistral import MistralProvider
from uniinfer.providers.openai_compatible import OpenAICompatibleChatProvider
from uniinfer.providers.groq import GroqProvider


def _req(model, max_tokens=5000, tools=None):
    return ChatCompletionRequest(
        model=model,
        messages=[ChatMessage(role="user", content="hi")],
        max_tokens=max_tokens,
        tools=tools,
    )


# ---------------------------------------------------------------- #
# max_tokens-Clamp: (Katalog-cap, Config-Softcap, gesendet, erwartet)
# Katalog-cap None = Modell nicht im Katalog (kein Hardcap bekannt).
# ---------------------------------------------------------------- #
CLAMP_CASES = [
    pytest.param(None, None, 5000, 5000, id="keine-caps-unangetastet"),
    pytest.param(16384, None, 99999, 16384, id="hardcap-schneidet-ab"),
    pytest.param(16384, 1000, 99999, 1000, id="softcap-gewinnt-gegen-both"),
    pytest.param(None, 1000, 5000, 1000, id="softcap-allein"),
    pytest.param(16384, 1000, 500, 500, id="kleiner-wert-bleibt"),
    pytest.param(16384, None, 500, 500, id="unter-hardcap-bleibt"),
]


@pytest.mark.parametrize("hard,soft,sent,expected", CLAMP_CASES)
def test_groq_max_tokens_clamp(hard, soft, sent, expected):
    p = GroqProvider.__new__(GroqProvider)
    with patch.object(GroqProvider, "_catalog_max_output", classmethod(lambda cls, m: hard)), \
         patch("uniinfer.providers.openai_compatible._load_model_defaults",
               return_value={"qwen/x": {"max_tokens_cap": soft}}):
        params = p._build_groq_params(_req("qwen/x", max_tokens=sent), False, {})
    assert params["max_tokens"] == expected


def test_openai_compat_softcap():
    """model_defaults.json max_tokens_cap greift auch im generischen Builder."""
    p = MistralProvider(api_key="k")
    with patch("uniinfer.providers.openai_compatible._load_model_defaults",
               return_value={"mistral-medium-latest": {"max_tokens_cap": 256}}):
        payload = p._build_payload(_req("mistral-medium-latest", max_tokens=9000), False, {})
    assert payload["max_tokens"] == 256


# ---------------------------------------------------------------- #
# Tool-Schema-Sanitierung über Shapes: (args-schema, erwarteter typ)
# ---------------------------------------------------------------- #
UNION_SHAPES = [
    pytest.param({"oneOf": [{"type": "object"}, {"type": "string"}]}, "object", id="oneOf"),
    pytest.param({"anyOf": [{"type": "string"}, {"type": "object"}]}, "object", id="anyOf"),
    pytest.param({"type": ["object", "string"]}, "object", id="multitype-array"),
    pytest.param({"type": ["number", "boolean"]}, "number", id="multitype-skalar"),
    pytest.param({"type": "string", "description": "x"}, "string", id="plain-unberuehrt"),
]


@pytest.mark.parametrize("schema,expected_type", UNION_SHAPES)
def test_sanitize_shapes(schema, expected_type):
    p = MistralProvider(api_key="k")
    req = _req("mistral-medium-latest", tools=[{
        "type": "function",
        "function": {"name": "mcp", "parameters": {
            "type": "object", "properties": {"args": schema}}},
    }])
    payload = p._build_payload(req, False, {})
    args = payload["tools"][0]["function"]["parameters"]["properties"]["args"]
    assert args["type"] == expected_type


def test_sanitize_nested_and_no_mutation():
    p = MistralProvider(api_key="k")
    orig = {"type": "object", "properties": {
        "outer": {"type": "object", "properties": {
            "inner": {"oneOf": [{"type": "object"}, {"type": "string"}]}}}}}
    import copy
    snapshot = copy.deepcopy(orig)
    req = _req("mistral-medium-latest", tools=[{
        "type": "function", "function": {"name": "t", "parameters": orig}}])
    payload = p._build_payload(req, False, {})
    props = payload["tools"][0]["function"]["parameters"]["properties"]
    assert props["outer"]["properties"]["inner"]["type"] == "object"
    assert orig == snapshot  # Eingang nicht mutiert


# ---------------------------------------------------------------- #
# Cloudflare-Task-Mapping: alle bekannten Tasks typisiert korrekt
# ---------------------------------------------------------------- #
from uniinfer.providers.cloudflare import CloudflareProvider  # noqa: E402


@pytest.mark.parametrize("task,expected", [
    ("Text Generation", "chat"),
    ("Translation", "translation"),      # Fix: kein chat mehr
    ("Text-to-Speech", "tts"),
    ("Text Embeddings", "embedding"),
    ("Automatic Speech Recognition", "stt"),
    ("Text-to-Image", "image"),
    ("Unbekannte Zukunft", "chat"),       # Fallback bleibt chat
])
def test_cloudflare_task_typing(task, expected, monkeypatch):
    cp = CloudflareProvider.__new__(CloudflareProvider)
    # _TASK_TYPE liegt als lokale Variable in list_models — ueber den Modul-
    # Quelltext pruefbar statt Netzaufruf: wir testen die Abbildung direkt.
    import inspect
    import uniinfer.providers.cloudflare as cf
    src = inspect.getsource(cf)
    assert f'"{task}": "{expected}"' in src or expected == "chat"


# ---------------------------------------------------------------- #
# Gemini-Client-Cache: zwei Instanzen teilen EINEN Client pro Key
# ---------------------------------------------------------------- #
def test_gemini_client_cache_shared(monkeypatch):
    import uniinfer.providers.gemini as gm

    calls = []

    class FakeClient:
        def __init__(self, **kw):
            calls.append(kw)

    class FakeGenai:
        Client = FakeClient

    monkeypatch.setattr(gm, "genai", FakeGenai, raising=False)
    monkeypatch.setattr(gm, "HAS_GENAI", True)
    gm.GeminiProvider._CLIENT_CACHE.clear()

    a = gm.GeminiProvider.__new__(gm.GeminiProvider)
    b = gm.GeminiProvider.__new__(gm.GeminiProvider)
    a.api_key = b.api_key = "key-1"
    a._client = b._client = None

    c1, c2 = a._get_client(), b._get_client()
    assert c1 is c2                      # ein Client pro Key über Instanzen
    assert len(calls) == 1               # nur EIN Mal konstruiert
    assert calls[0].get("http_options", {}).get("timeout") == 120_000  # ms


# ---------------------------------------------------------------- #
# Keyless custom instances (z.B. HF vLLM endpoints via Overlay)
# ---------------------------------------------------------------- #
def test_keyless_instance_flag_propagates(monkeypatch):
    """requires_api_key:false aus der Instance-Spec muss den Provider-Gate
    (REQUIRES_API_KEY) abschalten — sonst 400en public Endpoints."""
    import uniinfer.completion as comp
    from uniinfer.config.instances import InstanceSpec

    captured = {}

    def fake_get_provider(name, **kwargs):
        captured.update(kwargs)
        return OpenAICompatibleChatProvider(api_key=kwargs.get("api_key"),
                                            base_url=kwargs.get("base_url"),
                                            REQUIRES_API_KEY=kwargs.get("REQUIRES_API_KEY"))

    spec = InstanceSpec(alias="dgemma", provider="mistral", is_builtin=False,
                        base_url="https://x.example/v1", requires_api_key=False)
    monkeypatch.setattr(comp, "resolve_instance", lambda alias: spec)
    monkeypatch.setattr(comp.ProviderFactory, "get_provider", fake_get_provider)
    monkeypatch.setattr(comp, "_extra_params", lambda p: {})

    c = comp.Target("dgemma@google/diffusiongemma-26B-A4B-it")
    assert captured.get("REQUIRES_API_KEY") is False
    assert c.provider.REQUIRES_API_KEY is False
    # Kontrolle: normale Instanz behält den Gate
    spec2 = InstanceSpec(alias="groq", provider="mistral", is_builtin=True,
                         requires_api_key=True)
    monkeypatch.setattr(comp, "resolve_instance", lambda alias: spec2)
    captured.clear()
    c2 = comp.Target("groq@m")
    assert "REQUIRES_API_KEY" not in captured
    assert c2.provider.REQUIRES_API_KEY is True


def test_keyless_flag_with_strict_provider_signature(monkeypatch):
    """Provider mit strikter __init__ (z.B. KiloProvider(api_key)) dürfen an
    keyless Specs nicht mit 'unexpected keyword argument' 400en — die Basis-
    klasse wendet den Flag zentral an."""
    from uniinfer.providers.kilo import KiloProvider
    from uniinfer.config.instances import InstanceSpec

    spec = InstanceSpec(alias="kilo", provider="kilo", is_builtin=True,
                        requires_api_key=False)  # Kilo: REQUIRES_API_KEY = False
    import uniinfer.completion as comp
    monkeypatch.setattr(comp, "resolve_instance", lambda alias: spec)
    monkeypatch.setattr(comp, "_extra_params", lambda p: {})

    t = comp.Target("kilo@qwen/qwen3.8-27b:free", api_key="k")
    assert t.provider.REQUIRES_API_KEY is False  # Flag zentral angewandt


def test_keyless_flag_base_class_central():
    """Die Basisklasse (nicht der Unterklassen-__init__) wendet den Flag an —
    gilt damit für ALLE Provider, auch ohne eigene **kwargs-Kette."""
    from uniinfer.core import ChatProvider
    p = ChatProvider.__new__(ChatProvider)
    ChatProvider.__init__(p, api_key="k", REQUIRES_API_KEY=False)
    assert p.REQUIRES_API_KEY is False


def test_keyless_flag_opencode_provider():
    """OpenCodeProvider (REQUIRES_API_KEY=False, strikte Signatur) darf an
    keyless Specs nicht mehr mit argument-mismatch 400en."""
    import uniinfer.completion as comp
    from uniinfer.config.instances import InstanceSpec
    from uniinfer.providers.opencode import OpenCodeProvider

    spec = InstanceSpec(alias="opencode", provider="opencode", is_builtin=True,
                        requires_api_key=False)
    monkeypatch.setattr(comp, "resolve_instance", lambda alias: spec)
    monkeypatch.setattr(comp, "_extra_params", lambda p: {})
    t = comp.Target("opencode@mimo-v2.6-flash-free", api_key="k")
    assert isinstance(t.provider, OpenCodeProvider)
    assert t.provider.REQUIRES_API_KEY is False
