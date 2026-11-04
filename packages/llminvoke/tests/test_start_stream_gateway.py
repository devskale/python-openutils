"""_start_stream Gateway-Zweig — Alias-only-Provider (hab) müssen streamen.

Vorfall 2026-11-04 (``dgx-hab-episoden``): der Streaming-Pfad rief auch mit
konfiguriertem Gateway ``create_provider(ref.provider, base_url=...)`` auf —
die ProviderFactory resolved per NAME und wirft für Instanz-Aliase (hab)
ValueError "not registered", egal ob ein base_url mitgegeben wird. Der
Primary-Kettenplatz verbrannte pro Step an diesem Fehler plus 20–30 s an
Backups mit toten Upstreams (ttfb 32–44 s, „Trägheit"-Episode). Der Fix: mit
``cfg.base_url`` streamt der Pfad via OpenAI-SDK gegen den Gateway (routing
per provider@model-Präfix) — gleiche Semantik wie ``_invoke_via_openai_sdk``
auf dem Non-Stream-Pfad.

Diese Tests pinnen den Zweig fest — hermetisch, kein Netz.
"""

from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llminvoke
from llminvoke import ChatMessage, ModelRef, _start_stream, _stream_via_openai_sdk
from llminvoke.config import ResolvedConfig, RetryPolicy


def _cfg(base_url=None, bare_model=False, attempts=1):
    return ResolvedConfig(
        primary=ModelRef(provider="hab", model="qwen3.6-35b"),
        backups=[],
        temperature=0.7,
        max_tokens=128,
        retry=RetryPolicy(attempts=attempts),
        dsgvo_required=False,
        request_kwargs={},
        base_url=base_url,
        bare_model=bare_model,
        bearer="tok",
    )


def _sdk_chunks():
    """Fake raw OpenAI-SDK-Stream: delta-chunks + terminaler usage-chunk."""
    def delta(content=None, reasoning_content=None):
        d = types.SimpleNamespace(content=content, reasoning_content=reasoning_content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=d)], usage=None)

    usage = types.SimpleNamespace(
        model_dump=lambda: {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    )
    return [
        delta(),                                    # role-delta, kein Payload
        delta(content="{ \"a\": 1"),
        delta(content=" }"),
        types.SimpleNamespace(choices=[], usage=usage),  # terminal usage
    ]


class TestStartStreamGateway:
    def test_alias_provider_streams_via_gateway(self, monkeypatch):
        """DER Vorfall: hab@… + cfg.base_url → SDK-Pfad, kein create_provider."""
        def _no_direct(provider, **kw):
            raise AssertionError("direct provider path must not run with base_url set")

        monkeypatch.setattr(llminvoke, "create_provider", _no_direct)

        captured = {}

        def _fake_sdk(**kw):
            captured.update(kw)
            return iter(  # bereits adaptierte Chunks (Duck-Typ-Vertrag)
                [types.SimpleNamespace(
                    message=types.SimpleNamespace(content="x", reasoning_content=None),
                    thinking=None, usage=None)]
            )

        monkeypatch.setattr(llminvoke, "_stream_via_openai_sdk", _fake_sdk)
        stream, first = _start_stream(
            ModelRef(provider="hab", model="qwen3.6-35b"),
            _cfg(base_url="https://gw.example/v1"),
            [ChatMessage(role="user", content="hi")],
            {},
        )
        assert first.message.content == "x"
        assert captured["provider"] == "hab"
        assert captured["base_url"] == "https://gw.example/v1"
        assert captured["bearer"] == "tok"

    def test_empty_gateway_stream_raises_empty_response(self, monkeypatch):
        monkeypatch.setattr(llminvoke, "_stream_via_openai_sdk", lambda **kw: iter([]))
        try:
            _start_stream(
                ModelRef(provider="hab", model="qwen3.6-35b"),
                _cfg(base_url="https://gw.example/v1"),
                [ChatMessage(role="user", content="hi")],
                {},
            )
            raise AssertionError("expected empty_response")
        except RuntimeError as exc:
            assert "empty_response" in str(exc)

    def test_direct_path_without_base_url_unchanged(self):
        """Ohne Gateway: unbekannter Provider wirft weiterhin 'not registered'."""
        try:
            _start_stream(
                ModelRef(provider="hab", model="qwen3.6-35b"),
                _cfg(),
                [ChatMessage(role="user", content="hi")],
                {},
            )
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "not registered" in str(exc)


class TestSdkStreamAdapter:
    def test_adapt_maps_deltas_reasoning_and_usage(self, monkeypatch):
        fake = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(
            create=lambda **kw: _sdk_chunks())))
        monkeypatch.setattr("openai.OpenAI", lambda **kw: fake, raising=False)

        import llminvoke as li
        chunks = list(_stream_via_openai_sdk(
            base_url="https://gw.example/v1", bearer="tok",
            provider="hab", model="qwen3.6-35b",
            messages=[ChatMessage(role="user", content="hi")],
            temperature=0.7, max_tokens=64, request_kwargs={},
        ))
        assert len(chunks) == 4
        assert chunks[0].message.content is None          # role-delta durchgereicht
        assert chunks[1].message.content == '{ "a": 1'
        assert chunks[3].usage["total_tokens"] == 18      # terminaler usage-chunk
        assert llminvoke._chunk_has_payload(chunks[1])    # Vertrag: payload erkennbar
