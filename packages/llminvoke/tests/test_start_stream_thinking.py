"""_start_stream / _chunk_has_payload — der Thinking-First-Token-Vertrag.

Vorfall „Hilti-Reeval-Synthese 5/5 first-token" (2026-10-06): _start_stream
prüfte nur ``message.content``/``message.reasoning_content``. Der uniinfer-
Contract legt Reasoning aber an ``chunk.thinking`` (TUProvider) — während der
ganzen Denkphase flossen reasoning-Chunks und wurden STILL KONSUMIERT, ohne
als first chunk zurückzukehren. iter_llm_events lieferte kein Ereignis, bis
das Denken endete (exakter Synthese-Prompt: reasoning ab 0.9s, content ab
46.1s) — jede Erst-Event-Frist des Aufrufers (agentos: 20s) feuerte, obwohl
Bytes flossen, und die Denkphase ging als Ereignisse komplett verloren.

Diese Tests pinnen den Drei-Felder-Vertrag fest (content, message.
reasoning_content, chunk.thinking) — hermetisch, kein Netz.
"""

from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llminvoke
from llminvoke import _chunk_has_payload


def _chunk(content=None, reasoning_content=None, thinking=None):
    msg = types.SimpleNamespace(content=content, reasoning_content=reasoning_content)
    return types.SimpleNamespace(message=msg, thinking=thinking, usage=None)


class TestChunkHasPayload:
    def test_content(self):
        assert _chunk_has_payload(_chunk(content="x")) is True

    def test_message_reasoning_content(self):
        assert _chunk_has_payload(_chunk(reasoning_content="denk")) is True

    def test_chunk_thinking(self):
        """DER Vorfall: uniinfer-Contract-Feld chunk.thinking."""
        assert _chunk_has_payload(_chunk(thinking="denk")) is True

    def test_leer(self):
        assert _chunk_has_payload(_chunk()) is False
        assert _chunk_has_payload(_chunk(content="", thinking="")) is False


class TestStartStream:
    def _patch_provider(self, monkeypatch, chunks):
        """create_provider → Fake, dessen stream_complete chunks yielded."""
        recorded = {}

        class _FakeProv:
            def stream_complete(self, request):
                recorded["model"] = request.model
                yield from chunks

        def _factory(provider, **kw):
            return _FakeProv()

        monkeypatch.setattr(llminvoke, "create_provider", _factory)
        return recorded

    def _cfg(self):
        from llminvoke.config import ModelRef, ResolvedConfig, RetryPolicy

        return ResolvedConfig(
            primary=ModelRef(provider="tu", model="tu@qwen-test"),
            base_url="http://x", bearer="k",
            retry=RetryPolicy(attempts=1),
        )

    def test_first_chunk_ist_thinking(self, monkeypatch):
        """Reasoning-Chunk muss SOFORT als first chunk zurückkommen (nicht konsumiert werden)."""
        chunks = [_chunk(), _chunk(thinking="erster Gedanke"), _chunk(content="Antwort")]
        self._patch_provider(monkeypatch, chunks)
        stream, first = llminvoke._start_stream(
            types.SimpleNamespace(provider="tu", model="tu@qwen-test"),
            self._cfg(), [], {},
        )
        assert first.thinking == "erster Gedanke"
        rest = list(stream)  # der Iterator steht NACH dem first chunk
        assert [c.message.content for c in rest] == ["Antwort"]

    def test_reine_content_weiterhin_ok(self, monkeypatch):
        chunks = [_chunk(), _chunk(content="ok")]
        self._patch_provider(monkeypatch, chunks)
        _, first = llminvoke._start_stream(
            types.SimpleNamespace(provider="tu", model="tu@qwen-test"),
            self._cfg(), [], {},
        )
        assert first.message.content == "ok"

    def test_nur_leere_chunks_empty_response(self, monkeypatch):
        self._patch_provider(monkeypatch, [_chunk(), _chunk()])
        import pytest

        with pytest.raises(RuntimeError, match="empty_response"):
            llminvoke._start_stream(
                types.SimpleNamespace(provider="tu", model="tu@qwen-test"),
                self._cfg(), [], {},
            )
