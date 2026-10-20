"""stream_llm(include_reasoning=True) — markierte Stücke statt stummer Denkphase.

stream_llm liefert per Default nur Content-Text (Contract für agentos-Schritte
& Co — Reasoning darf Antworttexte nicht kontaminieren). Wer die Denkphase
live sehen will, musste bisher auf die Tupel-API iter_llm_events ausweichen.
include_reasoning=True liefert dieselben Stücke als StreamPiece — str-Subclass
mit ``.kind`` — damit Consumer join() weiterhin nutzen können und trotzdem auf
reasoning/content verzweigen können.

Hermetisch, kein Netz (create_provider → Fake).
"""

from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llminvoke
from llminvoke import StreamPiece, stream_llm


def _chunk(content=None, reasoning_content=None, thinking=None):
    msg = types.SimpleNamespace(content=content, reasoning_content=reasoning_content)
    return types.SimpleNamespace(message=msg, thinking=thinking, usage=None)


def _usage_chunk():
    return types.SimpleNamespace(
        message=types.SimpleNamespace(content=None, reasoning_content=None),
        thinking=None,
        usage={"prompt_tokens": 3, "completion_tokens": 5},
    )


def _cfg():
    from llminvoke.config import ModelRef, ResolvedConfig, RetryPolicy

    return ResolvedConfig(
        primary=ModelRef(provider="tu", model="qwen-test"),
        base_url="http://x", bearer="k",
        retry=RetryPolicy(attempts=1),
    )


def _patch(monkeypatch, chunks):
    class _FakeProv:
        def stream_complete(self, request):
            yield from chunks

    monkeypatch.setattr(llminvoke, "create_provider", lambda p, **kw: _FakeProv())


def _run(monkeypatch, chunks, **kw):
    _patch(monkeypatch, chunks)
    return list(stream_llm(config=_cfg(),
                           messages=[{"role": "user", "content": "x"}], **kw))


class TestDefaultOhneFlag:
    def test_reasoning_verworfen_content_plain_str(self, monkeypatch):
        out = _run(monkeypatch, [_chunk(thinking="Gedanke "), _chunk(content="Antwort")])
        assert out == ["Antwort"]
        assert all(type(p) is str for p in out)


class TestIncludeReasoning:
    def test_stuecke_markiert_und_geordnet(self, monkeypatch):
        out = _run(monkeypatch, [
            _chunk(thinking="Ich denk "),
            _chunk(thinking="nach."),
            _chunk(content="42"),
            _usage_chunk(),
        ], include_reasoning=True)
        assert [(p.kind, str(p)) for p in out] == [
            ("reasoning", "Ich denk "), ("reasoning", "nach."), ("content", "42")]

    def test_str_contract_erhalten(self, monkeypatch):
        out = _run(monkeypatch,
                   [_chunk(thinking="Gedanke"), _chunk(content="A"), _chunk(content="B")],
                   include_reasoning=True)
        assert "".join(out) == "GedankeAB"
        assert all(isinstance(p, str) for p in out)
        assert isinstance(out[0], StreamPiece)

    def test_denkphase_ab_erstem_chunk(self, monkeypatch):
        """Hilti-Regression: reasoning vom ALLERERSTEN chunk muss fließen."""
        out = _run(monkeypatch, [_chunk(thinking="sofort"), _chunk(content="ok")],
                   include_reasoning=True)
        assert out[0].kind == "reasoning"

    def test_message_reasoning_content_feld(self, monkeypatch):
        """Provider, die reasoning an message.reasoning_content legen."""
        out = _run(monkeypatch, [_chunk(reasoning_content="r1"), _chunk(content="c1")],
                   include_reasoning=True)
        assert [(p.kind, str(p)) for p in out] == [("reasoning", "r1"), ("content", "c1")]
