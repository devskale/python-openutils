"""call_llm_with_meta: which model actually served the call.

The gateway echoes the concrete model in the response body (auto@free
resolves proxy-side; fallbacks can serve a backup) — meta.model must carry
that echo, fall back to the requested ref when absent, and be None when the
whole chain fails. Mock-based, no LLM (same seams as test_retry_backup).
"""
from __future__ import annotations

import pytest

import llminvoke
from llminvoke import call_llm_with_meta
from llminvoke.config import ModelRef, ResolvedConfig, RetryPolicy


class FakeResp:
    def __init__(self, text: str, model: str | None = None, usage: dict | None = None) -> None:
        self.text = text
        if model is not None:
            self.model = model
        self.usage = usage


def cfg(primary="p1@m1", backups=()) -> ResolvedConfig:
    return ResolvedConfig(
        primary=ModelRef.parse(primary),
        backups=[ModelRef.parse(b) for b in backups],
        retry=RetryPolicy(attempts=1, base_delay=0.0),
    )


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(llminvoke, "extract_response_text", lambda r: r.text)
    monkeypatch.setattr("time.sleep", lambda s: None)
    return monkeypatch


def test_meta_carries_gateway_echo(patched):
    """Gateway-Echo gewinnt: auto@free served mistral@… — meta zeigt es."""
    mp = patched
    mp.setattr(
        llminvoke,
        "invoke_llm",
        lambda **kw: FakeResp("hi", model="mistral@codestral-2508", usage={"total_tokens": 3}),
    )
    text, meta = call_llm_with_meta("x", config=cfg(primary="auto@free"))
    assert text == "hi"
    assert meta["requested"] == "auto@free"
    assert meta["model"] == "mistral@codestral-2508"
    assert meta["provider"] == "mistral"
    assert meta["usage"] == {"total_tokens": 3}


def test_meta_falls_back_to_requested_when_no_echo(patched):
    mp = patched
    mp.setattr(llminvoke, "invoke_llm", lambda **kw: FakeResp("hi"))
    _, meta = call_llm_with_meta("x", config=cfg())
    assert meta["model"] == "p1@m1"
    assert meta["provider"] == "p1"


def test_meta_shows_backup_that_served(patched):
    """Backup-Fall: primary timed out → backup antwortet. requested = das
    Kettenglied, das den Call bekam; model = sein Echo."""
    mp = patched

    def _invoke(*, model, provider, **kw):
        if model == "m1":
            raise Exception("timeout")
        return FakeResp("from-backup", model="p2@m2")

    mp.setattr(llminvoke, "invoke_llm", _invoke)
    _, meta = call_llm_with_meta("x", config=cfg(backups=("p2@m2",)))
    assert meta is not None
    assert meta["requested"] == "p2@m2"
    assert meta["model"] == "p2@m2"


def test_meta_none_when_chain_exhausted(patched):
    mp = patched
    mp.setattr(llminvoke, "invoke_llm", lambda **kw: (_ for _ in ()).throw(Exception("429")))
    text, meta = call_llm_with_meta("x", config=cfg(backups=("p2@m2",)))
    assert text == ""
    assert meta is None
