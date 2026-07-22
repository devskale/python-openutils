"""llminvoke retry/backup/error-classification tests (mock-based, no LLM).

Validates the retry loop in ``call_llm`` → ``_try_model``:
  - transient errors (429/5xx/timeout/network) retry with backoff
  - permanent errors (auth/context/not_found/bad_request) skip retry, walk to backup
  - empty response is a failure (retried, then backup)
  - Retry-After is honored
  - primary exhaustion walks the backup chain; all-fail returns ""

The LLM seam (``invoke_llm``) + ``extract_response_text`` + ``time.sleep`` are
monkeypatched — no network, no real model, no real delay.
"""
from __future__ import annotations

import pytest

import llminvoke
from llminvoke import call_llm
from llminvoke import LLMChainExhausted
from llminvoke.config import ModelRef, ResolvedConfig, RetryPolicy


class FakeResp:
    """Stand-in for a uniinfer response; extract_response_text is mocked to read .text."""
    def __init__(self, text: str) -> None:
        self.text = text


def make_invoke(behaviors: dict):
    """Build a fake invoke_llm keyed by model name.

    ``behaviors`` maps model → list of actions (consumed in order, last repeats):
      ("ok", text) | ("raise", message) | ("empty",)
    """
    calls: list[str] = []
    idx: dict[str, int] = {}

    def _invoke(*, model, provider, messages, **kw):
        calls.append(model)
        actions = behaviors.get(model) or behaviors.get("*", [])
        i = idx.get(model, 0)
        idx[model] = i + 1
        action = actions[min(i, len(actions) - 1)] if actions else ("ok", "")
        if action[0] == "raise":
            raise Exception(action[1])
        return FakeResp("" if action[0] == "empty" else action[1])

    return _invoke, calls


def cfg(primary="p1@m1", backups=(), attempts=3, base_delay=0.0, max_delay=30.0) -> ResolvedConfig:
    return ResolvedConfig(
        primary=ModelRef.parse(primary),
        backups=[ModelRef.parse(b) for b in backups],
        retry=RetryPolicy(attempts=attempts, base_delay=base_delay, max_delay=max_delay),
    )


@pytest.fixture
def patched(monkeypatch):
    """Mock extract_response_text + time.sleep; return (monkeypatch, sleeps)."""
    monkeypatch.setattr(llminvoke, "extract_response_text", lambda r: r.text)
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    return monkeypatch, sleeps


# ── transient errors retry, then succeed ──────────────────────────────

def test_429_rate_limit_retries_then_succeeds(patched):
    mp, sleeps = patched
    inv, calls = make_invoke({"m1": [("raise", "429 too many requests"), ("ok", "done")]})
    mp.setattr(llminvoke, "invoke_llm", inv)
    assert call_llm("x", config=cfg()) == "done"
    assert calls == ["m1", "m1"]          # retried once
    assert len(sleeps) == 1               # backoff before the retry


def test_500_server_error_retries_then_succeeds(patched):
    mp, sleeps = patched
    inv, calls = make_invoke({"m1": [("raise", "500 internal server error"), ("ok", "done")]})
    mp.setattr(llminvoke, "invoke_llm", inv)
    assert call_llm("x", config=cfg()) == "done"
    assert calls == ["m1", "m1"]


def test_timeout_retries_then_succeeds(patched):
    mp, _ = patched
    inv, calls = make_invoke({"m1": [("raise", "connection timed out"), ("ok", "done")]})
    mp.setattr(llminvoke, "invoke_llm", inv)
    assert call_llm("x", config=cfg()) == "done"
    assert calls == ["m1", "m1"]


def test_network_error_retries_then_succeeds(patched):
    mp, _ = patched
    inv, calls = make_invoke({"m1": [("raise", "connection refused"), ("ok", "done")]})
    mp.setattr(llminvoke, "invoke_llm", inv)
    assert call_llm("x", config=cfg()) == "done"
    assert calls == ["m1", "m1"]


# ── permanent errors: no retry, walk to backup immediately ────────────

def test_auth_error_no_retry_walks_to_backup(patched):
    mp, sleeps = patched
    inv, calls = make_invoke({"m1": [("raise", "401 unauthorized")], "m2": [("ok", "backup")]})
    mp.setattr(llminvoke, "invoke_llm", inv)
    assert call_llm("x", config=cfg(backups=("p2@m2",))) == "backup"
    assert calls == ["m1", "m2"]          # m1 once (no retry), then backup
    assert sleeps == []                   # no backoff — permanent → immediate backup


def test_bad_request_no_retry_walks_to_backup(patched):
    mp, sleeps = patched
    inv, calls = make_invoke({"m1": [("raise", "400 bad request")], "m2": [("ok", "backup")]})
    mp.setattr(llminvoke, "invoke_llm", inv)
    assert call_llm("x", config=cfg(backups=("p2@m2",))) == "backup"
    assert calls == ["m1", "m2"]


def test_not_found_no_retry_walks_to_backup(patched):
    mp, sleeps = patched
    inv, calls = make_invoke({"m1": [("raise", "404 model not found")], "m2": [("ok", "backup")]})
    mp.setattr(llminvoke, "invoke_llm", inv)
    assert call_llm("x", config=cfg(backups=("p2@m2",))) == "backup"
    assert calls == ["m1", "m2"]


def test_context_exceeded_no_retry_walks_to_backup(patched):
    mp, sleeps = patched
    inv, calls = make_invoke({"m1": [("raise", "context window exceeded")], "m2": [("ok", "backup")]})
    mp.setattr(llminvoke, "invoke_llm", inv)
    assert call_llm("x", config=cfg(backups=("p2@m2",))) == "backup"
    assert calls == ["m1", "m2"]


# ── backup chain walking ──────────────────────────────────────────────

def test_primary_exhausted_walks_to_backup(patched):
    mp, _ = patched
    inv, calls = make_invoke({"m1": [("raise", "500 internal server error")] * 3, "m2": [("ok", "backup")]})
    mp.setattr(llminvoke, "invoke_llm", inv)
    assert call_llm("x", config=cfg(backups=("p2@m2",), attempts=3)) == "backup"
    assert calls == ["m1", "m1", "m1", "m2"]   # 3 retries on primary, then backup


def test_all_models_fail_returns_empty(patched):
    mp, _ = patched
    inv, _calls = make_invoke({"m1": [("raise", "500")] * 3, "m2": [("raise", "500")] * 3})
    mp.setattr(llminvoke, "invoke_llm", inv)
    assert call_llm("x", config=cfg(backups=("p2@m2",), attempts=3)) == ""


def test_empty_response_retried_then_backup(patched):
    mp, _ = patched
    inv, calls = make_invoke({"m1": [("empty",)] * 3, "m2": [("ok", "backup")]})
    mp.setattr(llminvoke, "invoke_llm", inv)
    assert call_llm("x", config=cfg(backups=("p2@m2",), attempts=3)) == "backup"
    assert calls == ["m1", "m1", "m1", "m2"]   # empty = failure → retried → backup


# ── Retry-After honored ───────────────────────────────────────────────

def test_retry_after_honored(patched):
    mp, sleeps = patched
    inv, _calls = make_invoke({"m1": [("raise", "429 too many requests retry-after: 5"), ("ok", "done")]})
    mp.setattr(llminvoke, "invoke_llm", inv)
    assert call_llm("x", config=cfg(max_delay=30.0)) == "done"
    assert sleeps == [5.0]                # honored Retry-After, not the backoff default


# ── fo-t7: raise_on_empty (silent '' → loud Failure) ──────────────────

def test_raise_on_empty_raises_when_chain_exhausted(patched):
    """All models return empty → call_llm(raise_on_empty=True) raises
    LLMChainExhausted (fo-t7), instead of the silent ''."""
    mp, _sleeps = patched
    inv, _calls = make_invoke({"m1": [("empty",)], "m2": [("empty",)]})
    mp.setattr(llminvoke, "invoke_llm", inv)
    with pytest.raises(LLMChainExhausted):
        call_llm("x", config=cfg(primary="p1@m1", backups=("p2@m2",)), raise_on_empty=True)


def test_raise_on_empty_raises_when_chain_errors(patched):
    """All models error (not just empty) → also raises with raise_on_empty."""
    mp, _sleeps = patched
    inv, _calls = make_invoke({"m1": [("raise", "boom")]})
    mp.setattr(llminvoke, "invoke_llm", inv)
    with pytest.raises(LLMChainExhausted):
        call_llm("x", config=cfg(), raise_on_empty=True)


def test_default_still_returns_empty_string(patched):
    """Without raise_on_empty, a failed chain still returns '' (backward compat)."""
    mp, _sleeps = patched
    inv, _calls = make_invoke({"m1": [("empty",)]})
    mp.setattr(llminvoke, "invoke_llm", inv)
    assert call_llm("x", config=cfg()) == ""


def test_raise_on_empty_not_triggered_on_success(patched):
    """A normal response never raises, even with raise_on_empty=True."""
    mp, _sleeps = patched
    inv, _calls = make_invoke({"m1": [("ok", "hello")]})
    mp.setattr(llminvoke, "invoke_llm", inv)
    assert call_llm("x", config=cfg(), raise_on_empty=True) == "hello"
