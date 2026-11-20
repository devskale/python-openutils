"""decide() — System-One-Decision-Calls via llminvoke (hab@LiquidAI/d1-2B).

Vorfall-Klasse gibberish-digit-mush (2026-11-20): Grauzonen-Adjudikation mit
Wahrscheinlichkeiten statt Completion-JSON-Bettelei. Hier: Auflösung, Payload,
Retry (429 + Retry-After), Erschöpfung, Antworten-Validierung — Transport gemockt.
"""

import sys

import pytest

import llminvoke
from llminvoke import LLMChainExhausted, decide


def _answers(choice="no"):
    return {"garbling": {"type": "choice", "choice": choice,
                         "probabilities": {"fully": 0.02, "partially": 0.08, "no": 0.9}}}


def _ok_response(choice="no"):
    return {"answers": _answers(choice), "usage": {"requests": 1}}


class _Resp:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body.encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_happy_path_liefert_answers(monkeypatch):
    calls = {}

    def _fake_post(url, bearer, payload, timeout=45.0):
        calls.update(url=url, bearer=bearer, payload=payload)
        return _ok_response("partially")

    monkeypatch.setattr(llminvoke, "_systemone_post", _fake_post)
    out = decide("Seiten-Text", {"garbling": {"type": "choice", "criteria": {}}},
                 model="hab@LiquidAI/d1-2B", package="pdf2md")
    assert out["garbling"]["choice"] == "partially"
    assert calls["url"].endswith("/systemone")
    assert calls["payload"]["model"] == "hab@LiquidAI/d1-2B"
    assert calls["payload"]["state"] == "Seiten-Text"


def test_retry_bei_429_dann_erfolg(monkeypatch):
    attempts = {"n": 0}
    sleeps = []

    def _fake_post(url, bearer, payload, timeout=45.0):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("HTTP 429: too many requests")
        return _ok_response("fully")

    monkeypatch.setattr(llminvoke, "_systemone_post", _fake_post)
    monkeypatch.setattr(llminvoke.time, "sleep", lambda s: sleeps.append(s))
    out = decide("x", {}, model="hab@LiquidAI/d1-2B")
    assert out["garbling"]["choice"] == "fully"
    assert attempts["n"] == 2
    assert sleeps, "Backoff muss laufen"


def test_erschoepfung_raist_chain_exhausted(monkeypatch):
    def _always_429(url, bearer, payload, timeout=45.0):
        raise RuntimeError("HTTP 429: nope")

    monkeypatch.setattr(llminvoke, "_systemone_post", _always_429)
    monkeypatch.setattr(llminvoke.time, "sleep", lambda s: None)
    with pytest.raises(LLMChainExhausted):
        decide("x", {}, model="hab@LiquidAI/d1-2B")


def test_leere_answers_sind_fehler_kein_stilles_none(monkeypatch):
    def _bad(url, bearer, payload, timeout=45.0):
        return {"answers": {}, "usage": {}}

    monkeypatch.setattr(llminvoke, "_systemone_post", _bad)
    monkeypatch.setattr(llminvoke.time, "sleep", lambda s: None)
    with pytest.raises(LLMChainExhausted):
        decide("x", {}, model="hab@LiquidAI/d1-2B")


def test_gateway_default_greift(monkeypatch):
    """Ohne Overlay: base_url defaultet auf den uniinfer-Gateway — decide()
    muss daraus {base}/systemone bauen (nicht api.openai o.ä.)."""
    seen = {}

    def _fake_post(url, bearer, payload, timeout=45.0):
        seen["url"] = url
        return _ok_response()

    monkeypatch.setattr(llminvoke, "_systemone_post", _fake_post)
    decide("x", {}, model="hab@LiquidAI/d1-2B")
    assert "/systemone" in seen["url"]
    assert "uniinfer" in seen["url"] or "skale.dev" in seen["url"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def test_task_resolved_model_wenn_kein_explicit(monkeypatch):
    """model=None → resolve_model(package, task).primary — keine Hardcodierung
    am Call-Site, models.yml ist Source of Truth."""
    from llminvoke.config import ResolvedConfig, ModelRef, RetryPolicy

    seen = {}

    def _fake_post(url, bearer, payload, timeout=45.0):
        seen["model"] = payload["model"]
        return _ok_response()

    monkeypatch.setattr(llminvoke, "_systemone_post", _fake_post)
    cfg = ResolvedConfig(primary=ModelRef("hab", "LiquidAI/d1-2B"), retry=RetryPolicy(attempts=1), base_url="https://uniinfer.skale.dev/v1")
    monkeypatch.setattr(llminvoke, "resolve_model", lambda **kw: cfg)
    decide("x", {}, package="pdf2md", task="decision")  # kein model-Param!
    assert seen["model"] == "hab@LiquidAI/d1-2B"
