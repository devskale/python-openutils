"""reasoning_effort-Transport im TU-Provider (2026-11-09, Aqueduct-Vorfall).

llminvoke `thinking: medium` schreibt top-level reasoning_effort — Aqueduct
ignoriert das still (bzw. 400 je Client). Der TU-Provider hebt graduierte
Werte auf den dokumentierten chat_template_kwargs-Transport — gleiche
Prezedenz wie die OFF-Injektion (Caller-ctk gewinnt).
"""
from __future__ import annotations

from uniinfer.core import ChatCompletionRequest, ChatMessage
from uniinfer.providers.tu import TUProvider


def _payload(reasoning_effort, ctk=None) -> dict:
    p = TUProvider(api_key="x")
    req = ChatCompletionRequest(
        model="qwen-3.8-27b",
        messages=[ChatMessage(role="user", content="hi")],
        max_tokens=100,
        reasoning_effort=reasoning_effort,
        chat_template_kwargs=ctk,
    )
    return p._prepare_payload(req)


def test_graduierte_effort_landet_in_ctk():
    ctk = _payload("medium").get("chat_template_kwargs") or {}
    assert ctk.get("reasoning_effort") == "medium"


def test_off_injektion_bleibt():
    ctk = _payload("none").get("chat_template_kwargs") or {}
    assert ctk.get("enable_thinking") is False


def test_caller_ctk_gewinnt():
    ctk = _payload("medium", {"reasoning_effort": "low"}).get("chat_template_kwargs") or {}
    assert ctk.get("reasoning_effort") == "low"


def test_ohne_effort_kein_ctk_reasoning():
    ctk = _payload(None).get("chat_template_kwargs")
    assert not (ctk or {}).get("reasoning_effort")
