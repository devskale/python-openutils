"""GLM-5.x /nothink-Soft-Switch im TU-Provider (live verifiziert 2026-11-02).

Aqueduct honoriert für glm-5.x weder ``enable_thinking`` noch ``thinking``
(Reasoning streamt trotzdem — Content erst nach 60s+), während der
Prompt-Switch ``/nothink`` sofort wirkt (0,8s TTFT). Der Provider pinnt ihn
zusätzlich zur ersten User-Message, WENN Reasoning-off angefragt ist —
nur für glm-5.x (qwen/deepseek lesen die Knobs).
"""

from __future__ import annotations

from uniinfer.core import ChatCompletionRequest, ChatMessage
from uniinfer.providers.tu import TUProvider


def _payload(model: str, **kw) -> dict:
    provider = TUProvider(api_key="test-key")
    req = ChatCompletionRequest(
        model=model,
        messages=[ChatMessage(role="user", content="Sag: OK")],
        max_tokens=32,
        **kw,
    )
    return provider._prepare_payload(req)


def test_glm5_reasoning_off_prepends_nothink():
    p = _payload("glm-5.3-744b-experimental", chat_template_kwargs={"enable_thinking": False})
    first_user = p["messages"][0]
    assert first_user["content"].startswith("/nothink\n")
    assert "Sag: OK" in first_user["content"]
    # Knob bleibt zusätzlich drin (Familien, die ihn lesen)
    assert p["chat_template_kwargs"]["enable_thinking"] is False


def test_glm5_reasoning_effort_none_prepends_nothink():
    p = _payload("glm-5.3-744b-experimental", reasoning_effort="none")
    assert p["messages"][0]["content"].startswith("/nothink\n")


def test_glm5_ohne_reasoning_off_uneaingegriffen():
    p = _payload("glm-5.3-744b-experimental")
    assert p["messages"][0]["content"] == "Sag: OK"


def test_qwen_bekommt_keinen_prompt_switch():
    # qwen liest enable_thinking — kein /nothink im Prompt
    p = _payload("qwen-3.6-35b", chat_template_kwargs={"enable_thinking": False})
    assert p["messages"][0]["content"] == "Sag: OK"
    assert p["chat_template_kwargs"]["enable_thinking"] is False


def test_nothink_nicht_doppelt():
    provider = TUProvider(api_key="test-key")
    req = ChatCompletionRequest(
        model="glm-5.3-744b-experimental",
        messages=[ChatMessage(role="user", content="/nothink\nSag: OK")],
        max_tokens=32,
        chat_template_kwargs={"enable_thinking": False},
    )
    p = provider._prepare_payload(req)
    assert p["messages"][0]["content"].count("/nothink") == 1


def test_system_prompt_bleibt_user_wird_geswitched():
    provider = TUProvider(api_key="test-key")
    req = ChatCompletionRequest(
        model="glm-5.3-744b-experimental",
        messages=[
            ChatMessage(role="system", content="Du bist ein Extraktor."),
            ChatMessage(role="user", content="Extrahiere: X"),
        ],
        max_tokens=32,
        chat_template_kwargs={"enable_thinking": False},
    )
    p = provider._prepare_payload(req)
    assert p["messages"][0]["content"] == "Du bist ein Extraktor."
    assert p["messages"][1]["content"].startswith("/nothink\nExtrahiere")
