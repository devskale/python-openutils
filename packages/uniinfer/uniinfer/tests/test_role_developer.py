"""Regression: the OpenAI ``developer`` role is accepted per-provider.

``developer`` is OpenAI's newer role for system instructions (reasoning models;
emitted by the official SDKs and @ai-sdk/openai). The proxy must accept the full
OpenAI role set, but most backends (Mistral, Gemini, Ollama, TU/vLLM, …) reject
``developer`` with 422 — so the proxy collapses it to ``system`` for them, while
preserving it for providers whose API accepts it natively (OpenAI).
"""
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient


def _post_with_developer_role(model: str):
    """POST a developer+user payload; return (response, captured forwarded messages)."""
    from uniinfer.proxy_app import app

    captured: dict = {}

    async def fake_acomplete(messages, **kw):
        captured["messages"] = messages
        return SimpleNamespace(
            message=SimpleNamespace(content="OK", tool_calls=None, role="assistant"),
            thinking=None, finish_reason="stop", usage=None,
        )

    with patch("uniinfer.proxy_routers.chat.verify_provider_access", return_value="k"), \
         patch("uniinfer.proxy_routers.chat.Target") as mt:
        mt.return_value.acomplete = fake_acomplete
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "developer", "content": "You are terse."},
                    {"role": "user", "content": "Say OK"},
                ],
                "max_tokens": 16,
                "stream": False,
            },
        )
    return resp, captured


def test_developer_normalized_to_system_for_mistral():
    """Backends without 'developer' (Mistral) get it collapsed to 'system'."""
    resp, cap = _post_with_developer_role("mistral@mistral-tiny-latest")
    assert resp.status_code == 200, resp.text
    assert [m["role"] for m in cap["messages"]] == ["system", "user"], \
        "developer should be normalized to system for Mistral"


def test_developer_preserved_for_openai():
    """Providers that natively accept 'developer' (OpenAI) keep it as-is."""
    resp, cap = _post_with_developer_role("openai@gpt-4o")
    assert resp.status_code == 200, resp.text
    assert [m["role"] for m in cap["messages"]] == ["developer", "user"], \
        "developer should be preserved for OpenAI"
