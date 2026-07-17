"""OpenAI passthrough: unmapped OpenAI params reach the backend, not dropped.

The proxy is a passthrough for OpenAI params it doesn't model explicitly
(top_p, response_format, seed, stream_options, logprobs, …) so new OpenAI
features flow to OpenAI-compatible backends without a per-field code change.
Captured via schema extra="allow" → request.extra → provider payload.
"""
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from uniinfer import ChatCompletionRequest, ChatMessage
from uniinfer.providers.mistral import MistralProvider


def test_payload_forwards_request_extra():
    """_build_payload merges request.extra into the outbound payload."""
    provider = MistralProvider(api_key="k")
    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="hi")],
        model="mistral-tiny-latest",
        temperature=0.5,
        extra={"top_p": 0.1, "seed": 42, "response_format": {"type": "json_object"}},
    )
    payload = provider._build_payload(req, False, {})
    assert payload["top_p"] == 0.1
    assert payload["seed"] == 42
    assert payload["response_format"] == {"type": "json_object"}
    # standard modeled fields are unaffected
    assert payload["model"] == "mistral-tiny-latest"
    assert payload["temperature"] == 0.5


def test_extra_forward_denylist_strips():
    """A provider's EXTRA_FORWARD_DENY removes a field even if a client sent it."""

    class Strict(MistralProvider):
        EXTRA_FORWARD_DENY = frozenset({"seed"})

    payload = Strict(api_key="k")._build_payload(
        ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hi")],
            model="x",
            extra={"seed": 42, "top_p": 0.1},
        ),
        False,
        {},
    )
    assert "seed" not in payload
    assert payload["top_p"] == 0.1


def test_proxy_forwards_unknown_openai_params():
    """An OpenAI client sending unmapped params has them passed through as `extra`."""
    from uniinfer.proxy_app import app

    captured: dict = {}

    async def fake_acomplete(messages, **kw):
        captured.update(kw)
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
                "model": "mistral@mistral-tiny-latest",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
                "top_p": 0.1,
                "seed": 42,
                "response_format": {"type": "json_object"},
            },
        )

    assert resp.status_code == 200, resp.text
    extra = captured.get("extra", {})
    assert extra.get("top_p") == 0.1
    assert extra.get("seed") == 42
    assert extra.get("response_format") == {"type": "json_object"}
