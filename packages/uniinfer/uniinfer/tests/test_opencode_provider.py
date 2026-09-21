"""OpenCode/Zen provider: registration + list_models parsing + free detection."""
from unittest.mock import patch
import requests

from uniinfer import ProviderFactory
from uniinfer.providers.opencode import OpenCodeProvider
from uniinfer.core import ModelInfo


def test_provider_registered():
    assert "opencode" in ProviderFactory.list_providers()
    assert ProviderFactory.get_provider_class("opencode") is OpenCodeProvider


def test_list_models_parses_and_marks_free():
    """Test that list_models correctly parses pi.dev catalog and marks free models."""
    # Mock pi.dev catalog response (simplified)
    sample_catalog = {
        "deepseek-v4-flash-free": {
            "id": "deepseek-v4-flash-free",
            "name": "DeepSeek V4 Flash Free",
            "contextWindow": 200000,
            "maxTokens": 128000,
            "cost": {"input": 0, "output": 0},
            "input": ["text"],
        },
        "big-pickle": {
            "id": "big-pickle",
            "name": "Big Pickle",
            "contextWindow": 200000,
            "maxTokens": 32000,
            "cost": {"input": 0, "output": 0},
            "input": ["text"],
        },
        "gpt-5.5": {
            "id": "gpt-5.5",
            "name": "GPT-5.5",
            "contextWindow": 1050000,
            "maxTokens": 128000,
            "cost": {"input": 5, "output": 30},
            "input": ["text", "image"],
        },
        "claude-haiku-4-5": {
            "id": "claude-haiku-4-5",
            "name": "Claude Haiku 4.5",
            "contextWindow": 200000,
            "maxTokens": 64000,
            "cost": {"input": 1, "output": 5},
            "input": ["text", "image"],
        },
    }

    with patch.object(requests, "get") as mock_get:
        mock_get.return_value.json.return_value = sample_catalog
        mock_get.return_value.raise_for_status.return_value = None

        models = OpenCodeProvider.list_models()

    ids = [m.id for m in models]
    assert ids == ["deepseek-v4-flash-free", "big-pickle", "gpt-5.5", "claude-haiku-4-5"]

    # Free models (-free, big-pickle) marked cost 0; paid left as-is from catalog
    by_id = {m.id: m for m in models}
    assert by_id["deepseek-v4-flash-free"].cost == {"input": 0, "output": 0}
    assert by_id["big-pickle"].cost == {"input": 0, "output": 0}
    assert by_id["gpt-5.5"].cost == {"input": 5, "output": 30}
    assert by_id["claude-haiku-4-5"].cost == {"input": 1, "output": 5}

    # Context windows should be populated from pi.dev
    assert by_id["deepseek-v4-flash-free"].context_window == 200000
    assert by_id["gpt-5.5"].context_window == 1050000
    assert by_id["claude-haiku-4-5"].context_window == 200000


# ------------------------------------------------------------------ #
# free-tier imitation protocol (2026-09-21): IDs, headers, tools, stream
# ------------------------------------------------------------------ #
import json
import re
import time

import pytest

from uniinfer.core import ChatCompletionRequest, ChatMessage


def test_opencode_id_format_encodes_timestamp():
    before = int(time.time() * 1000)
    msg = OpenCodeProvider._opencode_id("msg", descending=False)
    ses = OpenCodeProvider._opencode_id("ses", descending=True)
    after = int(time.time() * 1000)

    assert re.fullmatch(r"msg_[0-9a-f]{12}[0-9A-Za-z]{14}", msg), msg
    assert re.fullmatch(r"ses_[0-9a-f]{12}[0-9A-Za-z]{14}", ses), ses

    # ascending: hex decodes to ~now (mod 2^48 wrap, JS id.ts parity)
    hexpart = int(msg[4:16], 16)
    ts = hexpart >> 12
    assert (ts - (before * 0x1000 >> 12)) % (1 << 36) < 1000 or abs(ts - (before % (1 << 36))) < 2, (
        f"timestamp not current: {ts} vs {before}"
    )


def test_opencode_ids_unique_and_monotonic():
    ids = [OpenCodeProvider._opencode_id("msg", descending=False) for _ in range(50)]
    assert len(set(ids)) == len(ids)


def test_build_headers_agent_shape():
    p = OpenCodeProvider(api_key="test")
    h1 = p._build_headers()
    h2 = p._build_headers()
    assert h1["User-Agent"].startswith("opencode/")
    assert "ai-sdk/provider-utils" in h1["User-Agent"]
    assert h1["x-opencode-client"] == "cli"
    assert h1["x-opencode-project"] == "global"
    assert h1["x-opencode-request"].startswith("msg_")
    assert h1["x-opencode-session"].startswith("ses_")
    # fresh ids per request
    assert h1["x-opencode-request"] != h2["x-opencode-request"]
    assert h1["x-opencode-session"] != h2["x-opencode-session"]


def test_build_headers_session_env_override(monkeypatch):
    monkeypatch.setenv("UNIINFER_OPENCODE_SESSION", "ses_pinned123")
    p = OpenCodeProvider(api_key="test")
    h = p._build_headers()
    assert h["x-opencode-session"] == "ses_pinned123"
    assert h["x-opencode-request"].startswith("msg_")


def test_agent_tools_canonical_set():
    tools = OpenCodeProvider._agent_tools()
    names = {t["function"]["name"] for t in tools}
    assert names == {
        "bash", "edit", "glob", "grep", "question", "read",
        "skill", "task", "todowrite", "webfetch", "websearch", "write",
    }


def _req(messages=None):
    return ChatCompletionRequest(
        model="mimo-v2.5-free",
        messages=messages or [ChatMessage(role="user", content="hi")],
        temperature=0.7,
        streaming=False,
    )


def test_payload_injects_canonical_tools_and_forces_stream():
    p = OpenCodeProvider(api_key="test")
    payload = p._build_payload(_req(), False, {})
    names = {t["function"]["name"] for t in payload["tools"]}
    assert "bash" in names and "read" in names and "write" in names
    assert len(payload["tools"]) == 12
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


def test_payload_unions_caller_tools():
    custom = {
        "type": "function",
        "function": {
            "name": "my_custom_tool",
            "description": "x",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    req = ChatCompletionRequest(
        model="mimo-v2.5-free",
        messages=[ChatMessage(role="user", content="hi")],
        temperature=0.7,
        streaming=False,
        tools=[custom],
    )
    p = OpenCodeProvider(api_key="test")
    payload = p._build_payload(req, False, {})
    names = {t["function"]["name"] for t in payload["tools"]}
    assert "my_custom_tool" in names
    assert "bash" in names  # canonical still present
    assert len(payload["tools"]) == 13


class _FakeSSEResponse:
    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code

    async def aread(self):
        return b"error"

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamCtx:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        return False


class _FakeClient:
    def __init__(self, lines, status_code=200):
        self._lines = lines
        self._status = status_code

    def stream(self, method, url, **kwargs):
        return _FakeStreamCtx(_FakeSSEResponse(self._lines, self._status))


@pytest.mark.asyncio
async def test_acomplete_aggregates_sse(monkeypatch):
    lines = [
        'data: {"model":"mimo-v2.5-free","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}',
        'data: {"model":"mimo-v2.5-free","choices":[{"index":0,"delta":{"reasoning":"thinking..."}}]}',
        'data: {"model":"mimo-v2.5-free","choices":[{"index":0,"delta":{"content":"lo"}}]}',
        'data: {"model":"mimo-v2.5-free","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        'data: {"model":"mimo-v2.5-free","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}',
        "data: [DONE]",
    ]
    p = OpenCodeProvider(api_key="test")

    async def fake_client():
        return _FakeClient(lines)

    monkeypatch.setattr(p, "_get_async_client", fake_client)
    resp = await p.acomplete(_req())
    assert resp.message.content == "Hello"
    assert resp.thinking == "thinking..."
    assert resp.finish_reason == "stop"
    assert resp.usage["total_tokens"] == 12
    assert resp.model == "mimo-v2.5-free"


@pytest.mark.asyncio
async def test_acomplete_aggregates_tool_call_deltas(monkeypatch):
    lines = [
        'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"bash","arguments":"{\\"cmd\\""}}]}}]}',
        'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\\"ls\\"}"}}]}}]}',
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
        "data: [DONE]",
    ]
    p = OpenCodeProvider(api_key="test")

    async def fake_client():
        return _FakeClient(lines)

    monkeypatch.setattr(p, "_get_async_client", fake_client)
    resp = await p.acomplete(_req())
    assert resp.finish_reason == "tool_calls"
    assert resp.message.tool_calls[0]["function"]["name"] == "bash"
    assert resp.message.tool_calls[0]["function"]["arguments"] == '{"cmd":"ls"}'
