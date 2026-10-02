"""OpenCode/Zen provider: dynamic catalog, dialect routing, free-tier protocol."""
from unittest.mock import AsyncMock, MagicMock, patch
import json
import re
import time

import pytest
import httpx
import requests

import uniinfer.providers.opencode as oc_module
from uniinfer import ProviderFactory
from uniinfer.providers.opencode import OpenCodeProvider, _parse_docs_mdx
from uniinfer.providers.openai_compatible import normalize_tool_history
from uniinfer.providers.kilo import KiloProvider
from uniinfer.core import ChatCompletionRequest, ChatMessage, ModelInfo


MDX_SAMPLE = """# Zen

## Endpoints

| Model | Model ID | Endpoint | AI SDK Package |
| --- | --- | --- | --- |
| Big Pickle | big-pickle | `https://opencode.ai/zen/v1/chat/completions` | `@ai-sdk/openai-compatible` |
| MiMo-V2.5 Free | mimo-v2.5-free | `https://opencode.ai/zen/v1/chat/completions` | `@ai-sdk/openai-compatible` |
| Muse Spark 1.3 Contributor Free | muse-spark-1.3-contributor-free | `https://opencode.ai/zen/v1/responses` | `@ai-sdk/openai` |
| Jev 1.13 | jev-1.13 | `https://opencode.ai/zen/v1/systemone` | - |
| Jev 1.13 Free | jev-1.13-free | `https://opencode.ai/zen/v1/systemone` | - |
| Claude Sonnet 5 | claude-sonnet-5 | `https://opencode.ai/zen/v1/messages` | `@ai-sdk/anthropic` |
| Gemini 3.8 Flash | gemini-3.8-flash | `https://opencode.ai/zen/v1/models/gemini-3.8-flash` | `@ai-sdk/google` |
| Grok 4.7 | grok-4.7 | `https://opencode.ai/zen/v1/responses` | `@ai-sdk/openai` |

## Pricing

| Model | Input | Output | Cached Read | Cached Write |
| --- | --- | --- | --- | --- |
| Big Pickle | Free | Free | Free | - |
| MiMo-V2.5 Free | Free | Free | Free | - |
| Jev 1.13 Free | Free | Free | - | - |
| Jev 1.13 | $0.042 | Free | - | - |
| Grok 4.7 (≤ 200K tokens) | $2.00 | $6.00 | $0.50 | - |
| Grok 4.7 (> 200K tokens) | $4.00 | $12.00 | $1.00 | - |
"""


def _prime_docs(monkeypatch, mdx=MDX_SAMPLE):
    monkeypatch.setattr(oc_module, "_DOCS_CACHE", _parse_docs_mdx(mdx))
    monkeypatch.setattr(oc_module, "_DOCS_CACHE_TS", time.time())


def test_provider_registered():
    assert "opencode" in ProviderFactory.list_providers()
    assert ProviderFactory.get_provider_class("opencode") is OpenCodeProvider


# ------------------------------------------------------------------ #
# docs table parsing (dialect + pricing source of truth)
# ------------------------------------------------------------------ #
def test_parse_docs_mdx_endpoints_and_pricing():
    tables = _parse_docs_mdx(MDX_SAMPLE)
    ep = tables["endpoints"]
    assert ep["jev-1.13-free"]["endpoint"].endswith("/systemone")
    assert ep["muse-spark-1.3-contributor-free"]["endpoint"].endswith("/responses")
    assert ep["big-pickle"]["endpoint"].endswith("/chat/completions")
    assert ep["claude-sonnet-5"]["endpoint"].endswith("/messages")
    pr = tables["pricing"]
    assert pr["big-pickle"] == {"input": 0.0, "output": 0.0}
    assert pr["jev-1.13"] == {"input": 0.042, "output": 0.0}
    # tiered: first (≤) row wins
    assert pr["grok-4.7"] == {"input": 2.0, "output": 6.0}


def test_dialect_routing_from_docs(monkeypatch):
    _prime_docs(monkeypatch)
    assert OpenCodeProvider._dialect_for("jev-1.13-free") == "systemone"
    assert OpenCodeProvider._dialect_for("muse-spark-1.3-contributor-free") == "responses"
    assert OpenCodeProvider._dialect_for("big-pickle") == "chat"
    assert OpenCodeProvider._dialect_for("claude-sonnet-5") == "anthropic"
    assert OpenCodeProvider._dialect_for("gemini-3.8-flash") == "google"
    # unknown id → chat default
    assert OpenCodeProvider._dialect_for("brand-new-model") == "chat"


# ------------------------------------------------------------------ #
# free-tier imitation protocol: IDs, headers, tools, stream
# ------------------------------------------------------------------ #
def test_opencode_id_format_and_freshness():
    before = int(time.time() * 1000)
    msg = OpenCodeProvider._opencode_id("msg", descending=False)
    ses = OpenCodeProvider._opencode_id("ses", descending=True)
    after = int(time.time() * 1000)
    assert re.fullmatch(r"msg_[0-9a-f]{12}[0-9A-Za-z]{14}", msg), msg
    assert re.fullmatch(r"ses_[0-9a-f]{12}[0-9A-Za-z]{14}", ses), ses
    ts = int(msg[4:16], 16) >> 12
    assert abs(ts - (before % (1 << 36))) <= 2, f"id timestamp not current: {ts}"


def test_opencode_ids_unique():
    ids = [OpenCodeProvider._opencode_id("msg", descending=False) for _ in range(50)]
    assert len(set(ids)) == len(ids)


def test_build_headers_agent_shape():
    p = OpenCodeProvider(api_key="test")
    h1 = p._build_headers()
    h2 = p._build_headers()
    assert h1["User-Agent"].startswith("opencode/")
    assert h1["x-opencode-client"] == "cli"
    assert h1["x-opencode-project"] == "global"
    assert h1["x-opencode-request"] != h2["x-opencode-request"]
    assert h1["x-opencode-session"] != h2["x-opencode-session"]


def test_build_headers_session_env_override(monkeypatch):
    monkeypatch.setenv("UNIINFER_OPENCODE_SESSION", "ses_pinned123")
    p = OpenCodeProvider(api_key="test")
    assert p._build_headers()["x-opencode-session"] == "ses_pinned123"


def test_agent_tools_canonical_set():
    tools = OpenCodeProvider._agent_tools()
    names = {t["function"]["name"] for t in tools}
    assert {"bash", "edit", "read", "write", "grep", "webfetch"} <= names
    assert len(tools) == 12


def test_agent_tools_flat_format():
    flat = OpenCodeProvider._agent_tools_flat()
    assert len(flat) == 12
    bash = next(t for t in flat if t["name"] == "bash")
    assert bash["type"] == "function"
    assert "description" in bash and "parameters" in bash


def _req(model="mimo-v2.5-free", messages=None):
    return ChatCompletionRequest(
        model=model,
        messages=messages or [ChatMessage(role="user", content="hi")],
        temperature=0.7,
        streaming=False,
    )


def test_chat_payload_injects_tools_and_forces_stream():
    p = OpenCodeProvider(api_key="test")
    payload = p._build_payload(_req(), False, {})
    assert len(payload["tools"]) == 12
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


def test_chat_payload_unions_caller_tools():
    custom = {"type": "function", "function": {"name": "my_tool", "description": "x",
               "parameters": {"type": "object", "properties": {}}}}
    req = ChatCompletionRequest(
        model="mimo-v2.5-free",
        messages=[ChatMessage(role="user", content="hi")],
        temperature=0.7, streaming=False, tools=[custom],
    )
    p = OpenCodeProvider(api_key="test")
    payload = p._build_payload(req, False, {})
    names = {t["function"]["name"] for t in payload["tools"]}
    assert "my_tool" in names and "bash" in names


# ------------------------------------------------------------------ #
# SSE fakes
# ------------------------------------------------------------------ #
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
    def __init__(self, lines=None, status_code=200):
        self._lines = lines or []

    def stream(self, method, url, **kwargs):
        return _FakeStreamCtx(_FakeSSEResponse(self._lines, self.status_code
                                               if hasattr(self, "status") else 200))


class _FakePostResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code
        self.text = json.dumps(data)

    def json(self):
        return self._data


class _FakePostClient:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _FakePostResponse(self._data, self.status_code)


@pytest.mark.asyncio
async def test_chat_acomplete_aggregates_sse(monkeypatch):
    lines = [
        'data: {"model":"mimo-v2.5-free","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}',
        'data: {"model":"mimo-v2.5-free","choices":[{"index":0,"delta":{"reasoning":"thinking..."}}]}',
        'data: {"model":"mimo-v2.5-free","choices":[{"index":0,"delta":{"content":"lo"}}]}',
        'data: {"model":"mimo-v2.5-free","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        'data: {"model":"mimo-v2.5-free","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}',
        "data: [DONE]",
    ]
    p = OpenCodeProvider(api_key="test")
    monkeypatch.setattr(p, "_get_async_client", _awaitable(_FakeClient(lines)))
    resp = await p.acomplete(_req())
    assert resp.message.content == "Hello"
    assert resp.thinking == "thinking..."
    assert resp.finish_reason == "stop"
    assert resp.usage["total_tokens"] == 12


def _awaitable(value):
    async def _inner():
        return value
    return _inner


# ------------------------------------------------------------------ #
# systemone (jev) dialect
# ------------------------------------------------------------------ #
_JEV_OK = {
    "model": "jev-1.13-free",
    "answers": {"is_urgent": {"type": "noul", "noul": 0.96}},
    "usage": {"input_tokens": 290, "output_tokens": 23},
    "cost": "0",
}


def _jev_req(content):
    return _req(model="jev-1.13-free", messages=[ChatMessage(role="user", content=content)])


@pytest.mark.asyncio
async def test_jev_routes_to_systemone(monkeypatch):
    _prime_docs(monkeypatch)
    p = OpenCodeProvider(api_key="test")
    client = _FakePostClient(_JEV_OK)
    monkeypatch.setattr(p, "_get_async_client", _awaitable(client))
    resp = await p.acomplete(
        _jev_req(json.dumps({"state": "payments failed", "questions": {"q": {"type": "noul"}}}))
    )
    url, kwargs = client.calls[0]
    assert url.endswith("/systemone")
    assert kwargs["json"]["model"] == "jev-1.13-free"
    assert json.loads(resp.message.content) == _JEV_OK["answers"]
    assert resp.usage == {"prompt_tokens": 290, "completion_tokens": 23, "total_tokens": 313}


@pytest.mark.asyncio
async def test_jev_stream_single_chunk(monkeypatch):
    _prime_docs(monkeypatch)
    p = OpenCodeProvider(api_key="test")
    monkeypatch.setattr(p, "_get_async_client", _awaitable(_FakePostClient(_JEV_OK)))
    chunks = [c async for c in p.astream_complete(
        _jev_req(json.dumps({"state": "x", "questions": {}})))]
    assert len(chunks) == 1
    assert json.loads(chunks[0].message.content)["is_urgent"]["noul"] == 0.96


@pytest.mark.asyncio
async def test_jev_rejects_non_json(monkeypatch):
    _prime_docs(monkeypatch)
    p = OpenCodeProvider(api_key="test")
    with pytest.raises(Exception) as exc:
        await p.acomplete(_jev_req("just text"))
    assert "JSON" in str(exc.value) or "state" in str(exc.value)


# ------------------------------------------------------------------ #
# responses dialect (muse-spark)
# ------------------------------------------------------------------ #
_RESPONSES_LINES = [
    "event: response.created",
    "data: " + json.dumps({"type": "response.created", "response": {"id": "resp_1", "model": "muse-spark-1.3-contributor-free"}}),
    "event: response.output_text.delta",
    "data: " + json.dumps({"type": "response.output_text.delta", "delta": "PO"}),
    "data: " + json.dumps({"type": "response.output_text.delta", "delta": "NG"}),
    "event: response.output_item.done",
    "data: " + json.dumps({"type": "response.output_item.done", "item": {"type": "function_call", "call_id": "call_9", "name": "bash", "arguments": json.dumps({"cmd": "ls"})}}),
    "event: response.completed",
    "data: " + json.dumps({"type": "response.completed", "response": {"usage": {"input_tokens": 100, "output_tokens": 5, "total_tokens": 105}}}),
]


@pytest.mark.asyncio
async def test_responses_routes_and_aggregates(monkeypatch):
    _prime_docs(monkeypatch)
    p = OpenCodeProvider(api_key="test")
    monkeypatch.setattr(p, "_get_async_client", _awaitable(_FakeClient(_RESPONSES_LINES)))
    resp = await p.acomplete(_req(model="muse-spark-1.3-contributor-free"))
    assert resp.message.content == "PONG"
    assert resp.finish_reason == "tool_calls"
    assert resp.message.tool_calls[0]["function"]["name"] == "bash"
    assert resp.usage == {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105}


@pytest.mark.asyncio
async def test_responses_stream_yields_chunks(monkeypatch):
    _prime_docs(monkeypatch)
    p = OpenCodeProvider(api_key="test")
    monkeypatch.setattr(p, "_get_async_client", _awaitable(_FakeClient(_RESPONSES_LINES)))
    chunks = [c async for c in p.astream_complete(_req(model="muse-spark-1.3-contributor-free"))]
    contents = [c.message.content for c in chunks if c.message.content]
    assert "".join(contents) == "PONG"
    assert any(c.usage for c in chunks)
    tool_chunks = [c for c in chunks if c.message.tool_calls]
    assert tool_chunks and tool_chunks[0].message.tool_calls[0]["function"]["name"] == "bash"


def test_responses_payload_flat_tools_and_input(monkeypatch):
    _prime_docs(monkeypatch)
    p = OpenCodeProvider(api_key="test")
    req = ChatCompletionRequest(
        model="muse-spark-1.3-contributor-free",
        messages=[
            ChatMessage(role="system", content="be nice"),
            ChatMessage(role="user", content="hi"),
        ],
        temperature=0.7, streaming=True, max_tokens=123,
    )
    payload = p._build_responses_payload(req)
    assert payload["input"][0]["role"] == "developer"
    assert payload["input"][0]["content"][0]["type"] == "input_text"
    assert payload["max_output_tokens"] == 1024  # floored (reasoning headroom)
    assert payload["stream"] is True and payload["store"] is False
    assert len(payload["tools"]) == 12
    assert all("function" not in t for t in payload["tools"])  # flat format
    assert payload["tools"][0]["name"]


@pytest.mark.asyncio
async def test_unsupported_dialect_clear_error(monkeypatch):
    _prime_docs(monkeypatch)
    p = OpenCodeProvider(api_key="test")
    with pytest.raises(Exception) as exc:
        await p.acomplete(_req(model="claude-sonnet-5"))
    assert "anthropic" in str(exc.value)


# ------------------------------------------------------------------ #
# dynamic list_models (live zen ids + docs tables + pi.dev enrichment)
# ------------------------------------------------------------------ #
def test_list_models_dynamic_merge(monkeypatch):
    _prime_docs(monkeypatch)

    def fake_get(url, **kwargs):
        class R:
            def __init__(self, payload):
                self._p = payload
            def raise_for_status(self):
                pass
            def json(self):
                return self._p
        if "/zen/v1/models" in url:
            return R({"data": [{"id": "big-pickle"}, {"id": "mimo-v2.5-free"},
                               {"id": "jev-1.13-free"}, {"id": "muse-spark-1.3-contributor-free"},
                               {"id": "brand-new-model-free"}]})
        if "pi.dev" in url:
            return R({"mimo-v2.5-free": {"name": "MiMo V2.5", "contextWindow": 200000,
                                          "maxTokens": 32000, "cost": {"input": 0, "output": 0},
                                          "input": ["text"], "reasoning": True}})
        raise AssertionError(f"unexpected url {url}")

    with patch.object(requests, "get", side_effect=fake_get):
        models = OpenCodeProvider.list_models()
    by_id = {m.id: m for m in models}
    assert set(by_id) == {"big-pickle", "mimo-v2.5-free", "jev-1.13-free",
                          "muse-spark-1.3-contributor-free", "brand-new-model-free"}
    # free from docs pricing
    assert by_id["big-pickle"].access == "free"
    assert by_id["jev-1.13-free"].access == "free"
    assert by_id["mimo-v2.5-free"].access == "free"
    # enrichment from pi.dev
    assert by_id["mimo-v2.5-free"].context_window == 200000
    assert by_id["mimo-v2.5-free"].capabilities == {"reasoning": True}
    # unknown id without pricing → suffix heuristic, no hardcode
    assert by_id["brand-new-model-free"].access == "free"
    # dialect info carried in raw
    assert by_id["jev-1.13-free"].raw["endpoint"].endswith("/systemone")
    assert by_id["muse-spark-1.3-contributor-free"].raw["endpoint"].endswith("/responses")


def test_list_models_falls_back_to_docs_when_zen_down(monkeypatch):
    _prime_docs(monkeypatch)

    def fake_get(url, **kwargs):
        class R:
            def raise_for_status(self):
                pass
            def json(self):
                return {}
        if "pi.dev" in url:
            return R()
        raise ConnectionError("zen down")

    with patch.object(requests, "get", side_effect=fake_get):
        models = OpenCodeProvider.list_models()
    ids = {m.id for m in models}
    assert "jev-1.13-free" in ids and "big-pickle" in ids


# ------------------------------------------------------------------ #
# buffered responses backend: output only on response.completed
# ------------------------------------------------------------------ #
def test_responses_buffered_completed(monkeypatch):
    import json as _json
    lines = [
        "event: response.completed",
        "data: " + _json.dumps({
            "type": "response.completed",
            "response": {
                "usage": {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
                "output": [
                    {"type": "message", "role": "assistant",
                     "content": [{"type": "output_text", "text": "BUF"}]},
                    {"type": "function_call", "call_id": "c1", "name": "grep",
                     "arguments": "{}"},
                ],
            },
        }),
    ]

    async def run():
        _prime_docs(monkeypatch)
        p = OpenCodeProvider(api_key="test")
        monkeypatch.setattr(p, "_get_async_client", _awaitable(_FakeClient(lines)))
        resp = await p.acomplete(_req(model="muse-spark-1.3-contributor-free"))
        return resp

    import asyncio
    resp = asyncio.run(run())
    assert resp.message.content == "BUF"
    assert resp.finish_reason == "tool_calls"
    assert resp.message.tool_calls[0]["function"]["name"] == "grep"
    assert resp.usage["total_tokens"] == 9


def test_responses_payload_tool_items(monkeypatch):
    """Tool results must become function_call_output items (not role=tool),
    and assistant tool calls function_call items (responses-API shape)."""
    _prime_docs(monkeypatch)
    p = OpenCodeProvider(api_key="test")
    req = ChatCompletionRequest(
        model="muse-spark-1.3-contributor-free",
        messages=[
            ChatMessage(role="user", content="2+2?"),
            ChatMessage(
                role="assistant", content=None,
                tool_calls=[{"id": "call_1", "type": "function",
                             "function": {"name": "calc", "arguments": '{"x":4}'}}],
            ),
            ChatMessage(role="tool", tool_call_id="call_1", content="4"),
            ChatMessage(role="user", content="weiter"),
        ],
        streaming=True,
    )
    payload = p._build_responses_payload(req)
    items = payload["input"]
    # no role="tool" anywhere
    assert all(i.get("role") != "tool" for i in items)
    # assistant function_call item present
    assert any(i.get("type") == "function_call" and i.get("call_id") == "call_1"
               and i.get("name") == "calc" and "x" in i.get("arguments", "") for i in items)
    # tool result as function_call_output with matching call_id
    assert any(i.get("type") == "function_call_output" and i.get("call_id") == "call_1"
               and i.get("output") == "4" for i in items)
    # order: function_call before its function_call_output
    pos_fc = [i for i, it in enumerate(items) if it.get("type") == "function_call"]
    pos_out = [i for i, it in enumerate(items) if it.get("type") == "function_call_output"]
    assert pos_fc and pos_out and pos_fc[0] < pos_out[0]


class TestChatDialectToolHistory:
    """chat dialect: Zen's gateway rejects any transcript where a tool result is
    not introduced by exactly one assistant message carrying matching
    tool_calls — with a bare ``400 invalid_request_error: Upstream request
    failed`` that names nothing. Two real shapes violate it (verified against
    the live gateway): pi's split assistant runs and results orphaned by a
    mid-session provider switch. The normaliser rewrites only those."""

    @staticmethod
    def _tc(i):
        return {"id": i, "type": "function", "function": {"name": "bash", "arguments": "{}"}}

    @staticmethod
    def _a(*calls):
        return {"role": "assistant", "content": None, "tool_calls": list(calls)}

    @staticmethod
    def _t(i, body):
        return {"role": "tool", "tool_call_id": i, "content": body}

    def _run(self, messages):
        from uniinfer.providers.openai_compatible import normalize_tool_history
        return normalize_tool_history(messages)

    def test_split_assistant_run_is_folded(self):
        """assistant/tool/assistant/tool (pi's shape) becomes ONE assistant turn
        with all calls, followed by all results in order — the shape the
        gateway accepts."""
        out = self._run([
            {"role": "user", "content": "hi"},
            self._a(self._tc("c1")), self._t("c1", "a"),
            self._a(self._tc("c2")), self._t("c2", "b"),
            {"role": "user", "content": "weiter"},
        ])
        assert [m["role"] for m in out] == ["user", "assistant", "tool", "tool", "user"]
        assert [c["id"] for c in out[1]["tool_calls"]] == ["c1", "c2"]
        assert [m["tool_call_id"] for m in out[2:4]] == ["c1", "c2"]
        # ordering preserved: both results before the next user turn
        assert out[4]["content"] == "weiter"

    def test_three_call_run_is_folded(self):
        out = self._run([
            {"role": "user", "content": "hi"},
            self._a(self._tc("c1")), self._t("c1", "a"),
            self._a(self._tc("c2")), self._t("c2", "b"),
            self._a(self._tc("c3")), self._t("c3", "c"),
            {"role": "user", "content": "w"},
        ])
        assert [c["id"] for c in out[1]["tool_calls"]] == ["c1", "c2", "c3"]
        assert [m["tool_call_id"] for m in out[2:5]] == ["c1", "c2", "c3"]
        assert out[5]["content"] == "w"

    def test_accepted_shape_survives(self):
        """One assistant message followed directly by its results is the shape
        the gateway accepts — results stay in place."""
        msgs = [
            {"role": "user", "content": "hi"},
            self._a(self._tc("c1"), self._tc("c2")),
            self._t("c1", "a"), self._t("c2", "b"),
            {"role": "user", "content": "w"},
        ]
        out = self._run(msgs)
        assert out == msgs

    def test_single_tool_turn_survives(self):
        msgs = [
            {"role": "user", "content": "hi"},
            self._a(self._tc("c1")),
            self._t("c1", "a"),
            {"role": "user", "content": "w"},
        ]
        assert self._run(msgs) == msgs

    def test_tool_after_text_only_assistant_is_demoted(self):
        """assistant(TEXT) tool(...) — a provider-switched turn without calls —
        must not send the result as role=tool (rejected upstream)."""
        out = self._run([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Alles klar — Fix 1 und 2."},
            self._t("c1", "grep output"),
            {"role": "user", "content": "weiter"},
        ])
        assert all(m["role"] != "tool" for m in out)
        assert any("grep output" in (m.get("content") or "") for m in out)

    def test_orphan_result_becomes_user_turn(self):
        out = self._run([
            {"role": "user", "content": "hi"},
            self._t("unknown-id", "output"),
            {"role": "user", "content": "weiter"},
        ])
        assert [m["role"] for m in out] == ["user", "user", "user"]
        assert "output" in out[1]["content"]

    def test_plain_conversation_untouched(self):
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hallo"},
            {"role": "user", "content": "w"},
        ]
        assert self._run(msgs) == msgs

    def test_build_payload_applies_the_normaliser(self):
        """The hook is wired into the chat payload, not just a helper."""
        provider = OpenCodeProvider(api_key="k")
        req = ChatCompletionRequest(
            model="space-bunny-free",
            messages=[
                ChatMessage(role="user", content="hi"),
                ChatMessage(role="assistant", content=None,
                            tool_calls=[{"id": "c1", "type": "function",
                                         "function": {"name": "bash", "arguments": "{}"}}]),
                ChatMessage(role="tool", tool_call_id="c1", content="a"),
                ChatMessage(role="assistant", content=None,
                            tool_calls=[{"id": "c2", "type": "function",
                                         "function": {"name": "bash", "arguments": "{}"}}]),
                ChatMessage(role="tool", tool_call_id="c2", content="b"),
                ChatMessage(role="user", content="weiter"),
            ],
        )
        messages = provider._build_payload(req, False, {})["messages"]
        # the split run is folded: one assistant turn, both calls, both results
        assert [c["id"] for c in messages[1]["tool_calls"]] == ["c1", "c2"]
        assert [m.get("role") for m in messages] == ["user", "assistant", "tool", "tool", "user"]
        assert [m.get("tool_call_id") for m in messages[2:4]] == ["c1", "c2"]


class TestStrictToolHistoryFlag:
    """The normaliser is opt-in per provider (STRICT_TOOL_HISTORY): permissive
    backends must keep receiving the transcript untouched."""

    def test_base_default_is_off(self):
        from uniinfer.providers.openai_compatible import OpenAICompatibleChatProvider as B
        assert B.STRICT_TOOL_HISTORY is False

    def test_opencode_opts_in(self):
        assert OpenCodeProvider.STRICT_TOOL_HISTORY is True

    def test_off_flag_leaves_messages_untouched(self):
        """With the flag off (base default), a broken shape passes through as-is."""
        from uniinfer.providers.openai_compatible import OpenAICompatibleChatProvider as B
        import json as _json

        class Permissive(B):
            BASE_URL = "https://example.invalid/v1"
            PROVIDER_ID = "permissive"

        p = Permissive(api_key="k")
        req = ChatCompletionRequest(
            model="m",
            messages=[
                ChatMessage(role="user", content="hi"),
                ChatMessage(role="assistant", content=None,
                            tool_calls=[{"id": "c1", "type": "function",
                                         "function": {"name": "bash", "arguments": "{}"}}]),
                ChatMessage(role="tool", tool_call_id="c1", content="a"),
                ChatMessage(role="assistant", content=None,
                            tool_calls=[{"id": "c2", "type": "function",
                                         "function": {"name": "bash", "arguments": "{}"}}]),
                ChatMessage(role="tool", tool_call_id="c2", content="b"),
                ChatMessage(role="user", content="weiter"),
            ],
        )
        messages = p._build_payload(req, False, {})["messages"]
        # untouched: the split run survives verbatim
        assert [m.get("role") for m in messages] == ["user", "assistant", "tool", "assistant", "tool", "user"]

    def test_on_flag_normalises(self):
        import json as _json
        p = OpenCodeProvider(api_key="k")
        req = ChatCompletionRequest(
            model="space-bunny-free",
            messages=[
                ChatMessage(role="user", content="hi"),
                ChatMessage(role="assistant", content=None,
                            tool_calls=[{"id": "c1", "type": "function",
                                         "function": {"name": "bash", "arguments": "{}"}}]),
                ChatMessage(role="tool", tool_call_id="c1", content="a"),
                ChatMessage(role="assistant", content=None,
                            tool_calls=[{"id": "c2", "type": "function",
                                         "function": {"name": "bash", "arguments": "{}"}}]),
                ChatMessage(role="tool", tool_call_id="c2", content="b"),
                ChatMessage(role="user", content="weiter"),
            ],
        )
        messages = p._build_payload(req, False, {})["messages"]
        assert [m.get("role") for m in messages] == ["user", "assistant", "tool", "tool", "user"]
        assert [c["id"] for c in messages[1]["tool_calls"]] == ["c1", "c2"]


class TestNormalizeToolCallIds:
    """kilo answers a transcript carrying ids another backend minted (TU vLLM's
    chatcmpl-tool-*) with a silent empty completion. NORMALIZE_TOOL_CALL_IDS
    rewrites every id to call_<n> — same mapping on both sides of each pair."""

    @staticmethod
    def _transcript():
        return [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "chatcmpl-tool-aba455040e1e2db3", "type": "function",
                 "function": {"name": "bash", "arguments": "{}"}},
                {"id": "chatcmpl-tool-a1415dddfcd65862", "type": "function",
                 "function": {"name": "read", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "chatcmpl-tool-aba455040e1e2db3", "content": "a"},
            {"role": "tool", "tool_call_id": "chatcmpl-tool-a1415dddfcd65862", "content": "b"},
            {"role": "assistant", "content": "fertig"},
            {"role": "user", "content": "weiter"},
        ]

    def test_pairing_survives_the_rewrite(self):
        from uniinfer.providers.openai_compatible import normalize_tool_call_ids
        out = normalize_tool_call_ids(self._transcript())
        ids = [c["id"] for c in out[1]["tool_calls"]]
        refs = [m["tool_call_id"] for m in out if m["role"] == "tool"]
        # every result references the rewritten id of its call
        assert refs == ids[:2]
        # short neutral format, no chatcmpl-tool- prefix anywhere
        assert all(i.startswith("call_") and len(i) <= 13 for i in ids + refs)
        assert "chatcmpl-tool-" not in json.dumps(out)

    def test_deterministic(self):
        from uniinfer.providers.openai_compatible import normalize_tool_call_ids
        once = json.dumps(normalize_tool_call_ids(self._transcript()))
        twice = json.dumps(normalize_tool_call_ids(self._transcript()))
        assert once == twice

    def test_unmatched_tool_call_id_left_alone(self):
        from uniinfer.providers.openai_compatible import normalize_tool_call_ids
        msgs = [{"role": "tool", "tool_call_id": "orphan-1", "content": "x"}]
        assert normalize_tool_call_ids(msgs)[0]["tool_call_id"] == "orphan-1"

    def test_kilo_opts_in_opencode_does_not(self):
        assert KiloProvider.NORMALIZE_TOOL_CALL_IDS is True
        assert OpenCodeProvider.NORMALIZE_TOOL_CALL_IDS is False

    def test_kilo_payload_rewrites_ids(self):
        p = KiloProvider(api_key="k")
        req = ChatCompletionRequest(
            model="stealth/space-bunny-alpha",
            messages=[
                ChatMessage(role="user", content="hi"),
                ChatMessage(role="assistant", content=None,
                            tool_calls=[{"id": "chatcmpl-tool-deadbeef", "type": "function",
                                         "function": {"name": "bash", "arguments": "{}"}}]),
                ChatMessage(role="tool", tool_call_id="chatcmpl-tool-deadbeef", content="out"),
            ],
        )
        messages = p._build_payload(req, False, {})["messages"]
        call_id = messages[1]["tool_calls"][0]["id"]
        assert call_id.startswith("call_")
        assert messages[2]["tool_call_id"] == call_id


class TestEmptyCompletionRetry:
    """Flaky gateways (kilo LB roulette) answer finish=stop with zero content —
    sometimes. The same request replayed on a fresh connection lands on a
    healthy replica and the empty attempt consumed no tokens, so the replay is
    free. Flag EMPTY_COMPLETION_RETRIES gates it (default 0 = relay as-is).
    Tested on KiloProvider: it walks the base-class paths the retry lives in."""

    @staticmethod
    def _empty_sse():
        return (
            'data: {"id": "x", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant"}}]}\n'
            'data: {"id": "x", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}\n'
            "data: [DONE]\n"
        )

    @staticmethod
    def _full_sse():
        return (
            'data: {"id": "y", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant"}}]}\n'
            'data: {"id": "y", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "recovered"}}]}\n'
            'data: {"id": "y", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"total_tokens": 5}}\n'
            "data: [DONE]\n"
        )

    @staticmethod
    def _sse_response(text):
        import httpx as _httpx

        async def lines():
            for ln in text.split("\n"):
                yield ln

        r = AsyncMock(spec=_httpx.Response)
        r.status_code = 200
        r.aiter_lines = lines
        return r

    @staticmethod
    def _cm(response):
        class _CM:
            async def __aenter__(self):
                return response

            async def __aexit__(self, *a):
                return None

        return _CM()

    @staticmethod
    def _stream_client(sse_text):
        c = AsyncMock(spec=httpx.AsyncClient)
        c.is_closed = False
        c.stream.return_value = TestEmptyCompletionRetry._cm(
            TestEmptyCompletionRetry._sse_response(sse_text))
        return c

    def _provider(self, retries, clients):
        p = KiloProvider(api_key="k")
        p.EMPTY_COMPLETION_RETRIES = retries
        queue = list(clients)

        async def fake_get():
            return queue.pop(0)

        p._get_async_client = fake_get
        p._retry_client = lambda: queue.pop(0)
        return p

    @staticmethod
    def _request():
        return ChatCompletionRequest(model="stealth/space-bunny-alpha",
                                     messages=[ChatMessage(role="user", content="hi")])

    @pytest.mark.asyncio
    async def test_stream_replays_after_empty(self):
        dead = self._stream_client(self._empty_sse())
        alive = self._stream_client(self._full_sse())
        p = self._provider(retries=1, clients=[dead, alive])
        chunks = [c async for c in p.astream_complete(self._request())]

        visible = [c for c in chunks if (c.message.content or "").strip()]
        assert [c.message.content for c in visible] == ["recovered"]
        finishes = [c for c in chunks if c.finish_reason == "stop"]
        assert len(finishes) == 1  # the dead attempt's terminal state never leaked

    @pytest.mark.asyncio
    async def test_stream_relay_when_retries_exhausted(self):
        dead = self._stream_client(self._empty_sse())
        p = self._provider(retries=0, clients=[dead])
        chunks = [c async for c in p.astream_complete(self._request())]
        finishes = [c for c in chunks if c.finish_reason == "stop"]
        assert len(finishes) == 1
        assert all(not (c.message.content or "").strip() for c in chunks)

    @pytest.mark.asyncio
    async def test_nonstream_replays_after_empty(self):
        import httpx as _httpx

        empty = MagicMock(spec=_httpx.Response)
        empty.status_code = 200
        empty.json.return_value = {"choices": [{"message": {"role": "assistant", "content": ""},
                                                 "finish_reason": "stop"}]}
        full = MagicMock(spec=_httpx.Response)
        full.status_code = 200
        full.json.return_value = {"choices": [{"message": {"role": "assistant", "content": "recovered"},
                                                "finish_reason": "stop"}]}
        dead = AsyncMock(spec=_httpx.AsyncClient)
        dead.is_closed = False
        dead.post.return_value = empty
        alive = AsyncMock(spec=_httpx.AsyncClient)
        alive.is_closed = False
        alive.post.return_value = full

        p = self._provider(retries=1, clients=[dead, alive])
        resp = await p.acomplete(self._request())
        assert resp.message.content == "recovered"
        dead.post.assert_called_once()
        alive.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_nonstream_no_retry_when_disabled(self):
        import httpx as _httpx

        empty = MagicMock(spec=_httpx.Response)
        empty.status_code = 200
        empty.json.return_value = {"choices": [{"message": {"role": "assistant", "content": ""},
                                                 "finish_reason": "stop"}]}
        dead = AsyncMock(spec=_httpx.AsyncClient)
        dead.is_closed = False
        dead.post.return_value = empty
        p = self._provider(retries=0, clients=[dead])
        resp = await p.acomplete(self._request())
        assert not (resp.message.content or "").strip()
        dead.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_two_empty_then_healthy(self):
        sse = self._empty_sse()
        full = self._full_sse()
        clients = [self._stream_client(sse), self._stream_client(sse), self._stream_client(full)]
        p = self._provider(retries=2, clients=clients)
        chunks = [c async for c in p.astream_complete(self._request())]
        visible = [c for c in chunks if (c.message.content or "").strip()]
        assert [c.message.content for c in visible] == ["recovered"]


class TestEmptyCompletionRateLimitShadow:
    """After the retry budget, an empty completion is surfaced as a real 429 —
    kilo's free tier throttles with 200 + empty instead of a proper 429, and a
    silent empty answer looks like a model bug to the client. Providers without
    the opt-in (retries=0) keep relaying empty as before."""

    def _provider(self, retries, sse_texts):
        p = KiloProvider(api_key="k")
        p.EMPTY_COMPLETION_RETRIES = retries
        queue = list(sse_texts)
        clients = []
        for t in queue:
            c = AsyncMock(spec=httpx.AsyncClient)
            c.is_closed = False
            c.stream.return_value = TestEmptyCompletionRetry._cm(
                TestEmptyCompletionRetry._sse_response(t))
            clients.append(c)

        async def fake_get():
            return clients.pop(0)

        p._get_async_client = fake_get
        p._retry_client = lambda: clients.pop(0)
        return p

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises_rate_limit(self):
        from uniinfer.errors import RateLimitError

        def post_client(payload_content):
            c = AsyncMock(spec=httpx.AsyncClient)
            c.is_closed = False
            r = MagicMock(spec=httpx.Response)
            r.status_code = 200
            r.json.return_value = {"choices": [{"message": {"role": "assistant", "content": payload_content},
                                                 "finish_reason": "stop"}]}
            c.post.return_value = r
            return c

        empty_c = post_client("")
        second_c = post_client("")  # retry auch leer -> 429-shadow
        p = KiloProvider(api_key="k")
        p.EMPTY_COMPLETION_RETRIES = 1
        queue = [empty_c, second_c]

        async def fake_get():
            return queue.pop(0)

        p._get_async_client = fake_get
        p._retry_client = lambda: queue.pop(0)

        with pytest.raises(RateLimitError) as ei:
            await p.acomplete(TestEmptyCompletionRetry._request())
        assert ei.value.status_code == 429
        empty_c.post.assert_called_once()
        second_c.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_retry_exhausted_raises_rate_limit(self):
        from uniinfer.errors import RateLimitError
        empty = TestEmptyCompletionRetry._empty_sse()
        p = self._provider(retries=1, sse_texts=[empty, empty])
        with pytest.raises(RateLimitError):
            async for _ in p.astream_complete(TestEmptyCompletionRetry._request()):
                pass

    @pytest.mark.asyncio
    async def test_zero_retries_still_relays(self):
        """No opt-in (retries=0): empty relays as before — no 429 mapping."""
        empty = TestEmptyCompletionRetry._empty_sse()
        p = self._provider(retries=0, sse_texts=[empty])
        chunks = [c async for c in p.astream_complete(TestEmptyCompletionRetry._request())]
        assert any(c.finish_reason == "stop" for c in chunks)


class TestRetryAfterVisibility:
    """'Wie lange warten?' — die Wartezeit muss beim Client ankommen: upstream
    Retry-After wird geparsed, der 429-Chip traegt retry_after + Klartext."""

    def test_parse_retry_after_seconds_and_date(self):
        from email.utils import format_datetime
        from datetime import datetime, timezone, timedelta

        from uniinfer.providers.openai_compatible import parse_retry_after
        assert parse_retry_after({"Retry-After": "30"}) == 30.0
        assert parse_retry_after({"retry-after": "7"}) == 7.0
        future = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=90))
        got = parse_retry_after({"Retry-After": future})
        assert 80 <= got <= 90
        assert parse_retry_after({}) is None
        assert parse_retry_after({"Retry-After": "garbage"}) is None

    def test_upstream_429_relay_carries_retry_after(self):
        """acomplete on a 429: RateLimitError.retry_after comes from the header."""
        import httpx as _httpx

        limited = MagicMock(spec=_httpx.Response)
        limited.status_code = 429
        limited.text = "slow down"
        limited.headers = {"Retry-After": "45"}
        c = AsyncMock(spec=_httpx.AsyncClient)
        c.is_closed = False
        c.post.return_value = limited
        p = KiloProvider(api_key="k")
        p.EMPTY_COMPLETION_RETRIES = 1

        async def fake_get():
            return c

        p._get_async_client = fake_get
        import asyncio
        from uniinfer import ChatCompletionRequest, ChatMessage
        from uniinfer.errors import RateLimitError

        async def main():
            with pytest.raises(RateLimitError) as ei:
                await p.acomplete(ChatCompletionRequest(
                    model="stealth/space-bunny-alpha",
                    messages=[ChatMessage(role="user", content="hi")]))
            return ei.value.retry_after

        assert asyncio.run(main()) == 45.0


class TestNormalizerSafetyProperties:
    """The normaliser rewrites only rejected shapes; these properties must hold
    for every input, so future rule changes cannot silently corrupt history."""

    def test_idempotent_on_all_fixtures(self):
        """normalize(normalize(x)) == normalize(x) — a second pass must be a
        no-op, otherwise repeated requests drift."""
        from uniinfer.providers.openai_compatible import normalize_tool_history

        tc = lambda i: {"id": i, "type": "function", "function": {"name": "bash", "arguments": "{}"}}
        A = lambda *t: {"role": "assistant", "content": None, "tool_calls": list(t)}
        T = lambda i, c: {"role": "tool", "tool_call_id": i, "content": c}
        U = lambda c: {"role": "user", "content": c}
        fixtures = [
            [U("hi"), A(tc("c1")), T("c1", "a"), A(tc("c2")), T("c2", "b"), U("w")],
            [U("hi"), T("orphan", "x"), U("w")],
            [U("hi"), A(tc("c1"), tc("c2")), T("c1", "a"), T("c2", "b"), U("w")],
            [{"role": "assistant", "content": "text"}, T("c9", "y")],
            [U("hi"), A(tc("c1")), T("c1", "a")],
        ]
        for msgs in fixtures:
            once = normalize_tool_history(msgs)
            twice = normalize_tool_history(once)
            assert once == twice

    def test_no_text_loss(self):
        """Every text the caller sent is still present after normalising —
        the rewrite may regroup but never drop."""
        from uniinfer.providers.openai_compatible import normalize_tool_history

        tc = lambda i: {"id": i, "type": "function", "function": {"name": "bash", "arguments": "{}"}}
        msgs = [
            {"role": "user", "content": "erster turn"},
            {"role": "assistant", "content": None, "tool_calls": [tc("c1")]},
            {"role": "tool", "tool_call_id": "c1", "content": "ergebnis-eins"},
            {"role": "assistant", "content": "zweiter text"},
            {"role": "user", "content": "orphan folgt"},
            {"role": "tool", "tool_call_id": "unknown", "content": "verwaistes ergebnis"},
            {"role": "user", "content": "schluss"},
        ]
        blob = json.dumps(normalize_tool_history(msgs))
        for text in ("erster turn", "ergebnis-eins", "zweiter text",
                     "orphan folgt", "verwaistes ergebnis", "schluss"):
            assert text in blob
