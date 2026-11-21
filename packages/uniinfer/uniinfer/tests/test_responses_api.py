"""Hermetic tests for the Responses-API normalization (translation + router).

The translation core is pure (no I/O); the router tests patch the dispatch
seam. Live behavior (real providers) is covered by the issue's smoke steps.
"""
import json

import pytest
from fastapi.testclient import TestClient

from uniinfer.responses_translate import (
    ResponsesEventEmitter,
    completion_to_response,
    reject_unsupported,
    request_to_chat,
)


def _chunk(content=None, tool_calls=None, finish=None, usage=None):
    delta = {}
    if content:
        delta["content"] = content
    if tool_calls:
        delta["tool_calls"] = tool_calls
    ch = {"choices": [{"index": 0, "delta": delta}]}
    if finish:
        ch["choices"][0]["finish_reason"] = finish
    if usage:
        ch["usage"] = usage
    return ch


class TestRequestTranslation:
    def test_string_input(self):
        chat = request_to_chat({"model": "x", "input": "hallo"})
        assert chat["messages"] == [{"role": "user", "content": "hallo"}]

    def test_instructions_become_system(self):
        chat = request_to_chat({"model": "x", "input": "hi", "instructions": "Sei kurz."})
        assert chat["messages"][0] == {"role": "system", "content": "Sei kurz."}
        assert chat["messages"][1]["role"] == "user"

    def test_content_parts(self):
        chat = request_to_chat(
            {
                "model": "x",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Was ist das?"},
                            {"type": "input_image", "image_url": "https://x/y.png"},
                        ],
                    }
                ],
            }
        )
        content = chat["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "Was ist das?"}
        assert content[1]["type"] == "image_url"

    def test_function_roundtrip(self):
        chat = request_to_chat(
            {
                "model": "x",
                "input": [
                    {"type": "message", "role": "user", "content": "Wetter?"},
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": '{"city":"Wien"}',
                    },
                    {"type": "function_call_output", "call_id": "call_1", "output": '{"t":20}'},
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Wetter",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            }
        )
        roles = [m["role"] for m in chat["messages"]]
        assert roles == ["user", "assistant", "tool"]
        assert chat["messages"][1]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert chat["messages"][2]["tool_call_id"] == "call_1"
        assert chat["tools"][0]["function"]["name"] == "get_weather"

    def test_unsupported(self):
        assert reject_unsupported({"previous_response_id": "resp_1"})
        assert reject_unsupported({"tools": [{"type": "web_search"}]})
        assert reject_unsupported({"input": "x"}) is None


class TestResponseTranslation:
    def _full(self, content="Hallo", tool_calls=None, usage=None, finish="stop"):
        class _Msg:
            pass

        m = _Msg()
        m.content = content
        m.tool_calls = tool_calls
        r = _Msg()
        r.message = m
        r.usage = usage or {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        r.finish_reason = finish
        return r

    def test_text_response(self):
        out = completion_to_response(self._full(), "tu@x")
        assert out["object"] == "response" and out["status"] == "completed"
        assert out["output"][0]["type"] == "message"
        assert out["output"][0]["content"][0]["text"] == "Hallo"
        assert out["usage"]["input_tokens"] == 5
        assert out["usage"]["output_tokens"] == 3

    def test_tool_call_response(self):
        tcs = [{"id": "call_9", "function": {"name": "f", "arguments": "{}"}}]
        out = completion_to_response(self._full(content=None, tool_calls=tcs), "tu@x")
        assert out["output"][0]["type"] == "function_call"
        assert out["output"][0]["call_id"] == "call_9"

    def test_incomplete_on_length(self):
        out = completion_to_response(self._full(finish="length"), "tu@x")
        assert out["status"] == "incomplete"
        assert out["incomplete_details"] == {"reason": "max_output_tokens"}


class TestEventEmitter:
    def _events(self, chunks):
        em = ResponsesEventEmitter("tu@x")
        out = list(em.start())
        for c in chunks:
            out += list(em.feed(c))
        out += list(em.finish())
        return out

    def test_event_sequence(self):
        events = self._events(
            [_chunk(content="Hallo "), _chunk(content="Welt"), _chunk(finish="stop")]
        )
        types = [json.loads(e.split("data: ", 1)[1])["type"] for e in events]
        assert types[0] == "response.created"
        assert "response.output_item.added" in types
        assert "response.content_part.added" in types
        assert types.count("response.output_text.delta") == 2
        assert "response.output_text.done" in types
        assert types[-1] == "response.completed"
        completed = json.loads(events[-1].split("data: ", 1)[1])["response"]
        text = completed["output"][0]["content"][0]["text"]
        assert text == "Hallo Welt"

    def test_sse_format(self):
        events = self._events([_chunk(content="x")])
        assert events[0].startswith("event: response.created\ndata: ")

    def test_tool_call_stream(self):
        events = self._events(
            [
                _chunk(tool_calls=[{"index": 0, "id": "call_1", "function": {"name": "f", "arguments": "{"}}]),
                _chunk(tool_calls=[{"index": 0, "function": {"arguments": "}"}}]),
            ]
        )
        payload_types = [json.loads(e.split("data: ", 1)[1])["type"] for e in events]
        assert "response.function_call_arguments.delta" in payload_types
        completed = json.loads(events[-1].split("data: ", 1)[1])["response"]
        fc = completed["output"][0]
        assert fc["type"] == "function_call" and fc["arguments"] == "{}"

    def test_usage_from_tail_chunk(self):
        events = self._events(
            [_chunk(content="ok"), _chunk(finish="stop", usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3})]
        )
        completed = json.loads(events[-1].split("data: ", 1)[1])["response"]
        assert completed["usage"]["total_tokens"] == 3

    def test_failed_event(self):
        em = ResponsesEventEmitter("tu@x")
        ev = em.failed("upstream broke")
        payload = json.loads(ev.split("data: ", 1)[1])
        assert payload["type"] == "response.failed"
        assert payload["response"]["status"] == "failed"


@pytest.fixture()
def client():
    from uniinfer.proxy_app import app

    return TestClient(app, raise_server_exceptions=False)


class _FakeFull:
    def __init__(self):
        class M:
            content = "Antwort via responses"
            tool_calls = None

        class R:
            message = M()
            usage = {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8}
            finish_reason = "stop"

        self.message = R.message
        self.usage = R.usage
        self.finish_reason = R.finish_reason


class TestRouter:
    def test_nonstream(self, client, monkeypatch):
        import uniinfer.proxy_routers.responses as resp_mod

        class _Plan:
            provider_name = "tu"
            provider_model = "tu@d1-3b"
            provider_api_key = "k"
            auto_chain = None
            routed_model = None

            class target:
                @staticmethod
                async def aclose():
                    pass

        async def fake_complete(plan, **kw):
            return _FakeFull()

        monkeypatch.setattr(resp_mod, "resolve_dispatch", lambda *a, **k: _Plan())
        monkeypatch.setattr(resp_mod, "guarded_complete", fake_complete)
        monkeypatch.setattr(resp_mod, "make_meter", lambda *a, **k: lambda *a2, **k2: None)
        monkeypatch.setattr(resp_mod, "normalize_roles", lambda msgs, pn: msgs)

        r = client.post("/v1/responses", json={"model": "tu@d1-3b", "input": "hi"})
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "response"
        assert body["output"][0]["content"][0]["text"] == "Antwort via responses"

    def test_unsupported_400(self, client):
        r = client.post(
            "/v1/responses", json={"model": "tu@d1-3b", "input": "hi", "previous_response_id": "resp_1"}
        )
        assert r.status_code == 400
        assert "previous_response_id" in r.json()["detail"]

    def test_bad_model_400(self, client):
        r = client.post("/v1/responses", json={"model": "noproject", "input": "hi"})
        assert r.status_code in (400, 401, 500)


class TestCloudflareEmptyChoicesStream:
    """Regression: gpt-oss on CF emits SSE chunks with "choices": [] — the
    provider stream parser indexed [0] and crashed ('list index out of range'),
    killing the whole stream instead of skipping the chunk. Found live
    2026-11-21 via /v1/responses stream validation."""

    def test_empty_choices_chunk_is_skipped(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        from uniinfer.core import ChatCompletionRequest
        from uniinfer.providers.cloudflare import CloudflareProvider

        provider = CloudflareProvider(api_key="k", account_id="acct")

        mock_response = MagicMock()
        mock_response.status_code = 200
        lines = [
            'data: {"choices": [], "usage": {"prompt_tokens": 1}}',
            'data: {"choices": [{"delta": {"content": "ok"}}]}',
            "data: [DONE]",
        ]

        async def aiter_lines():
            for line in lines:
                yield line

        mock_response.aiter_lines = aiter_lines

        class _FakeClient:
            def stream(self, *a, **k):
                return _Ctx(mock_response)

        async def run():
            chunks = []
            with patch.object(provider, "_get_async_client", AsyncMock(return_value=_FakeClient())):
                from uniinfer.core import ChatMessage

                req = ChatCompletionRequest(
                    messages=[ChatMessage(role="user", content="hi")], model="m"
                )
                async for c in provider.astream_complete(req):
                    chunks.append(c)
            return chunks

        class _Ctx:
            def __init__(self, resp):
                self._r = resp

            async def __aenter__(self):
                return self._r

            async def __aexit__(self, *a):
                return False

        chunks = asyncio.run(run())
        assert len(chunks) == 1
        assert chunks[0].message.content == "ok"
