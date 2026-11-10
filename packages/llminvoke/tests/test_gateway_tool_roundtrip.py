"""Gateway-Pfad (_invoke_via_openai_sdk) muss Tool-Konversationen tragen
(User-Regel 2026-11-10: „immer über uniinfer/llminvoke — kein openai-SDK-
Bypass im agentos-ReactAgent"). Vorher fielen tool_calls in Anfrage UND
Antwort weg; tools in extra_body wurden vom Gateway still ignoriert.
"""

from unittest import mock

from llminvoke import _invoke_via_openai_sdk
from uniinfer import ChatMessage


class _TC:
    """openai-SDK-artiger tool_call (hat model_dump)."""
    def __init__(self, name, args, cid):
        self.id = cid
        self.function = type("F", (), {"name": name, "arguments": args})()
    def model_dump(self):
        return {"id": self.id, "type": "function",
                "function": {"name": self.function.name, "arguments": self.function.arguments}}


class _Msg:
    def __init__(self, content, tool_calls=None):
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, msg, finish_reason="tool_calls"):
        self.message = msg
        self.finish_reason = finish_reason


class _Resp:
    def __init__(self, msg):
        self.model = "tu@x"
        self.choices = [_Choice(msg)]
        self.usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})()
    def model_dump(self):
        return {}


def test_tools_top_level_nicht_extra_body():
    captured = {}
    class FakeCompletions:
        def create(self, **kw):
            captured.update(kw)
            return _Resp(_Msg(None, tool_calls=[_TC("gisa", "{}", "c1")]))
    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()
    with mock.patch("openai.OpenAI", return_value=FakeClient()):
        r = _invoke_via_openai_sdk(
            base_url="http://gw", bearer="t", provider="tu", model="m",
            messages=[ChatMessage(role="user", content="hi")],
            temperature=0.2, max_tokens=100,
            request_kwargs={"tools": [{"type": "function"}],
                            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        )
    assert captured["tools"] == [{"type": "function"}], "tools müssen top-level sein"
    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "tools" not in (captured["extra_body"] or {})


def test_assistent_tool_calls_in_und_aus():
    captured = {}
    class FakeCompletions:
        def create(self, **kw):
            captured.update(kw)
            return _Resp(_Msg("denke...", tool_calls=[_TC("recht_lesen", '{"frage":"x"}', "c9")]))
    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()
    hist = [
        {"role": "user", "content": "frage"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ergebnis"},
    ]
    with mock.patch("openai.OpenAI", return_value=FakeClient()):
        r = _invoke_via_openai_sdk(
            base_url="http://gw", bearer="t", provider="tu", model="m",
            messages=hist,  # rohe dicts VERBATIM
            temperature=0.2, max_tokens=100, request_kwargs={},
        )
    assert captured["messages"] == hist, "dict-messages unverändert durchreichen"
    assert r.message.tool_calls and r.message.tool_calls[0]["id"] == "c9"
    assert r.message.tool_calls[0]["function"]["name"] == "recht_lesen"


def test_chatmessage_objekt_mit_tool_feldern():
    captured = {}
    class FakeCompletions:
        def create(self, **kw):
            captured.update(kw)
            return _Resp(_Msg("ok"))
    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()
    m = ChatMessage(role="tool", content="res", tool_call_id="c2")
    with mock.patch("openai.OpenAI", return_value=FakeClient()):
        _invoke_via_openai_sdk(base_url="http://gw", bearer="t", provider="tu", model="m",
                               messages=[m], temperature=0.2, max_tokens=10, request_kwargs={})
    sent = captured["messages"][0]
    assert sent["role"] == "tool" and sent["tool_call_id"] == "c2" and sent["content"] == "res"
