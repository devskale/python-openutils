"""Regression: Mistral trailing-assistant prefill must carry prefix=True.

A trailing assistant message is the OpenAI-compatible prefill/continuation
pattern. Mistral rejects a trailing assistant without ``prefix=True`` (400:
"Expected last role User or Tool (or Assistant with prefix True)"). Since
``prefix`` isn't an OpenAI field, the Mistral provider sets it on the last
message when it's an assistant turn.
"""
from uniinfer import ChatMessage
from uniinfer.providers.mistral import MistralProvider


def _flatten(*msgs):
    return MistralProvider(api_key="k")._flatten_messages(list(msgs))


def test_trailing_assistant_gets_prefix():
    flat = _flatten(
        ChatMessage(role="user", content="Say OK"),
        ChatMessage(role="assistant", content="{"),
    )
    assert flat[-1]["role"] == "assistant"
    assert flat[-1]["prefix"] is True


def test_intermediate_assistant_gets_no_prefix():
    """Only the LAST assistant message is a prefill; mid-conversation turns don't get prefix."""
    flat = _flatten(
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hello"),
        ChatMessage(role="user", content="bye"),
    )
    assert all("prefix" not in m for m in flat)


def test_non_assistant_trailing_gets_no_prefix():
    flat = _flatten(ChatMessage(role="user", content="hi"))
    assert "prefix" not in flat[-1]
