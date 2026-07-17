"""Fast smoke tests for GLM-5.x streaming behavior and the leak-repair layer.

These exist to quickly *validate or falsify* hypotheses about the reported
GLM-5.2-preview looping/degenerate-output bug ("0,0.0.0.0..."), without needing
live provider calls. They run in milliseconds under plain pytest.

Hypotheses covered
------------------
- H1 (model fault): the upstream model emits degenerate repeating tokens; the
  proxy merely forwards them. -> ``test_degenerate_content_passes_through``.
- H2 (leak-repair buffering fault): the ``_TAIL_KEEP`` rolling buffer drops or
  re-emits content in a way that *creates* a loop. -> ``test_*_leak_*`` and
  ``test_tail_buffering_does_not_duplicate``.
- H3 (streaming-generator fault): finish/flush logic loses content or emits a
  spurious trailing stop. -> ``test_streaming_*``.
- H4 (reasoning leak): leaked XML in thinking is stripped. ->
  ``test_reasoning_leak_stripped``.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

import pytest
from unittest.mock import MagicMock

from uniinfer.core import ChatCompletionResponse, ChatMessage
from uniinfer.proxy_services.glm_leak_repair import GlmLeakInterceptor
from uniinfer.proxy_services.streaming import astream_response_generator


# --------------------------------------------------------------------------- #
# Repetition detector (mirrors the symptom the user reported: "0,0.0.0.0...")
# --------------------------------------------------------------------------- #
def detect_degenerate_repetition(text: str, min_run: int = 8) -> bool:
    """Return True if `text` contains a short token-unit repeated back-to-back,
    which is the hallmark of a stuck model.

    Catches both single-char stuck output ('aaaa...', '0000000') AND the actual
    reported symptom ('0,0.0.0.0.0.0...' = repeating 2- or 3-char unit). Checks
    repetition periods of 1, 2 and 3.

    Cheap, deterministic, useful as an assertion helper.
    """
    if not text or len(text) < min_run:
        return False
    n = len(text)
    for period in (1, 2, 3):
        needed = min_run
        run = 1
        for i in range(period, n, period):
            prev = text[i - period:i]
            cur = text[i:i + period]
            if cur == prev and len(cur) == period:
                run += 1
                if run * period >= needed:
                    return True
            else:
                run = 1
    return False


# --------------------------------------------------------------------------- #
# H1: degenerate upstream content must pass through unchanged (proxy innocent)
# --------------------------------------------------------------------------- #
def test_repetition_detector_flags_symptom():
    # The exact reported symptom and close variants.
    assert detect_degenerate_repetition("0,0.0.0.0.0.0.0.0.0.0.0.0.0")
    assert detect_degenerate_repetition("0.0.0.0.0.0.0.0.0.0.0.0")
    assert detect_degenerate_repetition("aaaaaaaa")


def test_repetition_detector_ignores_normal_text():
    assert not detect_degenerate_repetition("Hello, this is a normal sentence.")
    assert not detect_degenerate_repetition("1,2,3,4,5,6")  # counter, not stuck
    assert not detect_degenerate_repetition("The quick brown fox jumps.")


def test_degenerate_content_passes_through_interceptor():
    """If the model emits garbage, the interceptor (no tools offered) must not
    drop, duplicate, or alter it beyond its normal tail buffering."""
    gi = GlmLeakInterceptor(tools=None, model="glm-5.2-744b-preview")
    garbage = "0,0.0.0.0.0.0.0.0.0.0.0.0.0.0"
    safe, leak_done = gi.feed(garbage)
    # Either flushed immediately or buffered in the rolling tail; never lost.
    tail = gi.flush_tail()
    reconstructed = (safe or "") + (tail or "")
    assert reconstructed == garbage
    assert not leak_done
    assert not gi.has_leak()


# --------------------------------------------------------------------------- #
# H2: leak-repair correctness on the documented GLM-5.2 truncated-opener bug
# --------------------------------------------------------------------------- #
def _feed_all(gi: GlmLeakInterceptor, *chunks: str) -> tuple[str, list[dict] | None]:
    """Feed chunks sequentially; return (concatenated_safe, tool_calls_or_None)."""
    out = []
    tcs = None
    for c in chunks:
        safe, done = gi.feed(c)
        if safe:
            out.append(safe)
        if done and tcs is None:
            tcs = gi.reconstructed_tool_calls()
    tail = gi.flush_tail()
    if tail and not gi._leaking:
        out.append(tail)
    return "".join(out), tcs


def test_intact_toolcall_leak_is_repaired():
    tools = [{"function": {"name": "bash"}}]
    gi = GlmLeakInterceptor(tools=tools, model="glm-5.2-744b-preview")
    leaked = 'Running it.\n<tool_call>bash<arg_key>command</arg_key><arg_value>echo ok</arg_value></tool_call>'
    safe, tcs = _feed_all(gi, *leaked)
    assert "<arg_key>" not in safe
    assert "</tool_call>" not in safe
    assert "Running it." in safe
    assert tcs is not None
    assert tcs[0]["function"]["name"] == "bash"
    assert json.loads(tcs[0]["function"]["arguments"])["command"] == "echo ok"


def test_truncated_opener_leak_is_repaired_and_name_snapped():
    """The common GLM-5.2 bug: opening <tool_call> is dropped and the tool name
    fuses with preceding prose, e.g. 'Thebash<arg_key>...'."""
    tools = [{"function": {"name": "bash"}}]
    gi = GlmLeakInterceptor(tools=tools, model="glm-5.2-744b-preview")
    leaked = "I will use the table Thebash<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>"
    safe, tcs = _feed_all(gi, leaked)
    assert "<arg_key>" not in safe
    assert "Thebash" not in safe  # fused name must be removed from content
    assert "table" in safe  # real prose preserved
    assert tcs is not None
    assert tcs[0]["function"]["name"] == "bash"
    assert json.loads(tcs[0]["function"]["arguments"])["command"] == "ls"


def test_split_across_chunks_leak_is_repaired():
    """A leak marker split across two stream chunks must still be caught
    thanks to the rolling tail buffer."""
    tools = [{"function": {"name": "bash"}}]
    gi = GlmLeakInterceptor(tools=tools, model="glm-5.2-744b-preview")
    safe, tcs = _feed_all(
        gi,
        "Some prose here bash<arg_",      # marker split
        "key>command</arg_key><arg_value>pwd</arg_value></tool_call>",
    )
    assert "<arg_key>" not in safe
    assert "Some prose here" in safe
    assert tcs is not None
    assert tcs[0]["function"]["name"] == "bash"


def test_no_double_emit_when_structured_toolcalls_present():
    """If the provider already sent structured tool_calls, the interceptor must
    NOT reconstruct a second one (prevents duplicate/looped tool dispatch)."""
    tools = [{"function": {"name": "bash"}}]
    gi = GlmLeakInterceptor(tools=tools, model="glm-5.2-744b-preview")
    gi.note_structured_tool_calls()
    _feed_all(gi, "<tool_call>bash<arg_key>command</arg_key><arg_value>x</arg_value></tool_call>")
    assert gi.reconstructed_tool_calls() is None


def test_tail_buffering_does_not_duplicate():
    """Short normal content (under _TAIL_KEEP) must come out exactly once after
    flush_tail — the buffer must never re-emit bytes it already flushed."""
    gi = GlmLeakInterceptor(tools=None, model="glm-5.2-744b-preview")
    msg = "Hello world"
    safe, _ = gi.feed(msg)
    # Under _TAIL_KEEP so nothing is flushed mid-stream; all held in tail.
    assert safe in (None, "")
    assert gi.flush_tail() == msg
    assert gi.flush_tail() is None  # not re-emitted


# --------------------------------------------------------------------------- #
# H4: leaked XML in reasoning_content is stripped
# --------------------------------------------------------------------------- #
def test_reasoning_leak_stripped():
    gi = GlmLeakInterceptor(tools=[{"function": {"name": "bash"}}], model="glm-5.2-744b-preview")
    reasoning = "Let me think. <tool_call>bash<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>"
    safe, _ = gi.feed(reasoning)
    tail = gi.flush_tail()
    combined = (safe or "") + (tail or "")
    assert "<arg_key>" not in combined
    assert "</tool_call>" not in combined
    assert "Let me think." in combined


# --------------------------------------------------------------------------- #
# H3: streaming generator integration (end-to-end, mocked upstream)
# --------------------------------------------------------------------------- #
async def _aiter(chunks: list[Any]) -> AsyncIterator[Any]:
    for c in chunks:
        await asyncio.sleep(0)
        yield c


def _dict_delta(content: str | None = None, finish: str | None = None) -> dict:
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    choice: dict[str, Any] = {"index": 0, "delta": delta}
    if finish:
        choice["finish_reason"] = finish
    return {"choices": [choice]}


def _raw_chunk(content: str | None = None, finish: str | None = None, thinking: str | None = None):
    """A raw ChatCompletionResponse chunk, as Target.astream_complete now yields."""
    return ChatCompletionResponse(
        message=ChatMessage(role="assistant", content=content),
        provider="glm",
        model="glm-5.2-744b-preview",
        usage={},
        raw_response={},
        finish_reason=finish,
        thinking=thinking,
    )


def _collect_stream(gen) -> tuple[str, str | None, list[dict]]:
    """Drive the SSE generator; return (content, finish_reason, tool_call_deltas)."""
    content_parts: list[str] = []
    finish_reason = None
    tool_calls: list[dict] = []

    async def drive():
        async for raw in gen:
            assert raw.startswith("data: "), raw
            payload = raw[len("data: "):].strip()
            if payload == "[DONE]":
                return
            data = json.loads(payload)
            if "error" in data:
                raise AssertionError(f"stream error: {data['error']}")
            for choice in data.get("choices", []):
                delta = choice.get("delta", {}) or {}
                if delta.get("content"):
                    content_parts.append(delta["content"])
                if delta.get("tool_calls"):
                    tool_calls.extend(delta["tool_calls"])
                if choice.get("finish_reason"):
                    finish_reason_ = choice["finish_reason"]
                    return "".join(content_parts), finish_reason_, tool_calls
            # continue after loop
        return "".join(content_parts), None, tool_calls

    res = asyncio.run(drive())
    return res


def _run_stream(content_chunks: list[str], tools=None) -> tuple[str, str | None, list[dict]]:
    def fake_astream_complete(messages, *, temperature, max_tokens, **kw):
        async def _g():
            for c in content_chunks:
                yield _raw_chunk(content=c)
            yield _raw_chunk(finish="stop")
        return _g()

    fake_target = MagicMock()
    fake_target.provider_model = "glm-5.2-744b-preview"
    fake_target.astream_complete = fake_astream_complete

    gen = astream_response_generator(
        target=fake_target,
        messages=[{"role": "user", "content": "hi"}],
        temp=0.7,
        max_tok=1024,
        tools=tools,
    )
    return _collect_stream(gen)


def test_streaming_normal_content_round_trips():
    content, finish, tcs = _run_stream(["Hello ", "world"])
    assert content == "Hello world"
    assert finish == "stop"
    assert tcs == []


def test_streaming_degenerate_content_is_forwarded_not_amplified():
    """The actual reported symptom: degenerate '0,0.0...' tokens. The proxy must
    forward them exactly once, finish cleanly, and not loop/repeat internally."""
    garbage = ["0,0.", "0.0.0.", "0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0"]
    content, finish, tcs = _run_stream(garbage)
    assert content == "".join(garbage)
    assert finish == "stop"
    assert tcs == []
    # Sanity: exactly one occurrence of the input (no doubling by the proxy).
    assert content.count("0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0") == 1


def test_streaming_leaked_toolcall_is_repaired_end_to_end():
    tools = [{"type": "function", "function": {"name": "bash"}}]
    leak = [
        "I'll run it. ",
        "bash<arg_key>command</arg_key><arg_value>echo hi</arg_value></tool_call>",
    ]
    content, finish, tcs = _run_stream(leak, tools=tools)
    assert "<arg_key>" not in content
    assert "</tool_call>" not in content
    assert "I'll run it." in content
    assert tcs and tcs[0]["function"]["name"] == "bash"
    assert json.loads(tcs[0]["function"]["arguments"])["command"] == "echo hi"
    # KNOWN GAP (validated here, see test_leaked_toolcall_finish_reason_gap):
    # the proxy forwards the upstream "stop" instead of upgrading to
    # "tool_calls". A well-behaved client dispatches via the tool_calls delta
    # regardless; a finish_reason-strict client may not.
    assert finish == "stop"


def test_leaked_toolcall_finish_reason_gap():
    """Documents the GLM leak + vLLM finish_reason mismatch.

    When GLM leaks a tool call into content, TU/vLLM emits finish_reason="stop".
    The leak-repair layer reconstructs a structured tool_calls delta but does NOT
    rewrite the finish_reason. If this assertion ever flips to "tool_calls",
    the gap has been fixed in streaming.py.
    """
    tools = [{"type": "function", "function": {"name": "bash"}}]
    leak = ["bash<arg_key>command</arg_key><arg_value>x</arg_value></tool_call>"]
    content, finish, tcs = _run_stream(leak, tools=tools)
    assert tcs, "expected reconstructed tool_call delta"
    assert finish == "stop"  # <- the gap; should ideally be "tool_calls"


def test_streaming_empty_content_still_finishes():
    content, finish, tcs = _run_stream([])
    assert content == ""
    assert finish == "stop"


if __name__ == "__main__":
    # Allow `python test_glm_smoke.py` for instant local validation.
    pytest.main([__file__, "-v"])
