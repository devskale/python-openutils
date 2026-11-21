"""Responses-API translation — the normalization of chat onto OpenAI Responses.

uniinfer IS the normalization layer: ``POST /v1/responses`` is a surface every
chat provider speaks, translated here (issue uniinfer-responses-api, B').

Two halves, both pure and unit-testable:

- request side: Responses request → chat messages/tools/params
- response side: chat chunks (``format_chunk_to_openai`` dicts — the same
  normalized deltas the chat relay consumes) → Responses objects and SSE
  events (``response.created`` … ``response.completed``)

v1 scope is stateless: input, instructions, max_output_tokens, temperature,
stream, client-side function calling. previous_response_id / server-side
tools / background are rejected with actionable 400s (native fast-path is a
follow-up).
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional

_UNSUPPORTED = {
    "previous_response_id": "stateful chaining is not translated (v1); resend the full input or use a native-Responses backend directly",
    "background": "background runs are not translated (v1)",
    "truncation": "truncation strategy is not translated (v1)",
}


def _sse(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


def reject_unsupported(body: dict) -> Optional[str]:
    """Return an actionable error message for v1-unsupported params, else None."""
    for key, msg in _UNSUPPORTED.items():
        if body.get(key) not in (None, False):
            return f"'{key}': {msg}"
    tools = body.get("tools") or []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") not in (None, "function"):
            return (
                f"server-side tool '{t.get('type')}' is not translated (v1); "
                "only client-side 'function' tools work on every provider — "
                "native server tools need a native-Responses backend"
            )
    return None


def _content_parts_to_chat(content: Any) -> Any:
    """Responses content (string | parts[]) → chat content (string | parts[])."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content
    out = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype in ("input_text", "output_text", "text"):
            out.append({"type": "text", "text": part.get("text", "")})
        elif ptype == "input_image":
            img: dict[str, Any] = {"url": part.get("image_url")}
            if part.get("detail"):
                img["detail"] = part["detail"]
            out.append({"type": "image_url", "image_url": img})
        else:
            out.append(part)
    return out


def request_to_chat(body: dict) -> dict:
    """Responses request → chat completion kwargs (messages, tools, params)."""
    messages: list[dict] = []
    if body.get("instructions"):
        messages.append({"role": "system", "content": body["instructions"]})

    items = body.get("input")
    if items is None:
        items = []
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]
    for item in items:
        if not isinstance(item, dict):
            continue
        itype = item.get("type", "message")
        if itype == "message":
            messages.append(
                {
                    "role": item.get("role", "user"),
                    "content": _content_parts_to_chat(item.get("content")),
                }
            )
        elif itype == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", "{}"),
                            },
                        }
                    ],
                    "content": None,
                }
            )
        elif itype == "function_call_output":
            output = item.get("output", "")
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or "",
                    "content": output,
                }
            )
        elif itype == "reasoning":
            continue

    tools: Optional[list[dict]] = None
    if body.get("tools"):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters") or {"type": "object", "properties": {}},
                },
            }
            for t in body["tools"]
            if isinstance(t, dict) and t.get("type", "function") == "function" and t.get("name")
        ] or None

    chat: dict[str, Any] = {
        "messages": messages,
        "temperature": body.get("temperature", 0.7),
        "tools": tools,
        "tool_choice": body.get("tool_choice"),
        "max_tokens": body.get("max_output_tokens"),
    }
    return chat


def _usage_to_responses(usage: Any) -> Optional[dict]:
    if usage is None:
        return None
    if isinstance(usage, dict):
        pt = usage.get("prompt_tokens", 0) or 0
        ct = usage.get("completion_tokens", 0) or 0
        tt = usage.get("total_tokens", (pt + ct)) or (pt + ct)
        reasoning = 0
        det = usage.get("completion_tokens_details") or {}
        if isinstance(det, dict):
            reasoning = det.get("reasoning_tokens", 0) or 0
    else:
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        tt = getattr(usage, "total_tokens", pt + ct) or (pt + ct)
        reasoning = 0
        det = getattr(usage, "completion_tokens_details", None)
        if det is not None and getattr(det, "reasoning_tokens", None):
            reasoning = det.reasoning_tokens
    return {
        "input_tokens": pt,
        "output_tokens": ct,
        "total_tokens": tt,
        "output_tokens_details": {"reasoning_tokens": reasoning},
    }


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def completion_to_response(
    full_content,
    provider_model: str,
) -> dict:
    """Non-stream: ChatCompletionResponse → Responses response object."""
    msg = getattr(full_content, "message", None)
    text = getattr(msg, "content", None) if msg else None
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    tool_calls = getattr(msg, "tool_calls", None) if msg else None
    finish_reason = getattr(full_content, "finish_reason", None)

    output: list[dict] = []
    if text:
        output.append(
            {
                "id": _new_id("msg"),
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": text, "annotations": []}
                ],
            }
        )
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        output.append(
            {
                "id": _new_id("fc"),
                "type": "function_call",
                "status": "completed",
                "call_id": tc.get("id") or _new_id("call"),
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments", "{}"),
            }
        )

    status = "completed"
    incomplete = None
    if finish_reason == "length":
        status = "incomplete"
        incomplete = {"reason": "max_output_tokens"}

    return {
        "id": _new_id("resp"),
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": provider_model,
        "output": output,
        "usage": _usage_to_responses(getattr(full_content, "usage", None)),
        "incomplete_details": incomplete,
        "parallel_tool_calls": bool(tool_calls and len(tool_calls) > 1),
        "error": None,
        "metadata": {},
    }


class ResponsesEventEmitter:
    """Stateful chat-delta → Responses-SSE translator (one response).

    Feed it ``format_chunk_to_openai`` dicts; it yields ``event: …\\n\\ndata: …``
    strings, opening with ``response.created`` and closing with
    ``response.completed`` carrying the assembled response object.
    """

    def __init__(self, provider_model: str) -> None:
        self.provider_model = provider_model
        self.response_id = _new_id("resp")
        self.created_at = int(time.time())
        self._text = []
        self._calls: dict[int, dict] = {}
        self._usage: Optional[dict] = None
        self._finish_reason: Optional[str] = None
        self._msg_item_id: Optional[str] = None
        self._part_open = False
        self._started = False

    def _base_response(self, status: str, output: list[dict]) -> dict:
        return {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.provider_model,
            "output": output,
            "usage": self._usage,
            "error": None,
        }

    def start(self):
        """Yield the opening events (response.created + in_progress)."""
        self._started = True
        yield _sse({"type": "response.created", "response": self._base_response("in_progress", [])})
        yield _sse({"type": "response.in_progress", "response": self._base_response("in_progress", [])})

    def feed(self, chunk: dict):
        """Yield events for one normalized chat chunk (or None when idle)."""
        if not self._started:
            yield from self.start()
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        if chunk.get("usage"):
            self._usage = _usage_to_responses(chunk["usage"])

        text = delta.get("content")
        if text:
            if self._msg_item_id is None:
                self._msg_item_id = _new_id("msg")
                yield _sse(
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "id": self._msg_item_id,
                            "type": "message",
                            "role": "assistant",
                            "status": "in_progress",
                            "content": [],
                        },
                    }
                )
                yield _sse(
                    {
                        "type": "response.content_part.added",
                        "item_id": self._msg_item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    }
                )
                self._part_open = True
            self._text.append(text)
            yield _sse(
                {
                    "type": "response.output_text.delta",
                    "item_id": self._msg_item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": text,
                }
            )

        for frag in delta.get("tool_calls") or []:
            if not isinstance(frag, dict):
                continue
            idx = frag.get("index", 0) or 0
            call = self._calls.get(idx)
            fn = frag.get("function") or {}
            if call is None:
                call = {
                    "id": _new_id("fc"),
                    "call_id": frag.get("id") or _new_id("call"),
                    "name": fn.get("name") or "",
                    "arguments": "",
                }
                self._calls[idx] = call
                yield _sse(
                    {
                        "type": "response.output_item.added",
                        "output_index": 1 + idx,
                        "item": {
                            "id": call["id"],
                            "type": "function_call",
                            "status": "in_progress",
                            "call_id": call["call_id"],
                            "name": call["name"],
                            "arguments": "",
                        },
                    }
                )
            if fn.get("name"):
                call["name"] = fn["name"]
            if fn.get("arguments"):
                call["arguments"] += fn["arguments"]
                yield _sse(
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": call["id"],
                        "output_index": 1 + idx,
                        "delta": fn["arguments"],
                    }
                )

        if choice.get("finish_reason"):
            self._finish_reason = choice["finish_reason"]

    def failed(self, error: Any) -> str:
        """Terminal ``response.failed`` event for mid-stream upstream errors."""
        if not self._started:
            self._started = True
        resp = self._base_response("failed", [])
        resp["error"] = error if isinstance(error, dict) else {"message": str(error)}
        return _sse({"type": "response.failed", "response": resp})

    def finish(self):
        """Yield the closing events and the completed response object."""
        if not self._started:
            yield from self.start()
        if self._part_open:
            yield _sse(
                {
                    "type": "response.output_text.done",
                    "item_id": self._msg_item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": "".join(self._text),
                }
            )
            yield _sse(
                {
                    "type": "response.content_part.done",
                    "item_id": self._msg_item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": "".join(self._text),
                        "annotations": [],
                    },
                }
            )
            yield _sse(
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "id": self._msg_item_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "".join(self._text),
                                "annotations": [],
                            }
                        ],
                    },
                }
            )
        for idx, call in sorted(self._calls.items()):
            yield _sse(
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": call["id"],
                    "output_index": 1 + idx,
                    "arguments": call["arguments"],
                }
            )
            yield _sse(
                {
                    "type": "response.output_item.done",
                    "output_index": 1 + idx,
                    "item": {
                        "id": call["id"],
                        "type": "function_call",
                        "status": "completed",
                        "call_id": call["call_id"],
                        "name": call["name"],
                        "arguments": call["arguments"],
                    },
                }
            )

        output: list[dict] = []
        if self._msg_item_id:
            output.append(
                {
                    "id": self._msg_item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "".join(self._text),
                            "annotations": [],
                        }
                    ],
                }
            )
        for idx, call in sorted(self._calls.items()):
            output.append(
                {
                    "id": call["id"],
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call["call_id"],
                    "name": call["name"],
                    "arguments": call["arguments"],
                }
            )
        status = "completed"
        incomplete = None
        if self._finish_reason == "length":
            status = "incomplete"
            incomplete = {"reason": "max_output_tokens"}
        final = self._base_response(status, output)
        final["incomplete_details"] = incomplete
        yield _sse({"type": "response.completed", "response": final})
