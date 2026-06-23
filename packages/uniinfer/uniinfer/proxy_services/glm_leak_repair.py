"""GLM-5.x streaming tool-call leak repair.

The TU Aqueduct vLLM serving layer (tool parser `glm47`) intermittently fails to
intercept GLM-5.2's native XML tool-call format during streaming. Instead of a
structured `tool_calls` delta, the call leaks into `content` as raw XML, e.g.:

    ...normal prose...<tool_call>bash<arg_key>command</arg_key>
    <arg_value>echo "ok"</arg_value></tool_call>

often with the opening `<tool_call>` tag truncated by the buggy parser, e.g.

    ...prose Thebash<arg_key>command</arg_key><arg_value>...e</arg_value></tool_call>

This module detects the leak mid-stream, suppresses the leaked bytes from the
content stream, and reconstructs a structured OpenAI-style `tool_calls` entry
that the agent loop can consume normally.

Known-upstream: vLLM #39757, #42400, #36857, vllm-ascend #8154.

It is deliberately best-effort and defensive: if anything looks ambiguous, it
leaves the content untouched so the behaviour never gets *worse* than the raw
upstream output.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("uniioai_proxy")

# Markers of GLM-5.x's native XML tool-call format.
_LEAK_MARKERS = ("<arg_key>", "</tool_call>", "<arg_value>", "<tool_call>")
_ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)
_OPEN_TAG_RE = re.compile(r"</?tool_call>")
# Tool-name token: word chars / dash / dot right before the first <arg_key>.
_NAME_TOKEN_RE = re.compile(r"([A-Za-z0-9_.\-]+)\s*<arg_key>")

# How much un-flushed content tail we keep around so a tag that straddles two
# stream chunks can still be detected and repaired.
_TAIL_KEEP = 64


class GlmLeakInterceptor:
    """Per-stream accumulator that repairs leaked GLM XML tool calls.

    Usage:
        gi = GlmLeakInterceptor(tools=tools)
        for content_chunk in ...:
            safe, leak_done = gi.feed(content_chunk)
            if safe is not None:
                yield content delta with `safe`
            if leak_done:
                tc = gi.reconstructed_tool_calls()  # emit as tool_calls delta
    """

    def __init__(self, tools: list[dict] | None = None, model: str | None = None):
        self.tools = tools or []
        self.model = model or ""
        # Content we have not yet flushed downstream (small rolling tail).
        self._pending = ""
        # Once a leak is detected we stop flushing and collect the leaked XML.
        self._leaking = False
        self._leak_buf = ""
        self._flushed_leak = False
        # Track whether *any* structured tool_calls came through the normal path
        # (if so, we don't need to repair anything).
        self.saw_structured_tool_calls = False

    # ------------------------------------------------------------------ public

    def note_structured_tool_calls(self) -> None:
        """Call this if a real `delta.tool_calls` is seen in the stream."""
        self.saw_structured_tool_calls = True

    def feed(self, content: str | None) -> tuple[str | None, bool]:
        """Accept one content chunk; return (safe_to_flush, leak_complete).

        - safe_to_flush: content that is definitely not part of a leaked tool
          call and should be forwarded to the client now. None => nothing.
        - leak_complete: True once the full leaked XML (ending in
          </tool_call>) has been collected and can be reconstructed.
        """
        if not content:
            return None, False

        if self._leaking:
            self._leak_buf += content
            done = "</tool_call>" in self._leak_buf
            return None, done

        self._pending += content

        if not self._contains_leak_marker(self._pending):
            # No leak yet: flush everything except a small tail (a marker could
            # be split across chunks).
            if len(self._pending) > _TAIL_KEEP:
                flush = self._pending[:-_TAIL_KEEP]
                self._pending = self._pending[-_TAIL_KEEP:]
                return flush, False
            return None, False

        # A leak marker is present. Split into clean content + leaked region.
        clean, leaked = self._split_at_leak(self._pending)
        self._leaking = True
        self._leak_buf = leaked
        done = "</tool_call>" in self._leak_buf
        self._pending = ""
        return clean, done

    def flush_tail(self) -> str | None:
        """Flush any leftover pending content at stream end (no leak detected)."""
        if self._leaking or self._flushed_leak:
            return None
        if self._pending:
            tail = self._pending
            self._pending = ""
            return tail
        return None

    def has_leak(self) -> bool:
        return self._leaking and bool(self._leak_buf)

    def reconstructed_tool_calls(self) -> list[dict] | None:
        """Parse the collected leak buffer into OpenAI tool_calls entries.

        Returns None if reconstruction fails (caller should fall back to
        emitting the raw content unchanged).
        """
        if not self.has_leak():
            return None
        if self.saw_structured_tool_calls:
            # Real tool_calls already streamed; don't double-emit.
            return None

        buf = self._leak_buf
        # Strip surviving <tool_call> tags.
        cleaned = _OPEN_TAG_RE.sub("", buf)

        args = _ARG_RE.findall(cleaned)
        if not args:
            logger.warning(
                "GLM leak repair: could not parse args from %r (model=%s)",
                buf, self.model,
            )
            return None

        # Build arguments dict. Preserve order; later keys override duplicates.
        arguments: dict[str, Any] = {}
        for key, value in args:
            key = key.strip()
            value = value.strip()
            # Common case: a single "command"/"code" style arg holding JSON or text.
            try:
                arguments[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                arguments[key] = value

        # If exactly one arg and its value is a dict, unwrap it (many tools want
        # the object directly).
        if len(arguments) == 1:
            only = next(iter(arguments.values()))
            if isinstance(only, dict):
                arguments = only

        name = self._extract_name(cleaned)

        tool_call = {
            "id": "chatcmpl-glmleak",
            "type": "function",
            # Streaming tool_call deltas MUST carry an `index` so OpenAI-style
            # clients (incl. pi) can coalesce argument fragments by call slot.
            "index": 0,
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }
        logger.info(
            "GLM leak repair: reconstructed tool_call name=%s args=%s (model=%s)",
            name, tool_call["function"]["arguments"], self.model,
        )
        self._flushed_leak = True
        return [tool_call]

    # ------------------------------------------------------------------ private

    @staticmethod
    def _contains_leak_marker(text: str) -> bool:
        return any(m in text for m in _LEAK_MARKERS)

    def _split_at_leak(self, text: str) -> tuple[str, str]:
        """Split accumulated text into (clean_content, leaked_region).

        The leaked region reliably ends at </tool_call> and contains
        <arg_key>/<arg_value> pairs. Its start is either an intact <tool_call>
        tag or — when the opener is truncated (the common GLM-5.2 bug) — the
        tool-name token immediately preceding the first <arg_key>.
        """
        arg_pos = text.find("<arg_key>")
        close_pos = text.find("</tool_call>")
        open_pos = text.find("<tool_call>")

        # Prefer an intact opening tag if it sits before the args.
        if open_pos != -1 and open_pos < arg_pos:
            return text[:open_pos], text[open_pos:]

        # No intact opener: the tool name sits right before <arg_key>. Snap the
        # boundary to the start of that name token. We then validate/match the
        # name against the known tools list to correct any over-capture.
        if arg_pos != -1:
            m = _NAME_TOKEN_RE.search(text)
            if m:
                start = m.start(1)
            else:
                start = arg_pos
            start = self._snap_boundary_for_tool_name(text, start, arg_pos)
            return text[:start], text[start:]

        # Only </tool_call> seen (no arg_key yet): cut at the closing tag's region.
        if close_pos != -1:
            return text[:close_pos], text[close_pos:]

        return text, ""

    def _snap_boundary_for_tool_name(self, text: str, start: int, arg_pos: int) -> int:
        """If the captured name token includes trailing real content because
        the opener was truncated, try to align `start` to the actual tool name
        using the known tools list.

        Example: "...the table Thebash<arg_key>" with tools=[bash] -> we want
        start positioned at "bash", not at "Thebash".
        """
        candidate = text[start:arg_pos]
        names = self._known_tool_names()
        if not names:
            return start
        # Longest-name-first to avoid matching "sh" inside "bash".
        for tname in sorted(names, key=len, reverse=True):
            if candidate.endswith(tname) or tname in candidate:
                idx = candidate.rfind(tname)
                if idx >= 0:
                    return start + idx
        return start

    def _extract_name(self, cleaned: str) -> str:
        """Extract the tool name from the cleaned leak buffer, matching the
        known tools list when possible."""
        m = _NAME_TOKEN_RE.search(cleaned)
        raw = m.group(1).strip() if m else ""
        names = self._known_tool_names()
        if not raw and names:
            return names[0]
        if not names:
            return raw or "unknown"
        for tname in sorted(names, key=len, reverse=True):
            if tname == raw or raw.endswith(tname) or tname in raw:
                return tname
        return raw or (names[0] if names else "unknown")

    def _known_tool_names(self) -> list[str]:
        out = []
        for t in self.tools or []:
            fn = t.get("function") if isinstance(t, dict) else None
            if fn and fn.get("name"):
                out.append(fn["name"])
        return out
