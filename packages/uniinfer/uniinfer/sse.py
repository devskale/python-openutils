"""Canonical SSE decoding — one seam instead of per-provider hand parsing.

Wraps the OpenAI SDK's battle-tested ``SSEDecoder`` (the same import LiteLLM
uses in production) so no call site hand-rolls ``startswith("data: ")`` chains
anymore. Correctly handles multi-line ``data:`` fields, ``event:`` names, CRLF,
BOM, comment lines, and ``id:`` fields per the SSE spec.

If the SDK ever moves its internal module, this file is the single import
site to fix.
"""
from __future__ import annotations

from typing import AsyncIterable, Iterable, Iterator

from openai._streaming import ServerSentEvent, SSEDecoder

DONE = "[DONE]"


def sse_events(lines: Iterable[str]) -> Iterator[ServerSentEvent]:
    """Decode already-split SSE lines into events (sync).

    Tolerates CRLF line endings (``aiter_lines`` keeps the trailing ``\r``)
    and flushes a pending event at stream end (some servers omit the final
    blank line before closing).
    """
    decoder = SSEDecoder()
    for line in lines:
        event = decoder.decode(line.rstrip("\r") if line.endswith("\r") else line)
        if event is not None:
            yield event
    tail = decoder.decode("")
    if tail is not None:
        yield tail


async def sse_events_aiter(lines: AsyncIterable[str]) -> AsyncIterable[ServerSentEvent]:
    """Decode an async line stream (``response.aiter_lines()``) into events."""
    decoder = SSEDecoder()
    async for line in lines:
        event = decoder.decode(line.rstrip("\r") if line.endswith("\r") else line)
        if event is not None:
            yield event
    tail = decoder.decode("")
    if tail is not None:
        yield tail


__all__ = ["DONE", "ServerSentEvent", "sse_events", "sse_events_aiter"]
