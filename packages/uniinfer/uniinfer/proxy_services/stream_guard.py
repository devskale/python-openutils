"""Empty-stream + first-token-stall guard at the proxy relay seam.

Incident (2026-10-22, ``dgx-vllm-leerstrom-unter-last`` /
``uniinfer-edge-grosse-hab-streams-leer``): upstreams end SSE streams with
HTTP 200 and ZERO content — vLLM's instant-EOS under load. The relay forwards
that as a clean empty "success"; clients (agentos, llminvoke's backup chain)
see success-with-nothing and burn retries on a masked provider failure
(5x ``unparseable response ('')`` in the night logs).

This module wraps :func:`astream_response_generator` WITHOUT touching its
internals (RC#5 priming, GLM leak repair and close-race handling stay sealed).
Everything the client receives flows through the wrapper's ``yield``, so the
guard observes the wire shape itself:

* **EMPTY-STREAM** — a relay pass ends with no content / reasoning /
  tool_calls event (the 200-with-empty-body signature). One internal retry
  against the same provider (transient instant-EOS heals on replay — same
  finding as the TU provider's own healing), then a recognizable error chunk
  ``{"error": {"type": "empty_stream", "code": 502}}``. If nothing reached the
  client yet, a real ``ProviderError(502)`` is raised instead — the router
  turns that into a true HTTP 502 before the SSE 200 is committed, so the
  llminvoke backup chain (hab→tu→gemini) routes around the dead upstream.
* **FIRST-TOKEN-STALL** — no contentful event within
  ``UNIINFER_FIRST_TOKEN_STALL_S`` (default 30 s; the server-side view
  complementing agentos' 20-s client watchdog). The stalled pass is aborted
  (client-disconnect-equivalent close — the relay's own finally records stats
  and releases the upstream) and retried / failed like the empty case.

The ``[DONE]`` terminator of each pass is held back until the verdict is
known: on retry the next pass simply continues the SSE stream (a second
``role: assistant`` delta is legal, OpenAI SDK clients concatenate deltas);
on failure the error chunk replaces the clean termination. A pass that
already emitted an error chunk (idle timeout, provider error) is forwarded
unchanged — the failure is visible, the guard adds nothing. The FIRST event
of each pass is also held until a second event proves the pass alive (see
``held_first`` in the code) — that keeps the real-HTTP-502 window open
through the relay's cancellation-absorbing priming phase.

Telemetry: every detection emits an ``ALARM {json}`` line in the
llminvoke/robotni-alarms schema so the empty stream never hides behind a
200 again.

Config (env, read per request — no restart to tune):

* ``UNIINFER_EMPTY_STREAM_DETECT``    — "0"/"off" disables the empty verdict
  (stall guard independent, see below). Default on.
* ``UNIINFER_EMPTY_STREAM_RETRIES``   — internal retries per request before
  failing. Default 1.
* ``UNIINFER_FIRST_TOKEN_STALL_S``    — 0 disables the stall guard. Default 30.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, AsyncGenerator

from uniinfer.core import ChatCompletionResponse
from uniinfer.errors import ProviderError
from uniinfer.proxy_services.streaming import (
    _close_upstream_stream,
    astream_response_generator,
)

logger = logging.getLogger("uniioai_proxy")

#: Substring pre-filter before json parsing — the relay only puts these keys
#: into a chunk when the value is non-null (exclude_none / explicit sets), so
#: a miss proves the chunk carries no payload. Cheap at high chunk rates.
_PAYLOAD_KEYS = (
    '"content"', '"tool_calls"', '"reasoning_content"', '"thinking"',
    '"reasoning_details"',
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, "")))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def empty_guard_enabled() -> bool:
    """Empty-stream verdict on? (env, read at request time)."""
    return _env_bool("UNIINFER_EMPTY_STREAM_DETECT", True)


def empty_guard_retry_budget() -> int:
    """Internal retries before an empty verdict fails the request."""
    return max(0, _env_int("UNIINFER_EMPTY_STREAM_RETRIES", 1))


def emit_guard_alarm(error_type: str, provider_model: str, message: str) -> None:
    """Emit an ALARM-marker line in the llminvoke/robotni-alarms schema.

    robotni-alarms ingests ``ALARM {json}`` lines (the contract
    llminvoke.config.emit_alarm produces); the proxy's own detections use the
    same marker so they land in the same feed without a new channel. Only the
    ``ALARM `` marker + the JSON after it is stable; the logger prefix varies.
    """
    provider = provider_model.split("@", 1)[0] if "@" in provider_model else provider_model
    entry = {
        "ts": time.time(),
        "severity": "alarm",
        "provider": provider,
        "model": provider_model,
        "error_type": error_type,
        "message": message[:500],
        "package": "uniinfer",
        "client": None,
    }
    logger.warning("ALARM %s", json.dumps(entry, ensure_ascii=False))


def nonstream_masked_empty(response: ChatCompletionResponse) -> bool:
    """True when a non-streaming completion is a masked empty failure.

    Empty content is legitimate when the budget went to thinking
    (``thinking``/``reasoning_details`` present) or the answer is a tool call
    — the same exemptions the TU provider's rate-limit-shadow guard uses.
    """
    msg = getattr(response, "message", None)
    if msg is None:
        return False
    if msg.content:
        return False
    if msg.tool_calls:
        return False
    if getattr(response, "thinking", None):
        return False
    if getattr(response, "reasoning_details", None):
        return False
    return True


class _PassObserver:
    """Observes one relay pass by inspecting the SSE events it yields.

    Pure (no I/O) — directly unit-testable. ``observe`` takes one yielded
    relay event (an SSE frame string) and classifies it; the verdict
    properties answer the guard's two questions: did anything contentful
    reach the client, and did the relay itself already surface an error.
    """

    __slots__ = ("content_events", "saw_error")

    def __init__(self) -> None:
        self.content_events = 0
        self.saw_error = False

    def observe(self, event: str) -> None:
        if not event.startswith("data: {"):
            return  # keep-alive comments, [DONE], non-JSON frames
        # Cheap pre-filter: a chunk without any payload key substring cannot
        # be contentful (the relay only emits these keys for non-null values).
        if not any(key in event for key in _PAYLOAD_KEYS):
            if '"error"' not in event:
                return
        try:
            data = json.loads(event[6:])
        except ValueError:
            return
        if not isinstance(data, dict):
            return
        if isinstance(data.get("error"), dict):
            self.saw_error = True
            return
        for choice in data.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content") or delta.get("tool_calls") \
                    or delta.get("reasoning_content") or delta.get("thinking"):
                self.content_events += 1
                return

    @property
    def has_payload(self) -> bool:
        return self.content_events > 0


class _StallAborted(Exception):
    """Internal: the pass produced no contentful event within the stall budget."""


async def astream_guarded(
    *,
    target,
    messages: list[dict],
    temp: float,
    max_tok: int,
    tools: list[dict] | None = None,
    tool_choice: Any | None = None,
    request_id: str | None = None,
    reasoning_effort: str | None = None,
    chat_template_kwargs: dict | None = None,
    extra: dict | None = None,
) -> AsyncGenerator[str, None]:
    """Relay ``astream_response_generator`` with empty-stream protection.

    Same signature/semantics as the wrapped generator — a drop-in seam for the
    chat router. Holds each pass's ``[DONE]`` until the verdict is known;
    retries empty/stalled passes against the same provider; fails loudly
    (error chunk 502, or a real ProviderError 502 while nothing was sent).
    """
    detect = empty_guard_enabled()
    retries = empty_guard_retry_budget()
    stall_s = max(0.0, _env_float("UNIINFER_FIRST_TOKEN_STALL_S", 30.0))

    if not detect and stall_s <= 0:
        async for event in astream_response_generator(
            target=target, messages=messages, temp=temp, max_tok=max_tok,
            tools=tools, tool_choice=tool_choice, request_id=request_id,
            reasoning_effort=reasoning_effort,
            chat_template_kwargs=chat_template_kwargs, extra=extra,
        ):
            yield event
        return

    model_name = target.provider_model
    stream_label = f"[{request_id}] " if request_id else ""
    yielded = False
    attempt = 0  # retries consumed so far

    while True:
        _pass_t0 = time.monotonic()
        observer = _PassObserver()
        # First event of the pass is HELD until a second event proves the pass
        # is alive. The relay's priming phase absorbs cancellation (its
        # ``except BaseException`` turns an aborted pull into an early welcome
        # chunk), so a stall abort during prime would otherwise arrive as a
        # yielded welcome — closing the real-502 window before it opened. While
        # the hold is up, NOTHING has reached the client: an exhausted verdict
        # can still raise ProviderError(502) and the router answers with a true
        # HTTP 502 instead of an SSE error chunk mid-200.
        held_first: str | None = None
        inner = astream_response_generator(
            target=target, messages=messages, temp=temp, max_tok=max_tok,
            tools=tools, tool_choice=tool_choice, request_id=request_id,
            reasoning_effort=reasoning_effort,
            chat_template_kwargs=chat_template_kwargs, extra=extra,
        )
        stalled = False
        try:
            while True:
                # Stall budget: pass start → first contentful event. Keep-alive
                # comments and the welcome chunk do NOT satisfy it (they carry
                # no payload) — a silent-but-alive upstream still stalls. Once
                # payload flowed, the relay's own idle/heartbeat machinery owns
                # mid-stream staleness. Independent of UNIINFER_EMPTY_STREAM_DETECT.
                if stall_s > 0 and not observer.has_payload:
                    remaining = stall_s - (time.monotonic() - _pass_t0)
                    if remaining <= 0:
                        raise _StallAborted
                    try:
                        event = await asyncio.wait_for(inner.__anext__(), timeout=remaining)
                    except asyncio.TimeoutError:
                        raise _StallAborted from None
                else:
                    event = await inner.__anext__()
                if event.startswith("data: [DONE]"):
                    break  # held — the verdict decides what follows
                if held_first is None and not yielded:
                    held_first = event
                    continue
                if held_first is not None:
                    observer.observe(held_first)
                    yield held_first
                    yielded = True
                    held_first = None
                observer.observe(event)
                yield event
                yielded = True
        except _StallAborted:
            stalled = True
        except StopAsyncIteration:
            pass
        finally:
            # Race-safe close of the relay generator: no-op when it exhausted
            # normally; on stall (wait_for cancelled + awaited the pull, so the
            # relay's finally already ran) or exception, releases it like a
            # client disconnect would.
            await _close_upstream_stream(inner)

        if stalled:
            emit_guard_alarm(
                "first_token_stall",
                model_name,
                f"FIRST-TOKEN-STALL: no contentful event within {stall_s:.0f}s "
                f"(attempt {attempt + 1}/{retries + 1}) — pass aborted",
            )
            logger.warning(
                "%sFirst-token stall on %s — no payload within %.0fs, aborting pass",
                stream_label, model_name, stall_s,
            )
        elif not detect:
            yield "data: [DONE]\n\n"
            return
        elif observer.saw_error:
            # The relay already surfaced a failure (idle timeout, provider
            # error) — visible to the client, nothing to mask.
            yield "data: [DONE]\n\n"
            return
        elif observer.has_payload:
            yield "data: [DONE]\n\n"
            return
        else:
            emit_guard_alarm(
                "empty_stream",
                model_name,
                f"EMPTY-STREAM: upstream ended 200 with zero content "
                f"(attempt {attempt + 1}/{retries + 1})",
            )
            logger.warning(
                "%sEmpty stream from %s — 200 with no content (attempt %d/%d)",
                stream_label, model_name, attempt + 1, retries + 1,
            )

        # Verdict is stall or empty: retry while budget lasts.
        if attempt < retries:
            attempt += 1
            continue

        # Budget spent — fail loudly. A real ProviderError while nothing was
        # yielded lets the router answer with a true HTTP 502 (backup chains
        # route around it); once SSE bytes went out, the error chunk replaces
        # the clean termination.
        if stalled:
            message = (
                f"First-token stall: no contentful event within {stall_s:.0f}s "
                f"after {attempt + 1} attempt(s)"
            )
        else:
            message = (
                f"Empty stream: upstream returned 200 with no content after "
                f"{attempt + 1} attempt(s)"
            )
        if not yielded:
            raise ProviderError(message, status_code=502)
        error_chunk = {
            "error": {
                "message": message,
                "type": "first_token_stall" if stalled else "empty_stream",
                "code": 502,
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        return
