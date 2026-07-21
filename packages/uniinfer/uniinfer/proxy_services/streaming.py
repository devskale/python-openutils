import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, AsyncGenerator

from uniinfer.errors import ProviderError, UniInferError
from uniinfer.proxy_schemas.chat import (
    ChoiceDelta,
    StreamingChatCompletionChunk,
    StreamingChoice,
)
from uniinfer.proxy_services.glm_leak_repair import GlmLeakInterceptor

logger = logging.getLogger("uniioai_proxy")


def is_openai_strict_mode() -> bool:
    return False


def normalize_nonstream_content(content: Any, tool_calls: Any) -> str | None:
    if tool_calls:
        return content
    if content is None and is_openai_strict_mode():
        return ""
    return content


def format_chunk_to_openai(response, provider_model: str) -> dict[str, Any]:
    """Format a raw ChatCompletionResponse chunk into an OpenAI-compatible dict.

    The proxy SSE layer consumes OpenAI-shaped chunks; Target yields raw
    ChatCompletionResponse, so this conversion sits at the proxy seam.
    """
    chunk_data = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": provider_model,
        "choices": [],
    }
    delta: dict[str, Any] = {}
    if response.message:
        if response.message.content:
            delta["content"] = response.message.content
        if response.message.tool_calls:
            delta["tool_calls"] = response.message.tool_calls
        if response.message.role:
            delta["role"] = response.message.role
    if getattr(response, "thinking", None):
        delta["thinking"] = response.thinking
    choice_data: dict[str, Any] = {"index": 0, "delta": delta}
    if getattr(response, "finish_reason", None):
        choice_data["finish_reason"] = response.finish_reason
    chunk_data["choices"] = [choice_data]
    if getattr(response, "usage", None):
        chunk_data["usage"] = response.usage
    return chunk_data


async def astream_response_generator(
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
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    model_name = target.provider_model

    stream_label = f"[{request_id}] " if request_id else ""
    logger.info("%sAsync stream start for %s", stream_label, model_name)
    _stats_t0 = time.monotonic()
    _stats_usage: dict = {}
    _stats_status = 200
    chunk_count = 0
    heartbeat_count = 0
    last_yield_time = time.monotonic()
    idle_warning_threshold = 10.0
    heartbeat_interval = float(
        os.getenv("UNIINFER_STREAM_HEARTBEAT", "120")
    )  # Increased for reasoning models
    idle_timeout = float(
        os.getenv("UNIINFER_STREAM_IDLE_TIMEOUT", "300")
    )  # Increased for reasoning models

    first_chunk_data = StreamingChatCompletionChunk(
        id=completion_id,
        created=created_time,
        model=model_name,
        choices=[StreamingChoice(delta=ChoiceDelta(role="assistant"))],
    )
    yield f"data: {first_chunk_data.model_dump_json(exclude_none=True)}\n\n"

    seen_tool_calls = False
    sent_finish_reason = False

    # GLM-5.x leak repair: detects leaked XML tool-calls in streamed content
    # and reconstructs them as structured tool_calls. Only active when tools
    # are offered; harmless for normal chat.
    leak_repair = GlmLeakInterceptor(tools=tools, model=model_name)
    # The TU vLLM glm47 tool parser also leaks <tool_call> XML into
    # reasoning_content (thinking). Strip it there too so the agent's thinking
    # renderer doesn't display raw XML; we don't reconstruct tool_calls from
    # reasoning leaks (the content path / structured deltas handle that).
    reasoning_leak_repair = GlmLeakInterceptor(tools=tools, model=model_name)

    try:
        async_iter = target.astream_complete(
            messages,
            temperature=temp,
            max_tokens=max_tok,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            chat_template_kwargs=chat_template_kwargs,
            extra=extra,
        ).__aiter__()
        while True:
            try:
                raw_chunk = await asyncio.wait_for(
                    async_iter.__anext__(), timeout=heartbeat_interval
                )
                # Capture usage if the backend emits it (often on the final chunk).
                _raw = getattr(raw_chunk, "raw_response", None)
                _chunk_usage = getattr(raw_chunk, "usage", None)
                if isinstance(_chunk_usage, dict) and _chunk_usage:
                    _stats_usage = _chunk_usage
                elif isinstance(_raw, dict) and isinstance(_raw.get("usage"), dict):
                    _stats_usage = _raw["usage"]
                # Target yields raw ChatCompletionResponse; convert to the
                # OpenAI-dict shape the chunk-shaping logic below consumes.
                chunk = format_chunk_to_openai(raw_chunk, model_name)
            except asyncio.TimeoutError:
                now = time.monotonic()
                idle_for = now - last_yield_time
                if idle_for >= idle_timeout:
                    error_chunk = {
                        "error": {
                            "message": f"Stream idle timeout after {idle_for:.2f}s",
                            "type": "stream_timeout",
                            "code": None,
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    sent_finish_reason = True
                    break

                heartbeat_count += 1
                yield ": keep-alive\n\n"
                continue
            except StopAsyncIteration:
                break

            if isinstance(chunk, dict):
                choice = chunk.get("choices", [{}])[0]
                delta = (
                    choice.get("delta", {})
                    if isinstance(choice.get("delta", {}), dict)
                    else {}
                )

                # Apply GLM leak repair to streamed content (only when tools offered).
                raw_content = delta.get("content") if delta else None
                extra_tc_chunks = []
                if raw_content:
                    safe, leak_done = leak_repair.feed(raw_content)
                    if safe is None:
                        # Interceptor is buffering a tail; hold ALL content back.
                        delta.pop("content", None)
                    elif safe == "":
                        delta.pop("content", None)
                    else:
                        delta["content"] = safe
                    if leak_done:
                        tcs = leak_repair.reconstructed_tool_calls()
                        if tcs:
                            extra_tc_chunks.append(tcs)
                if delta.get("tool_calls"):
                    seen_tool_calls = True
                    leak_repair.note_structured_tool_calls()

                # normalise thinking -> reasoning_content
                if delta:
                    if is_openai_strict_mode():
                        delta.pop("reasoning_content", None)
                        delta.pop("thinking", None)
                    else:
                        if delta.get("thinking") and not delta.get("reasoning_content"):
                            delta["reasoning_content"] = delta["thinking"]
                        delta.pop("thinking", None)
                        # Strip leaked tool-call XML from reasoning_content too
                        # (the glm47 parser leaks into thinking as well as content).
                        raw_reasoning = delta.get("reasoning_content")
                        if raw_reasoning:
                            rsafe, _ = reasoning_leak_repair.feed(raw_reasoning)
                            if rsafe is None or rsafe == "":
                                delta.pop("reasoning_content", None)
                            else:
                                delta["reasoning_content"] = rsafe

                if choice.get("finish_reason"):
                    # The provider is signalling completion. Flush any content
                    # still held in the leak-repair rolling tail BEFORE we emit
                    # the finish chunk, otherwise the buffered tail is lost
                    # (the post-loop flush_tail() only runs when no finish_reason
                    # was ever sent).
                    tail = leak_repair.flush_tail()
                    if tail:
                        tail_chunk = {
                            "id": chunk.get("id", completion_id),
                            "object": "chat.completion.chunk",
                            "created": chunk.get("created", created_time),
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {"content": tail}}],
                        }
                        yield f"data: {json.dumps(tail_chunk)}\n\n"
                        chunk_count += 1
                        last_yield_time = time.monotonic()
                    rtail = reasoning_leak_repair.flush_tail()
                    if rtail:
                        rtail_chunk = {
                            "id": chunk.get("id", completion_id),
                            "object": "chat.completion.chunk",
                            "created": chunk.get("created", created_time),
                            "model": model_name,
                            "choices": [
                                {"index": 0, "delta": {"reasoning_content": rtail}}
                            ],
                        }
                        yield f"data: {json.dumps(rtail_chunk)}\n\n"
                        chunk_count += 1
                        last_yield_time = time.monotonic()
                    sent_finish_reason = True

                # Yield the (possibly repaired) content chunk if it still has something.
                if delta or choice.get("finish_reason"):
                    yield f"data: {json.dumps(chunk)}\n\n"
                    chunk_count += 1
                    last_yield_time = time.monotonic()

                # Emit any reconstructed tool_calls as their own delta(s).
                for tcs in extra_tc_chunks:
                    tc_obj = {
                        "id": chunk.get("id", completion_id),
                        "object": "chat.completion.chunk",
                        "created": chunk.get("created", created_time),
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {"tool_calls": tcs}}],
                    }
                    yield f"data: {json.dumps(tc_obj)}\n\n"
                    chunk_count += 1
                    last_yield_time = time.monotonic()
                    seen_tool_calls = True
            else:
                chunk_finish_reason = getattr(chunk, "finish_reason", None)

                # Handle error finish_reason from provider (e.g., preemption detection)
                if chunk_finish_reason == "error":
                    import sys

                    print(
                        f"[DEBUG] STREAMING: Got error finish_reason, raw_response={chunk.raw_response}",
                        file=sys.stderr,
                        flush=True,
                    )
                    error_msg = "Stream error"
                    if chunk.raw_response and isinstance(chunk.raw_response, dict):
                        error_msg = chunk.raw_response.get("error", error_msg)
                    error_chunk = {
                        "error": {
                            "message": error_msg,
                            "type": "provider_error",
                            "code": None,
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    sent_finish_reason = True
                    continue

                if chunk.message:
                    choice_kwargs = {}
                    if chunk.message.tool_calls:
                        choice_kwargs["tool_calls"] = chunk.message.tool_calls
                        seen_tool_calls = True
                        leak_repair.note_structured_tool_calls()

                    # GLM leak repair: route content through the interceptor
                    # so leaked XML never reaches the client as prose.
                    raw_content = chunk.message.content
                    if raw_content:
                        safe, leak_done = leak_repair.feed(raw_content)
                        if safe is not None:
                            choice_kwargs["content"] = safe
                        if leak_done:
                            # Emit any reconstructed tool_calls as a delta.
                            tcs = leak_repair.reconstructed_tool_calls()
                            if tcs:
                                tc_chunk_data = StreamingChatCompletionChunk(
                                    id=completion_id,
                                    created=created_time,
                                    model=model_name,
                                    choices=[
                                        StreamingChoice(
                                            delta=ChoiceDelta(tool_calls=tcs)
                                        )
                                    ],
                                )
                                yield f"data: {tc_chunk_data.model_dump_json(exclude_none=True)}\n\n"
                                chunk_count += 1
                                last_yield_time = time.monotonic()
                                seen_tool_calls = True
                    else:
                        # No content this chunk; make sure the interceptor has a
                        # chance to see a (rare) empty-but-marker situation.
                        pass

                    if chunk.thinking and not is_openai_strict_mode():
                        rsafe, _ = reasoning_leak_repair.feed(chunk.thinking)
                        if rsafe is not None and rsafe != "":
                            choice_kwargs["reasoning_content"] = rsafe

                    choice_kwargs_with_delta = {
                        "delta": ChoiceDelta(**choice_kwargs)
                        if choice_kwargs
                        else ChoiceDelta()
                    }
                    if chunk_finish_reason:
                        # Flush any content still held in the leak-repair rolling
                        # tail before signalling completion (see dict branch).
                        tail = leak_repair.flush_tail()
                        if tail:
                            tail_chunk_data = StreamingChatCompletionChunk(
                                id=completion_id,
                                created=created_time,
                                model=model_name,
                                choices=[
                                    StreamingChoice(delta=ChoiceDelta(content=tail))
                                ],
                            )
                            yield f"data: {tail_chunk_data.model_dump_json(exclude_none=True)}\n\n"
                            chunk_count += 1
                            last_yield_time = time.monotonic()
                        rtail = reasoning_leak_repair.flush_tail()
                        if rtail:
                            rtail_chunk_data = StreamingChatCompletionChunk(
                                id=completion_id,
                                created=created_time,
                                model=model_name,
                                choices=[
                                    StreamingChoice(
                                        delta=ChoiceDelta(reasoning_content=rtail)
                                    )
                                ],
                            )
                            yield f"data: {rtail_chunk_data.model_dump_json(exclude_none=True)}\n\n"
                            chunk_count += 1
                            last_yield_time = time.monotonic()
                        choice_kwargs_with_delta["finish_reason"] = chunk_finish_reason
                        sent_finish_reason = True

                    if choice_kwargs or chunk_finish_reason:
                        chunk_data = StreamingChatCompletionChunk(
                            id=completion_id,
                            created=created_time,
                            model=model_name,
                            choices=[StreamingChoice(**choice_kwargs_with_delta)],
                        )
                        yield f"data: {chunk_data.model_dump_json(exclude_none=True)}\n\n"
                        chunk_count += 1
                        last_yield_time = time.monotonic()

        if not sent_finish_reason:
            # Flush any leftover pending content (e.g. a tail kept for
            # split-marker detection that never turned into a leak).
            tail = leak_repair.flush_tail()
            if tail:
                tail_chunk_data = StreamingChatCompletionChunk(
                    id=completion_id,
                    created=created_time,
                    model=model_name,
                    choices=[StreamingChoice(delta=ChoiceDelta(content=tail))],
                )
                yield f"data: {tail_chunk_data.model_dump_json(exclude_none=True)}\n\n"
                chunk_count += 1
            rtail = reasoning_leak_repair.flush_tail()
            if rtail:
                rtail_chunk_data = StreamingChatCompletionChunk(
                    id=completion_id,
                    created=created_time,
                    model=model_name,
                    choices=[
                        StreamingChoice(delta=ChoiceDelta(reasoning_content=rtail))
                    ],
                )
                yield f"data: {rtail_chunk_data.model_dump_json(exclude_none=True)}\n\n"
                chunk_count += 1
            finish_reason = (
                "tool_calls" if (seen_tool_calls or leak_repair.has_leak()) else "stop"
            )
            final_chunk_data = StreamingChatCompletionChunk(
                id=completion_id,
                created=created_time,
                model=model_name,
                choices=[
                    StreamingChoice(delta=ChoiceDelta(), finish_reason=finish_reason)
                ],
            )
            yield f"data: {final_chunk_data.model_dump_json(exclude_none=True)}\n\n"

            # OpenAI streaming contract: when the client requested
            # stream_options.include_usage, emit a terminal chunk with empty
            # choices and the accumulated usage, immediately before [DONE].
            # Without this, streaming consumers (pi, etc.) never see token counts
            # and cannot track context-window fill / cost.
            if _stats_usage:
                _stream_opts = (extra or {}).get("stream_options") or {}
                if _stream_opts.get("include_usage"):
                    usage_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model_name,
                        "choices": [],
                        "usage": _stats_usage,
                    }
                    yield f"data: {json.dumps(usage_chunk)}\n\n"

    except (UniInferError, ValueError) as e:
        _stats_status = int(getattr(e, "status_code", 500) or 500)
        message = str(e)
        if isinstance(e, ProviderError) and e.response_body:
            message = f"{message} | Provider Response: {e.response_body}"
        error_chunk = {
            "error": {
                "message": message,
                "type": type(e).__name__,
                "code": getattr(e, "status_code", None),
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
    except Exception as e:
        _stats_status = 500
        error_chunk = {
            "error": {
                "message": f"Unexpected server error: {type(e).__name__}",
                "type": "internal_server_error",
                "code": None,
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
    finally:
        # Always record — runs on success, error, and early client disconnect.
        try:
            from uniinfer.proxy_services.stats import get_stats

            get_stats().record(
                model_name,
                status=_stats_status,
                latency_ms=(time.monotonic() - _stats_t0) * 1000,
                usage=_stats_usage or None,
            )
        except Exception:
            pass

    logger.info(
        "%sAsync stream end for %s (chunks=%s, heartbeats=%s)",
        stream_label,
        model_name,
        chunk_count,
        heartbeat_count,
    )
    yield "data: [DONE]\n\n"
