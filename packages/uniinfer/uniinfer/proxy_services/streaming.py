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

logger = logging.getLogger("uniioai_proxy")


def is_openai_strict_mode() -> bool:
    return False


def normalize_nonstream_content(content: Any, tool_calls: Any) -> str | None:
    if tool_calls:
        return content
    if content is None and is_openai_strict_mode():
        return ""
    return content


async def astream_response_generator(
    *,
    astream_completion,
    messages: list[dict],
    provider_model: str,
    temp: float,
    max_tok: int,
    provider_api_key: str | None,
    base_url: str | None,
    tools: list[dict] | None = None,
    tool_choice: Any | None = None,
    request_id: str | None = None,
    reasoning_effort: str | None = None,
    think: bool | str | None = None,
) -> AsyncGenerator[str, None]:
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    model_name = provider_model

    stream_label = f"[{request_id}] " if request_id else ""
    logger.info("%sAsync stream start for %s", stream_label, model_name)
    chunk_count = 0
    heartbeat_count = 0
    last_yield_time = time.monotonic()
    idle_warning_threshold = 10.0
    heartbeat_interval = float(os.getenv("UNIINFER_STREAM_HEARTBEAT", "120"))  # Increased for reasoning models
    idle_timeout = float(os.getenv("UNIINFER_STREAM_IDLE_TIMEOUT", "300"))  # Increased for reasoning models

    first_chunk_data = StreamingChatCompletionChunk(
        id=completion_id,
        created=created_time,
        model=model_name,
        choices=[StreamingChoice(delta=ChoiceDelta(role="assistant"))],
    )
    yield f"data: {first_chunk_data.model_dump_json(exclude_none=True)}\n\n"

    seen_tool_calls = False
    sent_finish_reason = False

    try:
        async_iter = astream_completion(
            messages,
            provider_model,
            temp,
            max_tok,
            provider_api_key=provider_api_key,
            base_url=base_url,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
        ).__aiter__()
        while True:
            try:
                chunk = await asyncio.wait_for(async_iter.__anext__(), timeout=heartbeat_interval)
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
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {}) or {}
                    if not isinstance(delta, dict):
                        continue

                    if is_openai_strict_mode():
                        delta.pop("reasoning_content", None)
                        delta.pop("thinking", None)
                    else:
                        if delta.get("thinking") and not delta.get("reasoning_content"):
                            delta["reasoning_content"] = delta["thinking"]
                        delta.pop("thinking", None)

                yield f"data: {json.dumps(chunk)}\n\n"
                chunk_count += 1
                now = time.monotonic()
                if now - last_yield_time > idle_warning_threshold:
                    logger.warning("%sAsync stream idle gap %.2fs for %s", stream_label, now - last_yield_time, model_name)
                last_yield_time = now

                choice = chunk.get("choices", [{}])[0]
                if choice.get("finish_reason"):
                    sent_finish_reason = True
                delta = choice.get("delta", {})
                if delta.get("tool_calls"):
                    seen_tool_calls = True
            else:
                chunk_finish_reason = getattr(chunk, "finish_reason", None)
                
                # Handle error finish_reason from provider (e.g., preemption detection)
                if chunk_finish_reason == "error":
                    import sys
                    print(f"[DEBUG] STREAMING: Got error finish_reason, raw_response={chunk.raw_response}", file=sys.stderr, flush=True)
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
                    if chunk.message.content:
                        choice_kwargs["content"] = chunk.message.content
                    if chunk.thinking and not is_openai_strict_mode():
                        choice_kwargs["reasoning_content"] = chunk.thinking
                    if chunk.message.tool_calls:
                        choice_kwargs["tool_calls"] = chunk.message.tool_calls
                        seen_tool_calls = True

                    choice_kwargs_with_delta = {
                        "delta": ChoiceDelta(**choice_kwargs) if choice_kwargs else ChoiceDelta()
                    }
                    if chunk_finish_reason:
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
            finish_reason = "tool_calls" if seen_tool_calls else "stop"
            final_chunk_data = StreamingChatCompletionChunk(
                id=completion_id,
                created=created_time,
                model=model_name,
                choices=[StreamingChoice(delta=ChoiceDelta(), finish_reason=finish_reason)],
            )
            yield f"data: {final_chunk_data.model_dump_json(exclude_none=True)}\n\n"

    except (UniInferError, ValueError) as e:
        message = str(e)
        if isinstance(e, ProviderError) and e.response_body:
            message = f"{message} | Provider Response: {e.response_body}"
        error_chunk = {"error": {"message": message, "type": type(e).__name__, "code": getattr(e, 'status_code', None)}}
        yield f"data: {json.dumps(error_chunk)}\n\n"
    except Exception as e:
        error_chunk = {
            "error": {
                "message": f"Unexpected server error: {type(e).__name__}",
                "type": "internal_server_error",
                "code": None,
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

    logger.info("%sAsync stream end for %s (chunks=%s, heartbeats=%s)", stream_label, model_name, chunk_count, heartbeat_count)
    yield "data: [DONE]\n\n"
