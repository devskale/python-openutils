"""llminvoke — shared LLM invocation layer for kontext.one.

Bridges uniinfer (the inference library) + credgoo (API key manager) behind
one small interface. Three levels:

  invoke_llm  — one-shot: credgoo → provider → request → .complete() → response
  call_llm    — full: invoke_llm + retry/backoff + extract_response_text → str
  stream_llm  — streaming: credgoo → provider → request → .stream_complete() → Iterator[str]

All three share provider+request construction internally. Consumers pass
model + provider (resolved by their own config); the module handles the rest.
"""
from __future__ import annotations

import time
from typing import Iterator

from credgoo import get_api_key
from uniinfer import (
    ChatCompletionRequest,
    ChatMessage,
    ProviderFactory,
    extract_response_text,
)

__all__ = ["invoke_llm", "call_llm", "stream_llm", "create_provider"]
__version__ = "0.1.0"


def _build_messages(
    prompt: str | None,
    messages: list[ChatMessage] | None,
    system_prompt: str | None,
) -> list[ChatMessage]:
    """Resolve messages: explicit > built from prompt/system_prompt."""
    if messages is not None:
        return messages
    msgs: list[ChatMessage] = []
    if system_prompt:
        msgs.append(ChatMessage(role="system", content=system_prompt))
    msgs.append(ChatMessage(role="user", content=prompt or ""))
    return msgs


def create_provider(provider: str):
    """credgoo key → ProviderFactory.get_provider. Public for consumers that
    need the raw provider (e.g. custom streaming with chunk inspection)."""
    api_key = get_api_key(provider)
    return ProviderFactory.get_provider(provider, api_key=api_key)


def invoke_llm(
    *,
    model: str,
    provider: str,
    messages: list[ChatMessage],
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **request_kwargs,
):
    """One-shot invocation: credgoo → provider → request → .complete().

    Returns the raw ChatCompletionResponse (for consumers that need
    usage data, error classification, or custom extraction).
    """
    prov = create_provider(provider)
    request = ChatCompletionRequest(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=False,
        **request_kwargs,
    )
    return prov.complete(request)


def call_llm(
    prompt: str | None = None,
    *,
    model: str,
    provider: str,
    messages: list[ChatMessage] | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    max_attempts: int = 1,
    **request_kwargs,
) -> str:
    """Full invocation with retry + thinking-model extraction. Returns text.

    Args:
        prompt: the user prompt (used if messages is None).
        messages: explicit message list (overrides prompt — for multipart/image).
        system_prompt: convenience — prepends a system message (ignored if messages given).
        model / provider: routed through uniinfer (credgoo key resolved internally).
        temperature / max_tokens: request sampling params.
        max_attempts: retry count (1 = no retry). Backoff is 2s, doubled.
        **request_kwargs: passthrough to ChatCompletionRequest (e.g. chat_template_kwargs).

    Returns the extracted text, or "" if all attempts yield nothing.
    Raises the last exception if all attempts fail with errors.
    """
    msgs = _build_messages(prompt, messages, system_prompt)
    last_exc: Exception | None = None

    for attempt in range(max(1, max_attempts)):
        try:
            response = invoke_llm(
                model=model,
                provider=provider,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_tokens,
                **request_kwargs,
            )
            text = extract_response_text(response)
            if text:
                return text
            # empty response — retry if attempts remain
        except Exception as exc:
            last_exc = exc
        # backoff before retry (skip on last attempt)
        if attempt < max_attempts - 1:
            time.sleep(2 ** (attempt + 1))

    if last_exc is not None:
        raise last_exc
    return ""


def stream_llm(
    prompt: str | None = None,
    *,
    model: str,
    provider: str,
    messages: list[ChatMessage] | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **request_kwargs,
) -> Iterator[str]:
    """Streaming invocation. Yields chunk texts as they arrive.

    One-shot (no retry — streaming retry is caller's concern). Consumers
    accumulate or display chunks as needed.

    Args: same as call_llm (minus max_attempts).
    Yields: str chunk texts (may be empty strings for keep-alive chunks).
    """
    msgs = _build_messages(prompt, messages, system_prompt)
    prov = create_provider(provider)
    request = ChatCompletionRequest(
        messages=msgs,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True,
        **request_kwargs,
    )
    for chunk in prov.stream_complete(request):
        content = getattr(getattr(chunk, "message", None), "content", None)
        if isinstance(content, str) and content:
            yield content
