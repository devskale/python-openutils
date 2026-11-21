"""Shared completion dispatch — the seam every surface builds on.

resolve (auto-routing) → auth (own-key) → Target construction, guarded
complete (empty-stream guard), guarded stream (priming + 429 fallback without
HTTP re-entry), error translation, metering. chat.py and responses.py both
cross this seam; OpenAI-dialect shaping (chat chunks vs ``response.*`` events)
stays per-surface by design — the dialects are different state machines, not
two formats of one stream.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Optional

from fastapi import HTTPException

from uniinfer.auth import verify_provider_access
from uniinfer.completion import Target
from uniinfer.errors import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
    UniInferError,
)
from uniinfer.proxy_services.key_usage import get_key_usage
from uniinfer.proxy_services.stats import get_stats
from uniinfer.proxy_services.stream_guard import (
    astream_guarded,
    empty_guard_enabled,
    empty_guard_retry_budget,
    emit_guard_alarm,
    nonstream_masked_empty,
)

logger = logging.getLogger("uniioai_proxy")

_AUTO_MAX_FALLBACKS = 2

Meter = Callable[[str, Optional[int], Optional[dict]], None]


@dataclass
class DispatchPlan:
    """Resolved dispatch for one completion request.

    ``auto_chain`` is None for non-routed requests ([] = routed, exhausted);
    ``routed_model`` echoes the original ``auto@selector`` when re-routing
    happened (for x-routed-model headers and logs).
    """

    provider_name: str
    provider_model: str
    target: Target
    provider_api_key: Optional[str]
    auto_chain: Optional[list[str]] = None
    routed_model: Optional[str] = None
    # carried for service-level fallback re-auth (never logged, never serialized)
    api_bearer_token: Optional[str] = field(default=None, repr=False)
    own_key: Optional[str] = field(default=None, repr=False)
    base_url: Optional[str] = field(default=None, repr=False)


def token_hash(token: Optional[str]) -> Optional[str]:
    """Caller identity for personal routing preferences (sha256 of the bearer)."""
    return hashlib.sha256(token.encode()).hexdigest() if token else None


def normalize_roles(messages: list[dict], provider_name: str) -> list[dict]:
    """Collapse OpenAI's ``developer`` role to ``system`` for providers that
    reject it (most backends 422 on developer). Providers natively accepting
    it keep it. Idempotent."""
    from uniinfer.proxy_routers.chat import _DEVELOPER_ROLE_PROVIDERS

    if provider_name in _DEVELOPER_ROLE_PROVIDERS:
        return messages
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "developer":
            m["role"] = "system"
    return out


def resolve_dispatch(
    provider_model: str,
    api_bearer_token: Optional[str],
    own_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> DispatchPlan:
    """Parse + auth + Target for a ``provider@model`` request.

    Auto-routing (``auto@selector``) resolves to a concrete provider@model and
    primes the fallback chain. Raises HTTPException on auth failure; ValueError
    surfaces from parse for the caller's 400 mapping.
    """
    from uniinfer.completion import parse_provider_model
    from uniinfer.proxy_services.router import resolve_auto, resolve_auto_candidates

    provider_name, _ = parse_provider_model(provider_model)
    provider_api_key = verify_provider_access(
        api_bearer_token, provider_name, own_key=own_key
    )
    target = Target(provider_model, provider_api_key, base_url)
    auto_chain: Optional[list[str]] = None
    routed_model: Optional[str] = None

    if provider_name == "auto":
        _resolved = resolve_auto(provider_model, token_hash=token_hash(api_bearer_token))
        if _resolved is None:
            raise HTTPException(
                status_code=503,
                detail="auto-routing found no available free model "
                "(all excluded, degraded, parked, or over quota)",
            )
        provider_model, _route_meta = _resolved
        provider_name, _ = parse_provider_model(provider_model)
        routed_model = target.provider_model
        provider_api_key = verify_provider_access(
            api_bearer_token, provider_name, own_key=own_key
        )
        target = Target(provider_model, provider_api_key, base_url)
        chain = resolve_auto_candidates(
            provider_model if routed_model is None else routed_model,
            token_hash=token_hash(api_bearer_token),
        )
        tried = {provider_model}
        auto_chain = [c for c in chain if c not in tried][:_AUTO_MAX_FALLBACKS]

    return DispatchPlan(
        provider_name=provider_name,
        provider_model=provider_model,
        target=target,
        provider_api_key=provider_api_key,
        auto_chain=auto_chain,
        routed_model=routed_model,
        api_bearer_token=api_bearer_token,
        own_key=own_key,
        base_url=base_url,
    )


def error_to_http_exception(
    e: BaseException,
    provider_name: str,
    *,
    meter: Optional[Meter] = None,
) -> HTTPException:
    """Translate a dispatch error to the canonical HTTPException.

    ``meter(status, latency_ms, usage)`` is called for provider/unknown errors
    so surfaces record stats exactly like the chat path always did.
    """
    if isinstance(e, ValueError):
        return HTTPException(status_code=400, detail=str(e))
    if isinstance(e, AuthenticationError):
        detail = f"{e} | Response: {e.response_body}" if e.response_body else str(e)
        return HTTPException(status_code=getattr(e, "status_code", 401) or 401, detail=detail)
    if isinstance(e, RateLimitError):
        detail = f"{e} | Response: {e.response_body}" if e.response_body else str(e)
        _ra = getattr(e, "retry_after", None)
        return HTTPException(
            status_code=getattr(e, "status_code", 429) or 429,
            detail=detail,
            headers={"Retry-After": str(max(1, int(_ra)))} if _ra else None,
        )
    if isinstance(e, HTTPException):
        return e
    if isinstance(e, ProviderError):
        detail = f"Provider Error ({provider_name}): {e}"
        if e.response_body:
            detail = f"{detail} | Response: {e.response_body}"
        if meter:
            meter(getattr(e, "status_code", 500) or 500, None, None)
        return HTTPException(status_code=getattr(e, "status_code", 500) or 500, detail=detail)
    if isinstance(e, UniInferError):
        return HTTPException(status_code=500, detail=f"UniInfer Error: {e}")
    if meter:
        meter(500, None, None)
    return HTTPException(
        status_code=500, detail=f"Internal Server Error: {type(e).__name__}"
    )


def make_meter(
    api_bearer_token: Optional[str],
    provider_model: str,
    provider_api_key: Optional[str],
    t0: float,
) -> Meter:
    """Stats + key-usage recording with the chat path's exact semantics."""

    def meter(status: int, latency_ms: Optional[float], usage: Optional[dict]) -> None:
        get_key_usage().note(api_bearer_token, provider_model, provider_api_key)
        get_stats().record(
            provider_model,
            status=status,
            latency_ms=(latency_ms if latency_ms is not None
                        else (time.monotonic() - t0) * 1000),
            usage=usage,
        )

    return meter


async def guarded_complete(
    plan: DispatchPlan,
    *,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    tools: Optional[list[dict]] = None,
    tool_choice: Any = None,
    reasoning_effort: Optional[str] = None,
    chat_template_kwargs: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> "object":
    """Non-stream complete with the empty-stream guard (masked-failure retry).

    Same semantics as the chat non-stream loop: a 200 completion with no
    content and no tool_calls/thinking is a masked provider failure — retry
    once (budget via env), then raise ProviderError so callers route around.
    """
    _guard_left = empty_guard_retry_budget() if empty_guard_enabled() else 0
    while True:
        full_content = await plan.target.acomplete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            chat_template_kwargs=chat_template_kwargs,
            extra=extra,
        )
        if plan.auto_chain is not None:
            from uniinfer.proxy_services.router import clear_dead_parks

            clear_dead_parks(plan.provider_model)
        if nonstream_masked_empty(full_content):
            _attempts = empty_guard_retry_budget() + 1
            if _guard_left > 0:
                _guard_left -= 1
                emit_guard_alarm(
                    "empty_stream",
                    plan.provider_model,
                    f"EMPTY-STREAM: non-stream completion empty "
                    f"(retrying, attempt {empty_guard_retry_budget() - _guard_left}/{_attempts})",
                )
                continue
            emit_guard_alarm(
                "empty_stream",
                plan.provider_model,
                f"EMPTY-STREAM: non-stream completion empty after {_attempts} attempt(s)",
            )
            from uniinfer.errors import ProviderError as _PE

            raise _PE("empty completion (masked provider failure)")
        return full_content


async def guarded_stream_events(
    plan: DispatchPlan,
    *,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    tools: Optional[list[dict]] = None,
    tool_choice: Any = None,
    reasoning_effort: Optional[str] = None,
    chat_template_kwargs: Optional[dict] = None,
    extra: Optional[dict] = None,
    request_id: Optional[str] = None,
    usage_note: Optional[Callable[[], None]] = None,
) -> AsyncGenerator[dict, None]:
    """Guarded stream yielding NEUTRAL chat-delta dicts (decoded from the
    battle-tested chat SSE relay — encode+decode is exact and keeps streaming.py
    untouched). Error payloads raise ProviderError; keep-alives/[DONE] vanish.

    On an open-time 429 the failed model is parked, the next chain candidate
    is re-resolved here (new Target, fresh guard), and the stream restarts —
    surfaces never re-enter their HTTP handler for fallback.
    """
    import json as _json

    from uniinfer.errors import ProviderError as _PE
    from uniinfer.sse import DONE, sse_events

    def _decode(item: str) -> Optional[dict]:
        for ev in sse_events(item.splitlines()):
            payload = ev.data
            if not payload or payload == DONE:
                continue
            try:
                d = _json.loads(payload)
            except ValueError:
                continue
            return d if isinstance(d, dict) else None
        return None

    _chain = list(plan.auto_chain) if plan.auto_chain is not None else None
    while True:
        gen = astream_guarded(
            target=plan.target,
            messages=messages,
            temp=temperature,
            max_tok=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            request_id=request_id,
            reasoning_effort=reasoning_effort,
            chat_template_kwargs=chat_template_kwargs,
            extra=extra,
            usage_note=usage_note,
        )
        try:
            _first = await gen.__anext__()
        except RateLimitError as e:
            if _chain:
                from uniinfer.proxy_services.router import note_quota_hit

                note_quota_hit(plan.provider_model, getattr(e, "retry_after", None))
                _next = _chain.pop(0)
                logger.warning(
                    "[auto-route] stream 429 on %s, falling through to %s",
                    plan.provider_model,
                    _next,
                )
                from uniinfer.completion import parse_provider_model as _ppm

                _pn, _ = _ppm(_next)
                plan.provider_name = _pn
                plan.provider_model = _next
                plan.provider_api_key = verify_provider_access(
                    plan.api_bearer_token, _pn, own_key=plan.own_key
                )
                plan.target = Target(_next, plan.provider_api_key, plan.base_url)
                continue
            raise
        for item in (_first,):
            d = _decode(item)
            if d is None:
                continue
            if "error" in d:
                raise _PE(str(d["error"]))
            yield d
        async for item in gen:
            d = _decode(item)
            if d is None:
                continue
            if "error" in d:
                raise _PE(str(d["error"]))
            yield d
        return
