"""llminvoke — shared LLM invocation layer for kontext.one.

Bridges uniinfer (the inference library) + credgoo (API key manager) behind
one small interface. Three invocation levels, all config-aware::

    call_llm   — retry + backoff + backup chain + extract → str   (recommended)
    invoke_llm — one-shot → raw response                        (escape hatch)
    stream_llm — streaming → Iterator[str]                      (before-first-token retry;
                  include_reasoning=True → StreamPiece-Stücke mit .kind, Denkphase live)

Model config (which model, params, backups, retry, DSGVO) resolves through
``resolve_model`` per the ADR 0004 precedence chain::

    env var  >  team-settings (DB)  >  clients.yml (runtime)  >  catalog default

See ``config.py`` for the resolver + ``ResolvedConfig`` shape.
"""
from __future__ import annotations

import os
import time
from dataclasses import replace
from typing import Iterator

from credgoo import get_api_key
from uniinfer import (
    ChatCompletionRequest,
    ChatMessage,
    ProviderFactory,
    extract_response_text,
)

from llminvoke.config import (
    PERMANENT_ERRORS,
    ModelRef,
    ResolvedConfig,
    RetryPolicy,
    _extract_retry_after,
    classify_error,
    emit_alarm,
    get_model_info,
    is_dsgvo_provider,
    resolve_model,
)

__version__ = "0.3.0"

__all__ = [
    # invocation
    "invoke_llm",
    "call_llm",
    "stream_llm",
    "StreamPiece",
    "iter_llm_events",
    "create_provider",
    # fo-t7: empty-chain failure signal
    "LLMChainExhausted",
    # config (ADR 0004)
    "resolve_model",
    "ResolvedConfig",
    "RetryPolicy",
    "ModelRef",
    "get_model_info",
    "is_dsgvo_provider",
    "classify_error",
    "emit_alarm",
    "call_llm_with_usage",
    "call_llm_with_meta",
]


# ════════════════════════════════════════════════════════════════════════
# Provider + message helpers
# ════════════════════════════════════════════════════════════════════════

def create_provider(provider: str, *, base_url: str | None = None, api_key: str | None = None):
    """credgoo key → ProviderFactory.get_provider. Public for consumers that
    need the raw provider (e.g. custom streaming with chunk inspection).

    ``base_url`` (the llm-gateway endpoint) + ``api_key`` override the provider's
    own — used by the gateway path (invoke_llm with a resolved base_url+bearer).
    """
    key = api_key if api_key is not None else get_api_key(provider)
    kwargs: dict = {"api_key": key}
    if base_url:
        kwargs["base_url"] = base_url
    return ProviderFactory.get_provider(provider, **kwargs)


def _build_messages(
    prompt: str | None,
    messages: list[ChatMessage] | None,
    system_prompt: str | None,
) -> list[ChatMessage]:
    """Resolve messages: explicit > built from prompt/system_prompt.

    Explicit dict messages are coerced to ChatMessage: callers passing raw
    ``{'role': ..., 'content': ...}`` dicts otherwise crash the GATEWAY path
    with a cryptic upstream 400 ("messages[0].user.content: Field required") —
    _invoke_via_openai_sdk serializes via getattr(), and getattr on a dict
    returns None (Vorfall: TU-smoke 2026-09-05 — alle Nicht-vllm-Routen 400ten
    auf content=null; die vllm-Route tolerierte es und maskierte den Bug).
    """
    if messages is not None:
        return [
            m if isinstance(m, ChatMessage) else ChatMessage(**m)
            for m in messages
        ]
    msgs: list[ChatMessage] = []
    if system_prompt:
        msgs.append(ChatMessage(role="system", content=system_prompt))
    msgs.append(ChatMessage(role="user", content=prompt or ""))
    return msgs


def _resolve_call_config(
    *,
    config: ResolvedConfig | None,
    package: str | None,
    client: str | None,
    task: str | None,
    model: str | None,
    provider: str | None,
    temperature: float | None,
    max_tokens: int | None,
    max_attempts: int | None,
    env_prefix: str | None,
) -> ResolvedConfig:
    """Build a ResolvedConfig from call_llm/stream_llm args + apply overrides.

    Precedence within a single call:
      1. explicit ``config=`` (caller resolved already)
      2. ``package=``/``task=`` → resolve_model (catalog + box overlay)
      3. explicit ``model=``+``provider=`` → single-model config (backward compat)

    Per-call overrides (temperature, max_tokens, max_attempts) layer on top.
    """
    if config is not None:
        cfg = config
    elif package or client or task:
        cfg = resolve_model(package, client, task, env_prefix=env_prefix)
    elif model and provider:
        # Resolve base_url/bearer from env + clients.yml default so explicit-model
        # calls still route via the gateway when configured. Without this, callers
        # passing model+provider (e.g. FAP audit steps) bypassed the proxy → direct
        # uniinfer → "Event loop is closed". resolve_model() reads env OPENAI_BASE_URL
        # + clients.yml default; we keep its endpoint and override only the primary.
        cfg = resolve_model(env_prefix=env_prefix)
        cfg = replace(cfg, primary=ModelRef(provider=provider, model=model))
    else:
        raise ValueError(
            "call_llm needs one of: config=, package=/client=/task=, or model=+provider="
        )

    # explicit model/provider override the resolved primary (but keep backups/retry)
    if model and provider and cfg.primary.model != model:
        cfg = replace(cfg, primary=ModelRef(provider=provider, model=model))

    overrides: dict = {}
    if temperature is not None:
        overrides["temperature"] = float(temperature)
    if max_tokens is not None:
        overrides["max_tokens"] = int(max_tokens)
    if max_attempts is not None:
        # backward compat: legacy per-model retry count
        overrides["retry"] = replace(cfg.retry, attempts=max(1, int(max_attempts)))
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg


# ════════════════════════════════════════════════════════════════════════
# invoke_llm — one-shot (the escape hatch; no retry, no backup)
# ════════════════════════════════════════════════════════════════════════

def _invoke_via_openai_sdk(
    *,
    base_url: str,
    bearer: str | None,
    provider: str,
    model: str,
    bare_model: bool = False,
    messages: list[ChatMessage],
    temperature: float,
    max_tokens: int,
    request_kwargs: dict,
):
    """Gateway call via the STANDARD SYNC openai SDK (no asyncio).

    Why: uniinfer's provider.complete() bridges to its async httpx client via
    asyncio.run(), which closes the event loop after every call; the process-
    global client's connection pool then fills with zombie connections bound to
    dead loops → "Event loop is closed" under high call counts (the PA
    stepAssess storm). The openai SDK's own client manages its lifecycle sanely
    and never crosses an event loop, so this path is immune. Returns a uniinfer
    ChatCompletionResponse so the rest of llminvoke (extract/usage/alarm) is
    unchanged. Model id is ``provider@model`` (the proxy routes by prefix).
    """
    from openai import OpenAI
    from uniinfer import ChatCompletionResponse

    import httpx
    # Fail-fast on stalls: a read timeout (no bytes for 90s) beats a 300s total
    # hang when the upstream stalls (HTTP 000 — connection up, zero bytes).
    # llminvoke then retries (timeout is transient → _try_model backoff), and a
    # 429 raises openai.RateLimitError → classify_error → rate_limited →
    # retry-after-backoff (already wired). connect/pool stay tight so a dead
    # peer fails in seconds, not minutes.
    client = OpenAI(
        base_url=base_url, api_key=bearer or "missing",
        timeout=httpx.Timeout(connect=10.0, read=90.0, write=30.0, pool=10.0),
    )
    model_id = model if bare_model else f"{provider}@{model}"
    def _to_openai_msg(m):
        # Tool-fähige Konversation (2026-11-10, User: „immer über uniinfer"):
        # rohe OpenAI-Shape-Dicts (assistant.tool_calls, tool-Rolle mit
        # tool_call_id) VERBATIM durchreichen; ChatMessage-Objekte inklusive
        # ihrer Tool-Felder konvertieren — vorher fielen tool_calls sowohl in
        # der Anfrage als auch in der Antwort weg.
        if isinstance(m, dict):
            return m
        d = {"role": getattr(m, "role", None) or "user", "content": getattr(m, "content", None)}
        if getattr(m, "tool_calls", None):
            d["tool_calls"] = [
                tc if isinstance(tc, dict) else (tc.model_dump() if hasattr(tc, "model_dump") else dict(tc))
                for tc in m.tool_calls
            ]
        if getattr(m, "tool_call_id", None):
            d["tool_call_id"] = m.tool_call_id
        rd = getattr(m, "reasoning_details", None)
        if rd and isinstance(d, dict):
            d.setdefault("extra_fields", {})["reasoning_details"] = rd
        return d

    openai_messages = [_to_openai_msg(m) for m in messages]
    # vLLM extras (chat_template_kwargs.enable_thinking, …) ride in extra_body →
    # merged into the top-level request JSON → forwarded by the proxy to vLLM.
    # OpenAI-Request-Parameter, die TOP-LEVEL gehören (Tool-Calls!), von
    # vLLM-Extras (extra_body) trennen — tools in extra_body ignoriert der
    # Gateway still (Vorfall: ReactAgent-Rewiring).
    _TOP_LEVEL = ("tools", "tool_choice", "parallel_tool_calls", "response_format", "stream")
    top_kwargs = {k: request_kwargs[k] for k in _TOP_LEVEL if k in request_kwargs}
    extra = {k: v for k, v in request_kwargs.items() if k not in _TOP_LEVEL and k != "extra_body"}
    if isinstance(request_kwargs.get("extra_body"), dict):
        extra = {**request_kwargs["extra_body"], **extra}
    resp = client.chat.completions.create(
        model=model_id,
        messages=openai_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra or None,
        **top_kwargs,
    )
    # Crash loud (golden rule): no choices = gateway/upstream malfunction, not a
    # legitimate empty. Surface it as an explicit error so it alarms + retries
    # visibly — never degrade into a silent empty-content verdict.
    if not resp.choices:
        raise RuntimeError(
            f"gateway {base_url} returned no choices for {model_id}: "
            f"{resp.model_dump() if hasattr(resp, 'model_dump') else resp!r}"
        )
    choice = resp.choices[0]
    msg = choice.message
    u = getattr(resp, "usage", None)
    usage = (
        {
            "prompt_tokens": getattr(u, "prompt_tokens", 0),
            "completion_tokens": getattr(u, "completion_tokens", 0),
            "total_tokens": getattr(u, "total_tokens", 0),
        }
        if u
        else {}
    )
    _tcs = getattr(msg, "tool_calls", None) or None
    if _tcs:
        _tcs = [
            tc if isinstance(tc, dict) else (tc.model_dump() if hasattr(tc, "model_dump") else dict(tc))
            for tc in _tcs
        ]
    return ChatCompletionResponse(
        message=ChatMessage(
            role=(getattr(msg, "role", None) or "assistant"),
            content=getattr(msg, "content", None),
            tool_calls=_tcs,
            tool_call_id=getattr(msg, "tool_call_id", None),
        ),
        provider=provider,
        model=getattr(resp, "model", model_id),
        usage=usage,
        raw_response=resp.model_dump() if hasattr(resp, "model_dump") else {},
        finish_reason=getattr(choice, "finish_reason", None),
        thinking=getattr(msg, "reasoning_content", None),
    )


def invoke_llm(
    *,
    model: str,
    provider: str,
    messages: list[ChatMessage],
    temperature: float = 0.7,
    max_tokens: int = 4096,
    base_url: str | None = None,
    bearer: str | None = None,
    bare_model: bool = False,
    **request_kwargs,
):
    """One-shot invocation: credgoo → provider → request → .complete().

    Returns the raw ChatCompletionResponse (for consumers that need
    usage data, error classification, or custom extraction). No retry, no
    backup — agentos uses this inside its own chain/breaker loop.

    ``base_url`` (the llm-gateway endpoint): when set, the call goes through the
    STANDARD openai SDK (sync) against that gateway with ``bearer``; the model
    id sent is the FULL ``provider@model`` (the proxy routes by the prefix). This
    bypasses uniinfer's async client entirely (no asyncio → no event-loop bug).
    Without ``base_url``, the named provider + bare model use the legacy direct
    path (uniinfer provider.complete()).
    """
    if base_url:
        return _invoke_via_openai_sdk(
            base_url=base_url, bearer=bearer, provider=provider, model=model,
            messages=messages, temperature=temperature, max_tokens=max_tokens,
            request_kwargs=request_kwargs, bare_model=bare_model,
        )
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


# ════════════════════════════════════════════════════════════════════════
# call_llm — retry + backoff + backup chain (the recommended entry point)
# ════════════════════════════════════════════════════════════════════════

def with_retry(
    fn,
    *,
    retry=None,
    fail_fast: bool = False,
    alarm_provider: str | None = None,
    alarm_model: str | None = None,
    package: str | None = None,
    client: str | None = None,
):
    """Retry ANY LLM-shaped callable with the canonical retry semantics.

    One seam for robustness (User 2026-11-10: model errors 429/500 can
    always happen — the whole workflow must be robust): consumers that need
    the RAW response (tool-calling agents) get the same retry/backoff/
    Retry-After/alarm treatment that ``call_llm`` gives text callers.

    ``fn() -> response`` — raise on failure; transient errors (rate_limited/
    timeout/network/server_error/…) retry with backoff, permanent errors
    (auth/context/not_found/bad_request) stop immediately. Returns the
    response, or ``None`` if exhausted/permanent (caller decides: backup
    chain, raise, or degrade loudly).
    """
    from .config import RetryPolicy, classify_error, _extract_retry_after

    r = retry if isinstance(retry, RetryPolicy) else RetryPolicy()
    attempts = 1 if fail_fast else r.attempts
    err: BaseException = RuntimeError("not attempted")
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            err = exc
        kind = classify_error(err)
        if kind in PERMANENT_ERRORS:
            break
        if attempt < attempts - 1:
            ra = _extract_retry_after(err) if kind == "rate_limited" else None
            time.sleep(r.delay_for(attempt + 1, ra))
    if alarm_provider or alarm_model:
        emit_alarm(
            "alarm", alarm_provider or "?", alarm_model or "?",
            classify_error(err), str(err), package=package, client=client,
        )
    return None


def _try_model(
    ref: ModelRef,
    cfg: ResolvedConfig,
    msgs: list[ChatMessage],
    fail_fast: bool,
    request_kwargs: dict,
    *,
    package: str | None,
    client: str | None,
) -> tuple[str | None, object | None]:
    """Try one model with retry/backoff. Returns (text, response) on success,
    (None, None) on failure.

    - fail_fast: one shot, no retry.
    - transient errors (429/timeout/network): retry with backoff (honors Retry-After).
    - permanent errors (auth/context-exceeded): stop retrying this model immediately.
    - empty response: treated as a failure (Q8) — retried, then walks to backup.
    """
    attempts = 1 if fail_fast else cfg.retry.attempts
    merged_kwargs = {**cfg.request_kwargs, **request_kwargs}
    err: BaseException = RuntimeError("not attempted")
    for attempt in range(attempts):
        try:
            response = invoke_llm(
                model=ref.model,
                provider=ref.provider,
                messages=msgs,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                base_url=cfg.base_url,
                bearer=cfg.bearer,
                bare_model=cfg.bare_model,
                **merged_kwargs,
            )
            text = extract_response_text(response)
            if text:
                return text, response
            err = RuntimeError("empty_response")  # empty = failure (Q8)
        except Exception as exc:
            err = exc

        kind = classify_error(err)
        if kind in PERMANENT_ERRORS:
            break  # don't retry permanent errors on this model
        # transient → backoff before next attempt (skip on last)
        if attempt < attempts - 1:
            ra = _extract_retry_after(err) if kind == "rate_limited" else None
            time.sleep(cfg.retry.delay_for(attempt + 1, ra))

    emit_alarm(
        "alarm", ref.provider, ref.model,
        classify_error(err), str(err),
        package=package, client=client,
    )
    return None, None


class LLMChainExhausted(RuntimeError):
    """The entire model chain failed (all models returned empty/errored).

    Raised by ``call_llm(raise_on_empty=True)`` so a CLI can turn a silent ``""``
    into a loud Failure (fo-t7, ADR 0007 Failure plane) — distinct from the
    Defect plane: this lowers ``job.success`` + emits a ROBOTNI error.
    """


def call_llm(
    prompt: str | None = None,
    *,
    config: ResolvedConfig | None = None,
    package: str | None = None,
    client: str | None = None,
    task: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    messages: list[ChatMessage] | None = None,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_attempts: int | None = None,
    fail_fast: bool = False,
    raise_on_empty: bool = False,
    env_prefix: str | None = None,
    **request_kwargs,
) -> str:
    """Full invocation: retry + backoff + backup chain + extract → text.

    Config resolution (ADR 0004): pass ``config=`` (pre-resolved),
    ``package=``/``client=``/``task=`` (resolve from catalog + clients.yml),
    or ``model=``+``provider=`` (explicit single-model, backward compat).

    Args:
        prompt: user prompt (used if messages is None).
        messages: explicit message list (overrides prompt — for multipart/image).
        system_prompt: prepends a system message (ignored if messages given).
        temperature / max_tokens: per-call overrides on the resolved config.
        max_attempts: backward-compat override for retry attempts.
        fail_fast: opt-in — skip retry, escalate to backup on any error.
        env_prefix: e.g. ``"PDF2MD_VLM"`` — ``<PREFIX>_MODEL``/``_PROVIDER``
            override the primary (Q12); backups still flow from config.
        **request_kwargs: passthrough to ChatCompletionRequest.

    Returns the extracted text, or ``""`` if the entire chain fails.
    """
    cfg = _resolve_call_config(
        config=config, package=package, client=client, task=task,
        model=model, provider=provider,
        temperature=temperature, max_tokens=max_tokens,
        max_attempts=max_attempts, env_prefix=env_prefix,
    )
    msgs = _build_messages(prompt, messages, system_prompt)
    pkg = package or _package_from_env()
    cli = client or os.environ.get("KONTEXT_CLIENT", "").strip() or None

    for ref in cfg.chain:
        text, _response = _try_model(ref, cfg, msgs, fail_fast, request_kwargs, package=pkg, client=cli)
        if text is not None:
            return text
    if raise_on_empty:
        raise LLMChainExhausted(
            f"LLM chain exhausted ({len(cfg.chain)} model(s)) — all returned empty/failed"
        )
    return ""  # entire chain exhausted


def _stamp_usage_details_flat(usage: dict) -> dict:
    """Provider-Usage-Details FLACH aufstempeln: ``cached_tokens`` (in) +
    ``reasoning_tokens`` — DER eine Normalisierer an der Invocation-Seam.

    OpenAI-/vLLM-kompatible Endpunkte (Aqueduct, DGX) melden die Details nur
    VERSCHACHTELT: ``prompt_tokens_details.cached_tokens`` (wie viel vom
    Prompt kam aus dem Prefix-Cache) und ``completion_tokens_details.
    reasoning_tokens`` (Denk-Tokens). Konsumenten wollen in/out/cached-in
    FLACH lesen (agentos TokenTracker → llm-Block in der Kriterien-History)
    — bisher warf jede Schicht sie still weg. Bereits flache Keys gewinnen,
    die rohen Felder bleiben unangetastet, das Dict wird kopiert (nie mutiert).
    """
    ptd = usage.get("prompt_tokens_details")
    ctd = usage.get("completion_tokens_details")
    flat: dict = {}
    if not usage.get("cached_tokens") and isinstance(ptd, dict) and ptd.get("cached_tokens"):
        flat["cached_tokens"] = int(ptd["cached_tokens"])
    if not usage.get("reasoning_tokens") and isinstance(ctd, dict) and ctd.get("reasoning_tokens"):
        flat["reasoning_tokens"] = int(ctd["reasoning_tokens"])
    return {**usage, **flat} if flat else usage


def _extract_usage(response: object) -> dict | None:
    """Best-effort usage extraction from a raw response."""
    if response is None:
        return None
    usage = getattr(response, "usage", None)
    if isinstance(usage, dict):
        return _stamp_usage_details_flat(usage)
    return None


def _extract_served_meta(ref: ModelRef, response: object) -> dict:
    """Which model actually answered: gateway echo > requested ref.

    The gateway echoes the concrete model in the response body (auto@free
    resolves proxy-side; fallback chains can serve a backup) — so the echo
    wins. The requested ref is the fallback for providers that don't echo.
    ``provider`` is parsed from the echoed ``provider@model`` when possible
    (an auto@free echo serves from a concrete provider, not "auto").
    """
    served = getattr(response, "model", None)
    served = served if isinstance(served, str) and served else str(ref)
    provider = served.split("@", 1)[0] if "@" in served else ref.provider
    return {
        "requested": str(ref),
        "model": served,
        "provider": provider,
    }


def call_llm_with_meta(
    prompt: str | None = None,
    *,
    config: ResolvedConfig | None = None,
    package: str | None = None,
    client: str | None = None,
    task: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    messages: list[ChatMessage] | None = None,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_attempts: int | None = None,
    fail_fast: bool = False,
    env_prefix: str | None = None,
    **request_kwargs,
) -> tuple[str, dict | None]:
    """Same as call_llm but also returns WHICH model served the call.

    Returns ``(text, meta)`` with ``meta = {"requested", "model", "provider",
    "usage"}``: ``requested`` is the chain member the call landed on (a backup
    that served shows up here, not the configured primary), ``model`` is that
    call's echoed id — the gateway echoes the concrete model in the response
    body (auto@free resolves proxy-side), falling back to ``requested`` when
    the provider doesn't echo. ``meta`` is ``None`` when the entire chain
    fails.
    """
    cfg = _resolve_call_config(
        config=config, package=package, client=client, task=task,
        model=model, provider=provider,
        temperature=temperature, max_tokens=max_tokens,
        max_attempts=max_attempts, env_prefix=env_prefix,
    )
    msgs = _build_messages(prompt, messages, system_prompt)
    pkg = package or _package_from_env()
    cli = client or os.environ.get("KONTEXT_CLIENT", "").strip() or None

    for ref in cfg.chain:
        text, response = _try_model(ref, cfg, msgs, fail_fast, request_kwargs, package=pkg, client=cli)
        if text is not None:
            meta = _extract_served_meta(ref, response)
            meta["usage"] = _extract_usage(response)
            return text, meta
    return "", None


def call_llm_with_usage(
    prompt: str | None = None,
    *,
    config: ResolvedConfig | None = None,
    package: str | None = None,
    client: str | None = None,
    task: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    messages: list[ChatMessage] | None = None,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_attempts: int | None = None,
    fail_fast: bool = False,
    env_prefix: str | None = None,
    **request_kwargs,
) -> tuple[str, dict | None]:
    """Same as call_llm but also returns a usage dict (prompt/completion/total tokens).

    Usage is extracted from the raw response on success; ``None`` if unavailable.
    Same retry/backoff/backup chain as ``call_llm``.
    """
    cfg = _resolve_call_config(
        config=config, package=package, client=client, task=task,
        model=model, provider=provider,
        temperature=temperature, max_tokens=max_tokens,
        max_attempts=max_attempts, env_prefix=env_prefix,
    )
    msgs = _build_messages(prompt, messages, system_prompt)
    pkg = package or _package_from_env()
    cli = client or os.environ.get("KONTEXT_CLIENT", "").strip() or None

    for ref in cfg.chain:
        text, response = _try_model(ref, cfg, msgs, fail_fast, request_kwargs, package=pkg, client=cli)
        if text is not None:
            return text, _extract_usage(response)
    return "", None


# ════════════════════════════════════════════════════════════════════════
# stream_llm — streaming with before-first-token retry (Q20)
# ════════════════════════════════════════════════════════════════════════

class StreamPiece(str):
    """Markiertes Stream-Stück: ``kind`` ∈ {"reasoning", "content"}.

    str-Subclass, damit der Iterator[str]-Contract von ``stream_llm`` bestehen
    bleibt — join()/isinstance(str) funktionieren unverändert. Consumer mit
    ``include_reasoning=True`` verzweigen auf ``.kind`` (Denkphase als Progress
    zeigen, Antwort separat akkumulieren), ohne die Tupel-API von
    ``iter_llm_events`` zu brauchen.
    """

    kind: str

    def __new__(cls, text: str, kind: str) -> "StreamPiece":
        obj = super().__new__(cls, text)
        obj.kind = kind
        return obj


def _chunk_has_payload(chunk) -> bool:
    """Erster substanzieller Token? — derselbe Zwei-Stellen-Vertrag wie
    :func:`_chunk_events`: reasoning kann an ``chunk.message.reasoning_content``
    (manche Provider) ODER an ``chunk.thinking`` (uniinfer-Contract: TUProvider
    legt reasoning dort ab) ankommen. Content an ``chunk.message.content``.

    Vorfall „Hilti-Reeval-Synthese 5/5 first-token" (2026-10-06): _start_stream
    prüfte nur message.content/message.reasoning_content — während der gesamten
    Denkphase flossen reasoning-Chunks über ``chunk.thinking`` und wurden STILL
    KONSUMIERT, ohne zurückzukehren. iter_llm_events lieferte dadurch kein
    Ereignis, bis das Denken endete (gemessen am exakten Synthese-Prompt: erstes
    reasoning am TU-Gateway nach 0.9s, erster content nach 46s) — jede
    Erst-Event-Frist des Aufrufers (agentos: 20s) feuerte, obwohl Bytes
    flossen, und die Denkphase ging als Ereignisse komplett verloren.
    """
    msg = getattr(chunk, "message", None)
    if isinstance(getattr(msg, "content", None), str) and msg.content:
        return True
    r = getattr(msg, "reasoning_content", None)
    if not (isinstance(r, str) and r):
        r = getattr(chunk, "thinking", None)
    return isinstance(r, str) and bool(r)


def _stream_via_openai_sdk(
    *,
    base_url: str,
    bearer: str | None,
    provider: str,
    model: str,
    bare_model: bool = False,
    messages: list[ChatMessage],
    temperature: float,
    max_tokens: int,
    request_kwargs: dict,
):
    """Gateway STREAM via the sync openai SDK — yields duck-typed chunks.

    Streaming twin of :func:`_invoke_via_openai_sdk` (same event-loop
    immunity, same timeouts). The gateway routes by the ``provider@model``
    prefix, so instance aliases (hab, …) that uniinfer's ProviderFactory
    doesn't know become streamable: ``create_provider('hab')`` raises
    "not registered" even WITH a base_url override (the factory resolves
    by provider NAME before any kwarg reaches a constructor) — Vorfall
    2026-11-04 (dgx-hab-episoden): der Primary-Kettenplatz verbrannte pro
    Step an diesem ValueError plus 20–30 s an toten Backups (ttfb 32–44 s).
    """
    from types import SimpleNamespace

    from openai import OpenAI

    import httpx
    client = OpenAI(
        base_url=base_url, api_key=bearer or "missing",
        timeout=httpx.Timeout(connect=10.0, read=90.0, write=30.0, pool=10.0),
    )
    model_id = model if bare_model else f"{provider}@{model}"
    def _to_openai_msg(m):
        # Tool-fähige Konversation (2026-11-10, User: „immer über uniinfer"):
        # rohe OpenAI-Shape-Dicts (assistant.tool_calls, tool-Rolle mit
        # tool_call_id) VERBATIM durchreichen; ChatMessage-Objekte inklusive
        # ihrer Tool-Felder konvertieren — vorher fielen tool_calls sowohl in
        # der Anfrage als auch in der Antwort weg.
        if isinstance(m, dict):
            return m
        d = {"role": getattr(m, "role", None) or "user", "content": getattr(m, "content", None)}
        if getattr(m, "tool_calls", None):
            d["tool_calls"] = [
                tc if isinstance(tc, dict) else (tc.model_dump() if hasattr(tc, "model_dump") else dict(tc))
                for tc in m.tool_calls
            ]
        if getattr(m, "tool_call_id", None):
            d["tool_call_id"] = m.tool_call_id
        rd = getattr(m, "reasoning_details", None)
        if rd and isinstance(d, dict):
            d.setdefault("extra_fields", {})["reasoning_details"] = rd
        return d

    openai_messages = [_to_openai_msg(m) for m in messages]
    # Top-Level-OpenAI-Parameter (Tool-Calls!) NICHT in extra_body — derselbe
    # Vertrag wie _invoke_via_openai_sdk (in extra_body ignoriert der Gateway
    # tools still).
    _TOP_LEVEL = ("tools", "tool_choice", "parallel_tool_calls", "response_format")
    top_kwargs = {k: request_kwargs[k] for k in _TOP_LEVEL if k in request_kwargs}
    extra = {k: v for k, v in request_kwargs.items() if k not in _TOP_LEVEL and k != "extra_body"}
    if isinstance(request_kwargs.get("extra_body"), dict):
        extra = {**request_kwargs["extra_body"], **extra}
    raw_stream = client.chat.completions.create(
        model=model_id,
        messages=openai_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        extra_body=extra or None,
        **top_kwargs,
    )

    def _adapt(raw_iter):
        for raw in raw_iter:
            delta = raw.choices[0].delta if raw.choices else None
            content = getattr(delta, "content", None) if delta is not None else None
            reasoning = getattr(delta, "reasoning_content", None) if delta is not None else None
            if not reasoning and delta is not None:
                reasoning = getattr(delta, "thinking", None)
            # Tool-Call-DELTAS durchreichen (2026-11-10): OpenAI-Streaming
            # liefert tool_calls in Fragmenten ({index, id?, function:{name?,
            # arguments-part}}) — roh yielden, die Akkumulation gehört EINER
            # Stelle (assemble_tool_call_deltas), aufgerufen von iter_llm_events.
            tcd = getattr(delta, "tool_calls", None) if delta is not None else None
            finish = getattr(raw.choices[0], "finish_reason", None) if raw.choices else None
            usage = getattr(raw, "usage", None)
            usage = None if usage is None else (
                usage if isinstance(usage, dict)
                else usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
            )
            yield SimpleNamespace(
                message=SimpleNamespace(
                    content=content, reasoning_content=reasoning,
                    tool_calls_delta=tcd,
                ),
                thinking=None,
                usage=usage,
                finish_reason=finish,
            )

    return _adapt(raw_stream)


def _start_stream(
    ref: ModelRef,
    cfg: ResolvedConfig,
    msgs: list[ChatMessage],
    request_kwargs: dict,
):
    """Open a stream + capture the first non-empty chunk, retrying transient errors.

    Returns ``(stream_iterator, first_chunk)`` — the first chunk with content,
    reasoning_content ODER thinking (siehe :func:`_chunk_has_payload`).
    Raises on exhaustion.
    This is the "before first token" phase — cheap to retry/backup (Q20).
    """
    attempts = cfg.retry.attempts
    err: BaseException = RuntimeError("not attempted")
    for attempt in range(attempts):
        try:
            if cfg.base_url:
                # Gateway-Zweig (Vorfall 2026-11-04, dgx-hab-episoden): der
                # Gateway routet per provider@model-Präfix — auch Alias-only-
                # Instanzen (hab), die die ProviderFactory nicht kennt. Der
                # alte Weg (create_provider mit base_url-Override) warf für
                # Aliase ValueError "not registered" (die Factory resolved
                # per Name, bevor das base_url-kwarg irgendwo ankam) und
                # verheizte den Primary-Kettenplatz + 20–30 s an toten
                # Backups pro Step (ttfb 32–44 s).
                stream = _stream_via_openai_sdk(
                    base_url=cfg.base_url, bearer=cfg.bearer,
                    provider=ref.provider, model=ref.model,
                    bare_model=cfg.bare_model, messages=msgs,
                    temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                    request_kwargs=request_kwargs,
                )
            else:
                # Direkter Provider-Call (base_url None → Provider-Default,
                # z.B. TU-Aqueduct) braucht das BARE model — aqueduct 400t auf
                # 'tu@glm-…' („no healthy deployments", stream starb leer und
                # die Kette fiel auf hab-qwen zurück; strukt2meta-Vorfall).
                prov = create_provider(ref.provider, api_key=cfg.bearer)
                request = ChatCompletionRequest(
                    messages=msgs,
                    model=ref.model,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    streaming=True,
                    **request_kwargs,
                )
                stream = prov.stream_complete(request)
            for chunk in stream:
                if _chunk_has_payload(chunk):
                    return stream, chunk
            err = RuntimeError("empty_response")  # stream yielded nothing
        except Exception as exc:
            err = exc

        kind = classify_error(err)
        if kind in PERMANENT_ERRORS:
            break
        if attempt < attempts - 1:
            ra = _extract_retry_after(err) if kind == "rate_limited" else None
            time.sleep(cfg.retry.delay_for(attempt + 1, ra))

    raise err


def stream_llm(
    prompt: str | None = None,
    *,
    config: ResolvedConfig | None = None,
    package: str | None = None,
    client: str | None = None,
    task: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    messages: list[ChatMessage] | None = None,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_attempts: int | None = None,
    env_prefix: str | None = None,
    include_reasoning: bool = False,
    **request_kwargs,
) -> Iterator[str]:
    """Streaming invocation. Yields chunk texts as they arrive.

    Before the first token: behaves like ``call_llm`` — transient errors retry
    with backoff, then escalate to backup (Q20). Once the first token flows,
    the stream runs to completion with no retry/backup (cannot un-stream).

    Args: same resolution shape as ``call_llm`` (minus ``fail_fast``).
        include_reasoning: True → liefert ``StreamPiece``-Stücke (str-Subclass
            mit ``.kind`` ∈ {"reasoning", "content"}) statt nur Content-Text —
            die Denkphase wird live sichtbar, der str-Contract bleibt intakt.
            Reasoning+usage als Events: ``iter_llm_events``.
    Yields: str chunk texts (StreamPiece bei include_reasoning=True).
    """
    cfg = _resolve_call_config(
        config=config, package=package, client=client, task=task,
        model=model, provider=provider,
        temperature=temperature, max_tokens=max_tokens,
        max_attempts=max_attempts, env_prefix=env_prefix,
    )
    msgs = _build_messages(prompt, messages, system_prompt)
    pkg = package or _package_from_env()
    cli = client or os.environ.get("KONTEXT_CLIENT", "").strip() or None

    for ref in cfg.chain:
        try:
            stream, first = _start_stream(ref, cfg, msgs, request_kwargs)
        except Exception as exc:
            emit_alarm(
                "alarm", ref.provider, ref.model,
                classify_error(exc), str(exc),
                package=pkg, client=cli,
            )
            continue  # next backup

        # first token obtained — drain the rest, no backup (Q20)
        def _emit(chunk):
            if include_reasoning:
                for kind, text in _chunk_events(chunk):
                    if kind != "usage":
                        yield StreamPiece(text, kind)
                return
            content = getattr(getattr(chunk, "message", None), "content", None)
            if isinstance(content, str) and content:
                yield content

        yield from _emit(first)
        try:
            for chunk in stream:
                yield from _emit(chunk)
        except Exception as exc:
            emit_alarm(
                "alarm", ref.provider, ref.model,
                classify_error(exc), str(exc) + " (mid-stream)",
                package=pkg, client=cli,
            )
        return  # done — even if mid-stream errored, we don't backup


def assemble_tool_call_deltas(frags: list) -> list[dict]:
    """OpenAI-Streaming-Tool-Call-Fragmente zu vollen tool_call-Dicts summieren.

    Der Stream liefert je Fragment ``{index, id?, type?, function: {name?,
    arguments-teilstring}}`` — id/name im ersten Fragment, arguments werden
    über alle Fragmente konateniert. EINE Akkumulations-Stelle (llminvoke),
    Konsumenten bekommen fertige OpenAI-Shape-Dicts
    (``{id, type: 'function', function: {name, arguments}}``) — sortiert nach
    index, wie das Modell sie beabsichtigte.
    """
    acc: dict[int, dict] = {}
    for frag in frags or []:
        idx = getattr(frag, "index", None)
        if idx is None and isinstance(frag, dict):
            idx = frag.get("index")
        if idx is None:
            continue
        slot = acc.setdefault(int(idx), {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
        fid = getattr(frag, "id", None) or (frag.get("id") if isinstance(frag, dict) else None)
        if fid:
            slot["id"] = fid
        ftype = getattr(frag, "type", None) or (frag.get("type") if isinstance(frag, dict) else None)
        if ftype:
            slot["type"] = ftype
        fn = getattr(frag, "function", None) or (frag.get("function") if isinstance(frag, dict) else None)
        fname = (getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else None)) if fn else None
        fargs = (getattr(fn, "arguments", None) or (fn.get("arguments") if isinstance(fn, dict) else None)) if fn else None
        if fname:
            slot["function"]["name"] += str(fname)
        if fargs:
            slot["function"]["arguments"] += str(fargs)
    return [acc[k] for k in sorted(acc)]


def _chunk_events(chunk):
    """Yield (kind, text) events from a streaming chunk: reasoning + content + usage.

    Der terminale Usage-Chunk (vLLM/OpenAI stream_options.include_usage — der
    Proxy reicht ihn durch) wird als ``("usage", dict)``-Event ausgeliefert,
    damit Streaming-Konsumenten Token-Zahlen behalten (klark0-UI zeigt
    Dauer+Tokens; Vorfall: nach Streaming-Upgrade alles 0).

    Reasoning kommt je nach Provider an TWO Stellen an:
    - ``chunk.message.reasoning_content`` (manche Provider)
    - ``chunk.thinking`` (uniinfer-Contract: TUProvider legt reasoning dort ab)
    Beide pruefen, erster gewinnt.
    """
    msg = getattr(chunk, "message", None)
    r = getattr(msg, "reasoning_content", None)
    if not (isinstance(r, str) and r):
        r = getattr(chunk, "thinking", None)
    if isinstance(r, str) and r:
        yield ("reasoning", r)
    c = getattr(msg, "content", None)
    if isinstance(c, str) and c:
        yield ("content", c)
    # Tool-Call-Fragmente (Streaming-Tool-Agenten): roh durchreichen — die
    # Akkumulation macht iter_llm_events via assemble_tool_call_deltas.
    tcd = getattr(msg, "tool_calls_delta", None)
    if tcd:
        yield ("tool_calls_delta", tcd)
    # finish_reason (letzer Content-Chunk): 'length' = max_tokens-Stopp —
    # Konsumenten (Agent) failen darauf laut statt abgeschnittene Antworten.
    fin = getattr(chunk, "finish_reason", None)
    if fin:
        yield ("finish", fin)
    # Terminaler Usage-Chunk: choices leer, usage gefüllt → als Event durchreichen
    # (Details flach aufgestempelt — in/out/cached-in, siehe _stamp_usage_details_flat)
    u = getattr(chunk, "usage", None)
    if u:
        d = u if isinstance(u, dict) else (u.model_dump() if hasattr(u, "model_dump") else dict(u))
        if d:
            yield ("usage", _stamp_usage_details_flat(d))


def iter_llm_events(
    prompt: str | None = None,
    *,
    config: ResolvedConfig | None = None,
    package: str | None = None,
    client: str | None = None,
    task: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    messages: list[ChatMessage] | None = None,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_attempts: int | None = None,
    env_prefix: str | None = None,
    **request_kwargs,
):
    """Streaming invocation yielding ``(kind, text)`` events — ``('reasoning', …)`` (the
    thinking) + ``('content', …)`` (the answer).

    Same resolution + retry/backup as ``call_llm`` / ``stream_llm``; once the first
    token flows, the stream runs to completion with no retry/backup (cannot un-stream).
    Use this (not ``stream_llm``) when you want to surface the model's **thinking**
    live (e.g. a streaming audit/Begründung).
    """
    cfg = _resolve_call_config(
        config=config, package=package, client=client, task=task,
        model=model, provider=provider,
        temperature=temperature, max_tokens=max_tokens,
        max_attempts=max_attempts, env_prefix=env_prefix,
    )
    msgs = _build_messages(prompt, messages, system_prompt)
    pkg = package or _package_from_env()
    cli = client or os.environ.get("KONTEXT_CLIENT", "").strip() or None

    # task/profile request_kwargs (temperature-Profile, nothink-Flags,
    # chat_template_kwargs …) MIT den per-call kwargs mergen — sonst verliert
    # der Stream die halbe Konfiguration (Bug: thinking-Events verschwanden,
    # weil enable_thinking-Overrides nie ankamen).
    merged_kwargs = {**cfg.request_kwargs, **request_kwargs}

    for ref in cfg.chain:
        try:
            stream, first = _start_stream(ref, cfg, msgs, merged_kwargs)
        except Exception as exc:
            emit_alarm(
                "alarm", ref.provider, ref.model,
                classify_error(exc), str(exc),
                package=pkg, client=cli,
            )
            continue  # next backup
        # first token obtained — drain the rest, no backup (Q20)
        _tc_frags: list = []
        for kind, payload in _chunk_events(first):
            if kind == "tool_calls_delta":
                _tc_frags.extend(payload)
                continue
            yield (kind, payload)
        try:
            for chunk in stream:
                for kind, payload in _chunk_events(chunk):
                    if kind == "tool_calls_delta":
                        _tc_frags.extend(payload)
                        continue
                    yield (kind, payload)
        except Exception as exc:
            emit_alarm(
                "alarm", ref.provider, ref.model,
                classify_error(exc), str(exc) + " (mid-stream)",
                package=pkg, client=cli,
            )
        # Terminale Ereignisse NACH dem Stream: akkumulierte Tool-Calls als
        # EIN Event (OpenAI-Shape-Dicts) — Streaming-Tool-Agenten (ReactAgent)
        # konsumieren fertig montierte Calls statt Fragmente zu puddeln.
        if _tc_frags:
            yield ("tool_calls", assemble_tool_call_deltas(_tc_frags))
        return  # done — even if mid-stream errored, we don't backup


# ════════════════════════════════════════════════════════════════════════
# Backward-compat: read_model_config (worker /api/worker/version uses this)
# ════════════════════════════════════════════════════════════════════════

def _package_from_env() -> str | None:
    """Best-effort package name from call stack (for alarm attribution)."""
    return os.environ.get("KONTEXT_PACKAGE", "").strip() or None


def read_model_config(config_path: str | None = None, *, env_prefix: str | None = None) -> dict:
    """Read a package's model config, normalized to {default, tiers, tasks}.

    .. deprecated:: 0.2.0
        Prefer :func:`resolve_model` (ADR 0004). This shim remains for the
        worker ``/api/worker/version`` endpoint and legacy callers.
    """
    import json

    result: dict = {"default": None, "tiers": {}, "tasks": {}}

    if config_path and os.path.isfile(config_path):
        try:
            cfg = json.load(open(config_path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cfg = {}

        p, m = cfg.get("provider"), cfg.get("model")
        if p and m:
            result["default"] = f"{p}@{m}"

        llm = cfg.get("llm") or {}
        if isinstance(llm, dict):
            dp, dm = llm.get("default_provider"), llm.get("default_model")
            if dp and dm:
                result["default"] = f"{dp}@{dm}"
            for tier_name, tier_cfg in (llm.get("tiers") or {}).items():
                if isinstance(tier_cfg, dict):
                    models = tier_cfg.get("models") or []
                    if models:
                        result["tiers"][tier_name] = models[0]

        for key, val in cfg.items():
            if key == "llm":
                continue
            if isinstance(val, dict) and val.get("provider") and val.get("model"):
                result["tasks"][key] = f"{val['provider']}@{val['model']}"

    if env_prefix:
        ep = os.environ.get(f"{env_prefix}_PROVIDER", "")
        em = os.environ.get(f"{env_prefix}_MODEL", "")
        if ep and em:
            result["default"] = f"{ep}@{em}"

    return result
