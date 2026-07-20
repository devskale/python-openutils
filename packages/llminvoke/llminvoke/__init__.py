"""llminvoke — shared LLM invocation layer for kontext.one.

Bridges uniinfer (the inference library) + credgoo (API key manager) behind
one small interface. Three invocation levels, all config-aware::

    call_llm   — retry + backoff + backup chain + extract → str   (recommended)
    invoke_llm — one-shot → raw response                        (escape hatch)
    stream_llm — streaming → Iterator[str]                      (before-first-token retry)

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

__version__ = "0.2.0"

__all__ = [
    # invocation
    "invoke_llm",
    "call_llm",
    "stream_llm",
    "create_provider",
    # config (ADR 0004)
    "resolve_model",
    "ResolvedConfig",
    "RetryPolicy",
    "ModelRef",
    "get_model_info",
    "is_dsgvo_provider",
    "classify_error",
    "emit_alarm",
    "read_model_config",
]


# ════════════════════════════════════════════════════════════════════════
# Provider + message helpers
# ════════════════════════════════════════════════════════════════════════

def create_provider(provider: str):
    """credgoo key → ProviderFactory.get_provider. Public for consumers that
    need the raw provider (e.g. custom streaming with chunk inspection)."""
    api_key = get_api_key(provider)
    return ProviderFactory.get_provider(provider, api_key=api_key)


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
      2. ``package=``/``client=``/``task=`` → resolve_model (catalog + clients.yml)
      3. explicit ``model=``+``provider=`` → single-model config (backward compat)

    Per-call overrides (temperature, max_tokens, max_attempts) layer on top.
    """
    if config is not None:
        cfg = config
    elif package or client or task:
        cfg = resolve_model(package, client, task, env_prefix=env_prefix)
    elif model and provider:
        cfg = ResolvedConfig(primary=ModelRef(provider=provider, model=model))
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
    usage data, error classification, or custom extraction). No retry, no
    backup — agentos uses this inside its own chain/breaker loop.
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


# ════════════════════════════════════════════════════════════════════════
# call_llm — retry + backoff + backup chain (the recommended entry point)
# ════════════════════════════════════════════════════════════════════════

def _try_model(
    ref: ModelRef,
    cfg: ResolvedConfig,
    msgs: list[ChatMessage],
    fail_fast: bool,
    request_kwargs: dict,
    *,
    package: str | None,
    client: str | None,
) -> str | None:
    """Try one model with retry/backoff. Returns text on success, None on failure.

    - fail_fast: one shot, no retry.
    - transient errors (429/timeout/network): retry with backoff (honors Retry-After).
    - permanent errors (auth/context-exceeded): stop retrying this model immediately.
    - empty response: treated as a failure (Q8) — retried, then walks to backup.
    """
    attempts = 1 if fail_fast else cfg.retry.attempts
    err: BaseException = RuntimeError("not attempted")
    for attempt in range(attempts):
        try:
            response = invoke_llm(
                model=ref.model,
                provider=ref.provider,
                messages=msgs,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                **request_kwargs,
            )
            text = extract_response_text(response)
            if text:
                return text
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
    return None


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
        text = _try_model(ref, cfg, msgs, fail_fast, request_kwargs, package=pkg, client=cli)
        if text is not None:
            return text
    return ""  # entire chain exhausted


# ════════════════════════════════════════════════════════════════════════
# stream_llm — streaming with before-first-token retry (Q20)
# ════════════════════════════════════════════════════════════════════════

def _start_stream(
    ref: ModelRef,
    cfg: ResolvedConfig,
    msgs: list[ChatMessage],
    request_kwargs: dict,
):
    """Open a stream + capture the first non-empty chunk, retrying transient errors.

    Returns ``(stream_iterator, first_chunk_str)``. Raises on exhaustion.
    This is the "before first token" phase — cheap to retry/backup (Q20).
    """
    attempts = cfg.retry.attempts
    err: BaseException = RuntimeError("not attempted")
    for attempt in range(attempts):
        try:
            prov = create_provider(ref.provider)
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
                content = getattr(getattr(chunk, "message", None), "content", None)
                if isinstance(content, str) and content:
                    return stream, content
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
    **request_kwargs,
) -> Iterator[str]:
    """Streaming invocation. Yields chunk texts as they arrive.

    Before the first token: behaves like ``call_llm`` — transient errors retry
    with backoff, then escalate to backup (Q20). Once the first token flows,
    the stream runs to completion with no retry/backup (cannot un-stream).

    Args: same resolution shape as ``call_llm`` (minus ``fail_fast``).
    Yields: str chunk texts.
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
        yield first
        try:
            for chunk in stream:
                content = getattr(getattr(chunk, "message", None), "content", None)
                if isinstance(content, str) and content:
                    yield content
        except Exception as exc:
            emit_alarm(
                "alarm", ref.provider, ref.model,
                classify_error(exc), str(exc) + " (mid-stream)",
                package=pkg, client=cli,
            )
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
