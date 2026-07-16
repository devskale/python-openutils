"""Performance probes — the opt-in perf probe set (``PERF_PROBES``).

Throughput (maxspeed across context sizes), context ceiling, rate-limit ceiling,
and token-speed (TTFT + decode tok/s; normal / large-context / cached). Built on
the capability core primitives (``ProbeTarget``, ``_complete_quiet``,
``_completion_target``). Separated from the feature-probe matrix in ``core.py``
because perf probes are a distinct, opt-in probe set carrying their own
measurement machinery. ``core.run_capabilities`` imports ``PERF_PROBES`` lazily
to avoid an import cycle.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

import httpx

from .core import (
    ProbeResult,
    ProbeTarget,
    _complete_quiet,
    _completion_target,
    _ms,
    _short_error,
)


async def perf_maxspeed(t: ProbeTarget) -> ProbeResult:
    """Throughput (tok/min) swept across varying context sizes.

    Token-frugal by design: tiny capped output, one run per size, thinking OFF —
    long perf runs exhaust token-limited (free-tier) models fast. Sizes that
    exceed the model's declared context are skipped.
    """
    started = time.monotonic()
    sizes = getattr(t, "perf_context_sizes", None) or (128, 1024)
    declared_ctx = (getattr(t, "profile", {}).get("context_length") or 0) or None
    rows = []
    for n in sizes:
        if declared_ctx and n >= declared_ctx:
            rows.append(
                {
                    "context_tokens": n,
                    "tok_s": None,
                    "tok_min": None,
                    "note": "exceeds declared ctx",
                }
            )
            continue
        try:
            tok_s = await _measure_throughput(t, n)
        except Exception as e:  # noqa: BLE001
            rows.append(
                {
                    "context_tokens": n,
                    "tok_s": None,
                    "tok_min": None,
                    "note": type(e).__name__,
                }
            )
            continue
        rows.append(
            {
                "context_tokens": n,
                "tok_s": tok_s,
                "tok_min": round(tok_s * 60) if tok_s else None,
            }
        )
    valid = [r["tok_min"] for r in rows if r.get("tok_min")]
    best = max(valid) if valid else None
    parts = []
    for r in rows:
        v = r.get("tok_min")
        parts.append(
            f"{r['context_tokens']}t→{v if v is not None else r.get('note','?')}"
            + (" tok/min" if v is not None else "")
        )
    status = "pass" if valid else "fail"
    head = f"max≈{best} tok/min | " if best else ""
    return ProbeResult(
        "perf_maxspeed",
        status,
        evidence=head + " ; ".join(parts),
        latency_ms=_ms(started),
        detail={"sweep": rows, "declared_context": declared_ctx},
    )


async def perf_context(t: ProbeTarget) -> ProbeResult:
    """Context ceiling: declared (from probe) vs empirical grow (capped)."""
    started = time.monotonic()
    try:
        declared = t.profile.get("context_length") if hasattr(t, "profile") else None
        return ProbeResult(
            "perf_context",
            "pass",
            evidence=f"declared_context={declared}",
            latency_ms=_ms(started),
            detail={
                "declared": declared,
                "note": "empirical grow-probe not run by default (expensive)",
            },
        )
    except Exception as e:  # noqa: BLE001
        return ProbeResult(
            "perf_context",
            "error",
            _short_error(e),
            latency_ms=_ms(started),
        )


async def perf_ratelimit(t: ProbeTarget) -> ProbeResult:
    """Rate-limit ceiling. Non-invasive: reads the proxy's learned limit if a
    proxy base is configured; otherwise skips (empirical burst-probe is TODO)."""
    proxy = getattr(t, "proxy_base_url", None)
    if not proxy:
        return ProbeResult(
            "perf_ratelimit",
            "skip",
            evidence="no proxy base; empirical burst-probe not implemented",
        )
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=t.timeout) as client:
            r = await client.get(
                f"{proxy.rstrip('/')}/v1/system/rate-limits",
                headers={"Authorization": f"Bearer {t.api_key}"} if t.api_key else {},
            )
            r.raise_for_status()
            data = r.json()
        entry = None
        for key, val in (data.get("limits") or {}).items():
            if t.provider_model in key or t.model_name in key:
                entry = val
                break
        if not entry:
            return ProbeResult(
                "perf_ratelimit",
                "skip",
                evidence="no learned limit for this model",
                latency_ms=_ms(started),
            )
        return ProbeResult(
            "perf_ratelimit",
            "pass",
            evidence=f"learned_rpm={entry}",
            latency_ms=_ms(started),
            detail={"entry": entry},
        )
    except Exception as e:  # noqa: BLE001
        return ProbeResult(
            "perf_ratelimit",
            "error",
            _short_error(e),
            latency_ms=_ms(started),
        )


async def _measure_throughput(t: ProbeTarget, context_tokens: int) -> Optional[float]:
    """Measure output tok/s for a generation prefilled with ~context_tokens.

    Thinking OFF, output capped at 32 tokens (a short count) — so the token
    cost is ~context_tokens input + a few dozen output.
    """
    filler = "lorem ipsum dolor sit amet " * max(1, context_tokens // 6)
    filler = filler[: context_tokens * 4]
    messages = [
        {
            "role": "user",
            "content": f"Context: {filler}\n\nCount from 1 to 30, one number per line.",
        }
    ]
    started = time.monotonic()
    resp = await asyncio.wait_for(
        _complete_quiet(t, messages, max_tokens=32), t.timeout
    )
    elapsed = max(time.monotonic() - started, 1e-3)
    usage = getattr(resp, "usage", {}) or {}
    out_tokens = usage.get("completion_tokens") or usage.get("eval_count") or 0
    if not out_tokens:
        return None
    return round(out_tokens / elapsed, 2)


def _filler_messages(context_tokens: int, question: str) -> list[dict]:
    """A prompt of ~context_tokens of filler followed by a short question."""
    unit = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
    filler = unit * max(1, context_tokens // 10)
    filler = filler[: context_tokens * 5]
    return [{"role": "user", "content": f"{filler}\n\n{question}"}]


async def _stream_timed(t: ProbeTarget, messages: list[dict], max_tokens: int) -> dict:
    """Stream a completion and time it: TTFT (first token) + decode window.

    Decode tok/s uses the decode window (end − first token), isolating
    generation speed from prefill/latency — avoids the short-output artifact.
    Output tokens are estimated from word count (rough proxy).
    """
    t0 = time.monotonic()
    first: float | None = None
    parts: list[str] = []
    async for chunk in _completion_target(t).astream_complete(
        messages,
        temperature=0.0,
        max_tokens=max_tokens,
        reasoning_effort="none",
    ):
        c = getattr(chunk.message, "content", "") or ""
        if c:
            if first is None:
                first = time.monotonic()
            parts.append(c)
    end = time.monotonic()
    text = "".join(parts)
    ttft = (first - t0) if first is not None else None
    window = (end - first) if first is not None else (end - t0)
    out_est = max(1, len(text.split()))
    decode = (out_est / window) if window > 0 else None
    return {
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "decode_tok_s": round(decode, 1) if decode else None,
        "out_tokens_est": out_est,
        "wall_s": round(end - t0, 3),
    }


async def _cached_pair(t: ProbeTarget, context_tokens: int) -> dict:
    """Send an identical large prompt twice: cold (prefill) vs warm (cached)."""
    msgs = _filler_messages(
        context_tokens, "In one short sentence, what is the above text about?"
    )
    cold = await _stream_timed(t, msgs, 48)
    warm = await _stream_timed(t, msgs, 48)  # byte-identical -> prefix-cache eligible
    ct, wt = cold.get("ttft_s"), warm.get("ttft_s")
    speedup = round(ct / wt, 2) if (ct and wt and wt > 0) else None
    return {
        "cold_ttft_s": ct,
        "warm_ttft_s": wt,
        "speedup": speedup,
        "caching": "active" if (speedup and speedup >= 1.5) else "none/low",
    }


async def perf_tokenspeed(t: ProbeTarget) -> ProbeResult:
    """Token speed across three regimes: normal, large-context sweep, cached.

    Streaming-based: TTFT (prefill+queue) and decode tok/s (generation, isolated
    from latency). Cached = identical large prompt twice -> cold vs warm TTFT.
    """
    started = time.monotonic()
    try:
        declared = (getattr(t, "profile", {}).get("context_length") or 0) or None
        heavy = getattr(t, "heavy_perf", False)  # opt-in: large-context sweep + cached
        normal = await asyncio.wait_for(
            _stream_timed(
                t,
                [
                    {
                        "role": "user",
                        "content": "List 40 creative uses for a brick, one per line.",
                    }
                ],
                120,
            ),
            t.timeout,
        )
        sweep: list[dict] = []
        cached: dict | None = None
        if heavy:
            for s in (sz for sz in (1024, 8192) if not (declared and sz > declared)):
                try:
                    sweep.append(
                        {
                            "ctx": s,
                            **await asyncio.wait_for(
                                _stream_timed(
                                    t,
                                    _filler_messages(
                                        s, "In one sentence, summarize the text above."
                                    ),
                                    64,
                                ),
                                t.timeout,
                            ),
                        }
                    )
                except Exception as e:  # noqa: BLE001
                    sweep.append({"ctx": s, "error": type(e).__name__})
            cache_ctx = (
                4096 if (not declared or 4096 <= declared) else max(512, declared // 2)
            )
            try:
                cached = await asyncio.wait_for(_cached_pair(t, cache_ctx), t.timeout)
            except Exception as e:  # noqa: BLE001
                cached = {"error": type(e).__name__}
        parts = [f"normal TTFT {normal['ttft_s']}s · {normal['decode_tok_s']} tok/s"]
        if sweep:
            parts.append(
                "large "
                + ", ".join(f"{x['ctx']}t:{x.get('ttft_s', '?')}s" for x in sweep)
            )
        if cached:
            sp = cached.get("speedup")
            parts.append(
                f"cached {cached.get('cold_ttft_s', '?')}→{cached.get('warm_ttft_s', '?')}s"
                + (f" ({sp}×)" if sp else "")
            )
        detail: dict = {"normal": normal}
        if heavy:
            detail["large_context"] = sweep
            detail["cached"] = cached
        return ProbeResult(
            "perf_tokenspeed",
            "pass",
            evidence=" | ".join(parts),
            latency_ms=_ms(started),
            detail=detail,
        )
    except Exception as e:  # noqa: BLE001
        return ProbeResult(
            "perf_tokenspeed", "error", _short_error(e), latency_ms=_ms(started)
        )


PERF_PROBES: dict[str, Any] = {
    "tokenspeed": perf_tokenspeed,
    "maxspeed": perf_maxspeed,
    "context": perf_context,
    "ratelimit": perf_ratelimit,
}
