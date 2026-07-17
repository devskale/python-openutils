#!/usr/bin/env python3
"""Performance probes against a uniinfer proxy (testsuite tier 3).

HOW WE COUNT TOKENS (the definition):
  Total output tokens = ``usage.completion_tokens``. Per the OpenAI spec this
  ALREADY INCLUDES reasoning/thinking tokens; ``completion_tokens_details.
  reasoning_tokens`` is a subset (breakdown), NOT additive — adding it again
  double-counts. So throughput = completion_tokens / generation_time, and that
  covers thinking. We report reasoning_tokens separately when the provider
  exposes it, so the thinking share is visible.

GENERATION vs PREFILL:
  throughput = completion_tokens / (total - TTFT), where TTFT is time to the
  first token (visible content OR reasoning). This isolates generation (incl.
  thinking) from prefill/queue — NOT tokens/total (prefill crushes short runs).

We measure reasoning ON (the realistic thinking-model workload — thinking
included) AND reasoning OFF (pure visible generation) for comparison.

Env: PROXY_URL, PROXY_AUTH, MODEL, RUNS (latency samples, default 4)
Run: uv run python testsuite/perf.py
"""
import json
import os
import statistics
import time

import httpx

PROXY_URL = os.getenv("PROXY_URL", "http://127.0.0.1:8013")
AUTH = os.getenv("PROXY_AUTH", "")
MODEL = os.getenv("MODEL", "ollama@qwen3.5:0.8b")
RUNS = int(os.getenv("RUNS", "4"))
GEN_TOKENS = 512  # large enough that prefill doesn't dominate the decode window

H = {"Content-Type": "application/json"}
if AUTH:
    H["Authorization"] = f"Bearer {AUTH}"

PROMPT = "Write a numbered list of 15 fruits, one per line. Then explain briefly why each is healthy."


def _post(body, timeout=120):
    return httpx.post(f"{PROXY_URL}/v1/chat/completions", headers=H, json=body, timeout=timeout, verify=False)


def _stream(body, timeout=120):
    return httpx.stream("POST", f"{PROXY_URL}/v1/chat/completions", headers=H, json=body, timeout=timeout, verify=False)


def _first_token_time(reasoning):
    """TTFT: time to the first generated token — visible content OR reasoning
    (so thinking-generation time is part of the decode window, not hidden)."""
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": GEN_TOKENS, "stream": True, "reasoning_effort": reasoning}
    first = None
    t0 = time.perf_counter()
    with _stream(body) as r:
        for line in r.iter_lines():
            line = line.strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                delta = json.loads(line[6:])["choices"][0].get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    first = time.perf_counter()
                    break
            except Exception:  # noqa: BLE001
                pass
    return (first - t0) if first else None


def _nonstream(reasoning):
    """Real completion_tokens (incl. thinking) + reasoning breakdown + wall time."""
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": GEN_TOKENS, "reasoning_effort": reasoning}
    t0 = time.perf_counter()
    r = _post(body)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    usage = r.json().get("usage", {})
    out = int(usage.get("completion_tokens", 0) or 0)
    details = usage.get("completion_tokens_details") or {}
    rtok = int(details.get("reasoning_tokens", 0) or 0)
    return out, rtok, dt


def _probe(reasoning):
    """One mode: (completion_tokens, reasoning_tokens, total, ttft, gen_rate)."""
    ttft = _first_token_time(reasoning)
    out, rtok, total = _nonstream(reasoning)
    decode = max(total - (ttft or 0), 1e-6)
    return out, rtok, total, ttft, (out / decode if out else 0.0)


def measure_latency():
    body = {"model": MODEL, "messages": [{"role": "user", "content": "OK"}],
            "max_tokens": 5, "reasoning_effort": "none"}
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        _post(body, timeout=30)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def measure_ttft_at_ctx(ctx_tokens, reasoning="none"):
    filler = "lorem ipsum dolor sit amet " * max(1, ctx_tokens // 6)
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": f"Context: {filler}\n\nReply with: OK."}],
            "max_tokens": 20, "stream": True, "reasoning_effort": reasoning}
    first = None
    t0 = time.perf_counter()
    with _stream(body) as r:
        for line in r.iter_lines():
            line = line.strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                d = json.loads(line[6:])["choices"][0].get("delta", {})
                if d.get("content") or d.get("reasoning_content"):
                    first = time.perf_counter()
                    break
            except Exception:  # noqa: BLE001
                pass
    return (first - t0) if first else None


def _f(x, unit=""):
    return f"{x:.3f}{unit}" if x is not None else "n/a"


def _report_line(label, out, rtok, total, ttft, rate):
    think = f" ({rtok} thinking)" if rtok else ""
    print(f"  {label:<13}: {out} tok{think} | total {total:.2f}s "
          f"(prefill {_f(ttft,'s')} + decode {max(total-(ttft or 0),0):.2f}s) "
          f"-> {rate:.0f} tok/s")


print(f"Perf against {PROXY_URL}  model={MODEL}  gen_tokens={GEN_TOKENS}")
print("counting: completion_tokens (incl. thinking); reasoning_tokens shown as subset")
print("-" * 64)

# Probe reasoning ON (detects thinking) and OFF (pure visible generation).
out, rtok, total, ttft, rate = _probe("high")
out2, rtok2, total2, ttft2, rate2 = _probe("none")
thinking_available = rtok > 0

if thinking_available:
    _report_line("reasoning=high", out, rtok, total, ttft, rate)
    _report_line("reasoning=none", out2, rtok2, total2, ttft2, rate2)
else:
    # No thinking tokens reported (non-thinking model, or provider doesn't break
    # them out) — just the baseline: TTFT + output tok/s.
    _report_line("output", out2, rtok2, total2, ttft2, rate2)

lat = measure_latency()
print(f"  latency       : {lat * 1000:.0f} ms (tiny req, median of {RUNS})")
print("  context scaling (stream TTFT vs prefilled context, reasoning=none):")
for ctx in (50, 500):
    t = measure_ttft_at_ctx(ctx)
    print(f"    ctx~{ctx:<4}: TTFT={_f(t, 's')}")

print("-" * 64)
if thinking_available:
    print(f"Headline: ~{rate:.0f} tok/s incl thinking ({rtok} tok), "
          f"~{rate2:.0f} tok/s pure; TTFT {_f(ttft,'s')}")
else:
    print(f"Headline: ~{rate2:.0f} output tok/s; TTFT {_f(ttft2,'s')} "
          f"(provider doesn't itemize reasoning; completion_tokens includes any thinking)")
