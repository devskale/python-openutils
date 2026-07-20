"""Probe + tok/s benchmark for ANY OpenAI-compatible endpoint.

Provider-agnostic and endpoint-agnostic: point it at a uniinfer proxy, vLLM,
Ollama, OpenAI, ... — anything that speaks /v1/chat/completions. No uniinfer
specifics, so it drops into any OpenAI deployment.

Config (args or env):
  --base-url / BASEURL   full base incl. /v1   (default https://localhost:8123/v1)
  --bearer   / BEARER    API key / bearer      (default: none — open endpoints)
  --model    / MODEL     chat model id         (default ollama@qwen3.5:0.8b)
  --reasoning REASONING  send reasoning_effort (none|low|high|off). Omit for
                         strict OpenAI compat. For uniinfer thinking models use
                         `none` to skip reasoning, or `high` to measure it too.
  --quick-tokens N       gen tokens for the quick run   (default 256)
  --long-tokens  N       gen tokens for the long run    (default 1024)
  --runs N               long-run repetition             (default 3)
  --skip-long            only the quick run

Reasoning-model note: thinking models spend the output-token budget on
reasoning before the visible answer, so a SMALL max_tokens starves the answer
(and looks like a hang / empty response). When --reasoning is high/on we
auto-bump the token budget; pass --long-tokens explicitly for big reasoning
runs. Use generous values.

Usage:
  uv run python testsuite/bench_openai.py \
      --base-url https://localhost:8123/v1 --bearer "$PROXY_AUTH" \
      --model tu@qwen-3.6-35b --reasoning none

  # Adopt to ANY other OpenAI-compatible endpoint -- just swap base-url + bearer:
  uv run python testsuite/bench_openai.py \
      --base-url https://other-host:8000/v1 --bearer "sk-..." --model my-model
  # (OpenAI, vLLM, Ollama, LM Studio, ... all speak /v1/chat/completions)

  # or via env vars (no flags):
  BASEURL=https://other-host:8000/v1 BEARER=sk-... MODEL=my-model python bench_openai.py
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import httpx

# Fail-fast reachability gate (seconds): how long to wait on the alive probe
# before declaring a model down. Single-digit so reachability is known within
# seconds. Override with DOWN_TIMEOUT. The tok/s runs use a longer budget since,
# by then, the model is already confirmed alive.
# Self-contained on purpose: this file only needs `httpx`, so it drops into any
# OpenAI-compatible deployment (copy it anywhere, `pip install httpx`).
LIVENESS_TIMEOUT = float(os.getenv("DOWN_TIMEOUT", "8"))

PROMPT = (
    "Write a thorough, well-structured explanation of how transformer attention "
    "works, with a concrete small example. Be detailed and complete."
)


def client(timeout: float = 300.0) -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(connect=15, read=timeout, write=60, pool=60), verify=False)


def reason_field(args: argparse.Namespace) -> dict:
    return {"reasoning_effort": args.reasoning} if args.reasoning else {}


def mark(status: str) -> str:
    return {"pass": "✅", "fail": "❌", "info": "ℹ️", "skip": "⏭️"}.get(status, "•")


def probe(base: str, auth: str, model: str, reasoning: dict) -> float | None:
    print(f"\n{mark('info')} PROBE — chat completion (alive?)")
    # Reachability probe: always reasoning_effort=none so a reasoning model
    # can't burn the gate on a long chain. (The --reasoning flag applies to the
    # tok/s runs, not to "is it reachable".)
    body = {"model": model, "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "max_tokens": 32, "reasoning_effort": "none"}
    H = {"Content-Type": "application/json"}
    if auth:
        H["Authorization"] = f"Bearer {auth}"
    t0 = time.perf_counter()
    try:
        with client(timeout=LIVENESS_TIMEOUT) as c:
            r = c.post(f"{base}/chat/completions", headers=H, json=body)
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            print(f"  {mark('fail')} HTTP {r.status_code}: {r.text[:160]}")
            return None
        content = (r.json().get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        if content:
            print(f"  {mark('pass')} responded in {dt:.2f}s -> {content[:40]!r}")
            return dt
        print(f"  {mark('fail')} HTTP 200 but empty content (model may need larger max_tokens)")
        return None
    except Exception as e:  # noqa: BLE001
        print(f"  {mark('fail')} {type(e).__name__}: {e}")
        return None


def quick_tps(base: str, auth: str, model: str, reasoning: dict, gen_tokens: int) -> dict | None:
    print(f"\n{mark('info')} QUICK tok/s — streaming, gen_tokens≈{gen_tokens}")
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": gen_tokens, "stream": True,
            "stream_options": {"include_usage": True}}
    body.update(reasoning)
    H = {"Content-Type": "application/json"}
    if auth:
        H["Authorization"] = f"Bearer {auth}"
    t0 = time.perf_counter()
    first = None
    toks = None
    try:
        with client() as c:
            with c.stream("POST", f"{base}/chat/completions", headers=H, json=body) as s:
                for line in s.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    if line == "data: [DONE]":
                        break
                    try:
                        d = __import__("json").loads(line[6:])
                    except Exception:  # noqa: BLE001
                        continue
                    delta = d.get("choices", [{}])[0].get("delta", {})
                    if (delta.get("content") or delta.get("reasoning_content")) and first is None:
                        first = time.perf_counter()
                    u = d.get("usage")
                    if u:
                        toks = u.get("completion_tokens")
        total = time.perf_counter() - t0
        if toks is None:
            ttft = (first - t0) if first else None
            extra = f" TTFT={ttft:.2f}s" if ttft is not None else ""
            print(f"  {mark('info')} streamed but no usage returned (content flowed:{first is not None}){extra}")
            return None
        ttft = (first - t0) if first else None
        gen = max(total - (ttft or 0), 1e-6)
        print(f"  {mark('pass')} completion_tokens={toks}  TTFT={ttft:.2f}s  "
              f"gen_tps={toks / gen:.0f}  overall_tps={toks / total:.0f}  total={total:.2f}s")
        return {"toks": toks, "ttft": ttft, "total": total, "gen_tps": toks / gen, "overall_tps": toks / total}
    except Exception as e:  # noqa: BLE001
        print(f"  {mark('fail')} {type(e).__name__}: {e}")
        return None


def long_tps(base: str, auth: str, model: str, reasoning: dict, gen_tokens: int, runs: int) -> dict | None:
    print(f"\n{mark('info')} LONG tok/s — non-streaming, gen_tokens≈{gen_tokens}, {runs} run(s)")
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": gen_tokens}
    body.update(reasoning)
    H = {"Content-Type": "application/json"}
    if auth:
        H["Authorization"] = f"Bearer {auth}"
    rates, lats = [], []
    try:
        with client() as c:
            for i in range(runs):
                t0 = time.perf_counter()
                r = c.post(f"{base}/chat/completions", headers=H, json=body)
                dt = time.perf_counter() - t0
                if r.status_code != 200:
                    print(f"  {mark('fail')} run{i + 1}: HTTP {r.status_code}: {r.text[:120]}")
                    return None
                u = r.json().get("usage", {})
                toks = u.get("completion_tokens", 0)
                rates.append(toks / dt)
                lats.append(dt)
                print(f"  {mark('pass')} run{i + 1}: {toks} tok in {dt:.2f}s -> {toks / dt:.0f} tok/s")
        med = statistics.median(rates)
        print(f"  {mark('pass')} median {med:.0f} tok/s  (latency median {statistics.median(lats):.2f}s)")
        return {"median_tps": med, "median_lat": statistics.median(lats)}
    except Exception as e:  # noqa: BLE001
        print(f"  {mark('fail')} {type(e).__name__}: {e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe + tok/s for any OpenAI-compatible endpoint")
    ap.add_argument("--base-url", default=os.getenv("BASEURL", "https://localhost:8123/v1"))
    ap.add_argument("--bearer", default=os.getenv("BEARER", ""))
    ap.add_argument("--model", default=os.getenv("MODEL", "ollama@qwen3.5:0.8b"))
    ap.add_argument("--reasoning", default=os.getenv("REASONING", ""),
                   help="reasoning_effort to send (none|low|high|off). Omit for strict compat.")
    ap.add_argument("--quick-tokens", type=int, default=int(os.getenv("QUICK_TOKENS", "256")))
    ap.add_argument("--long-tokens", type=int, default=int(os.getenv("LONG_TOKENS", "1024")))
    ap.add_argument("--runs", type=int, default=int(os.getenv("RUNS", "3")))
    ap.add_argument("--skip-long", action="store_true")
    args = ap.parse_args()

    # Reasoning models need room: a small max_tokens starves the visible answer.
    if args.reasoning in ("high", "on"):
        args.quick_tokens = max(args.quick_tokens, 2048)
        args.long_tokens = max(args.long_tokens, 4096)

    reasoning = reason_field(args)
    print(f"base={args.base_url}  model={args.model}  "
          f"auth={'set' if args.bearer else 'none'}  reasoning={args.reasoning or 'default'}")

    if probe(args.base_url, args.bearer, args.model, reasoning) is None:
        print("\nPROBE FAILED — endpoint/model not responding. Aborting.")
        sys.exit(1)

    quick_tps(args.base_url, args.bearer, args.model, reasoning, args.quick_tokens)
    if not args.skip_long:
        long_tps(args.base_url, args.bearer, args.model, reasoning, args.long_tokens, args.runs)

    print(f"\nDone. (slow? increase --long-tokens for reasoning models, or set --reasoning none)")


if __name__ == "__main__":
    main()
