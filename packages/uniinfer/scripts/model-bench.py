#!/usr/bin/env python3
"""model-bench — repeatable upstream/latency testbench for the uniioai proxy.

Answers three questions in one run:
  1. Which models are affected (all / few)?
  2. Is the PROXY (unii) the problem?
  3. Is the UPSTREAM (TU/aqueduct) the problem?

Modes:
  direct  — raw httpx straight at the TU upstream (needs credgoo 'tu' key).
            Run this ON amd:  uv run python scripts/model-bench.py --mode direct
  proxy   — full production path (proxy :8124 or https://uniinfer.skale.dev).
            Needs a grant: --grant-file FILE | UNII_GRANT env | ~/.pi/agent/auth.json
  both    — direct first, then proxy; same models, same box, back to back.

Verdict logic (mode=both, same moment):
  direct slow            -> UPSTREAM (proxy exonerated)
  direct fast, proxy slow -> PROXY/POOL suspect
  all models slow         -> upstream global; one model -> replica pool of that model

Usage:
  uv run python scripts/model-bench.py --mode both --models tu@deepseek-v4-flash-284b \
      --attempts 10 --concurrency 5 --grant-file /tmp/unii_grant
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

TU_DIRECT_BASE = "https://aqueduct.ai.datalab.tuwien.ac.at/v1"
WEDGE_10 = 10.0   # seconds — suspicious
WEDGE_30 = 30.0   # seconds — wedged


def resolve_grant(args) -> str | None:
    if args.grant_file:
        return Path(args.grant_file).read_text().strip()
    if os.environ.get("UNII_GRANT"):
        return os.environ["UNII_GRANT"].strip()
    p = Path("~/.pi/agent/auth.json").expanduser()
    try:
        return json.loads(p.read_text())["unii"]["key"]
    except Exception:
        return None


async def one_attempt(client: httpx.AsyncClient, base: str, model: str,
                      token: str, timeout: float) -> dict:
    """Returns {ttfb, total, ok, done, error} for a streamed chat probe."""
    body = {"model": model, "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 30, "stream": True}
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    t0 = time.monotonic()
    res = {"model": model, "ttfb": None, "total": None, "ok": False, "done": False, "error": None}
    try:
        async with client.stream("POST", f"{base}/chat/completions", json=body,
                                 headers=headers) as r:
            lines = 0
            async for line in r.aiter_lines():
                if line.startswith("data:"):
                    if "[DONE]" in line:
                        res["done"] = True
                        break
                    if res["ttfb"] is None:
                        res["ttfb"] = round(time.monotonic() - t0, 2)
                    lines += 1
                    if "error" in line[:120] and not res["error"]:
                        res["error"] = line[:140]
            res["total"] = round(time.monotonic() - t0, 2)
            res["ok"] = r.status_code == 200 and res["ttfb"] is not None
            res["status"] = r.status_code
    except Exception as e:
        res["total"] = round(time.monotonic() - t0, 2)
        res["error"] = f"{type(e).__name__}: {str(e)[:80]}"
    return res


async def run_matrix(base: str, token: str, models: list[str], attempts: int,
                     concurrency: int, timeout: float, label: str) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {}
    sem = asyncio.Semaphore(concurrency)

    async def guarded(client, m):
        async with sem:
            return await one_attempt(client, base, m, token, timeout)

    print(f"\n=== {label}  ({base})  attempts={attempts} concurrency={concurrency} ===", flush=True)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for m in models:
            tasks = [guarded(client, m) for _ in range(attempts)]
            results[m] = await asyncio.gather(*tasks)
    return results


def summarize(label: str, results: dict[str, list[dict]]) -> dict[str, dict]:
    print(f"\n--- {label}: Zusammenfassung ---")
    print(f"{'model':44s} {'ok':>4s} {'wedge>10s':>9s} {'wedge>30s':>9s} {'ttfb min/med/max':>22s}")
    summary = {}
    for m, rows in results.items():
        tt = [r["ttfb"] for r in rows if r["ttfb"] is not None]
        ok = sum(1 for r in rows if r["ok"])
        w10 = sum(1 for t in tt if t > WEDGE_10)
        w30 = sum(1 for t in tt if t > WEDGE_30)
        if tt:
            stat = f"{min(tt):5.1f} / {statistics.median(tt):5.1f} / {max(tt):5.1f}"
        else:
            stat = "kein Token"
        print(f"{m:44s} {ok:4d} {w10:9d} {w30:9d} {stat:>22s}")
        errs = [r["error"] for r in rows if r["error"]]
        for e in errs[:2]:
            print(f"    err: {e}")
        summary[m] = {"ok": ok, "n": len(rows), "wedge10": w10, "wedge30": w30,
                      "ttfb_min": min(tt) if tt else None,
                      "ttfb_med": statistics.median(tt) if tt else None,
                      "ttfb_max": max(tt) if tt else None}
    return summary


def verdict(direct: dict | None, proxy: dict | None) -> None:
    if not direct or not proxy:
        return
    print("\n=== VERDICT (direct vs proxy, gleiche Minute) ===")
    for m in direct:
        d, p = direct[m], proxy.get(m)
        if not p:
            continue
        d_med = d["ttfb_med"] or 999
        p_med = p["ttfb_med"] or 999
        if d_med > WEDGE_10 and p_med > WEDGE_10:
            v = "UPSTREAM (beide Pfade langsam) — unii entlastet"
        elif d_med <= WEDGE_10 and p_med > WEDGE_10:
            v = "PROXY-VERDACHT (direkt schnell, via unii langsam)"
        else:
            v = "OK (beide schnell)"
        print(f"{m:44s} direct_med={d_med:6.1f}s proxy_med={p_med:6.1f}s -> {v}")


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["direct", "proxy", "both"], default="both")
    ap.add_argument("--models", nargs="+", default=["tu@deepseek-v4-flash-284b"])
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=150.0)
    ap.add_argument("--proxy-base", default="http://127.0.0.1:8124/v1",
                    help="default: local proxy on amd; use https://uniinfer.skale.dev/v1 for the full public path")
    ap.add_argument("--grant-file", default=None, help="file containing the unii bearer grant")
    args = ap.parse_args()

    if args.mode in ("proxy", "both"):
        grant = resolve_grant(args)
        if not grant:
            print("FEHLER: proxy-Modus braucht einen Grant (--grant-file / UNII_GRANT / ~/.pi/agent/auth.json)", file=sys.stderr)
            return 2

    direct_sum = proxy_sum = None
    if args.mode in ("direct", "both"):
        from credgoo import get_api_key
        key = get_api_key("tu")
        direct_models = [m.split("@", 1)[1] for m in args.models]
        res = await run_matrix(TU_DIRECT_BASE, key, direct_models, args.attempts,
                               args.concurrency, args.timeout, "TU DIREKT")
        direct_sum = summarize("TU DIREKT", res)

    if args.mode in ("proxy", "both"):
        grant = resolve_grant(args)
        res = await run_matrix(args.proxy_base, grant, args.models, args.attempts,
                               args.concurrency, args.timeout, "VIA UNII (proxy)")
        proxy_sum = summarize("VIA UNII", res)

    verdict(direct_sum, proxy_sum)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
