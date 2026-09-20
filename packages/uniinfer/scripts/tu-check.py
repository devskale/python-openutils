#!/usr/bin/env python3
"""tu-check — quick health check of all TU chat models, direct AND via the proxy.

One probe per model per path (quota-friendly). Prints a status table + verdict.
Exit code: 0 = all healthy · 1 = degraded (something slow/missing) · 2 = TU down.

Usage (on amd):
    uv run python scripts/tu-check.py                # direct + proxy
    uv run python scripts/tu-check.py --json         # machine-readable (cron)
    uv run python scripts/tu-check.py --direct-only  # ohne Grant
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

TU_DIRECT_BASE = "https://aqueduct.ai.datalab.tuwien.ac.at/v1"
CHAT_MODELS = [
    "deepseek-v4-flash-284b",
    "qwen-3.5-397b",
    "qwen-3.6-35b",
    "qwen-3.6-35b-vllm",
]
HEALTHY_S = 5.0    # <= this TTFB = healthy
DEGRADED_S = 20.0  # <= this = degraded; above / no token / error = down


def resolve_grant() -> str | None:
    if os_env := __import__("os").environ.get("UNII_GRANT"):
        return os_env.strip()
    p = Path("~/.pi/agent/auth.json").expanduser()
    try:
        return json.loads(p.read_text())["unii"]["key"]
    except Exception:
        return None


async def probe(client: httpx.AsyncClient, base: str, token: str, model: str,
                timeout: float, prefix: str = "") -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 20, "stream": True}
    t0 = time.monotonic()
    out = {"model": model, "ttfb": None, "done": False, "error": None, "status": None}
    try:
        async with client.stream("POST", f"{base}/chat/completions", json=body,
                                 headers={"Authorization": f"Bearer {token}",
                                          "Content-Type": "application/json"}) as r:
            out["status"] = r.status_code
            if r.status_code != 200:
                body_text = (await r.aread()).decode(errors="replace")[:120]
                out["error"] = f"HTTP {r.status_code}: {body_text}"
                out["total"] = round(time.monotonic() - t0, 1)
                return out
            async for line in r.aiter_lines():
                if line.startswith("data:"):
                    if "[DONE]" in line:
                        out["done"] = True
                        break
                    if out["ttfb"] is None:
                        out["ttfb"] = round(time.monotonic() - t0, 1)
                    if "error" in line[:120] and not out["error"]:
                        out["error"] = line[:90]
    except Exception as e:
        out["error"] = f"{type(e).__name__} after {time.monotonic()-t0:.0f}s"
    return out


def verdict(res: dict) -> tuple[str, int]:
    """Returns (status_word, severity) — 0 healthy, 1 degraded, 2 down."""
    if res["error"] or (res["ttfb"] is None and not res["done"]):
        return "DOWN", 2
    t = res["ttfb"]
    if t is None or t > DEGRADED_S:
        return "DEGRADED", 1
    if t > HEALTHY_S:
        return "DEGRADED", 1
    return "healthy", 0


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timeout", type=float, default=100.0)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--direct-only", action="store_true", help="skip proxy path (no grant needed)")
    ap.add_argument("--proxy-base", default="http://127.0.0.1:8124/v1")
    args = ap.parse_args()

    from credgoo import get_api_key
    key = get_api_key("tu")
    grant = None if args.direct_only else resolve_grant()

    checks = [("direct", TU_DIRECT_BASE, key)]
    if grant:
        checks.append(("proxy", args.proxy_base, grant))

    all_results: dict[str, dict] = {}
    worst = 0
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        for label, base, token in checks:
            for m in CHAT_MODELS:
                model_id = m if label == "proxy" else m  # proxy needs "tu@<model>"
                res = await probe(client, base, token, (f"tu@{m}" if label == "proxy" else m), args.timeout)
                status, sev = verdict(res)
                worst = max(worst, sev)
                all_results[f"{label}:{m}"] = {**res, "status": status}
                if not args.json:
                    tt = f"{res['ttfb']:6.1f}s" if res["ttfb"] is not None else "     —  "
                    extra = f"  {res['error']}" if res["error"] else ""
                    print(f"{label:6s} {m:26s} TTFB={tt} [{status}]{extra}", flush=True)

    if args.json:
        print(json.dumps({"checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                          "worst_severity": worst, "results": all_results}, indent=1))
        return worst

    if not args.json:
        counts = {"healthy": 0, "DEGRADED": 0, "DOWN": 0}
        for r in all_results.values():
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        print(f"\nVERDICT: {counts['healthy']} healthy · {counts['DEGRADED']} degraded · "
              f"{counts['DOWN']} down  ->  "
              + ("TU GESUND" if worst == 0 else "TU DEGRADIERT" if worst == 1 else "TU DOWN/PROBLEME"))
    return worst


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
