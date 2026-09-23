#!/usr/bin/env python3
"""unii-check — proxy health + model catalog in one command.

Queries the uniioai proxy: /health (status, memory, TU telemetry, per-model
TTFT/latency) and /v1/models (catalog size, per-provider breakdown).
Exit code: 0 ok · 1 warn/degraded · 2 down/unreachable.

Usage:
    uv run python scripts/unii-check.py                       # local proxy (amd)
    uv run python scripts/unii-check.py --base https://uniinfer.skale.dev
    uv run python scripts/unii-check.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

DEFAULT_BASE = "http://127.0.0.1:8124"


def _get(url: str, timeout: float = 15.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def check(base: str, timeout: float) -> tuple[dict, dict, int]:
    severity = 0
    health: dict = {}
    models: dict = {}
    try:
        health = _get(f"{base}/health", timeout)
    except Exception as e:
        print(f"health: UNERREICHBAR ({type(e).__name__}: {e})")
        return {}, {}, 2

    status = health.get("status", "?")
    severity = {"ok": 0, "warn": 1, "crit": 2}.get(status, 2)

    print(f"status: {status} | uptime: {health.get('uptime_seconds')}s | "
          f"loop: {health.get('event_loop_latency_ms')}ms | version: {health.get('version')}")
    mem = health.get("memory", {})
    print(f"mem: {mem.get('rss_mb')}MB rss | headroom: {mem.get('headroom_mb')}MB | peak: {mem.get('peak_mb')}MB")

    tu = (health.get("upstream") or {}).get("tu")
    if tu:
        keys = ("open_stalls", "body_stalls", "prime_retries", "prime_timeouts",
                "rate_limits", "throttled", "in_flight", "stuck_streams")
        print("tu:", json.dumps({k: tu.get(k) for k in keys}))
        lpt = tu.get("last_prime_timeout")
        if lpt:
            print(f"  last_prime_timeout: {lpt.get('model')} vor {lpt.get('ago_s')}s")

    m24 = health.get("models_24h") or []
    if m24:
        print("models_24h (top):")
        for m in m24:
            print(f"   {str(m.get('model')):44s} req={m.get('req'):>4} "
                  f"ttft={str(m.get('avg_ttft_s')):>6s}s lat={m.get('avg_latency_s')}s")

    try:
        models = _get(f"{base}/v1/models", timeout)
        data = models.get("data", [])
        provs: dict[str, int] = {}
        for m in data:
            p = m.get("provider") or str(m.get("id", "")).split("@", 1)[0]
            provs[p] = provs.get(p, 0) + 1
        print(f"catalog: {len(data)} modelle, {len(provs)} provider")
        for p, n in sorted(provs.items(), key=lambda x: -x[1]):
            print(f"   {p:16s} {n:4d}")
        if not data:
            severity = max(severity, 1)
    except Exception as e:
        print(f"catalog: FEHLER ({type(e).__name__}: {e})")
        severity = max(severity, 1)

    return health, models, severity


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    health, models, severity = check(args.base.rstrip("/"), args.timeout)
    if args.json:
        print(json.dumps({"health": health, "model_count": len(models.get("data", []))}, indent=1))
    return severity


if __name__ == "__main__":
    sys.exit(main())
