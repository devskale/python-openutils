"""Capability probe + matrix for ANY provider's models, through the proxy.

Provider-agnostic — the whole point of uniinfer. Two phases:

  1. READINESS  — hit each model with the cheap `chat` probe (a real
     completion). Only models that return `pass` go on to phase 2.
  2. MATRIX      — run the capability matrix (tool_calling, structured_output,
     image, thinking_on/off) on up to N ready models and print a comparison.

Model selection (exactly one of):
  --models tu@qwen-3.6-35b,mistral@mistral-small-latest   explicit provider@model list
  --provider tu,mistral                                      auto-discover chat models (via /v1/models)

Config: PROXY_URL / PROXY_AUTH (or PROXYHOST/PROXY_PORT/PROXY_KEY via .env).
Self-signed HTTPS is accepted.

Usage:
  uv run python testsuite/test_via_proxy.py --provider tu
  uv run python testsuite/test_via_proxy.py --provider tu --max 3
  uv run python testsuite/test_via_proxy.py --models tu@qwen-3.6-35b,mistral@mistral-small-latest
"""
from __future__ import annotations

import argparse
import sys

from _proxy_common import (
    auth_header,
    LIVENESS_TIMEOUT,
    make_client,
    mark,
    proxy_auth,
    proxy_base_url,
)

# High-value capability gaps we always exercise in phase 2.
MATRIX_PROBES = ["chat", "tool_calling", "structured_output", "image", "thinking_on", "thinking_off"]
# Per-probe timeout for the matrix phase: healthy probes finish <20s; a hung
# probe fails here instead of waiting the full client default.
PROBE_TIMEOUT = 60.0


def discover_models(provider: str, base: str) -> list[str]:
    """Chat models for one provider, via the unauthenticated /v1/models list."""
    with make_client(timeout=LIVENESS_TIMEOUT) as c:
        r = c.get(f"{base}/v1/models")
        r.raise_for_status()
        out = []
        for m in r.json().get("data", []):
            if m.get("provider") != provider:
                continue
            if m.get("type") not in (None, "chat"):
                continue
            mid = m.get("id")
            if mid:
                out.append(f"{provider}@{mid}")
        return out


def capabilities(base: str, model: str, probes: str | None, timeout: float = PROBE_TIMEOUT) -> dict:
    """GET /v1/system/capabilities for one model (optionally a probe subset)."""
    url = f"{base}/v1/system/capabilities?model={model}"
    if probes:
        url += "&probes=" + probes
    try:
        with make_client(timeout=timeout) as c:
            r = c.get(url, headers=auth_header())
            return r.json()
    except Exception as e:  # noqa: BLE001
        return {"error": type(e).__name__, "detail": str(e)[:200]}


def _status(r: dict, name: str) -> tuple[str, str, float | None]:
    res = next((x for x in r.get("results", []) if x["name"] == name), {})
    return res.get("status", "?"), res.get("evidence", "")[:60], res.get("latency_ms")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Capability probe + matrix for any provider's models, via the proxy")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--models", help="comma-separated provider@model list")
    src.add_argument("--provider", help="comma-separated provider list (auto-discovers chat models)")
    ap.add_argument("--max", type=int, default=3, help="max models for the matrix phase (default 3)")
    ap.add_argument("--probes", default=",".join(MATRIX_PROBES),
                    help="phase-2 probe subset (comma-separated)")
    args = ap.parse_args()

    base = proxy_base_url()
    auth = proxy_auth()
    print(f"Proxy: {base}  (auth {'set' if auth else 'none'})\n")

    models: list[str] = []
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        for p in [p.strip() for p in args.provider.split(",") if p.strip()]:
            found = discover_models(p, base)
            print(f"provider {p}: {len(found)} chat model(s) discovered")
            models.extend(found)
    if not models:
        sys.exit("No models selected.")

    # ---- Phase 1: readiness ----
    print("\n" + "=" * 64)
    print("  PHASE 1 — READINESS (chat probe = real completion)")
    print("=" * 64)
    ready: list[str] = []
    for m in models:
        rep = capabilities(base, m, "chat", LIVENESS_TIMEOUT)
        if "error" in rep:
            print(f"  {mark('error')} {m:36} {rep['error']}: {rep.get('detail', '')[:60]}")
            continue
        st, ev, lat = _status(rep, "chat")
        lat_s = f" {int(lat)}ms" if lat is not None else ""
        print(f"  {mark(st)} {m:36} chat={st:5}{lat_s}  {ev}")
        if st == "pass":
            ready.append(m)
    print(f"\n  Ready: {len(ready)}/{len(models)}")
    if not ready:
        sys.exit("No ready models — aborting before matrix phase.")

    # ---- Phase 2: matrix ----
    print("\n" + "=" * 64)
    print("  PHASE 2 — CAPABILITY MATRIX")
    print("=" * 64)
    for m in ready[: args.max]:
        print(f"\n  --- {m} ---")
        rep = capabilities(base, m, args.probes)
        if "error" in rep:
            print(f"    {mark('error')} {rep['error']}: {rep.get('detail', '')[:60]}")
            continue
        for r in rep.get("results", []):
            st = r["status"]
            lat = r.get("latency_ms")
            lat_s = f" {int(lat)}ms" if lat is not None else ""
            print(f"    {mark(st)} {r['name']:18} {st:5}{lat_s}  {r.get('evidence', '')[:70]}")

    print("\n" + "=" * 64)
    print("  DONE")
    print("=" * 64)


if __name__ == "__main__":
    main()
