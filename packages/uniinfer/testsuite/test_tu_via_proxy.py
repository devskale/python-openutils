"""Probe + test tu models through the amd1 proxy.

Reads proxy config from `.env` (PROXYHOST, PROXY_PORT, PROXY_KEY) — never
hardcodes tokens. Two phases, matching the user's instruction "probe first
before test":

  1. READINESS PROBE — hit each LLM tu model with the cheap `chat` probe
     (a real completion). Only models that return `pass` are "ready".
  2. FULL MATRIX — run the capability matrix (tool_calling, structured_output,
     image, thinking_on/off) on up to N ready models and print a comparison.

Usage:
    cd packages/uniinfer
    uv run python testsuite/test_tu_via_proxy.py
    uv run python testsuite/test_tu_via_proxy.py --max 3
    uv run python testsuite/test_tu_via_proxy.py --models qwen-3.5-397b,gemma-4-e2b-it
"""
import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

from dotenv import load_dotenv

# LLM-capable tu models (excludes embedding/TTS/ASR/image-gen).
LLM_MODELS = [
    "qwen-3.5-397b",
    "qwen-3.6-35b",
    "glm-5.2-744b-preview",
    "gemma-4-e2b-it",
]

# Probes that exercise real capabilities (the high-value gaps).
MATRIX_PROBES = ["chat", "tool_calling", "structured_output", "image", "thinking_on", "thinking_off"]


def load_config() -> tuple[str, str]:
    """Load proxy base URL + bearer token from .env. Never hardcode."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)
    host = os.getenv("PROXYHOST", "localhost")
    port = os.getenv("PROXY_PORT", "8123")
    key = os.getenv("PROXY_KEY")
    if not key:
        sys.exit(f"PROXY_KEY not set in {env_path}")
    base = f"https://{host}:{port}/v1"
    return base, key


def probe(base: str, key: str, model: str, probes: str | None = None) -> dict:
    """Call the proxy capabilities endpoint. Returns the parsed report dict."""
    url = f"{base}/system/capabilities?model={model}"
    if probes:
        url += f"&probes={probes}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "detail": e.read().decode()[:200]}
    except Exception as e:
        return {"error": type(e).__name__, "detail": str(e)[:200]}


def status_mark(status: str) -> str:
    return {"pass": "✅", "fail": "❌", "skip": "⏭️ ", "error": "💥"}.get(status, "❓")


def phase1_readiness(base: str, key: str, models: list[str]) -> list[str]:
    """Probe each model with the cheap chat probe. Return models that are ready."""
    print("=" * 64)
    print("  PHASE 1 — READINESS PROBE (chat probe = real completion)")
    print("=" * 64)
    ready = []
    for m in models:
        model = f"tu@{m}"
        rep = probe(base, key, model, probes="chat")
        if "error" in rep:
            print(f"  💥 {m:24} {rep['error']}: {rep.get('detail','')[:50]}")
            continue
        results = rep.get("results", [])
        r = next((x for x in results if x["name"] == "chat"), {})
        status = r.get("status", "?")
        ev = r.get("evidence", "")[:50]
        lat = r.get("latency_ms")
        lat_s = f" {lat:.0f}ms" if lat else ""
        print(f"  {status_mark(status)} {m:24} chat={status:5}{lat_s}  {ev}")
        if status == "pass":
            ready.append(m)
    print(f"\n  Ready: {len(ready)}/{len(models)} -> {ready}\n")
    return ready


def phase2_matrix(base: str, key: str, models: list[str]) -> None:
    """Run the full capability matrix on each ready model and compare."""
    print("=" * 64)
    print("  PHASE 2 — CAPABILITY MATRIX (tool_calling, structured, image, thinking)")
    print("=" * 64)
    for m in models:
        model = f"tu@{m}"
        print(f"\n  --- {model} ---")
        rep = probe(base, key, model, probes=",".join(MATRIX_PROBES))
        if "error" in rep:
            print(f"    💥 {rep['error']}: {rep.get('detail','')[:60]}")
            continue
        for r in rep.get("results", []):
            name = r["name"]
            status = r["status"]
            ev = r.get("evidence", "")[:70]
            lat = r.get("latency_ms")
            lat_s = f" {lat:.0f}ms" if lat else ""
            print(f"    {status_mark(status)} {name:18} {status:5}{lat_s}  {ev}")


def main():
    parser = argparse.ArgumentParser(description="Probe + test tu models via the amd1 proxy")
    parser.add_argument("--models", help="comma-separated model list (default: all LLM tu models)")
    parser.add_argument("--max", type=int, default=3, help="max models to run the full matrix on (default 3)")
    args = parser.parse_args()

    models = args.models.split(",") if args.models else LLM_MODELS
    base, key = load_config()
    print(f"Proxy: {base}  (key from .env)\n")

    ready = phase1_readiness(base, key, models)
    if not ready:
        sys.exit("No ready models — aborting before test phase.")

    selected = ready[: args.max]
    phase2_matrix(base, key, selected)

    print("\n" + "=" * 64)
    print("  DONE")
    print("=" * 64)


if __name__ == "__main__":
    main()
