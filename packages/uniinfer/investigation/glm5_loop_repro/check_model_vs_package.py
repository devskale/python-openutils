"""Decisive isolation test: is the GLM-5.2-preview degeneration a model/serving
bug, or a uniinfer-package bug?

A/B with an IDENTICAL prompt and IDENTICAL params:
  A) DIRECT to TU Aqueduct (https://aqueduct.ai.datalab.tuwien.ac.at/v1)
  B) THROUGH the uniinfer proxy (http://localhost:8124/v1)

uniinfer forwards a plain OpenAI payload (no tool-parser injection, no content
transformation), so if BOTH paths degenerate identically, the fault is upstream
(TU vLLM serving / the GLM model), and uniinfer is exonerated. If only B
degenerates, the package is implicated.

Both use: temperature=0.2, max_tokens=220, stream=true, same ~180k prompt
(the size that reproduced the '0,0.0...' loop earlier).
"""
import json
import time

import httpx
from credgoo import get_api_key

TU_KEY = get_api_key("tu")
TU_DIRECT = "https://aqueduct.ai.datalab.tuwien.ac.at/v1/chat/completions"
PROXY = "http://localhost:8124/v1/chat/completions"
PROXY_KEY = "test23@test34"

MODEL_DIRECT = "glm-5.2-744b-preview"   # raw model id as TU expects it
MODEL_PROXY = "tu@glm-5.2-744b-preview"  # proxy-prefixed id

MAXTOK = 220
TARGET_TOKENS = 180_000

BLOCK = (
    "The migration working group reviewed the staging cluster logs and found no "
    "anomalies; latency p99 held at 142ms, disk at 63 percent, and the on-call "
    "rotation handled two low-severity tickets. Budget code 4729. Tea was served. "
)
PER = int(len(BLOCK.split()) * 1.3)
ASK = ("\n\nIgnoring everything above, answer in one short sentence: what is "
       "2+2, and what is the capital of Austria?")


def looks_degenerate(s: str) -> bool:
    if not s or len(s) < 8:
        return False
    n = len(s)
    for period in (1, 2, 3):
        run = 1
        for i in range(period, n, period):
            if s[i:i + period] == s[i - period:i] and len(s[i:i + period]) == period:
                run += 1
                if run * period >= 10:
                    return True
            else:
                run = 1
    return False


def run(label: str, url: str, headers: dict, model: str, prompt: str):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": MAXTOK,
        "stream": True,
    }
    parts, rparts = [], []
    t0 = time.monotonic()
    finish = None
    err = None
    try:
        with httpx.Client(timeout=400.0) as c:
            with c.stream("POST", url, headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    return (f"HTTP {resp.status_code}: {resp.read().decode()[:200]}",
                            None, None, None)
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[len("data: "):].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        d = json.loads(payload)
                    except Exception:
                        continue
                    if "error" in d:
                        err = json.dumps(d["error"])[:200]
                        break
                    for ch in d.get("choices", []):
                        delta = ch.get("delta", {}) or {}
                        if delta.get("content"):
                            parts.append(delta["content"])
                        if delta.get("reasoning_content"):
                            rparts.append(delta["reasoning_content"])
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
    except Exception as e:
        return f"EXC {type(e).__name__}: {e}", None, None, None
    content = "".join(parts)
    reasoning = "".join(rparts)
    dt = time.monotonic() - t0
    return (err or f"ok {dt:.1f}s"), content, reasoning, finish


def main():
    reps = max(1, TARGET_TOKENS // PER)
    prompt = BLOCK * reps + ASK
    approx_in = int(len(prompt.split()) * 1.3)
    print(f"prompt ~{approx_in} input tokens, max_tokens={MAXTOK}, temp=0.2, stream=true")
    print(f"identical prompt sent to both targets.\n")

    direct_headers = {"Authorization": f"Bearer {TU_KEY}", "Content-Type": "application/json"}
    proxy_headers = {"Authorization": f"Bearer {PROXY_KEY}", "Content-Type": "application/json"}

    print("=== A) DIRECT to TU Aqueduct (bypasses uniinfer) ===")
    status_a, ca, ra, fa = run("DIRECT", TU_DIRECT, direct_headers, MODEL_DIRECT, prompt)
    if ca is None:
        print(f"  FAILED: {status_a}\n")
    else:
        deg = looks_degenerate(ca) or looks_degenerate(ra)
        both = (ca + " " + ra).strip()
        print(f"  degenerate={deg} finish={fa} {status_a}")
        print(f"  head={(both[:90] or '(empty)')!r}")
        print(f"  tail={(both[-90:] or '(empty)')!r}\n")

    print("=== B) THROUGH uniinfer proxy ===")
    status_b, cb, rb, fb = run("PROXY", PROXY, proxy_headers, MODEL_PROXY, prompt)
    if cb is None:
        print(f"  FAILED: {status_b}\n")
    else:
        deg = looks_degenerate(cb) or looks_degenerate(rb)
        both = (cb + " " + rb).strip()
        print(f"  degenerate={deg} finish={fb} {status_b}")
        print(f"  head={(both[:90] or '(empty)')!r}")
        print(f"  tail={(both[-90:] or '(empty)')!r}\n")

    print("=== VERDICT ===")
    if ca is not None and cb is not None:
        da = looks_degenerate(ca) or looks_degenerate(ra)
        db = looks_degenerate(cb) or looks_degenerate(rb)
        if da and db:
            print("  BOTH paths degenerate -> MODEL / TU-serving fault. "
                  "uniinfer package EXONERATED.")
        elif db and not da:
            print("  Only proxy degenerates -> uniinfer package IMPLICATED.")
        elif da and not db:
            print("  Only direct degenerates -> inconsistent; re-run.")
        else:
            print("  Neither degenerate at this size; bump context and re-run.")


if __name__ == "__main__":
    main()
