import os
"""Reproduce the GLM-5.2-preview degeneration at extreme context (~240k tokens).

Evidence (logs/tu_raw_chat.log, 2026-06-30): a real request carried 249,038
input tokens against a 262,144-token limit. Homogeneous 100k padding does NOT
trigger the bug, so we push toward the actual ceiling and capture the full
streamed output (content + reasoning) to detect the '0,0.0.0...' loop.

Streams GLM-5.2-preview and (control) qwen-3.5-397b, writes full output to
/tmp/glm_ctx_<model>.txt, and prints a degeneration verdict.
"""
import json
import time

import httpx

BASE = "http://localhost:8124/v1"
KEY = os.getenv("PROXY_KEY")

# Target input-token budgets near the 262144 ceiling.
TARGETS = [180_000, 230_000, 245_000]

BLOCK = (
    "The migration working group reviewed the staging cluster logs and found no "
    "anomalies; latency p99 held at 142ms, disk at 63 percent, and the on-call "
    "rotation handled two low-severity tickets. Budget code 4729. Tea was served. "
)
# ~33 tokens per BLOCK by the word heuristic.
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


def stream_call(model: str, target_tokens: int, max_tokens: int = 1024):
    reps = max(1, target_tokens // PER)
    prompt = BLOCK * reps + ASK
    approx_in = int(len(prompt.split()) * 1.3)
    parts, rparts = [], []
    t0 = time.monotonic()
    finish = None
    err = None
    try:
        with httpx.Client(timeout=600.0) as c:
            with c.stream("POST", f"{BASE}/chat/completions",
                          headers={"Authorization": f"Bearer {KEY}"},
                          json={
                              "model": model,
                              "messages": [{"role": "user", "content": prompt}],
                              "max_tokens": max_tokens,
                              "temperature": 0.2,
                              "stream": True,
                          }) as resp:
                if resp.status_code != 200:
                    return approx_in, None, None, None, f"HTTP {resp.status_code}: {resp.read().decode()[:300]}"
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
                        err = str(d["error"])
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
        return approx_in, None, None, None, f"EXC {type(e).__name__}: {e}"

    dt = time.monotonic() - t0
    return approx_in, "".join(parts), "".join(rparts), finish, (err or f"ok {dt:.1f}s")


def main():
    print("=" * 84)
    print("EXTREME-CONTEXT REPRO: GLM-5.2-preview near the 256k ceiling")
    print("=" * 84)
    for model in ["tu@glm-5.2-744b-preview", "tu@qwen-3.5-397b"]:
        print(f"\n### {model}")
        for tgt in TARGETS:
            approx_in, content, reasoning, finish, status = stream_call(model, tgt)
            tag = model.split("@")[1]
            if content is not None:
                path = f"/tmp/glm_ctx_{tag}_{tgt}.txt"
                with open(path, "w") as f:
                    f.write(f"status={status} finish={finish} approx_in={approx_in}\n")
                    f.write("--- reasoning ---\n")
                    f.write((reasoning or "") + "\n")
                    f.write("--- content ---\n")
                    f.write((content or "") + "\n")
                cdeg = looks_degenerate(content or "")
                rdeg = looks_degenerate(reasoning or "")
                verdict = "DEGENERATE" if (cdeg or rdeg) else "clean"
                print(f"  target~{tgt:>6} (in~{approx_in:>6}): {verdict} "
                      f"(content={cdeg} reasoning={rdeg}) finish={finish} {status}")
                print(f"    content[:120]={(content or '')[:120]!r}")
                print(f"    written -> {path}")
            else:
                print(f"  target~{tgt:>6} (in~{approx_in:>6}): FAILED {status}")


if __name__ == "__main__":
    main()
