"""Validate GLM-5.2-preview degeneration threshold, determinism, and qwen's
behavior at/over its ~200k limit.

max_tokens is kept small (220): the loop onset is in the very first output
tokens, so this still detects degeneration while cutting generation time.

Writes one line per request (flushed) to stdout so we can poll incrementally.
"""
import json
import sys
import time

import httpx

BASE = "http://localhost:8124/v1"
KEY = "test23@test34"
MAXTOK = 220

BLOCK = (
    "The migration working group reviewed the staging cluster logs and found no "
    "anomalies; latency p99 held at 142ms, disk at 63 percent, and the on-call "
    "rotation handled two low-severity tickets. Budget code 4729. Tea was served. "
)
PER = int(len(BLOCK.split()) * 1.3)  # ~33 tokens/block
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


def stream_call(model: str, target_tokens: int):
    reps = max(1, target_tokens // PER)
    prompt = BLOCK * reps + ASK
    approx_in = int(len(prompt.split()) * 1.3)
    parts, rparts = [], []
    t0 = time.monotonic()
    finish = None
    err = None
    try:
        with httpx.Client(timeout=400.0) as c:
            with c.stream("POST", f"{BASE}/chat/completions",
                          headers={"Authorization": f"Bearer {KEY}"},
                          json={
                              "model": model,
                              "messages": [{"role": "user", "content": prompt}],
                              "max_tokens": MAXTOK,
                              "temperature": 0.2,
                              "stream": True,
                          }) as resp:
                if resp.status_code != 200:
                    return approx_in, None, None, None, f"HTTP {resp.status_code}: {resp.read().decode()[:200]}"
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
        return approx_in, None, None, None, f"EXC {type(e).__name__}: {e}"
    return approx_in, "".join(parts), "".join(rparts), finish, (err or f"ok {time.monotonic()-t0:.1f}s")


def report(model, target, label=""):
    approx_in, content, reasoning, finish, status = stream_call(model, target)
    if content is None:
        print(f"[{model.split('@')[1]:18} target~{target:>6} ~in{label}] -> FAILED {status}", flush=True)
        return
    cdeg = looks_degenerate(content)
    rdeg = looks_degenerate(reasoning)
    verdict = "DEGENERATE" if (cdeg or rdeg) else "clean"
    both = (content + " " + reasoning).strip()
    head = both[:70].replace("\n", " ")
    print(f"[{model.split('@')[1]:18} target~{target:>6} ~in={approx_in:>6}{label}] "
          f"{verdict:10} (c={int(cdeg)} r={int(rdeg)}) finish={finish} {status} "
          f"| head={head!r}", flush=True)


def main():
    print(f"=== GLM-5.2-preview (~500k ctx) threshold bisect + determinism "
          f"(max_tokens={MAXTOK}) ===", flush=True)
    # 100k was clean earlier, 180k degenerate -> bisect.
    for t in [110_000, 130_000, 150_000, 170_000]:
        report("tu@glm-5.2-744b-preview", t)
    # determinism: re-run 150k and 180k
    report("tu@glm-5.2-744b-preview", 150_000, label=" rerun")
    report("tu@glm-5.2-744b-preview", 180_000, label=" rerun")
    # over-limit probe: does TU accept ~400k for GLM?
    report("tu@glm-5.2-744b-preview", 400_000, label=" overlimit")

    print(f"\n=== qwen-3.5-397b (~200k ctx) at/over limit ===", flush=True)
    for t in [150_000, 190_000, 230_000, 280_000]:
        report("tu@qwen-3.5-397b", t)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
