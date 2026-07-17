"""Does max_tokens drive the GLM-5.2-preview degeneration, or does input size?

Hypothesis under test: 'max_tokens should not be more than ~50k' (i.e. a large
max_tokens is what triggers the loop).

Prior finding: 183k INPUT with max_tokens=220 (tiny) already degenerates. So we
expect max_tokens to be irrelevant. This script isolates the variable:

  Matrix (DIRECT to TU, model glm-5.2-744b-preview, temp=0.2, stream):
    INPUT:   8k (small)        | 100k (clean-large) | 183k (degenerate-large)
    MAXTOK:  220, 8192, 50000

  Early-break: stop reading once degeneration is detected OR ~400 output tokens
  collected, so a max_tokens=50000 case never actually generates 50k tokens.

If max_tokens were the driver, small input + large max_tokens would degenerate.
If input size is the driver, only the large-input rows degenerate, regardless of
max_tokens.
"""
import json
import time

import httpx
from credgoo import get_api_key

TU_KEY = get_api_key("tu")
URL = "https://aqueduct.ai.datalab.tuwien.ac.at/v1/chat/completions"
MODEL = "glm-5.2-744b-preview"

BLOCK = (
    "The migration working group reviewed the staging cluster logs and found no "
    "anomalies; latency p99 held at 142ms, disk at 63 percent, and the on-call "
    "rotation handled two low-severity tickets. Budget code 4729. Tea was served. "
)
PER = int(len(BLOCK.split()) * 1.3)
ASK = ("\n\nIgnoring everything above, answer in one short sentence: what is "
       "2+2, and what is the capital of Austria?")

EARLY_BREAK_TOKENS = 400


def looks_degenerate(s: str) -> bool:
    if not s or len(s) < 8:
        return False
    n = len(s)
    # Check small periods 1..6 to catch units like '0', '0\n', '0.0', ': 0\n'.
    for period in range(1, 7):
        run = 1
        for i in range(period, n, period):
            cur = s[i:i + period]
            prev = s[i - period:i]
            if cur == prev and len(cur) == period:
                run += 1
                if run * period >= 12:
                    return True
            else:
                run = 1
    return False


def run(input_tokens: int, max_tokens: int):
    reps = max(1, input_tokens // PER)
    prompt = BLOCK * reps + ASK
    approx_in = int(len(prompt.split()) * 1.3)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "stream": True,
    }
    parts, rparts = [], []
    t0 = time.monotonic()
    finish = None
    err = None
    broke = False
    try:
        with httpx.Client(timeout=400.0) as c:
            with c.stream("POST", URL,
                          headers={"Authorization": f"Bearer {TU_KEY}",
                                   "Content-Type": "application/json"},
                          json=body) as resp:
                if resp.status_code != 200:
                    return approx_in, f"HTTP {resp.status_code}: {resp.read().decode()[:160]}"
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
                        err = json.dumps(d["error"])[:160]
                        break
                    for ch in d.get("choices", []):
                        delta = ch.get("delta", {}) or {}
                        if delta.get("content"):
                            parts.append(delta["content"])
                        if delta.get("reasoning_content"):
                            rparts.append(delta["reasoning_content"])
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
                    combined = "".join(parts) + "".join(rparts)
                    if looks_degenerate(combined) or len(combined.split()) > EARLY_BREAK_TOKENS:
                        broke = True
                        break
    except Exception as e:
        return approx_in, f"EXC {type(e).__name__}: {e}"

    content = "".join(parts)
    reasoning = "".join(rparts)
    deg = looks_degenerate(content) or looks_degenerate(reasoning)
    both = (content + " " + reasoning).strip()
    dt = time.monotonic() - t0
    note = " (early-break)" if broke else ""
    return approx_in, (f"degenerate={deg} finish={finish} {dt:.1f}s{note} "
                       f"| head={both[:60]!r}")


def main():
    # Focused matrix: does max_tokens matter, and is small input reliably bad?
    plan = [
        # (input_tokens, max_tokens, label)
        (8_000, 220, "small-in small-mt"),
        (8_000, 50_000, "small-in huge-mt"),
        (8_000, 220, "small-in small-mt (rerun)"),
        (8_000, 220, "small-in small-mt (rerun2)"),
        (183_000, 220, "large-in small-mt"),
        (183_000, 50_000, "large-in huge-mt"),
    ]
    print(f"DIRECT to TU, model={MODEL}, temp=0.2, stream. early-break at "
          f">{EARLY_BREAK_TOKENS} out tokens or degeneration. periods 1..6.\n")
    print(f"{'label':32} | {'~in':>8} | {'maxt':>6} | result")
    print("-" * 100)
    for inp, mt, label in plan:
        approx_in, res = run(inp, mt)
        print(f"{label:32} | {approx_in:>8} | {mt:>6} | {res}", flush=True)


if __name__ == "__main__":
    main()
