"""Scaling probe: find the context size at which GLM-5.2-preview starts emitting
degenerate / looping output in content OR reasoning_content.

Streams each request so we can also see the live token stream, and flags any
short-unit repetition (the reported "0,0.0.0..." symptom).
"""
import time

import httpx

BASE = "http://localhost:8124/v1"
KEY = "test23@test34"

# Roughly ~1.3 tokens/word. Sizes are approximate token budgets for the padding.
SIZES = [8_000, 20_000, 40_000, 70_000, 100_000]
ASK = "\n\nIgnore all of the above. Reply with exactly one word: PONG"


def looks_degenerate(s: str) -> bool:
    if not s or len(s) < 8:
        return False
    n = len(s)
    for period in (1, 2, 3):
        run = 1
        for i in range(period, n, period):
            if s[i:i + period] == s[i - period:i] and len(s[i:i + period]) == period:
                run += 1
                if run * period >= 8:
                    return True
            else:
                run = 1
    return False


def approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def stream_call(model: str, target_tokens: int, max_tokens: int = 1200):
    # Build padding to ~target_tokens. Repeat a varied block so it isn't a
    # single trivial token (which could itself cause loops).
    block = ("The quarterly report notes that revenue increased modestly while "
             "operational costs remained stable across regions. Tea was served. ")
    per_block = approx_tokens(block)
    reps = max(1, target_tokens // per_block)
    padding = block * reps
    prompt = padding + ASK
    actual = approx_tokens(prompt)

    parts, rparts = [], []
    t0 = time.monotonic()
    finish = None
    try:
        with httpx.Client(timeout=300.0) as c:
            with c.stream(
                "POST",
                f"{BASE}/chat/completions",
                headers={"Authorization": f"Bearer {KEY}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                    "stream": True,
                },
            ) as resp:
                if resp.status_code != 200:
                    body = resp.read().decode()[:200]
                    return actual, f"HTTP {resp.status_code}: {body}"
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[len("data: "):].strip()
                    if payload == "[DONE]":
                        break
                    import json as _j
                    try:
                        d = _j.loads(payload)
                    except Exception:
                        continue
                    if "error" in d:
                        return actual, f"STREAM ERR: {d['error']}"
                    for ch in d.get("choices", []):
                        delta = ch.get("delta", {}) or {}
                        if delta.get("content"):
                            parts.append(delta["content"])
                        if delta.get("reasoning_content"):
                            rparts.append(delta["reasoning_content"])
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
    except Exception as e:
        return actual, f"ERROR: {type(e).__name__}: {e}"

    dt = time.monotonic() - t0
    content = "".join(parts)
    reasoning = "".join(rparts)
    cdeg = looks_degenerate(content)
    rdeg = looks_degenerate(reasoning)
    flag = ""
    if cdeg or rdeg:
        flag = f"  <<< DEGENERATE (content={cdeg}, reasoning={rdeg})"
    sample = (reasoning[-160:] if reasoning else content)
    return actual, (f"({dt:.1f}s, finish={finish}) "
                    f"content={content[:80]!r} reasoning_tail={sample!r}{flag}")


def main():
    print("=" * 82)
    print("SCALING PROBE: GLM-5.2-preview degeneration vs context size (stream)")
    print("=" * 82)
    for target in SIZES:
        actual, out = stream_call("tu@glm-5.2-744b-preview", target)
        print(f"  target~{target:>6} tok (actual ~{actual:>6}): {out}")
    print()
    print("--- control: qwen-3.5-397b at the same sizes ---")
    for target in SIZES:
        actual, out = stream_call("tu@qwen-3.5-397b", target)
        print(f"  target~{target:>6} tok (actual ~{actual:>6}): {out}")


if __name__ == "__main__":
    main()
