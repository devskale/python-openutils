import os
"""Live check: does GLM-5.2-preview degenerate on large context?

Compares a short prompt vs a large-context prompt on tu@glm-5.2-744b-preview,
with tu@qwen-3.5-397b as a control. Non-streaming, short max_tokens so a
degenerate/looping reply is obvious.
"""
import time

import httpx

BASE = "http://localhost:8124/v1"
KEY = os.getenv("PROXY_KEY")

SHORT_PROMPT = "Say 'OK' and nothing else."

# ~6k tokens of filler to bloat the context, then a trivial ask.
PADDING = ("This is padding context about the weather, budgets, and tea. " * 600)
LARGE_PROMPT = (
    PADDING
    + "\n\nBased only on the above, reply with the single word: READY"
)


def call(model: str, prompt: str, max_tokens: int = 1500, stream: bool = False):
    t0 = time.monotonic()
    try:
        with httpx.Client(timeout=180.0) as c:
            if not stream:
                r = c.post(
                    f"{BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {KEY}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.2,
                    },
                )
                dt = time.monotonic() - t0
                if r.status_code != 200:
                    return f"HTTP {r.status_code} ({dt:.1f}s): {r.text[:200]}"
                data = r.json()
                choice = data["choices"][0]
                msg = choice.get("message", {}) or {}
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or ""
                flag = "  <<< DEGENERATE" if looks_degenerate(reasoning + content) else ""
                return (
                    f"OK ({dt:.1f}s, finish={choice.get('finish_reason')}): "
                    f"content={content[:200]!r} reasoning_len={len(reasoning)}{flag}"
                )
            else:
                # streaming: collect content, watch for repetition
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
                    parts = []
                    rparts = []
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
                        for ch in d.get("choices", []):
                            delta = (ch.get("delta", {}) or {})
                            if delta.get("content"):
                                parts.append(delta["content"])
                            if delta.get("reasoning_content"):
                                rparts.append(delta["reasoning_content"])
                    dt = time.monotonic() - t0
                    content = "".join(parts)
                    reasoning = "".join(rparts)
                    flag = "  <<< DEGENERATE" if looks_degenerate(reasoning + content) else ""
                    return (
                        f"STREAM OK ({dt:.1f}s): content={content[:200]!r} "
                        f"reasoning_len={len(reasoning)}{flag}"
                    )
    except Exception as e:
        return f"ERROR ({time.monotonic()-t0:.1f}s): {type(e).__name__}: {e}"


def looks_degenerate(s: str) -> bool:
    if not s:
        return False
    # Repeating short unit (period 1/2/3) of length >= 8.
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


def main():
    models = ["tu@glm-5.2-744b-preview", "tu@qwen-3.5-397b"]
    print("=" * 78)
    print("LIVE CHECK: GLM-5.2-preview degeneration vs context size")
    print("=" * 78)

    print("\n--- SHORT context (non-stream) ---")
    for m in models:
        out = call(m, SHORT_PROMPT)
        print(f"  {m:32s} {out}")

    print("\n--- LARGE context (non-stream) ---")
    for m in models:
        out = call(m, LARGE_PROMPT)
        flag = "  <<< DEGENERATE" if looks_degenerate(out) else ""
        print(f"  {m:32s} {out}{flag}")

    print("\n--- LARGE context (stream) ---")
    for m in models:
        out = call(m, LARGE_PROMPT, stream=True)
        flag = "  <<< DEGENERATE" if looks_degenerate(out) else ""
        print(f"  {m:32s} {out}{flag}")


if __name__ == "__main__":
    main()
