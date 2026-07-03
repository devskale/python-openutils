"""Probe the most likely real-world trigger: large context WITH tools offered.

The TU glm47 tool parser is the known-fragile path (see glm_leak_repair.py).
Homogeneous padding does NOT trigger degeneration even at 100k tokens, so we
test the combination: large multi-turn context + tools. This mirrors real
agent traffic (pi) far more closely than a single padded user turn.
"""
import time

import httpx

BASE = "http://localhost:8124/v1"
KEY = "test23@test34"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name."},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command."},
                },
                "required": ["command"],
            },
        },
    },
]


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


def build_messages(target_tokens: int):
    """Large multi-turn history ending in a simple tool-using ask."""
    block = ("Earlier we discussed the migration plan and reviewed the logs "
             "from the staging cluster. No anomalies were found. Tea was served. ")
    per_block = approx_tokens(block)
    reps = max(1, target_tokens // per_block)
    filler = block * reps
    return [
        {"role": "system", "content": "You are a helpful assistant with tools."},
        {"role": "user", "content": filler + "\n\nWhat is the weather in Vienna?"},
    ]


def stream_call(model: str, target_tokens: int, with_tools: bool, max_tokens: int = 1500):
    msgs = build_messages(target_tokens)
    actual = sum(approx_tokens(m["content"]) for m in msgs)
    parts, rparts, tcs = [], [], []
    t0 = time.monotonic()
    finish = None
    body = {
        "model": model,
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": True,
    }
    if with_tools:
        body["tools"] = TOOLS

    try:
        with httpx.Client(timeout=300.0) as c:
            with c.stream("POST", f"{BASE}/chat/completions",
                          headers={"Authorization": f"Bearer {KEY}"}, json=body) as resp:
                if resp.status_code != 200:
                    return actual, f"HTTP {resp.status_code}: {resp.read().decode()[:200]}"
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
                        if delta.get("tool_calls"):
                            tcs.append("tc")
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
    except Exception as e:
        return actual, f"ERROR: {type(e).__name__}: {e}"

    dt = time.monotonic() - t0
    content = "".join(parts)
    reasoning = "".join(rparts)
    cdeg, rdeg = looks_degenerate(content), looks_degenerate(reasoning)
    flag = f"  <<< DEGENERATE(content={cdeg},reasoning={rdeg})" if (cdeg or rdeg) else ""
    sample = (reasoning[-120:] if reasoning else content)
    return actual, (f"({dt:.1f}s,finish={finish},tc={len(tcs)}) "
                    f"content={content[:60]!r} reasoning_tail={sample!r}{flag}")


def main():
    print("=" * 84)
    print("TOOLS + LARGE CONTEXT PROBE (the known-fragile glm47 path)")
    print("=" * 84)
    for tgt in [8_000, 40_000, 100_000]:
        a, o = stream_call("tu@glm-5.2-744b-preview", tgt, with_tools=True)
        print(f"  GLM  tools=Y target~{tgt:>6} (actual ~{a:>6}): {o}")
    print()
    for tgt in [8_000, 40_000, 100_000]:
        a, o = stream_call("tu@glm-5.2-744b-preview", tgt, with_tools=False)
        print(f"  GLM  tools=N target~{tgt:>6} (actual ~{a:>6}): {o}")
    print()
    for tgt in [8_000, 40_000, 100_000]:
        a, o = stream_call("tu@qwen-3.5-397b", tgt, with_tools=True)
        print(f"  QWEN tools=Y target~{tgt:>6} (actual ~{a:>6}): {o}")


if __name__ == "__main__":
    main()
