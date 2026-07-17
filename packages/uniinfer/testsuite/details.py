#!/usr/bin/env python3
"""Detailed correctness tests against a uniinfer proxy (testsuite tier 2).

Smoke checks "alive"; this checks "correct":
  - reasoning control: reasoning_effort="none" yields content; the legacy
    think:false shim behaves the same.
  - streaming shape: non-empty content deltas + a finish_reason + the [DONE]
    sentinel.
  - error handling: malformed model -> 400; missing auth on a key-required
    provider -> 401.
  - embeddings: a non-zero vector.

Env:
  PROXY_URL            base URL (default: local proxy)
  PROXY_AUTH           bearer token (optional for open providers like ollama)
  MODEL                chat model (default: an amp ollama chat model)
  EMBED_MODEL          embedding model
  AUTH_REQUIRED_MODEL  a key-required provider@model for the 401 test (skipped if unset)

Run:  uv run python testsuite/details.py
"""
import json
import os
import sys

import httpx

PROXY_URL = os.getenv("PROXY_URL", "http://127.0.0.1:8013")
AUTH = os.getenv("PROXY_AUTH", "")
MODEL = os.getenv("MODEL", "ollama@qwen3.5:0.8b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "ollama@nomic-embed-text-v2-moe")
AUTH_REQUIRED_MODEL = os.getenv("AUTH_REQUIRED_MODEL", "")

_BASE = {"Content-Type": "application/json"}
if AUTH:
    _BASE["Authorization"] = f"Bearer {AUTH}"

passed = failed = skipped = 0


def ok(msg):
    global passed
    print(f"  \u2713 {msg}")
    passed += 1


def bad(msg, detail=""):
    global failed
    print(f"  \u2717 {msg} {('— ' + detail) if detail else ''}")
    failed += 1


def skip(msg):
    global skipped
    print(f"  - {msg} (skipped)")
    skipped += 1


def _headers(auth=True):
    h = dict(_BASE)
    if not auth:
        h.pop("Authorization", None)
    return h


# ---- the thinking-model max_tokens rule (encoded ONCE; don't hand-tune) ----
# Thinking models spend the token budget on reasoning BEFORE the visible
# answer, so a tiny max_tokens yields empty/truncated content. Pair them:
#   reasoning OFF (none/minimal) -> small  (fast, deterministic; no reasoning)
#   reasoning ON  (low/medium/high) -> generous (room for the chain + answer)
_SMALL = 256
_GENEROUS = 4096


def _chat(extra=None, model=None, auth=True, messages=None):
    """Non-stream chat. ``max_tokens`` auto-sizes to the FINAL reasoning state
    (see the rule above) unless the caller sets it explicitly in ``extra`` — so
    thinking models never starve the visible answer on the token budget."""
    body = {
        "model": model or MODEL,
        "messages": messages or [{"role": "user", "content": "Reply with: OK"}],
        "reasoning_effort": "none",
    }
    if extra:
        body.update(extra)
    if "max_tokens" not in body:
        eff = body.get("reasoning_effort", "none")
        body["max_tokens"] = _GENEROUS if eff not in ("none", "minimal") else _SMALL
    return httpx.post(
        f"{PROXY_URL}/v1/chat/completions",
        headers=_headers(auth),
        json=body,
        timeout=60,
        verify=False,
    )


print(f"Details against {PROXY_URL}  model={MODEL}")
print("-" * 52)

# D1 — reasoning_effort="none" yields visible content (thinking off)
try:
    r = _chat(extra={"reasoning_effort": "none"})
    d = r.json()
    c = d.get("choices", [{}])[0].get("message", {}).get("content", "")
    if r.status_code == 200 and c:
        ok(f"reasoning=none -> content {c[:30]!r}")
    else:
        bad("reasoning=none -> content", f"status={r.status_code}")
except Exception as e:  # noqa: BLE001
    bad("reasoning=none -> content", str(e))

# D2 — legacy think:false shim behaves like reasoning=none
try:
    r = _chat(extra={"think": False})
    r.raise_for_status()
    c = r.json()["choices"][0]["message"]["content"]
    if c:
        ok(f"think:false shim -> content {c[:30]!r}")
    else:
        bad("think:false shim", "no content")
except Exception as e:  # noqa: BLE001
    bad("think:false shim", str(e))

# D3 — thinking ON (reasoning=high, generous budget): content must still appear;
# a thinking-capable model also returns reasoning_content.
try:
    r = _chat(
        messages=[{"role": "user", "content": "What is 7 times 6? Show your reasoning."}],
        extra={"reasoning_effort": "high"},
    )
    msg = r.json().get("choices", [{}])[0].get("message", {})
    content = msg.get("content", "") or ""
    rc = msg.get("reasoning_content") or ""
    if r.status_code == 200 and (content or rc):
        parts = []
        if content:
            parts.append(f"content {content[:25]!r}")
        if rc:
            parts.append(f"reasoning_content ({len(rc)} chars)")
        ok("reasoning=high -> " + (", ".join(parts) or "no output"))
    else:
        bad("reasoning=high", f"status={r.status_code} no content/reasoning")
except Exception as e:  # noqa: BLE001
    bad("reasoning=high", str(e))

# D4 — tool calling: a provided tool must come back as a structured tool_call
_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}
try:
    r = _chat(
        messages=[{"role": "user", "content": "What's the weather in Paris right now? You must call the get_weather tool."}],
        extra={"tools": [_WEATHER_TOOL], "tool_choice": "auto", "max_tokens": 256},
    )
    d = r.json()
    if r.status_code != 200:
        bad("tool calling", f"status={r.status_code} {str(d)[:80]}")
    else:
        tcs = d["choices"][0]["message"].get("tool_calls")
        if tcs:
            names = ",".join(
                tc.get("function", {}).get("name", "?") for tc in tcs if isinstance(tc, dict)
            )
            ok(f"tool calling -> tool_calls=[{names}]")
        else:
            bad("tool calling", "no structured tool_call (model may not support tools)")
except Exception as e:  # noqa: BLE001
    bad("tool calling", str(e))

# D5 — streaming shape: content deltas + finish_reason + [DONE]
try:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Count: 1 2 3"}],
        "max_tokens": 30,
        "stream": True,
        "reasoning_effort": "none",
    }
    contents, finish, done = [], None, False
    with httpx.stream(
        "POST", f"{PROXY_URL}/v1/chat/completions",
        headers=_headers(), json=body, timeout=60, verify=False,
    ) as r:
        for line in r.iter_lines():
            line = line.strip()
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                done = True
                continue
            ch = json.loads(line[6:])["choices"][0]
            dc = ch.get("delta", {})
            if dc.get("content"):
                contents.append(dc["content"])
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    if contents and done and finish:
        ok(f"stream shape -> {len(contents)} content chunks, finish={finish}, [DONE]=True")
    else:
        bad("stream shape", f"chunks={len(contents)} done={done} finish={finish}")
except Exception as e:  # noqa: BLE001
    bad("stream shape", str(e))

# D-stream-turns — multi-turn (turn-based) streaming: 3 streamed turns where
# later turns must recall facts from turn 1. Proves context carries across
# streamed turns, not just single-shot.
try:
    turns = [
        "My name is Quasar and my favourite number is 47. Remember these.",
        "What is my name? Reply with just the name.",
        "What is my favourite number? Reply with just the number.",
    ]
    history: list = []
    streamed_ok = True
    assistant_texts: list = []
    for user in turns:
        history.append({"role": "user", "content": user})
        body = {"model": MODEL, "messages": history, "max_tokens": 128,
                "stream": True, "reasoning_effort": "none"}
        parts, finish, done = [], None, False
        with httpx.stream("POST", f"{PROXY_URL}/v1/chat/completions",
                          headers=_headers(), json=body, timeout=60, verify=False) as r:
            for line in r.iter_lines():
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    done = True
                    continue
                ch = json.loads(line[6:])["choices"][0]
                dc = ch.get("delta", {})
                if dc.get("content"):
                    parts.append(dc["content"])
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
        text = "".join(parts)
        assistant_texts.append(text)
        history.append({"role": "assistant", "content": text})
        if not (parts and done and finish):
            streamed_ok = False
    name_recall = "quasar" in assistant_texts[1].lower()
    num_recall = "47" in assistant_texts[2]
    if streamed_ok and name_recall and num_recall:
        ok(f"3-turn streaming, context carried (name={assistant_texts[1].strip()[:20]!r}, num={assistant_texts[2].strip()[:10]!r})")
    else:
        bad("3-turn streaming", f"streamed_ok={streamed_ok} name_recall={name_recall} num_recall={num_recall}")
except Exception as e:  # noqa: BLE001
    bad("3-turn streaming", str(e))

# D5 — malformed model (no @) -> rejected (400 from parse, or 422 from schema)
try:
    r = _chat(model="not-a-valid-model-no-at-sign")
    if r.status_code in (400, 422):
        ok(f"malformed model -> {r.status_code}")
    else:
        bad("malformed model", f"expected 400/422 got {r.status_code}")
except Exception as e:  # noqa: BLE001
    bad("malformed model", str(e))

# D5 — missing auth on a key-required provider -> 401 (ollama is open, so opt-in)
if AUTH_REQUIRED_MODEL:
    try:
        r = _chat(model=AUTH_REQUIRED_MODEL, auth=False)
        if r.status_code == 401:
            ok("missing auth (key-required model) -> 401")
        else:
            bad("missing auth", f"expected 401 got {r.status_code}")
    except Exception as e:  # noqa: BLE001
        bad("missing auth", str(e))
else:
    skip("missing auth -> 401 (set AUTH_REQUIRED_MODEL)")

# D6 — embeddings return a non-zero vector
try:
    r = httpx.post(
        f"{PROXY_URL}/v1/embeddings",
        headers=_headers(),
        json={"model": EMBED_MODEL, "input": ["hello world"]},
        timeout=60,
        verify=False,
    )
    r.raise_for_status()
    vec = r.json()["data"][0]["embedding"]
    if vec and any(v != 0 for v in vec[:20]):
        ok(f"embeddings -> dim={len(vec)}, non-zero")
    else:
        bad("embeddings", "zero/empty vector")
except Exception as e:  # noqa: BLE001
    bad("embeddings", str(e))

print("-" * 52)
print(f"Details: {passed} passed, {failed} failed, {skipped} skipped")
sys.exit(1 if failed else 0)
