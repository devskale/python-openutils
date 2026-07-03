"""Large-document task probe: throw a big coherent doc at the model and ask it
to summarize / extract. This stresses the reasoning path far more than
homogeneous padding, and mirrors the real-world situation where GLM-5.2-preview
reportedly degenerates into '0,0.0.0...' loops.

Generates a synthetic but structured document (sections, headings, numbers,
lists) so the model has genuine material to compress.
"""
import random
import time

import httpx

BASE = "http://localhost:8124/v1"
KEY = ""


def approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def build_doc(target_words: int) -> str:
    rng = random.Random(42)
    topics = [
        "infrastructure provisioning", "budget reconciliation", "incident response",
        "data retention policy", "capacity planning", "access control review",
        "disaster recovery drills", "vendor lifecycle", "compliance auditing",
        "feature rollout strategy", "telemetry pipeline", "on-call rotation",
        "release engineering", "secret rotation", "network segmentation",
    ]
    findings = [
        "no anomalies detected", "latency p99 elevated by 12%", "two stale alerts",
        "disk usage at 67%", "a misconfigured ingress rule", "three stale branches",
        "an expired certificate in staging", "cache hit ratio at 0.41",
        "queue depth growing slowly", "a flaky integration test",
    ]
    actions = [
        "re-ran the suite", "patched the rule", "opened a ticket", "escalated to on-call",
        "rolled back the change", "added a guardrail", "notified the owner",
        "scheduled a review", "increased the threshold", "documented the workaround",
    ]
    words = 0
    out = ["# Operations Report — Q3\n", "## Overview\n"]
    section = 1
    while words < target_words:
        out.append(f"\n## {section}. {rng.choice(topics).title()}\n")
        for _ in range(rng.randint(3, 6)):
            para = (
                f"On {rng.choice(['Monday','Tuesday','Wednesday'])} the team reviewed "
                f"{rng.choice(topics)}. Findings: {rng.choice(findings)}. "
                f"Next steps: we {rng.choice(actions)} and {rng.choice(actions)}. "
                f"Metric M{rng.randint(1,9)} settled at {rng.randint(10,990)} units, "
                f"which is {'within' if rng.random()>0.5 else 'above'} tolerance. "
                f"Stakeholders were informed. Tea was served. Budget code {rng.randint(1000,9999)}."
            )
            out.append(para + "\n")
            words += len(para.split())
        out.append("- item a\n- item b\n- item c\n")
        words += 6
        section += 1
    return "".join(out)


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


def stream_call(model: str, doc_words: int, max_tokens: int = 2000):
    doc = build_doc(doc_words)
    prompt = (doc + "\n\n## Task\nSummarize the key findings and the most important "
                    "action items from the report above in a concise bullet list.")
    actual = approx_tokens(prompt)
    parts, rparts = [], []
    t0 = time.monotonic()
    finish = None
    try:
        with httpx.Client(timeout=400.0) as c:
            with c.stream("POST", f"{BASE}/chat/completions",
                          headers={"Authorization": f"Bearer {KEY}"},
                          json={
                              "model": model,
                              "messages": [{"role": "user", "content": prompt}],
                              "max_tokens": max_tokens,
                              "temperature": 0.3,
                              "stream": True,
                          }) as resp:
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
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
    except Exception as e:
        return actual, f"ERROR: {type(e).__name__}: {e}"

    dt = time.monotonic() - t0
    content = "".join(parts)
    reasoning = "".join(rparts)
    cdeg, rdeg = looks_degenerate(content), looks_degenerate(reasoning)
    flag = f"  <<< DEGENERATE(content={cdeg},reasoning={rdeg})" if (cdeg or rdeg) else ""
    return actual, (f"({dt:.1f}s,finish={finish}) content_len={len(content)} "
                    f"reasoning_len={len(reasoning)}\n      content_head={content[:140]!r}"
                    f"\n      reasoning_tail={(reasoning[-140:] if reasoning else '')!r}{flag}")


def main():
    print("=" * 84)
    print("LARGE-DOC SUMMARIZE PROBE (coherent doc + real compression work)")
    print("=" * 84)
    for words in [2_000, 8_000, 20_000]:
        a, o = stream_call("tu@glm-5.2-744b-preview", words)
        print(f"\nGLM  doc~{words} words (~{a} tok):")
        print(f"  {o}")
    print()
    for words in [2_000, 8_000, 20_000]:
        a, o = stream_call("tu@qwen-3.5-397b", words)
        print(f"\nQWEN doc~{words} words (~{a} tok):")
        print(f"  {o}")


if __name__ == "__main__":
    main()
