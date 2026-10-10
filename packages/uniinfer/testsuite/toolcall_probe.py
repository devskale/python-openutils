"""toolcall_probe — repeated tool-call reliability probe for ANY provider/model.

Single-shot checks (testsuite/test_models.py) miss intermittent defects: the
qwen-vllm issue (docs/bug-qwen-vllm-toolcall.md) showed a 1-in-3 failure rate
that one probe call cannot see. This script runs N identical rounds and
classifies every failure mode we have observed in the wild:

  emit facet (round 1 — "call the tool"):
    prose_call     tool call leaked into content as text, no structured tool_calls
    empty_length   finish_reason=length, no content, no tool_calls (abort before the call)

  roundtrip facet (round 2 — result fed back, "summarise, do NOT call again"):
    re_call        model ignores the result and invokes the same tool again
    ignored_result no re-call, but the sentinel value from the result is absent
    hallucinated   no re-call, sentinel absent, but a *different* concrete value
                   appears in the summary (invented result content)

Sentinel trick: the tool result carries a distinctive, unmistakable value
("31.7 degrees, purple wind"). A summary that does not contain it did not read
the result; one that contains a different number invented it.

Modes:
  proxy   provider@model via the uniinfer proxy (PROXY_URL / PROXY_AUTH)
  direct  provider via ProviderFactory (no proxy; needs the provider's key)

Usage:
  uv run python testsuite/toolcall_probe.py --models tu@qwen-3.6-35b-vllm --runs 10
  uv run python testsuite/toolcall_probe.py --models kilo@stepfun/step-5-preview-free,opencode@step-5-preview-free
  uv run python testsuite/toolcall_probe.py --direct tu --models qwen-3.6-35b-vllm
  uv run python testsuite/toolcall_probe.py --models tu@qwen-3.6-35b-vllm --stream

Exit code: 0 = clean · 1 = intermittent defects · 2 = broken/down.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field

import httpx
from _proxy_common import auth_header, proxy_base_url

CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Calculate a math expression",
        "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]},
    },
}
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
    },
}
# Distinctive sentinel: a summary without "31.7" did not read the result; a
# summary naming a different concrete value invented one.
SENTINEL_RESULT = "31.7 degrees, purple wind"
SENTINEL_TOKEN = "31.7"
PROSE_CALL_RE = re.compile(r"\b\w+\s*\([^)]*\)")
# vLLM-style leak: the call arrives as a JSON object in content instead of a
# structured tool_calls entry, e.g. {"call": {"function": "calculator", ...}}
JSON_CALL_RE = re.compile(r"\{\s*\"(?:call|function|tool_call|tool_calls)\"\s*:")
OTHER_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

EMIT_PROMPT = "What is 2+2? Use the calculator tool."
ROUNDTRIP_PROMPT = "Summarise the weather result in one short sentence. Use the result you already got — do NOT call any tool again."


@dataclass
class RunResult:
    model: str
    run: int
    emit: str                       # ok | prose_call | empty_length | error
    emit_detail: str = ""
    roundtrip: str = "skipped"      # ok | re_call | ignored_result | hallucinated | error | skipped
    roundtrip_detail: str = ""
    latency_s: float = 0.0
    # Full raw content of a defect run — only populated when --dump is active,
 # so a leak parser can be designed from real samples instead of guesses.
    emit_content: str | None = None
    roundtrip_content: str | None = None


@dataclass
class ModelReport:
    model: str
    runs: list[RunResult] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.runs:
            for facet in (r.emit, r.roundtrip):
                if facet == "skipped":
                    continue
                out[facet] = out.get(facet, 0) + 1
        return out

    def defect_rate(self) -> float:
        total = sum(1 for r in self.runs for f in (r.emit, r.roundtrip) if f != "skipped")
        bad = sum(1 for r in self.runs for f in (r.emit, r.roundtrip)
                  if f not in ("ok", "skipped"))
        return (bad / total) if total else 0.0

    def verdict(self) -> str:
        c = self.counts()
        if not self.runs or all(r.emit == "error" for r in self.runs):
            return "DOWN"
        if c.get("prose_call") or c.get("empty_length"):
            return "BROKEN (tool calls not reliably emitted)"
        if c.get("re_call") or c.get("ignored_result") or c.get("hallucinated"):
            return "UNRELIABLE (tool results not reliably consumed)"
        return "CLEAN"


def _classify_emit(finish_reason: str | None, content: str, tool_calls: list) -> tuple[str, str]:
    if tool_calls:
        return "ok", f"tool={tool_calls[0].get('function', {}).get('name')}"
    if finish_reason == "length" and not content:
        return "empty_length", "finish=length, no content, no tool_calls (abort before the call)"
    if content and PROSE_CALL_RE.search(content):
        return "prose_call", f"call leaked into content: {content[:60]!r}"
    if content and JSON_CALL_RE.search(content):
        return "prose_call", f"call leaked as JSON text: {content[:60]!r}"
    return "ok", f"plain text answer (no call): {content[:60]!r}"


def _is_defect(*facets: str) -> bool:
    return any(f not in ("ok", "skipped") for f in facets)


def _classify_roundtrip(content: str, tool_calls: list) -> tuple[str, str]:
    if tool_calls:
        names = [tc.get("function", {}).get("name") for tc in tool_calls]
        return "re_call", f"re-invoked {names} despite the result"
    text = content or ""
    if SENTINEL_TOKEN in text:
        return "ok", "summary uses the result"
    others = [n for n in OTHER_NUMBER_RE.findall(text) if n != "31" and n != "7"]
    if others:
        return "hallucinated", f"sentinel absent, invented value(s) {others[:3]}: {text[:60]!r}"
    return "ignored_result", f"sentinel absent: {text[:60]!r}"


async def _proxy_round(client, base: str, model: str, messages: list, tools: list,
                       stream: bool, timeout: float) -> tuple[str | None, str, list]:
    """One completion via the proxy. Returns (finish_reason, content, tool_calls)."""
    body: dict = {"model": model, "messages": messages, "tools": tools,
                  "tool_choice": "auto", "max_tokens": 1024}
    if stream:
        body["stream"] = True
        content_parts: list[str] = []
        tool_calls: list[dict] = []
        finish = None
        async with client.stream("POST", f"{base}/chat/completions", json=body,
                                 headers={**auth_header(), "Content-Type": "application/json"},
                                 timeout=timeout) as r:
            if r.status_code != 200:
                err = (await r.aread()).decode(errors="replace")[:200]
                raise RuntimeError(f"HTTP {r.status_code}: {err}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict) and data.get("error"):
                    raise RuntimeError(f"stream error chunk: {data['error']}")
                choice = (data.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content_parts.append(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", len(tool_calls))
                    while len(tool_calls) <= idx:
                        tool_calls.append({"id": None, "type": "function",
                                           "function": {"name": "", "arguments": ""}})
                    agg = tool_calls[idx]
                    if tc.get("id"):
                        agg["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        agg["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        agg["function"]["arguments"] += fn["arguments"]
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
        return finish, "".join(content_parts), [t for t in tool_calls if t["function"]["name"]]

    r = await client.post(f"{base}/chat/completions", json=body,
                          headers={**auth_header(), "Content-Type": "application/json"},
                          timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    return choice.get("finish_reason"), msg.get("content") or "", msg.get("tool_calls") or []


async def _direct_round(provider, messages: list, tools: list) -> tuple[str | None, str, list]:
    """One completion via ProviderFactory (no proxy)."""
    from uniinfer.core import ChatCompletionRequest, ChatMessage

    req = ChatCompletionRequest(
        model=provider_model_id,
        messages=[ChatMessage(role=m["role"], content=m.get("content"),
                              tool_calls=m.get("tool_calls"), tool_call_id=m.get("tool_call_id"))
                  for m in messages],
        tools=tools, tool_choice="auto", max_tokens=1024,
    )
    resp = await provider.acomplete(req)
    return resp.finish_reason, resp.message.content or "", resp.message.tool_calls or []


async def probe_model(model: str, runs: int, stream: bool, timeout: float,
                      base: str | None, provider=None,
                      dump_path: str | None = None) -> ModelReport:
    report = ModelReport(model=model)
    for i in range(1, runs + 1):
        t0 = time.monotonic()
        rr = RunResult(model=model, run=i, emit="error")
        try:
            if base:
                async with httpx.AsyncClient(verify=False, timeout=timeout) as client:
                    finish, content, tcs = await _proxy_round(
                        client, base, model,
                        [{"role": "user", "content": EMIT_PROMPT}], [CALC_TOOL], stream, timeout)
            else:
                finish, content, tcs = await _direct_round(
                    provider, [{"role": "user", "content": EMIT_PROMPT}], [CALC_TOOL])
            rr.emit, rr.emit_detail = _classify_emit(finish, content, tcs)

            if rr.emit == "ok" and tcs:
                # Round 2: feed a sentinel result back and demand a summary.
                call = tcs[0]
                messages = [
                    {"role": "user", "content": EMIT_PROMPT},
                    {"role": "assistant", "content": None, "tool_calls": tcs},
                    {"role": "tool", "tool_call_id": call.get("id"), "content": SENTINEL_RESULT},
                    {"role": "user", "content": ROUNDTRIP_PROMPT},
                ]
                if base:
                    async with httpx.AsyncClient(verify=False, timeout=timeout) as client:
                        f2, c2, tcs2 = await _proxy_round(
                            client, base, model, messages, [CALC_TOOL], stream, timeout)
                else:
                    f2, c2, tcs2 = await _direct_round(provider, messages, [CALC_TOOL])
                rr.roundtrip, rr.roundtrip_detail = _classify_roundtrip(c2, tcs2)
        except Exception as e:  # noqa: BLE001
            rr.emit = "error"
            rr.emit_detail = f"{type(e).__name__}: {str(e)[:120]}"
            rr.roundtrip = "error"
        rr.latency_s = round(time.monotonic() - t0, 1)
        report.runs.append(rr)
        flag = "✅" if (rr.emit == "ok" and rr.roundtrip in ("ok", "skipped")) else "❌"
        print(f"  {flag} run{i:<2} emit={rr.emit:<12} roundtrip={rr.roundtrip:<15} "
              f"{rr.emit_detail[:60]}" +
              (f" | {rr.roundtrip_detail[:60]}" if rr.roundtrip != "skipped" else ""), flush=True)
        if dump_path and _is_defect(rr.emit, rr.roundtrip):
            with open(dump_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "model": model, "run": i, "emit": rr.emit,
                    "emit_content": content if rr.emit != "ok" else None,
                    "roundtrip": rr.roundtrip,
                    "roundtrip_content": c2 if rr.roundtrip not in ("ok", "skipped") else None,
                }, ensure_ascii=False) + "\n")
    return report


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True,
                    help="comma-separated provider@model (proxy) or bare model (direct)")
    ap.add_argument("--direct", metavar="PROVIDER",
                    help="probe this provider directly via ProviderFactory (no proxy)")
    ap.add_argument("--runs", type=int, default=10, help="rounds per model (default 10)")
    ap.add_argument("--stream", action="store_true",
                    help="probe via SSE streaming (default: non-streaming, matches the issue repro)")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--dump", metavar="FILE",
                    help="append full content of every defect run as JSON lines "
                         "(leak samples for parser design)")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    base = None if args.direct else proxy_base_url()
    provider = None
    global provider_model_id
    if args.direct:
        from uniinfer import ProviderFactory
        provider = ProviderFactory.get_provider(args.direct)
        provider_model_id = None  # set per model below

    reports: list[ModelReport] = []
    for model in models:
        if args.direct:
            provider_model_id = model
            label = f"{args.direct}@{model}"
        else:
            label = model
        print(f"\n{'=' * 70}\n  {label}  ({args.runs} runs, "
              f"{'stream' if args.stream else 'non-stream'}"
              f"{', direct' if args.direct else ', via proxy'})\n{'=' * 70}", flush=True)
        reports.append(await probe_model(label, args.runs, args.stream, args.timeout,
                                          base, provider, args.dump))

    worst = 0
    print(f"\n{'=' * 70}\n  SUMMARY\n{'=' * 70}")
    for rep in reports:
        c = rep.counts()
        rate = rep.defect_rate()
        verdict = rep.verdict()
        worst = max(worst, 0 if verdict == "CLEAN" else 1 if verdict.startswith("UNRELIABLE") else 2)
        detail = " ".join(f"{k}={v}" for k, v in sorted(c.items()) if k != "ok")
        print(f"  {rep.model:44s} {verdict:42s} defects={rate:.0%} {detail}")

    if args.json:
        print(json.dumps([{
            "model": r.model,
            "verdict": r.verdict(),
            "defect_rate": round(r.defect_rate(), 3),
            "counts": r.counts(),
            "runs": [vars(x) for x in r.runs],
        } for r in reports], indent=1))
    return worst


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
