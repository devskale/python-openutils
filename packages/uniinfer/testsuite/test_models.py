"""LLM model validation for ANY provider, through the proxy.

Provider-agnostic — the whole point of uniinfer. For each selected model it
checks, in order:
  1. alive    — coherent response to a trivial prompt
  2. tool_use — calls a supplied get_weather tool
  3. thinking — emits reasoning_content when reasoning_effort=high
                (auto-skipped when the model doesn't reason — not a failure)

Model selection (exactly one of):
  --models provider@model,...   explicit list
  --provider p1,p2,...          auto-discover chat models (via /v1/models)

Config: PROXY_URL / PROXY_AUTH (or PROXYHOST/PROXY_PORT/PROXY_KEY via .env).

Usage:
  uv run python testsuite/test_models.py --provider tu
  uv run python testsuite/test_models.py --models tu@qwen-3.6-35b
  uv run python testsuite/test_models.py --provider tu --skip-tool-use
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field

from _proxy_common import auth_header, LIVENESS_TIMEOUT, make_client, mark, proxy_auth, proxy_base_url

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string", "description": "City name"}},
            "required": ["location"],
        },
    },
}


@dataclass
class TestResult:
    model: str
    test: str
    status: str  # pass | fail | skip | error
    detail: str
    latency_s: float = 0.0


@dataclass
class ModelReport:
    model: str
    results: list[TestResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == "pass")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == "fail")

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == "skip")


def chat(base: str, model: str, messages: list[dict], timeout: float = 180, **extra) -> tuple[dict, int]:
    """One non-stream chat completion. reasoning_effort defaults to 'none'."""
    body = {"model": model, "messages": messages, "reasoning_effort": "none"}
    body.update(extra)
    headers = {"Content-Type": "application/json", **auth_header()}
    with make_client(timeout=timeout) as c:
        r = c.post(f"{base}/v1/chat/completions", headers=headers, json=body)
        try:
            return r.json(), r.status_code
        except Exception:  # noqa: BLE001
            return {}, r.status_code


def discover_models(provider: str, base: str) -> list[str]:
    """Chat models for one provider, via the unauthenticated /v1/models list."""
    with make_client() as c:
        r = c.get(f"{base}/v1/models")
        r.raise_for_status()
        out = []
        for m in r.json().get("data", []):
            if m.get("provider") != provider:
                continue
            if m.get("type") not in (None, "chat"):
                continue
            mid = m.get("id")
            if mid:
                out.append(f"{provider}@{mid}")
        return out


def check_alive(base: str, model: str) -> TestResult:
    t0 = time.monotonic()
    try:
        d, code = chat(base, model, [{"role": "user", "content": "Reply with exactly: I am alive."}], max_tokens=64, timeout=LIVENESS_TIMEOUT)
        lat = time.monotonic() - t0
        if code != 200:
            return TestResult(model, "alive", "fail", f"HTTP {code}", lat)
        content = (d.get("choices", [{}])[0].get("message", {}).get("content") or "").strip().lower()
        ok = "alive" in content and len(content) > 0
        return TestResult(model, "alive", "pass" if ok else "fail",
                          f"content={content[:120]!r} latency={lat:.1f}s", lat)
    except Exception as e:  # noqa: BLE001
        return TestResult(model, "alive", "error", f"{type(e).__name__}: {e}", time.monotonic() - t0)


def check_tool_use(base: str, model: str) -> TestResult:
    t0 = time.monotonic()
    try:
        d, code = chat(base, model,
                       [{"role": "user", "content": "What is the weather in Vienna?"}],
                       tools=[WEATHER_TOOL], tool_choice="auto", max_tokens=256)
        lat = time.monotonic() - t0
        if code != 200:
            return TestResult(model, "tool_use", "fail", f"HTTP {code}", lat)
        tc = d.get("choices", [{}])[0].get("message", {}).get("tool_calls")
        if tc:
            fn = tc[0].get("function", {})
            return TestResult(model, "tool_use", "pass",
                              f"tool={fn.get('name')}({fn.get('arguments', '')}) latency={lat:.1f}s", lat)
        return TestResult(model, "tool_use", "fail", "no tool_calls", lat)
    except Exception as e:  # noqa: BLE001
        return TestResult(model, "tool_use", "error", f"{type(e).__name__}: {e}", time.monotonic() - t0)


def check_thinking(base: str, model: str) -> TestResult:
    t0 = time.monotonic()
    try:
        d, code = chat(base, model,
                       [{"role": "user", "content": "What color is the sky on a clear day? Think step by step."}],
                       max_tokens=2048, reasoning_effort="high")
        lat = time.monotonic() - t0
        if code != 200:
            return TestResult(model, "thinking", "fail", f"HTTP {code}", lat)
        msg = d.get("choices", [{}])[0].get("message", {})
        rc = msg.get("reasoning_content") or ""
        if rc and len(rc.strip()) > 10:
            return TestResult(model, "thinking", "pass", f"reasoning_content={len(rc)} chars latency={lat:.1f}s", lat)
        # No reasoning emitted: the model simply doesn't think. Not a failure.
        return TestResult(model, "thinking", "skip",
                          "no reasoning_content (model does not think) — not a failure", lat)
    except Exception as e:  # noqa: BLE001
        return TestResult(model, "thinking", "error", f"{type(e).__name__}: {e}", time.monotonic() - t0)


def run_model(base: str, model: str, skip_tool_use: bool) -> ModelReport:
    print(f"\n{'=' * 60}\n  {model}\n{'=' * 60}")
    report = ModelReport(model=model)
    checks = [("alive", check_alive), ("tool_use", check_tool_use), ("thinking", check_thinking)]
    alive: TestResult | None = None
    for label, fn in checks:
        if label == "tool_use" and skip_tool_use:
            print(f"  {mark('skip')} tool_use: skipped")
            report.results.append(TestResult(model, "tool_use", "skip", "skipped by --skip-tool-use"))
            continue
        # If the alive check failed, the model is down — skip the rest rather
        # than waiting again on each subsequent check.
        if label != "alive" and alive is not None and alive.status in ("fail", "error"):
            print(f"  {mark('skip')} {label}: model down (alive failed) — skipped")
            report.results.append(
                TestResult(model, label, "skip", "model down (alive failed) — skipped"))
            continue
        r = fn(base, model)
        print(f"  {mark(r.status)} {r.test:9} {r.detail}")
        report.results.append(r)
        if label == "alive":
            alive = r
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM model validation for any provider, via the proxy")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--models", help="comma-separated provider@model list")
    src.add_argument("--provider", help="comma-separated provider list (auto-discovers chat models)")
    ap.add_argument("--skip-tool-use", action="store_true", help="skip the tool_use check")
    args = ap.parse_args()

    base = proxy_base_url()
    auth = proxy_auth()
    print(f"Proxy: {base}  (auth {'set' if auth else 'none'})\n")

    models: list[str] = []
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        for p in [p.strip() for p in args.provider.split(",") if p.strip()]:
            found = discover_models(p, base)
            print(f"provider {p}: {len(found)} chat model(s) discovered")
            models.extend(found)
    if not models:
        sys.exit("No models selected.")

    reports = [run_model(base, m, args.skip_tool_use) for m in models]

    print(f"\n{'=' * 64}\n  SUMMARY\n{'=' * 64}")
    total_pass = total_fail = total_skip = 0
    for rep in reports:
        total_pass += rep.passed
        total_fail += rep.failed
        total_skip += rep.skipped
        status = "✅" if rep.failed == 0 else "❌"
        print(f"  {status} {rep.model:36s}  {rep.passed} pass  {rep.failed} fail  {rep.skipped} skip")
        for r in rep.results:
            print(f"      {mark(r.status)} {r.test:9} {r.detail[:80]}")
    print(f"\n  Total: {total_pass} passed, {total_fail} failed, {total_skip} skipped "
          f"({total_pass + total_fail + total_skip} checks)")
    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
