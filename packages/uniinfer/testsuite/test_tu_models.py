"""TU LLM model validation testsuite.

Tests each model for:
  1. Alive — simple question, expects coherent response
  2. Tool use — model must call a provided tool
  3. Thinking mode — model returns reasoning_content (where supported)

Usage:
    cd packages/uniinfer
    uv run python testsuite/test_tu_models.py
    uv run python testsuite/test_tu_models.py --model gemma-4-26b
    uv run python testsuite/test_tu_models.py --skip-tool-use
"""

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, field
from uniinfer.providers.tu import TUProvider
from uniinfer.core import ChatCompletionRequest, ChatMessage

LLM_MODELS = [
    "gemma-4-26b",
    "glm-4.7-355b",
    "mistral-small-3.2-24b",
    "qwen-3.5-397b",
    "qwen-coder-30b",
]

THINKING_MODELS = {"glm-4.7-355b", "qwen-3.5-397b"}

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
            },
            "required": ["location"],
        },
    },
}


@dataclass
class TestResult:
    model: str
    test: str
    passed: bool
    detail: str
    latency_s: float = 0.0


@dataclass
class ModelReport:
    model: str
    results: list[TestResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)


def fmt(msg: str, ok: bool | None = None) -> str:
    if ok is True:
        return f"  ✅ {msg}"
    if ok is False:
        return f"  ❌ {msg}"
    return f"  ⏳ {msg}"


async def test_alive(provider: TUProvider, model: str) -> TestResult:
    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Reply with exactly: I am alive.")],
        model=model,
        temperature=0.0,
        max_tokens=4096 if model in THINKING_MODELS else 64,
    )
    t0 = time.monotonic()
    try:
        resp = await provider.acomplete(req)
    except Exception as e:
        latency = time.monotonic() - t0
        return TestResult(model=model, test="alive", passed=False, detail=f"ERROR: {type(e).__name__}: {e} latency={latency:.1f}s", latency_s=latency)
    latency = time.monotonic() - t0
    content_raw = resp.message.content or ""
    content = content_raw.strip().lower()
    thinking_raw = resp.thinking or ""
    has_alive = "alive" in content or "alive" in thinking_raw.lower()
    ok = has_alive and len(content) > 0
    detail = f'content="{content_raw[:120]}" thinking={len(thinking_raw)}chars finish={resp.finish_reason} latency={latency:.1f}s'
    return TestResult(model=model, test="alive", passed=ok, detail=detail, latency_s=latency)


async def test_tool_use(provider: TUProvider, model: str) -> TestResult:
    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="What is the weather in Vienna?")],
        model=model,
        temperature=0.0,
        max_tokens=256,
        tools=[WEATHER_TOOL],
        tool_choice="auto",
    )
    t0 = time.monotonic()
    try:
        resp = await provider.acomplete(req)
    except Exception as e:
        latency = time.monotonic() - t0
        return TestResult(model=model, test="tool_use", passed=False, detail=f"ERROR: {type(e).__name__}: {e} latency={latency:.1f}s", latency_s=latency)
    latency = time.monotonic() - t0
    tc = resp.message.tool_calls
    ok = tc is not None and len(tc) > 0
    detail = f"tool_calls={tc} latency={latency:.1f}s" if not ok else f"tool={tc[0]['function']['name']}({tc[0]['function'].get('arguments','')}) latency={latency:.1f}s"
    return TestResult(model=model, test="tool_use", passed=ok, detail=detail, latency_s=latency)


async def test_thinking(provider: TUProvider, model: str) -> TestResult | None:
    if model not in THINKING_MODELS:
        return None
    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="What color is the sky on a clear day? Think step by step.")],
        model=model,
        temperature=0.0,
        max_tokens=4096,
    )
    t0 = time.monotonic()
    try:
        resp = await provider.acomplete(req)
    except Exception as e:
        latency = time.monotonic() - t0
        return TestResult(model=model, test="thinking", passed=False, detail=f"ERROR: {type(e).__name__}: {e} latency={latency:.1f}s", latency_s=latency)
    latency = time.monotonic() - t0
    thinking = resp.thinking
    ok = thinking is not None and len(str(thinking).strip()) > 10
    detail = f"thinking={'yes' if thinking else 'none'} len={len(str(thinking or ''))} latency={latency:.1f}s"
    return TestResult(model=model, test="thinking", passed=ok, detail=detail, latency_s=latency)


async def run_model(model: str, skip_tool_use: bool = False) -> ModelReport:
    print(f"\n{'='*60}")
    print(f"  {model}")
    print(f"{'='*60}")

    provider = TUProvider()
    report = ModelReport(model=model)

    # Alive
    print(fmt("alive"))
    r = await test_alive(provider, model)
    print(fmt(f"{r.test}: {r.detail}", r.passed))
    report.results.append(r)

    # Tool use
    if not skip_tool_use:
        print(fmt("tool_use"))
        r = await test_tool_use(provider, model)
        print(fmt(f"{r.test}: {r.detail}", r.passed))
        report.results.append(r)

    # Thinking
    print(fmt("thinking"))
    r = await test_thinking(provider, model)
    if r is None:
        print(fmt("thinking: skipped (not a thinking model)"))
    else:
        print(fmt(f"{r.test}: {r.detail}", r.passed))
        report.results.append(r)

    return report


async def main(models: list[str], skip_tool_use: bool = False):
    print("TU Model Validation Testsuite")
    print(f"Models: {', '.join(models)}")
    print(f"Tool use: {'skipped' if skip_tool_use else 'enabled'}")
    print(f"Thinking models: {', '.join(sorted(THINKING_MODELS))}")

    reports: list[ModelReport] = []
    for model in models:
        report = await run_model(model, skip_tool_use=skip_tool_use)
        reports.append(report)

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    total_pass = 0
    total_fail = 0
    for report in reports:
        status = "✅" if report.failed == 0 else "❌"
        print(f"  {status} {report.model:30s}  {report.passed} passed  {report.failed} failed")
        for r in report.results:
            tag = "✅" if r.passed else "❌"
            print(f"      {tag} {r.test:15s} {r.detail[:80]}")
        total_pass += report.passed
        total_fail += report.failed

    print(f"\n  Total: {total_pass} passed, {total_fail} failed out of {total_pass + total_fail}")
    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TU model validation testsuite")
    parser.add_argument("--model", "-m", action="append", dest="models", help="Model to test (repeatable, default: all LLMs)")
    parser.add_argument("--skip-tool-use", action="store_true", help="Skip tool use tests")
    args = parser.parse_args()
    models = args.models or LLM_MODELS
    asyncio.run(main(models, skip_tool_use=args.skip_tool_use))
