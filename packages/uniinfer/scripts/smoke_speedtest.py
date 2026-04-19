"""Smoke speed test for uniinfer chat models.

Measures per-model:
- Time to first token (TFT)
- Time to first thinking token (if applicable)
- Thinking token count
- Text token count
- Total tokens per second
- Total wall time

Usage:
    uv run python3 scripts/smoke_speedtest.py
    uv run python3 scripts/smoke_speedtest.py -p tu -m qwen-coder-30b
    uv run python3 scripts/smoke_speedtest.py --runs 3
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import json
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


QUESTIONS = [
    "Erkläre mir bitte Transformer in maschinellem Lernen in einfachen Worten und auf deutsch.",
    "Was ist der Unterschied zwischen Supervised und Unsupervised Learning? Erkläre kurz.",
    "Beschreibe kurz, wie ein neuronales Netzwerk funktioniert.",
    "Was ist Transfer Learning und warum ist es nützlich?",
    "Erkläre den Begriff Gradient Descent in einfachen Worten.",
    "Was ist der Unterschied zwischen CNN und RNN? Nenne je ein Anwendungsgebiet.",
    "Was sind Embeddings im Kontext von NLP? Erkläre kurz.",
    "Erkläre den Attention-Mechanismus in einfachen Worten.",
    "Was ist Fine-Tuning und wann verwendet man es?",
    "Was ist der Unterschied zwischen Batch und Stochastic Gradient Descent?",
]


@dataclass
class SpeedResult:
    provider: str
    model: str
    run: int
    prompt: str
    tft: float = 0.0
    tft_thinking: float | None = None
    thinking_chars: int = 0
    text_chars: int = 0
    thinking_tokens: int = 0
    text_tokens: int = 0
    total_tokens: int = 0
    tok_per_sec: float = 0.0
    wall_time: float = 0.0
    finish_reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        d = {
            "provider": self.provider,
            "model": self.model,
            "run": self.run,
            "tft": round(self.tft, 3),
            "tft_thinking": round(self.tft_thinking, 3) if self.tft_thinking is not None else None,
            "thinking_tokens": self.thinking_tokens,
            "text_tokens": self.text_tokens,
            "total_tokens": self.total_tokens,
            "tok_per_sec": round(self.tok_per_sec, 1),
            "wall_time": round(self.wall_time, 3),
            "finish_reason": self.finish_reason,
        }
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class ModelResult:
    provider: str
    model: str
    runs: list[SpeedResult] = field(default_factory=list)

    def avg(self, attr: str) -> float:
        vals = [getattr(r, attr) for r in self.runs if getattr(r, attr) and not r.error]
        return sum(vals) / len(vals) if vals else 0.0

    def avg_dict(self) -> dict:
        ok = [r for r in self.runs if not r.error]
        if not ok:
            return {"error": all(r.error for r in self.runs)}
        return {
            "provider": self.provider,
            "model": self.model,
            "runs": len(ok),
            "avg_tft": round(self.avg("tft"), 3),
            "avg_tft_thinking": round(self.avg("tft_thinking"), 3) if any(r.tft_thinking for r in ok) else None,
            "avg_thinking_tokens": round(self.avg("thinking_tokens")),
            "avg_text_tokens": round(self.avg("text_tokens")),
            "avg_total_tokens": round(self.avg("total_tokens")),
            "avg_tok_per_sec": round(self.avg("tok_per_sec"), 1),
            "avg_wall_time": round(self.avg("wall_time"), 3),
            "finish_reason": ok[-1].finish_reason,
        }


def run_single(provider_name: str, model: str, prompt: str, run_num: int, max_tokens: int | None = None) -> SpeedResult:
    from uniinfer import ChatMessage, ChatCompletionRequest
    from uniinfer.config.providers import PROVIDER_CONFIGS
    from credgoo import get_api_key

    result = SpeedResult(provider=provider_name, model=model, run=run_num, prompt=prompt)

    credgoo_service = provider_name
    try:
        api_key = get_api_key(service=credgoo_service)
    except Exception as e:
        result.error = f"no api key: {e}"
        return result

    extra = PROVIDER_CONFIGS.get(provider_name, {}).get("extra_params", {})
    kwargs = {k: v for k, v in extra.items() if k in ("base_url", "account_id")}
    kwargs["api_key"] = api_key

    try:
        from uniinfer.providers import ChatProvider
        cls = PROVIDER_CONFIGS.get(provider_name, {}).get("provider_class")
        if cls:
            provider = cls(**kwargs)
        else:
            from uniinfer import ProviderFactory
            provider = ProviderFactory().get_provider(name=provider_name, **kwargs)
    except Exception as e:
        result.error = f"provider init: {e}"
        return result

    messages = [ChatMessage(role="user", content=prompt)]
    request = ChatCompletionRequest(
        messages=messages,
        model=model,
        streaming=True,
        max_tokens=max_tokens,
    )

    start = time.time()
    first_token_time = None
    first_thinking_time = None
    text = ""
    thinking = ""
    last_usage = None

    try:
        for chunk in provider.stream_complete(request):
            if chunk.message.content:
                if first_token_time is None:
                    first_token_time = time.time()
                text += chunk.message.content
            if chunk.thinking:
                if first_thinking_time is None:
                    first_thinking_time = time.time()
                thinking += chunk.thinking
            if chunk.usage:
                last_usage = chunk.usage
            if chunk.finish_reason:
                result.finish_reason = chunk.finish_reason
    except Exception as e:
        if text or thinking:
            pass
        else:
            result.error = str(e)
            result.wall_time = time.time() - start
            return result

    end = time.time()

    result.tft = (first_token_time - start) if first_token_time else 0.0
    result.tft_thinking = (first_thinking_time - start) if first_thinking_time else None
    result.thinking_chars = len(thinking)
    result.text_chars = len(text)
    result.wall_time = end - start

    if last_usage:
        result.total_tokens = last_usage.get("total_tokens", 0) or 0
        result.text_tokens = (last_usage.get("completion_tokens", 0) or 0) - (last_usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0) or 0)
        result.thinking_tokens = last_usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0) or 0
        if result.total_tokens == 0:
            result.total_tokens = result.thinking_tokens + result.text_tokens
    else:
        chars_per_token = 4
        result.thinking_tokens = len(thinking) // chars_per_token
        result.text_tokens = len(text) // chars_per_token
        result.total_tokens = result.thinking_tokens + result.text_tokens

    gen_time = (end - first_token_time) if first_token_time else 0.0
    result.tok_per_sec = result.total_tokens / gen_time if gen_time > 0 else 0.0

    return result


def print_single(r: SpeedResult):
    if r.error:
        print(f"  run {r.run}: ERROR - {r.error}")
        return
    parts = [
        f"run {r.run}",
        f"tft={r.tft:.2f}s",
    ]
    if r.tft_thinking is not None:
        parts.append(f"tft_think={r.tft_thinking:.2f}s")
    if r.thinking_tokens:
        parts.append(f"think_tok={r.thinking_tokens}")
    parts.append(f"text_tok={r.text_tokens}")
    parts.append(f"total={r.total_tokens}")
    parts.append(f"tok/s={r.tok_per_sec:.1f}")
    parts.append(f"wall={r.wall_time:.2f}s")
    if r.finish_reason:
        parts.append(f"({r.finish_reason})")
    print(f"  {' | '.join(parts)}")


def print_summary(model_result: ModelResult):
    a = model_result.avg_dict()
    if "error" in a:
        print(f"  ALL RUNS FAILED")
        return
    print(f"  ── avg over {a['runs']} runs ──")
    parts = [
        f"tft={a['avg_tft']:.2f}s",
    ]
    if a["avg_tft_thinking"] is not None:
        parts.append(f"tft_think={a['avg_tft_thinking']:.2f}s")
    if a["avg_thinking_tokens"]:
        parts.append(f"think_tok={a['avg_thinking_tokens']:.0f}")
    parts.append(f"text_tok={a['avg_text_tokens']:.0f}")
    parts.append(f"total={a['avg_total_tokens']:.0f}")
    parts.append(f"tok/s={a['avg_tok_per_sec']:.1f}")
    parts.append(f"wall={a['avg_wall_time']:.2f}s")
    if a["finish_reason"]:
        parts.append(f"({a['finish_reason']})")
    print(f"  {' | '.join(parts)}")


def main():
    parser = argparse.ArgumentParser(description="Smoke speed test for uniinfer chat models")
    parser.add_argument("-p", "--provider", type=str, default=None,
                        help="Provider to test (default: tu)")
    parser.add_argument("-m", "--model", type=str, default=None,
                        help="Model to test (default: provider default)")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Multiple models to test")
    parser.add_argument("--runs", type=int, default=1, help="Runs per model (default: 1)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max tokens to generate (default: none)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for prompt selection")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    provider = args.provider or "tu"
    models = args.models or ([args.model] if args.model else None)

    if not models:
        from uniinfer.config.providers import PROVIDER_CONFIGS
        models = [PROVIDER_CONFIGS.get(provider, {}).get("default_model", "")]
        if not models[0]:
            print(f"Error: no model specified and no default for provider '{provider}'")
            sys.exit(1)

    prompt = random.choice(QUESTIONS)
    print(f"Provider: {provider}")
    print(f"Models:   {', '.join(models)}")
    print(f"Runs:     {args.runs}")
    print(f"Prompt:   {prompt}")
    print()

    all_results: list[ModelResult] = []
    json_results: list[dict] = []

    for model in models:
        print(f"── {provider}/{model} ──")
        mr = ModelResult(provider=provider, model=model)

        for run in range(1, args.runs + 1):
            if args.runs > 1:
                run_prompt = random.choice(QUESTIONS)
            else:
                run_prompt = prompt
            r = run_single(provider, model, run_prompt, run, args.max_tokens)
            mr.runs.append(r)
            print_single(r)

        print_summary(mr)
        print()
        all_results.append(mr)
        json_results.append(mr.avg_dict())

    if args.json:
        print(json.dumps(json_results, indent=2))


if __name__ == "__main__":
    main()
