"""Quick speed & performance test for TU models with varying context lengths."""

import time
import asyncio
import sys
from uniinfer.core import ChatCompletionRequest, ChatMessage
from uniinfer.providers.tu import TUProvider

MODELS = ["qwen-3.6-35b", "gemma-4-e2b-it"]
CONTEXT_TOKENS = [100, 500, 1000, 2000]
GENERATION_TOKENS = 4096
RUNS = 2

def build_prompt(num_chars: int) -> str:
    word = "The quick brown fox jumps over the lazy dog. "
    repeated = (word * ((num_chars // len(word)) + 1))[:num_chars]
    return f"Context:\n{repeated}\n\nReply with a single sentence summarizing what this text is about."

def count_tokens_approx(text: str) -> int:
    return len(text.split())

async def test_streaming(provider: TUProvider, model: str, prompt: str) -> dict:
    request = ChatCompletionRequest(
        model=model,
        messages=[ChatMessage(role="user", content=prompt)],
        streaming=True,
        max_tokens=GENERATION_TOKENS,
    )
    
    start = time.perf_counter()
    first_token_time = None
    first_thinking_time = None
    total_content = ""
    total_thinking = ""
    content_tokens = 0
    thinking_tokens = 0
    finish_reason = None
    
    try:
        async for chunk in provider.astream_complete(request):
            now = time.perf_counter()
            if chunk.message.content:
                if first_token_time is None:
                    first_token_time = now
                total_content += chunk.message.content
                content_tokens += 1
            if chunk.thinking:
                if first_thinking_time is None:
                    first_thinking_time = now
                total_thinking += chunk.thinking
                thinking_tokens += 1
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
    except Exception as e:
        return {"error": str(e), "model": model}
    
    end = time.perf_counter()
    total_time = end - start
    # Use whichever came first (content or thinking) for TTFT
    ttft_source = first_token_time or first_thinking_time
    ttft = ttft_source - start if ttft_source else None
    gen_time = (end - ttft_source) if ttft_source else None
    total_tokens = content_tokens + thinking_tokens
    tps = total_tokens / gen_time if gen_time and gen_time > 0 else 0
    
    return {
        "model": model,
        "total_time_s": round(total_time, 2),
        "ttft_s": round(ttft, 3) if ttft else None,
        "gen_time_s": round(gen_time, 2) if gen_time else None,
        "content_tokens": content_tokens,
        "thinking_tokens": thinking_tokens,
        "total_tokens": total_tokens,
        "tps": round(tps, 1),
        "finish_reason": finish_reason,
        "output_preview": total_content[:100].replace("\n", " "),
        "thinking_preview": total_thinking[:80].replace("\n", " ") if total_thinking else None,
    }

async def run_tests():
    provider = TUProvider()
    
    print(f"{'Model':<18} {'Ctx~chars':<10} {'Total(s)':<10} {'TTFT(s)':<10} {'Gen(s)':<10} {'Content':<9} {'Think':<7} {'TPS':<8} {'Finish':<10} Run", flush=True)
    print("-" * 115, flush=True)
    
    for model in MODELS:
        for ctx_chars in CONTEXT_TOKENS:
            prompt = build_prompt(ctx_chars)
            ctx_words = count_tokens_approx(prompt)
            
            for run in range(1, RUNS + 1):
                result = await test_streaming(provider, model, prompt)
                
                if "error" in result:
                    print(f"{model:<18} {ctx_chars:<10} {'ERROR: ' + result['error'][:60]:<80} R{run}", flush=True)
                    continue
                
                print(
                    f"{result['model']:<18} "
                    f"~{ctx_words}w/{ctx_chars}c  "
                    f"{result['total_time_s']:<10} "
                    f"{result['ttft_s'] or '-':<10} "
                    f"{result['gen_time_s'] or '-':<10} "
                    f"{result['content_tokens']:<9} "
                    f"{result['thinking_tokens']:<7} "
                    f"{result['tps']:<8} "
                    f"{(result['finish_reason'] or '-'):<10} "
                    f"R{run}",
                    flush=True
                )
                if result.get('output_preview'):
                    print(f"  -> {result['output_preview']}", flush=True)
                if result.get('thinking_preview'):
                    print(f"  [think] {result['thinking_preview']}", flush=True)
                
                # Small delay between runs
                await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(run_tests())
