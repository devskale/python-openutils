"""Quick smoke test for TU chat models via uniioai-proxy."""
import asyncio
import httpx

BASE_URL = "http://localhost:8123/v1"
API_KEY = "test23@test34"

# Chat models to test (excluding TTS/STT/image models)
CHAT_MODELS = [
    "tu@glm-4.7-355b",
    "tu@glm-4.5v-106b",
    "tu@qwen-coder-30b",
    "tu@mistral-small-3.2-24b",
]

async def test_model(model: str) -> dict:
    """Test a single model with a simple prompt."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Say 'OK' and nothing else."}],
                    "max_tokens": 50
                }
            )
            
            if response.status_code != 200:
                return {"model": model, "status": "FAIL", "error": f"HTTP {response.status_code}: {response.text[:200]}"}
            
            data = response.json()
            choice = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content")
            reasoning = choice.get("message", {}).get("reasoning_content")
            finish_reason = choice.get("finish_reason")
            
            return {
                "model": model,
                "status": "PASS",
                "content": content[:100] if content else None,
                "has_reasoning": bool(reasoning),
                "finish_reason": finish_reason
            }
        except Exception as e:
            return {"model": model, "status": "FAIL", "error": str(e)[:200]}

async def main():
    print("="*60)
    print("SMOKE TEST: TU Chat Models via uniioai-proxy")
    print("="*60)
    print(f"Base URL: {BASE_URL}")
    print(f"Models to test: {len(CHAT_MODELS)}")
    print("-"*60)
    
    results = []
    for model in CHAT_MODELS:
        print(f"Testing {model}...", end=" ", flush=True)
        result = await test_model(model)
        results.append(result)
        
        if result["status"] == "PASS":
            content_preview = result.get("content", "") or "(reasoning only)"
            print(f"✅ PASS - {content_preview[:50]}")
            if result.get("has_reasoning"):
                print(f"         (has reasoning)")
        else:
            print(f"❌ FAIL - {result.get('error', 'unknown')}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    
    if failed > 0:
        print("\nFailed models:")
        for r in results:
            if r["status"] == "FAIL":
                print(f"  - {r['model']}: {r.get('error', 'unknown')}")

if __name__ == "__main__":
    asyncio.run(main())
