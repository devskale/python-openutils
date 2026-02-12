#!/usr/bin/env python
"""
Test script for Gemini provider with uniinfer.

NOTE: Gemini requires an API key. Get your key at https://console.cloud.google.com/
"""
import asyncio
import os
from uniinfer import GeminiProvider, ChatMessage, ChatCompletionRequest

def test_gemini_sync():
    """Test synchronous completion with Gemini."""
    print("=" * 60)
    print("Testing Gemini Synchronous Completion")
    print("=" * 60)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  Skipping: GEMINI_API_KEY not set")
        print("Get your API key at https://console.cloud.google.com/")
        return
    
    provider = GeminiProvider(api_key=api_key)
    
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Say hello!")],
        model="gemini-1.5-flash"
    )
    
    try:
        response = provider.complete(request)
        print(f"\nResponse: {response.message.content}")
        print(f"Model: {response.model}")
        print(f"Provider: {response.provider}")
        print(f"Usage: {response.usage}")
        print("\n✅ Synchronous test passed!")
    except Exception as e:
        print(f"\n❌ Synchronous test failed: {e}")
        raise

async def test_gemini_async():
    """Test asynchronous completion with Gemini."""
    print("\n" + "=" * 60)
    print("Testing Gemini Asynchronous Completion")
    print("=" * 60)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  Skipping: GEMINI_API_KEY not set")
        print("Get your API key at https://console.cloud.google.com/")
        return
    
    provider = GeminiProvider(api_key=api_key)
    
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="What is 2+2?")],
        model="gemini-1.5-flash"
    )
    
    try:
        response = await provider.acomplete(request)
        print(f"\nResponse: {response.message.content}")
        print(f"Model: {response.model}")
        print(f"Provider: {response.provider}")
        print(f"Usage: {response.usage}")
        print("\n✅ Asynchronous test passed!")
        await provider.close()
    except Exception as e:
        print(f"\n❌ Asynchronous test failed: {e}")
        raise

def test_gemini_stream():
    """Test streaming completion with Gemini."""
    print("\n" + "=" * 60)
    print("Testing Gemini Streaming Completion")
    print("=" * 60)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  Skipping: GEMINI_API_KEY not set")
        print("Get your API key at https://console.cloud.google.com/")
        return
    
    provider = GeminiProvider(api_key=api_key)
    
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Count to 5")],
        model="gemini-1.5-flash"
    )
    
    try:
        full_response = ""
        print("\nStreaming response:")
        for chunk in provider.stream_complete(request):
            content = chunk.message.content
            if content:
                full_response += content
                print(content, end="", flush=True)
        print()
        print(f"\nFull response: {full_response}")
        print("\n✅ Streaming test passed!")
    except Exception as e:
        print(f"\n❌ Streaming test failed: {e}")
        raise

async def test_gemini_async_stream():
    """Test async streaming completion with Gemini."""
    print("\n" + "=" * 60)
    print("Testing Gemini Asynchronous Streaming Completion")
    print("=" * 60)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  Skipping: GEMINI_API_KEY not set")
        print("Get your API key at https://console.cloud.google.com/")
        return
    
    provider = GeminiProvider(api_key=api_key)
    
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="List 3 colors")],
        model="gemini-1.5-flash"
    )
    
    try:
        full_response = ""
        print("\nAsync streaming response:")
        async for chunk in provider.astream_complete(request):
            content = chunk.message.content
            if content:
                full_response += content
                print(content, end="", flush=True)
        print()
        print(f"\nFull response: {full_response}")
        print("\n✅ Async streaming test passed!")
        await provider.close()
    except Exception as e:
        print(f"\n❌ Async streaming test failed: {e}")
        raise

async def test_gemini_models():
    """Test listing available models from Gemini."""
    print("\n" + "=" * 60)
    print("Testing Gemini List Models")
    print("=" * 60)
    
    try:
        models = GeminiProvider.list_models()
        print(f"\nFound {len(models)} models")
        print("Available models:")
        for i, model in enumerate(models):
            print(f"  {i+1}. {model.get('id')}")
        print("\n✅ List models test passed!")
    except Exception as e:
        print(f"\n❌ List models test failed: {e}")
        raise

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Gemini Provider Test Suite")
    print("=" * 60)
    print("\nNote: Gemini requires an API key")
    print("Get your key at https://console.cloud.google.com/")
    print("Set it via: export GEMINI_API_KEY=your_key")
    print()
    
    # Test sync methods first
    test_gemini_sync()
    test_gemini_stream()
    
    # Run async tests
    async def run_async_tests():
        await test_gemini_async()
        await test_gemini_async_stream()
        await test_gemini_models()
    
    asyncio.run(run_async_tests())
    
    print("\n" + "=" * 60)
    print("All tests completed successfully! 🎉")
    print("=" * 60)

if __name__ == "__main__":
    main()
