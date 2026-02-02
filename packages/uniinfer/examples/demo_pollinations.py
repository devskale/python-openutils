#!/usr/bin/env python
"""
Test script for Pollinations provider with uniinfer.

NOTE: Pollinations requires an API key. Get your key at https://enter.pollinations.ai
"""
import asyncio
import os
from uniinfer import PollinationsProvider, ChatMessage, ChatCompletionRequest

def test_pollinations_sync():
    """Test synchronous completion with Pollinations."""
    print("=" * 60)
    print("Testing Pollinations Synchronous Completion")
    print("=" * 60)
    
    # Get API key from environment
    api_key = os.environ.get("POLLINATIONS_API_KEY")
    if not api_key:
        print("⚠️  Skipping: POLLINATIONS_API_KEY not set")
        print("Get your API key at https://enter.pollinations.ai")
        return
    
    provider = PollinationsProvider(api_key=api_key)
    
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Say hello!")],
        model="openai"
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

async def test_pollinations_async():
    """Test asynchronous completion with Pollinations."""
    print("\n" + "=" * 60)
    print("Testing Pollinations Asynchronous Completion")
    print("=" * 60)
    
    api_key = os.environ.get("POLLINATIONS_API_KEY")
    if not api_key:
        print("⚠️  Skipping: POLLINATIONS_API_KEY not set")
        print("Get your API key at https://enter.pollinations.ai")
        return
    
    provider = PollinationsProvider(api_key=api_key)
    
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="What is 2+2?")],
        model="openai"
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

def test_pollinations_stream():
    """Test streaming completion with Pollinations."""
    print("\n" + "=" * 60)
    print("Testing Pollinations Streaming Completion")
    print("=" * 60)
    
    api_key = os.environ.get("POLLINATIONS_API_KEY")
    if not api_key:
        print("⚠️  Skipping: POLLINATIONS_API_KEY not set")
        print("Get your API key at https://enter.pollinations.ai")
        return
    
    provider = PollinationsProvider(api_key=api_key)
    
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Count to 5")],
        model="openai"
    )
    
    try:
        full_response = ""
        print("\nStreaming response:")
        for chunk in provider.stream_complete(request):
            content = chunk.message.content
            full_response += content
            print(content, end="", flush=True)
        print()
        print(f"\nFull response: {full_response}")
        print("\n✅ Streaming test passed!")
    except Exception as e:
        print(f"\n❌ Streaming test failed: {e}")
        raise

async def test_pollinations_async_stream():
    """Test async streaming completion with Pollinations."""
    print("\n" + "=" * 60)
    print("Testing Pollinations Asynchronous Streaming Completion")
    print("=" * 60)
    
    api_key = os.environ.get("POLLINATIONS_API_KEY")
    if not api_key:
        print("⚠️  Skipping: POLLINATIONS_API_KEY not set")
        print("Get your API key at https://enter.pollinations.ai")
        return
    
    provider = PollinationsProvider(api_key=api_key)
    
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="List 3 colors")],
        model="openai"
    )
    
    try:
        full_response = ""
        print("\nAsync streaming response:")
        async for chunk in provider.astream_complete(request):
            content = chunk.message.content
            full_response += content
            print(content, end="", flush=True)
        print()
        print(f"\nFull response: {full_response}")
        print("\n✅ Async streaming test passed!")
        await provider.close()
    except Exception as e:
        print(f"\n❌ Async streaming test failed: {e}")
        raise

async def test_pollinations_models():
    """Test listing available models from Pollinations."""
    print("\n" + "=" * 60)
    print("Testing Pollinations List Models")
    print("=" * 60)
    
    try:
        models = PollinationsProvider.list_models()
        print(f"\nFound {len(models)} models")
        print("First 10 models:")
        for i, model in enumerate(models[:10]):
            print(f"  {i+1}. {model}")
        print("\n✅ List models test passed!")
    except Exception as e:
        print(f"\n❌ List models test failed: {e}")
        raise

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Pollinations Provider Test Suite")
    print("=" * 60)
    print("\nNote: Pollinations requires an API key")
    print("Get your key at https://enter.pollinations.ai")
    print("Set it via: export POLLINATIONS_API_KEY=your_key")
    print()
    
    # Test sync methods first
    test_pollinations_sync()
    test_pollinations_stream()
    
    # Run async tests
    async def run_async_tests():
        await test_pollinations_async()
        await test_pollinations_async_stream()
        await test_pollinations_models()
    
    asyncio.run(run_async_tests())
    
    print("\n" + "=" * 60)
    print("All tests completed successfully! 🎉")
    print("=" * 60)

if __name__ == "__main__":
    main()
