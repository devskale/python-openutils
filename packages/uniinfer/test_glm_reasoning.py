"""Test glm-4.5v-106b model on tu and localhost providers with reasoning tokens."""
import os
import sys
import asyncio
from dotenv import load_dotenv

load_dotenv()

from uniinfer import ChatMessage, ChatCompletionRequest
from uniinfer.providers.tu import TUProvider
from uniinfer.providers.openai_compatible import OpenAICompatibleChatProvider


def test_tu_provider():
    """Test glm-4.5v-106b on TU provider."""
    print("\n" + "="*60)
    print("Testing TU Provider with glm-4.5v-106b")
    print("="*60)
    
    try:
        provider = TUProvider()
        
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="user", content="What is 15 * 17? Please show your reasoning.")
            ],
            model="glm-4.5v-106b",
            temperature=0.7,
            max_tokens=500,
            streaming=False,
            reasoning_effort="medium"
        )
        
        print(f"Model: {request.model}")
        print(f"Reasoning effort: {request.reasoning_effort}")
        print("-"*40)
        
        response = provider.complete(request)
        
        print(f"Provider: {response.provider}")
        print(f"Model used: {response.model}")
        print(f"Finish reason: {response.finish_reason}")
        print(f"Usage: {response.usage}")
        
        if response.thinking:
            print(f"\n[THINKING/REASONING]:\n{response.thinking}\n")
        
        print(f"\n[RESPONSE]:\n{response.message.content}\n")
        
        return True
    except Exception as e:
        print(f"Error testing TU provider: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tu_provider_streaming():
    """Test glm-4.5v-106b on TU provider with streaming."""
    print("\n" + "="*60)
    print("Testing TU Provider (Streaming) with glm-4.5v-106b")
    print("="*60)
    
    try:
        provider = TUProvider()
        
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="user", content="Count from 1 to 5, explaining each number.")
            ],
            model="glm-4.5v-106b",
            temperature=0.7,
            max_tokens=300,
            streaming=True,
            reasoning_effort="low"
        )
        
        print(f"Model: {request.model}")
        print(f"Reasoning effort: {request.reasoning_effort}")
        print("-"*40)
        
        thinking_content = ""
        response_content = ""
        
        async def stream():
            nonlocal thinking_content, response_content
            async for chunk in provider.astream_complete(request):
                if chunk.thinking:
                    thinking_content += chunk.thinking
                    print(f"[THINKING] {chunk.thinking}", end="", flush=True)
                if chunk.message.content:
                    response_content += chunk.message.content
                    print(chunk.message.content, end="", flush=True)
        
        asyncio.run(stream())
        print("\n")
        
        if thinking_content:
            print(f"\n[FULL THINKING]:\n{thinking_content}\n")
        
        print(f"\n[FULL RESPONSE]:\n{response_content}\n")
        
        return True
    except Exception as e:
        print(f"Error testing TU provider (streaming): {e}")
        import traceback
        traceback.print_exc()
        return False


def test_localhost_provider():
    """Test glm-4.5v-106b on localhost uniioai-proxy."""
    print("\n" + "="*60)
    print("Testing Localhost Proxy with glm-4.5v-106b")
    print("="*60)
    
    try:
        # uniioai-proxy runs on localhost:8123/v1 with bearer test23@test34
        base_url = "http://localhost:8123/v1"
        api_key = "test23@test34"
        
        provider = OpenAICompatibleChatProvider(base_url=base_url, api_key=api_key)
        
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="user", content="What is 15 * 17? Please show your reasoning.")
            ],
            model="tu@glm-4.5v-106b",  # Use tu@ prefix for proxy
            temperature=0.7,
            max_tokens=500,
            streaming=False,
            reasoning_effort="medium"
        )
        
        print(f"Model: {request.model}")
        print(f"Base URL: {base_url}")
        print(f"Reasoning effort: {request.reasoning_effort}")
        print("-"*40)
        
        response = provider.complete(request)
        
        print(f"Provider: {response.provider}")
        print(f"Model used: {response.model}")
        print(f"Finish reason: {response.finish_reason}")
        print(f"Usage: {response.usage}")
        
        if response.thinking:
            print(f"\n[THINKING/REASONING]:\n{response.thinking}\n")
        
        print(f"\n[RESPONSE]:\n{response.message.content}\n")
        
        return True
    except Exception as e:
        print(f"Error testing localhost provider: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_localhost_provider_streaming():
    """Test glm-4.5v-106b on localhost uniioai-proxy with streaming."""
    print("\n" + "="*60)
    print("Testing Localhost Proxy (Streaming) with glm-4.5v-106b")
    print("="*60)
    
    try:
        base_url = "http://localhost:8123/v1"
        api_key = "test23@test34"
        
        provider = OpenAICompatibleChatProvider(base_url=base_url, api_key=api_key)
        
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="user", content="Count from 1 to 5, explaining each number.")
            ],
            model="tu@glm-4.5v-106b",
            temperature=0.7,
            max_tokens=300,
            streaming=True,
            reasoning_effort="low"
        )
        
        print(f"Model: {request.model}")
        print(f"Base URL: {base_url}")
        print(f"Reasoning effort: {request.reasoning_effort}")
        print("-"*40)
        
        thinking_content = ""
        response_content = ""
        
        async def stream():
            nonlocal thinking_content, response_content
            async for chunk in provider.astream_complete(request):
                if chunk.thinking:
                    thinking_content += chunk.thinking
                    print(f"[THINKING] {chunk.thinking}", end="", flush=True)
                if chunk.message.content:
                    response_content += chunk.message.content
                    print(chunk.message.content, end="", flush=True)
        
        asyncio.run(stream())
        print("\n")
        
        if thinking_content:
            print(f"\n[FULL THINKING]:\n{thinking_content}\n")
        
        print(f"\n[FULL RESPONSE]:\n{response_content}\n")
        
        return True
    except Exception as e:
        print(f"Error testing localhost provider (streaming): {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing glm-4.5v-106b model with reasoning tokens")
    print("="*60)
    
    # Test TU provider
    tu_success = test_tu_provider()
    
    # Test TU streaming
    tu_stream_success = test_tu_provider_streaming()
    
    # Test localhost proxy
    localhost_success = test_localhost_provider()
    
    # Test localhost streaming
    localhost_stream_success = test_localhost_provider_streaming()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"TU Provider (non-streaming):       {'✓ PASS' if tu_success else '✗ FAIL'}")
    print(f"TU Provider (streaming):           {'✓ PASS' if tu_stream_success else '✗ FAIL'}")
    print(f"Localhost Proxy (non-streaming):   {'✓ PASS' if localhost_success else '✗ FAIL'}")
    print(f"Localhost Proxy (streaming):       {'✓ PASS' if localhost_stream_success else '✗ FAIL'}")
