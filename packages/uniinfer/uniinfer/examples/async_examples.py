"""
Async usage examples for UniInfer providers.

This file demonstrates how to use async methods for chat completions and streaming.
"""
import asyncio
from uniinfer import ChatMessage, ChatCompletionRequest
from uniinfer.providers import OpenAIProvider, AnthropicProvider, MistralProvider, OllamaProvider


async def async_completion_example():
    """
    Example of async chat completion with OpenAI.
    """
    # Create provider with your API key
    provider = OpenAIProvider(api_key="your-api-key-here")

    # Create a request
    request = ChatCompletionRequest(
        messages=[
            ChatMessage(role="user", content="Hello, how are you?")
        ],
        model="gpt-3.5-turbo",
        temperature=0.7
    )

    # Make async completion request
    response = await provider.acomplete(request)

    print(f"Response: {response.message.content}")
    print(f"Provider: {response.provider}")
    print(f"Model: {response.model}")
    print(f"Usage: {response.usage}")


async def async_streaming_example():
    """
    Example of async streaming with OpenAI.
    """
    provider = OpenAIProvider(api_key="your-api-key-here")

    request = ChatCompletionRequest(
        messages=[
            ChatMessage(role="user", content="Tell me a short story.")
        ],
        model="gpt-3.5-turbo",
        temperature=0.7
    )

    # Stream async responses
    full_content = ""
    async for chunk in provider.astream_complete(request):
        content = chunk.message.content
        if content:
            full_content += content
            print(content, end="", flush=True)

    print(f"\n\nFull response: {full_content}")


async def multiple_providers_example():
    """
    Example of using multiple providers concurrently.
    """
    messages = [ChatMessage(role="user", content="Say hello!")]
    
    # Create multiple provider instances
    providers = [
        OpenAIProvider(api_key="your-openai-key"),
        AnthropicProvider(api_key="your-anthropic-key"),
        MistralProvider(api_key="your-mistral-key"),
    ]

    # Make requests concurrently
    tasks = []
    for provider in providers:
        request = ChatCompletionRequest(
            messages=messages,
            model=provider.__class__.__name__.replace("Provider", "").lower(),
            temperature=0.7
        )
        tasks.append(provider.acomplete(request))

    # Wait for all requests to complete
    responses = await asyncio.gather(*tasks)

    # Print all responses
    for i, response in enumerate(responses):
        print(f"\nProvider {i+1} ({response.provider}):")
        print(f"  Response: {response.message.content[:100]}...")


async def batch_completion_example():
    """
    Example of processing multiple requests with async.
    """
    provider = OpenAIProvider(api_key="your-api-key-here")

    # Create multiple requests
    requests = [
        ChatCompletionRequest(
            messages=[ChatMessage(role="user", content=f"What is {num} + {num}?")],
            model="gpt-3.5-turbo"
        )
        for num in range(1, 6)
    ]

    # Process all requests concurrently
    tasks = [provider.acomplete(req) for req in requests]
    responses = await asyncio.gather(*tasks)

    # Print results
    for i, response in enumerate(responses, 1):
        print(f"Question {i}: {requests[i-1].messages[0].content}")
        print(f"Answer: {response.message.content}\n")


async def ollama_async_example():
    """
    Example of async usage with Ollama (local provider).
    """
    # Ollama runs locally, adjust base_url as needed
    provider = OllamaProvider(base_url="http://localhost:11434")

    request = ChatCompletionRequest(
        messages=[
            ChatMessage(role="user", content="Hello!")
        ],
        model="llama2"
    )

    response = await provider.acomplete(request)
    print(f"Ollama Response: {response.message.content}")


async def anthropic_async_stream_example():
    """
    Example of async streaming with Anthropic Claude.
    """
    provider = AnthropicProvider(api_key="your-anthropic-key")

    request = ChatCompletionRequest(
        messages=[
            ChatMessage(role="user", content="Explain quantum computing in simple terms.")
        ],
        model="claude-3-sonnet-20240229",
        max_tokens=500
    )

    full_text = ""
    async for chunk in provider.astream_complete(request):
        content = chunk.message.content
        if content:
            full_text += content
            print(content, end="", flush=True)

    print(f"\n\nTotal characters: {len(full_text)}")


async def main():
    """
    Run all async examples.
    """
    print("=== Async Completion Example ===")
    # await async_completion_example()
    
    print("\n=== Async Streaming Example ===")
    # await async_streaming_example()
    
    print("\n=== Multiple Providers Example ===")
    # await multiple_providers_example()
    
    print("\n=== Batch Completion Example ===")
    # await batch_completion_example()
    
    print("\n=== Ollama Async Example ===")
    # await ollama_async_example()
    
    print("\n=== Anthropic Async Stream Example ===")
    # await anthropic_async_stream_example()
    
    print("\n=== All examples completed! ===")
    print("\nNote: Uncomment the await statements above to run each example.")
    print("Replace 'your-api-key-here' with your actual API keys.")


if __name__ == "__main__":
    asyncio.run(main())
