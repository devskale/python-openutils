#!/usr/bin/env python3
"""
Test script for tool calling functionality across different providers.
"""
import os
from uniinfer.uniioai import get_completion, get_provider_api_key
from credgoo import get_api_key

# Define a simple weather tool
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use"
                }
            },
            "required": ["location"]
        }
    }
}

def test_provider_tool_calling(provider_name, model_name):
    """Test tool calling for a specific provider."""
    print(f"\n{'='*60}")
    print(f"Testing {provider_name}@{model_name}")
    print(f"{'='*60}\n")
    
    # Get API key
    try:
        bearer = os.getenv('CREDGOO_BEARER_TOKEN')
        encryption = os.getenv('CREDGOO_ENCRYPTION_KEY')
        
        if bearer and encryption:
            api_key = get_provider_api_key(f"{bearer}@{encryption}", provider_name)
        else:
            api_key = get_api_key(service=provider_name)
    except Exception as e:
        print(f"Error getting API key: {e}")
        return
    
    # Test message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's the weather like in "},
                {"type": "text", "text": "Paris?"}
            ]
        }
    ]
    
    try:
        # Make request with tools
        print("Making request with tool definitions...")
        response = get_completion(
            messages=messages,
            provider_model_string=f"{provider_name}@{model_name}",
            provider_api_key=api_key,
            tools=[WEATHER_TOOL],
            tool_choice="auto"
        )
        
        # Check response type
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print("✅ Tool call detected!")
            print("\nResponse type: ChatMessage with tool_calls")
            print(f"Content: {response.content}")
            print("\nTool calls:")
            for i, tool_call in enumerate(response.tool_calls, 1):
                print(f"\n  Tool call {i}:")
                print(f"    ID: {tool_call.get('id', 'N/A')}")
                print(f"    Type: {tool_call.get('type', 'N/A')}")
                if 'function' in tool_call:
                    print(f"    Function: {tool_call['function'].get('name', 'N/A')}")
                    print(f"    Arguments: {tool_call['function'].get('arguments', 'N/A')}")
        elif isinstance(response, str):
            print("ℹ️  String response (no tool calls)")
            print(f"\nResponse: {response}")
        else:
            print("ℹ️  ChatMessage response (no tool calls)")
            print(f"\nContent: {response.content if hasattr(response, 'content') else response}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run tests for different providers."""
    print("\n" + "="*60)
    print("Tool Calling Test Suite")
    print("="*60)
    
    # Test configurations: (provider, model)
    test_configs = [
        ("mistral", "mistral-small"),
        ("tu", "qwen-coder-30b"),
        # Uncomment to test Gemini (requires google-genai package)
        # ("gemini", "gemini-1.5-flash"),
    ]
    
    for provider, model in test_configs:
        try:
            test_provider_tool_calling(provider, model)
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error testing {provider}: {e}")
            continue
    
    print("\n" + "="*60)
    print("Test suite completed")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
