
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load Configuration
CONFIG_FILE = "test_compliance_config.json"
try:
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    API_KEY = config.get("api_key")
    BASE_URL = config.get("base_url")
    MODELS_TO_TEST = config.get("models_to_test", [])
except FileNotFoundError:
    print(f"Error: Config file '{CONFIG_FILE}' not found.")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: Failed to decode JSON from '{CONFIG_FILE}'.")
    sys.exit(1)


def test_model_compliance(client, model):
    print(f"\n{'='*50}")
    print(f"Testing Model: {model}")
    print(f"{'='*50}")

    # 1. Test Non-Streaming Chat Completion
    print("\n[1] Testing Non-Streaming Chat Completion...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Say 'Hello, World!' and nothing else."}],
            temperature=0.1
        )
        print("  - Response received.")

        # Validation
        assert response.choices[0].message.content is not None, "Content should not be None"
        assert response.choices[0].finish_reason in [
            "stop", "length", "tool_calls"], f"Invalid finish_reason: {response.choices[0].finish_reason}"
        print(f"  - Content: {response.choices[0].message.content}")
        print(f"  - Finish Reason: {response.choices[0].finish_reason}")
        print("  ✅ Non-Streaming Test Passed")
    except Exception as e:
        print(f"  ❌ Non-Streaming Test Failed: {e}")

    # 2. Test Streaming Chat Completion
    print("\n[2] Testing Streaming Chat Completion...")
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Count from 1 to 3."}],
            stream=True,
            temperature=0.1
        )

        print("  - Stream started.")
        collected_content = ""
        finish_reason = None

        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    collected_content += delta.content
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

        print(f"  - Full Streamed Content: {collected_content}")
        print(f"  - Final Finish Reason: {finish_reason}")

        assert len(collected_content) > 0, "Streamed content should not be empty"
        assert finish_reason is not None, "Finish reason should be present in the last chunk"
        print("  ✅ Streaming Test Passed")
    except Exception as e:
        print(f"  ❌ Streaming Test Failed: {e}")

    # 3. Test Tool Calling (if supported)
    # We will assume these models support tools for the sake of the compliance check
    print("\n[3] Testing Tool Calling (Weather Function)...")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "What's the weather like in Boston?"}],
            tools=tools,
            tool_choice="auto",
        )

        message = response.choices[0].message
        tool_calls = message.tool_calls
        finish_reason = response.choices[0].finish_reason

        print(f"  - Finish Reason: {finish_reason}")

        if tool_calls:
            print(f"  - Tool Calls received: {len(tool_calls)}")
            for tc in tool_calls:
                print(f"    - Name: {tc.function.name}")
                print(f"    - Arguments: {tc.function.arguments}")

            assert finish_reason == "tool_calls", f"Finish reason should be 'tool_calls', got '{finish_reason}'"
            print("  ✅ Tool Calling Test Passed")
        else:
            print(
                f"  ⚠️ No tool calls generated. Response content: {message.content}")
            print("  (This might be model behavior, not necessarily a compliance failure if the model refused to call the tool)")

    except Exception as e:
        print(f"  ❌ Tool Calling Test Failed: {e}")

    # 4. Test Max Tokens
    print("\n[4] Testing Max Tokens (max_tokens=5)...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Write a long story about a dragon."}],
            max_tokens=5,
            temperature=0.1
        )
        print("  - Response received.")

        # Validation
        finish_reason = response.choices[0].finish_reason
        content = response.choices[0].message.content
        print(f"  - Content: {content}")
        print(f"  - Finish Reason: {finish_reason}")

        assert finish_reason == "length", f"Finish reason should be 'length', got '{finish_reason}'"
        assert len(content) > 0, "Content should not be empty"
        print("  ✅ Max Tokens Test Passed")
    except Exception as e:
        print(f"  ❌ Max Tokens Test Failed: {e}")


def main():
    print(f"Connecting to {BASE_URL} with API Key: {API_KEY[:4]}***")

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

    # List Models
    print("\n[0] Listing Available Models...")
    try:
        models = client.models.list()
        print(f"  - Found {len(models.data)} models.")
        model_ids = [m.id for m in models.data]
        # print(f"  - Models: {model_ids}")
    except Exception as e:
        print(f"  ❌ List Models Failed: {e}")
        return

    for model in MODELS_TO_TEST:
        if model not in model_ids:
            print(
                f"\n⚠️ Warning: Model '{model}' not found in listed models. Attempting to test anyway...")

        test_model_compliance(client, model)


if __name__ == "__main__":
    main()
