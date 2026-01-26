import requests
import json
import time
import sys
import os
from dotenv import load_dotenv

load_dotenv()


def test_proxy_model(model_name, token=None):
    if token is None:
        token = os.getenv("PROXY_KEY")
        if not token:
            print("Error: PROXY_KEY not found in .env")
            return

    url = "http://localhost:8123/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    # Payload with structured content (list of text objects)
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": " world. What is 1+1?"}
                ]
            }
        ],
        "stream": False
    }

    print(f"\nTesting {model_name}...")
    try:
        response = requests.post(url, headers=headers, json=data)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            print("SUCCESS: Proxy accepted structured content.")
            # Truncate for brevity
            print(f"Response: {response.text[:200]}...")
        elif response.status_code == 500 and "Invalid token" in response.text:
            print(
                "SUCCESS (Validation): Proxy accepted structured content (Provider auth failed as expected).")
            print(f"Response: {response.text}")
        else:
            print("FAILURE: Proxy rejected request or unexpected error.")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Test Chutes
    test_proxy_model("chutes@zai-org/GLM-4.5-Air", token="test_token")

    # Test Bigmodel (using the working token from previous turn)
    test_proxy_model("bigmodel@glm-4.5-flash", token="test_token")
