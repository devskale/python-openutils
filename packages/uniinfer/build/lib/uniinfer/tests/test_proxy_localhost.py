import requests
import json

import os
from dotenv import load_dotenv

load_dotenv()

# Configuration from opencode.json
BASE_URL = "http://localhost:8123/v1/chat/completions"
API_KEY = os.getenv("PROXY_KEY")
MODEL = "tu@qwen-coder-30b"


def test_proxy_streaming_tool_call():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # Define a tool (xlsx skill simulation)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "xlsx",
                "description": "Create an Excel file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to process into excel"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    data = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "Here is some data: Sales for Q1 were $1M, Q2 were $1.2M. Summarize this into a xlsx file."}
        ],
        "tools": tools,
        "stream": True
    }

    print(f"Connecting to {BASE_URL} with model {MODEL}...")

    try:
        response = requests.post(
            BASE_URL, headers=headers, json=data, stream=True)
        response.raise_for_status()

        print("Response received. Reading stream...")

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    json_str = decoded_line[6:]
                    if json_str == "[DONE]":
                        print("\n[DONE] received.")
                        break
                    try:
                        chunk = json.loads(json_str)
                        # Print relevant parts of the chunk
                        delta = chunk['choices'][0]['delta']
                        finish_reason = chunk['choices'][0]['finish_reason']

                        if 'content' in delta and delta['content']:
                            print(
                                f"Content: {delta['content']}", end="", flush=True)
                        if 'tool_calls' in delta:
                            print(f"\nTool Call: {delta['tool_calls']}")
                        if finish_reason:
                            print(f"\nFinish Reason: {finish_reason}")

                    except json.JSONDecodeError:
                        print(f"\nFailed to decode: {json_str}")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    test_proxy_streaming_tool_call()
