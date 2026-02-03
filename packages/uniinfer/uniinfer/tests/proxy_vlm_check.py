import requests
import base64
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("PROXY_KEY")
if not api_key:
    print("Error: PROXY_KEY not found in .env")
    exit(1)

api_token_header = f"Bearer {api_key}"

url = "http://127.0.0.1:8123/v1/chat/completions"
image_path = "examples/image.jpg"

if not os.path.exists(image_path):
    print(f"Image not found: {image_path}")
    exit(1)

print("Reading image...")
with open(image_path, "rb") as f:
    b64_data = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "model": "tu@glm-4.5v-106b",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image briefly."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_data}"}}
            ]
        }
    ],
    "stream": True
}

print(f"Sending request to {url}...")

try:
    response = requests.post(url, json=payload, headers={
        "Content-Type": "application/json",
        "Authorization": api_token_header
    }, stream=True, timeout=120)

    print(f"Response status: {response.status_code}")
    if response.status_code != 200:
        print(f"Error: {response.text}")
        exit(1)

    print("Streaming response:")
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            print(f"Chunk: {line_str}")

except Exception as e:
    print(f"Exception: {e}")
