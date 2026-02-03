import requests
import os
from dotenv import load_dotenv

load_dotenv()

def test_proxy():
    url = "http://localhost:8123/v1/chat/completions"
    
    token = os.getenv("PROXY_KEY")
    if not token:
        print("Error: PROXY_KEY not found in .env")
        return

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    # Payload with structured content (list of text objects)
    data = {
        "model": "bigmodel@glm-4-flash", # Using Bigmodel provider
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

    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, headers=headers, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("SUCCESS: Proxy accepted structured content.")
        else:
            print("FAILURE: Proxy rejected request.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Wait a bit for proxy to start if running in same script (not here)
    test_proxy()
