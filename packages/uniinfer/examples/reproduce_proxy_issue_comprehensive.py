import requests
import os
from dotenv import load_dotenv

load_dotenv()

PROXY_URL = "http://localhost:8123/v1/chat/completions"
PROXY_KEY = os.getenv("PROXY_KEY")

if not PROXY_KEY:
    print("Warning: PROXY_KEY not found in .env, using placeholder")
    PROXY_KEY = "placeholder"

TEST_CASES = [
    # (provider, model, token, stream)
    ("bigmodel", "glm-4-flash", PROXY_KEY, False), # Working token
    ("bigmodel", "glm-4-flash", PROXY_KEY, True),  # Working token, Streaming
    ("chutes", "zai-org/GLM-4.5-Air", "test_token", False),
    ("chutes", "zai-org/GLM-4.5-Air", "test_token", True),
    ("openrouter", "moonshotai/moonlight-16b-a3b-instruct:free", "test_token", False),
    ("mistral", "mistral-tiny", "test_token", False),
    ("tu", "openai/RedHatAI/DeepSeek-R1-0528-quantized.w4a16", "test_token", False),
    ("openai", "gpt-3.5-turbo", "test_token", False),
]

def run_test(provider, model, token, stream):
    print(f"Testing {provider}@{model} (Stream={stream})...")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    # Structured content
    content = [
        {"type": "text", "text": "Hello, "},
        {"type": "text", "text": "are you working?"}
    ]
    
    data = {
        "model": f"{provider}@{model}",
        "messages": [{"role": "user", "content": content}],
        "stream": stream
    }
    
    try:
        response = requests.post(PROXY_URL, headers=headers, json=data, stream=stream)
        
        if response.status_code == 200:
            print("  ✅ SUCCESS: 200 OK")
            if stream:
                # Consume stream to check for errors
                content_received = False
                for line in response.iter_lines():
                    if line:
                        content_received = True
                if content_received:
                    print("  ✅ Stream consumed successfully")
                else:
                    print("  ⚠️ Stream was empty")
            else:
                print(f"  Response: {response.text[:100]}...")
                
        elif response.status_code == 422:
            print("  ❌ FAILURE: 422 Unprocessable Entity (Validation Failed)")
            print(f"  Response: {response.text}")
            
        elif response.status_code in [401, 500]:
            # Check if it's an auth error from the provider (which means validation passed)
            text = ""
            try:
                text = response.text
            except Exception:
                pass
                
            if "Invalid token" in text or "Authentication" in text or "API key" in text or "Unauthorized" in text or "401" in text:
                 print(f"  ✅ SUCCESS (Validation): {response.status_code} Provider Auth Error (Expected)")
            else:
                 print(f"  ⚠️ WARNING: {response.status_code} Unexpected Error")
                 print(f"  Response: {text}")
        else:
            print(f"  ❌ FAILURE: {response.status_code}")
            print(f"  Response: {response.text}")
            
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")

if __name__ == "__main__":
    print("Starting Comprehensive Structured Content Test")
    print("============================================")
    for provider, model, token, stream in TEST_CASES:
        run_test(provider, model, token, stream)
        print("-" * 40)
