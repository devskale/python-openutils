import argparse
import json
import random
import requests

def build_tool_def():
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    }

def request_tool_call(proxy, token, model, location):
    url = f"{proxy.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": f"What is the weather in {location}?"}],
        "tools": [build_tool_def()],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "stream": False
    }
    resp = requests.post(url, headers=headers, data=json.dumps(body))
    resp.raise_for_status()
    return resp.json()

def create_fake_weather(location):
    styles = [
        f"ULTRA GREAT sunshine in {location} with blue skies",
        f"HORRIBLE TYPHOON hitting {location} with flying cows",
        f"EPIC BLIZZARD covering {location} in instant ice cream",
        f"MONSOON DELUXE soaking {location} in chocolate rain"
    ]
    return random.choice(styles)

def send_tool_result(proxy, token, model, user_location, tool_call, fake_weather):
    url = f"{proxy.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": f"What is the weather in {user_location}?"},
            {"role": "assistant", "content": None, "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": tool_call.get("id"), "content": fake_weather}
        ],
        "stream": False
    }
    resp = requests.post(url, headers=headers, data=json.dumps(body))
    resp.raise_for_status()
    return resp.json()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", default="http://0.0.0.0:8123")
    parser.add_argument("--token", required=True)
    parser.add_argument("--model", default="moonshot@kimi-latest")
    parser.add_argument("--location", default="Vienna")
    args = parser.parse_args()

    first = request_tool_call(args.proxy, args.token, args.model, args.location)
    choice = (first.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        print(json.dumps(first, indent=2))
        return
    counts = {}
    for t in tool_calls:
        name = (t.get("function") or {}).get("name") or "unknown"
        counts[name] = counts.get(name, 0) + 1
    tc = tool_calls[0]
    args_raw = (tc.get("function") or {}).get("arguments") or "{}"
    try:
        args_json = json.loads(args_raw)
    except Exception:
        args_json = {}
    loc = args_json.get("location", args.location)
    fake = create_fake_weather(loc)
    follow = send_tool_result(args.proxy, args.token, args.model, loc, tc, fake)
    summary_task = ""
    s = fake.lower()
    if "typhoon" in s or "storm" in s:
        summary_task = f"Shelter in place in {loc}, avoid travel"
    elif "blizzard" in s or "ice" in s:
        summary_task = f"Stock up and prepare heating for {loc}"
    elif "monsoon" in s or "rain" in s:
        summary_task = f"Carry waterproof gear and plan indoor day in {loc}"
    elif "sunshine" in s or "blue skies" in s:
        summary_task = f"Plan outdoor picnic in {loc}"
    else:
        summary_task = f"Stay alert and check updates for {loc}"
    print("Tools triggered:")
    for name, num in counts.items():
        print(f"- {name}: {num}")
    print("\nRoundtrip result:")
    print(json.dumps(follow, indent=2))
    print("\nSummary:")
    print(f"- Tool response: {fake}")
    print(f"- Task: {summary_task}")

if __name__ == "__main__":
    main()
