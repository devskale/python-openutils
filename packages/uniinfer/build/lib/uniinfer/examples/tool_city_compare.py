import argparse
import json
import random
import re
import requests

def call_proxy(proxy, token, body):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    resp = requests.post(f"{proxy.rstrip('/')}/v1/chat/completions", headers=headers, data=json.dumps(body))
    resp.raise_for_status()
    return resp.json()

def extract_content(resp):
    try:
        return ((resp.get("choices") or [{}])[0].get("message") or {}).get("content")
    except Exception:
        return None

def generate_cities(proxy, token, model):
    prompt = "Return JSON: {\"cities\": [\"CITY_A\", \"CITY_B\"]} with two random well-known cities, no extra text."
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False}
    resp = call_proxy(proxy, token, body)
    content = extract_content(resp)
    cities = []
    if content:
        m = re.search(r"\{[\s\S]*\}", content)
        if m:
            try:
                data = json.loads(m.group(0))
                cities = data.get("cities") or []
            except Exception:
                pass
    if len(cities) < 2:
        fallback = ["Vienna", "Graz", "Paris", "Tokyo", "New York", "Cairo", "Mumbai", "Osaka", "Berlin", "Seoul"]
        cities = random.sample(fallback, 2)
    return cities[:2]

def tool_def():
    return {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}

def request_tool_call(proxy, token, model, city):
    body = {"model": model, "messages": [{"role": "user", "content": f"What is the weather in {city}?"}], "tools": [tool_def()], "tool_choice": {"type": "function", "function": {"name": "get_weather"}}, "stream": False}
    return call_proxy(proxy, token, body)

def create_fake_weather(city):
    styles = [f"ULTRA GREAT sunshine in {city} with blue skies", f"HORRIBLE TYPHOON hitting {city} with flying cows", f"EPIC BLIZZARD covering {city} in instant ice cream", f"MONSOON DELUXE soaking {city} in chocolate rain"]
    return random.choice(styles)

def send_tool_result(proxy, token, model, city, tc, fake):
    body = {"model": model, "messages": [{"role": "user", "content": f"What is the weather in {city}?"}, {"role": "assistant", "content": None, "tool_calls": [tc]}, {"role": "tool", "tool_call_id": tc.get("id"), "content": fake}], "stream": False}
    return call_proxy(proxy, token, body)

def score(fake):
    s = fake.lower()
    sc = 0
    if "sunshine" in s: sc += 2  # noqa: E701
    if "blue skies" in s: sc += 1  # noqa: E701
    if "typhoon" in s: sc -= 3  # noqa: E701
    if "blizzard" in s: sc -= 2  # noqa: E701
    if "monsoon" in s: sc -= 2  # noqa: E701
    if "rain" in s: sc -= 1  # noqa: E701
    if "ice" in s: sc -= 1  # noqa: E701
    return sc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", default="http://0.0.0.0:8123")
    parser.add_argument("--token", required=True)
    parser.add_argument("--model", default="moonshot@kimi-latest")
    args = parser.parse_args()

    cities = generate_cities(args.proxy, args.token, args.model)
    print(f"Cities: {cities[0]}, {cities[1]}")

    results = []
    total_counts = {}
    for city in cities:
        first = request_tool_call(args.proxy, args.token, args.model, city)
        msg = ((first.get("choices") or [{}])[0]).get("message") or {}
        tool_calls = msg.get("tool_calls") or []
        counts = {}
        for t in tool_calls:
            name = (t.get("function") or {}).get("name") or "unknown"
            counts[name] = counts.get(name, 0) + 1
            total_counts[name] = total_counts.get(name, 0) + 1
        tc = tool_calls[0] if tool_calls else {"id": "get_weather:0", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps({"location": city})}}
        fake = create_fake_weather(city)
        follow = send_tool_result(args.proxy, args.token, args.model, city, tc, fake)
        results.append({"city": city, "fake": fake, "score": score(fake), "counts": counts, "final": follow})

    best = max(results, key=lambda r: r["score"]) if results else None

    print("\nTools triggered:")
    for name, num in total_counts.items():
        print(f"- {name}: {num}")

    print("\nWeather summaries:")
    for r in results:
        print(f"- {r['city']}: {r['fake']} (score {r['score']})")

    if best:
        print("\nPreference:")
        print(f"- Choose {best['city']} because: {best['fake']}")

if __name__ == "__main__":
    main()

