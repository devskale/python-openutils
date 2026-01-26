import argparse
import json
import base64
import time
from pathlib import Path
import requests

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", default="http://0.0.0.0:8123")
    parser.add_argument("--token", required=True)
    parser.add_argument("--model", default="pollinations@turbo")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--size", default="256x256")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="./images")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    url = f"{args.proxy.rstrip('/')}/v1/images/generations"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {args.token}"}
    body = {
        "model": args.model,
        "prompt": args.prompt,
        "n": args.n,
        "size": args.size,
        "seed": args.seed,
    }
    resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=120)
    resp.raise_for_status()
    data = resp.json()

    ts = int(time.time())
    def slugify(s: str) -> str:
        return "".join(c.lower() if c.isalnum() else "-" for c in s).strip("-")[:60]

    prompt_slug = slugify(args.prompt)
    model_slug = slugify(args.model)

    saved = []
    for i, item in enumerate(data.get("data", [])):
        b64 = item.get("b64_json")
        if not b64:
            continue
        fname = outdir / f"{model_slug}_{prompt_slug}_{args.size}_{args.seed}_{ts}_{i}.png"
        fname.write_bytes(base64.b64decode(b64))
        saved.append(str(fname))

    print(json.dumps({"saved": saved, "model": data.get("model"), "count": len(saved)}, indent=2))

if __name__ == "__main__":
    main()

