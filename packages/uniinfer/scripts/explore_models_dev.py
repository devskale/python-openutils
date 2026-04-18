"""Explorative script: fetch models.dev, map to uniinfer providers, derive types."""

import json
import urllib.request
from pathlib import Path

MODELS_DEV_URL = "https://models.dev/api.json"
CACHE_PATH = Path(__file__).parent / "_models_dev_cache.json"

UNIINFER_TO_MODELS_DEV = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "google",
    "mistral": "mistral",
    "groq": "groq",
    "cohere": "cohere",
    "openrouter": "openrouter",
    "ollama": "ollama-cloud",
    "chutes": "chutes",
    "cloudflare": "cloudflare-workers-ai",
    "minimax": "minimax",
    "upstage": "upstage",
    "stepfun": "stepfun",
    "moonshot": "moonshotai",
    "huggingface": "huggingface",
    "zai": "zai",
    "zai-code": "zai",
    "sambanova": "nova",
    "ngc": "nvidia",
}


def fetch_models_dev() -> dict:
    if CACHE_PATH.exists():
        print(f"Using cached data from {CACHE_PATH}")
        with open(CACHE_PATH) as f:
            return json.load(f)

    print(f"Fetching {MODELS_DEV_URL} ...")
    req = urllib.request.Request(MODELS_DEV_URL, headers={"User-Agent": "uniinfer-explore/1.0"})
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
    CACHE_PATH.write_bytes(raw)
    return json.loads(raw)


def derive_type(model: dict) -> str:
    family = model.get("family", "").lower()
    modalities = model.get("modalities", {})
    input_mod = modalities.get("input", [])
    output_mod = modalities.get("output", [])

    if "embed" in family:
        return "embed"
    if output_mod == ["audio"]:
        return "tts"
    if input_mod == ["audio"] and output_mod == ["text"]:
        return "stt"
    return "chat"


def main():
    data = fetch_models_dev()
    print(f"models.dev: {len(data)} providers loaded\n")

    print("=" * 90)
    print(f"{'Provider':<14} {'md_prov':<26} {'Models':>5} {'Chat':>5} {'Emb':>4} {'TTS':>4} {'STT':>4} {'Dep':>4}")
    print("=" * 90)

    total_models = 0
    total_by_type = {"chat": 0, "embed": 0, "tts": 0, "stt": 0}
    total_deprecated = 0

    for ui_id, md_id in UNIINFER_TO_MODELS_DEV.items():
        md_provider = data.get(md_id)
        if not md_provider:
            print(f"{ui_id:<14} {md_id:<26} {'NOT FOUND':>5}")
            continue

        models = md_provider.get("models", {})
        counts = {"chat": 0, "embed": 0, "tts": 0, "stt": 0, "deprecated": 0}

        for model_id, model in models.items():
            t = derive_type(model)
            counts[t] += 1
            total_by_type[t] += 1
            if model.get("status") == "deprecated":
                counts["deprecated"] += 1
                total_deprecated += 1

        total_models += len(models)
        print(
            f"{ui_id:<14} {md_id:<26} {len(models):>5}"
            f" {counts['chat']:>5} {counts['embed']:>4}"
            f" {counts['tts']:>4} {counts['stt']:>4} {counts['deprecated']:>4}"
        )

    print("=" * 90)
    print(
        f"{'TOTAL':<14} {'':<26} {total_models:>5}"
        f" {total_by_type['chat']:>5} {total_by_type['embed']:>4}"
        f" {total_by_type['tts']:>4} {total_by_type['stt']:>4} {total_deprecated:>4}"
    )

    # Show a sample enriched model entry for each type
    print("\n\n--- Sample enriched entries per type ---\n")
    seen_types = set()

    for ui_id, md_id in UNIINFER_TO_MODELS_DEV.items():
        md_provider = data.get(md_id)
        if not md_provider:
            continue
        for model_id, model in md_provider.get("models", {}).items():
            t = derive_type(model)
            if t in seen_types:
                continue
            seen_types.add(t)

            status = model.get("status") or "active"
            ctx = model.get("limit", {}).get("context")
            max_out = model.get("limit", {}).get("output")
            cost_in = model.get("cost", {}).get("input")
            cost_out = model.get("cost", {}).get("output")
            input_mod = model.get("modalities", {}).get("input", [])
            output_mod = model.get("modalities", {}).get("output", [])

            print(f"[{t.upper()}] {ui_id}/{model_id}")
            print(f"  name:          {model.get('name')}")
            print(f"  family:        {model.get('family')}")
            print(f"  status:        {status}")
            print(f"  context:       {ctx or 'N/A'}")
            print(f"  max_output:    {max_out or 'N/A'}")
            print(f"  input:         {input_mod}")
            print(f"  output:        {output_mod}")
            print(f"  cost:          ${cost_in}/M in, ${cost_out}/M out" if cost_in else f"  cost:          N/A")
            print(f"  reasoning:     {model.get('reasoning')}")
            print(f"  tool_call:     {model.get('tool_call')}")
            print(f"  open_weights:  {model.get('open_weights')}")
            print(f"  release_date:  {model.get('release_date')}")
            print(f"  knowledge:     {model.get('knowledge')}")
            print()

            if len(seen_types) == 4:
                break
        if len(seen_types) == 4:
            break

    # Check: are there any models with no context window?
    print("\n--- Models missing context window (first 10 per provider) ---\n")
    for ui_id, md_id in UNIINFER_TO_MODELS_DEV.items():
        md_provider = data.get(md_id)
        if not md_provider:
            continue
        missing = []
        for model_id, model in md_provider.get("models", {}).items():
            ctx = model.get("limit", {}).get("context")
            if not ctx:
                missing.append(model_id)
        if missing:
            print(f"  {ui_id}: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    # Check: openai embedding models
    print("\n--- Spot check: openai embedding models ---\n")
    md_provider = data.get("openai")
    if md_provider:
        for model_id, model in md_provider.get("models", {}).items():
            if "embed" in model_id.lower() or "embed" in model.get("family", "").lower():
                t = derive_type(model)
                print(f"  {model_id} → {t} (family: {model.get('family')})")

    # Check: groq TTS/STT models
    print("\n--- Spot check: groq TTS/STT models ---\n")
    md_provider = data.get("groq")
    if md_provider:
        for model_id, model in md_provider.get("models", {}).items():
            t = derive_type(model)
            if t in ("tts", "stt"):
                print(f"  {model_id} → {t} (input: {model.get('modalities',{}).get('input',[])}, output: {model.get('modalities',{}).get('output',[])})")

    # Check: chat models with image input (multimodal)
    print("\n--- Chat models with image input (first 5 per provider) ---\n")
    for ui_id, md_id in UNIINFER_TO_MODELS_DEV.items():
        md_provider = data.get(md_id)
        if not md_provider:
            continue
        multimodal = []
        for model_id, model in md_provider.get("models", {}).items():
            t = derive_type(model)
            input_mod = model.get("modalities", {}).get("input", [])
            if t == "chat" and "image" in input_mod:
                multimodal.append(model_id)
        if multimodal:
            print(f"  {ui_id}: {multimodal[:5]}{'...' if len(multimodal) > 5 else ''}")


if __name__ == "__main__":
    main()
