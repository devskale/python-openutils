"""Generate models.json by calling list_models() on all installed providers."""

import json
import sys
import dataclasses
from uniinfer.core import ModelInfo
import logging
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("generate_models")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "uniinfer" / "models" / "models.json"
MODELS_DEV_CACHE = PROJECT_ROOT / "scripts" / "_models_dev_cache.json"
MODELS_DEV_URL = "https://models.dev/api.json"

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


def discover_providers():
    """Return list of (provider_id, provider_class, credgoo_service)."""
    from uniinfer import ProviderFactory, EmbeddingProviderFactory
    from uniinfer.core import TTSProvider, STTProvider
    from uniinfer.config.providers import PROVIDER_CONFIGS
    import uniinfer.providers as mod

    entries = []

    # Chat providers
    for name, cls in ProviderFactory._providers.items():
        svc = PROVIDER_CONFIGS.get(name, {}).get("credgoo_service", name)
        entries.append((name, cls, svc, "chat"))

    # Embedding providers
    for name, cls in EmbeddingProviderFactory._providers.items():
        svc = PROVIDER_CONFIGS.get(name, {}).get("credgoo_service", name)
        entries.append((name, cls, svc, "embed"))

    # TTS/STT providers — not in factories, scan module
    for attr_name in sorted(dir(mod)):
        cls = getattr(mod, attr_name, None)
        if cls is None or not isinstance(cls, type):
            continue
        if not issubclass(cls, (TTSProvider, STTProvider)):
            continue
        if cls in (TTSProvider, STTProvider):
            continue
        # Deduplicate (don't re-add if already in chat/embed factories)
        provider_id = getattr(cls, "PROVIDER_ID", attr_name.lower().replace("provider", "").replace("_", "-"))
        existing_ids = {e[0] for e in entries}
        if provider_id in existing_ids:
            continue
        svc = PROVIDER_CONFIGS.get(provider_id, {}).get("credgoo_service", provider_id)
        kind = "tts" if issubclass(cls, TTSProvider) and not issubclass(cls, STTProvider) else "stt"
        entries.append((provider_id, cls, svc, kind))

    return entries


def model_info_to_dict(m) -> dict:
    """Serialize a ModelInfo to a JSON-safe dict."""
    d = {"id": m.id}
    for field in dataclasses.fields(m):
        val = getattr(m, field.name)
        if val is None or field.name == "id":
            continue
        if field.name == "raw":
            continue
        d[field.name] = val
    return d


def fetch_provider_models(provider_id, cls, credgoo_service, kind):
    """Call list_models() on a provider, return list of dicts."""
    kwargs = {}
    try:
        from credgoo.credgoo import get_api_key
        api_key = get_api_key(credgoo_service)
        kwargs["api_key"] = api_key
    except Exception:
        log.warning("  %s: no API key via credgoo '%s'", provider_id, credgoo_service)
        try:
            models = cls.list_models()
        except Exception as e:
            log.warning("  %s: failed without key: %s", provider_id, e)
            return []
        return [model_info_to_dict(m) for m in models]

    # Provider-specific extra params (e.g. cloudflare account_id, ollama base_url)
    from uniinfer.config.providers import PROVIDER_CONFIGS
    extra = PROVIDER_CONFIGS.get(provider_id, {}).get("extra_params", {})
    kwargs.update(extra)

    try:
        models = cls.list_models(**kwargs)
    except Exception as e:
        log.warning("  %s: list_models() failed: %s", provider_id, e)
        import traceback
        traceback.print_exc()
        return []

    return [model_info_to_dict(m) for m in models]


def load_type_overrides() -> dict:
    """Load curated type overrides from type_overrides.json."""
    overrides_path = PROJECT_ROOT / "uniinfer" / "models" / "type_overrides.json"
    if not overrides_path.exists():
        return {}
    try:
        with open(overrides_path) as f:
            data = json.load(f)
        return data.get("models", {})
    except Exception:
        return {}


def fetch_models_dev() -> dict:
    """Load models.dev data from cache, or fetch fresh."""
    if MODELS_DEV_CACHE.exists():
        log.info("Using cached models.dev data")
        with open(MODELS_DEV_CACHE) as f:
            return json.load(f)

    log.info("Fetching models.dev ...")
    import urllib.request
    req = urllib.request.Request(MODELS_DEV_URL, headers={"User-Agent": "uniinfer-generate/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    MODELS_DEV_CACHE.write_bytes(raw)
    return json.loads(raw)


def merge_models_dev(models: list[dict], dev_provider: dict) -> list[dict]:
    """Enrich models with models.dev data.

    Priority: live API data wins over models.dev.
    models.dev fills: name, context_window, max_output, cost, modalities,
    capabilities, dimensions (for embed), release_date, knowledge_cutoff.
    """
    dev_models = dev_provider.get("models", {})
    enriched = 0

    for m in models:
        dev = dev_models.get(m["id"])
        if not dev:
            continue

        if not m.get("name") and dev.get("name"):
            m["name"] = dev["name"]

        if not m.get("context_window"):
            ctx = dev.get("limit", {}).get("context")
            if ctx:
                m["context_window"] = ctx

        if not m.get("max_output"):
            out = dev.get("limit", {}).get("output")
            if out:
                m["max_output"] = out

        if not m.get("cost") and dev.get("cost"):
            m["cost"] = dev["cost"]

        if not m.get("modalities") and dev.get("modalities"):
            m["modalities"] = dev["modalities"]

        dev_caps = {}
        for key in ("reasoning", "tool_call", "structured_output", "vision"):
            if dev.get(key) is not None:
                dev_caps[key] = dev[key]
        if dev_caps:
            existing = m.get("capabilities") or {}
            merged = {**dev_caps, **existing}
            m["capabilities"] = merged

        if m.get("type") == "embed" and not m.get("dimensions"):
            dim = dev.get("limit", {}).get("output")
            if dim:
                m["dimensions"] = dim

        if dev.get("release_date"):
            m["release_date"] = dev["release_date"]
        if dev.get("knowledge"):
            m["knowledge_cutoff"] = dev["knowledge"]

        enriched += 1

    if enriched:
        log.info("  models.dev: enriched %d/%d models", enriched, len(models))
    return models


def probe_embed_dimensions(provider_id, cls, credgoo_service, model_ids):
    """Probe embedding endpoint to get vector dimensions for each model.

    Only used for providers whose /v1/models returns bare data (no dimensions).
    Currently limited to TU which exposes embed models without dimension info.
    """
    dimensions = {}
    if provider_id != "tu":
        return dimensions

    try:
        from credgoo.credgoo import get_api_key
        api_key = get_api_key(credgoo_service)
    except Exception:
        return dimensions

    import requests
    base_url = getattr(cls, "BASE_URL", None)
    if not base_url:
        return dimensions

    # Strip trailing /v1 to get base, then re-add
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        embed_url = base + "/embeddings"
    else:
        embed_url = base + "/v1/embeddings"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for mid in model_ids:
        try:
            resp = requests.post(embed_url, json={"model": mid, "input": "hello"}, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                emb = data["data"][0]["embedding"]
                dimensions[mid] = len(emb)
                log.info("    %s: dimensions=%d", mid, len(emb))
        except Exception as e:
            log.debug("    %s: probe failed: %s", mid, e)

    return dimensions


def main():
    log.info("Discovering providers...")
    providers = discover_providers()
    log.info("Found %d providers\n", len(providers))

    type_overrides = load_type_overrides()
    log.info("Loaded %d type overrides", len(type_overrides))

    models_dev = fetch_models_dev()
    dev_providers_used = set()

    result = {}
    total_models = 0

    for provider_id, cls, credgoo_service, kind in providers:
        log.info("Fetching %s (%s) ...", provider_id, kind)
        models = fetch_provider_models(provider_id, cls, credgoo_service, kind)
        if models:
            # Apply type overrides (always wins), then derive_type as fallback
            for m in models:
                mid = m["id"]
                if mid in type_overrides:
                    m["type"] = type_overrides[mid]
                elif m.get("type", "chat") == "chat":
                    mi = ModelInfo(id=mid, modalities=m.get("modalities"))
                    derived = mi.derive_type()
                    if derived != "chat":
                        m["type"] = derived

            # Merge models.dev enrichment
            dev_pid = UNIINFER_TO_MODELS_DEV.get(provider_id)
            if dev_pid and dev_pid in models_dev:
                models = merge_models_dev(models, models_dev[dev_pid])
                dev_providers_used.add(dev_pid)

            # Probe embed models for dimensions (TU only)
            embed_ids = [m["id"] for m in models if m.get("type") == "embed"]
            if embed_ids:
                dims = probe_embed_dimensions(provider_id, cls, credgoo_service, embed_ids)
                for m in models:
                    if m["id"] in dims:
                        m["dimensions"] = dims[m["id"]]  # live probe wins over models.dev

            result[provider_id] = {
                "provider_class": cls.__name__,
                "kind": kind,
                "models": models,
            }
            total_models += len(models)
            log.info("  → %d models", len(models))
        else:
            log.info("  → 0 models (skipped)")

    output = {
        "_meta": {
            "version": "1.0.0",
            "generated": datetime.now(timezone.utc).isoformat(),
            "source": "live provider APIs + models.dev",
            "models_dev_providers": sorted(dev_providers_used),
            "total_models": total_models,
            "total_providers": len(result),
        },
        "providers": result,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log.info("\nWrote %s (%d models across %d providers)",
             OUTPUT_PATH, total_models, len(result))

    # Summary table
    log.info("\n{'Provider':<20} {'Kind':<8} {'Models':>6}")
    log.info("-" * 36)
    for pid, pdata in sorted(result.items()):
        log.info("%-20s %-8s %6d", pid, pdata["kind"], len(pdata["models"]))


if __name__ == "__main__":
    main()
