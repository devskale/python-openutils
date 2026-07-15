"""Generate models.json by calling list_models() on all installed providers."""

import json
import sys
import enum
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
MODEL_HISTORY_PATH = PROJECT_ROOT / "uniinfer" / "models" / "_model_history.json"
SPEED_RESULTS_PATH = PROJECT_ROOT / "uniinfer" / "models" / "_speed_results.json"

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
    "cloudflare": "cloudflare-ai-gateway",
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


def _json_safe(obj):
    """Recursively coerce to JSON-serializable primitives.

    Provider list_models() can leak non-primitive objects into ModelInfo fields
    (e.g. the anthropic SDK's ThinkingCapability in capabilities.thinking). Left
    raw, such objects crash json.dump mid-stream and truncate the catalog.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if dataclasses.is_dataclass(obj):
        return {f.name: _json_safe(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    # Pydantic / arbitrary objects: best-effort dict of public attrs, else str().
    attrs = {k: v for k, v in getattr(obj, "__dict__", {}).items() if not k.startswith("_")}
    if attrs:
        return {k: _json_safe(v) for k, v in attrs.items()}
    return str(obj)


def model_info_to_dict(m) -> dict:
    """Serialize a ModelInfo to a JSON-safe dict."""
    d = {"id": m.id}
    for field in dataclasses.fields(m):
        val = getattr(m, field.name)
        if val is None or field.name == "id":
            continue
        if field.name == "raw":
            continue
        d[field.name] = _json_safe(val)
    return d


def fetch_provider_models(provider_id, cls, credgoo_service, kind):
    """Call list_models() on a provider, return list of dicts."""
    kwargs = {}
    try:
        from credgoo import get_api_key
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

    # Provider-specific extra params for list_models (only infra params, not chat params)
    from uniinfer.config.providers import PROVIDER_CONFIGS
    extra = PROVIDER_CONFIGS.get(provider_id, {}).get("extra_params", {})
    list_models_params = {k: v for k, v in extra.items() if k in ("base_url", "account_id", "api_key")}
    kwargs.update(list_models_params)

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


import re


def _normalize_model_id(mid: str) -> str:
    """Strip date suffixes and prefixes for fuzzy matching.

    Examples:
        gpt-5.4-nano-2026-03-17 → gpt-5.4-nano
        step-2-16k-202411 → step-2-16k
        @cf/baai/bge-m3 → baai/bge-m3
    """
    # Strip @cf/ or workers-ai/@cf/ prefix
    mid = re.sub(r'^(?:workers-ai/)?@cf/', '', mid)
    # Strip date suffixes like -2026-03-17, -202411, -20250605
    mid = re.sub(r'-\d{4}(?:-\d{2}(?:-\d{2})?)?$', '', mid)
    return mid


def _build_dev_lookup(dev_models: dict) -> dict[str, dict]:
    """Build a lookup dict keyed by normalized model ID."""
    lookup = {}
    for mid, data in dev_models.items():
        norm = _normalize_model_id(mid)
        if norm not in lookup:
            lookup[norm] = data
    return lookup


PRUNE_DAYS = 90


def load_model_history() -> dict[str, dict]:
    """Load {provider/model_id: {first_seen, last_seen}} from history file.

    Migrates legacy format (string value) to new object format.
    """
    if MODEL_HISTORY_PATH.exists():
        with open(MODEL_HISTORY_PATH) as f:
            raw = json.load(f)
    else:
        return {}
    migrated = {}
    for key, val in raw.items():
        if isinstance(val, str):
            migrated[key] = {"first_seen": val, "last_seen": val}
        elif isinstance(val, dict) and "first_seen" in val:
            if "last_seen" not in val:
                val["last_seen"] = val["first_seen"]
            migrated[key] = val
        else:
            continue
    return migrated


def save_model_history(history: dict[str, dict]) -> None:
    MODEL_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2, sort_keys=True)


def merge_speed_results(models: list[dict], provider_id: str) -> list[dict]:
    """Merge speed test results from _speed_results.json into model dicts."""
    if not SPEED_RESULTS_PATH.exists():
        return models
    with open(SPEED_RESULTS_PATH) as f:
        speed_data = json.load(f)
    if not speed_data:
        return models
    enriched = 0
    for m in models:
        key = f"{provider_id}/{m['id']}"
        if key in speed_data:
            sr = speed_data[key]
            m["speed"] = sr
            enriched += 1
    if enriched:
        log.info("  speed: merged results for %d/%d models", enriched, len(models))
    return models


def _days_between(d1: str, d2: str) -> int:
    """Days between two YYYY-MM-DD strings."""
    from datetime import datetime
    a = datetime.strptime(d1, "%Y-%m-%d")
    b = datetime.strptime(d2, "%Y-%m-%d")
    return abs((a - b).days)


def update_model_history(
    history: dict[str, dict],
    current_providers: dict[str, dict],
    today: str,
) -> tuple[dict[str, dict], list[dict], list[dict]]:
    """Update history with current models.

    Returns (history, stale_models, pruned_models).

    - Models seen today: last_seen = today
    - New models: first_seen = last_seen = today
    - Models not seen today but seen before: last_seen unchanged (stale)
    - Models not seen for > PRUNE_DAYS: pruned (removed from history)
    """
    current_keys = set()
    for pid, pdata in current_providers.items():
        for m in pdata["models"]:
            key = f"{pid}/{m['id']}"
            current_keys.add(key)
            if key not in history:
                history[key] = {"first_seen": today, "last_seen": today}
            else:
                history[key]["last_seen"] = today

    stale = []
    pruned = []
    for key, entry in list(history.items()):
        if key not in current_keys:
            days_missing = _days_between(entry["last_seen"], today)
            pid, mid = key.split("/", 1)
            if days_missing >= PRUNE_DAYS:
                pruned.append({
                    "id": mid, "provider": pid,
                    "first_seen": entry["first_seen"],
                    "last_seen": entry["last_seen"],
                    "days_missing": days_missing,
                })
                del history[key]
            else:
                stale.append({
                    "id": mid, "provider": pid,
                    "first_seen": entry["first_seen"],
                    "last_seen": entry["last_seen"],
                    "days_missing": days_missing,
                })

    return history, stale, pruned


def merge_models_dev(models: list[dict], dev_provider: dict, type_overrides: dict = None) -> list[dict]:
    """Enrich models with models.dev data.

    Priority: live API data wins over models.dev.
    models.dev fills: name, context_window, max_output, cost, modalities,
    capabilities, dimensions (for embed), release_date, knowledge_cutoff.

    Matching: exact ID first, then normalized (strip date suffixes, @cf/ prefix).
    """
    dev_models = dev_provider.get("models", {})
    dev_lookup = _build_dev_lookup(dev_models)
    enriched = 0
    fuzzy_matched = 0

    for m in models:
        dev = dev_models.get(m["id"])
        if not dev:
            norm = _normalize_model_id(m["id"])
            dev = dev_lookup.get(norm)
            if dev:
                fuzzy_matched += 1
        if not dev:
            continue

        if not m.get("name") and dev.get("name"):
            m["name"] = dev["name"]

        if not m.get("context_window"):
            ctx = dev.get("limit", {}).get("context")
            if ctx and ctx > 0:
                m["context_window"] = ctx

        if not m.get("max_output"):
            out = dev.get("limit", {}).get("output")
            if out and out > 0:
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
            if dim and dim > 0:
                m["dimensions"] = dim
        # Also check type_overrides for embed detection
        if not m.get("dimensions") and type_overrides and type_overrides.get(m["id"]) == "embed":
            dim = dev.get("limit", {}).get("output")
            if dim and dim > 0:
                m["dimensions"] = dim

        if dev.get("release_date"):
            m["release_date"] = dev["release_date"]
        if dev.get("knowledge"):
            m["knowledge_cutoff"] = dev["knowledge"]

        enriched += 1

    if enriched:
        log.info("  models.dev: enriched %d/%d models (exact: %d, fuzzy: %d)",
                 enriched, len(models), enriched - fuzzy_matched, fuzzy_matched)
    return models


def probe_ollama_show_metadata(base_url, api_key, model_ids):
    """Probe Ollama /api/show for context_length and embedding_length.

    Returns dict mapping model_id -> {context_window, dimensions}.
    Only fills values that are missing (zero/None) from live API.
    """
    metadata = {}
    import requests
    show_url = base_url.rstrip("/") + "/api/show"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for mid in model_ids:
        try:
            resp = requests.post(show_url, json={"model": mid}, headers=headers, timeout=15)
            if resp.status_code != 200:
                log.debug("    %s: /api/show returned %d", mid, resp.status_code)
                continue
            mi = resp.json().get("model_info", {})
            if not mi:
                continue
            arch = mi.get("general.architecture", "")
            ctx = mi.get(f"{arch}.context_length") or mi.get("general.context_length")
            dims = mi.get(f"{arch}.embedding_length") or mi.get("general.embedding_length")
            entry = {}
            if ctx and ctx > 0:
                entry["context_window"] = ctx
            if dims and dims > 0:
                entry["dimensions"] = dims
            if entry:
                metadata[mid] = entry
                log.info("    %s: ctx=%s dims=%s", mid, entry.get("context_window", "-"), entry.get("dimensions", "-"))
        except Exception as e:
            log.debug("    %s: /api/show failed: %s", mid, e)

    return metadata


def main():
    log.info("Discovering providers...")
    providers = discover_providers()
    log.info("Found %d providers\n", len(providers))

    type_overrides = load_type_overrides()
    log.info("Loaded %d type overrides", len(type_overrides))

    models_dev = fetch_models_dev()
    dev_providers_used = set()

    model_history = load_model_history()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    new_models_count = 0

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

            # Probe Ollama embed models via /api/show for ctx + dims
            embed_ids = [m["id"] for m in models if m.get("type") == "embed"]
            if embed_ids and provider_id == "ollama":
                from uniinfer.config.providers import PROVIDER_CONFIGS
                ollama_base = PROVIDER_CONFIGS.get("ollama", {}).get("extra_params", {}).get("base_url")
                ollama_key = None
                try:
                    from credgoo import get_api_key
                    ollama_key = get_api_key(credgoo_service)
                except Exception:
                    pass
                if ollama_base:
                    meta = probe_ollama_show_metadata(ollama_base, ollama_key, embed_ids)
                    for m in models:
                        if m["id"] in meta:
                            entry = meta[m["id"]]
                            if not m.get("context_window") and "context_window" in entry:
                                m["context_window"] = entry["context_window"]
                            if not m.get("dimensions") and "dimensions" in entry:
                                m["dimensions"] = entry["dimensions"]

            # Merge speed test results
            models = merge_speed_results(models, provider_id)

            if provider_id in result:
                # Merge models from a second factory (e.g. embed) for the same provider.
                # Deduplicate by model ID, preferring the first (chat) entry.
                existing_ids = {m["id"] for m in result[provider_id]["models"]}
                added = 0
                for m in models:
                    if m["id"] not in existing_ids:
                        result[provider_id]["models"].append(m)
                        existing_ids.add(m["id"])
                        added += 1
                total_models += added
                log.info("  → merged %d new models (total %d)", added, len(result[provider_id]["models"]))
            else:
                result[provider_id] = {
                    "provider_class": cls.__name__,
                    "kind": kind,
                    "models": models,
                }
                total_models += len(models)
                log.info("  → %d models", len(models))
        else:
            log.info("  → 0 models (skipped)")

    # Track first_seen / last_seen, detect stale and pruned models
    model_history, stale_models, pruned_models = update_model_history(model_history, result, today)

    # Apply first_seen + last_seen to all models
    for pid, pdata in result.items():
        for m in pdata["models"]:
            key = f"{pid}/{m['id']}"
            entry = model_history.get(key)
            if entry:
                m["first_seen"] = entry["first_seen"]
                m["last_seen"] = entry["last_seen"]
            else:
                m["first_seen"] = today
                m["last_seen"] = today

    # Check for newly seen models
    new_models = [k for k in {f"{pid}/{m['id']}" for pid, pd in result.items() for m in pd["models"]} if model_history[k]["first_seen"] == today]
    new_models_count = len(new_models)
    if new_models:
        log.info("\nNew models (%d):", new_models_count)
        for k in sorted(new_models)[:20]:
            log.info("  + %s", k)
        if new_models_count > 20:
            log.info("  ... and %d more", new_models_count - 20)

    # Report stale models (missing from provider but within prune window)
    if stale_models:
        log.info("\nStale models (%d, will be pruned after %d days):", len(stale_models), PRUNE_DAYS)
        for sm in sorted(stale_models, key=lambda x: -x["days_missing"])[:20]:
            log.info("  ~ %s/%s (last_seen: %s, %d days ago)",
                     sm["provider"], sm["id"], sm["last_seen"], sm["days_missing"])
        if len(stale_models) > 20:
            log.info("  ... and %d more", len(stale_models) - 20)

    # Report pruned models (gone for > PRUNE_DAYS, removed from history)
    if pruned_models:
        log.info("\nPruned models (%d, removed from catalog):", len(pruned_models))
        for pm in pruned_models[:20]:
            log.info("  x %s/%s (last_seen: %s, %d days ago)",
                     pm["provider"], pm["id"], pm["last_seen"], pm["days_missing"])
        if len(pruned_models) > 20:
            log.info("  ... and %d more", len(pruned_models) - 20)

    # Write stale models list (for /v1/models/stale endpoint)
    stale_output_path = PROJECT_ROOT / "uniinfer" / "models" / "_stale_models.json"
    with open(stale_output_path, "w") as f:
        json.dump(stale_models, f, indent=2, ensure_ascii=False)

    save_model_history(model_history)
    log.info("Saved model history (%d entries)", len(model_history))

    output = {
        "_meta": {
            "version": "1.0.0",
            "generated": datetime.now(timezone.utc).isoformat(),
            "source": "live provider APIs + models.dev",
            "models_dev_providers": sorted(dev_providers_used),
            "total_models": total_models,
            "total_providers": len(result),
            "new_models": new_models_count,
            "stale_models": len(stale_models),
            "pruned_models": len(pruned_models),
            "prune_after_days": PRUNE_DAYS,
        },
        "providers": result,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: stage to a tmp file then rename, so a crash/timeout mid-dump
    # never leaves a truncated catalog (the live file stays intact until done).
    tmp = OUTPUT_PATH.parent / (OUTPUT_PATH.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    tmp.replace(OUTPUT_PATH)

    log.info("\nWrote %s (%d models across %d providers)",
             OUTPUT_PATH, total_models, len(result))

    # Summary table
    log.info("\n{'Provider':<20} {'Kind':<8} {'Models':>6}")
    log.info("-" * 36)
    for pid, pdata in sorted(result.items()):
        log.info("%-20s %-8s %6d", pid, pdata["kind"], len(pdata["models"]))


if __name__ == "__main__":
    main()
