"""Generate models.json by calling list_models() on all installed providers."""

import json
import sys
import dataclasses
import logging
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("generate_models")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "uniinfer" / "models" / "models.json"


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


def main():
    log.info("Discovering providers...")
    providers = discover_providers()
    log.info("Found %d providers\n", len(providers))

    result = {}
    total_models = 0

    for provider_id, cls, credgoo_service, kind in providers:
        log.info("Fetching %s (%s) ...", provider_id, kind)
        models = fetch_provider_models(provider_id, cls, credgoo_service, kind)
        if models:
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
            "source": "live provider APIs",
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
