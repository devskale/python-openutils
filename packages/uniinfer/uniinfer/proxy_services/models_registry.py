import os
import sys
import time
import logging
import dataclasses
from typing import Any

from starlette.concurrency import run_in_threadpool

logger = logging.getLogger("uniioai_proxy")

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PREDEFINED_MODELS = [
    "mistral@mistral-tiny-latest",
    "ollama@qwen2.5:3b",
    "openrouter@google/gemma-3-12b-it:free",
    "arli@Mistral-Nemo-12B-Instruct-2407",
    "internlm@internlm3-latest",
    "stepfun@step-1-flash",
    "upstage@solar-mini-250401",
    "zai@glm-4.5-flash",
    "zai-code@glm-4.5",
    "ngc@google/gemma-3-27b-it",
    "cohere@command-r",
    "moonshot@kimi-latest",
    "groq@llama3-8b-8192",
    "gemini@models/gemma-3-27b-it",
    "chutes@Qwen/Qwen3-235B-A22B",
    "pollinations@grok",
]


def get_refetch_interval_seconds() -> float:
    raw = os.getenv("REFETCHTIME", "24")
    try:
        hours = float(raw)
        if hours <= 0:
            raise ValueError
    except ValueError:
        logger.warning("Invalid REFETCHTIME value '%s', defaulting to 24 hours", raw)
        hours = 24.0
    return hours * 3600.0


def models_file_is_stale() -> bool:
    models_json = os.path.join(PACKAGE_ROOT, "models", "models.json")
    if not os.path.exists(models_json):
        return True
    age_seconds = time.time() - os.path.getmtime(models_json)
    return age_seconds > get_refetch_interval_seconds()


async def refresh_models_file() -> dict[str, Any]:
    """Re-generate models.json by calling list_models() on all providers."""
    cmd = [sys.executable, "-m", "scripts.generate_models"]

    scripts_dir = os.path.join(PACKAGE_ROOT, "scripts")
    if not os.path.exists(scripts_dir):
        scripts_dir = os.path.join(os.path.dirname(PACKAGE_ROOT), "scripts")

    result = await run_in_threadpool(
        subprocess.run,
        [sys.executable, os.path.join(scripts_dir, "generate_models.py")],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        logger.error("generate_models.py failed: %s", result.stderr)

    return {
        "status": "success" if result.returncode == 0 else "error",
        "message": "Models updated successfully" if result.returncode == 0 else result.stderr,
    }


async def ensure_fresh_models_file() -> None:
    if not models_file_is_stale():
        return
    try:
        await refresh_models_file()
    except Exception as e:
        logger.warning("Failed to refresh stale models file: %s", e)


def list_all_models_from_factories() -> list[dict]:
    """Build a flat OpenAI-compatible model list from models.json."""
    from uniinfer.core import ModelInfo

    models_json = os.path.join(PACKAGE_ROOT, "models", "models.json")
    if not os.path.exists(models_json):
        return [{"id": m, "object": "model", "owned_by": "skaledev"} for m in PREDEFINED_MODELS]

    try:
        import json
        with open(models_json) as f:
            data = json.load(f)
    except Exception as e:
        logger.error("Error reading models.json: %s", e)
        return [{"id": m, "object": "model", "owned_by": "skaledev"} for m in PREDEFINED_MODELS]

    result = []
    for provider_id, provider_data in data.get("providers", {}).items():
        for model in provider_data.get("models", []):
            entry = {
                "id": model["id"],
                "object": "model",
                "owned_by": model.get("owned_by", "skaledev"),
            }
            # Derive type from name/modalities if stored type is generic
            stored_type = model.get("type", "chat")
            mi = ModelInfo(id=model["id"], modalities=model.get("modalities"))
            derived_type = mi.derive_type()
            if derived_type != "chat":
                entry["type"] = derived_type
            elif stored_type and stored_type != "chat":
                entry["type"] = stored_type
            if model.get("context_window"):
                entry["context_window"] = model["context_window"]
            if model.get("max_output"):
                entry["max_output"] = model["max_output"]
            if model.get("type") and model["type"] != "chat":
                entry["type"] = model["type"]
            if model.get("capabilities"):
                entry["capabilities"] = model["capabilities"]
            if model.get("modalities"):
                entry["modalities"] = model["modalities"]
            if model.get("cost"):
                entry["cost"] = model["cost"]
            result.append(entry)
    return result


def update_provider_in_cache(provider_name: str, models_list) -> None:
    """Update a single provider's models in the models.json cache file.

    Called after a live API fetch so subsequent /v1/models calls get fresh data
    for that provider without hitting the API again.
    """
    import json
    from datetime import datetime, timezone
    from uniinfer.core import ModelInfo
    import dataclasses

    models_json = os.path.join(PACKAGE_ROOT, "models", "models.json")

    if not models_list:
        return

    # Serialize ModelInfo objects to dicts
    model_dicts = []
    for m in models_list:
        if isinstance(m, ModelInfo):
            d = {"id": m.id}
            for field in dataclasses.fields(m):
                val = getattr(m, field.name)
                if val is None or field.name in ("id", "raw"):
                    continue
                d[field.name] = val
            model_dicts.append(d)
        else:
            model_dicts.append({"id": str(m)})

    # Load existing cache or create new structure
    data = {}
    if os.path.exists(models_json):
        try:
            with open(models_json) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read models.json for update: %s", e)

    # Ensure structure
    if "_meta" not in data:
        data["_meta"] = {"version": "1.0.0"}
    if "providers" not in data:
        data["providers"] = {}

    data["_meta"]["generated"] = datetime.now(timezone.utc).isoformat()
    data["providers"][provider_name] = {
        "provider_class": "",
        "kind": "chat",
        "models": model_dicts,
    }

    # Recount totals
    total = sum(len(p.get("models", [])) for p in data["providers"].values())
    data["_meta"]["total_models"] = total
    data["_meta"]["total_providers"] = len(data["providers"])

    # Persist
    try:
        os.makedirs(os.path.dirname(models_json), exist_ok=True)
        with open(models_json, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Updated cache for provider %s: %d models", provider_name, len(model_dicts))
    except OSError as e:
        logger.warning("Failed to write models.json: %s", e)


def parse_models_file() -> list[str]:
    """Deprecated: models.txt is no longer used. Returns predefined fallback."""
    return PREDEFINED_MODELS


import subprocess
