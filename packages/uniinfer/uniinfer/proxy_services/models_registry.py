import os
import sys
import time
import asyncio
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


_refresh_lock = asyncio.Lock()


async def refresh_models_file() -> dict[str, Any]:
    """Re-generate models.json (single-flight: only one refresh at a time).

    The lock stops concurrent /v1/models requests (and the manual refresh
    endpoint) from each spawning their own generate_models.py subprocess, which
    would stack memory on small hosts. generate_models.py also holds its own
    flock, so even an overlap with the systemd timer is a harmless no-op.
    """
    if _refresh_lock.locked():
        return {"status": "skipped", "message": "A models refresh is already in progress"}
    async with _refresh_lock:
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
            return {"status": "error", "message": result.stderr}
        return {"status": "success", "message": "Models updated successfully"}


async def ensure_fresh_models_file() -> None:
    if not models_file_is_stale():
        return
    if _refresh_lock.locked():
        # A refresh is already running (on-demand or timer); don't stack spawns.
        return
    try:
        await refresh_models_file()
    except Exception as e:
        logger.warning("Failed to refresh stale models file: %s", e)


def _overrides_file() -> str:
    return os.path.join(PACKAGE_ROOT, "models", "model_overrides.json")


def load_model_overrides() -> dict:
    """Load all model overrides from model_overrides.json."""
    path = _overrides_file()
    if not os.path.exists(path):
        return {"models": {}}
    try:
        import json
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {"models": {}}


def save_model_override(model_id: str, fields: dict) -> None:
    """Save fields for a model into model_overrides.json.
    
    Also updates type_overrides.json if 'type' field is present.
    """
    import json
    from datetime import datetime, timezone

    data = load_model_overrides()
    if model_id not in data["models"]:
        data["models"][model_id] = {}
    data["models"][model_id].update(fields)
    data["models"][model_id]["_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    os.makedirs(os.path.dirname(_overrides_file()), exist_ok=True)
    with open(_overrides_file(), "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # If type changed, also update type_overrides.json
    if "type" in fields:
        type_path = os.path.join(PACKAGE_ROOT, "models", "type_overrides.json")
        try:
            with open(type_path) as f:
                td = json.load(f)
        except Exception:
            td = {"_meta": {"description": "Curated model type assignments."}, "models": {}}
        td["models"][model_id] = fields["type"]
        with open(type_path, "w") as f:
            json.dump(td, f, indent=2, ensure_ascii=False)

    logger.info("Saved override for %s: %s", model_id, list(fields.keys()))


def delete_model_override(model_id: str) -> bool:
    """Delete all overrides for a model. Returns True if anything was deleted."""
    import json
    data = load_model_overrides()
    if model_id not in data["models"]:
        return False
    del data["models"][model_id]
    with open(_overrides_file(), "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return True


def load_catalog(provider_filter: str | None = None) -> dict:
    """Load the raw nested models.json catalog, optionally filtered by provider(s).

    Unlike list_all_models_from_factories (which flattens into an OpenAI-style
    list and merges overrides), this returns the catalog in its native nested
    shape: {"_meta": {...}, "providers": {pid: {provider_class, kind, models}}}.

    Args:
        provider_filter: Comma-separated provider IDs to include.
            None or empty string returns all providers.

    Returns:
        Dict with '_meta' and 'providers' keys. Meta totals are recomputed
        to reflect the filtered subset.
    """
    import json

    models_json = os.path.join(PACKAGE_ROOT, "models", "models.json")
    empty = {
        "_meta": {"generated": None, "total_models": 0, "total_providers": 0},
        "providers": {},
    }
    if not os.path.exists(models_json):
        return empty
    try:
        with open(models_json) as f:
            data = json.load(f)
    except Exception as e:
        logger.error("Error reading models.json: %s", e)
        return empty

    providers = data.get("providers", {})
    if provider_filter:
        wanted = {p.strip() for p in provider_filter.split(",") if p.strip()}
        providers = {k: v for k, v in providers.items() if k in wanted}

    total = sum(len(p.get("models", [])) for p in providers.values())
    meta = dict(data.get("_meta", {}))
    meta["total_models"] = total
    meta["total_providers"] = len(providers)
    meta["filtered"] = bool(provider_filter)
    return {"_meta": meta, "providers": providers}


def load_stale_models() -> list[dict]:
    """Load stale models from _stale_models.json (written by generate_models.py)."""
    path = os.path.join(PACKAGE_ROOT, "models", "_stale_models.json")
    if not os.path.exists(path):
        return []
    try:
        import json
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


def list_all_models_from_factories() -> list[dict]:
    """Build a flat OpenAI-compatible model list from models.json.

    Each model gets a 'freshness' field: 'fresh' (last_seen == generated date)
    or 'stale' (last_seen < generated date).
    """
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

    # Load type overrides (authoritative type database)
    type_overrides = {}
    overrides_path = os.path.join(PACKAGE_ROOT, "models", "type_overrides.json")
    try:
        import json
        with open(overrides_path) as f:
            type_overrides = json.load(f).get("models", {})
    except Exception:
        pass

    # Load model overrides (context, dimensions, cost, etc.)
    model_overrides = {}
    mo_path = os.path.join(PACKAGE_ROOT, "models", "model_overrides.json")
    try:
        import json
        with open(mo_path) as f:
            model_overrides = json.load(f).get("models", {})
    except Exception:
        pass

    result = []
    for provider_id, provider_data in data.get("providers", {}).items():
        for model in provider_data.get("models", []):
            override = model_overrides.get(model["id"], {})
            entry = {
                "id": model["id"],
                "object": "model",
                "owned_by": model.get("owned_by", "skaledev"),
                "provider": provider_id,
            }
            # Resolve type: overrides > type_overrides > stored > derive
            model_type = None
            if override.get("type"):
                model_type = override["type"]
            elif model["id"] in type_overrides:
                model_type = type_overrides[model["id"]]
            elif model.get("type") and model["type"] != "chat":
                model_type = model["type"]
            if not model_type:
                mi = ModelInfo(id=model["id"], modalities=model.get("modalities"))
                model_type = mi.derive_type()
            entry["type"] = model_type
            # Fields: override > models.json > skip
            for field in ("context_window", "max_output", "dimensions", "cost", "capabilities",
                          "modalities", "first_seen", "deprecation_date", "deprecation_replacement",
                          "status", "release_date", "knowledge_cutoff", "name", "speed"):
                val = override.get(field) if field in override else model.get(field)
                if val is not None:
                    entry[field] = val

            # Freshness: based on last_seen vs models.json generated date
            generated_date = data.get("_meta", {}).get("generated", "")[:10]
            last_seen = model.get("last_seen") or generated_date
            entry["last_seen"] = last_seen
            if last_seen == generated_date:
                entry["freshness"] = "fresh"
                entry["days_since_seen"] = 0
            else:
                try:
                    from datetime import datetime
                    days = (datetime.strptime(generated_date, "%Y-%m-%d")
                            - datetime.strptime(last_seen, "%Y-%m-%d")).days
                except Exception:
                    days = 0
                entry["freshness"] = "stale"
                entry["days_since_seen"] = days

            result.append(entry)
    return result


def update_provider_in_cache(provider_name: str, models_list) -> None:
    """Update a single provider's models in the models.json cache file.

    Called after a live API fetch so subsequent /v1/models calls get fresh data
    for that provider without hitting the API again. Preserves first_seen/last_seen
    from existing cache entries and _model_history.json.
    """
    import json
    from datetime import datetime, timezone
    from uniinfer.core import ModelInfo
    import dataclasses

    models_json = os.path.join(PACKAGE_ROOT, "models", "models.json")
    history_path = os.path.join(PACKAGE_ROOT, "models", "_model_history.json")

    if not models_list:
        return

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Load existing cache to preserve first_seen/last_seen
    existing_models = {}
    if os.path.exists(models_json):
        try:
            with open(models_json) as f:
                data = json.load(f)
            for m in data.get("providers", {}).get(provider_name, {}).get("models", []):
                existing_models[m["id"]] = m
        except (json.JSONDecodeError, OSError):
            data = {}
    else:
        data = {}

    # Load model history for first_seen on truly new models
    history = {}
    if os.path.exists(history_path):
        try:
            with open(history_path) as f:
                raw = json.load(f)
            for k, v in raw.items():
                if isinstance(v, dict) and "first_seen" in v:
                    history[k] = v
        except Exception:
            pass

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

    # Merge first_seen/last_seen into each model
    for m in model_dicts:
        mid = m["id"]
        # 1. Preserve from existing cache entry
        old = existing_models.get(mid, {})
        if old.get("first_seen"):
            m["first_seen"] = old["first_seen"]
            m["last_seen"] = today
        else:
            # 2. Fall back to _model_history.json
            hkey = f"{provider_name}/{mid}"
            hentry = history.get(hkey)
            if hentry:
                m["first_seen"] = hentry["first_seen"]
                m["last_seen"] = today
            else:
                # 3. Brand new model — record first_seen as today
                m["first_seen"] = today
                m["last_seen"] = today

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
            # Compact JSON: machine-read catalog, not hand-edited.
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        logger.info("Updated cache for provider %s: %d models", provider_name, len(model_dicts))
    except OSError as e:
        logger.warning("Failed to write models.json: %s", e)


def parse_models_file() -> list[str]:
    """Deprecated: models.txt is no longer used. Returns predefined fallback."""
    return PREDEFINED_MODELS


import subprocess
