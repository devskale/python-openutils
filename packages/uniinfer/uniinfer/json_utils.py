import json
import os
import datetime
import logging

logger = logging.getLogger(__name__)

# Save models to a JSON file


def touch_json(json_file='models.json'):
    # ensure directory exists and file is present
    base_dir = os.getenv("UNIINFER_HOME", os.path.expanduser("~/.uniinfer"))
    json_path = os.path.join(base_dir, json_file)

    try:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        if not os.path.exists(json_path):
            with open(json_path, 'w') as f:
                json.dump({}, f)
        # load and return data + path
        with open(json_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
        return data, json_path
    except (PermissionError, OSError) as e:
        logger.warning(
            f"Failed to access {json_path}: {e}. Model tracking will be disabled.")
        return {}, None


def update_models(models, provider_name, json_file='models.json'):
    logger.info(
        f"Updating models for provider: {provider_name} in {json_file}")
    # load or initialize JSON
    existing_models, json_path = touch_json(json_file)

    if json_path is None:
        return

    # ensure top-level structure
    providers = existing_models.get("providers", {})

    # preserve old entries
    old_entries = providers.get(provider_name, {}).get("modellist", [])
    old_map = {e["name"]: e for e in old_entries}

    now = datetime.datetime.now().isoformat()
    model_entries = []
    for m in models:
        model_id = str(m) if hasattr(m, '__str__') and not isinstance(m, str) else m
        if model_id in old_map:
            model_entries.append(old_map[model_id])
        else:
            model_entries.append({"name": model_id, "created": now, "accessed": None})

    providers[provider_name] = {"modellist": model_entries}
    existing_models["providers"] = providers

    # persist back
    try:
        with open(json_path, 'w') as f:
            json.dump(existing_models, f, indent=2)
        logger.info(f"Models saved to {json_path}")
    except (PermissionError, OSError) as e:
        logger.error(f"Failed to save models to {json_path}: {e}")


def update_model_accessed(model_name, provider_name, json_file='models.json'):
    # load or initialize JSON
    existing_models, json_path = touch_json(json_file)

    if json_path is None:
        return

    # ensure top-level structure (setdefault so a new provider/model persists)
    providers = existing_models.setdefault("providers", {})
    provider_data = providers.setdefault(provider_name, {})
    # Catalog key is 'models' (current); read 'modellist' as a legacy fallback,
    # then normalize onto 'models' so any append below persists.
    modellist = provider_data.get("models") or provider_data.get("modellist") or []
    provider_data["models"] = modellist

    now = datetime.datetime.now().isoformat()
    for model_entry in modellist:
        # Match on 'id' (the model identifier, e.g. "deepseek-v4-flash-284b"),
        # with 'name' (display name) as a fallback.
        if model_entry.get("id") == model_name or model_entry.get("name") == model_name:
            model_entry["accessed"] = now
            model_entry["accessed_count"] = model_entry.get("accessed_count", 0) + 1
            break
    else:
        # The model was served successfully but isn't in the catalog yet (e.g. a
        # newly added upstream model). Register it so the catalog stays current
        # and access is tracked from here on — instead of warning every request.
        modellist.append({
            "id": model_name,
            "type": "chat",
            "status": "active",
            "owned_by": provider_name,
            "first_seen": now,
            "accessed": now,
            "accessed_count": 1,
        })
        logger.info("Registered new model '%s' (provider '%s') into the catalog.", model_name, provider_name)

    # Persist back
    try:
        with open(json_path, 'w') as f:
            json.dump(existing_models, f, indent=2)
    except (PermissionError, OSError) as e:
        logger.error(f"Failed to update model access in {json_path}: {e}")
