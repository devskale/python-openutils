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


def parse_models_file() -> list[str]:
    """Legacy: parse models.txt and return provider@model entries.

    Used only as fallback when models.json doesn't exist.
    """
    models_txt = os.path.join(PACKAGE_ROOT, "models.txt")
    models: list[str] = []
    if not os.path.exists(models_txt):
        return PREDEFINED_MODELS

    current_provider = None
    try:
        import re
        with open(models_txt, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                provider_match = re.match(r"Available models for (\w+):", line)
                if provider_match:
                    current_provider = provider_match.group(1)
                    continue
                if line.startswith("- ") and current_provider:
                    model_name = line[2:].strip()
                    models.append(f"{current_provider}@{model_name}")
    except Exception as e:
        logger.error("Error reading models file: %s", e)
        return PREDEFINED_MODELS

    return models if models else PREDEFINED_MODELS


import subprocess
