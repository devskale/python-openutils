import os
import re
import sys
import time
import logging
import subprocess
from typing import Any

from starlette.concurrency import run_in_threadpool

logger = logging.getLogger("uniioai_proxy")

# The models.txt file is expected to be in the package root (one level up from uniioai_proxy.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UNIINFER_DIR = os.path.dirname(SCRIPT_DIR)
PACKAGE_ROOT = os.path.dirname(UNIINFER_DIR)
MODELS_FILE_PATH = os.path.join(PACKAGE_ROOT, "models.txt")

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
    if not os.path.exists(MODELS_FILE_PATH):
        return True
    age_seconds = time.time() - os.path.getmtime(MODELS_FILE_PATH)
    return age_seconds > get_refetch_interval_seconds()


async def refresh_models_file() -> dict[str, Any]:
    cmd = ["uniinfer", "-l", "--list-models"]

    # Prefer installed CLI; fallback to module execution.
    if not _which("uniinfer"):
        cmd = [sys.executable, "-m", "uniinfer.uniinfer_cli", "-l", "--list-models"]

    result = await run_in_threadpool(
        subprocess.run,
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    with open(MODELS_FILE_PATH, "w") as f:
        f.write(result.stdout)

    return {
        "status": "success",
        "message": "Models updated successfully",
        "output_length": len(result.stdout),
    }


async def ensure_fresh_models_file() -> None:
    if not models_file_is_stale():
        return
    try:
        await refresh_models_file()
    except Exception as e:
        logger.warning("Failed to refresh stale models file: %s", e)


def parse_models_file() -> list[str]:
    """Parse models.txt and return provider@model entries."""
    models: list[str] = []
    if not os.path.exists(MODELS_FILE_PATH):
        return PREDEFINED_MODELS

    current_provider = None
    try:
        with open(MODELS_FILE_PATH, "r") as f:
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


def _which(binary: str) -> bool:
    try:
        import shutil

        return shutil.which(binary) is not None
    except Exception:
        return False
