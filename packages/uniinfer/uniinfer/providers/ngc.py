from __future__ import annotations
"""
NVIDIA GPU Cloud (NGC) provider implementation.
Uses OpenAI-compatible API.

Access: universally free — per build.nvidia.com FAQ, "All models are free to
prototype with": the free tier runs on rate limits (~40 RPM, most models), no
per-token billing. The /models API carries no pricing (id/owned_by/created
only), so all models are tagged access='free'.
"""
from typing import Optional, List

import requests

from .openai_compatible import OpenAICompatibleChatProvider


class NGCProvider(OpenAICompatibleChatProvider):
    """
    Provider for NVIDIA GPU Cloud (NGC) API.
    NGC provides an OpenAI-compatible API for various models.
    """

    BASE_URL = "https://integrate.api.nvidia.com/v1"
    PROVIDER_ID = "ngc"
    ERROR_PROVIDER_NAME = "ngc"
    DEFAULT_MODEL: str | None = None

    def __init__(self, api_key: Optional[str] = None, base_url: str = BASE_URL, **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)

    @classmethod
    def list_models(cls, api_key: Optional[str] = None, base_url: str = BASE_URL) -> list[ModelInfo]:
        from ..core import ModelInfo
        """List available models from NGC catalog."""
        if api_key is None:
            try:
                from credgoo import get_api_key
                api_key = get_api_key("ngc")
            except ImportError:
                return []

        if not api_key:
            return []

        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(f"{base_url}/models", headers=headers)
            response.raise_for_status()
            models_data = response.json()
            return [ModelInfo(id=model["id"], owned_by=model.get("owned_by"), access="free", created=model.get("created"), raw=model) for model in models_data.get("data", [])]
        except Exception:
            return []
