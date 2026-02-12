"""
NVIDIA GPU Cloud (NGC) provider implementation.
Uses OpenAI-compatible API.
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
    def list_models(cls, api_key: Optional[str] = None, base_url: str = BASE_URL) -> List[str]:
        """List available models from NGC catalog."""
        if api_key is None:
            try:
                from credgoo.credgoo import get_api_key
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
            return [model["id"] for model in models_data.get("data", [])]
        except Exception:
            return []
