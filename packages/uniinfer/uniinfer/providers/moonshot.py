"""
Moonshot provider implementation.
"""
from typing import Optional

import requests

from .openai_compatible import OpenAICompatibleChatProvider


class MoonshotProvider(OpenAICompatibleChatProvider):
    """
    Provider for Moonshot AI API.
    """

    BASE_URL = "https://api.moonshot.cn/v1"
    PROVIDER_ID = "moonshot"
    ERROR_PROVIDER_NAME = "moonshot"
    DEFAULT_MODEL = "moonshot-v1-8k"

    def __init__(self, api_key: Optional[str] = None, base_url: str = BASE_URL, **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """List available models from Moonshot AI."""
        if api_key is None:
            try:
                from credgoo.credgoo import get_api_key
                api_key = get_api_key("moonshot")
            except ImportError:
                return []

        if not api_key:
            return []

        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get("https://api.moonshot.cn/v1/models", headers=headers)
            response.raise_for_status()
            models_data = response.json()
            return [model["id"] for model in models_data.get("data", [])]
        except Exception:
            return []
