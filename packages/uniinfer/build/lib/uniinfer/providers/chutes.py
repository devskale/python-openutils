"""
Chutes provider implementation.

Chutes is a unified API to access multiple AI models from different providers.
"""
import requests
from typing import Optional

from .openai_compatible import OpenAICompatibleChatProvider


class ChutesProvider(OpenAICompatibleChatProvider):
    """
    Provider for Chutes API.

    Chutes provides a unified interface to access multiple AI models from
    different providers, including Anthropic, OpenAI, and more.
    """

    BASE_URL = "https://llm.chutes.ai/v1"
    PROVIDER_ID = "chutes"
    ERROR_PROVIDER_NAME = "Chutes"
    DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key=api_key, base_url=self.BASE_URL)

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """List available models from Chutes."""
        if not api_key:
            try:
                from credgoo.credgoo import get_api_key
                api_key = get_api_key("chutes")
            except ImportError:
                return []

        if not api_key:
            return []

        try:
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            response = requests.get("https://llm.chutes.ai/v1/models", headers=headers)
            response.raise_for_status()
            models = response.json().get("data", [])
            return [model["id"] for model in models]
        except Exception:
            return []
