"""
InternLM provider implementation.
"""
from typing import Optional

import requests

from .openai_compatible import OpenAICompatibleChatProvider


class InternLMProvider(OpenAICompatibleChatProvider):
    """
    Provider for InternLM API.
    """

    BASE_URL = "https://chat.intern-ai.org.cn/api/v1"
    PROVIDER_ID = "internlm"
    ERROR_PROVIDER_NAME = "InternLM"
    DEFAULT_MODEL = "internlm3-latest"

    def __init__(self, api_key: Optional[str] = None, base_url: str = BASE_URL, **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)

    def _get_default_payload_params(self, stream: bool) -> dict[str, float | int]:
        return {
            "n": 1,
            "top_p": 0.9,
        }

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """List available models from InternLM."""
        if not api_key:
            from credgoo.credgoo import get_api_key
            api_key = get_api_key("internlm")
            if not api_key:
                return []

        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get("https://chat.intern-ai.org.cn/api/v1/models", headers=headers)
            response.raise_for_status()
            return [model["id"] for model in response.json().get("data", [])]
        except Exception:
            return []
