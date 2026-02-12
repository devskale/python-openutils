"""
Bigmodel (Z.ai) provider implementation.
"""
from typing import Optional

import requests

from .openai_compatible import OpenAICompatibleChatProvider


class BigmodelProvider(OpenAICompatibleChatProvider):
    BASE_URL = "https://api.z.ai/api/paas/v4/"
    PROVIDER_ID = "bigmodel"
    ERROR_PROVIDER_NAME = "Bigmodel"
    DEFAULT_MODEL = "glm-4.7"
    CREDGOO_SERVICE = "bigmodel"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        if not api_key:
            try:
                from credgoo.credgoo import get_api_key
                api_key = get_api_key(self.CREDGOO_SERVICE)
            except Exception:
                api_key = None
        super().__init__(api_key=api_key, base_url=base_url or self.BASE_URL, **kwargs)

    @classmethod
    def list_models(cls, api_key: Optional[str] = None, base_url: Optional[str] = None) -> list[str]:
        if not api_key:
            raise ValueError("API key is required to list models")

        guaranteed_models = ["glm-4.7", "glm-4-flash", "glm-4.5-flash", "glm-4.5"]
        effective_base_url = base_url or cls.BASE_URL
        url = f"{effective_base_url.rstrip('/')}/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            from ..errors import map_provider_error
            raise map_provider_error(
                "Bigmodel",
                Exception(f"Bigmodel API error: {response.status_code} - {response.text}"),
                status_code=response.status_code,
                response_body=response.text,
            )

        data = response.json()
        api_models = [model["id"] for model in data.get("data", [])]
        return list(dict.fromkeys(api_models + guaranteed_models))


class ZAIProvider(BigmodelProvider):
    PROVIDER_ID = "zai"
    ERROR_PROVIDER_NAME = "ZAI"
    DEFAULT_MODEL = "glm-4.7"
    CREDGOO_SERVICE = "zai"


class ZAICodingProvider(BigmodelProvider):
    BASE_URL = "https://api.z.ai/api/coding/paas/v4"
    PROVIDER_ID = "zai-coding"
    ERROR_PROVIDER_NAME = "ZAI-Coding"
    DEFAULT_MODEL = "glm-4.5"
    CREDGOO_SERVICE = "zai-code"
