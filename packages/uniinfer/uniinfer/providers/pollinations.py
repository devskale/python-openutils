"""
Pollinations provider implementation.
"""
from typing import Optional

import requests

from ..errors import map_provider_error
from .openai_compatible import OpenAICompatibleChatProvider


class PollinationsProvider(OpenAICompatibleChatProvider):
    """
    Provider for Pollinations OpenAI-compatible API.
    """

    BASE_URL = "https://text.pollinations.ai/openai"
    PROVIDER_ID = "pollinations"
    ERROR_PROVIDER_NAME = "Pollinations"
    DEFAULT_MODEL = "openai"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(api_key=api_key, base_url=base_url or self.BASE_URL)

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list[str]:
        """
        List available models from Pollinations.
        """
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        endpoints = [
            "https://text.pollinations.ai/openai/v1/models",
            "https://gen.pollinations.ai/openai/v1/models",
            "https://gen.pollinations.ai/models",
        ]

        last_error = None
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, headers=headers, timeout=30)
                if response.status_code != 200:
                    last_error = map_provider_error(
                        "Pollinations",
                        Exception(f"Pollinations API error: {response.status_code} - {response.text}"),
                        status_code=response.status_code,
                        response_body=response.text,
                    )
                    continue

                data = response.json()
                if isinstance(data, dict) and "data" in data:
                    return [m.get("id", "") for m in data["data"] if isinstance(m, dict) and m.get("id")]
                if isinstance(data, list):
                    return [m.get("name", "") for m in data if isinstance(m, dict) and m.get("name")]
                return []
            except Exception as e:
                last_error = e

        if last_error:
            raise map_provider_error("Pollinations", last_error)
        return []
