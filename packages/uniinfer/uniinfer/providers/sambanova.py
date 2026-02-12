"""
SambaNova provider implementation.
"""
from typing import List, Optional

import requests

from ..errors import map_provider_error
from .openai_compatible import OpenAICompatibleChatProvider


class SambanovaProvider(OpenAICompatibleChatProvider):
    """
    Provider for SambaNova AI API.
    """

    BASE_URL = "https://api.sambanova.ai/v1"
    PROVIDER_ID = "sambanova"
    ERROR_PROVIDER_NAME = "sambanova"
    DEFAULT_MODEL = "Meta-Llama-3.1-8B-Instruct"

    def __init__(self, api_key: Optional[str] = None, base_url: str = BASE_URL, **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)

    @classmethod
    def list_models(cls, api_key: Optional[str] = None, base_url: str = BASE_URL) -> List[str]:
        """
        List available models from SambaNova.
        """
        if api_key is None:
            try:
                from credgoo.credgoo import get_api_key
                api_key = get_api_key("sambanova")
            except ImportError:
                api_key = None

        if api_key is None:
            return [
                "Meta-Llama-3.1-8B-Instruct",
                "sambastudio-7b",
                "sambastudio-13b",
                "sambastudio-20b",
                "sambastudio-70b",
            ]

        try:
            response = requests.get(
                f"{base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30,
            )
            if response.status_code != 200:
                raise map_provider_error(
                    "sambanova",
                    Exception(f"SambaNova API error: {response.status_code} - {response.text}"),
                    status_code=response.status_code,
                    response_body=response.text,
                )
            data = response.json()
            return [model["id"] for model in data.get("data", []) if isinstance(model, dict) and model.get("id")]
        except Exception:
            return [
                "Meta-Llama-3.1-8B-Instruct",
                "sambastudio-7b",
                "sambastudio-13b",
                "sambastudio-20b",
                "sambastudio-70b",
            ]
