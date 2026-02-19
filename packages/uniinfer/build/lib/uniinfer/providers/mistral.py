"""
Mistral AI provider implementation.
"""
import requests
from typing import Optional

from ..errors import map_provider_error
from .openai_compatible import OpenAICompatibleChatProvider


class MistralProvider(OpenAICompatibleChatProvider):
    """
    Provider for Mistral AI API.
    """

    BASE_URL = "https://api.mistral.ai/v1"
    PROVIDER_ID = "mistral"
    ERROR_PROVIDER_NAME = "Mistral"
    DEFAULT_MODEL: str | None = None

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key=api_key, base_url=self.BASE_URL)

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """
        List available models from Mistral AI.

        Args:
            api_key (Optional[str]): The Mistral API key. If not provided,
                                     it attempts to retrieve it using credgoo.

        Returns:
            list: A list of available model names.

        Raises:
            ValueError: If no API key is provided or found.
            Exception: If the API request fails.
        """
        if not api_key:
            try:
                from credgoo.credgoo import get_api_key
                api_key = get_api_key("mistral")
            except ImportError:
                raise ValueError("credgoo not installed. Please provide an API key or install credgoo.")
            except Exception as e:
                raise ValueError(f"Failed to get Mistral API key from credgoo: {e}")

        if not api_key:
            raise ValueError("Mistral API key is required. Provide it directly or configure credgoo.")

        endpoint = "https://api.mistral.ai/v1/models"
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        response = requests.get(endpoint, headers=headers)

        if response.status_code != 200:
            raise map_provider_error(
                "Mistral",
                Exception(f"Failed to fetch models: {response.status_code} - {response.text}"),
                status_code=response.status_code,
                response_body=response.text,
            )

        models_data = response.json()
        return [model["id"] for model in models_data["data"]]
