"""
OpenRouter provider implementation.

OpenRouter is a unified API to access multiple AI models from different providers.
"""
import requests
from typing import Optional

from ..errors import map_provider_error
from .openai_compatible import OpenAICompatibleChatProvider


class OpenRouterProvider(OpenAICompatibleChatProvider):
    """
    Provider for OpenRouter API.

    OpenRouter provides a unified interface to access multiple AI models from
    different providers, including Anthropic, OpenAI, and more.
    """

    BASE_URL = "https://openrouter.ai/api/v1"
    PROVIDER_ID = "openrouter"
    ERROR_PROVIDER_NAME = "OpenRouter"
    DEFAULT_MODEL = "moonshotai/moonlight-16b-a3b-instruct:free"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key=api_key, base_url=self.BASE_URL)

    def _get_extra_headers(self) -> dict[str, str]:
        return {
            "HTTP-Referer": "https://github.com/uniinfer",
            "X-Title": "UniInfer",
        }

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """
        List available models from OpenRouter.

        Args:
            api_key (Optional[str]): The OpenRouter API key. If not provided,
                                     it attempts to retrieve it using credgoo.

        Returns:
            list: A list of available free model IDs.

        Raises:
            ValueError: If no API key is provided or found.
            Exception: If the API request fails.
        """
        if not api_key:
            try:
                from credgoo.credgoo import get_api_key
                api_key = get_api_key("openrouter")
            except ImportError:
                raise ValueError("credgoo not installed. Please provide an API key or install credgoo.")
            except Exception as e:
                raise ValueError(f"Failed to get OpenRouter API key from credgoo: {e}")

        if not api_key:
            raise ValueError("OpenRouter API key is required. Provide it directly or configure credgoo.")

        endpoint = "https://openrouter.ai/api/v1/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/uniinfer",
            "X-Title": "UniInfer",
        }

        try:
            response = requests.get(endpoint, headers=headers)

            if response.status_code != 200:
                error_msg = f"OpenRouter API error: {response.status_code} - {response.text}"
                raise map_provider_error(
                    "OpenRouter",
                    Exception(error_msg),
                    status_code=response.status_code,
                    response_body=response.text,
                )

            models = response.json().get("data", [])
            free_models = [
                model["id"]
                for model in models
                if model.get("pricing", {}).get("prompt") == "0"
                and model.get("pricing", {}).get("completion") == "0"
            ]
            return free_models
        except Exception as e:
            status_code = getattr(e.response, "status_code", None) if hasattr(e, "response") else None
            response_body = getattr(e.response, "text", None) if hasattr(e, "response") else None
            raise map_provider_error("OpenRouter", e, status_code=status_code, response_body=response_body)
