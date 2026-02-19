"""
ArliAI provider implementation.
"""
from typing import Optional

from .openai_compatible import OpenAICompatibleChatProvider


class ArliAIProvider(OpenAICompatibleChatProvider):
    """
    Provider for ArliAI API.
    """

    BASE_URL = "https://api.arliai.com/v1"
    PROVIDER_ID = "arli"
    ERROR_PROVIDER_NAME = "ArliAI"
    DEFAULT_MODEL = "Mistral-Nemo-12B-Instruct-2407"

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the ArliAI provider.

        Args:
            api_key (Optional[str]): The ArliAI API key.
            **kwargs: Additional configuration options.
        """
        super().__init__(api_key=api_key, base_url=self.BASE_URL, **kwargs)

    def _get_default_payload_params(self, stream: bool) -> dict:
        """ArliAI default parameters."""
        params = {
            "repetition_penalty": 1.1,
            "top_p": 0.9,
            "top_k": 40
        }
        if stream:
            params["max_tokens"] = 1024  # Default for streaming if not specified
        return params

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """
        List available models from ArliAI.

        Args:
            api_key (Optional[str]): The ArliAI API key.

        Returns:
            list: A list of available model names.
        """
        if not api_key:
            raise ValueError("API key is required to list models")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        try:
            import requests
            endpoints = [
                "https://api.arliai.com/v1/models/textgen-models",
                "https://api.arliai.com/v1/models",
            ]
            for endpoint in endpoints:
                try:
                    response = requests.get(endpoint, headers=headers, timeout=30)
                    if response.status_code == 200:
                        models_data = response.json()
                        if isinstance(models_data, list):
                            return [m.get("id") or m.get("name") for m in models_data if isinstance(m, dict) and (m.get("id") or m.get("name"))]
                        data = models_data.get("data", models_data)
                        if isinstance(data, list):
                            return [m.get("id") or m.get("name") for m in data if isinstance(m, dict) and (m.get("id") or m.get("name"))]
                except Exception:
                    continue
            return []
        except Exception:
            return []
