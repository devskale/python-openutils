"""
OpenAI provider implementation.
"""
import requests
from typing import Optional

from ..errors import map_provider_error
from .openai_compatible import OpenAICompatibleChatProvider


class OpenAIProvider(OpenAICompatibleChatProvider):
    """
    Provider for OpenAI API.
    """

    BASE_URL = "https://api.openai.com/v1"
    PROVIDER_ID = "openai"
    ERROR_PROVIDER_NAME = "OpenAI"
    DEFAULT_MODEL = "gpt-3.5-turbo"

    def __init__(self, api_key: Optional[str] = None, organization: Optional[str] = None):
        super().__init__(api_key=api_key, base_url=self.BASE_URL)
        self.organization = organization

    def _get_extra_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        return headers

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """
        List available models from OpenAI using the API.

        Returns:
            list: A list of available model IDs.
        """
        if not api_key:
            raise ValueError("API key is required to list models")

        url = "https://api.openai.com/v1/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                raise map_provider_error(
                    "OpenAI",
                    Exception(f"OpenAI API error: {response.status_code} - {response.text}"),
                    status_code=response.status_code,
                    response_body=response.text,
                )

            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        except Exception as e:
            status_code = getattr(e.response, "status_code", None) if hasattr(e, "response") else None
            response_body = getattr(e.response, "text", None) if hasattr(e, "response") else str(e)
            raise map_provider_error("OpenAI", e, status_code=status_code, response_body=response_body)
