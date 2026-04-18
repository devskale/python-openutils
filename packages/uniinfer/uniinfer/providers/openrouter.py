from __future__ import annotations
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
    def list_models(cls, api_key: Optional[str] = None) -> list[ModelInfo]:
        """
        List available models from OpenRouter.

        Args:
            api_key (Optional[str]): The OpenRouter API key. If not provided,
                                     it attempts to retrieve it using credgoo.

        Returns:
            list[ModelInfo]: A list of free model info objects.

        Raises:
            ValueError: If no API key is provided or found.
            Exception: If the API request fails.
        """
        from ..core import ModelInfo
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
            results = []
            for model in models:
                pricing = model.get("pricing", {})
                prompt_price = float(pricing["prompt"]) if pricing.get("prompt") else None
                completion_price = float(pricing["completion"]) if pricing.get("completion") else None
                if prompt_price is not None and prompt_price == 0 and completion_price is not None and completion_price == 0:
                    cost = {"input": 0.0, "output": 0.0}
                elif prompt_price is not None or completion_price is not None:
                    cost = {}
                    if prompt_price is not None:
                        cost["input"] = prompt_price * 1_000_000
                    if completion_price is not None:
                        cost["output"] = completion_price * 1_000_000
                else:
                    cost = None

                arch = model.get("architecture", {})
                modalities = None
                if arch.get("input_modalities") or arch.get("output_modalities"):
                    modalities = {
                        "input": arch.get("input_modalities", ["text"]),
                        "output": arch.get("output_modalities", ["text"]),
                    }

                results.append(ModelInfo(
                    id=model["id"],
                    name=model.get("name"),
                    type="chat",
                    context_window=model.get("context_length"),
                    cost=cost,
                    modalities=modalities,
                    owned_by=model.get("owned_by"),
                    created=model.get("created"),
                    raw=model,
                ))
            return results
        except Exception as e:
            status_code = getattr(e.response, "status_code", None) if hasattr(e, "response") else None
            response_body = getattr(e.response, "text", None) if hasattr(e, "response") else None
            raise map_provider_error("OpenRouter", e, status_code=status_code, response_body=response_body)
