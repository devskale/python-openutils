from __future__ import annotations
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
    def list_models(cls, api_key: Optional[str] = None) -> list[ModelInfo]:
        """
        List available models from ArliAI.

        Args:
            api_key (Optional[str]): The ArliAI API key.

        Returns:
            list[ModelInfo]: A list of model info objects.
        """
        from ..core import ModelInfo
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
                        data = models_data
                        if isinstance(models_data, dict):
                            data = models_data.get("data", [])
                        if not isinstance(data, list):
                            continue
                        results = []
                        for m in data:
                            if not isinstance(m, dict):
                                continue
                            mid = m.get("id") or m.get("name")
                            if not mid:
                                continue
                            capabilities = {}
                            if m.get("reasoning"):
                                capabilities["reasoning"] = True
                            if m.get("vlm"):
                                capabilities["vision"] = True
                            input_mods = ["text"]
                            if capabilities.get("vision"):
                                input_mods.append("image")
                            results.append(ModelInfo(
                                id=mid,
                                type="chat",
                                context_window=m.get("max_context"),
                                modalities={"input": input_mods, "output": ["text"]},
                                capabilities=capabilities or None,
                                owned_by=m.get("owned_by"),
                                raw=m,
                            ))
                        if results:
                            return results
                except Exception:
                    continue
            return []
        except Exception:
            return []
