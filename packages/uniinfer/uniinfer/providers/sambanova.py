from __future__ import annotations
"""
SambaNova provider implementation.

Access: free-tier-reachable — SambaNova Cloud's free tier reaches all models
on a balance_units budget (rate-limited: ~20 RPM/20 RPD/200K TPD, no CC), like
HF/pollinations (depletes, then 402 'balance_units: 0'). The API's `pricing`
field is the PAID tier (per-token $), NOT an access signal — so all models are
tagged access='free'. Grounded in ayautomate/toolfreebie; verified by probe.
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
    def list_models(cls, api_key: Optional[str] = None, base_url: str = BASE_URL) -> list[ModelInfo]:
        from ..core import ModelInfo
        """List available models from SambaNova."""
        if api_key is None:
            try:
                from credgoo import get_api_key
                api_key = get_api_key("sambanova")
            except ImportError:
                api_key = None

        if api_key is None:
            return []

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
            results = []
            for model in data.get("data", []):
                if not isinstance(model, dict) or not model.get("id"):
                    continue
                pricing = model.get("pricing", {})
                # SambaNova's free tier reaches all models (rate-limited: 20 RPM/20 RPD/200K
                # TPD) on a balance_units budget; the API `pricing` is the PAID tier (per-
                # token $), irrelevant to free access. So access='free', cost=None (free).
                results.append(ModelInfo(
                    id=model["id"],
                    type="chat",
                    context_window=model.get("context_length"),
                    max_output=model.get("max_completion_tokens"),
                    access="free",
                    owned_by=model.get("owned_by"),
                    raw=model,
                ))
            return results
        except Exception:
            return []
