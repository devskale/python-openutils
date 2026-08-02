from __future__ import annotations
"""
ArliAI provider implementation.
"""
from typing import Optional

from ..core import ModelInfo
from .openai_compatible import OpenAICompatibleChatProvider


class ArliAIProvider(OpenAICompatibleChatProvider):
    """
    Provider for ArliAI API.
    """

    BASE_URL = "https://api.arliai.com/v1"
    PROVIDER_ID = "arli"
    CREDGOO_SERVICE = "arli"
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
    def _model_info(cls, raw: dict) -> ModelInfo:
        mid = raw.get("id") or raw.get("name")
        caps: dict = {}
        if raw.get("reasoning"):
            caps["reasoning"] = True
        if raw.get("vlm"):
            caps["vision"] = True
        input_mods = ["text"] + (["image"] if caps.get("vision") else [])
        # (TRIAL)-prefixed = what a trial key reaches (tagged paid); bare = dead trials (free)
        return ModelInfo(
            id=mid, type="chat",
            access="paid" if mid.startswith("(TRIAL)") else "free",
            context_window=raw.get("max_context"),
            modalities={"input": input_mods, "output": ["text"]},
            capabilities=caps or None, owned_by=raw.get("owned_by"), raw=raw,
        )
