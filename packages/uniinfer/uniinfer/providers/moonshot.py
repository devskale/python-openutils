from __future__ import annotations
"""
Moonshot provider implementation.
"""
from typing import Optional

from ..core import ModelInfo
from .openai_compatible import OpenAICompatibleChatProvider


class MoonshotProvider(OpenAICompatibleChatProvider):
    """
    Provider for Moonshot AI API.
    """

    ACCESS_TIER = "paid"  # per-token billing, no free tier (web-grounded)
    BASE_URL = "https://api.moonshot.cn/v1"
    PROVIDER_ID = "moonshot"
    CREDGOO_SERVICE = "moonshot"
    ERROR_PROVIDER_NAME = "moonshot"
    DEFAULT_MODEL = "moonshot-v1-8k"

    def __init__(self, api_key: Optional[str] = None, base_url: str = BASE_URL, **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)

    @classmethod
    def _model_info(cls, raw: dict) -> ModelInfo:
        # paid per-token (trial credits; no free tier — Kimi K2's free tier is on Groq)
        caps = {}
        if raw.get("supports_image_in"):
            caps["vision"] = True
        input_mods = ["text"] + (["image"] if caps.get("vision") else [])
        return ModelInfo(
            id=raw["id"], type="chat", access="paid",
            context_window=raw.get("context_length"),
            modalities={"input": input_mods, "output": ["text"]},
            capabilities=caps or None,
            owned_by=raw.get("owned_by"), created=raw.get("created"), raw=raw,
        )
