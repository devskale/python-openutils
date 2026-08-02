from __future__ import annotations
"""
Upstage provider implementation.
"""
from typing import Optional

from ..core import ModelInfo
from .openai_compatible import OpenAICompatibleChatProvider


class UpstageProvider(OpenAICompatibleChatProvider):
    """
    Provider for Upstage AI Solar API.
    """

    BASE_URL = "https://api.upstage.ai/v1/solar"
    PROVIDER_ID = "upstage"
    CREDGOO_SERVICE = "upstage"
    ERROR_PROVIDER_NAME = "upstage"
    DEFAULT_MODEL = "solar-pro"

    def __init__(self, api_key: Optional[str] = None, base_url: str = BASE_URL, **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)

    @classmethod
    def _models_url(cls, base_url: str) -> str:
        # BASE_URL is /v1/solar (completions); the models endpoint is /v1/models
        return "https://api.upstage.ai/v1/models"

    @classmethod
    def _model_info(cls, raw: dict) -> ModelInfo:
        # paid per-token
        return ModelInfo(id=raw["id"], owned_by=raw.get("owned_by"), created=raw.get("created"), access="paid", raw=raw)
