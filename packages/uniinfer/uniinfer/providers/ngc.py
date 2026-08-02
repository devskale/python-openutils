from __future__ import annotations
"""
NVIDIA GPU Cloud (NGC) provider implementation.
Uses OpenAI-compatible API.

Access: universally free — per build.nvidia.com FAQ, "All models are free to
prototype with": the free tier runs on rate limits (~40 RPM, most models), no
per-token billing. The /models API carries no pricing (id/owned_by/created
only), so all models are tagged access='free'.
"""
from typing import Optional

from ..core import ModelInfo
from .openai_compatible import OpenAICompatibleChatProvider


class NGCProvider(OpenAICompatibleChatProvider):
    """
    Provider for NVIDIA GPU Cloud (NGC) API.
    NGC provides an OpenAI-compatible API for various models.
    """

    BASE_URL = "https://integrate.api.nvidia.com/v1"
    PROVIDER_ID = "ngc"
    CREDGOO_SERVICE = "ngc"
    ERROR_PROVIDER_NAME = "ngc"
    DEFAULT_MODEL: str | None = None

    def __init__(self, api_key: Optional[str] = None, base_url: str = BASE_URL, **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)

    @classmethod
    def _model_info(cls, raw: dict) -> ModelInfo:
        # universally free (build.nvidia.com FAQ); the API carries no pricing
        return ModelInfo(id=raw["id"], owned_by=raw.get("owned_by"), access="free", created=raw.get("created"), raw=raw)
