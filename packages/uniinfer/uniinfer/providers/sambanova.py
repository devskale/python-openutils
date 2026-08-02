from __future__ import annotations
"""
SambaNova provider implementation.

Access: free-tier-reachable — SambaNova Cloud's free tier reaches all models
on a balance_units budget (rate-limited: ~20 RPM/20 RPD/200K TPD, no CC), like
HF/pollinations (depletes, then 402 'balance_units: 0'). The API's `pricing`
field is the PAID tier (per-token $), NOT an access signal — so all models are
tagged access='free'. Grounded in ayautomate/toolfreebie; verified by probe.
"""
from typing import Optional

from ..core import ModelInfo
from .openai_compatible import OpenAICompatibleChatProvider


class SambanovaProvider(OpenAICompatibleChatProvider):
    """
    Provider for SambaNova AI API.
    """

    BASE_URL = "https://api.sambanova.ai/v1"
    PROVIDER_ID = "sambanova"
    CREDGOO_SERVICE = "sambanova"
    ERROR_PROVIDER_NAME = "sambanova"
    DEFAULT_MODEL = "Meta-Llama-3.1-8B-Instruct"

    def __init__(self, api_key: Optional[str] = None, base_url: str = BASE_URL, **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)

    @classmethod
    def _model_info(cls, raw: dict) -> ModelInfo:
        # free-tier-reachable (balance_units budget); API `pricing` is the paid tier — ignored
        return ModelInfo(
            id=raw["id"],
            type="chat",
            context_window=raw.get("context_length"),
            max_output=raw.get("max_completion_tokens"),
            access="free",
            owned_by=raw.get("owned_by"),
            raw=raw,
        )
