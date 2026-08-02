from __future__ import annotations
"""
StepFun provider implementation.

Access: paid — per-token billing (signup/trial credits only; no persistent
free tier). Grounded in yangmao.ai/models.dev; verified by probe (key returns
402 'exceeded your current quota' once trial credits are exhausted).
"""
from typing import Optional

from ..core import ModelInfo
from .openai_compatible import OpenAICompatibleChatProvider


class StepFunProvider(OpenAICompatibleChatProvider):
    """
    Provider for StepFun API (阶跃星辰).
    """

    BASE_URL = "https://api.stepfun.com/v1"
    PROVIDER_ID = "stepfun"
    CREDGOO_SERVICE = "stepfun"
    ERROR_PROVIDER_NAME = "stepfun"
    DEFAULT_MODEL = "step-1-8k"

    def __init__(self, api_key: Optional[str] = None, base_url: str = BASE_URL, **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)

    @classmethod
    def _model_info(cls, raw: dict) -> ModelInfo:
        # paid per-token (trial credits only; no free tier)
        return ModelInfo(id=raw["id"], owned_by=raw.get("owned_by"), created=raw.get("created"), access="paid", raw=raw)
