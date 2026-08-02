from __future__ import annotations
"""
Chutes provider implementation.

Chutes is a unified API to access multiple AI models from different providers.

Access: PAYG (pay-per-token, no markup, no free tier) — every model carries a
non-zero `price` in USD + TAO. Derived from pricing: 0/0 -> free, else paid
(all current models are paid). Grounded in chutes.ai/pricing; verified by probe
(a $0-balance key returns 402 'account balance is $0.0, pay with fiat or tao').
"""
import requests
from typing import Optional

from .openai_compatible import OpenAICompatibleChatProvider


class ChutesProvider(OpenAICompatibleChatProvider):
    """
    Provider for Chutes API.

    Chutes provides a unified interface to access multiple AI models from
    different providers, including Anthropic, OpenAI, and more.
    """

    BASE_URL = "https://llm.chutes.ai/v1"
    PROVIDER_ID = "chutes"
    ERROR_PROVIDER_NAME = "Chutes"
    DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731-TEE"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key=api_key, base_url=self.BASE_URL)

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list[ModelInfo]:
        from ..core import ModelInfo
        """List available models from Chutes."""
        if not api_key:
            try:
                from credgoo import get_api_key
                api_key = get_api_key("chutes")
            except ImportError:
                return []

        if not api_key:
            return []

        try:
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            response = requests.get("https://llm.chutes.ai/v1/models", headers=headers)
            response.raise_for_status()
            models = response.json().get("data", [])
            out = []
            for model in models:
                price = model.get("price") or {}
                in_usd = (price.get("input") or {}).get("usd")
                out_usd = (price.get("output") or {}).get("usd")
                is_free = in_usd == 0 and out_usd == 0
                cost = {"input": in_usd, "output": out_usd} if (in_usd is not None or out_usd is not None) else None  # per-million USD
                out.append(ModelInfo(
                    id=model["id"],
                    owned_by=model.get("owned_by"),
                    access="free" if is_free else "paid",
                    cost=cost,
                    raw=model,
                ))
            return out
        except Exception:
            return []
