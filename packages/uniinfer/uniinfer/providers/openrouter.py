from __future__ import annotations
"""
OpenRouter provider implementation.

OpenRouter is a unified API to access multiple AI models from different providers.
"""
from typing import Any, Optional

from ..core import ModelInfo
from .openai_compatible import OpenAICompatibleChatProvider, openrouter_reasoning_payload


class OpenRouterProvider(OpenAICompatibleChatProvider):
    """
    Provider for OpenRouter API.

    OpenRouter provides a unified interface to access multiple AI models from
    different providers, including Anthropic, OpenAI, and more.
    """

    BASE_URL = "https://openrouter.ai/api/v1"
    PROVIDER_ID = "openrouter"
    CREDGOO_SERVICE = "openrouter"
    ERROR_PROVIDER_NAME = "OpenRouter"
    DEFAULT_MODEL = "moonshotai/moonlight-16b-a3b-instruct:free"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key=api_key, base_url=self.BASE_URL)

    def _get_extra_headers(self) -> dict[str, str]:
        return {
            "HTTP-Referer": "https://github.com/uniinfer",
            "X-Title": "UniInfer",
        }

    def _reasoning_payload(self, reasoning_effort: Optional[str]) -> dict[str, Any]:
        """OpenRouter's reasoning dialect is the ``reasoning`` object."""
        return openrouter_reasoning_payload(reasoning_effort)

    # ----- list_models dialect (mechanics live on OpenAICompatibleChatProvider) -----
    @classmethod
    def _extra_request_headers(cls) -> dict:
        # app attribution for OpenRouter's leaderboard (optional, non-functional)
        return {"HTTP-Referer": "https://github.com/uniinfer", "X-Title": "UniInfer"}

    @classmethod
    def _model_info(cls, raw: dict) -> ModelInfo:
        pricing = raw.get("pricing", {})
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

        arch = raw.get("architecture", {})
        modalities = None
        if arch.get("input_modalities") or arch.get("output_modalities"):
            modalities = {
                "input": arch.get("input_modalities", ["text"]),
                "output": arch.get("output_modalities", ["text"]),
            }

        caps: dict = {}
        sp = raw.get("supported_parameters", [])
        if sp:
            if "tools" in sp or "tool_choice" in sp:
                caps["tool_call"] = True
            if "structured_outputs" in sp:
                caps["structured_outputs"] = True
            if "reasoning" in sp or "include_reasoning" in sp:
                caps["reasoning"] = True
        top_provider = raw.get("top_provider", {}) or {}
        kc = raw.get("knowledge_cutoff")
        if kc:
            caps["knowledge_cutoff"] = kc

        mid = raw["id"]
        # access from the :free convention — OpenRouter's reliable free signal
        # (cost-zero is a TRAP: google/lyria-3-* has pricing 0/0 but is paid,
        # verified via a no-budget key returning 'Insufficient credits').
        # openrouter/free is the virtual free tier.
        access = "free" if (mid.endswith(":free") or mid == "openrouter/free") else "paid"

        return ModelInfo(
            id=mid,
            name=raw.get("name"),
            type="chat",
            context_window=raw.get("context_length"),
            max_output=top_provider.get("max_completion_tokens"),
            cost=cost,
            access=access,
            modalities=modalities,
            capabilities=caps or None,
            owned_by=raw.get("owned_by"),
            created=raw.get("created"),
            raw=raw,
        )
