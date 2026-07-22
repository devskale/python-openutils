from __future__ import annotations
"""
OpenCode / Zen provider implementation.

OpenCode (opencode.ai) runs the "zen" model router — an OpenAI-compatible
endpoint that aggregates many models (DeepSeek, GPT, Gemini, Qwen, GLM,
MiniMax, Kimi, …). Several are free (id ends in ``-free``, plus ``big-pickle``):
deepseek-v4-flash-free, mimo-v2.5-free, hy3-free, nemotron-3-ultra-free,
north-mini-code-free.

Note: Claude models on OpenCode use the Anthropic-messages API
(``https://opencode.ai/zen``) and are NOT served by this OpenAI-compatible
provider (which targets ``/v1``).
"""
import requests
from typing import Optional

from .openai_compatible import OpenAICompatibleChatProvider


class OpenCodeProvider(OpenAICompatibleChatProvider):
    """Provider for the OpenCode/Zen model router (OpenAI-compatible)."""

    BASE_URL = "https://opencode.ai/zen/v1"
    PROVIDER_ID = "opencode"
    ERROR_PROVIDER_NAME = "OpenCode"
    DEFAULT_MODEL = "deepseek-v4-flash-free"
    CREDGOO_SERVICE = "opencode"
    # The router forwards native OpenAI multimodal content to vision-capable
    # upstreams (e.g. big-pickle, mimo-v2.5-free).
    PRESERVE_MULTIMODAL = True

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key=api_key, base_url=self.BASE_URL)

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list["ModelInfo"]:
        """List models from OpenCode/Zen via pi.dev catalog.

        The native ``/v1/models`` endpoint returns bare IDs with no metadata.
        pi.dev maintains an enriched catalog with context windows, max tokens,
        capabilities, and cost data — pull from there instead.

        Free models (id ends in ``-free``, plus ``big-pickle``) are marked
        cost 0 so the catalog surfaces them as free.
        """
        from ..core import ModelInfo
        try:
            # pi.dev has the enriched OpenCode catalog with context sizes
            r = requests.get("https://pi.dev/api/models/providers/opencode", timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception:
            return []
        out = []
        for mid, m in data.items():
            free = mid.endswith("-free") or mid == "big-pickle"
            out.append(
                ModelInfo(
                    id=mid,
                    name=m.get("name"),
                    owned_by="opencode",
                    context_window=m.get("contextWindow"),
                    max_output=m.get("maxTokens"),
                    cost={"input": 0, "output": 0} if free else m.get("cost"),
                    access="free" if free else m.get("access", "paid"),
                    capabilities=m.get("capabilities"),
                    modalities=m.get("input"),
                    raw=m,
                )
            )
        return out
