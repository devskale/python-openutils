from __future__ import annotations
"""
InternLM provider implementation.
"""
from typing import Optional

from .openai_compatible import OpenAICompatibleChatProvider


class InternLMProvider(OpenAICompatibleChatProvider):
    """
    Provider for InternLM API.
    """

    BASE_URL = "https://chat.intern-ai.org.cn/api/v1"
    PROVIDER_ID = "internlm"
    CREDGOO_SERVICE = "internlm"
    ERROR_PROVIDER_NAME = "InternLM"
    DEFAULT_MODEL = "internlm3-latest"

    def __init__(self, api_key: Optional[str] = None, base_url: str = BASE_URL, **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)

    def _get_default_payload_params(self, stream: bool) -> dict[str, float | int]:
        return {
            "n": 1,
            "top_p": 0.9,
        }

    # list_models + _model_info inherited from OpenAICompatibleChatProvider
    # (base _model_info = id + owned_by, which is InternLM's whole dialect).
