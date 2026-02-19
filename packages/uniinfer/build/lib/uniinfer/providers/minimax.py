"""
MiniMax provider implementation.
"""
from typing import Optional

from .anthropic_compatible import AnthropicCompatibleProvider


class MiniMaxProvider(AnthropicCompatibleProvider):
    """
    Provider for MiniMax Anthropic-compatible API.
    """

    BASE_URL = "https://api.minimax.io/anthropic"
    PROVIDER_ID = "minimax"
    ERROR_PROVIDER_NAME = "MiniMax"
    DEFAULT_MODEL = "MiniMax-M2.1"
    CREDGOO_SERVICE = "minimax"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(api_key=api_key, base_url=base_url or self.BASE_URL)

    @classmethod
    def list_models(cls, api_key: Optional[str] = None, **kwargs) -> list[str]:
        """
        MiniMax Anthropic-compatible API may not expose Anthropic model-list endpoint.
        Return known supported models as stable fallback.
        """
        default_models = ["MiniMax-M2.1", "MiniMax-M2.1-lightning", "MiniMax-M2"]
        try:
            models = super().list_models(api_key=api_key, **kwargs)
            return models or default_models
        except Exception:
            return default_models
