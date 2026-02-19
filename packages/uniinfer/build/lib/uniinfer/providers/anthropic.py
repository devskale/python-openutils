"""
Anthropic provider implementation.
"""
from typing import Optional

from .anthropic_compatible import AnthropicCompatibleProvider


class AnthropicProvider(AnthropicCompatibleProvider):
    """
    Provider for Anthropic Claude API.
    """

    BASE_URL = "https://api.anthropic.com"
    PROVIDER_ID = "anthropic"
    ERROR_PROVIDER_NAME = "Anthropic"
    DEFAULT_MODEL = "claude-3-sonnet-20240229"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(api_key=api_key, base_url=base_url or self.BASE_URL)
