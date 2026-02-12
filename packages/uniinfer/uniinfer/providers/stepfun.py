"""
StepFun provider implementation.
"""
from typing import Optional, List

import requests

from .openai_compatible import OpenAICompatibleChatProvider


class StepFunProvider(OpenAICompatibleChatProvider):
    """
    Provider for StepFun API (阶跃星辰).
    """

    BASE_URL = "https://api.stepfun.com/v1"
    PROVIDER_ID = "stepfun"
    ERROR_PROVIDER_NAME = "stepfun"
    DEFAULT_MODEL = "step-1-8k"

    def __init__(self, api_key: Optional[str] = None, base_url: str = BASE_URL, **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)

    @classmethod
    def list_models(cls, api_key: Optional[str] = None, base_url: str = BASE_URL) -> List[str]:
        """List available models from StepFun."""
        if api_key is None:
            try:
                from credgoo.credgoo import get_api_key
                api_key = get_api_key("stepfun")
            except ImportError:
                return []

        if not api_key:
            return []

        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(f"{base_url}/models", headers=headers)
            response.raise_for_status()
            models_data = response.json()
            return [model["id"] for model in models_data.get("data", [])]
        except Exception:
            return []
