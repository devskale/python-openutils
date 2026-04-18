from __future__ import annotations
"""
Pollinations provider implementation.
"""
from typing import Optional

import requests

from ..errors import map_provider_error
from .openai_compatible import OpenAICompatibleChatProvider


class PollinationsProvider(OpenAICompatibleChatProvider):
    """
    Provider for Pollinations OpenAI-compatible API.
    """

    BASE_URL = "https://text.pollinations.ai/openai"
    PROVIDER_ID = "pollinations"
    ERROR_PROVIDER_NAME = "Pollinations"
    DEFAULT_MODEL = "openai"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(api_key=api_key, base_url=base_url or self.BASE_URL)

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list[ModelInfo]:
        from ..core import ModelInfo
        """
        List available models from Pollinations.
        """
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        endpoints = [
            "https://gen.pollinations.ai/models",
            "https://text.pollinations.ai/openai/v1/models",
            "https://gen.pollinations.ai/openai/v1/models",
        ]

        last_error = None
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, headers=headers, timeout=30)
                if response.status_code != 200:
                    last_error = map_provider_error(
                        "Pollinations",
                        Exception(f"Pollinations API error: {response.status_code} - {response.text}"),
                        status_code=response.status_code,
                        response_body=response.text,
                    )
                    continue

                data = response.json()
                if isinstance(data, dict) and "data" in data:
                    results = []
                    for m in data["data"]:
                        if not isinstance(m, dict) or not m.get("id"):
                            continue
                        capabilities = {}
                        if m.get("reasoning"):
                            capabilities["reasoning"] = True
                        if m.get("vision"):
                            capabilities["vision"] = True
                        if m.get("tools"):
                            capabilities["tool_call"] = True
                        input_mods = m.get("input_modalities", ["text"])
                        output_mods = m.get("output_modalities", ["text"])
                        if capabilities.get("vision") and "image" not in input_mods:
                            input_mods = input_mods + ["image"]
                        results.append(ModelInfo(
                            id=m["id"],
                            name=m.get("name"),
                            type="chat",
                            modalities={"input": input_mods, "output": output_mods},
                            capabilities=capabilities or None,
                            raw=m,
                        ))
                    return results
                if isinstance(data, list):
                    results = []
                    for m in data:
                        if not isinstance(m, dict) or not m.get("name"):
                            continue
                        capabilities = {}
                        if m.get("reasoning"):
                            capabilities["reasoning"] = True
                        if m.get("vision"):
                            capabilities["vision"] = True
                        if m.get("tools"):
                            capabilities["tool_call"] = True
                        input_mods = m.get("input_modalities", ["text"])
                        output_mods = m.get("output_modalities", ["text"])
                        if capabilities.get("vision") and "image" not in input_mods:
                            input_mods = input_mods + ["image"]
                        results.append(ModelInfo(
                            id=m["name"],
                            name=m.get("name"),
                            type="chat",
                            modalities={"input": input_mods, "output": output_mods},
                            capabilities=capabilities or None,
                            raw=m,
                        ))
                    return results
                return []
            except Exception as e:
                last_error = e

        if last_error:
            raise map_provider_error("Pollinations", last_error)
        return []
