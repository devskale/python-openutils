from __future__ import annotations
"""
Kilo Gateway provider implementation.

Kilo Code (kilo.ai) runs the "Kilo AI Gateway" — an OpenAI-compatible, OpenRouter-
compatible router at ``https://api.kilo.ai/api/gateway`` that aggregates 300+
models from many providers (Anthropic, OpenAI, Google, xAI, DeepSeek, Qwen, NVIDIA,
...). The gateway's ``/models`` endpoint is public (no auth) and returns the same
schema as OpenRouter (``data[]`` with ``pricing``, ``architecture``, ``top_provider``,
``supported_parameters``, plus kilo-specific ``isFree``).

Free models (``isFree=true`` or id ending ``:free``) are usable anonymously —
rate-limited to 200 requests/hour per IP. Paid models need a "Kilo Gateway API key"
(``Authorization: Bearer $KILO_API_KEY``).

Caveats:
- Some free models route to NVIDIA trial endpoints, which log prompts/outputs for
  service improvement. Do not send confidential data to ``nvidia/*:free``.
- ``kilo-auto/*`` are virtual tiers (frontier/balanced/free/small/efficient) whose
  underlying model is chosen server-side and can change; their cost/context are
  approximate.
- ``openrouter/free`` and ``openrouter/auto-beta`` report ``pricing.prompt = "-1"``
  (sentinel for "varies") — mapped to ``cost=None``.
"""
import requests
from typing import Optional

from ..errors import map_provider_error
from .openai_compatible import OpenAICompatibleChatProvider


class KiloProvider(OpenAICompatibleChatProvider):
    """Provider for the Kilo AI Gateway (OpenAI/OpenRouter-compatible)."""

    BASE_URL = "https://api.kilo.ai/api/gateway"
    PROVIDER_ID = "kilo"
    ERROR_PROVIDER_NAME = "Kilo"
    DEFAULT_MODEL = "tencent/hy3:free"
    CREDGOO_SERVICE = "kilocode"
    # Free models are usable anonymously (200 req/hr per IP); paid models need a key.
    # A key is still resolved from credgoo when available so free-model requests
    # are authenticated (higher rate limits than the anonymous tier).
    REQUIRES_API_KEY = False
    # The gateway forwards native OpenAI multimodal content (image_url parts)
    # to vision-capable upstreams (e.g. stepfun/step-3.7-flash:free).
    PRESERVE_MULTIMODAL = True

    def __init__(self, api_key: Optional[str] = None):
        if not api_key:
            try:
                from credgoo import get_api_key
                api_key = get_api_key(self.CREDGOO_SERVICE)
            except Exception:
                api_key = None
        super().__init__(api_key=api_key, base_url=self.BASE_URL)

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list["ModelInfo"]:
        """List models from the Kilo Gateway.

        The ``/models`` endpoint is public (no auth required). When no key is
        passed, one is resolved from credgoo (``kilocode`` service) so the
        request is authenticated (higher rate limits than the anonymous tier);
        free models work either way. Free models (``isFree=true`` or id ending
        ``:free``) are marked cost 0/0 so the catalog surfaces them as free.
        """
        from ..core import ModelInfo
        if not api_key:
            try:
                from credgoo import get_api_key
                api_key = get_api_key(cls.CREDGOO_SERVICE)
            except Exception:
                api_key = None
        try:
            headers = {"accept": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            r = requests.get(
                "https://api.kilo.ai/api/gateway/models",
                headers=headers,
                timeout=30,
            )
            r.raise_for_status()
        except Exception as e:
            raise map_provider_error("Kilo", e)

        out: list[ModelInfo] = []
        for model in r.json().get("data", []):
            mid = model.get("id")
            if not mid:
                continue

            pricing = model.get("pricing", {}) or {}
            prompt_price = _parse_price(pricing.get("prompt"))
            completion_price = _parse_price(pricing.get("completion"))
            is_free = bool(model.get("isFree")) or mid.endswith(":free")

            if is_free:
                cost = {"input": 0.0, "output": 0.0}
            elif prompt_price is not None or completion_price is not None:
                cost = {}
                if prompt_price is not None:
                    cost["input"] = prompt_price * 1_000_000
                if completion_price is not None:
                    cost["output"] = completion_price * 1_000_000
            else:
                cost = None

            arch = model.get("architecture", {}) or {}
            modalities = None
            if arch.get("input_modalities") or arch.get("output_modalities"):
                modalities = {
                    "input": arch.get("input_modalities", ["text"]),
                    "output": arch.get("output_modalities", ["text"]),
                }

            supported_params = model.get("supported_parameters", []) or []
            capabilities: dict = {}
            if "tools" in supported_params or "tool_choice" in supported_params:
                capabilities["tool_call"] = True
            if "structured_outputs" in supported_params:
                capabilities["structured_outputs"] = True
            if "reasoning" in supported_params or "include_reasoning" in supported_params:
                capabilities["reasoning"] = True

            top_provider = model.get("top_provider", {}) or {}
            max_output = top_provider.get("max_completion_tokens")
            context_window = (
                top_provider.get("context_length")
                or model.get("context_length")
            )

            out.append(ModelInfo(
                id=mid,
                name=model.get("name"),
                type="chat",
                context_window=context_window,
                max_output=max_output,
                cost=cost,
                modalities=modalities,
                capabilities=capabilities or None,
                owned_by=model.get("owned_by") or (mid.split("/")[0] if "/" in mid else "kilo"),
                created=model.get("created"),
                raw=model,
            ))
        return out


def _parse_price(value) -> Optional[float]:
    """Parse a pricing string from the OpenRouter/Kilo schema.

    Returns ``None`` for missing values and for the ``"-1"`` sentinel (and any
    negative), which both OpenRouter and Kilo use to mean "price varies / not
    applicable" (e.g. for ``openrouter/free`` and ``openrouter/auto-beta``).
    """
    if value is None or value == "":
        return None
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if price < 0:
        return None
    return price
