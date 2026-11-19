"""Generic config-driven OpenAI-compatible provider for fleet instances.

Unlike the concrete subclasses (groq, kilo, openrouter, …) which hardcode their
identity, this class holds NO baked-in ``BASE_URL``/``PROVIDER_ID``/``CREDGOO_SERVICE``
— all of it comes from the instance config (``resolve_instance``). It is the
home for "any ``/v1`` endpoint with no special dialect": vLLM, Ollama-compat,
LM Studio, local servers, Together-style gateways. Provider-specific reasoning
dialects stay in their own subclasses; this one is pure passthrough.

Registered under ``openai-compat``.
"""
from __future__ import annotations

from typing import Optional

import httpx

from ..core import ChatProvider, ModelInfo
from .openai_compatible import OpenAICompatibleChatProvider


class OpenAICompatProvider(OpenAICompatibleChatProvider):
    """A faceless OpenAI-compatible endpoint — identity supplied by instance config."""

    BASE_URL = ""
    PROVIDER_ID = "openai-compat"
    ERROR_PROVIDER_NAME = "OpenAI-compat"
    DEFAULT_MODEL: Optional[str] = None
    # No CREDGOO_SERVICE: a custom instance defaults to its own alias name.
    # OpenAI list content (incl. image_url) is this provider's native dialect
    # (vLLM/Ollama/LM Studio all accept it); text-only models behind it answer
    # with a loud backend error instead of the flatten's silent image drop
    # (issue hab-qwen-image-multimodal). String-only legacy endpoints opt out
    # per instance: preserve_multimodal: false.
    PRESERVE_MULTIMODAL = True

    @classmethod
    def list_models(
        cls,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> list[ModelInfo]:
        """List models from ``{base_url}/models`` (OpenAI shape)."""
        if not base_url:
            return []
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            resp = httpx.get(
                f"{base_url.rstrip('/')}/models", headers=headers, timeout=15.0
            )
            resp.raise_for_status()
            data = resp.json().get("data", []) or resp.json().get("models", [])
            out: list[ModelInfo] = []
            for m in data:
                if isinstance(m, dict):
                    mid = m.get("id")
                    if not mid:
                        continue
                    # Upstream owned_by is a real discriminator (DGX model-proxy
                    # sends "sttts" for audio stack entries, "vllm" for vLLM)
                    # — keep it; only fall back to our class id when absent.
                    mi = ModelInfo(id=mid,
                                   owned_by=m.get("owned_by") or cls.PROVIDER_ID,
                                   raw=m)
                    # Derive type early so audio models (asr/tts/supertonic)
                    # never surface as chat candidates (issue
                    # uniinfer-hab-audio-models-mislabeled-as-chat).
                    derived = mi.derive_type()
                    if derived != "chat":
                        mi.type = derived
                    out.append(mi)
                elif m:
                    out.append(ModelInfo(id=str(m), owned_by=cls.PROVIDER_ID))
            return out
        except Exception:
            return []
