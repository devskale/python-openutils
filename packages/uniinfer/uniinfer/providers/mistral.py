from __future__ import annotations
"""
Mistral AI provider implementation.

Access: universally free — La Plateforme's free Experiment tier gives
rate-limited access to ALL models (incl. Mistral Large, Codestral) at $0
(~1B tokens/month); pay-as-you-go unlocks higher limits (mistral.ai/news,
pricepertoken, costbench). The /models API carries no pricing field, so all
models are tagged access='free'.
"""
from typing import Optional

from ..core import ModelInfo
from .openai_compatible import OpenAICompatibleChatProvider


class MistralProvider(OpenAICompatibleChatProvider):
    """
    Provider for Mistral AI API.
    """

    BASE_URL = "https://api.mistral.ai/v1"
    PROVIDER_ID = "mistral"
    CREDGOO_SERVICE = "mistral"
    ERROR_PROVIDER_NAME = "Mistral"
    DEFAULT_MODEL: str | None = None
    # Mistral requires prefix=True on a trailing assistant message (prefill) or
    # it 400s ("Expected last role User or Tool (or Assistant with prefix True)").
    # Declared via the base PREFILL_FLAG mechanism — prefix isn't an OpenAI field,
    # so the provider must set it.
    PREFILL_FLAG = "prefix"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key=api_key, base_url=self.BASE_URL)

    @classmethod
    def _model_info(cls, raw: dict) -> ModelInfo:
        # universally free (Experiment tier: all models at $0, ~1B tok/mo)
        caps_src = raw.get("capabilities", {})
        caps: dict = {}
        for key in ("reasoning", "vision", "function_calling", "audio", "ocr", "classification", "moderation"):
            if caps_src.get(key):
                caps[key] = True
        if caps.get("function_calling"):
            caps["tool_call"] = True
        if caps.get("vision"):
            caps.pop("vision", None)
        input_mods = ["text"]
        if caps_src.get("vision"):
            input_mods.append("image")
        if caps_src.get("ocr"):
            input_mods.append("pdf")
        output_mods = ["text"]
        if caps_src.get("audio_speech"):
            output_mods.append("audio")
        mtype = "chat"
        if caps_src.get("audio_transcription"):
            mtype = "stt"; input_mods = ["audio"]; output_mods = ["text"]
        elif caps_src.get("audio_speech") and not caps_src.get("completion_chat"):
            mtype = "tts"; input_mods = ["text"]; output_mods = ["audio"]
        deprecation = raw.get("deprecation")
        return ModelInfo(
            id=raw["id"], name=raw.get("name"), type=mtype, access="free",
            status="deprecated" if deprecation else "active",
            deprecation_date=deprecation,
            deprecation_replacement=raw.get("deprecation_replacement_model"),
            context_window=raw.get("max_context_length"), max_output=raw.get("max_tokens"),
            modalities={"input": input_mods, "output": output_mods},
            capabilities=caps or None, owned_by=raw.get("owned_by"),
            created=raw.get("created"), raw=raw,
        )
