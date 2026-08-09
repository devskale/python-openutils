"""Audio dispatch — the ``Target`` for TTS + STT.

Mirrors :mod:`uniinfer.completion` (``Target`` for chat) and
:mod:`uniinfer.images` (``ImageTarget`` for images): binds a ``provider@model``
+ API key, owns provider resolution + request building + dispatch behind one
small interface per modality. The HTTP router becomes a thin adapter.

TTS and STT are separate targets (not one module with two methods) because
they have genuinely different interfaces, request types, and response shapes
— one module would be a shallow namespace rather than a deep module.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Optional

from uniinfer.completion import parse_provider_model

# ---------------------------------------------------------------------------
# Provider registries — lazy (import on first use). Adding a TTS/STT provider
# is a new entry here, not a router branch.
# ---------------------------------------------------------------------------
_TTS_REGISTRY: dict[str, str] = {
    "tu": "uniinfer.providers.tu_tts.TuAITTSProvider",
    "openai": "uniinfer.providers.openai_tts.OpenAITTSProvider",
    "openaitts": "uniinfer.providers.openai_tts.OpenAITTSProvider",
}
_STT_REGISTRY: dict[str, str] = {
    "tu": "uniinfer.providers.tu_stt.TuAISTTProvider",
}


def _resolve(dotted: str):
    mod_path, _, attr = dotted.rpartition(".")
    return getattr(importlib.import_module(mod_path), attr)


# ---------------------------------------------------------------------------
# Result types — library-level, HTTP-agnostic. The router shapes these into
# HTTP responses (Response / JSONResponse).
# ---------------------------------------------------------------------------
@dataclass
class TTSResult:
    audio_content: bytes
    content_type: str


@dataclass
class STTResult:
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    segments: Optional[list] = None


class AudioGenerationError(Exception):
    """An audio provider failure carrying a status code for the router to map."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(body)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
class TTSTarget:
    """Bind a ``provider@model`` + key and own the text-to-speech dispatch.

    The single home for "synthesize speech" — the audio analog of
    :class:`uniinfer.completion.Target`. Resolves the TTS provider class at
    construction; the router is a thin HTTP adapter.

    Raises:
        ValueError: provider unknown or has no TTS implementation.
    """

    def __init__(self, provider_model: str, api_key: Optional[str] = None):
        self.provider_name, self.model_name = parse_provider_model(provider_model)
        self.api_key = api_key
        dotted = _TTS_REGISTRY.get(self.provider_name)
        if not dotted:
            raise ValueError(f"TTS not supported for provider '{self.provider_name}'")
        self._provider_cls = _resolve(dotted)

    async def asynthesize(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        response_format: str = "mp3",
        speed: float = 1.0,
        instructions: Optional[str] = None,
    ) -> TTSResult:
        from uniinfer import TTSRequest

        provider = self._provider_cls(api_key=self.api_key)
        request = TTSRequest(
            input=text,
            model=self.model_name,
            voice=voice,
            response_format=response_format,
            speed=speed,
            instructions=instructions,
        )
        response = await provider.agenerate_speech(request)
        return TTSResult(
            audio_content=response.audio_content,
            content_type=response.content_type,
        )


class STTTarget:
    """Bind a ``provider@model`` + key and own the speech-to-text dispatch.

    The single home for "transcribe audio." Analog of :class:`TTSTarget` for STT.

    Raises:
        ValueError: provider unknown or has no STT implementation.
    """

    def __init__(self, provider_model: str, api_key: Optional[str] = None):
        self.provider_name, self.model_name = parse_provider_model(provider_model)
        self.api_key = api_key
        dotted = _STT_REGISTRY.get(self.provider_name)
        if not dotted:
            raise ValueError(f"STT not supported for provider '{self.provider_name}'")
        self._provider_cls = _resolve(dotted)

    async def atranscribe(
        self,
        audio_content: bytes,
        *,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> STTResult:
        from uniinfer import STTRequest

        provider = self._provider_cls(api_key=self.api_key)
        request = STTRequest(
            file=audio_content,
            model=self.model_name,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
        )
        response = await provider.atranscribe(request)
        return STTResult(
            text=response.text,
            language=getattr(response, "language", None),
            duration=getattr(response, "duration", None),
            segments=getattr(response, "segments", None),
        )
