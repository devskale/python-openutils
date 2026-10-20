"""OpenAI-compatible TTS + STT for custom instances (``provider: openai-compat``).

Instance-driven audio: unlike the fixed-endpoint providers (openai, tu), these
classes take a mandatory ``base_url`` — the ``hab`` gateway pattern (dgx
model-proxy) serves ``/v1/audio/speech`` (vLLM-Omni TTS, e.g. Qwen3-TTS
CustomVoice) and ``/v1/audio/transcriptions`` (Nemotron/whisper STT) behind one
OpenAI-shaped audio surface. The router resolves the instance alias and hands
``base_url`` + bearer to the target; these providers just speak the dialect.
"""
from __future__ import annotations

from typing import Optional

from ..core import ModelInfo, STTProvider, STTRequest, STTResponse, TTSProvider, TTSRequest, TTSResponse


class OpenAICompatTTSProvider(TTSProvider):
    """TTS against ``{base_url}/audio/speech`` (JSON in, audio bytes out)."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        if not base_url:
            raise ValueError("base_url is required for openai-compat TTS instances")
        super().__init__(api_key)
        self.base_url = base_url.rstrip("/")

    @classmethod
    def list_models(cls, api_key: Optional[str] = None, **kwargs) -> list[ModelInfo]:
        return []

    async def agenerate_speech(self, request: TTSRequest, **provider_specific_kwargs) -> TTSResponse:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": request.model,
            "input": request.input,
            "voice": request.voice or "ono_anna",
            "response_format": request.response_format or "wav",
        }
        if request.speed and request.speed != 1.0:
            payload["speed"] = request.speed
        if request.instructions:
            payload["instructions"] = request.instructions
        payload.update(provider_specific_kwargs)
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(f"{self.base_url}/audio/speech",
                                     json=payload, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(
                f"openai-compat TTS upstream {resp.status_code}: {resp.text[:300]}")
        return TTSResponse(
            audio_content=resp.content,
            model=request.model or "",
            provider="openai-compat",
            content_type=resp.headers.get("Content-Type", "audio/wav"),
        )


class OpenAICompatSTTProvider(STTProvider):
    """STT against ``{base_url}/audio/transcriptions`` (multipart in, JSON out)."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        if not base_url:
            raise ValueError("base_url is required for openai-compat STT instances")
        super().__init__(api_key)
        self.base_url = base_url.rstrip("/")

    @classmethod
    def list_models(cls, api_key: Optional[str] = None, **kwargs) -> list[ModelInfo]:
        return []

    async def atranscribe(self, request: STTRequest, **provider_specific_kwargs) -> STTResponse:
        import httpx

        audio = getattr(request, "file", None)
        if not audio:
            raise ValueError("STTRequest carries no audio content")
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        files = {"file": ("audio.wav", audio, "application/octet-stream")}
        data = {"model": request.model or "whisper-large",
                "response_format": request.response_format or "json"}
        if request.language:
            data["language"] = request.language
        if request.prompt:
            data["prompt"] = request.prompt
        data.update(provider_specific_kwargs)
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(f"{self.base_url}/audio/transcriptions",
                                     files=files, data=data, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(
                f"openai-compat STT upstream {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        return STTResponse(
            text=body.get("text", ""),
            model=request.model or "",
            provider="openai-compat",
            language=body.get("language"),
            duration=body.get("duration_sec") or body.get("duration"),
        )
