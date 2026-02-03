"""
OpenAI TTS provider implementation.
"""
from typing import Optional

from ..core import TTSProvider, TTSRequest, TTSResponse
from ..errors import map_provider_error, UniInferError


class OpenAITTSProvider(TTSProvider):
    """
    Provider for OpenAI TTS API.

    OpenAI provides text-to-speech via their Audio API.
    """

    BASE_URL = "https://api.openai.com/v1"

    def __init__(self, api_key: Optional[str] = None, organization: Optional[str] = None):
        """
        Initialize the OpenAI TTS provider.

        Args:
            api_key (Optional[str]): The OpenAI API key.
            organization (Optional[str]): The OpenAI organization ID.
        """
        super().__init__(api_key)
        self.organization = organization

    @classmethod
    def list_models(cls, api_key: Optional[str] = None, **kwargs) -> list[str]:
        """List available TTS models from OpenAI."""
        return ["tts-1", "tts-1-hd"]

    async def agenerate_speech(
        self,
        request: TTSRequest,
        **provider_specific_kwargs
    ) -> TTSResponse:
        """Generate speech from text using OpenAI TTS."""
        if self.api_key is None:
            raise ValueError("OpenAI API key is required")

        client = await self._get_async_client()
        endpoint = f"{self.BASE_URL}/audio/speech"

        payload = {
            "model": request.model or "tts-1",
            "input": request.input,
            "voice": request.voice or "alloy",
            "response_format": request.response_format or "mp3",
            "speed": request.speed or 1.0
        }
        if request.instructions:
            payload["instructions"] = request.instructions

        payload.update(provider_specific_kwargs)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        try:
            response = await client.post(endpoint, headers=headers, json=payload, timeout=300.0)
            
            if response.status_code != 200:
                raise map_provider_error(
                    "OpenAI TTS",
                    Exception(response.text),
                    status_code=response.status_code,
                    response_body=response.text
                )
            
            content_type = response.headers.get("Content-Type", "audio/mpeg")
            return TTSResponse(
                audio_content=response.content,
                model=request.model or "tts-1",
                provider='openai',
                content_type=content_type,
                raw_response=response
            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("OpenAI TTS", e)
