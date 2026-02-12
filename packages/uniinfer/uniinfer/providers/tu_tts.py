"""
TU TTS provider implementation.
"""

from ..core import TTSProvider, TTSRequest, TTSResponse
from ..errors import map_provider_error, UniInferError


class TuAITTSProvider(TTSProvider):
    """
    Provider for TU AI TTS API.

    TU AI provides OpenAI-compatible TTS endpoints via Aqueduct.
    """

    BASE_URL = "https://aqueduct.ai.datalab.tuwien.ac.at/v1"

    def __init__(self, api_key: str | None = None, organization: str | None = None):
        """
        Initialize the TU AI TTS provider.

        Args:
            api_key (Optional[str]): The TU AI API key.
            organization (Optional[str]): The TU AI organization ID.
        """
        super().__init__(api_key)
        self.organization = organization

    @classmethod
    def list_models(cls, api_key: str | None = None, **kwargs) -> list[str]:
        """List available TTS models from TU AI."""
        return []

    async def agenerate_speech(
        self,
        request: TTSRequest,
        **provider_specific_kwargs
    ) -> TTSResponse:
        """Generate speech from text using TU AI asynchronously."""
        if self.api_key is None:
            raise ValueError("TU AI API key is required")

        client = await self._get_async_client()
        endpoint = f"{self.BASE_URL}/audio/speech"

        payload = {
            "model": request.model or "kokoro",
            "input": request.input,
            "response_format": request.response_format or "mp3",
            "speed": request.speed or 1.0,
            "voice": request.voice or "af_alloy"
        }
        if request.instructions:
            payload["instructions"] = request.instructions

        payload.update(provider_specific_kwargs)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        if self.organization:
            headers["TUW-Organization"] = self.organization

        try:
            response = await client.post(endpoint, headers=headers, json=payload, timeout=300.0)
            if response.status_code != 200:
                raise map_provider_error("TU", Exception(response.text), status_code=response.status_code, response_body=response.text)
            
            content_type = response.headers.get("Content-Type", "audio/mpeg")
            return TTSResponse(
                audio_content=response.content,
                model=request.model or "kokoro",
                provider='tu',
                content_type=content_type,
                raw_response=response
            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("TU", e)
