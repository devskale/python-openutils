"""
TU TTS provider implementation.
"""
import requests
from typing import Optional, List

from ..core import TTSProvider, TTSRequest, TTSResponse
from ..errors import map_provider_error


class TuAITTSProvider(TTSProvider):
    """
    Provider for TU AI TTS API.

    TU AI provides OpenAI-compatible TTS endpoints via Aqueduct.
    """

    BASE_URL = "https://aqueduct.ai.datalab.tuwien.ac.at/v1"

    def __init__(self, api_key: Optional[str] = None, organization: Optional[str] = None):
        """
        Initialize the TU AI TTS provider.

        Args:
            api_key (Optional[str]): The TU AI API key.
            organization (Optional[str]): The TU AI organization ID.
        """
        super().__init__(api_key)
        self.organization = organization

    @classmethod
    def list_models(cls, api_key: Optional[str] = None, **kwargs) -> List[str]:
        """
        List available TTS models from TU AI.

        Returns:
            List[str]: A list of available TTS model IDs.
        """
        # TU AI supports these TTS models according to the documentation
        return ["kokoro", "piper-thorsten"]

    def generate_speech(
        self,
        request: TTSRequest,
        **provider_specific_kwargs
    ) -> TTSResponse:
        """
        Generate speech from text using TU AI.

        Args:
            request (TTSRequest): The request to make.
            **provider_specific_kwargs: Additional TU AI-specific parameters.

        Returns:
            TTSResponse: The TTS response with audio content.

        Raises:
            Exception: If the request fails.
        """
        if self.api_key is None:
            raise ValueError("TU AI API key is required")

        endpoint = f"{self.BASE_URL}/audio/speech"

        # Prepare the request payload
        payload = {
            "model": request.model or "kokoro",  # Default TTS model
            "input": request.input,
            "response_format": request.response_format or "mp3",
            "speed": request.speed or 1.0
        }

        # Add optional parameters
        if request.voice:
            payload["voice"] = request.voice
        else:
            # Set default voice if not provided, as it is required by the API
            payload["voice"] = "af_alloy"
            
        if request.instructions:
            payload["instructions"] = request.instructions

        # Add any provider-specific parameters
        payload.update(provider_specific_kwargs)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Add organization header if provided
        if self.organization:
            headers["TUW-Organization"] = self.organization

        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=120
        )

        # Handle error response
        if response.status_code != 200:
            error_msg = f"TU AI API error: {response.status_code} - {response.text}"
            raise map_provider_error("TU", Exception(error_msg), status_code=response.status_code, response_body=response.text)

        # Determine content type from response
        content_type = response.headers.get("Content-Type", "audio/mpeg")

        return TTSResponse(
            audio_content=response.content,
            model=request.model or "kokoro",
            provider='tu',
            content_type=content_type,
            raw_response=response
        )
