"""
TU STT provider implementation.
"""
import httpx
import os
import requests
from typing import Any

from ..core import STTProvider, STTRequest, STTResponse
from ..errors import map_provider_error


class TuAISTTProvider(STTProvider):
    """
    Provider for TU AI STT (Speech-to-Text) API.

    TU AI provides OpenAI-compatible transcription endpoints via Aqueduct.
    """

    BASE_URL = "https://aqueduct.ai.datalab.tuwien.ac.at/v1"

    def __init__(self, api_key: str | None = None, organization: str | None = None):
        """
        Initialize the TU AI STT provider.

        Args:
            api_key (Optional[str]): The TU AI API key.
            organization (Optional[str]): The TU AI organization ID.
        """
        super().__init__(api_key)
        self.organization = organization

    @classmethod
    def list_models(cls, api_key: str | None = None, **kwargs) -> list[str]:
        """
        List available STT models from TU AI.

        Returns:
            List[str]: A list of available STT model IDs.
        """
        # TU AI supports these STT models according to the documentation
        return ["whisper-large", "whisper-1"]

    def transcribe(
        self,
        request: STTRequest,
        **provider_specific_kwargs
    ) -> STTResponse:
        """
        Transcribe audio to text using TU AI.

        Args:
            request (STTRequest): The request to make.
            **provider_specific_kwargs: Additional TU AI-specific parameters.

        Returns:
            STTResponse: The STT response with transcribed text.

        Raises:
            Exception: If the request fails.
        """
        if self.api_key is None:
            raise ValueError("TU AI API key is required")

        endpoint = f"{self.BASE_URL}/audio/transcriptions"

        # Prepare the multipart form data
        files = {}
        data = {
            "model": request.model or "whisper-large",  # Default STT model
            "response_format": request.response_format or "json",
            "temperature": request.temperature or 0.0
        }

        # Add optional parameters
        if request.language:
            data["language"] = request.language
        if request.prompt:
            data["prompt"] = request.prompt
        if request.timestamp_granularities:
            data["timestamp_granularities[]"] = request.timestamp_granularities

        # Add any provider-specific parameters
        data.update(provider_specific_kwargs)

        # Handle file input
        if isinstance(request.file, bytes):
            # File content provided as bytes
            files["file"] = ("audio.mp3", request.file, "audio/mpeg")
        elif isinstance(request.file, str):
            # File path provided
            if not os.path.exists(request.file):
                raise ValueError(f"Audio file not found: {request.file}")
            with open(request.file, 'rb') as f:
                file_content = f.read()
            # Determine MIME type from file extension
            ext = os.path.splitext(request.file)[1].lower()
            mime_types = {
                '.mp3': 'audio/mpeg',
                '.wav': 'audio/wav',
                '.m4a': 'audio/mp4',
                '.flac': 'audio/flac',
                '.ogg': 'audio/ogg',
                '.webm': 'audio/webm'
            }
            mime_type = mime_types.get(ext, 'audio/mpeg')
            files["file"] = (os.path.basename(request.file), file_content, mime_type)
        else:
            raise ValueError("File must be either a file path (str) or file content (bytes)")

        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        # Add organization header if provided
        if self.organization:
            headers["TUW-Organization"] = self.organization

        response = requests.post(
            endpoint,
            headers=headers,
            files=files,
            data=data,
            timeout=120
        )

        # Handle error response
        if response.status_code != 200:
            error_msg = f"TU AI API error: {response.status_code} - {response.text}"
            raise map_provider_error("TU", Exception(error_msg), status_code=response.status_code, response_body=response.text)

        # Parse the response
        response_data = response.json()

        # Extract transcription text
        text = response_data.get("text", "")
        
        # Extract optional fields
        language = response_data.get("language")
        duration = response_data.get("duration")
        segments = response_data.get("segments")

        return STTResponse(
            text=text,
            model=request.model or "whisper-large",
            provider='tu',
            language=language,
            duration=duration,
            segments=segments,
            raw_response=response_data
        )

    async def atranscribe(
        self,
        request: STTRequest,
        **provider_specific_kwargs
    ) -> STTResponse:
        """
        Transcribe audio to text using TU AI asynchronously.

        Args:
            request (STTRequest): The request to make.
            **provider_specific_kwargs: Additional TU AI-specific parameters.

        Returns:
            STTResponse: The STT response with transcribed text.

        Raises:
            Exception: If the request fails.
        """
        if self.api_key is None:
            raise ValueError("TU AI API key is required")

        endpoint = f"{self.BASE_URL}/audio/transcriptions"

        data = {
            "model": request.model or "whisper-large",
            "response_format": request.response_format or "json",
            "temperature": request.temperature or 0.0
        }

        if request.language:
            data["language"] = request.language
        if request.prompt:
            data["prompt"] = request.prompt
        if request.timestamp_granularities:
            data["timestamp_granularities[]"] = request.timestamp_granularities

        data.update(provider_specific_kwargs)

        files = {}
        if isinstance(request.file, bytes):
            files["file"] = ("audio.mp3", request.file, "audio/mpeg")
        elif isinstance(request.file, str):
            if not os.path.exists(request.file):
                raise ValueError(f"Audio file not found: {request.file}")
            with open(request.file, 'rb') as f:
                file_content = f.read()
            ext = os.path.splitext(request.file)[1].lower()
            mime_types = {
                '.mp3': 'audio/mpeg',
                '.wav': 'audio/wav',
                '.m4a': 'audio/mp4',
                '.flac': 'audio/flac',
                '.ogg': 'audio/ogg',
                '.webm': 'audio/webm'
            }
            mime_type = mime_types.get(ext, 'audio/mpeg')
            files["file"] = (os.path.basename(request.file), file_content, mime_type)
        else:
            raise ValueError("File must be either a file path (str) or file content (bytes)")

        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        if self.organization:
            headers["TUW-Organization"] = self.organization

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    endpoint,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=120.0
                )

                if response.status_code != 200:
                    error_msg = f"TU AI API error: {response.status_code} - {response.text}"
                    raise map_provider_error("TU", Exception(error_msg), status_code=response.status_code, response_body=response.text)

                response_data = response.json()

                text = response_data.get("text", "")

                language = response_data.get("language")
                duration = response_data.get("duration")
                segments = response_data.get("segments")

                return STTResponse(
                    text=text,
                    model=request.model or "whisper-large",
                    provider='tu',
                    language=language,
                    duration=duration,
                    segments=segments,
                    raw_response=response_data
                )
            except Exception as e:
                status_code = getattr(e.response, 'status_code', None) if hasattr(
                    e, 'response') else None
                response_body = getattr(e.response, 'text', None) if hasattr(
                    e, 'response') else str(e)
                raise map_provider_error(
                    "TU", e, status_code=status_code, response_body=response_body)
