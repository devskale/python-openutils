"""
TU STT provider implementation.
"""
import os

from ..core import STTProvider, STTRequest, STTResponse
from ..errors import map_provider_error, UniInferError


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
        """List available STT models from TU AI."""
        return []

    async def atranscribe(
        self,
        request: STTRequest,
        **provider_specific_kwargs
    ) -> STTResponse:
        """Transcribe audio to text using TU AI asynchronously."""
        if self.api_key is None:
            raise ValueError("TU AI API key is required")

        client = await self._get_async_client()
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
                '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.m4a': 'audio/mp4',
                '.flac': 'audio/flac', '.ogg': 'audio/ogg', '.webm': 'audio/webm'
            }
            mime_type = mime_types.get(ext, 'audio/mpeg')
            files["file"] = (os.path.basename(request.file), file_content, mime_type)
        else:
            raise ValueError("File must be either a file path (str) or file content (bytes)")

        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.organization:
            headers["TUW-Organization"] = self.organization

        try:
            response = await client.post(endpoint, headers=headers, files=files, data=data, timeout=300.0)
            if response.status_code != 200:
                raise map_provider_error("TU", Exception(response.text), status_code=response.status_code, response_body=response.text)
            
            response_data = response.json()
            return STTResponse(
                text=response_data.get("text", ""),
                model=request.model or "whisper-large",
                provider='tu',
                language=response_data.get("language"),
                duration=response_data.get("duration"),
                segments=response_data.get("segments"),
                raw_response=response_data
            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("TU", e)
