"""
TU embedding provider implementation.
"""
import json
from typing import Any, Optional

from ..core import EmbeddingProvider, EmbeddingRequest, EmbeddingResponse
from ..errors import map_provider_error, UniInferError


class TuAIEmbeddingProvider(EmbeddingProvider):
    """
    Provider for TU AI embeddings API.

    TU AI provides OpenAI-compatible embedding endpoints.
    """

    BASE_URL = "https://aqueduct.ai.datalab.tuwien.ac.at/v1"

    def __init__(self, api_key: str | None = None, organization: str | None = None):
        """
        Initialize the TU AI embedding provider.

        Args:
            api_key (Optional[str]): The TU AI API key.
            organization (Optional[str]): The TU AI organization ID.
        """
        super().__init__(api_key)
        self.organization = organization

    @classmethod
    def list_models(cls, api_key: str | None = None, **kwargs) -> list[str]:
        """List available models from TU AI using the API."""
        import requests
        if not api_key:
            return ["e5-mistral-7b"]
        
        url = f"{cls.BASE_URL}/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return [model["id"] for model in data.get("data", [])]
        except Exception:
            pass
        return ["e5-mistral-7b"]

    async def aembed(
        self,
        request: EmbeddingRequest,
        **provider_specific_kwargs
    ) -> EmbeddingResponse:
        """Make an async embedding request to TU AI."""
        if self.api_key is None:
            raise ValueError("TU AI API key is required")

        client = await self._get_async_client()
        endpoint = f"{self.BASE_URL}/embeddings"

        payload = {
            "model": request.model or "e5-mistral-7b",
            "input": request.input
        }
        payload.update(provider_specific_kwargs)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        if self.organization:
            headers["TUW-Organization"] = self.organization

        try:
            response = await client.post(endpoint, headers=headers, json=payload, timeout=60.0)
            if response.status_code != 200:
                raise map_provider_error("TU", Exception(response.text), status_code=response.status_code, response_body=response.text)
            
            resp_data = response.json()
            embeddings_data = resp_data.get("data", [])
            data = [{"object": "embedding", "embedding": item["embedding"], "index": item["index"]} for item in embeddings_data]

            return EmbeddingResponse(
                object="list",
                data=data,
                model=resp_data.get('model', request.model),
                usage=resp_data.get("usage", {}),
                provider='tu',
                raw_response=resp_data
            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("TU", e)
