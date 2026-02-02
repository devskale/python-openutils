"""
Ollama embedding provider implementation.
"""
import json
from typing import List, Dict, Any, Optional

from ..core import EmbeddingProvider, EmbeddingRequest, EmbeddingResponse
from ..errors import map_provider_error, UniInferError


def _normalize_base_url(base_url: str) -> str:
    """Normalize the base URL."""
    if not base_url.startswith(("http://", "https://")):
        base_url = "http://" + base_url
    if base_url.startswith("http://") and not base_url.startswith(("http://localhost", "http://127.0.0.1")):
        base_url = "https://" + base_url[len("http://"):]
    return base_url


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Provider for Ollama embeddings API.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://localhost:11434", **kwargs):
        """Initialize the Ollama embedding provider."""
        super().__init__(api_key)
        self.base_url = base_url

    @classmethod
    def list_models(cls, **kwargs) -> List[str]:
        """List available models from Ollama."""
        import requests
        try:
            raw_url = kwargs.get("base_url") or "localhost:11434"
            base_url = _normalize_base_url(raw_url)
            endpoint = f"{base_url}/api/tags"
            headers = {}
            if kwargs.get("api_key"):
                headers["Authorization"] = f"Bearer {kwargs['api_key']}"
            response = requests.get(endpoint, headers=headers)
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception:
            return ["nomic-embed-text"]

    async def aembed(
        self,
        request: EmbeddingRequest,
        **provider_specific_kwargs
    ) -> EmbeddingResponse:
        """Make an async embedding request to Ollama."""
        client = await self._get_async_client()
        base_url = _normalize_base_url(self.base_url)
        endpoint = f"{base_url}/api/embed"

        payload = {
            "model": request.model or "nomic-embed-text",
            "input": request.input
        }
        payload.update(provider_specific_kwargs)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = await client.post(endpoint, headers=headers, json=payload, timeout=60.0)
            if response.status_code != 200:
                raise map_provider_error("Ollama", Exception(response.text), status_code=response.status_code, response_body=response.text)
            
            resp_data = response.json()
            embeddings = resp_data.get("embeddings", [])
            data = [{"object": "embedding", "embedding": emb, "index": i} for i, emb in enumerate(embeddings)]

            usage = {
                "prompt_tokens": resp_data.get("prompt_eval_count", 0),
                "total_tokens": resp_data.get("prompt_eval_count", 0)
            }

            return EmbeddingResponse(
                object="list",
                data=data,
                model=resp_data.get('model', request.model),
                usage=usage,
                provider='ollama',
                raw_response=resp_data
            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("Ollama", e)
