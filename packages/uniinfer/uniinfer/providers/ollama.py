"""
Ollama provider implementation.
"""
import asyncio
import httpx
import json
import requests
from typing import Dict, Any, Iterator, Optional, AsyncIterator

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import map_provider_error, UniInferError


def _normalize_base_url(base_url: str) -> str:
    """
    Normalize the base URL to ensure it has a scheme and upgrade non-localhost http to https.

    Args:
        base_url (str): The base URL to normalize.

    Returns:
        str: The normalized base URL.
    """
    # Ensure scheme present; allow plain URLs
    if not base_url.startswith(("http://", "https://")):
        base_url = "http://" + base_url
    # Upgrade non-localhost http to https
    if base_url.startswith("http://") and not base_url.startswith(("http://localhost", "http://127.0.0.1")):
        base_url = "https://" + base_url[len("http://"):]
    return base_url


class OllamaProvider(ChatProvider):
    """
    Provider for Ollama API.

    Ollama is a local LLM serving tool that allows running various models locally.
    This provider requires a running Ollama instance.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://localhost:11434", **kwargs):
        """
        Initialize the Ollama provider.

        Args:
            api_key (Optional[str]): Not used for Ollama, but kept for API consistency.
            base_url (str): The base URL for the Ollama API (default: http://localhost:11434).
            **kwargs: Additional configuration options.
        """
        super().__init__(api_key)
        self.base_url = base_url

    @classmethod
    def list_models(cls, **kwargs) -> list:
        """
        List available models from Ollama.

        Args:
            **kwargs: Additional configuration options including base_url and api_key

        Returns:
            list: A list of available model names.
        """
        try:
            # Use provided base_url or fallback
            raw_url = kwargs.get("base_url") or getattr(
                cls, "base_url", "localhost:11434")
            base_url = _normalize_base_url(raw_url)
            if "molodetz.nl" in base_url:
                # Special case for molodetz.nl, which uses a different endpoint
                endpoint = f"{base_url}/models"
            else:
                endpoint = f"{base_url}/api/tags"
            print(f"Using Ollama endpoint: {endpoint}")  # Verbose logging

            headers = {}
            api_key = kwargs.get("api_key")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            response = requests.get(endpoint, headers=headers)
            response.raise_for_status()

            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            if not models:
                # Fallback to tags if models are not available
                models = [model["id"] for model in data.get("data", [])]
            print(f"Found {len(models)} available models")
            return models

        except Exception as e:
            print(f"Error listing models from Ollama: {str(e)}")
            # Fallback to default models if API call fails
            return [
                "error listing models",
            ]

    async def acomplete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """
        Make an async chat completion request to Ollama.
        """
        base_url = _normalize_base_url(self.base_url)
        endpoint = f"{base_url}/api/chat"

        payload = {
            "model": request.model or "llama2",
            "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
            "stream": False,
            "options": {}
        }

        if request.temperature is not None:
            payload["options"]["temperature"] = request.temperature

        if request.max_tokens is not None:
            payload["options"]["num_predict"] = request.max_tokens

        if provider_specific_kwargs:
            for key, value in provider_specific_kwargs.items():
                if key == "options" and isinstance(value, dict):
                    payload["options"].update(value)
                else:
                    payload[key] = value

        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        client = await self._get_async_client()
        try:
            response = await client.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=60.0
            )

            if response.status_code != 200:
                error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                raise map_provider_error("Ollama", Exception(error_msg), status_code=response.status_code, response_body=response.text)

            response_data = response.json()
            assistant_message = response_data["message"]

            message = ChatMessage(
                role=assistant_message.get("role", "assistant"),
                content=assistant_message.get("content", "")
            )

            usage = {
                "prompt_tokens": response_data.get("prompt_eval_count", 0),
                "completion_tokens": response_data.get("eval_count", 0),
                "total_tokens": (
                    response_data.get("prompt_eval_count", 0) +
                    response_data.get("eval_count", 0)
                )
            }

            return ChatCompletionResponse(
                message=message,
                provider='ollama',
                model=response_data.get('model', request.model),
                usage=usage,
                raw_response=response_data
            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("Ollama", e)

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> AsyncIterator[ChatCompletionResponse]:
        """
        Stream an async chat completion response from Ollama.
        """
        base_url = _normalize_base_url(self.base_url)
        endpoint = f"{base_url}/api/chat"

        payload = {
            "model": request.model or "llama2",
            "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
            "stream": True,
            "options": {}
        }

        if request.temperature is not None:
            payload["options"]["temperature"] = request.temperature

        if request.max_tokens is not None:
            payload["options"]["num_predict"] = request.max_tokens

        if provider_specific_kwargs:
            for key, value in provider_specific_kwargs.items():
                if key == "options" and isinstance(value, dict):
                    payload["options"].update(value)
                else:
                    payload[key] = value

        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        client = await self._get_async_client()
        try:
            async with client.stream(
                "POST",
                endpoint,
                headers=headers,
                json=payload,
                timeout=60.0
            ) as response:
                if response.status_code != 200:
                    error_msg = f"Ollama API error: {response.status_code} - {await response.aread()}"
                    raise map_provider_error("Ollama", Exception(error_msg), status_code=response.status_code, response_body=error_msg)

                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)

                            if "done" in data and data["done"]:
                                continue

                            content = ""
                            if "message" in data and "content" in data["message"]:
                                content = data["message"]["content"]

                            message = ChatMessage(
                                role="assistant", content=content)

                            yield ChatCompletionResponse(
                                message=message,
                                provider='ollama',
                                model=data.get('model', request.model),
                                usage={},
                                raw_response=data
                            )
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("Ollama", e)

