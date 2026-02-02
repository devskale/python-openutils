"""
Anthropic provider implementation.
"""
import json
import httpx
import requests
from typing import Dict, Any, Iterator, Optional, AsyncIterator

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import map_provider_error, UniInferError


class AnthropicProvider(ChatProvider):
    """
    Provider for Anthropic Claude API.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Anthropic provider.

        Args:
            api_key (Optional[str]): The Anthropic API key.
        """
        super().__init__(api_key)
        self.base_url = "https://api.anthropic.com/v1"
        self.api_version = "2023-06-01"  # Current Anthropic API version

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """
        List available models from Anthropic using the API.

        Returns:
            list: A list of available model names.
        """
        if not api_key:
            raise ValueError("API key is required to list models")

        url = "https://api.anthropic.com/v1/models"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise map_provider_error("Anthropic", Exception(f"Anthropic API error: {response.status_code} - {response.text}"), status_code=response.status_code, response_body=response.text)

        data = response.json()
        # The API returns a list of model objects under the "data" key
        return [model["id"] for model in data.get("data", [])]

    async def acomplete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """
        Make an async chat completion request to Anthropic.
        """
        if self.api_key is None:
            raise ValueError("Anthropic API key is required")

        endpoint = f"{self.base_url}/messages"

        messages = []
        system_content = None
        for msg in request.messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        payload = {
            "messages": messages,
            "model": request.model or "claude-3-sonnet-20240229",
            "temperature": request.temperature,
            "stream": False
        }

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        if system_content:
            payload["system"] = system_content

        payload.update(provider_specific_kwargs)

        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": self.api_key,
            "anthropic-version": self.api_version
        }

        client = await self._get_async_client()
        try:
            response = await client.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=60.0
            )

            if response.status_code != 200:
                error_msg = f"Anthropic API error: {response.status_code} - {response.text}"
                raise map_provider_error("Anthropic", Exception(error_msg), status_code=response.status_code, response_body=response.text)

            response_data = response.json()
            content = response_data["content"][0]["text"]

            message = ChatMessage(
                role="assistant",
                content=content
            )

            usage = {
                "input_tokens": response_data.get("usage", {}).get("input_tokens", 0),
                "output_tokens": response_data.get("usage", {}).get("output_tokens", 0),
                "total_tokens": response_data.get("usage", {}).get("input_tokens", 0) +
                response_data.get("usage", {}).get("output_tokens", 0)
            }

            return ChatCompletionResponse(
                message=message,
                provider='anthropic',
                model=response_data.get('model', request.model),
                usage=usage,
                raw_response=response_data
            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("Anthropic", e)

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> AsyncIterator[ChatCompletionResponse]:
        """
        Stream an async chat completion response from Anthropic.
        """
        if self.api_key is None:
            raise ValueError("Anthropic API key is required")

        endpoint = f"{self.base_url}/messages"

        messages = []
        system_content = None
        for msg in request.messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        payload = {
            "messages": messages,
            "model": request.model or "claude-3-sonnet-20240229",
            "temperature": request.temperature,
            "stream": True
        }

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        if system_content:
            payload["system"] = system_content

        payload.update(provider_specific_kwargs)

        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": self.api_key,
            "anthropic-version": self.api_version
        }

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
                    error_msg = f"Anthropic API error: {response.status_code} - {await response.aread()}"
                    raise map_provider_error("Anthropic", Exception(error_msg), status_code=response.status_code, response_body=error_msg)

                async for line in response.aiter_lines():
                    if line:
                        try:
                            if not line or line == 'data: [DONE]':
                                continue

                            if line.startswith('data: '):
                                line = line[6:]

                            event_data = json.loads(line)

                            if event_data.get('type') == 'content_block_delta':
                                delta = event_data.get('delta', {})
                                content = delta.get('text', '')

                                message = ChatMessage(
                                    role="assistant", content=content)

                                usage = {
                                    "input_tokens": 0,
                                    "output_tokens": 0,
                                    "total_tokens": 0
                                }

                                yield ChatCompletionResponse(
                                    message=message,
                                    provider='anthropic',
                                    model=event_data.get('model', request.model),
                                    usage=usage,
                                    raw_response=event_data
                                )
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("Anthropic", e)

