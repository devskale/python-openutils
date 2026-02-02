"""
Pollinations provider implementation.

Pollinations is a unified API to access multiple AI models from different providers.
"""
from typing import Optional, Iterator, AsyncIterator
import requests
import json
import urllib.parse
import asyncio

import httpx

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import map_provider_error


class PollinationsProvider(ChatProvider):
    """
    Provider for Pollinations API.

    Pollinations provides a unified interface to access multiple AI models from
    different providers, including Anthropic, OpenAI, and more.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Pollinations provider.

        Args:
            api_key (Optional[str]): The Pollinations API key.
        """
        super().__init__(api_key)
        self.base_url = "https://gen.pollinations.ai/v1/chat/completions"
        self._async_client = None

    def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0),
                limits=httpx.Limits(max_connections=100),
            )
        return self._async_client

    async def close(self):
        """Close async client."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """
        List available models from Pollinations.

        Args:
            api_key (Optional[str]): The Pollinations API key. If not provided,
                                     it attempts to retrieve it using credgoo.

        Returns:
            list: A list of available model IDs.

        Raises:
            ValueError: If no API key is provided or found.
            Exception: If the API request fails.
        """
        # Retrieve API key if not provided (though not strictly required for listing models on public endpoint)
        if not api_key:
            try:
                from credgoo.credgoo import get_api_key
                api_key = get_api_key('pollinations')
            except (ImportError, Exception):
                pass

        endpoint = "https://gen.pollinations.ai/models"
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        response = requests.get(endpoint, headers=headers)

        if response.status_code != 200:
            error_msg = f"Pollinations API error: {response.status_code} - {response.text}"
            raise map_provider_error("Pollinations", Exception(error_msg), status_code=response.status_code, response_body=response.text)

        models = response.json()

        # Handle list of dicts format (current gen.pollinations.ai format)
        if isinstance(models, list):
            return [model.get('name', '') for model in models if isinstance(model, dict) and model.get('name')]

        # Handle potential wrapped format
        if isinstance(models, dict) and 'data' in models:
            return [model.get('id', '') for model in models['data'] if model.get('id')]

        return []

    async def acomplete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """
        Make an async text generation request to Pollinations.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional Pollinations-specific parameters.

        Returns:
            ChatCompletionResponse: The completion response.

        Raises:
            Exception: If the request fails.
        """
        messages = []
        for msg in request.messages:
            content = msg.content
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                content = "".join(text_parts)

            messages.append({
                "role": msg.role,
                "content": content
            })

        payload = {
            "model": request.model or "openai",
            "messages": messages,
            "stream": False
        }

        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        payload.update(provider_specific_kwargs)

        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            client = self._get_async_client()
            response = await client.post(self.base_url, headers=headers, json=payload)

            if response.status_code != 200:
                error_msg = f"Pollinations API error: {response.status_code} - {response.text}"
                raise map_provider_error("Pollinations", Exception(error_msg), status_code=response.status_code, response_body=response.text)

            response_data = await response.json()

            if 'choices' not in response_data or not response_data['choices']:
                raise Exception("API returned invalid response")

            choice = response_data['choices'][0]
            content = choice.get('message', {}).get('content', '')

            message = ChatMessage(
                role="assistant",
                content=content
            )

            return ChatCompletionResponse(
                message=message,
                provider='pollinations',
                model=request.model,
                usage=response_data.get('usage', {}),
                raw_response=response_data
            )

        except httpx.HTTPStatusError as e:
            response_body = e.response.text
            raise map_provider_error("Pollinations", e, status_code=e.response.status_code, response_body=response_body)
        except httpx.RequestError as e:
            raise map_provider_error("Pollinations", e)

    def complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """
        Make a text generation request to Pollinations.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional Pollinations-specific parameters.

        Returns:
            ChatCompletionResponse: The completion response.

        Raises:
            Exception: If the request fails.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop is None or not loop.is_running():
            return asyncio.run(self.acomplete(request, **provider_specific_kwargs))
        
        import concurrent.futures
        import threading
        result_container = []
        exception_container = []
        
        def run_in_thread():
            try:
                result = asyncio.run(self.acomplete(request, **provider_specific_kwargs))
                result_container.append(result)
            except Exception as e:
                exception_container.append(e)
        
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()
        
        if exception_container:
            raise exception_container[0]
        return result_container[0]

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> AsyncIterator[ChatCompletionResponse]:
        """
        Stream an async chat completion response from Pollinations.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional Pollinations-specific parameters.

        Returns:
            AsyncIterator[ChatCompletionResponse]: An iterator of response chunks.

        Raises:
            Exception: If the request fails.
        """
        messages = []
        for msg in request.messages:
            content = msg.content
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                content = "".join(text_parts)

            messages.append({
                "role": msg.role,
                "content": content
            })

        payload = {
            "model": request.model or "openai",
            "messages": messages,
            "stream": True
        }

        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        payload.update(provider_specific_kwargs)

        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            client = self._get_async_client()
            async with client.stream("POST", self.base_url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    error_msg = f"Pollinations API error: {response.status_code} - {await response.aread()}"
                    raise map_provider_error("Pollinations", Exception(error_msg), status_code=response.status_code, response_body=error_msg)

                async for line in response.aiter_lines():
                    if line:
                        try:
                            line = line.decode('utf-8').strip() if isinstance(line, bytes) else line.strip()
                            if not line or not line.startswith('data: '):
                                continue
                            data = line[6:]
                            if data == '[DONE]':
                                break
                            chunk = data.strip()
                            if not chunk:
                                continue
                            data_json = json.loads(chunk)
                            if 'choices' not in data_json or not data_json['choices']:
                                continue
                            choice = data_json['choices'][0]
                            if 'delta' not in choice:
                                continue
                            delta = choice['delta']
                            content = delta.get('content', '')
                            role = delta.get('role', 'assistant')

                            if not content:
                                continue

                            message = ChatMessage(role=role, content=content)
                            usage = {}
                            model = data_json.get('model', request.model)
                            yield ChatCompletionResponse(
                                message=message,
                                provider='pollinations',
                                model=model,
                                usage=usage,
                                raw_response=data_json
                            )
                        except json.JSONDecodeError:
                            continue

        except httpx.HTTPStatusError as e:
            response_body = await e.response.aread()
            raise map_provider_error("Pollinations", e, status_code=e.response.status_code, response_body=response_body)
        except httpx.RequestError as e:
            raise map_provider_error("Pollinations", e)

    def stream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> Iterator[ChatCompletionResponse]:
        """
        Stream a chat completion response from Pollinations.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional Pollinations-specific parameters.

        Returns:
            Iterator[ChatCompletionResponse]: An iterator of response chunks.

        Raises:
            Exception: If the request fails.
        """
        async def _async_gen():
            async for chunk in self.astream_complete(request, **provider_specific_kwargs):
                yield chunk

        gen = _async_gen()
        try:
            while True:
                try:
                    yield asyncio.run(gen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            gen.aclose()
