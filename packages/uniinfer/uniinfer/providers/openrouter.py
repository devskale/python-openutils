"""
OpenRouter provider implementation.

OpenRouter is a unified API to access multiple AI models from different providers.
"""
import json
import requests
from typing import AsyncIterator, Optional

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import map_provider_error, UniInferError


class OpenRouterProvider(ChatProvider):
    """
    Provider for OpenRouter API.

    OpenRouter provides a unified interface to access multiple AI models from
    different providers, including Anthropic, OpenAI, and more.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenRouter provider.

        Args:
            api_key (Optional[str]): The OpenRouter API key.
        """
        super().__init__(api_key)
        self.base_url = "https://openrouter.ai/api/v1"

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """
        List available models from OpenRouter.

        Args:
            api_key (Optional[str]): The OpenRouter API key. If not provided,
                                     it attempts to retrieve it using credgoo.

        Returns:
            list: A list of available free model IDs.

        Raises:
            ValueError: If no API key is provided or found.
            Exception: If the API request fails.
        """
        # Determine the API key to use
        if not api_key:
            try:
                from credgoo.credgoo import get_api_key
                api_key = get_api_key('openrouter')
            except ImportError:
                raise ValueError(
                    "credgoo not installed. Please provide an API key or install credgoo.")
            except Exception as e:
                raise ValueError(
                    f"Failed to get OpenRouter API key from credgoo: {e}")

        if not api_key:
            raise ValueError(
                "OpenRouter API key is required. Provide it directly or configure credgoo.")

        endpoint = "https://openrouter.ai/api/v1/models"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/uniinfer",
            "X-Title": "UniInfer"
        }

        try:
            response = requests.get(endpoint, headers=headers)

            if response.status_code != 200:
                error_msg = f"OpenRouter API error: {response.status_code} - {response.text}"
                raise map_provider_error("OpenRouter", Exception(error_msg), status_code=response.status_code, response_body=response.text)

            models = response.json().get('data', [])
            free_models = [
                model['id']
                for model in models
                if model.get('pricing', {}).get('prompt') == '0'
                and model.get('pricing', {}).get('completion') == '0'
            ]
            return free_models
        except Exception as e:
            status_code = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            response_body = getattr(e.response, 'text', None) if hasattr(e, 'response') else None
            raise map_provider_error("OpenRouter", e, status_code=status_code, response_body=response_body)

    def _flatten_messages(self, messages: list) -> list:
        """
        Flatten message content if it's a list of text objects.
        Some proxies/providers only support string content.
        """
        flattened_messages = []
        for msg in messages:
            msg_dict = msg.to_dict()
            content = msg_dict.get("content")
            
            if isinstance(content, list):
                # Check if it's a list of text objects
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                
                # If we found text parts, join them
                if text_parts:
                    msg_dict["content"] = "".join(text_parts)
            
            flattened_messages.append(msg_dict)
        return flattened_messages

    async def acomplete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """
        Make an async chat completion request to OpenRouter.
        """
        if self.api_key is None:
            raise ValueError("OpenRouter API key is required")

        endpoint = f"{self.base_url}/chat/completions"

        payload = {
            "model": request.model or "moonshotai/moonlight-16b-a3b-instruct:free",
            "messages": self._flatten_messages(request.messages),
            "temperature": request.temperature,
            "stream": False
        }

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice

        payload.update(provider_specific_kwargs)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/uniinfer",
            "X-Title": "UniInfer"
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
                error_msg = f"OpenRouter API error: {response.status_code} - {response.text}"
                raise map_provider_error("OpenRouter", Exception(error_msg), status_code=response.status_code, response_body=response.text)

            response_data = response.json()
            choice = response_data['choices'][0]
            message = ChatMessage(
                role=choice['message']['role'],
                content=choice['message'].get('content'),
                tool_calls=choice['message'].get('tool_calls')
            )

            return ChatCompletionResponse(
                message=message,
                provider='openrouter',
                model=response_data.get('model', request.model),
                usage=response_data.get('usage', {}),
                raw_response=response_data
            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("OpenRouter", e)

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> AsyncIterator[ChatCompletionResponse]:
        """
        Stream an async chat completion response from OpenRouter.
        """
        if self.api_key is None:
            raise ValueError("OpenRouter API key is required")

        endpoint = f"{self.base_url}/chat/completions"

        payload = {
            "model": request.model or "moonshotai/moonlight-16b-a3b-instruct:free",
            "messages": self._flatten_messages(request.messages),
            "temperature": request.temperature,
            "stream": True
        }

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice

        payload.update(provider_specific_kwargs)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/uniinfer",
            "X-Title": "UniInfer"
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
                    error_msg = f"OpenRouter API error: {response.status_code} - {await response.aread()}"
                    raise map_provider_error("OpenRouter", Exception(error_msg), status_code=response.status_code, response_body=error_msg)

                async for line in response.aiter_lines():
                    if line:
                        try:
                            if line.startswith('data: '):
                                line = line[6:]

                            if not line or line == '[DONE]':
                                continue

                            data = json.loads(line)

                            if 'choices' not in data or not data['choices']:
                                continue

                            choice = data['choices'][0]
                            delta = choice.get('delta', {})

                            if not delta.get('content') and not delta.get('tool_calls'):
                                continue

                            message = ChatMessage(
                                role=delta.get('role', 'assistant'),
                                content=delta.get('content'),
                                tool_calls=delta.get('tool_calls')
                            )

                            yield ChatCompletionResponse(
                                message=message,
                                provider='openrouter',
                                model=data.get('model', request.model),
                                usage={},
                                raw_response=data
                            )
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("OpenRouter", e)
