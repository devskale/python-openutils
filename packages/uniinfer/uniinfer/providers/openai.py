"""
OpenAI provider implementation.
"""
import json
import requests
import httpx
from typing import Iterator, Optional, AsyncIterator

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import map_provider_error, UniInferError


class OpenAIProvider(ChatProvider):
    """
    Provider for OpenAI API.
    """

    def __init__(self, api_key: Optional[str] = None, organization: Optional[str] = None):
        """
        Initialize the OpenAI provider.

        Args:
            api_key (Optional[str]): The OpenAI API key.
            organization (Optional[str]): The OpenAI organization ID.
        """
        super().__init__(api_key)
        self.base_url = "https://api.openai.com/v1"
        self.organization = organization

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """
        List available models from OpenAI using the API.

        Returns:
            list: A list of available model IDs.
        """
        if not api_key:
            raise ValueError("API key is required to list models")

        url = "https://api.openai.com/v1/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                raise map_provider_error("OpenAI", Exception(
                    f"OpenAI API error: {response.status_code} - {response.text}"), status_code=response.status_code, response_body=response.text)

            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        except Exception as e:
            status_code = getattr(e.response, 'status_code', None) if hasattr(
                e, 'response') else None
            response_body = getattr(e.response, 'text', None) if hasattr(
                e, 'response') else str(e)
            raise map_provider_error(
                "OpenAI", e, status_code=status_code, response_body=response_body)

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
        Make an async chat completion request to OpenAI.
        """
        if self.api_key is None:
            raise ValueError("OpenAI API key is required")

        endpoint = f"{self.base_url}/chat/completions"

        payload = {
            "model": request.model or "gpt-3.5-turbo",
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
            "Authorization": f"Bearer {self.api_key}"
        }

        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        client = await self._get_async_client()
        try:
            response = await client.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=60.0
            )

            if response.status_code != 200:
                error_msg = f"OpenAI API error: {response.status_code} - {response.text}"
                raise map_provider_error("OpenAI", Exception(error_msg), status_code=response.status_code, response_body=response.text)

            response_data = response.json()
            choice = response_data['choices'][0]
            message = ChatMessage(
                role=choice['message']['role'],
                content=choice['message'].get('content'),
                tool_calls=choice['message'].get('tool_calls')
            )

            return ChatCompletionResponse(
                message=message,
                provider='openai',
                model=response_data.get('model', request.model),
                usage=response_data.get('usage', {}),
                raw_response=response_data,
                finish_reason=choice.get('finish_reason')
            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("OpenAI", e)

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> AsyncIterator[ChatCompletionResponse]:
        """
        Stream an async chat completion response from OpenAI.
        """
        if self.api_key is None:
            raise ValueError("OpenAI API key is required")

        endpoint = f"{self.base_url}/chat/completions"

        payload = {
            "model": request.model or "gpt-3.5-turbo",
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
            "Authorization": f"Bearer {self.api_key}"
        }

        if self.organization:
            headers["OpenAI-Organization"] = self.organization

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
                    error_msg = f"OpenAI API error: {response.status_code} - {await response.aread()}"
                    raise map_provider_error("OpenAI", Exception(error_msg), status_code=response.status_code, response_body=error_msg)

                async for line in response.aiter_lines():
                    if line:
                        try:
                            if not line or line == 'data: [DONE]' or not line.startswith('data: '):
                                continue

                            data_str = line[6:]
                            data = json.loads(data_str)

                            if len(data['choices']) > 0:
                                choice = data['choices'][0]
                                finish_reason = choice.get('finish_reason')
                                delta = choice.get('delta', {})

                                if not delta.get('content') and not delta.get('tool_calls') and not finish_reason:
                                    continue

                                message = ChatMessage(
                                    role=delta.get('role', 'assistant'),
                                    content=delta.get('content'),
                                    tool_calls=delta.get('tool_calls')
                                )

                                yield ChatCompletionResponse(
                                    message=message,
                                    provider='openai',
                                    model=data.get('model', request.model),
                                    usage={},
                                    raw_response=data,
                                    finish_reason=finish_reason
                                )
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("OpenAI", e)
