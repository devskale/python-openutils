"""
Mistral AI provider implementation.
"""
import asyncio
import httpx
import json
import requests
from typing import Dict, Any, Iterator, Optional, AsyncIterator

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import map_provider_error


class MistralProvider(ChatProvider):
    """
    Provider for Mistral AI API.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Mistral provider.

        Args:
            api_key (Optional[str]): The Mistral API key.
        """
        super().__init__(api_key)
        self.base_url = "https://api.mistral.ai/v1"

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """
        List available models from Mistral AI.

        Args:
            api_key (Optional[str]): The Mistral API key. If not provided,
                                     it attempts to retrieve it using credgoo.

        Returns:
            list: A list of available model names.

        Raises:
            ValueError: If no API key is provided or found.
            Exception: If the API request fails.
        """
        if not api_key:
            try:
                from credgoo.credgoo import get_api_key
                # Get API key from credgoo
                api_key = get_api_key("mistral")
            except ImportError:
                raise ValueError(
                    "credgoo not installed. Please provide an API key or install credgoo.")
            except Exception as e:
                raise ValueError(
                    f"Failed to get Mistral API key from credgoo: {e}")

        if not api_key:
            raise ValueError(
                "Mistral API key is required. Provide it directly or configure credgoo.")

        endpoint = "https://api.mistral.ai/v1/models"
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        response = requests.get(endpoint, headers=headers)

        if response.status_code != 200:
            raise map_provider_error("Mistral", Exception(f"Failed to fetch models: {response.status_code} - {response.text}"), status_code=response.status_code, response_body=response.text)

        models_data = response.json()
        return [model["id"] for model in models_data["data"]]

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

    def complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """
        Make a chat completion request to Mistral AI.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional Mistral-specific parameters.

        Returns:
            ChatCompletionResponse: The completion response.

        Raises:
            Exception: If the request fails.
        """
        if self.api_key is None:
            raise ValueError("Mistral API key is required")

        endpoint = f"{self.base_url}/chat/completions"

        # Prepare the request payload
        payload = {
            "model": request.model,
            "messages": self._flatten_messages(request.messages),
            "temperature": request.temperature,
            "stream": False
        }

        # Add max_tokens if provided
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        # Add tools and tool_choice if provided
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice

        # Add any provider-specific parameters
        payload.update(provider_specific_kwargs)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        response = requests.post(
            endpoint,
            headers=headers,
            data=json.dumps(payload)
        )

        # Handle error response
        if response.status_code != 200:
            error_msg = f"Mistral API error: {response.status_code} - {response.text}"
            raise map_provider_error("Mistral", Exception(error_msg), status_code=response.status_code, response_body=response.text)

        # Parse the response
        response_data = response.json()
        choice = response_data['choices'][0]
        message = ChatMessage(
            role=choice['message']['role'],
            content=choice['message'].get('content'),
            tool_calls=choice['message'].get('tool_calls')
        )

        return ChatCompletionResponse(
            message=message,
            provider='mistral',
            model=response_data.get('model', request.model),
            usage=response_data.get('usage', {}),
            raw_response=response_data
        )

    def stream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> Iterator[ChatCompletionResponse]:
        """
        Stream a chat completion response from Mistral AI.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional Mistral-specific parameters.

        Returns:
            Iterator[ChatCompletionResponse]: An iterator of response chunks.

        Raises:
            Exception: If the request fails.
        """
        if self.api_key is None:
            raise ValueError("Mistral API key is required")

        endpoint = f"{self.base_url}/chat/completions"

        # Prepare the request payload
        payload = {
            "model": request.model,
            "messages": self._flatten_messages(request.messages),
            "temperature": request.temperature,
            "stream": True
        }

        # Add max_tokens if provided
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        # Add tools and tool_choice if provided
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice

        # Add any provider-specific parameters
        payload.update(provider_specific_kwargs)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        with requests.post(
            endpoint,
            headers=headers,
            data=json.dumps(payload),
            stream=True
        ) as response:
            # Handle error response
            if response.status_code != 200:
                error_msg = f"Mistral API error: {response.status_code} - {response.text}"
                raise map_provider_error("Mistral", Exception(error_msg), status_code=response.status_code, response_body=response.text)

            # Process the streaming response
            for line in response.iter_lines():
                if line:
                    # Parse the JSON data from the stream
                    try:
                        data = line.decode('utf-8')
                        if data.startswith('data: '):
                            data = data[6:]  # Remove 'data: ' prefix

                        # Skip empty lines or [DONE]
                        if not data or data == '[DONE]':
                            continue

                        chunk = json.loads(data)

                        if 'choices' in chunk:
                            choice = chunk['choices'][0]
                            role = choice['delta'].get('role', 'assistant')
                            # Get content and tool_calls from delta
                            content = choice['delta'].get('content')
                            tool_calls = choice['delta'].get('tool_calls')

                            # Create a message for this chunk
                            message = ChatMessage(
                                role=role,
                                content=content,
                                tool_calls=tool_calls
                            )

                            yield ChatCompletionResponse(
                                message=message,
                                provider='mistral',
                                model=chunk.get('model', request.model),
                                usage=chunk.get('usage', {}),
                                raw_response=chunk
                            )
                    except json.JSONDecodeError:
                        # Skip invalid JSON lines
                        continue

    async def acomplete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """
        Make an async chat completion request to Mistral AI.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional Mistral-specific parameters.

        Returns:
            ChatCompletionResponse: The completion response.

        Raises:
            Exception: If the request fails.
        """
        if self.api_key is None:
            raise ValueError("Mistral API key is required")

        endpoint = f"{self.base_url}/chat/completions"

        payload = {
            "model": request.model,
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

        async with httpx.AsyncClient() as client:
            response = await client.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=60.0
            )

            if response.status_code != 200:
                error_msg = f"Mistral API error: {response.status_code} - {response.text}"
                raise map_provider_error("Mistral", Exception(error_msg), status_code=response.status_code, response_body=response.text)

            response_data = response.json()
            choice = response_data['choices'][0]
            message = ChatMessage(
                role=choice['message']['role'],
                content=choice['message'].get('content'),
                tool_calls=choice['message'].get('tool_calls')
            )

            return ChatCompletionResponse(
                message=message,
                provider='mistral',
                model=response_data.get('model', request.model),
                usage=response_data.get('usage', {}),
                raw_response=response_data
            )

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> AsyncIterator[ChatCompletionResponse]:
        """
        Stream an async chat completion response from Mistral AI.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional Mistral-specific parameters.

        Returns:
            AsyncIterator[ChatCompletionResponse]: An async iterator of response chunks.

        Raises:
            Exception: If the request fails.
        """
        if self.api_key is None:
            raise ValueError("Mistral API key is required")

        endpoint = f"{self.base_url}/chat/completions"

        payload = {
            "model": request.model,
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

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                endpoint,
                headers=headers,
                json=payload,
                timeout=60.0
            ) as response:
                if response.status_code != 200:
                    error_msg = f"Mistral API error: {response.status_code} - {await response.aread()}"
                    raise map_provider_error("Mistral", Exception(error_msg), status_code=response.status_code, response_body=error_msg)

                async for line in response.aiter_lines():
                    if line:
                        try:
                            if line.startswith('data: '):
                                line = line[6:]

                            if not line or line == '[DONE]':
                                continue

                            chunk = json.loads(line)

                            if 'choices' in chunk:
                                choice = chunk['choices'][0]
                                role = choice['delta'].get('role', 'assistant')
                                content = choice['delta'].get('content')
                                tool_calls = choice['delta'].get('tool_calls')

                                message = ChatMessage(
                                    role=role,
                                    content=content,
                                    tool_calls=tool_calls
                                )

                                yield ChatCompletionResponse(
                                    message=message,
                                    provider='mistral',
                                    model=chunk.get('model', request.model),
                                    usage=chunk.get('usage', {}),
                                    raw_response=chunk
                                )
                        except json.JSONDecodeError:
                            continue
