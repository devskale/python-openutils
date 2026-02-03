"""
ArliAI provider implementation.
"""
import json
from typing import Optional, AsyncIterator

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import map_provider_error, UniInferError


class ArliAIProvider(ChatProvider):
    """
    Provider for ArliAI API.
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the ArliAI provider.

        Args:
            api_key (Optional[str]): The ArliAI API key.
            **kwargs: Additional configuration options.
        """
        super().__init__(api_key)
        self.base_url = "https://api.arliai.com/v1"

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """
        List available models from ArliAI.

        Args:
            api_key (Optional[str]): The ArliAI API key.

        Returns:
            list: A list of available model names.

        Raises:
            ValueError: If no API key is provided
            Exception: If API request fails
        """
        if not api_key:
            raise ValueError("API key is required to list models")

        endpoint = "https://api.arliai.com/v1/models"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        try:
            import requests
            response = requests.get(endpoint, headers=headers)

            if response.status_code != 200:
                raise map_provider_error("ArliAI", Exception(f"ArliAI API error: {response.status_code} - {response.text}"), status_code=response.status_code, response_body=response.text)

            models_data = response.json()
            return [model["id"] for model in models_data.get("data", [])
                    if model["id"] and model["id"][0].isalpha()]
        except Exception as e:
            status_code = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            response_body = getattr(e.response, 'text', None) if hasattr(e, 'response') else None
            raise map_provider_error("ArliAI", e, status_code=status_code, response_body=response_body)

    async def acomplete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """
        Make an async chat completion request to ArliAI.
        """
        if self.api_key is None:
            raise ValueError("ArliAI API key is required")

        endpoint = f"{self.base_url}/chat/completions"

        payload = {
            "model": request.model or "Mistral-Nemo-12B-Instruct-2407",
            "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
            "temperature": request.temperature,
            "stream": False
        }

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        # ArliAI default parameters
        if "repetition_penalty" not in provider_specific_kwargs:
            payload["repetition_penalty"] = 1.1
        if "top_p" not in provider_specific_kwargs:
            payload["top_p"] = 0.9
        if "top_k" not in provider_specific_kwargs:
            payload["top_k"] = 40

        payload.update(provider_specific_kwargs)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
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
                error_msg = f"ArliAI API error: {response.status_code} - {response.text}"
                raise map_provider_error("ArliAI", Exception(error_msg), status_code=response.status_code, response_body=response.text)

            response_data = response.json()
            choice = response_data.get("choices", [{}])[0]
            message_data = choice.get("message", {})

            message = ChatMessage(
                role=message_data.get("role", "assistant"),
                content=message_data.get("content", "")
            )

            usage = response_data.get("usage", {})

            return ChatCompletionResponse(
                message=message,
                provider='arli',
                model=response_data.get('model', request.model),
                usage=usage,
                raw_response=response_data
            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("ArliAI", e)

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> AsyncIterator[ChatCompletionResponse]:
        """
        Stream an async chat completion response from ArliAI.
        """
        if self.api_key is None:
            raise ValueError("ArliAI API key is required")

        endpoint = f"{self.base_url}/chat/completions"

        payload = {
            "model": request.model or "Mistral-Nemo-12B-Instruct-2407",
            "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
            "temperature": request.temperature,
            "stream": True
        }

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        else:
            payload["max_tokens"] = 1024

        if "repetition_penalty" not in provider_specific_kwargs:
            payload["repetition_penalty"] = 1.1
        if "top_p" not in provider_specific_kwargs:
            payload["top_p"] = 0.9
        if "top_k" not in provider_specific_kwargs:
            payload["top_k"] = 40

        payload.update(provider_specific_kwargs)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
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
                    error_msg = f"ArliAI API error: {response.status_code} - {await response.aread()}"
                    raise map_provider_error("ArliAI", Exception(error_msg), status_code=response.status_code, response_body=error_msg)

                async for line in response.aiter_lines():
                    if line:
                        try:
                            if not line or line == 'data: [DONE]':
                                continue

                            if line.startswith('data: '):
                                line = line[6:]

                            data = json.loads(line)

                            if 'choices' not in data or not data['choices']:
                                continue

                            choice = data['choices'][0]
                            content = ""
                            role = "assistant"

                            if 'delta' in choice:
                                delta = choice['delta']
                                content = delta.get('content', '')
                                role = delta.get('role', 'assistant')
                            elif 'message' in choice:
                                message = choice['message']
                                content = message.get('content', '')
                                role = message.get('role', 'assistant')

                            if not content:
                                continue

                            message = ChatMessage(role=role, content=content)
                            yield ChatCompletionResponse(
                                message=message,
                                provider='arli',
                                model=data.get('model', request.model),
                                usage={},
                                raw_response=data
                            )
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("ArliAI", e)
