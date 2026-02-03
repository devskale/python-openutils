"""
Pollinations provider implementation.

Pollinations is a unified API to access multiple AI models from different providers.
"""
from typing import Optional, AsyncIterator
import json

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import map_provider_error, UniInferError


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

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """
        List available models from Pollinations.
        """
        import requests
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

        try:
            response = requests.get(endpoint, headers=headers)
            if response.status_code != 200:
                error_msg = f"Pollinations API error: {response.status_code} - {response.text}"
                raise map_provider_error("Pollinations", Exception(error_msg), status_code=response.status_code, response_body=response.text)

            models = response.json()
            if isinstance(models, list):
                return [model.get('name', '') for model in models if isinstance(model, dict) and model.get('name')]
            if isinstance(models, dict) and 'data' in models:
                return [model.get('id', '') for model in models['data'] if model.get('id')]
            return []
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("Pollinations", e)

    async def acomplete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """
        Make an async text generation request to Pollinations.
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

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            client = await self._get_async_client()
            response = await client.post(self.base_url, headers=headers, json=payload, timeout=60.0)

            if response.status_code != 200:
                error_msg = f"Pollinations API error: {response.status_code} - {response.text}"
                raise map_provider_error("Pollinations", Exception(error_msg), status_code=response.status_code, response_body=response.text)

            response_data = response.json()
            if 'choices' not in response_data or not response_data['choices']:
                raise Exception("API returned invalid response")

            choice = response_data['choices'][0]
            content = choice.get('message', {}).get('content', '')

            message = ChatMessage(role="assistant", content=content)

            return ChatCompletionResponse(
                message=message,
                provider='pollinations',
                model=request.model,
                usage=response_data.get('usage', {}),
                raw_response=response_data
            )

        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("Pollinations", e)

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> AsyncIterator[ChatCompletionResponse]:
        """
        Stream an async chat completion response from Pollinations.
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

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            client = await self._get_async_client()
            async with client.stream("POST", self.base_url, headers=headers, json=payload, timeout=60.0) as response:
                if response.status_code != 200:
                    error_msg = f"Pollinations API error: {response.status_code} - {await response.aread()}"
                    raise map_provider_error("Pollinations", Exception(error_msg), status_code=response.status_code, response_body=error_msg)

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith('data: '):
                        continue
                    
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    
                    try:
                        data_json = json.loads(data)
                        if 'choices' not in data_json or not data_json['choices']:
                            continue
                        
                        choice = data_json['choices'][0]
                        if 'delta' not in choice:
                            continue
                        
                        delta = choice['delta']
                        content = delta.get('content', '')
                        if not content:
                            continue

                        yield ChatCompletionResponse(
                            message=ChatMessage(role=delta.get('role', 'assistant'), content=content),
                            provider='pollinations',
                            model=data_json.get('model', request.model),
                            usage={},
                            raw_response=data_json
                        )
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("Pollinations", e)
