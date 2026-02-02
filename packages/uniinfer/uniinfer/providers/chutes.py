"""
Chutes provider implementation.

Chutes is a unified API to access multiple AI models from different providers.
"""
import json
from typing import Dict, Any, Iterator, Optional, AsyncIterator

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import map_provider_error, UniInferError


class ChutesProvider(ChatProvider):
    """
    Provider for Chutes API.

    Chutes provides a unified interface to access multiple AI models from
    different providers, including Anthropic, OpenAI, and more.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Chutes provider.

        Args:
            api_key (Optional[str]): The Chutes API key.
        """
        super().__init__(api_key)
        self.base_url = "https://llm.chutes.ai/v1"

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """List available models from Chutes."""
        import requests
        if not api_key:
            try:
                from credgoo.credgoo import get_api_key
                api_key = get_api_key('chutes')
            except ImportError:
                return ["deepseek-ai/DeepSeek-V3"]

        if not api_key:
            return ["deepseek-ai/DeepSeek-V3"]

        try:
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            response = requests.get(f"https://llm.chutes.ai/v1/models", headers=headers)
            response.raise_for_status()
            models = response.json().get('data', [])
            return [model['id'] for model in models]
        except Exception:
            return ["deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1"]

    def _flatten_messages(self, messages: list) -> list:
        """Flatten message content if it's a list of text objects."""
        flattened_messages = []
        for msg in messages:
            msg_dict = msg.to_dict()
            content = msg_dict.get("content")
            if isinstance(content, list):
                text_parts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
                if text_parts:
                    msg_dict["content"] = "".join(text_parts)
            flattened_messages.append(msg_dict)
        return flattened_messages

    async def acomplete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """Make an async chat completion request to Chutes."""
        if self.api_key is None:
            raise ValueError("Chutes API key is required")

        client = await self._get_async_client()
        endpoint = f"{self.base_url}/chat/completions"

        payload = {
            "model": request.model or "deepseek-ai/DeepSeek-V3",
            "messages": self._flatten_messages(request.messages),
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice

        payload.update(provider_specific_kwargs)
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        try:
            response = await client.post(endpoint, headers=headers, json=payload, timeout=60.0)
            if response.status_code != 200:
                raise map_provider_error("Chutes", Exception(response.text), status_code=response.status_code, response_body=response.text)
            
            resp_data = response.json()
            choice = resp_data['choices'][0]
            message = ChatMessage(
                role=choice['message']['role'],
                content=choice['message'].get('content'),
                tool_calls=choice['message'].get('tool_calls')
            )
            return ChatCompletionResponse(
                message=message,
                provider='chutes',
                model=resp_data.get('model', request.model),
                usage=resp_data.get('usage', {}),
                raw_response=resp_data
            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("Chutes", e)

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> AsyncIterator[ChatCompletionResponse]:
        """Stream an async chat completion response from Chutes."""
        if self.api_key is None:
            raise ValueError("Chutes API key is required")

        client = await self._get_async_client()
        endpoint = f"{self.base_url}/chat/completions"

        payload = {
            "model": request.model or "deepseek-ai/DeepSeek-V3",
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
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        try:
            async with client.stream("POST", endpoint, headers=headers, json=payload, timeout=60.0) as response:
                if response.status_code != 200:
                    raise map_provider_error("Chutes", Exception(await response.aread()), status_code=response.status_code, response_body=await response.aread())
                
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data_json = json.loads(data_str)
                        if not data_json.get('choices'):
                            continue
                        choice = data_json['choices'][0]
                        delta = choice.get('delta', {})
                        
                        if not delta.get('content') and not delta.get('tool_calls'):
                            continue

                        yield ChatCompletionResponse(
                            message=ChatMessage(
                                role=delta.get('role', 'assistant'),
                                content=delta.get('content'),
                                tool_calls=delta.get('tool_calls')
                            ),
                            provider='chutes',
                            model=data_json.get('model', request.model),
                            usage={},
                            raw_response=data_json
                        )
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("Chutes", e)
