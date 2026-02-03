"""
Moonshot provider implementation.
"""
from typing import Optional, AsyncIterator

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import map_provider_error, UniInferError

try:
    from openai import OpenAI, AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class MoonshotProvider(ChatProvider):
    """
    Provider for Moonshot AI API.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.moonshot.cn/v1", **kwargs):
        """
        Initialize the Moonshot provider.

        Args:
            api_key (Optional[str]): The Moonshot API key.
            base_url (str): The base URL for the Moonshot API.
            **kwargs: Additional configuration options.
        """
        super().__init__(api_key)
        self.base_url = base_url

        if not HAS_OPENAI:
            raise ImportError(
                "openai package is required for the MoonshotProvider. "
                "Install it with: pip install openai"
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def aclose(self):
        """Close the Moonshot async client."""
        if hasattr(self, 'async_client'):
            await self.async_client.close()
        await super().aclose()

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """List available models from Moonshot AI."""
        import requests
        if api_key is None:
            try:
                from credgoo.credgoo import get_api_key
                api_key = get_api_key("moonshot")
            except ImportError:
                return []

        if not api_key:
            return []

        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get("https://api.moonshot.cn/v1/models", headers=headers)
            response.raise_for_status()
            models_data = response.json()
            return [model["id"] for model in models_data.get("data", [])]
        except Exception:
            return []

    async def acomplete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """Make an async chat completion request to Moonshot."""
        if self.api_key is None:
            raise ValueError("Moonshot API key is required")

        params = {
            "model": request.model or "moonshot-v1-8k",
            "messages": [msg.to_dict() for msg in request.messages],
            "temperature": request.temperature,
        }

        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens
        if request.tools:
            params["tools"] = request.tools
        if request.tool_choice:
            params["tool_choice"] = request.tool_choice

        params.update(provider_specific_kwargs)

        try:
            completion = await self.async_client.chat.completions.create(**params)
            response_message = completion.choices[0].message
            
            tool_calls = None
            if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
                tool_calls = [{"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in response_message.tool_calls]
            
            message = ChatMessage(role=response_message.role, content=response_message.content, tool_calls=tool_calls)
            usage = {}
            if hasattr(completion, 'usage'):
                usage = {"prompt_tokens": completion.usage.prompt_tokens, "completion_tokens": completion.usage.completion_tokens, "total_tokens": completion.usage.total_tokens}

            return ChatCompletionResponse(
                message=message,
                provider='moonshot',
                model=params["model"],
                usage=usage,
                raw_response=completion.model_dump()
            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("moonshot", e)

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> AsyncIterator[ChatCompletionResponse]:
        """Stream an async chat completion response from Moonshot."""
        if self.api_key is None:
            raise ValueError("Moonshot API key is required")

        params = {
            "model": request.model or "moonshot-v1-8k",
            "messages": [msg.to_dict() for msg in request.messages],
            "temperature": request.temperature,
            "stream": True
        }

        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens
        if request.tools:
            params["tools"] = request.tools
        if request.tool_choice:
            params["tool_choice"] = request.tool_choice

        params.update(provider_specific_kwargs)

        try:
            stream = await self.async_client.chat.completions.create(**params)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    content = getattr(delta, 'content', None)
                    tool_calls = None
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        tool_calls = [{"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in delta.tool_calls]
                    
                    if content or tool_calls:
                        yield ChatCompletionResponse(
                            message=ChatMessage(role="assistant", content=content, tool_calls=tool_calls),
                            provider='moonshot',
                            model=params["model"],
                            usage={},
                            raw_response=chunk.model_dump()
                        )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("moonshot", e)
