"""
NVIDIA GPU Cloud (NGC) provider implementation.
Uses OpenAI-compatible API.
"""
from typing import Optional, List, AsyncIterator

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import map_provider_error, UniInferError

try:
    from openai import OpenAI, AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class NGCProvider(ChatProvider):
    """
    Provider for NVIDIA GPU Cloud (NGC) API.
    NGC provides an OpenAI-compatible API for various models.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://integrate.api.nvidia.com/v1", **kwargs):
        """
        Initialize the NGC provider.

        Args:
            api_key (Optional[str]): The NGC API key (required if outside NGC environment).
            base_url (str): The base URL for the NGC API.
            **kwargs: Additional provider-specific configuration parameters.
        """
        super().__init__(api_key)
        self.base_url = base_url

        if not HAS_OPENAI:
            raise ImportError(
                "The 'openai' package is required to use the NGC provider. "
                "Install it with 'pip install openai>=1.0.0'"
            )

        self.client = OpenAI(base_url=base_url, api_key=self.api_key)
        self.async_client = AsyncOpenAI(base_url=base_url, api_key=self.api_key)
        self.config = kwargs

    async def aclose(self):
        """Close the NGC async client."""
        if hasattr(self, 'async_client'):
            await self.async_client.close()
        await super().aclose()

    @classmethod
    def list_models(cls, api_key: Optional[str] = None, base_url: str = "https://integrate.api.nvidia.com/v1") -> List[str]:
        """List available models from NGC catalog."""
        import requests
        if api_key is None:
            try:
                from credgoo.credgoo import get_api_key
                api_key = get_api_key("ngc")
            except ImportError:
                return []

        if not api_key:
            return []

        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(f"{base_url}/models", headers=headers)
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
        """Make an async chat completion request to NGC."""
        try:
            def _flatten(content):
                if isinstance(content, list):
                    return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
                return content

            messages = [{"role": msg.role, "content": _flatten(msg.content)} for msg in request.messages]

            params = {
                "model": request.model,
                "messages": messages,
                "temperature": request.temperature,
                "stream": False,
            }
            if request.max_tokens is not None:
                params["max_tokens"] = request.max_tokens
            if request.tools:
                params["tools"] = request.tools
            if request.tool_choice:
                params["tool_choice"] = request.tool_choice

            params.update(provider_specific_kwargs)

            completion = await self.async_client.chat.completions.create(**params)
            response_message = completion.choices[0].message
            
            tool_calls = None
            if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
                tool_calls = [{"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in response_message.tool_calls]
            
            message = ChatMessage(role="assistant", content=response_message.content, tool_calls=tool_calls)
            usage = {}
            if hasattr(completion, "usage"):
                usage = {"prompt_tokens": completion.usage.prompt_tokens, "completion_tokens": completion.usage.completion_tokens, "total_tokens": completion.usage.total_tokens}

            return ChatCompletionResponse(
                message=message,
                provider='ngc',
                model=request.model,
                usage=usage,
                raw_response=completion.model_dump()
            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("ngc", e)

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> AsyncIterator[ChatCompletionResponse]:
        """Stream an async chat completion response from NGC."""
        try:
            def _flatten(content):
                if isinstance(content, list):
                    return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
                return content

            messages = [{"role": msg.role, "content": _flatten(msg.content)} for msg in request.messages]

            params = {
                "model": request.model,
                "messages": messages,
                "temperature": request.temperature,
                "stream": True,
            }
            if request.max_tokens is not None:
                params["max_tokens"] = request.max_tokens
            if request.tools:
                params["tools"] = request.tools
            if request.tool_choice:
                params["tool_choice"] = request.tool_choice

            params.update(provider_specific_kwargs)

            stream = await self.async_client.chat.completions.create(**params)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    tool_calls = None
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        tool_calls = [{"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in delta.tool_calls]
                    
                    if content or tool_calls:
                        yield ChatCompletionResponse(
                            message=ChatMessage(role="assistant", content=content, tool_calls=tool_calls),
                            provider='ngc',
                            model=request.model,
                            usage={},
                            raw_response=chunk.model_dump()
                        )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("ngc", e)
