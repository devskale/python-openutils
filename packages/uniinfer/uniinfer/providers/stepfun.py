"""
StepFun provider implementation.
"""
from typing import Dict, Any, Iterator, Optional, List, AsyncIterator
import os

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import map_provider_error, UniInferError

try:
    from openai import OpenAI, AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class StepFunProvider(ChatProvider):
    """
    Provider for StepFun API (阶跃星辰).
    """

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.stepfun.com/v1", **kwargs):
        """
        Initialize the StepFun provider.

        Args:
            api_key (Optional[str]): The StepFun API key.
            base_url (str): The base URL for the StepFun API.
            **kwargs: Additional configuration options.
        """
        super().__init__(api_key)
        self.base_url = base_url

        if not HAS_OPENAI:
            raise ImportError(
                "openai package is required for the StepFunProvider. "
                "Install it with: pip install openai"
            )

        self.client = OpenAI(api_key=self.api_key or os.environ.get("STEPFUN_API_KEY"), base_url=self.base_url)
        self.async_client = AsyncOpenAI(api_key=self.api_key or os.environ.get("STEPFUN_API_KEY"), base_url=self.base_url)

    async def aclose(self):
        """Close the StepFun async client."""
        if hasattr(self, 'async_client'):
            await self.async_client.close()
        await super().aclose()

    @classmethod
    def list_models(cls, api_key: Optional[str] = None, base_url: str = "https://api.stepfun.com/v1") -> List[str]:
        """List available models from StepFun."""
        import requests
        if api_key is None:
            try:
                from credgoo.credgoo import get_api_key
                api_key = get_api_key("stepfun")
            except ImportError:
                return ["step-1-8k", "step-1-32k"]

        if not api_key:
            return ["step-1-8k", "step-1-32k"]

        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(f"{base_url}/models", headers=headers)
            response.raise_for_status()
            models_data = response.json()
            return [model["id"] for model in models_data.get("data", [])]
        except Exception:
            return ["step-1-8k", "step-1-32k", "step-2-8k", "step-2-32k"]

    async def acomplete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """Make an async chat completion request to StepFun."""
        if self.api_key is None:
            raise ValueError("StepFun API key is required")

        params = {
            "model": request.model or "step-1-8k",
            "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
            "temperature": request.temperature,
        }

        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens

        params.update(provider_specific_kwargs)

        try:
            completion = await self.async_client.chat.completions.create(**params)
            message = ChatMessage(role=completion.choices[0].message.role, content=completion.choices[0].message.content)
            usage = {}
            if hasattr(completion, 'usage'):
                usage = {"prompt_tokens": completion.usage.prompt_tokens, "completion_tokens": completion.usage.completion_tokens, "total_tokens": completion.usage.total_tokens}

            return ChatCompletionResponse(
                message=message,
                provider='stepfun',
                model=params["model"],
                usage=usage,
                raw_response=completion.model_dump()
            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("stepfun", e)

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> AsyncIterator[ChatCompletionResponse]:
        """Stream an async chat completion response from StepFun."""
        if self.api_key is None:
            raise ValueError("StepFun API key is required")

        params = {
            "model": request.model or "step-1-8k",
            "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
            "temperature": request.temperature,
            "stream": True
        }

        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens

        params.update(provider_specific_kwargs)

        try:
            stream = await self.async_client.chat.completions.create(**params)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield ChatCompletionResponse(
                        message=ChatMessage(role="assistant", content=content),
                        provider='stepfun',
                        model=params["model"],
                        usage={},
                        raw_response=chunk.model_dump()
                    )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("stepfun", e)
