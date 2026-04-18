from __future__ import annotations
"""
AI21 provider implementation.
"""
from typing import Optional, AsyncIterator
import os

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import map_provider_error, UniInferError
import requests

try:
    from ai21 import AI21Client, AsyncAI21Client
    from ai21.models.chat import ChatMessage as AI21ChatMessage
    HAS_AI21 = True
except ImportError:
    HAS_AI21 = False


class AI21Provider(ChatProvider):
    """
    Provider for AI21 API.

    AI21 offers Jamba language models through its own client library.
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the AI21 provider.

        Args:
            api_key (Optional[str]): The AI21 API key.
            **kwargs: Additional configuration options.
        """
        super().__init__(api_key)

        if not HAS_AI21:
            raise ImportError(
                "ai21 package is required for the AI21Provider. "
                "Install it with: uv pip install ai21"
            )

        self.client = AI21Client(api_key=self.api_key or os.environ.get("AI21_API_KEY"))
        self.async_client = AsyncAI21Client(api_key=self.api_key or os.environ.get("AI21_API_KEY"))

    async def aclose(self):
        """Close the AI21 async client."""
        # AsyncAI21Client doesn't have an explicit close in some versions, 
        # but we should call it if it exists.
        if hasattr(self, 'async_client') and hasattr(self.async_client, 'close'):
            await self.async_client.close()
        await super().aclose()

    @staticmethod
    def list_models(api_key: Optional[str] = None) -> list[ModelInfo]:
        from ..core import ModelInfo
        """List available AI21 models."""
        if not api_key:
            try:
                from credgoo.credgoo import get_api_key
                api_key = get_api_key("ai21")
            except (ImportError, Exception):
                return []
        if not api_key:
            return []
        try:
            response = requests.get(
                "https://api.ai21.com/studio/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30,
            )
            if response.status_code != 200:
                return []
            results = []
            for model in response.json().get("data", []):
                if not isinstance(model, dict) or not model.get("id"):
                    continue
                pricing = model.get("pricing", {})
                cost = None
                if pricing.get("prompt") or pricing.get("completion"):
                    cost = {}
                    if pricing.get("prompt"):
                        cost["input"] = float(pricing["prompt"]) * 1_000_000
                    if pricing.get("completion"):
                        cost["output"] = float(pricing["completion"]) * 1_000_000
                results.append(ModelInfo(
                    id=model["id"],
                    name=model.get("name"),
                    type="chat",
                    context_window=model.get("context_length"),
                    max_output=model.get("max_completion_tokens"),
                    cost=cost,
                    owned_by=model.get("owned_by"),
                    raw=model,
                ))
            return results
        except Exception:
            return []

    async def acomplete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """Make an async chat completion request to AI21."""
        messages = []
        for msg in request.messages:
            role = msg.role if msg.role in ['user', 'assistant', 'system'] else 'user'
            messages.append(AI21ChatMessage(content=msg.content, role=role))

        params = {
            "model": request.model or "jamba-1.6-mini",
            "messages": messages,
            "temperature": request.temperature if request.temperature is not None else 0.7,
        }
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens

        params.update(provider_specific_kwargs)

        try:
            completion = await self.async_client.chat.completions.create(**params)
            content = completion.choices[0].message.content
            role = completion.choices[0].message.role

            message = ChatMessage(role=role, content=content)
            usage = {}
            if hasattr(completion, 'usage'):
                usage = {
                    "prompt_tokens": getattr(completion.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(completion.usage, "completion_tokens", 0),
                    "total_tokens": getattr(completion.usage, "total_tokens", 0)
                }

            return ChatCompletionResponse(
                message=message,
                provider='ai21',
                model=params["model"],
                usage=usage,
                raw_response=completion.to_dict() if hasattr(completion, 'to_dict') else str(completion)
            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("ai21", e)

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> AsyncIterator[ChatCompletionResponse]:
        """Stream an async chat completion response from AI21."""
        # AI21 SDK might not support streaming in all versions. 
        # If it doesn't, we fall back to a single chunk.
        try:
            resp = await self.acomplete(request, **provider_specific_kwargs)
            yield resp
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("ai21", e)
