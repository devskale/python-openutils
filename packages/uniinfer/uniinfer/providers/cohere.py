"""
Cohere provider implementation.
"""
import os
from typing import Dict, Any, Iterator, Optional

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import map_provider_error, UniInferError

try:
    import cohere
    HAS_COHERE = True
except ImportError:
    HAS_COHERE = False


class CohereProvider(ChatProvider):
    """
    Provider for Cohere API.
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the Cohere provider.

        Args:
            api_key (Optional[str]): The Cohere API key.
            **kwargs: Additional configuration options.
        """
        if not api_key:
            from credgoo.credgoo import get_api_key
            api_key = get_api_key("cohere")
            if not api_key:
                raise ValueError("API key is required for CohereProvider")

        super().__init__(api_key)

        if not HAS_COHERE:
            raise ImportError(
                "cohere package is required for the CohereProvider. "
                "Install it with: pip install cohere"
            )

        # Initialize the Cohere clients
        self.client = cohere.ClientV2(api_key=self.api_key)
        self.async_client = cohere.AsyncClientV2(api_key=self.api_key)

    async def aclose(self):
        """Close the Cohere async client."""
        if hasattr(self, 'async_client'):
            # The cohere SDK handles closing via __aexit__ or explicit close if available
            # For now we'll just let it be GC'd if no explicit close is found, 
            # as typical SDK behavior.
            pass
        await super().aclose()

    async def acomplete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """
        Make an async chat completion request to Cohere.
        """
        if self.api_key is None:
            raise ValueError("Cohere API key is required")

        cohere_messages = []
        for msg in request.messages:
            cohere_messages.append({
                "role": msg.role,
                "content": msg.content
            })

        params = {
            "model": request.model or "command-r-plus-08-2024",
            "messages": cohere_messages,
        }

        if request.temperature is not None:
            params["temperature"] = request.temperature

        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens

        params.update(provider_specific_kwargs)

        try:
            response = await self.async_client.chat(**params)
            content = response.text

            message = ChatMessage(
                role="assistant",
                content=content
            )

            usage = {
                "input_tokens": getattr(response, "input_tokens", 0),
                "output_tokens": getattr(response, "output_tokens", 0),
                "total_tokens": getattr(response, "input_tokens", 0) + getattr(response, "output_tokens", 0)
            }

            return ChatCompletionResponse(
                message=message,
                provider='cohere',
                model=params["model"],
                usage=usage,
                raw_response=response
            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            status_code = getattr(e, 'status_code', None)
            response_body = str(e)
            if hasattr(e, 'response') and e.response is not None:
                status_code = getattr(e.response, 'status_code', status_code)
                response_body = getattr(e.response, 'text', response_body)
            elif hasattr(e, 'body') and isinstance(e.body, dict):
                response_body = str(e.body)
            
            raise map_provider_error("cohere", e, status_code=status_code, response_body=response_body)

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """
        List available models from Cohere.
        """
        if not HAS_COHERE:
            raise ImportError(
                "cohere package is required for the CohereProvider. "
                "Install it with: pip install cohere"
            )

        if not api_key:
            api_key = os.getenv("COHERE_API_KEY")
            if not api_key:
                try:
                    from credgoo.credgoo import get_api_key
                    api_key = get_api_key("cohere")
                except ImportError:
                    pass
        
        if not api_key:
            return []

        try:
            client = cohere.ClientV2(api_key=api_key)
            response = client.models.list()
            return [model.name for model in response.models]
        except Exception:
            return []

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> AsyncIterator[ChatCompletionResponse]:
        """
        Stream an async chat completion response from Cohere.
        """
        if self.api_key is None:
            raise ValueError("Cohere API key is required")

        cohere_messages = []
        for msg in request.messages:
            cohere_messages.append({
                "role": msg.role,
                "content": msg.content
            })

        params = {
            "model": request.model or "command-r-plus-08-2024",
            "messages": cohere_messages,
        }

        if request.temperature is not None:
            params["temperature"] = request.temperature

        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens

        params.update(provider_specific_kwargs)

        try:
            async for event in await self.async_client.chat_stream(**params):
                if event.type == "content-delta":
                    content = event.delta.message.content.text
                    
                    message = ChatMessage(
                        role="assistant",
                        content=content
                    )

                    yield ChatCompletionResponse(
                        message=message,
                        provider='cohere',
                        model=params["model"],
                        usage={},
                        raw_response=event
                    )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            status_code = getattr(e, 'status_code', None)
            response_body = str(e)
            if hasattr(e, 'response') and e.response is not None:
                status_code = getattr(e.response, 'status_code', status_code)
                response_body = getattr(e.response, 'text', response_body)
            elif hasattr(e, 'body') and isinstance(e.body, dict):
                response_body = str(e.body)
                
            raise map_provider_error("cohere", e, status_code=status_code, response_body=response_body)
