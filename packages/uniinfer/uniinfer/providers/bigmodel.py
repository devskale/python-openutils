"""
Bigmodel provider implementation.
"""
import requests
import json
import httpx
from typing import AsyncIterator, Optional

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import map_provider_error, UniInferError

try:
    from openai import OpenAI, AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class BigmodelProvider(ChatProvider):
    """
    Provider for Bigmodel AI API.

    Bigmodel AI is a China-based LLM provider that uses the OpenAI client format.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.z.ai/api/paas/v4/", **kwargs): 
        """
        Initialize the Bigmodel provider.

        Args:
            api_key (Optional[str]): The Bigmodel API key.
            base_url (str): The base URL for the Bigmodel API.
            **kwargs: Additional configuration options.
        """
        super().__init__(api_key)
        self.base_url = base_url

        # Initialize OpenAI clients if available; otherwise fall back to HTTP calls
        if HAS_OPENAI:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            self.async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        else:
            self.client = None
            self.async_client = None

    @classmethod
    def list_models(cls, api_key: Optional[str] = None, base_url: str = "https://api.z.ai/api/paas/v4/") -> list:
        """
        List available models from Bigmodel AI using the API.
        Ensures that glm-4-flash and glm-4.5-flash are always included.

        Args:
            api_key (Optional[str]): The Bigmodel API key.
            base_url (str): The base URL for the Bigmodel API.

        Returns:
            list: A list of available model IDs, including guaranteed models.

        Raises:
            ValueError: If API key is not provided.
            Exception: If the API request fails.
        """
        if not api_key:
            raise ValueError("API key is required to list models")
        
        # Define guaranteed models that should always be available
        guaranteed_models = ["glm-4.7", "glm-4-flash", "glm-4.5-flash"]
        
        url = f"{base_url.rstrip('/')}/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise map_provider_error("Bigmodel", Exception(f"Bigmodel API error: {response.status_code} - {response.text}"), status_code=response.status_code, response_body=response.text)

        data = response.json()
        api_models = [model["id"] for model in data.get("data", [])]
        
        # Combine API models with guaranteed models, removing duplicates while preserving order
        all_models = list(dict.fromkeys(api_models + guaranteed_models))
        
        return all_models

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
        Make an async chat completion request to Bigmodel.
        """
        if self.api_key is None:
            raise ValueError("Bigmodel API key is required")

        messages = self._flatten_messages(request.messages)
        params = {
            "model": request.model or "glm-4.7",
            "messages": messages,
            "temperature": request.temperature,
            "stream": False
        }

        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens

        if request.tools:
            params["tools"] = request.tools
        if request.tool_choice:
            params["tool_choice"] = request.tool_choice

        params.update(provider_specific_kwargs)
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            # Parse the returned message structure to `ChatMessage`
            choice = data.get("choices", [])[0]
            response_message = choice.get("message", {})

            tool_calls = None
            if response_message.get("tool_calls"):
                tool_calls = response_message.get("tool_calls")

            message = ChatMessage(
                role=response_message.get("role", "assistant"),
                content=response_message.get("content") or response_message.get("text") or None,
                tool_calls=tool_calls,
            )

            usage = data.get("usage", {}) or {}

            return ChatCompletionResponse(
                message=message,
                provider='bigmodel',
                model=params["model"],
                usage=usage,
                raw_response=data,
            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("Bigmodel", e)

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> AsyncIterator[ChatCompletionResponse]:
        """
        Stream an async chat completion response from Bigmodel.
        """
        if self.api_key is None:
            raise ValueError("Bigmodel API key is required")

        messages = self._flatten_messages(request.messages)
        params = {
            "model": request.model or "glm-4.7",
            "messages": messages,
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
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", url, json=params, headers=headers) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        # Handle SSE-style "data: ..." lines or raw JSON per-line
                        payload = line
                        if line.startswith("data:"):
                            payload = line.split("data:", 1)[1].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                        except Exception:
                            continue

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        tool_calls = delta.get("tool_calls") or None

                        if content is not None or tool_calls is not None:
                            message = ChatMessage(role="assistant", content=content, tool_calls=tool_calls)
                            yield ChatCompletionResponse(
                                message=message,
                                provider='bigmodel',
                                model=params["model"],
                                usage={},
                                raw_response=chunk,
                            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("Bigmodel", e)
