"""
InternLM provider implementation.
"""
import json
from typing import Optional, AsyncIterator

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import map_provider_error, UniInferError

try:
    from openai import OpenAI, AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class InternLMProvider(ChatProvider):
    """
    Provider for InternLM API.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://chat.intern-ai.org.cn/api/v1", **kwargs):
        """
        Initialize the InternLM provider.

        Args:
            api_key (Optional[str]): The InternLM API key.
            base_url (str): The base URL for the InternLM API.
            **kwargs: Additional configuration options.
        """
        super().__init__(api_key)
        self.base_url = base_url

        self.client = None
        self.async_client = None
        if HAS_OPENAI:
            self.client = OpenAI(api_key=self.api_key, base_url=f"{self.base_url}/")
            self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=f"{self.base_url}/")

    async def aclose(self):
        """Close the InternLM async client."""
        if self.async_client:
            await self.async_client.close()
        await super().aclose()

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """List available models from InternLM."""
        import requests
        if not api_key:
            from credgoo.credgoo import get_api_key
            api_key = get_api_key("internlm")
            if not api_key:
                return []

        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get("https://chat.intern-ai.org.cn/api/v1/models", headers=headers)
            response.raise_for_status()
            return [model["id"] for model in response.json().get("data", [])]
        except Exception:
            return []

    async def acomplete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """Make an async chat completion request to InternLM."""
        if self.api_key is None:
            raise ValueError("InternLM API key is required")

        if self.async_client:
            try:
                params = {
                    "model": request.model or "internlm3-latest",
                    "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
                    "temperature": request.temperature,
                }
                if request.max_tokens is not None:
                    params["max_tokens"] = request.max_tokens
                if "n" not in provider_specific_kwargs:
                    params["n"] = 1
                if "top_p" not in provider_specific_kwargs:
                    params["top_p"] = 0.9
                params.update(provider_specific_kwargs)

                completion = await self.async_client.chat.completions.create(**params)
                message = ChatMessage(role=completion.choices[0].message.role, content=completion.choices[0].message.content)
                usage = {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens
                }
                return ChatCompletionResponse(
                    message=message,
                    provider='internlm',
                    model=params["model"],
                    usage=usage,
                    raw_response=completion.model_dump()
                )
            except Exception as e:
                if isinstance(e, UniInferError):
                    raise
                raise map_provider_error("InternLM", e)
        else:
            # Fallback to direct httpx
            client = await self._get_async_client()
            endpoint = f"{self.base_url}/chat/completions"
            payload = {
                "model": request.model or "internlm3-latest",
                "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
                "temperature": request.temperature,
                "stream": False,
                "n": 1,
                "top_p": 0.9,
            }
            if request.max_tokens is not None:
                payload["max_tokens"] = request.max_tokens
            payload.update(provider_specific_kwargs)

            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
            try:
                response = await client.post(endpoint, headers=headers, json=payload, timeout=60.0)
                if response.status_code != 200:
                    raise map_provider_error("InternLM", Exception(response.text), status_code=response.status_code, response_body=response.text)
                
                resp_data = response.json()
                choice = resp_data.get("choices", [{}])[0]
                msg_data = choice.get("message", {})
                message = ChatMessage(role=msg_data.get("role", "assistant"), content=msg_data.get("content", ""))
                return ChatCompletionResponse(
                    message=message,
                    provider='internlm',
                    model=resp_data.get('model', request.model),
                    usage=resp_data.get("usage", {}),
                    raw_response=resp_data
                )
            except Exception as e:
                if isinstance(e, UniInferError):
                    raise
                raise map_provider_error("InternLM", e)

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> AsyncIterator[ChatCompletionResponse]:
        """Stream an async chat completion response from InternLM."""
        if self.api_key is None:
            raise ValueError("InternLM API key is required")

        if self.async_client:
            try:
                params = {
                    "model": request.model or "internlm3-latest",
                    "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
                    "temperature": request.temperature,
                    "stream": True,
                    "n": 1,
                    "top_p": 0.9,
                }
                if request.max_tokens is not None:
                    params["max_tokens"] = request.max_tokens
                params.update(provider_specific_kwargs)

                stream = await self.async_client.chat.completions.create(**params)
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        yield ChatCompletionResponse(
                            message=ChatMessage(role="assistant", content=content),
                            provider='internlm',
                            model=params["model"],
                            usage={},
                            raw_response=chunk.model_dump()
                        )
            except Exception as e:
                if isinstance(e, UniInferError):
                    raise
                raise map_provider_error("InternLM", e)
        else:
            # Fallback to direct httpx stream
            client = await self._get_async_client()
            endpoint = f"{self.base_url}/chat/completions"
            payload = {
                "model": request.model or "internlm3-latest",
                "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
                "temperature": request.temperature,
                "stream": True,
                "n": 1,
                "top_p": 0.9,
            }
            if request.max_tokens is not None:
                payload["max_tokens"] = request.max_tokens
            payload.update(provider_specific_kwargs)

            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
            try:
                async with client.stream("POST", endpoint, headers=headers, json=payload, timeout=60.0) as response:
                    if response.status_code != 200:
                        raise map_provider_error("InternLM", Exception(await response.aread()), status_code=response.status_code, response_body=await response.aread())
                    
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                yield ChatCompletionResponse(
                                    message=ChatMessage(role="assistant", content=content),
                                    provider='internlm',
                                    model=data.get('model', request.model),
                                    usage={},
                                    raw_response=data
                                )
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                if isinstance(e, UniInferError):
                    raise
                raise map_provider_error("InternLM", e)
