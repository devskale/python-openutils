from __future__ import annotations
"""
Cloudflare Workers AI provider implementation.
"""
import json
from typing import Optional, List, AsyncIterator

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import map_provider_error, AuthenticationError, UniInferError


class CloudflareProvider(ChatProvider):
    """
    Provider for Cloudflare Workers AI.
    """

    def __init__(self, api_key: Optional[str] = None, account_id: Optional[str] = None, **kwargs):
        """
        Initialize the Cloudflare Workers AI provider.

        Args:
            api_key (Optional[str]): The Cloudflare API token.
            account_id (Optional[str]): The Cloudflare account ID.
            **kwargs: Additional provider-specific configuration parameters.
        """
        super().__init__(api_key)

        if not api_key:
            raise AuthenticationError("Cloudflare API token is required")

        if not account_id:
            raise AuthenticationError("Cloudflare account ID is required")

        self.account_id = account_id
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Save any additional configuration
        self.config = kwargs

    @classmethod
    def list_models(cls, api_key: Optional[str] = None, account_id: Optional[str] = None,
                    author: Optional[str] = None, hide_experimental: Optional[bool] = None,
                    page: Optional[int] = None, per_page: Optional[int] = None,
                    search: Optional[str] = None, source: Optional[int] = None) -> list[ModelInfo]:
        """
        List available models from Cloudflare Workers AI.
        """
        from ..core import ModelInfo
        import requests
        if api_key is None:
            try:
                from credgoo import get_api_key
                api_key = get_api_key("cloudflare")
                if api_key is None:
                    raise ValueError(
                        "Failed to retrieve Cloudflare API key from credgoo")
            except ImportError:
                raise ValueError(
                    "Cloudflare API key is required when credgoo is not available")

        if account_id is None:
            raise ValueError("Cloudflare account ID is required")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        params = {}
        if author is not None:
            params["author"] = author
        if hide_experimental is not None:
            params["hide_experimental"] = hide_experimental
        if page is not None:
            params["page"] = page
        if per_page is not None:
            params["per_page"] = per_page
        if search is not None:
            params["search"] = search
        if source is not None:
            params["source"] = source

        try:
            response = requests.get(
                f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search",
                headers=headers,
                params=params
            )
            response.raise_for_status()

            models_data = response.json()
            model_list = []
            if models_data.get("success", False) and "result" in models_data:
                for model in models_data["result"]:
                    if isinstance(model, dict) and "name" in model:
                        model_name = model.get("name")
                        if model_name:
                            model_list.append(ModelInfo(id=model_name, owned_by=model.get("author"), raw=model))

            return model_list
        except Exception as e:
            status_code = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            response_body = getattr(e.response, 'text', None) if hasattr(e, 'response') else None
            raise map_provider_error("cloudflare", e, status_code=status_code, response_body=response_body)

    def _prepare_messages(self, messages: List[ChatMessage]) -> list[dict]:
        """Convert ChatMessage list to OpenAI-style messages dicts."""
        result = []
        for msg in messages:
            content = msg.content
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                content = "".join(parts)
            result.append({"role": msg.role, "content": content})
        return result

    async def acomplete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """
        Make an async chat completion request to Cloudflare Workers AI.
        """
        try:
            model = request.model or "@cf/meta/llama-3-8b-instruct"
            msgs = self._prepare_messages(request.messages)

            data = {
                "messages": msgs,
                "max_tokens": request.max_tokens if request.max_tokens is not None else 1024,
            }

            if request.temperature is not None:
                data["temperature"] = request.temperature

            for key, value in provider_specific_kwargs.items():
                data[key] = value

            client = await self._get_async_client()
            endpoint = self.base_url + model

            response = await client.post(
                endpoint,
                headers=self.headers,
                json=data,
                timeout=60.0
            )

            if response.status_code != 200:
                error_msg = f"Cloudflare API error: {response.status_code} - {response.text}"
                raise map_provider_error("cloudflare", Exception(error_msg), status_code=response.status_code, response_body=response.text)

            response_data = response.json()
            result = response_data.get("result", {})
            content = ""
            thinking = None

            if isinstance(result, dict):
                if result.get("object") == "chat.completion":
                    choice = result.get("choices", [{}])[0]
                    msg = choice.get("message", {})
                    content = msg.get("content", "") or ""
                    thinking = msg.get("reasoning_content")
                elif "response" in result:
                    content = result["response"]
                else:
                    content = str(result)
            elif isinstance(result, str):
                content = result
            else:
                content = str(result)

            msg_obj = ChatMessage(role="assistant", content=content)
            resp = ChatCompletionResponse(
                message=msg_obj,
                provider='cloudflare',
                model=model,
                usage={},
                raw_response=response_data
            )
            if thinking:
                resp.thinking = thinking

            return resp

        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("cloudflare", e)

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> AsyncIterator[ChatCompletionResponse]:
        """
        Stream an async chat completion response from Cloudflare Workers AI.
        """
        try:
            model = request.model or "@cf/meta/llama-3-8b-instruct"
            msgs = self._prepare_messages(request.messages)

            data = {
                "messages": msgs,
                "stream": True,
                "max_tokens": request.max_tokens if request.max_tokens is not None else 1024,
            }

            if request.temperature is not None:
                data["temperature"] = request.temperature

            for key, value in provider_specific_kwargs.items():
                data[key] = value

            client = await self._get_async_client()
            endpoint = self.base_url + model

            async with client.stream(
                "POST",
                endpoint,
                headers=self.headers,
                json=data,
                timeout=60.0
            ) as response:
                if response.status_code != 200:
                    error_msg = f"Cloudflare API error: {response.status_code} - {await response.aread()}"
                    raise map_provider_error("cloudflare", Exception(error_msg), status_code=response.status_code, response_body=error_msg)

                async for line in response.aiter_lines():
                    if line:
                        chunk_data = line.strip()
                        if chunk_data.startswith("data: "):
                            chunk_data = chunk_data[6:]

                        if chunk_data == "[DONE]":
                            continue

                        try:
                            json_chunk = json.loads(chunk_data)
                            chunk_text = ""

                            if isinstance(json_chunk, dict):
                                if "choices" in json_chunk:
                                    delta = json_chunk.get("choices", [{}])[0]
                                    d = delta.get("delta", {})
                                    chunk_text = d.get("content", "") or ""
                                    if not chunk_text:
                                        chunk_text = delta.get("text", "") or ""
                                elif "response" in json_chunk:
                                    chunk_text = json_chunk["response"]
                                elif "result" in json_chunk:
                                    r = json_chunk["result"]
                                    if isinstance(r, str):
                                        chunk_text = r
                                    elif isinstance(r, dict) and "response" in r:
                                        chunk_text = r["response"]

                            if not chunk_text:
                                continue

                            yield ChatCompletionResponse(
                                message=ChatMessage(role="assistant", content=chunk_text),
                                provider='cloudflare',
                                model=model,
                                usage={},
                                raw_response=json_chunk
                            )
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("cloudflare", e)
