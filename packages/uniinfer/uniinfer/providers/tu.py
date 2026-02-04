"""
OpenAI-compliant TU provider implementation.
"""
from typing import Any, AsyncIterator
from collections.abc import Iterator
import httpx
import json
import os
import asyncio
from datetime import datetime, timedelta

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage

from ..errors import map_provider_error, UniInferError
from ..logging_utils import log_raw_response

class TUProvider(ChatProvider):
    """TU (Tencent Unbounded) LLM Provider implementation."""

    _global_last_request_time: datetime | None = None
    _global_lock = asyncio.Lock()
    _min_request_interval = timedelta(seconds=2)

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """Initialize the TU provider.
        
        Args:
            api_key: The API key for TU. Defaults to TU_API_KEY env var.
            base_url: The base URL for the API.
        """
        self.api_key = api_key or os.getenv("TU_API_KEY")
        if not self.api_key:
            try:
                from credgoo.credgoo import get_api_key
                self.api_key = get_api_key("tu")
            except (ImportError, Exception):
                pass
        self.base_url = base_url or "https://aqueduct.ai.datalab.tuwien.ac.at/v1"
        self._async_client: httpx.AsyncClient | None = None
        
    async def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create the internal httpx.AsyncClient with TU configuration."""
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=httpx.Timeout(60.0, connect=10.0)
            )
        return self._async_client

    async def _throttle(self) -> None:
        """Simple async throttling to respect rate limits."""
        cls = type(self)
        async with cls._global_lock:
            if cls._global_last_request_time is not None:
                elapsed = datetime.now() - cls._global_last_request_time
                if elapsed < cls._min_request_interval:
                    wait_time = (cls._min_request_interval - elapsed).total_seconds()
                    await asyncio.sleep(wait_time)
            cls._global_last_request_time = datetime.now()

    def _prepare_payload(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """Prepare the request payload for the TU API.
        
        Args:
            request: The ChatCompletionRequest object.
            
        Returns:
            dict[str, Any]: The payload for the API request.
        """
        messages = []
        for m in request.messages:
            msg = {"role": m.role, "content": m.content}
            if m.tool_calls:
                msg["tool_calls"] = m.tool_calls
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            messages.append(msg)

        payload = {
            "model": request.model or "qwen-coder-30b",
            "messages": messages,
            "temperature": request.temperature,
            "stream": request.streaming
        }
        
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
            
        if request.tools:
            payload["tools"] = request.tools
            if request.tool_choice:
                payload["tool_choice"] = request.tool_choice
                
        return payload

    async def acomplete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Async completion implementation for TU."""
        await self._throttle()
        client = await self._get_async_client()
        payload = self._prepare_payload(request)

        try:
            response = await client.post("/chat/completions", json=payload)
            if response.status_code != 200:
                log_raw_response(
                    provider="tu",
                    operation="chat.completions",
                    raw_response={
                        "status_code": response.status_code,
                        "body": response.text,
                    },
                    log_file=os.path.join(os.getcwd(), "logs", "tu_raw_chat.log"),
                )
                raise map_provider_error("tu", Exception(f"TU API error: {response.status_code} - {response.text}"), status_code=response.status_code, response_body=response.text)
            
            log_raw_response(
                provider="tu",
                operation="chat.completions",
                raw_response=response.text,
                log_file=os.path.join(os.getcwd(), "logs", "tu_raw_chat.log"),
            )
            data = response.json()
            choice = data["choices"][0]
            message_data = choice["message"]
            
            message = ChatMessage(
                role=message_data["role"],
                content=message_data.get("content"),
                tool_calls=message_data.get("tool_calls"),
                tool_call_id=message_data.get("tool_call_id")
            )
            
            return ChatCompletionResponse(
                message=message,
                provider="tu",
                model=data.get("model", request.model or "qwen-coder-30b"),
                usage=data.get("usage", {}),
                raw_response=data,
                finish_reason=choice.get("finish_reason")
            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("tu", e)

    async def astream_complete(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionResponse]:
        """Async streaming completion implementation for TU."""
        request.streaming = True
        await self._throttle()
        client = await self._get_async_client()
        payload = self._prepare_payload(request)

        try:
            async with client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    log_raw_response(
                        provider="tu",
                        operation="chat.completions.stream",
                        raw_response={
                            "status_code": response.status_code,
                            "body": error_body.decode("utf-8", errors="replace"),
                        },
                        log_file=os.path.join(os.getcwd(), "logs", "tu_raw_chat.log"),
                    )
                    error_msg = f"TU API error: {response.status_code} - {error_body}"
                    raise map_provider_error("tu", Exception(error_msg), status_code=response.status_code, response_body=error_msg)
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.strip() == 'data: [DONE]':
                        break
                    if not line.startswith('data: '):
                        continue
                    log_raw_response(
                        provider="tu",
                        operation="chat.completions.stream",
                        raw_response={"line": line},
                        log_file=os.path.join(os.getcwd(), "logs", "tu_raw_chat.log"),
                    )
                    
                    try:
                        data_str = line[6:]
                        data = json.loads(data_str)
                        if 'choices' in data and len(data['choices']) > 0:
                            choice = data['choices'][0]
                            delta = choice.get('delta', {})
                            finish_reason = choice.get('finish_reason')
                            
                            if not delta.get('content') and not delta.get('tool_calls') and not finish_reason:
                                continue
                                
                            message = ChatMessage(
                                role=delta.get('role', 'assistant'),
                                content=delta.get('content'),
                                tool_calls=delta.get('tool_calls')
                            )
                            
                            yield ChatCompletionResponse(
                                message=message,
                                provider="tu",
                                model=data.get("model", request.model),
                                usage={},
                                raw_response=data,
                                finish_reason=finish_reason
                            )
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("tu", e)

    @classmethod
    def list_models(cls, api_key: str | None = None, **kwargs) -> list[str]:
        """List available models for TU.
        
        Args:
            api_key: API key if needed for listing models.
            **kwargs: Additional parameters (e.g. base_url).
            
        Returns:
            list[str]: A list of model identifiers.
        """
        if not api_key:
            api_key = os.getenv("TU_API_KEY")
        
        if not api_key:
            try:
                from credgoo.credgoo import get_api_key
                api_key = get_api_key("tu")
            except (ImportError, Exception):
                pass
        
        if not api_key:
            return []
            
        base_url = kwargs.get("base_url") or "https://aqueduct.ai.datalab.tuwien.ac.at/v1"
        
        try:
            import requests
            response = requests.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return [model["id"] for model in data.get("data", [])]
        except Exception:
            pass
            
        return []
