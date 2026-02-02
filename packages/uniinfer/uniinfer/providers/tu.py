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

from ..errors import map_provider_error

class TUProvider(ChatProvider):
    """TU (Tencent Unbounded) LLM Provider implementation."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """Initialize the TU provider.
        
        Args:
            api_key: The API key for TU. Defaults to TU_API_KEY env var.
            base_url: The base URL for the API.
        """
        self.api_key = api_key or os.getenv("TU_API_KEY")
        self.base_url = base_url or "https://aqueduct.ai.datalab.tuwien.ac.at/v1"
        self._async_client: httpx.AsyncClient | None = None
        
        # Throttling state
        self._last_request_time: datetime | None = None
        self._min_request_interval = timedelta(milliseconds=200) # 5 requests per second limit
        self._lock = asyncio.Lock()

    async def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create the async httpx client.
        
        Returns:
            httpx.AsyncClient: The async client instance.
        """
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
        async with self._lock:
            if self._last_request_time is not None:
                elapsed = datetime.now() - self._last_request_time
                if elapsed < self._min_request_interval:
                    wait_time = (self._min_request_interval - elapsed).total_seconds()
                    await asyncio.sleep(wait_time)
            self._last_request_time = datetime.now()

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
        """Async completion implementation for TU.
        
        Args:
            request: The ChatCompletionRequest object.
            
        Returns:
            ChatCompletionResponse: The response from the TU API.
            
        Raises:
            UniInferError: If the request fails.
        """
        await self._throttle()
        client = await self._get_async_client()
        payload = self._prepare_payload(request)

        try:
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Map OpenAI-compatible response to UniInfer format
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
            raise map_provider_error("tu", e)

    def _process_chunk(self, line: str, model_requested: str | None) -> ChatCompletionResponse | None:
        """Process a single line from the stream.
        
        Args:
            line: The raw line from the API.
            model_requested: The model that was requested.
            
        Returns:
            ChatCompletionResponse | None: The parsed response chunk, or None if it should be skipped.
        """
        if not line.startswith("data: "):
            return None
            
        line = line[6:].strip()
        if line == "[DONE]" or not line:
            return None
            
        try:
            data = json.loads(line)
            if not data.get("choices"):
                return None
                
            choice = data["choices"][0]
            delta = choice.get("delta", {})
            
            # Skip if no content and no tool calls (common in first/last chunks)
            if not delta.get("content") and not delta.get("tool_calls") and not choice.get("finish_reason"):
                return None

            message = ChatMessage(
                role=delta.get("role", "assistant"),
                content=delta.get("content"),
                tool_calls=delta.get("tool_calls"),
                tool_call_id=delta.get("tool_call_id")
            )
            
            return ChatCompletionResponse(
                message=message,
                provider="tu",
                model=data.get("model", model_requested or "qwen-coder-30b"),
                usage={}, # Usage is usually only in the last chunk or not at all in stream
                raw_response=data,
                finish_reason=choice.get("finish_reason")
            )
        except json.JSONDecodeError:
            return None

    async def astream_complete(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionResponse]:
        """Async streaming completion implementation for TU.
        
        Args:
            request: The ChatCompletionRequest object.
            
        Yields:
            ChatCompletionResponse: Chunks of the response from the TU API.
            
        Raises:
            UniInferError: If the request fails.
        """
        request.streaming = True
        await self._throttle()
        client = await self._get_async_client()
        payload = self._prepare_payload(request)

        try:
            async with client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    chunk = self._process_chunk(line, request.model)
                    if chunk:
                        yield chunk
        except Exception as e:
            raise map_provider_error("tu", e)

    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Sync completion wrapper.
        
        Args:
            request: The ChatCompletionRequest object.
            
        Returns:
            ChatCompletionResponse: The response from the TU API.
        """
        return asyncio.run(self.acomplete(request))

    def stream_complete(self, request: ChatCompletionRequest) -> Iterator[ChatCompletionResponse]:
        """Sync streaming completion wrapper.
        
        Args:
            request: The ChatCompletionRequest object.
            
        Returns:
            Iterator[ChatCompletionResponse]: An iterator over response chunks.
        """
        import requests
        request.streaming = True
        payload = self._prepare_payload(request)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            with requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=60
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        chunk = self._process_chunk(line_str, request.model)
                        if chunk:
                            yield chunk
        except Exception as e:
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
        api_key = api_key or os.getenv("TU_API_KEY")
        base_url = kwargs.get("base_url") or "https://aqueduct.ai.datalab.tuwien.ac.at/v1"
        
        if not api_key:
            return []
            
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
