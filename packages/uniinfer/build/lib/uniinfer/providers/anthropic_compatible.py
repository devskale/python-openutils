import json
import os
from typing import Any, AsyncIterator, Optional

from ..core import ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ChatProvider
from ..errors import UniInferError, map_provider_error

try:
    from anthropic import Anthropic, AsyncAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


class AnthropicCompatibleProvider(ChatProvider):
    BASE_URL = "https://api.anthropic.com"
    PROVIDER_ID = ""
    ERROR_PROVIDER_NAME = ""
    DEFAULT_MODEL: str | None = None
    DEFAULT_MAX_TOKENS = 1024
    CREDGOO_SERVICE: str | None = None

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        if not HAS_ANTHROPIC:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")
        if not api_key and self.CREDGOO_SERVICE:
            try:
                from credgoo.credgoo import get_api_key
                api_key = get_api_key(self.CREDGOO_SERVICE)
            except Exception:
                api_key = None
        super().__init__(api_key=api_key, **kwargs)
        self.base_url = base_url or self.BASE_URL
        default_headers = self._default_headers()
        self.client = Anthropic(api_key=self.api_key, base_url=self.base_url, default_headers=default_headers)
        self.async_client = AsyncAnthropic(api_key=self.api_key, base_url=self.base_url, default_headers=default_headers)

    def _default_headers(self) -> dict[str, str] | None:
        return None

    def _error_name(self) -> str:
        return self.ERROR_PROVIDER_NAME or self.PROVIDER_ID or self.__class__.__name__

    async def aclose(self):
        if hasattr(self, "async_client"):
            await self.async_client.close()
        await super().aclose()

    @classmethod
    def list_models(cls, api_key: Optional[str] = None, **kwargs) -> list[str]:
        if not HAS_ANTHROPIC:
            return []
        if not api_key and cls.CREDGOO_SERVICE:
            try:
                from credgoo.credgoo import get_api_key
                api_key = get_api_key(cls.CREDGOO_SERVICE)
            except Exception:
                api_key = None
        if not api_key:
            env_name = f"{cls.PROVIDER_ID.upper().replace('-', '_')}_API_KEY"
            api_key = os.getenv(env_name)
        if not api_key:
            raise ValueError(f"{cls.PROVIDER_ID} API key is required to list models")
        base_url = kwargs.get("base_url") or cls.BASE_URL
        try:
            client = Anthropic(api_key=api_key, base_url=base_url, default_headers=cls._class_default_headers())
            response = client.models.list()
            data = getattr(response, "data", []) or []
            return [getattr(model, "id", None) for model in data if getattr(model, "id", None)]
        except Exception as e:
            raise map_provider_error(cls.ERROR_PROVIDER_NAME or cls.PROVIDER_ID, e)

    @classmethod
    def _class_default_headers(cls) -> dict[str, str] | None:
        return None

    def _prepare_messages(self, request: ChatCompletionRequest) -> tuple[list[dict[str, Any]], str | None]:
        messages: list[dict[str, Any]] = []
        system_content: str | None = None
        for msg in request.messages:
            if msg.role == "system":
                if isinstance(msg.content, str):
                    system_content = msg.content if system_content is None else f"{system_content}\n{msg.content}"
                elif isinstance(msg.content, list):
                    parts = []
                    for part in msg.content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            parts.append(part.get("text", ""))
                    joined = "".join(parts)
                    if joined:
                        system_content = joined if system_content is None else f"{system_content}\n{joined}"
                continue
            content: Any = msg.content
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                content = content
            else:
                content = [{"type": "text", "text": ""}]
            messages.append({"role": msg.role, "content": content})
        return messages, system_content

    def _extract_message(self, response: Any) -> tuple[str | None, list[dict] | None]:
        content_text_parts: list[str] = []
        tool_calls: list[dict] = []
        for block in getattr(response, "content", []) or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                text = getattr(block, "text", None)
                if text:
                    content_text_parts.append(text)
            elif btype == "tool_use":
                tool_calls.append({
                    "id": getattr(block, "id", None),
                    "type": "function",
                    "function": {
                        "name": getattr(block, "name", None),
                        "arguments": json.dumps(getattr(block, "input", {}) or {}),
                    },
                })
        content = "".join(content_text_parts) if content_text_parts else None
        return content, tool_calls or None

    async def acomplete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        if self.api_key is None:
            raise ValueError(f"{self._error_name()} API key is required")

        messages, system_content = self._prepare_messages(request)
        params: dict[str, Any] = {
            "model": request.model or self.DEFAULT_MODEL,
            "messages": messages,
            "max_tokens": request.max_tokens or self.DEFAULT_MAX_TOKENS,
        }
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if system_content:
            params["system"] = system_content
        if request.tools:
            params["tools"] = request.tools
        if request.tool_choice:
            params["tool_choice"] = request.tool_choice
        params.update(provider_specific_kwargs)

        try:
            response = await self.async_client.messages.create(**params)
            content, tool_calls = self._extract_message(response)
            usage_obj = getattr(response, "usage", None)
            input_tokens = getattr(usage_obj, "input_tokens", 0) or 0
            output_tokens = getattr(usage_obj, "output_tokens", 0) or 0
            usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
            return ChatCompletionResponse(
                message=ChatMessage(role="assistant", content=content, tool_calls=tool_calls),
                provider=self.PROVIDER_ID,
                model=getattr(response, "model", params["model"]),
                usage=usage,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else response,
                finish_reason=getattr(response, "stop_reason", None),
            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error(self._error_name(), e)

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> AsyncIterator[ChatCompletionResponse]:
        if self.api_key is None:
            raise ValueError(f"{self._error_name()} API key is required")

        messages, system_content = self._prepare_messages(request)
        params: dict[str, Any] = {
            "model": request.model or self.DEFAULT_MODEL,
            "messages": messages,
            "max_tokens": request.max_tokens or self.DEFAULT_MAX_TOKENS,
            "stream": True,
        }
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if system_content:
            params["system"] = system_content
        if request.tools:
            params["tools"] = request.tools
        if request.tool_choice:
            params["tool_choice"] = request.tool_choice
        params.update(provider_specific_kwargs)

        try:
            stream = await self.async_client.messages.create(**params)
            async for event in stream:
                event_type = getattr(event, "type", None)
                if event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    text = getattr(delta, "text", None)
                    if text:
                        yield ChatCompletionResponse(
                            message=ChatMessage(role="assistant", content=text),
                            provider=self.PROVIDER_ID,
                            model=request.model or self.DEFAULT_MODEL or "",
                            usage={},
                            raw_response=event.model_dump() if hasattr(event, "model_dump") else event,
                        )
                elif event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if getattr(block, "type", None) == "tool_use":
                        tool_call = {
                            "id": getattr(block, "id", None),
                            "type": "function",
                            "function": {
                                "name": getattr(block, "name", None),
                                "arguments": json.dumps(getattr(block, "input", {}) or {}),
                            },
                        }
                        yield ChatCompletionResponse(
                            message=ChatMessage(role="assistant", content=None, tool_calls=[tool_call]),
                            provider=self.PROVIDER_ID,
                            model=request.model or self.DEFAULT_MODEL or "",
                            usage={},
                            raw_response=event.model_dump() if hasattr(event, "model_dump") else event,
                        )
                elif event_type == "message_stop":
                    stop_reason = getattr(event, "stop_reason", None)
                    if stop_reason is None:
                        message = getattr(event, "message", None)
                        stop_reason = getattr(message, "stop_reason", None)
                    yield ChatCompletionResponse(
                        message=ChatMessage(role="assistant", content=None),
                        provider=self.PROVIDER_ID,
                        model=request.model or self.DEFAULT_MODEL or "",
                        usage={},
                        raw_response=event.model_dump() if hasattr(event, "model_dump") else event,
                        finish_reason=stop_reason,
                    )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error(self._error_name(), e)
