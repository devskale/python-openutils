"""
Bigmodel (Z.ai) provider implementation.
"""
import asyncio
import logging
from typing import Optional

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import map_provider_error, UniInferError

logger = logging.getLogger(__name__)

_END = object()


def _safe_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _safe_dump(obj):
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    return obj


def _next_or_end(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return _END


class ZAIBaseProvider(ChatProvider):
    BASE_URL = "https://api.z.ai/api/paas/v4"
    PROVIDER_ID = "bigmodel"
    ERROR_PROVIDER_NAME = "Bigmodel"
    DEFAULT_MODEL = "glm-4.7"
    CREDGOO_SERVICE = "bigmodel"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        if not api_key:
            try:
                from credgoo.credgoo import get_api_key
                api_key = get_api_key(self.CREDGOO_SERVICE)
            except Exception:
                api_key = None
        super().__init__(api_key=api_key, **kwargs)
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        try:
            from zai import ZaiClient
        except ImportError as e:
            raise ImportError("zai-sdk is required for Z.ai providers. Install with: pip install zai-sdk") from e
        self.client = ZaiClient(api_key=self.api_key, base_url=self.base_url)

    @classmethod
    def list_models(cls, api_key: Optional[str] = None, base_url: Optional[str] = None) -> list[str]:
        if not api_key:
            raise ValueError("API key is required to list models")
        try:
            from zai import ZaiClient
            import requests
            effective_base_url = (base_url or cls.BASE_URL).rstrip("/")
            client = ZaiClient(api_key=api_key, base_url=effective_base_url)
            try:
                model_list = client.get("/models", cast_type=dict)
            except Exception:
                headers = {"Authorization": f"Bearer {api_key}"}
                response = requests.get(f"{effective_base_url}/models", headers=headers, timeout=30)
                response.raise_for_status()
                model_list = response.json()
            data = _safe_get(_safe_dump(model_list), "data", []) or []
            api_models = []
            for model in data:
                model_id = _safe_get(model, "id")
                if model_id:
                    api_models.append(model_id)
            return list(dict.fromkeys(api_models))
        except Exception as e:
            raise map_provider_error(cls.ERROR_PROVIDER_NAME, e)

    def _flatten_messages(self, messages: list[ChatMessage]) -> list[dict]:
        flattened_messages = []
        for msg in messages:
            msg_dict = msg.to_dict()
            content = msg_dict.get("content")
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                # Join text parts, or use placeholder if no text (e.g., image-only message)
                msg_dict["content"] = "".join(text_parts) if text_parts else "[content]"
            # Always include the message
            flattened_messages.append(msg_dict)
        return flattened_messages

    async def acomplete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        if self.api_key is None:
            raise ValueError(f"{self.ERROR_PROVIDER_NAME} API key is required")

        params = {
            "model": request.model or self.DEFAULT_MODEL,
            "messages": self._flatten_messages(request.messages),
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

        try:
            completion = await asyncio.to_thread(self.client.chat.completions.create, **params)
            completion_data = _safe_dump(completion) or {}
            choices = _safe_get(completion_data, "choices", []) or []
            choice = choices[0] if choices else {}
            message_data = _safe_get(choice, "message", {}) or {}
            message = ChatMessage(
                role=_safe_get(message_data, "role", "assistant"),
                content=_safe_get(message_data, "content"),
                tool_calls=_safe_get(message_data, "tool_calls"),
                tool_call_id=_safe_get(message_data, "tool_call_id"),
            )
            usage = _safe_get(completion_data, "usage", {}) or {}
            return ChatCompletionResponse(
                message=message,
                provider=self.PROVIDER_ID,
                model=_safe_get(completion_data, "model", params["model"]),
                usage=usage,
                raw_response=completion_data,
                finish_reason=_safe_get(choice, "finish_reason"),
            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error(self.ERROR_PROVIDER_NAME, e)

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ):
        if self.api_key is None:
            raise ValueError(f"{self.ERROR_PROVIDER_NAME} API key is required")

        flattened_messages = self._flatten_messages(request.messages)
        
        if not flattened_messages:
            logger.warning(f"ZAI {self.PROVIDER_ID}: No messages to send!")
        
        params = {
            "model": request.model or self.DEFAULT_MODEL,
            "messages": flattened_messages,
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

        try:
            stream = await asyncio.to_thread(self.client.chat.completions.create, **params)
            iterator = iter(stream)
            while True:
                chunk = await asyncio.to_thread(_next_or_end, iterator)
                if chunk is _END:
                    break
                chunk_data = _safe_dump(chunk) or {}
                choices = _safe_get(chunk_data, "choices", []) or []
                if not choices:
                    continue
                choice = choices[0]
                delta = _safe_get(choice, "delta", {}) or {}
                content = _safe_get(delta, "content")
                # Handle reasoning_content (Z.ai thinking models) - put in thinking field
                reasoning_content = _safe_get(delta, "reasoning_content")
                tool_calls = _safe_get(delta, "tool_calls")
                finish_reason = _safe_get(choice, "finish_reason")
                if content is None and reasoning_content is None and tool_calls is None and finish_reason is None:
                    continue
                yield ChatCompletionResponse(
                    message=ChatMessage(
                        role=_safe_get(delta, "role", "assistant"),
                        content=content,
                        tool_calls=tool_calls,
                    ),
                    provider=self.PROVIDER_ID,
                    model=_safe_get(chunk_data, "model", params["model"]),
                    usage=_safe_get(chunk_data, "usage", {}) or {},
                    raw_response=chunk_data,
                    finish_reason=finish_reason,
                    thinking=reasoning_content,  # Separate thinking content
                )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error(self.ERROR_PROVIDER_NAME, e)


class ZAIProvider(ZAIBaseProvider):
    PROVIDER_ID = "zai"
    ERROR_PROVIDER_NAME = "ZAI"
    DEFAULT_MODEL = "glm-4.7"
    CREDGOO_SERVICE = "zai"


class ZAICodeProvider(ZAIBaseProvider):
    BASE_URL = "https://api.z.ai/api/coding/paas/v4"
    PROVIDER_ID = "zai-code"
    ERROR_PROVIDER_NAME = "ZAI-Code"
    DEFAULT_MODEL = "glm-4.5"
    CREDGOO_SERVICE = "zai-code"
