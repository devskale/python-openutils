import json
from typing import Any, AsyncIterator, Optional

from ..core import ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ChatProvider
from ..errors import UniInferError, map_provider_error


class OpenAICompatibleChatProvider(ChatProvider):
    BASE_URL = ""
    PROVIDER_ID = ""
    ERROR_PROVIDER_NAME = ""
    DEFAULT_MODEL: str | None = None

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.base_url = base_url or self.BASE_URL

    def _flatten_messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
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
            flattened_messages.append(msg_dict)
        return flattened_messages

    def _get_extra_headers(self) -> dict[str, str]:
        return {}

    def _get_default_payload_params(self, stream: bool) -> dict[str, Any]:
        return {}

    def _build_payload(
        self,
        request: ChatCompletionRequest,
        stream: bool,
        provider_specific_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model or self.DEFAULT_MODEL,
            "messages": self._flatten_messages(request.messages),
            "temperature": request.temperature,
            "stream": stream,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice

        default_params = self._get_default_payload_params(stream)
        for key, value in default_params.items():
            if key not in provider_specific_kwargs:
                payload[key] = value

        payload.update(provider_specific_kwargs)
        return payload

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self._get_extra_headers())
        return headers

    def _error_name(self) -> str:
        return self.ERROR_PROVIDER_NAME or self.PROVIDER_ID or self.__class__.__name__

    def _completion_endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    async def acomplete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs,
    ) -> ChatCompletionResponse:
        if self.api_key is None:
            raise ValueError(f"{self._error_name()} API key is required")

        endpoint = self._completion_endpoint()
        payload = self._build_payload(request, False, provider_specific_kwargs)
        headers = self._build_headers()

        client = await self._get_async_client()
        try:
            response = await client.post(endpoint, headers=headers, json=payload, timeout=60.0)
            if response.status_code != 200:
                error_msg = f"{self._error_name()} API error: {response.status_code} - {response.text}"
                raise map_provider_error(
                    self._error_name(),
                    Exception(error_msg),
                    status_code=response.status_code,
                    response_body=response.text,
                )

            response_data = response.json()
            choice = response_data.get("choices", [{}])[0]
            message_data = choice.get("message", {})
            message = ChatMessage(
                role=message_data.get("role", "assistant"),
                content=message_data.get("content"),
                tool_calls=message_data.get("tool_calls"),
                tool_call_id=message_data.get("tool_call_id"),
            )
            # Handle reasoning_content (OpenAI o1/o3, Groq R1, etc.)
            reasoning_content = message_data.get("reasoning_content")

            return ChatCompletionResponse(
                message=message,
                provider=self.PROVIDER_ID,
                model=response_data.get("model", request.model),
                usage=response_data.get("usage", {}),
                raw_response=response_data,
                finish_reason=choice.get("finish_reason"),
                thinking=reasoning_content,
            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error(self._error_name(), e)

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs,
    ) -> AsyncIterator[ChatCompletionResponse]:
        if self.api_key is None:
            raise ValueError(f"{self._error_name()} API key is required")

        endpoint = self._completion_endpoint()
        payload = self._build_payload(request, True, provider_specific_kwargs)
        headers = self._build_headers()

        client = await self._get_async_client()
        try:
            async with client.stream(
                "POST",
                endpoint,
                headers=headers,
                json=payload,
                timeout=60.0,
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    error_text = error_body.decode("utf-8", errors="replace")
                    error_msg = f"{self._error_name()} API error: {response.status_code} - {error_text}"
                    raise map_provider_error(
                        self._error_name(),
                        Exception(error_msg),
                        status_code=response.status_code,
                        response_body=error_text,
                    )

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        line = line[6:].strip()
                    elif line.startswith("data:"):
                        line = line[5:].strip()
                    else:
                        continue

                    if not line or line == "[DONE]":
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")
                    role = delta.get("role", "assistant")
                    content = delta.get("content")
                    # Handle reasoning_content (OpenAI o1/o3, Groq R1, etc.)
                    reasoning_content = delta.get("reasoning_content")
                    tool_calls = delta.get("tool_calls")

                    if content is None and reasoning_content is None and tool_calls is None and finish_reason is None:
                        continue

                    yield ChatCompletionResponse(
                        message=ChatMessage(
                            role=role,
                            content=content,
                            tool_calls=tool_calls,
                        ),
                        provider=self.PROVIDER_ID,
                        model=data.get("model", request.model),
                        usage=data.get("usage", {}),
                        raw_response=data,
                        finish_reason=finish_reason,
                        thinking=reasoning_content,
                    )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error(self._error_name(), e)
