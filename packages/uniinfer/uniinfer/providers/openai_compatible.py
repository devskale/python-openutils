import json
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from ..core import REASONING_OFF, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ChatProvider
from ..errors import UniInferError, map_provider_error

_MODEL_DEFAULTS: dict[str, dict[str, Any]] | None = None
_MODEL_DEFAULTS_PATH = Path(__file__).resolve().parent.parent / "models" / "model_defaults.json"


def _load_model_defaults() -> dict[str, dict[str, Any]]:
    global _MODEL_DEFAULTS
    if _MODEL_DEFAULTS is None:
        try:
            with open(_MODEL_DEFAULTS_PATH) as f:
                _MODEL_DEFAULTS = json.load(f)
        except Exception:
            _MODEL_DEFAULTS = {}
    return _MODEL_DEFAULTS


def openrouter_reasoning_payload(reasoning_effort: Optional[str]) -> dict[str, Any]:
    """Map ``reasoning_effort`` to the OpenRouter/Kilo ``reasoning`` object.

    OpenRouter-style gateways (OpenRouter, Kilo) ignore the bare
    ``reasoning_effort`` field — on Kilo it actively breaks reasoning-capable
    models (they over-reason and emit no content). The curated ``reasoning``
    object is the correct dialect. Valid efforts: low/medium/high.
    ``none``/``minimal`` (``REASONING_OFF``) are omitted: many routed reasoning
    models reject disabling ("Reasoning is mandatory ... cannot be disabled"),
    so we let the model default rather than 400.
    """
    if not reasoning_effort:
        return {}
    effort = str(reasoning_effort).strip().lower()
    if effort in REASONING_OFF:
        return {}
    return {"reasoning": {"effort": effort}}


class OpenAICompatibleChatProvider(ChatProvider):
    # OpenAI params to NEVER forward even if a client sends them as extras
    # (none known yet; add here if one 400s a backend).
    EXTRA_FORWARD_DENY: frozenset[str] = frozenset()
    BASE_URL = ""
    PROVIDER_ID = ""
    ERROR_PROVIDER_NAME = ""
    DEFAULT_MODEL: str | None = None
    # OpenAI-compat: a trailing assistant message is a prefill (continuation).
    # Backends that need a flag to accept it declare the JSON key here (e.g.
    # Mistral's "prefix"); the base _flatten_messages sets it True on the last
    # message when it's an assistant turn. None = the backend accepts a trailing
    # assistant natively (no flag needed).
    PREFILL_FLAG: str | None = None
    # Whether completions require an API key. Most OpenAI-compatible backends do;
    # gateways with an anonymous free tier (e.g. Kilo) set this False so
    # acomplete/astream_complete skip the api_key guard for free models.
    REQUIRES_API_KEY: bool = True
    # Whether the backend accepts native OpenAI multimodal content (a list of
    # content parts, including image_url). When True, list content is forwarded
    # as-is so vision models receive images. When False (default, for text-only
    # backends), list content is flattened to a string (image parts dropped) —
    # the legacy behaviour. Set True on vision-capable gateways (Kilo, OpenCode).
    PRESERVE_MULTIMODAL: bool = False

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.base_url = base_url or self.BASE_URL

    def _flatten_messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        flattened_messages = []
        for msg in messages:
            msg_dict = msg.to_dict()
            content = msg_dict.get("content")
            if isinstance(content, list) and not self.PRESERVE_MULTIMODAL:
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                # Join text parts, or use placeholder if no text (e.g., image-only message)
                msg_dict["content"] = "".join(text_parts) if text_parts else "[content]"
            flattened_messages.append(msg_dict)
        # Trailing-assistant prefill: if this backend needs a continuation flag,
        # set it on the last message (only when it's an assistant turn).
        # Generalized from the Mistral-specific override — any backend declares
        # its flag via PREFILL_FLAG instead of overriding this method.
        if (
            self.PREFILL_FLAG
            and flattened_messages
            and flattened_messages[-1].get("role") == "assistant"
        ):
            flattened_messages[-1][self.PREFILL_FLAG] = True
        return flattened_messages

    def _get_extra_headers(self) -> dict[str, str]:
        return {}

    def _get_default_payload_params(self, stream: bool) -> dict[str, Any]:
        return {}

    def _reasoning_payload(self, reasoning_effort: Optional[str]) -> dict[str, Any]:
        """Map ``reasoning_effort`` to backend-specific payload fields.

        Base default is a no-op: ``reasoning_effort`` is dropped, preserving
        legacy behaviour and staying safe for backends that reject unknown
        params. Subclasses whose backend supports reasoning control override
        this with the correct dialect (e.g. OpenRouter/Kilo use the
        ``reasoning`` object via :func:`openrouter_reasoning_payload`).
        """
        return {}

    def _build_payload(
        self,
        request: ChatCompletionRequest,
        stream: bool,
        provider_specific_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        model_id = request.model or self.DEFAULT_MODEL

        defaults = {}
        model_defaults = _load_model_defaults()
        if model_id in model_defaults:
            defaults = model_defaults[model_id]

        payload: dict[str, Any] = {
            "model": model_id,
            "messages": self._flatten_messages(request.messages),
            "temperature": defaults.get("temperature", request.temperature),
            "stream": stream,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice

        if request.reasoning_effort is not None:
            payload.update(self._reasoning_payload(request.reasoning_effort))

        default_params = self._get_default_payload_params(stream)
        for key, value in default_params.items():
            if key not in provider_specific_kwargs:
                payload[key] = value

        payload.update(provider_specific_kwargs)
        # OpenAI passthrough: forward unmapped OpenAI params (top_p, response_format,
        # seed, stream_options, logprobs, …) verbatim so new OpenAI features reach
        # backends without a per-field code change. EXTRA_FORWARD_DENY strips any
        # that 400 a backend; empty by default (most backends ignore unknowns).
        if getattr(request, "extra", None):
            for _k, _v in request.extra.items():
                if _k not in self.EXTRA_FORWARD_DENY:
                    payload[_k] = _v
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
        if self.REQUIRES_API_KEY and self.api_key is None:
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
            # Also handle NGC 'reasoning' and Ollama 'thinking' fields
            reasoning_content = message_data.get("reasoning_content") or message_data.get("reasoning") or message_data.get("thinking")

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
        if self.REQUIRES_API_KEY and self.api_key is None:
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
                        # Terminal usage-only chunk (choices:[]). vLLM emits this
                        # when stream_options.include_usage is set; forward it so
                        # the proxy can emit usage to clients.
                        if data.get("usage"):
                            yield ChatCompletionResponse(
                                message=ChatMessage(role="assistant", content=None),
                                provider=self.PROVIDER_ID,
                                model=data.get("model", request.model),
                                usage=data["usage"],
                                raw_response=data,
                                finish_reason=None,
                                thinking=None,
                            )
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")
                    role = delta.get("role", "assistant")
                    content = delta.get("content")
                    # Handle reasoning_content (OpenAI o1/o3, Groq R1, etc.)
                    # Also handle NGC 'reasoning' and Ollama 'thinking' fields
                    reasoning_content = delta.get("reasoning_content") or delta.get("reasoning") or delta.get("thinking")
                    tool_calls = delta.get("tool_calls")

                    if content is None and reasoning_content is None and tool_calls is None and finish_reason is None:
                        # Empty-delta chunk — but vLLM emits usage on a
                        # choices:[{delta:{}}] chunk (not choices:[]). Forward it
                        # when it carries usage so the proxy can emit it.
                        if not data.get("usage"):
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
