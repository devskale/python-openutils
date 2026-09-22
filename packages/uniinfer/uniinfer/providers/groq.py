from __future__ import annotations
"""
Groq provider implementation with async support.
"""
import json
import os
from typing import Dict, Any, Iterator, Optional, List, AsyncIterator

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ModelInfo
from ..errors import map_provider_error, UniInferError

try:
    from groq import Groq, AsyncGroq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


class GroqProvider(ChatProvider):
    """
    Provider for Groq API.

    Groq is a high-performance LLM inference provider.
    """

    ACCESS_TIER = "free"  # universally free (forever-free tier, no CC)

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the Groq provider.

        Args:
            api_key (Optional[str]): The Groq API key.
            **kwargs: Additional configuration options.
        """
        if not api_key:
            from credgoo import get_api_key
            api_key = get_api_key("groq")

        super().__init__(api_key)

        if not HAS_GROQ:
            raise ImportError(
                "groq package is required for the GroqProvider. "
                "Install it with: uv pip install groq"
            )

        # Initialize the Groq clients
        self.client = Groq(api_key=self.api_key)
        self.async_client = AsyncGroq(api_key=self.api_key)

    _MAX_OUTPUT_CACHE: dict[str, Optional[int]] = {}

    @classmethod
    def _catalog_max_output(cls, model_id: Optional[str]) -> Optional[int]:
        """Model max_output from the packaged catalog (uniinfer/models/models.json).

        Groq hard-caps ``max_tokens`` per model (e.g. qwen3.8-27b: 16384) —
        sending more 400s the whole request. The catalog carries the cap since
        list_models() populates max_output; callers rarely know it.
        """
        if not model_id or model_id not in cls._MAX_OUTPUT_CACHE:
            try:
                from pathlib import Path
                p = Path(__file__).resolve().parent.parent / "models" / "models.json"
                data = json.loads(p.read_text())
                for m in (data.get("providers", {}).get("groq", {}) or {}).get("models", []):
                    v = m.get("max_output")
                    if isinstance(v, int) and v > 0:
                        cls._MAX_OUTPUT_CACHE[m["id"]] = v
            except Exception:
                pass
            cls._MAX_OUTPUT_CACHE.setdefault(model_id, None)
        return cls._MAX_OUTPUT_CACHE.get(model_id)

    def _build_groq_params(
        self,
        request: ChatCompletionRequest,
        stream: bool,
        provider_specific_kwargs: dict,
    ) -> dict[str, Any]:
        """Single param builder for sync/async/stream — one place for the
        max_tokens clamp (Groq rejects max_tokens > model cap with a 400)."""
        params: dict[str, Any] = {
            "model": request.model or "llama-3.1-8b",
            "messages": self._flatten_messages(request.messages),
            "temperature": request.temperature,
            "stream": stream,
        }
        if request.max_tokens is not None:
            cap = self._catalog_max_output(request.model)
            mt = min(request.max_tokens, cap) if cap else request.max_tokens
            # Weicher Config-Cap (model_defaults.json) neben dem Katalog-Hardcap:
            # Free-Tier-Limits liegen oft weit unter der API-Grenze.
            from .openai_compatible import _load_model_defaults
            soft = (_load_model_defaults().get(request.model) or {}).get("max_tokens_cap")
            if isinstance(soft, int) and soft > 0:
                mt = min(mt, soft)
            params["max_tokens"] = mt
        if request.tools:
            params["tools"] = request.tools
        if request.tool_choice:
            params["tool_choice"] = request.tool_choice
        params.update(provider_specific_kwargs)
        return params

    async def acomplete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """
        Make an async chat completion request to Groq.
        """
        messages = self._flatten_messages(request.messages)

        params = self._build_groq_params(request, False, provider_specific_kwargs)

        try:
            completion = await self.async_client.chat.completions.create(**params)

            response_message = completion.choices[0].message
            tool_calls = None
            if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
                tool_calls = [
                    {
                        "id": getattr(tc, 'id', None),
                        "type": getattr(tc, 'type', 'function'),
                        "function": {
                            "name": tc.function.name if hasattr(tc, 'function') else None,
                            "arguments": tc.function.arguments if hasattr(tc, 'function') else None
                        }
                    }
                    for tc in response_message.tool_calls
                ]
            message = ChatMessage(
                role=response_message.role,
                content=getattr(response_message, 'content', None),
                tool_calls=tool_calls
            )
            
            # Handle reasoning_content (Groq R1 models)
            reasoning_content = getattr(response_message, 'reasoning_content', None)

            usage = {}
            if hasattr(completion, 'usage'):
                usage = {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens
                }

            try:
                raw_response = completion.model_dump()
            except Exception:
                raw_response = str(completion)

            return ChatCompletionResponse(
                message=message,
                provider='groq',
                model=params["model"],
                usage=usage,
                raw_response=raw_response,
                thinking=reasoning_content,
            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("Groq", e)

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> AsyncIterator[ChatCompletionResponse]:
        """
        Stream an async chat completion response from Groq.
        """
        messages = self._flatten_messages(request.messages)

        params = self._build_groq_params(request, True, provider_specific_kwargs)

        try:
            stream = await self.async_client.chat.completions.create(**params)

            async for chunk in stream:
                if chunk.choices and hasattr(chunk.choices[0], 'delta'):
                    delta = chunk.choices[0].delta
                    content = getattr(delta, 'content', None)
                    # Handle reasoning_content (Groq R1 models)
                    reasoning_content = getattr(delta, 'reasoning_content', None)
                    tool_calls = None
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        tool_calls = [
                            {
                                "id": getattr(tc, 'id', None),
                                "type": getattr(tc, 'type', 'function'),
                                "function": {
                                    "name": tc.function.name if hasattr(tc, 'function') else None,
                                    "arguments": tc.function.arguments if hasattr(tc, 'function') else None
                                }
                            }
                            for tc in delta.tool_calls
                        ]

                    if content or tool_calls or reasoning_content:
                        message = ChatMessage(
                            role="assistant",
                            content=content,
                            tool_calls=tool_calls
                        )
                        yield ChatCompletionResponse(
                            message=message,
                            provider='groq',
                            model=params["model"],
                            usage={},
                            raw_response={"delta": {"content": content, "tool_calls": tool_calls, "reasoning_content": reasoning_content}},
                            thinking=reasoning_content
                        )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error("Groq", e)

    def _flatten_messages(self, msgs: List[ChatMessage]) -> List[Dict[str, Any]]:
        flattened = []
        for m in msgs:
            md = m.to_dict()
            content = md.get("content")
            if isinstance(content, list):
                parts: List[str] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                # Join text parts, or use placeholder if no text (e.g., image-only message)
                md["content"] = "".join(parts) if parts else "[content]"
            flattened.append(md)
        return flattened

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list[ModelInfo]:
        """
        List available models from Groq.

        Args:
            api_key (Optional[str]): The Groq API key. If not provided,
                it will try GROQ_API_KEY environment variable or credgoo.

        Returns:
            List[str]: A list of available model names.

        Raises:
            ValueError: If no API key can be found.
            Exception: If the API request fails.
        """
        if not HAS_GROQ:
            import logging
            logging.warning("groq package not installed — cannot list Groq models")
            return []

        # Prioritize the provided api_key parameter
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            try:
                from credgoo import get_api_key
                api_key = get_api_key("groq")
            except ImportError:
                api_key = None  # credgoo not available

        if not api_key:
            import logging
            logging.warning("No Groq API key found")
            return []

        try:
            client = Groq(api_key=api_key)
            models = client.models.list()
            out = []
            for model in models.data:
                in_mods = getattr(model, "input_modalities", None) or ["text"]
                out_mods = getattr(model, "output_modalities", None) or ["text"]
                caps = {"vision": True} if "image" in in_mods else None
                # Tool-Unterstützung aus supported_features (z.B. groq/compound
                # hat sie NICHT — Requests mit Tools 400en dort hart).
                feats = getattr(model, "supported_features", None) or []
                if feats:
                    caps = dict(caps or {})
                    caps["tool_call"] = "tools" in feats
                # Non-text outputs (TTS like orpheus) are not chat models —
                # typing them chat made them appear in chat-only client lists.
                if "text" not in out_mods:
                    mtype = "tts" if "audio" in out_mods else "other"
                else:
                    mtype = "chat"
                out.append(ModelInfo(
                    id=model.id,
                    owned_by=getattr(model, "owned_by", None),
                    created=getattr(model, "created", None),
                    type=mtype,
                    context_window=getattr(model, "context_window", None),
                    # Groq hard-caps max_tokens per model (e.g. qwen3.8-27b:
                    # 16384); carrying it lets callers/routers clamp.
                    max_output=(getattr(model, "max_completion_tokens", None)
                                or getattr(model, "max_output_length", None)),
                    access="free",  # universally free (forever-free tier, no CC; API carries no pricing)
                    status="active" if getattr(model, "active", True) else "deprecated",
                    modalities={"input": in_mods, "output": out_mods},
                    capabilities=caps,
                ))
            return out
        except Exception as e:
            import logging
            logging.warning("Failed to fetch Groq models: %s", str(e))
            return []

    def stream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> Iterator[ChatCompletionResponse]:
        """
        Stream a chat completion response from Groq.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional Groq-specific parameters.

        Returns:
            Iterator[ChatCompletionResponse]: An iterator of response chunks.

        Raises:
            Exception: If the request fails.
        """
        def _flatten_messages(msgs: List[ChatMessage]) -> List[Dict[str, Any]]:
            flattened = []
            for m in msgs:
                md = m.to_dict()
                content = md.get("content")
                if isinstance(content, list):
                    parts: List[str] = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            parts.append(part.get("text", ""))
                    if parts:
                        md["content"] = "".join(parts)
                    else:
                        md["content"] = "".join(str(p) for p in content)
                flattened.append(md)
            return flattened

        messages = _flatten_messages(request.messages)

        # Prepare parameters
        params = self._build_groq_params(request, True, provider_specific_kwargs)

        try:
            # Make the streaming request
            completion_stream = self.client.chat.completions.create(**params)

            for chunk in completion_stream:
                delta = chunk.choices[0].delta
                content = getattr(delta, 'content', None)
                # Handle reasoning_content (Groq R1 models)
                reasoning_content = getattr(delta, 'reasoning_content', None)
                tool_calls = None
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    tool_calls = [
                        {
                            "id": getattr(tc, 'id', None),
                            "type": getattr(tc, 'type', 'function'),
                            "function": {
                                "name": tc.function.name if hasattr(tc, 'function') else None,
                                "arguments": tc.function.arguments if hasattr(tc, 'function') else None
                            }
                        }
                        for tc in delta.tool_calls
                    ]

                if not content and not tool_calls and not reasoning_content:
                    continue

                message = ChatMessage(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls
                )

                # No usage stats in streaming mode
                usage = {}

                yield ChatCompletionResponse(
                    message=message,
                    provider='groq',
                    model=params["model"],
                    usage=usage,
                    raw_response={"delta": {"content": content, "tool_calls": tool_calls, "reasoning_content": reasoning_content}},
                    thinking=reasoning_content
                )
        except Exception as e:
            status_code = getattr(e, 'status_code', None)
            response_body = getattr(e, 'body', None) or getattr(e, 'response', None)
            if hasattr(response_body, 'text'):
                response_body = response_body.text
            raise map_provider_error("Groq", e, status_code=status_code, response_body=str(response_body) if response_body else None)
