"""Completion dispatch handle — the deep module behind chat completion.

A ``Target`` binds a ``provider@model`` string to a ready provider instance and
exposes the four completion paths (sync/async x stream/non-stream) behind one
small interface. It owns parse -> instantiate -> request-build -> dispatch ->
access-recording, collapsing the six duplicated copies that lived in provider_access.py
and capabilities. Embeddings stay separate (different factory + request type);
only the parse is shared. See CONTEXT.md.
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Iterator, Optional

from .config.providers import PROVIDER_CONFIGS
from .core import ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from .factory import ProviderFactory
from .json_utils import update_model_accessed

logger = logging.getLogger(__name__)


def parse_provider_model(provider_model: str) -> tuple[str, str]:
    """Split ``provider@model`` into (provider_name, model_name).

    Raises:
        ValueError: if the string is not ``provider@model`` or either side is
            empty. The library-level parse; the HTTP seam translates this to
            HTTPException (see proxy_app.parse_provider_model).
    """
    if "@" not in provider_model:
        raise ValueError(
            "Invalid provider_model format. Expected 'provider@modelname'."
        )
    provider_name, model_name = provider_model.split("@", 1)
    if not provider_name or not model_name:
        raise ValueError(
            "Invalid provider_model format. Provider or model name is empty."
        )
    return provider_name, model_name


def _extra_params(provider_name: str) -> dict[str, Any]:
    """Extra constructor params declared for a provider in PROVIDER_CONFIGS.

    Data-driven replacement for the old ``if provider == 'cloudflare'`` hardcode.
    Explicit caller kwargs (api_key, base_url) take precedence over these.
    """
    return dict(PROVIDER_CONFIGS.get(provider_name, {}).get("extra_params", {}))


class Target:
    """A completion dispatch handle for a ``provider@model``.

    Owns parse, provider instantiation (incl. config extra_params), request
    building, dispatch, and (optionally) model-access recording. The deep module
    behind chat completion — callers cross this seam instead of re-implementing
    the parse/instantiate/dispatch sequence.

    All four completion methods build the same uniform request (only the
    ``streaming`` flag and sync/async dispatch differ) and yield raw
    ``ChatCompletionResponse`` chunks (stream paths). Proxy-specific OpenAI/SSE
    shaping lives in the proxy layer, not here.

    Args:
        provider_model: ``provider@model`` string.
        api_key: resolved provider API key (None for providers like Ollama).
        base_url: optional provider base URL.
        record_access: if True (default), record model access on successful
            completion. Diagnostic callers (e.g. capability probes) pass False
            so probe traffic doesn't pollute usage metadata.
    """

    def __init__(
        self,
        provider_model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        record_access: bool = True,
    ) -> None:
        self.provider_model = provider_model
        self.provider_name, self.model_name = parse_provider_model(provider_model)
        # Explicit caller kwargs win over config extra_params defaults.
        kwargs: dict[str, Any] = {}
        kwargs.update(_extra_params(self.provider_name))
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        self.provider = ProviderFactory.get_provider(self.provider_name, **kwargs)
        self._record_access = record_access

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _build_request(
        self,
        messages: list[Any],
        *,
        temperature: float,
        max_tokens: Optional[int],
        streaming: bool,
        tools: Optional[list[dict]] = None,
        tool_choice: Any = None,
        reasoning_effort: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = None,
        extra: Optional[dict] = None,
    ) -> ChatCompletionRequest:
        msgs = [
            m if isinstance(m, ChatMessage) else ChatMessage(**m) for m in messages
        ]
        return ChatCompletionRequest(
            messages=msgs,
            model=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            chat_template_kwargs=chat_template_kwargs,
            extra=extra,
        )

    def _record(self, resp: Any) -> None:
        """Record model access after a successful non-stream completion."""
        if self._record_access and getattr(resp, "message", None):
            try:
                update_model_accessed(self.model_name, self.provider_name)
            except Exception as e:  # access tracking must never break completion
                logger.warning("update_model_accessed failed: %s", e)

    def _record_stream(self) -> None:
        """Record model access after a stream completes."""
        if self._record_access:
            try:
                update_model_accessed(self.model_name, self.provider_name)
            except Exception as e:  # access tracking must never break completion
                logger.warning("update_model_accessed failed: %s", e)

    # ------------------------------------------------------------------ #
    # dispatch — four paths over one request builder
    # ------------------------------------------------------------------ #
    def complete(
        self,
        messages: list[Any],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Any = None,
        reasoning_effort: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = None,
        extra: Optional[dict] = None,
    ) -> ChatCompletionResponse:
        req = self._build_request(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=False,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            chat_template_kwargs=chat_template_kwargs,
            extra=extra,
        )
        logger.info("Requesting response from %s (%s)", self.provider_name, self.model_name)
        resp = self.provider.complete(req)
        self._record(resp)
        return resp

    def stream_complete(
        self,
        messages: list[Any],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Any = None,
        reasoning_effort: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = None,
        extra: Optional[dict] = None,
    ) -> Iterator[ChatCompletionResponse]:
        req = self._build_request(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            chat_template_kwargs=chat_template_kwargs,
            extra=extra,
        )
        logger.info("Streaming response from %s (%s)", self.provider_name, self.model_name)
        for chunk in self.provider.stream_complete(req):
            yield chunk
        self._record_stream()

    async def acomplete(
        self,
        messages: list[Any],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Any = None,
        reasoning_effort: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = None,
        extra: Optional[dict] = None,
    ) -> ChatCompletionResponse:
        req = self._build_request(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=False,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            chat_template_kwargs=chat_template_kwargs,
            extra=extra,
        )
        logger.info(
            "Requesting async response from %s (%s)", self.provider_name, self.model_name
        )
        resp = await self.provider.acomplete(req)
        self._record(resp)
        return resp

    async def astream_complete(
        self,
        messages: list[Any],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Any = None,
        reasoning_effort: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = None,
        extra: Optional[dict] = None,
    ) -> AsyncIterator[ChatCompletionResponse]:
        req = self._build_request(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            chat_template_kwargs=chat_template_kwargs,
            extra=extra,
        )
        logger.info(
            "Streaming async response from %s (%s)", self.provider_name, self.model_name
        )
        async for chunk in self.provider.astream_complete(req):
            yield chunk
        self._record_stream()
