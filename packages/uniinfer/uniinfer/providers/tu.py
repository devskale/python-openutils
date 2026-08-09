from __future__ import annotations
"""
OpenAI-compliant TU provider implementation.
"""
from typing import Any, AsyncIterator
import asyncio
import httpx
import json
import logging
import os
from datetime import datetime

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, REASONING_OFF

from ..errors import map_provider_error, UniInferError
from ..logging_utils import log_raw_response

logger = logging.getLogger(__name__)

TU_BASE_URL = "https://aqueduct.ai.datalab.tuwien.ac.at/v1"
TU_STAGING_BASE_URL = "https://aqueduct-staging.ai.datalab.tuwien.ac.at/v1"

# Raw per-request/chunk logging to logs/tu_raw_chat.log is extremely verbose
# (every SSE line) and was filling disk (196 MB+). It is now gated behind
# the UNIINFER_DEBUG_RAW env var (off by default). Rare error-path diagnostics
# (preemption / empty stream) are still logged regardless of this flag.
def _raw_logging_enabled() -> bool:
    return os.getenv("UNIINFER_DEBUG_RAW", "").lower() in {"1", "true", "yes"}



def _parse_retry_after(headers: Any) -> float | None:
    """Parse a Retry-After header (delta-seconds or HTTP-date) if present."""
    raw = None
    if hasattr(headers, "get"):
        raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    raw = str(raw).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(raw)
        if dt is not None:
            delta = (dt - datetime.now()).total_seconds()
            return delta if delta > 0 else None
    except (TypeError, ValueError):
        pass
    return None


# Process-wide pool of httpx clients keyed by base_url — one client per
# upstream, reused across all requests (no per-request client creation → no
# native TLS/buffer leak under streaming load). See _get_async_client.
_TU_CLIENT_CACHE: dict[str, httpx.AsyncClient] = {}


class TUProvider(ChatProvider):
    """TU (Tencent Unbounded) LLM Provider implementation."""

    _DEFAULT_MAX_TOKENS = 8192
    ACCESS_TIER = "granted"  # key-granted: TU Wien Aqueduct (university-hosted, access via key — not a public free tier, not paid-$)
    _CREDGOO_SERVICE = "tu"
    _DEFAULT_BASE_URL = TU_BASE_URL
    # OpenAI passthrough params to NEVER forward to vLLM even if a client sends
    # them as extras. Empty by default (vLLM ignores unknowns).
    EXTRA_FORWARD_DENY: frozenset[str] = frozenset()

    def __init__(self, api_key: str | None = None, base_url: str | None = None, supports_reasoning_effort: bool = False):
        """Initialize the TU provider.
        
        Args:
            api_key: The API key for TU. Defaults to TU_API_KEY env var.
            base_url: The base URL for the API.
            supports_reasoning_effort: Whether the backend supports reasoning_effort parameter.
                                       Aqueduct-backed endpoints typically don't support this.
        """
        self.api_key = api_key or os.getenv("TU_API_KEY")
        if not self.api_key:
            try:
                from credgoo import get_api_key
                self.api_key = get_api_key(self._CREDGOO_SERVICE)
            except (ImportError, Exception):
                pass
        self.base_url = base_url or self._DEFAULT_BASE_URL
        self.supports_reasoning_effort = supports_reasoning_effort
        self._async_client: httpx.AsyncClient | None = None
        self._owns_client = True  # False once _get_async_client returns a pooled client
        
    async def _get_async_client(self) -> httpx.AsyncClient:
        """Get the shared (per-base_url) httpx.AsyncClient for TU.

        Pooled per base_url so every request reuses ONE client (and its
        connection pool) instead of minting one per Target — per-request
        creation leaked native TLS/buffer memory under streaming load. The
        client is cached process-wide; aclose() skips it (see _owns_client)."""
        # Respect a caller/test-injected client before consulting the pool.
        if self._async_client is not None and not self._async_client.is_closed:
            return self._async_client
        client = _TU_CLIENT_CACHE.get(self.base_url)
        if client is None or client.is_closed:
            # UNIINFER_TU_KEEPALIVE=0 disables connection keepalive (A/B for the
            # retained-per-request-buffer leak hypothesis: httpcore can hold
            # read/write buffers on kept-alive connections under streaming load).
            _keepalive = int(os.getenv("UNIINFER_TU_KEEPALIVE", "20"))
            limits = httpx.Limits(max_connections=100,
                                 max_keepalive_connections=_keepalive,
                                 keepalive_expiry=5.0)
            client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=httpx.Timeout(300.0, connect=30.0),  # 5 min timeout for large models
                limits=limits,
            )
            _TU_CLIENT_CACHE[self.base_url] = client
        self._async_client = client
        self._owns_client = False
        return client

    async def aclose(self) -> None:
        """TU's httpx client is process-pooled per base_url (see
        _get_async_client) — never close it per request."""
        return

    async def _post_with_ratelimit_retry(
        self, client: httpx.AsyncClient, url: str, payload: dict[str, Any], model: str, max_retries: int = 4
    ) -> httpx.Response:
        """POST with transparent 429 + transport-error retries.

        Upstream HTTP 429 is NOT throttled or internally retried — it is relayed
        to the caller as a RateLimitError immediately (the client does its own
        backoff). Only transient transport errors are retried a few times.
        """
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                response = await client.post(url, json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_exc = e
                logger.warning("[tu] network error on %s (attempt %d/%d): %s", model, attempt + 1, max_retries + 1, e)
                if attempt < max_retries:
                    await asyncio.sleep(min(2.0 * (attempt + 1), 8.0))
                    continue
                raise map_provider_error(self._CREDGOO_SERVICE, e)
            if response.status_code == 429:
                # Transparent rate-limit transport: no proxy-side throttle — just
                # relay the upstream 429 to the caller; the client does its own
                # backoff instead of hanging on an internal replay.
                logger.warning("[%s] 429 on model %s — relaying 429 to caller", self._CREDGOO_SERVICE, model)
                raise map_provider_error(
                    self._CREDGOO_SERVICE,
                    Exception(f"TU API error: 429 - {response.text}"),
                    status_code=429,
                    response_body=response.text,
                )
            return response
        if last_exc is not None:
            raise map_provider_error(self._CREDGOO_SERVICE, last_exc)
        raise map_provider_error(self._CREDGOO_SERVICE, Exception("TU API error: exhausted retries"))

    async def _open_stream_with_ratelimit_retry(
        self, client: httpx.AsyncClient, url: str, payload: dict[str, Any], model: str, max_retries: int = 4
    ):
        """Open a streaming POST with transparent 429 + transport-error retries.

        Returns the ``(context_manager, response)`` pair; the caller must exit
        the context manager (e.g. via ``finally``). Upstream 429 is relayed to
        the caller as a RateLimitError immediately (no throttle/retry); only
        transient transport errors are retried a few times.
        """
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                cm = client.stream("POST", url, json=payload)
                response = await cm.__aenter__()
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_exc = e
                logger.warning("[tu] network error on %s stream (attempt %d/%d): %s", model, attempt + 1, max_retries + 1, e)
                if attempt < max_retries:
                    await asyncio.sleep(min(2.0 * (attempt + 1), 8.0))
                    continue
                raise map_provider_error(self._CREDGOO_SERVICE, e)
            if response.status_code == 429:
                # Transparent rate-limit transport (see _post_with_ratelimit_retry):
                # no proxy-side throttle — surface the upstream 429 to the caller.
                logger.warning("[%s] 429 on model %s stream — relaying 429 to caller", self._CREDGOO_SERVICE, model)
                error_body = await response.aread()
                try:
                    await cm.__aexit__(None, None, None)
                except Exception:
                    pass
                raise map_provider_error(
                    self._CREDGOO_SERVICE,
                    Exception(f"TU API error: 429 - {error_body}"),
                    status_code=429,
                    response_body=error_body,
                )
            if response.status_code != 200:
                error_body = await response.aread()
                log_raw_response(
                    provider=self._CREDGOO_SERVICE,
                    operation="chat.completions.stream",
                    raw_response={
                        "status_code": response.status_code,
                        "body": error_body.decode("utf-8", errors="replace"),
                    },
                    log_file=os.path.join(os.getcwd(), "logs", "tu_raw_chat.log"),
                )
                try:
                    await cm.__aexit__(None, None, None)
                except Exception:
                    pass
                raise map_provider_error(
                    self._CREDGOO_SERVICE,
                    Exception(f"TU API error: {response.status_code} - {error_body}"),
                    status_code=response.status_code,
                    response_body=error_body,
                )
            return cm, response
        if last_exc is not None:
            raise map_provider_error(self._CREDGOO_SERVICE, last_exc)
        raise map_provider_error(self._CREDGOO_SERVICE, Exception("TU API error: exhausted stream retries"))

    def _prepare_payload(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """Prepare the request payload for the TU API.
        
        Args:
            request: The ChatCompletionRequest object.
            
        Returns:
            dict[str, Any]: The payload for the API request.
        """
        messages = []
        for m in request.messages:
            content = m.content
            # Preserve multimodal OpenAI-style content (text + image_url) for VLM models.
            # Fallback: if list is malformed/non-dict, keep only text parts.
            if isinstance(content, list):
                if all(isinstance(part, dict) and part.get("type") in {"text", "image_url"} for part in content):
                    content = content
                else:
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    content = "".join(text_parts) if text_parts else None

            msg = {"role": m.role, "content": content}
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
        payload["max_tokens"] = request.max_tokens or self._DEFAULT_MAX_TOKENS
            
        if request.tools:
            payload["tools"] = request.tools
            # `tool_choice="required"` is intentionally NOT supported on TU.
            # vLLM's required path uses constrained decoding, which conflicts
            # with the GLM-5.x reasoning/tool parser and produces deterministic
            # upstream 500s on reasoning models (e.g. glm-5.2-744b-preview).
            # See vLLM #42400, #39757, #36857 and vLLM forum #1945.
            # Fail fast with a clear message instead of forwarding a broken call.
            # To force a specific tool, pass a named tool_choice, e.g.:
            #   {"type": "function", "function": {"name": "..."}}
            if request.tool_choice == "required":
                raise ValueError(
                    "tool_choice='required' is not supported by the TU provider "
                    "(upstream vLLM constrained-decoding bug on reasoning models). "
                    "Use tool_choice='auto', or a named tool_choice to force a specific tool."
                )
            if request.tool_choice:
                payload["tool_choice"] = request.tool_choice
        
        if request.reasoning_effort and self.supports_reasoning_effort:
            payload["reasoning_effort"] = request.reasoning_effort

        # chat_template_kwargs — generic vLLM passthrough and the RELIABLE
        # thinking knob (top-level enable_thinking is silently ignored by
        # Qwen3.x / GLM-5.x; vLLM #35574). Forwarded verbatim: the escape hatch.
        ctk = dict(request.chat_template_kwargs or {})
        # reasoning_effort none/minimal disables reasoning (the cross-provider
        # contract). Inject the default unless the caller already expressed
        # intent via the escape hatch (explicit chat_template_kwargs wins).
        if request.reasoning_effort in REASONING_OFF and "enable_thinking" not in ctk:
            ctk["enable_thinking"] = False
        if ctk:
            payload["chat_template_kwargs"] = ctk

        # OpenAI passthrough: forward unmapped OpenAI params (top_p,
        # response_format, seed, stream_options, logprobs, …) verbatim so new
        # OpenAI features reach vLLM without a per-field code change. Critically
        # this carries stream_options.include_usage so vLLM emits a terminal
        # usage chunk — without it, streaming consumers never see token counts.
        if getattr(request, "extra", None):
            _deny = getattr(self, "EXTRA_FORWARD_DENY", None) or frozenset()
            for _k, _v in request.extra.items():
                if _k not in _deny:
                    payload[_k] = _v

        return payload

    async def acomplete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Async completion implementation for TU."""
        client = await self._get_async_client()
        payload = self._prepare_payload(request)

        try:
            response = await self._post_with_ratelimit_retry(client, "/chat/completions", payload, request.model)

            # Log raw response for debugging (gated: very verbose)
            raw_text = response.text
            if _raw_logging_enabled():
                log_raw_response(
                    provider=self._CREDGOO_SERVICE,
                    operation="chat.completions",
                    raw_response={
                        "status_code": response.status_code,
                        "body": raw_text[:1000] if raw_text else "(empty)",
                    },
                    log_file=os.path.join(os.getcwd(), "logs", "tu_raw_chat.log"),
                )
            
            if response.status_code != 200:
                raise map_provider_error(self._CREDGOO_SERVICE, Exception(f"TU API error: {response.status_code} - {raw_text}"), status_code=response.status_code, response_body=raw_text)

            if not raw_text or not raw_text.strip():
                raise map_provider_error(self._CREDGOO_SERVICE, Exception("TU API returned empty response"), status_code=500, response_body="(empty)")
            
            try:
                data = response.json()
            except Exception as json_err:
                raise map_provider_error(self._CREDGOO_SERVICE, Exception(f"TU API JSON parse error: {json_err}. Response: {raw_text[:500]}"), status_code=500, response_body=raw_text)
            choice = data["choices"][0]
            message_data = choice["message"]
            
            message = ChatMessage(
                role=message_data["role"],
                content=message_data.get("content"),
                tool_calls=message_data.get("tool_calls"),
                tool_call_id=message_data.get("tool_call_id")
            )
            
            # Handle reasoning_content (TU thinking models)
            reasoning_content = message_data.get("reasoning_content") or message_data.get("reasoning")
            
            return ChatCompletionResponse(
                message=message,
                provider=self._CREDGOO_SERVICE,
                model=data.get("model", request.model or "qwen-coder-30b"),
                usage=data.get("usage", {}),
                raw_response=data,
                finish_reason=choice.get("finish_reason"),
                thinking=reasoning_content
            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error(self._CREDGOO_SERVICE, e)

    async def astream_complete(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionResponse]:
        """Async streaming completion implementation for TU."""
        request.streaming = True
        client = await self._get_async_client()
        payload = self._prepare_payload(request)

        cm, response = await self._open_stream_with_ratelimit_retry(client, "/chat/completions", payload, request.model)
        try:
            chunks_yielded = 0  # Track if we receive any valid chunks
            received_done = False  # Track if we received [DONE] marker
            received_finish_reason = False  # Track if we received finish_reason
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.strip() == 'data: [DONE]':
                    received_done = True
                    break
                if not line.startswith('data: '):
                    continue
                # Per-chunk logging is extremely verbose and was filling disk.
                # Only enabled when UNIINFER_DEBUG_RAW=1.
                if _raw_logging_enabled():
                    log_raw_response(
                        provider=self._CREDGOO_SERVICE,
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
                        
                        if finish_reason:
                            received_finish_reason = True
                        
                        content = delta.get('content')
                        # Handle reasoning_content (TU thinking models)
                        reasoning_content = delta.get('reasoning_content') or delta.get('reasoning')
                        tool_calls = delta.get('tool_calls')
                        
                        if not content and not reasoning_content and not tool_calls and not finish_reason:
                            if not data.get("usage"):
                                continue
                            
                        chunks_yielded += 1
                        message = ChatMessage(
                            role=delta.get('role', 'assistant'),
                            content=content,
                            tool_calls=tool_calls
                        )
                        
                        yield ChatCompletionResponse(
                            message=message,
                            provider=self._CREDGOO_SERVICE,
                            model=data.get("model", request.model),
                            usage=data.get("usage") or {},
                            raw_response=data,
                            finish_reason=finish_reason,
                            thinking=reasoning_content  # Separate thinking content
                        )
                    elif data.get("usage"):
                        # Terminal usage-only chunk (choices:[]). vLLM emits this
                        # when stream_options.include_usage is set; forward it so
                        # the proxy can emit usage to clients.
                        yield ChatCompletionResponse(
                            message=ChatMessage(role="assistant", content=None),
                            provider=self._CREDGOO_SERVICE,
                            model=data.get("model", request.model),
                            usage=data["usage"],
                            raw_response=data,
                            finish_reason=None,
                            thinking=None,
                        )
                except json.JSONDecodeError:
                    continue
            
            # Detect incomplete stream (preemption) - stream ended without proper completion
            if chunks_yielded == 0:
                log_raw_response(
                    provider=self._CREDGOO_SERVICE,
                    operation="chat.completions.stream",
                    raw_response={"error": "Empty stream - no chunks received (possible preemption)"},
                    log_file=os.path.join(os.getcwd(), "logs", "tu_raw_chat.log"),
                )
                # Yield error response instead of raising - exception would be swallowed by StopAsyncIteration
                yield ChatCompletionResponse(
                    message=ChatMessage(role="assistant", content=""),
                    provider=self._CREDGOO_SERVICE,
                    model=request.model,
                    usage={},
                    raw_response={"error": "TU API returned empty stream - model may have preempted"},
                    finish_reason="error",
                    thinking=None
                )
                return  # Exit generator cleanly
            
            # Detect premature stream termination - stream had chunks but no finish_reason or [DONE]
            if not received_done and not received_finish_reason:
                import sys
                print(f"[DEBUG] PREEMPTION DETECTED: {chunks_yielded} chunks, no finish_reason or [DONE]", file=sys.stderr, flush=True)
                log_raw_response(
                    provider=self._CREDGOO_SERVICE,
                    operation="chat.completions.stream",
                    raw_response={"error": f"Stream terminated prematurely - {chunks_yielded} chunks but no finish_reason or [DONE] (possible preemption)"},
                    log_file=os.path.join(os.getcwd(), "logs", "tu_raw_chat.log"),
                )
                # Yield error response instead of raising - exception would be swallowed by StopAsyncIteration
                yield ChatCompletionResponse(
                    message=ChatMessage(role="assistant", content=""),
                    provider=self._CREDGOO_SERVICE,
                    model=request.model,
                    usage={},
                    raw_response={"error": f"TU API stream terminated prematurely after {chunks_yielded} chunks - model may have preempted"},
                    finish_reason="error",
                    thinking=None
                )
                return  # Exit generator cleanly
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error(self._CREDGOO_SERVICE, e)
        finally:
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass

    @classmethod
    def list_models(cls, api_key: str | None = None, **kwargs) -> list[ModelInfo]:
        from ..core import ModelInfo
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
                from credgoo import get_api_key
                api_key = get_api_key(cls._CREDGOO_SERVICE)
            except (ImportError, Exception):
                pass
        
        if not api_key:
            return []
            
        base_url = kwargs.get("base_url") or cls._DEFAULT_BASE_URL
        
        try:
            import requests
            response = requests.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                results = []
                for model in data.get("data", []):
                    mid = model["id"]
                    m = ModelInfo(id=mid, owned_by=model.get("owned_by"), created=model.get("created"), access="granted", raw=model)
                    m.type = m.derive_type()
                    results.append(m)
                return results
        except Exception:
            pass
            
        return []


class TUStagingProvider(TUProvider):
    """TU Staging provider — uses staging base URL and staging API key."""

    _CREDGOO_SERVICE = "tu-staging"
    _DEFAULT_BASE_URL = TU_STAGING_BASE_URL
