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
import time
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

# --- Stream robustness knobs -------------------------------------------------
# A TU backend that accepts a request but never answers (its wedged-replica
# failure mode) used to hold a stream open for the FULL httpx read timeout
# (300s) before the retry path could act — so a wedged stream produced several-
# minute hangs. These bound that window tightly:
#   * TU_STREAM_OPEN_TIMEOUT — max wait for response headers after POST
#   * TU_STREAM_GAP_TIMEOUT  — max idle time between SSE lines mid-stream
# A stream idle longer than the gap is treated as wedged: the pooled client is
# evicted and (if no chunk has reached the caller yet) the stream is replayed
# on a fresh connection. Both are overridable via env for targeted tuning.
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default

TU_STREAM_OPEN_TIMEOUT = _env_float("TU_STREAM_OPEN_TIMEOUT", 90.0)
TU_STREAM_GAP_TIMEOUT = _env_float("TU_STREAM_GAP_TIMEOUT", 60.0)


class StreamStalledError(Exception):
    """Internal: a TU stream produced no data within the idle-gap timeout
    (wedged backend). Caught in ``astream_complete`` so it can replay from a
    fresh connection while no chunk has been emitted yet, otherwise it is
    surfaced to the caller as an upstream error."""


class TUTelemetry:
    """Process-wide wedge/stall telemetry for the TU upstream, surfaced through
    the proxy /health ``upstream`` block so an operator can see at a glance how
    much is wedged or hanging right now."""

    def __init__(self) -> None:
        self.evictions = 0            # pooled httpx client evictions
        self.transport_retries = 0    # attempts after the first (retries)
        self.open_stalls = 0          # read timeouts while (re-)opening a stream
        self.body_stalls = 0          # idle-gap timeouts mid-stream
        self.stall_retries = 0        # successful replays after a pre-first-chunk stall
        self.rate_limits = 0          # upstream 429s relayed to the caller
        self.prime_retries = 0        # first-token window hit once (RC#5 retry)
        self.prime_timeouts = 0       # first token never arrived (2 windows) — 504 to caller
        self.last_prime_timeout: dict | None = None  # {model, ts(monotonic)}
        self.active_streams: dict[int, dict] = {}  # id -> {start,last,model}
        self._seq = 0

    def note_prime_retry(self, model: str) -> None:
        """First-token window elapsed once; the RC#5 retry is under way."""
        self.prime_retries += 1

    def note_prime_timeout(self, model: str) -> None:
        """Both first-token windows elapsed — caller gets a 504 chunk."""
        self.prime_timeouts += 1
        self.last_prime_timeout = {"model": model, "ts": time.monotonic()}

    def register_stream(self, model: str) -> int:
        self._seq += 1
        now = time.monotonic()
        self.active_streams[self._seq] = {"start": now, "last": now, "model": model}
        return self._seq

    def touch(self, sid: int) -> None:
        s = self.active_streams.get(sid)
        if s is not None:
            s["last"] = time.monotonic()

    def unregister(self, sid: int) -> None:
        self.active_streams.pop(sid, None)

    def snapshot(self, stalled_after_s: float) -> dict:
        now = time.monotonic()
        stuck = 0
        streams = []
        for s in self.active_streams.values():
            idle = now - s["last"]
            streams.append({
                "model": s["model"],
                "age_s": round(now - s["start"], 1),
                "idle_s": round(idle, 1),
            })
            if idle >= stalled_after_s:
                stuck += 1
        return {
            "evictions": self.evictions,
            "transport_retries": self.transport_retries,
            "open_stalls": self.open_stalls,
            "body_stalls": self.body_stalls,
            "stall_retries": self.stall_retries,
            "rate_limits": self.rate_limits,
            "prime_retries": self.prime_retries,
            "prime_timeouts": self.prime_timeouts,
            "last_prime_timeout": (
                {
                    "model": self.last_prime_timeout["model"],
                    "ago_s": round(now - self.last_prime_timeout["ts"], 1),
                }
                if self.last_prime_timeout
                else None
            ),
            "in_flight": len(streams),
            "stuck_streams": stuck,
            "stall_threshold_s": round(stalled_after_s, 1),
            "streams": streams,
        }


_TU_TELEMETRY = TUTelemetry()


async def clear_wedge_state() -> dict:
    """Force-drop + close every pooled TU client and reset the live wedge registry.

    Operator-triggered (proxy POST /debug/wedge/clear) when /health reports
    stuck streams. Closing the pooled httpx clients aborts any wedged in-flight
    request (kill the hang) and empties the pool, so every provider instance
    mints a brand-new connection (with its own API key) on its next request
    instead of reusing the wedged socket. Counters are kept for the historical
    picture; only the *live* stuck state is reset.

    Returns what was dropped/cleared so the caller (and /health) can confirm.
    """
    active = dict(_TU_TELEMETRY.active_streams)
    now = time.monotonic()
    cleared_streams = [
        {"model": s["model"], "idle_s": round(now - s["last"], 1)}
        for s in active.values()
    ]
    _TU_TELEMETRY.active_streams.clear()
    clients = list(_TU_CLIENT_CACHE.values())
    _TU_CLIENT_CACHE.clear()
    for client in clients:
        try:
            await client.aclose()
        except Exception:
            pass
    return {
        "closed_clients": len(clients),
        "cleared_streams": len(cleared_streams),
        "streams": cleared_streams,
    }


class TUProvider(ChatProvider):
    """TU (Tencent Unbounded) LLM Provider implementation."""

    _DEFAULT_MAX_TOKENS = 8192
    # Thinking-disable keys across the reasoning families served by TU (vLLM).
    # Qwen3.x/Gemma 4 read "enable_thinking"; DeepSeek V3.1/V4 + Holo2 read
    # "thinking". A reasoning-off intent fans BOTH out — each chat template
    # picks the key it understands and ignores the rest, so no per-model
    # registry is needed (verified live 2026-08-14: deepseek-v4-flash-284b
    # honors thinking:false, ignores enable_thinking; qwen-3.6-35b is the
    # inverse). New reasoning families pick one of these spellings for free.
    _THINKING_OFF_KNOBS: tuple[str, ...] = ("enable_thinking", "thinking")
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
        self._key_source = "explicit" if api_key else ("env" if os.getenv("TU_API_KEY") else "credgoo")
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
        
    def _new_async_client(self) -> httpx.AsyncClient:
        """Mint a fresh AsyncClient for this provider's base_url."""
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=httpx.Timeout(300.0, connect=30.0),  # 5 min timeout for large models
            http2=True,
        )

    def _refresh_credgoo_key(self) -> bool:
        """Refetch the API key from credgoo after an upstream 401.

        credgoo-issued keys rotate without notice; a long-running process (the
        amd proxy) holds the old key until restart. Returns True when the key
        actually changed (caller should rebuild the pooled client — the key is
        baked into its Authorization header — and retry). Explicit/env keys are
        never refreshed here: the caller owns them.
        """
        if self._key_source != "credgoo":
            return False
        try:
            from credgoo import get_api_key
            fresh = get_api_key(self._CREDGOO_SERVICE)
        except Exception as e:
            logger.warning("[%s] credgoo key refresh failed: %s", self._CREDGOO_SERVICE, e)
            return False
        if fresh and fresh != self.api_key:
            logger.info("[%s] credgoo key rotated — updated", self._CREDGOO_SERVICE)
            self.api_key = fresh
            return True
        return False

    def _replace_pooled_client(self) -> httpx.AsyncClient:
        """Evict the pooled client after a read timeout (wedged upstream backend).

        A backend that accepts the request but never answers (TU's wedged-
        replica failure mode) pins the pooled h2 connection — every multiplexed
        request on it hangs for the full read timeout, and so does every retry
        that reuses it. Replacing the pool entry forces the retry onto a fresh
        TCP/TLS connection (fresh load-balancer routing). The old client is
        deliberately NOT closed: in-flight requests on it must finish; it is
        dropped for GC instead.
        """
        _TU_TELEMETRY.evictions += 1
        replacement = self._new_async_client()
        _TU_CLIENT_CACHE[self.base_url] = replacement
        self._async_client = replacement
        self._owns_client = False
        logger.warning(
            "[%s] read timeout — pooled client evicted, retrying on a fresh connection",
            self._CREDGOO_SERVICE,
        )
        return replacement

    async def _get_async_client(self) -> httpx.AsyncClient:
        """Get the shared (per-base_url) httpx.AsyncClient for TU.

        Pooled per base_url so every request reuses ONE client (and its
        connection pool) instead of minting one per Target — per-request
        creation leaked native TLS/buffer memory under streaming load. The
        client is cached process-wide; aclose() skips it (see _owns_client)."""
        # Respect a caller/test-injected client before consulting the pool.
        if self._async_client is not None and not self._async_client.is_closed:
            # Re-sync with the process-wide pool: a concurrent request may have
            # evicted this client after a read timeout (see _replace_pooled_client).
            # Without this, long-lived provider instances would keep serving on
            # the wedged connection forever. Injected clients (_owns_client) are
            # exempt — the caller owns their lifecycle.
            if not self._owns_client:
                pooled = _TU_CLIENT_CACHE.get(self.base_url)
                if pooled is not None and pooled is not self._async_client and not pooled.is_closed:
                    self._async_client = pooled
            return self._async_client
        client = _TU_CLIENT_CACHE.get(self.base_url)
        if client is None or client.is_closed:
            # HTTP/2 multiplexing: one connection per host serves all concurrent
            # streams, so the pool never reaches the contention that triggers the
            # httpcore #1093 connection-slot leak (HTTP/1.1-specific). ALPN falls
            # back to HTTP/1.1 for hosts that don't speak h2. Replaces the earlier
            # keepalive=0 workaround (which cost a TLS handshake per request).
            client = self._new_async_client()
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
            except httpx.TimeoutException as e:
                last_exc = e
                logger.warning("[%s] read timeout on %s (attempt %d/%d): %s", self._CREDGOO_SERVICE, model, attempt + 1, max_retries + 1, e)
                if attempt < max_retries:
                    # A read timeout on a wedged backend poisons the pooled
                    # connection for every request on it — evict it so this
                    # retry (and all concurrent requests) get fresh routing.
                    _TU_TELEMETRY.transport_retries += 1
                    if not self._owns_client:
                        client = self._replace_pooled_client()
                    await asyncio.sleep(min(2.0 * (attempt + 1), 8.0))
                    continue
                raise map_provider_error(self._CREDGOO_SERVICE, e)
            except httpx.TransportError as e:
                last_exc = e
                logger.warning("[tu] network error on %s (attempt %d/%d): %s", model, attempt + 1, max_retries + 1, e)
                if attempt < max_retries:
                    # A wedged-backend hang typically ends as a TransportError
                    # (LB kills the silent connection) — same pool poisoning as
                    # a read timeout, so same eviction + fresh-connection retry.
                    _TU_TELEMETRY.transport_retries += 1
                    if not self._owns_client:
                        client = self._replace_pooled_client()
                    await asyncio.sleep(min(2.0 * (attempt + 1), 8.0))
                    continue
                raise map_provider_error(self._CREDGOO_SERVICE, e)
            if response.status_code == 401 and attempt < max_retries:
                # Key rotation: credgoo-issued keys rotate without notice and the
                # pooled client carries the old one in its Authorization header.
                # Refetch → rebuild → retry; if the key is unchanged (or not
                # credgoo-sourced) raise immediately (permanent error).
                if self._refresh_credgoo_key():
                    if not self._owns_client:
                        client = self._replace_pooled_client()
                    continue
                raise map_provider_error(
                    self._CREDGOO_SERVICE,
                    Exception(f"TU API error: 401 - {response.text}"),
                    status_code=401,
                    response_body=response.text,
                )
            if response.status_code == 429:
                # Transparent rate-limit transport: no proxy-side throttle — just
                # relay the upstream 429 to the caller; the client does its own
                # backoff instead of hanging on an internal replay.
                _TU_TELEMETRY.rate_limits += 1
                logger.warning("[%s] 429 on model %s — relaying 429 to caller", self._CREDGOO_SERVICE, model)
                raise map_provider_error(
                    self._CREDGOO_SERVICE,
                    Exception(f"TU API error: 429 - {response.text}"),
                    status_code=429,
                    response_body=response.text,
                    retry_after=_parse_retry_after(response.headers),
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
        # Bound how long we wait for response headers after POST. TU's wedged
        # backend accepts but never answers; a per-request read timeout here fails
        # that case in TU_STREAM_OPEN_TIMEOUT seconds instead of the client's 300s
        # streaming budget. The injected/retry client reuses its own pool timeout.
        base_t = getattr(client, "timeout", None)
        if isinstance(base_t, httpx.Timeout):
            open_timeout = httpx.Timeout(
                connect=base_t.connect, read=TU_STREAM_OPEN_TIMEOUT,
                write=base_t.write, pool=base_t.pool,
            )
        else:
            open_timeout = httpx.Timeout(TU_STREAM_OPEN_TIMEOUT, connect=30.0)
        for attempt in range(max_retries + 1):
            try:
                cm = client.stream("POST", url, json=payload, timeout=open_timeout)
                response = await cm.__aenter__()
            except httpx.TimeoutException as e:
                last_exc = e
                _TU_TELEMETRY.open_stalls += 1
                logger.warning("[%s] read timeout on %s stream (attempt %d/%d): %s", self._CREDGOO_SERVICE, model, attempt + 1, max_retries + 1, e)
                if attempt < max_retries:
                    # Same wedged-backend eviction as the non-streaming path:
                    # retry on a fresh connection instead of re-hanging on the
                    # poisoned pooled one.
                    _TU_TELEMETRY.transport_retries += 1
                    if not self._owns_client:
                        client = self._replace_pooled_client()
                    await asyncio.sleep(min(2.0 * (attempt + 1), 8.0))
                    continue
                raise map_provider_error(self._CREDGOO_SERVICE, e)
            except httpx.TransportError as e:
                last_exc = e
                logger.warning("[tu] network error on %s stream (attempt %d/%d): %s", model, attempt + 1, max_retries + 1, e)
                if attempt < max_retries:
                    # Wedged-backend hangs usually die as TransportError (LB
                    # kills the silent connection) — evict + retry fresh, same
                    # as the non-streaming path.
                    _TU_TELEMETRY.transport_retries += 1
                    if not self._owns_client:
                        client = self._replace_pooled_client()
                    await asyncio.sleep(min(2.0 * (attempt + 1), 8.0))
                    continue
                raise map_provider_error(self._CREDGOO_SERVICE, e)
            if response.status_code == 401 and attempt < max_retries:
                # Same key-rotation handling as the non-streaming path: refetch
                # from credgoo, rebuild the pooled client (Authorization header),
                # retry. Unchanged key → permanent auth error.
                if self._refresh_credgoo_key():
                    try:
                        await cm.__aexit__(None, None, None)
                    except Exception:
                        pass
                    if not self._owns_client:
                        client = self._replace_pooled_client()
                    continue
                error_body = await response.aread()
                try:
                    await cm.__aexit__(None, None, None)
                except Exception:
                    pass
                raise map_provider_error(
                    self._CREDGOO_SERVICE,
                    Exception(f"TU API error: 401 - {error_body}"),
                    status_code=401,
                    response_body=error_body,
                )
            if response.status_code == 429:
                # Transparent rate-limit transport (see _post_with_ratelimit_retry):
                # no proxy-side throttle — surface the upstream 429 to the caller.
                _TU_TELEMETRY.rate_limits += 1
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
                    retry_after=_parse_retry_after(response.headers),
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
        # contract). Different reasoning families expose DIFFERENT disable keys
        # here, so we inject the union of every known "thinking off" spelling:
        # Qwen3.x/Gemma read enable_thinking, DeepSeek V3.1/V4 + Holo2 read
        # thinking. The chat template consumes the key it understands and
        # ignores the others — no per-model branch needed. Explicit caller
        # knobs win over the injected default (escape-hatch precedence).
        if request.reasoning_effort in REASONING_OFF and not (set(ctk) & set(self._THINKING_OFF_KNOBS)):
            for _knob in self._THINKING_OFF_KNOBS:
                ctk.setdefault(_knob, False)
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
            choice = (data.get("choices") or [{}])[0]
            message_data = choice.get("message", {}) or {}

            content = message_data.get("content")
            tool_calls = message_data.get("tool_calls")
            # Handle reasoning_content (TU thinking models)
            reasoning_content = message_data.get("reasoning_content") or message_data.get("reasoning")

            # TU throttles (>25 req/min) with 200 + empty content instead of a
            # proper 429. Empty is only legitimate when thinking consumed the
            # whole max_tokens budget (reasoning present) or the answer is a
            # tool call — anything else is a rate-limit shadow: map it onto the
            # existing 429 path so client-side backoff engages.
            if not content and not reasoning_content and not tool_calls:
                raise map_provider_error(
                    self._CREDGOO_SERVICE,
                    Exception("TU API returned empty content (suspected rate-limit shadow)"),
                    status_code=429,
                    response_body=raw_text[:500],
                )

            message = ChatMessage(
                role=message_data.get("role", "assistant"),
                content=content,
                tool_calls=tool_calls,
                tool_call_id=message_data.get("tool_call_id")
            )

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
        """Async streaming completion implementation for TU.

        A stream that produces no data within ``TU_STREAM_GAP_TIMEOUT`` is
        treated as wedged: the pooled client is evicted and, while no chunk has
        reached the caller yet (nothing to replay), the stream is replayed on a
        fresh connection. Once data has been emitted it can no longer be
        replayed, so a stall there is surfaced as an upstream error. The retry
        window here only covers body stalls; ``_open_stream_with_ratelimit_retry``
        still owns the open-phase (headers) retries.
        """
        request.streaming = True
        payload = self._prepare_payload(request)
        max_retries = 4
        last_stall: StreamStalledError | None = None
        for attempt in range(max_retries + 1):
            client = await self._get_async_client()
            cm = response = None
            chunks_yielded = 0  # Track if we receive any valid chunks
            received_done = False  # Track if we received [DONE] marker
            received_finish_reason = False  # Track if we received finish_reason
            received_payload = False  # Track if any content/reasoning/tool_calls arrived
            try:
                cm, response = await self._open_stream_with_ratelimit_retry(
                    client, "/chat/completions", payload, request.model
                )
                sid = _TU_TELEMETRY.register_stream(request.model)
                try:
                    it = response.aiter_lines()
                    while True:
                        try:
                            line = await asyncio.wait_for(it.__anext__(), timeout=TU_STREAM_GAP_TIMEOUT)
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError:
                            # Wedged: no data within the idle gap. Evict the
                            # pooled connection and (if nothing emitted yet)
                            # replay fresh — handled in the except below.
                            _TU_TELEMETRY.body_stalls += 1
                            _TU_TELEMETRY.touch(sid)
                            raise StreamStalledError(request.model) from None
                        _TU_TELEMETRY.touch(sid)
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
                                if content or reasoning_content or tool_calls:
                                    received_payload = True

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
                finally:
                    _TU_TELEMETRY.unregister(sid)

            except StreamStalledError as e:
                last_stall = e
                if chunks_yielded > 0 or attempt >= max_retries:
                    # Already streamed data to the caller (can't replay without
                    # duplicating the prefix) or out of attempts: clear the
                    # wedged connection for concurrent requests and surface the
                    # stall as an error.
                    if not self._owns_client:
                        client = self._replace_pooled_client()
                    raise map_provider_error(
                        self._CREDGOO_SERVICE,
                        Exception(f"TU stream stalled after {chunks_yielded} chunks on {request.model}"),
                    ) from e
                # Nothing emitted yet → safe to replay the whole stream on a
                # fresh connection (fresh LB routing) instead of erroring out.
                _TU_TELEMETRY.stall_retries += 1
                if not self._owns_client:
                    client = self._replace_pooled_client()
                await asyncio.sleep(min(2.0 * (attempt + 1), 8.0))
                continue
            except Exception as e:
                if isinstance(e, UniInferError):
                    raise
                raise map_provider_error(self._CREDGOO_SERVICE, e)
            finally:
                if cm is not None:
                    try:
                        await cm.__aexit__(None, None, None)
                    except Exception:
                        pass

            # Stream ended: run the (unchanged) completeness guards.
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

            # Stream completed (finish_reason/[DONE]) but never carried
            # content, reasoning, or tool calls — TU's silent rate-limit shadow
            # (200, empty). Surface it as an error marker (same pattern as
            # preemption above) instead of an empty "success" the caller
            # would retry blindly.
            if not received_payload:
                log_raw_response(
                    provider=self._CREDGOO_SERVICE,
                    operation="chat.completions.stream",
                    raw_response={"error": "Stream completed with empty content (suspected rate-limit shadow)"},
                    log_file=os.path.join(os.getcwd(), "logs", "tu_raw_chat.log"),
                )
                yield ChatCompletionResponse(
                    message=ChatMessage(role="assistant", content=""),
                    provider=self._CREDGOO_SERVICE,
                    model=request.model,
                    usage={},
                    raw_response={"error": "TU API stream completed with empty content (suspected rate-limit shadow)"},
                    finish_reason="error",
                    thinking=None
                )
                return  # Exit generator cleanly

            return
        raise map_provider_error(
            self._CREDGOO_SERVICE,
            last_stall or Exception("TU stream exhausted retries"),
        )

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
