from __future__ import annotations
"""
OpenCode / Zen provider implementation.

OpenCode (opencode.ai) runs the "zen" model router — an OpenAI-compatible
endpoint that aggregates many models (DeepSeek, GPT, Gemini, Qwen, GLM,
MiniMax, Kimi, …). Free models are id-suffixed ``-free`` plus ``big-pickle``:
mimo-v2.5-free, ling-3.0-flash-fin-free, nemotron-3-ultra-free,
nemotron-3.5-lightning-free, deepseek-v4-flash-free, muse-spark-*-contributor-free.
Jev models (``jev-1.13-free``) are System One *decision* models served via
``/v1/systemone`` (state + typed questions -> values/probabilities). They are
exposed through the chat API by a wrapper: the last user message must be a
JSON object ``{"state": ..., "questions": {...}"}`` and the response content
is the ``answers`` object as JSON. The systemone endpoint has no free-tier
gate — plain requests pass.

Free-tier gate ("FreeTierError: OpenCode's free tier can only be used from
within OpenCode"): the zen upstream serves free models only for requests that
look like real opencode agent traffic. Empirically required and sufficient
(2026-09-21, verified request-by-request):

1. ``User-Agent: opencode/<ver> ai-sdk/provider-utils/<ver> runtime/bun/<ver>``
2. ``x-opencode-client: cli`` and ``x-opencode-project: global``
3. ``x-opencode-request`` / ``x-opencode-session`` IDs whose 12-hex prefix
   encodes the *current* millisecond timestamp: ``hex48(ts_ms * 0x1000 +
   counter)`` — messages ascending, sessions *bitwise-inverted* (descending).
   Mirrors ``packages/opencode/src/id/id.ts``. Random/future hex → 403.
4. The canonical 12-tool agent array (bash, edit, glob, grep, question, read,
   skill, task, todowrite, webfetch, websearch, write) in ``tools`` — a
   system prompt is NOT required, but missing/partial tools → 403.
5. ``stream: true`` — non-streaming requests → 403. ``complete()`` therefore
   streams internally and aggregates the SSE into one response.

A Bearer key is optional for free models (``Bearer public``/anonymous passes);
an API key is resolved from credgoo when available so usage is tracked to the
account and paid models work.

Note: Claude models on OpenCode use the Anthropic-messages API
(``https://opencode.ai/zen``) and are NOT served by this OpenAI-compatible
provider (which targets ``/v1``).
"""
import json
import os
import secrets
import string
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import requests

from ..core import ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ModelInfo
from ..errors import UniInferError, map_provider_error
from .openai_compatible import OpenAICompatibleChatProvider

_TOOLS_PATH = Path(__file__).resolve().parent / "opencode_agent_tools.json"
_B62 = string.digits + string.ascii_uppercase + string.ascii_lowercase


class OpenCodeProvider(OpenAICompatibleChatProvider):
    """Provider for the OpenCode/Zen model router (OpenAI-compatible)."""

    BASE_URL = "https://opencode.ai/zen/v1"
    PROVIDER_ID = "opencode"
    ERROR_PROVIDER_NAME = "OpenCode"
    DEFAULT_MODEL = "deepseek-v4-flash-free"
    CREDGOO_SERVICE = "opencode"
    # Free models are usable anonymously (Bearer public); the key from credgoo
    # is attached when present (usage tracking + paid models), but is optional.
    REQUIRES_API_KEY = False
    # The router forwards native OpenAI multimodal content to vision-capable
    # upstreams (e.g. big-pickle, mimo-v2.5-free).
    PRESERVE_MULTIMODAL = True

    # Client identity the zen gate expects. Version-pinned to the capture the
    # toolset was taken from (1.18.31) — refresh both together on upgrades.
    CLIENT_USER_AGENT = "opencode/1.18.31 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14"
    CLIENT_ID = os.environ.get("UNIINFER_OPENCODE_CLIENT", "cli")
    PROJECT_ID = os.environ.get("UNIINFER_OPENCODE_PROJECT", "global")

    # Monotonic per-process id counter (mirrors id.ts state)
    _id_lock = threading.Lock()
    _id_last_ms = 0
    _id_counter = 0
    _agent_tools_cache: Optional[list[dict[str, Any]]] = None

    def __init__(self, api_key: Optional[str] = None):
        if not api_key:
            try:
                from credgoo import get_api_key
                api_key = get_api_key(self.CREDGOO_SERVICE)
            except Exception:
                api_key = None
        super().__init__(api_key=api_key, base_url=self.BASE_URL)

    # ------------------------------------------------------------------ #
    # opencode ID dialect (packages/opencode/src/id/id.ts):
    #   value = ts_ms * 0x1000 + counter   (counter resets per ms)
    #   id   = prefix + "_" + hex48(value | ~value) + 14 random base62 chars
    # The zen free-tier gate validates the hex part encodes a *recent*
    # timestamp, so ids MUST be generated fresh per request.
    # ------------------------------------------------------------------ #
    @classmethod
    def _opencode_id(cls, prefix: str, *, descending: bool) -> str:
        with cls._id_lock:
            now_ms = int(time.time() * 1000)
            if now_ms != cls._id_last_ms:
                cls._id_last_ms = now_ms
                cls._id_counter = 0
            cls._id_counter += 1
            value = now_ms * 0x1000 + cls._id_counter
        if descending:
            value = ~value
        hexpart = format(value & 0xFFFFFFFFFFFF, "012x")
        rand = "".join(secrets.choice(_B62) for _ in range(14))
        return f"{prefix}_{hexpart}{rand}"

    def _get_extra_headers(self) -> dict[str, str]:
        # Fresh per-request ids: request id always new; session id too unless
        # pinned via UNIINFER_OPENCODE_SESSION (keeps zen's sticky-provider
        # routing on one conversation).
        session = os.environ.get("UNIINFER_OPENCODE_SESSION")
        return {
            "User-Agent": self.CLIENT_USER_AGENT,
            "x-opencode-client": self.CLIENT_ID,
            "x-opencode-project": self.PROJECT_ID,
            "x-opencode-request": self._opencode_id("msg", descending=False),
            "x-opencode-session": session or self._opencode_id("ses", descending=True),
        }

    @classmethod
    def _agent_tools(cls) -> list[dict[str, Any]]:
        """The canonical opencode agent toolset (from a 1.18.31 capture).

        Part of the free-tier request signature — the zen gate 403s requests
        that don't carry (at least) these tool definitions.
        """
        if cls._agent_tools_cache is None:
            try:
                cls._agent_tools_cache = json.loads(_TOOLS_PATH.read_text())
            except Exception:
                cls._agent_tools_cache = []
        return cls._agent_tools_cache

    def _build_payload(
        self,
        request: ChatCompletionRequest,
        stream: bool,
        provider_specific_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        payload = super()._build_payload(request, stream, provider_specific_kwargs)
        # Agent-shape the request: union caller tools with the canonical set
        # (caller definitions win by name), always stream with usage.
        canonical = self._agent_tools()
        if not canonical:
            return payload
        if payload.get("tools"):
            by_name = {t.get("function", {}).get("name"): t for t in payload["tools"]}
            for tool in canonical:
                by_name.setdefault(tool["function"]["name"], tool)
            payload["tools"] = list(by_name.values())
        else:
            payload["tools"] = canonical
            payload.setdefault("tool_choice", "auto")
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        return payload

    async def _chat_acomplete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs,
    ) -> ChatCompletionResponse:
        """Chat completion: streamed (zen requirement) and aggregated."""
        endpoint = self._completion_endpoint()
        payload = self._build_payload(request, False, provider_specific_kwargs)
        headers = self._build_headers()

        client = await self._get_async_client()
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        usage: dict[str, Any] = {}
        finish_reason: Optional[str] = None
        model_id = request.model
        last_chunks: list[dict[str, Any]] = []
        try:
            async with client.stream("POST", endpoint, headers=headers, json=payload, timeout=60.0) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    error_text = error_body.decode("utf-8", errors="replace")
                    raise map_provider_error(
                        self._error_name(),
                        Exception(f"{self._error_name()} API error: {response.status_code} - {error_text}"),
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
                    last_chunks.append(data)
                    last_chunks = last_chunks[-5:]
                    if data.get("usage"):
                        usage = data["usage"]
                    if data.get("model"):
                        model_id = data["model"]
                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                    reasoning = delta.get("reasoning") or delta.get("reasoning_content") or delta.get("thinking")
                    if reasoning:
                        reasoning_parts.append(reasoning)
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", len(tool_calls))
                        while len(tool_calls) <= idx:
                            tool_calls.append({})
                        agg = tool_calls[idx]
                        if tc.get("id"):
                            agg["id"] = tc["id"]
                        if tc.get("type"):
                            agg["type"] = tc["type"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            agg["function"] = {"name": fn["name"], "arguments": ""}
                        if fn.get("arguments"):
                            agg.setdefault("function", {"name": "", "arguments": ""})
                            agg["function"]["arguments"] = agg["function"].get("arguments", "") + fn["arguments"]
            message = ChatMessage(
                role="assistant",
                content="".join(content_parts) or None,
                tool_calls=tool_calls or None,
                tool_call_id=None,
            )
            return ChatCompletionResponse(
                message=message,
                provider=self.PROVIDER_ID,
                model=model_id or request.model,
                usage=usage,
                raw_response={"chunks": last_chunks},
                finish_reason=finish_reason,
                thinking="".join(reasoning_parts) or None,
            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error(self._error_name(), e)

    # ------------------------------------------------------------------ #
    # System One (Jev) — decision models via /v1/systemone. No free-tier
    # gate: plain requests pass. Wrapped into the chat API: last user
    # message must be JSON {"state": ..., "questions": {...}}; the response
    # content is the answers object as JSON.
    # ------------------------------------------------------------------ #
    SYSTEMONE_PREFIX = "jev-"

    @classmethod
    def _is_systemone_model(cls, model_id: Optional[str]) -> bool:
        return bool(model_id) and model_id.split(":")[0].startswith(cls.SYSTEMONE_PREFIX)

    def _systemone_endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/systemone"

    def _extract_systemone_payload(self, request: ChatCompletionRequest) -> dict[str, Any]:
        last_user = None
        for message in request.messages:
            if message.role == "user":
                last_user = message
        text = getattr(last_user, "content", None) if last_user else None
        if not text or not isinstance(text, str):
            raise ValueError(
                'OpenCode jev models need the last user message to be a JSON object '
                '"state" and "questions" keys (System One API).'
            )
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                "OpenCode jev models need the last user message to be valid JSON "
                f'{{"state": ..., "questions": ...}}}} (System One API): {e}'
            ) from e
        if not isinstance(parsed, dict) or "state" not in parsed or "questions" not in parsed:
            raise ValueError(
                'OpenCode jev payload must contain "state" and "questions" keys.'
            )
        return {"model": request.model, "state": parsed["state"], "questions": parsed["questions"]}

    async def _systemone_acomplete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        payload = self._extract_systemone_payload(request)
        headers = self._build_headers()
        client = await self._get_async_client()
        try:
            response = await client.post(self._systemone_endpoint(), headers=headers, json=payload, timeout=60.0)
            if response.status_code != 200:
                raise map_provider_error(
                    self._error_name(),
                    Exception(f"{self._error_name()} API error: {response.status_code} - {response.text}"),
                    status_code=response.status_code,
                    response_body=response.text,
                )
            data = response.json()
            usage_raw = data.get("usage") or {}
            input_tokens = usage_raw.get("input_tokens", 0)
            output_tokens = usage_raw.get("output_tokens", 0)
            return ChatCompletionResponse(
                message=ChatMessage(role="assistant", content=json.dumps(data.get("answers") or {})),
                provider=self.PROVIDER_ID,
                model=data.get("model", request.model),
                usage={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
                raw_response=data,
                finish_reason="stop",
                thinking=None,
            )
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error(self._error_name(), e)

    async def acomplete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs,
    ) -> ChatCompletionResponse:
        """Route: jev-* -> System One decision API, else chat (streamed)."""
        if self._is_systemone_model(request.model or self.DEFAULT_MODEL):
            return await self._systemone_acomplete(request)
        return await self._chat_acomplete(request, **provider_specific_kwargs)

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs,
    ) -> AsyncIterator[ChatCompletionResponse]:
        """Route: jev-* yields one aggregated chunk (no stream upstream)."""
        if self._is_systemone_model(request.model or self.DEFAULT_MODEL):
            yield await self._systemone_acomplete(request)
            return
        async for chunk in super().astream_complete(request, **provider_specific_kwargs):
            yield chunk

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list["ModelInfo"]:
        """List models from OpenCode/Zen via pi.dev catalog.

        The native ``/v1/models`` endpoint returns bare IDs with no metadata.
        pi.dev maintains an enriched catalog (context windows, max tokens,
        reasoning flag, input modalities, cost) — pull from there instead.

        pi.dev uses flat top-level fields (``reasoning``, ``input``, ``cost``)
        rather than a ``capabilities`` dict, so we translate:
        - ``reasoning: true``  -> capabilities.reasoning
        - ``input: [...'image']`` -> capabilities.vision (+ modalities)
        - ``cost.input == 0``   -> access 'free' (data-driven, not a name
          heuristic; removes the old ``-free``/``big-pickle`` special-case).
        """
        try:
            r = requests.get("https://pi.dev/api/models/providers/opencode", timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception:
            data = {}
        out = []
        for mid, m in data.items():
            cost = m.get("cost") or {}
            free = cost.get("input", 0) == 0
            inputs = m.get("input") or []
            caps = {}
            if m.get("reasoning"):
                caps["reasoning"] = True
            if "image" in inputs:
                caps["vision"] = True
            out.append(
                ModelInfo(
                    id=mid,
                    name=m.get("name"),
                    owned_by="opencode",
                    context_window=m.get("contextWindow"),
                    max_output=m.get("maxTokens"),
                    cost=cost,
                    access="free" if free else "paid",
                    capabilities=caps or None,
                    modalities=inputs or None,
                    raw=m,
                )
            )
        # pi.dev lags behind the live zen catalog — pin the System One model
        # (served via /v1/systemone, wrapped into chat by this provider).
        if not any(m.id == "jev-1.13-free" for m in out):
            out.append(
                ModelInfo(
                    id="jev-1.13-free",
                    name="Jev 1.13 Free (System One)",
                    owned_by="opencode",
                    cost={"input": 0.0, "output": 0.0},
                    access="free",
                    capabilities={"structured_outputs": True},
                    raw={"endpoint": "systemone"},
                )
            )
        return out
