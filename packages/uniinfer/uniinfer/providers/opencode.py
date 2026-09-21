from __future__ import annotations
"""
OpenCode / Zen provider implementation.

OpenCode (opencode.ai) runs the "zen" model router — an OpenAI-compatible
endpoint that aggregates many models (DeepSeek, GPT, Gemini, Qwen, GLM,
MiniMax, Kimi, …). Free models are id-suffixed ``-free`` plus ``big-pickle``.

**Everything model-specific is derived dynamically** — no hardcoded model
lists or prefixes:

- model universe: live ``GET /zen/v1/models`` (always current)
- endpoint dialect + display names + pricing: the official docs tables,
  parsed from the opencode repo's ``zen.mdx`` (raw.githubusercontent.com,
  cached 1h). This maps each model id to its dialect:
  ``chat`` (/chat/completions), ``responses`` (/responses — muse-spark,
  gpt family), ``systemone`` (/systemone — jev decision models),
  ``anthropic``/``google`` (unsupported here → clear error)
- metadata (context window, caps): pi.dev catalog enrichment where present

Free-tier gate ("FreeTierError: can only be used from within OpenCode") —
empirically required and sufficient per dialect (2026-09-21):

- chat: opencode User-Agent + client/project headers + timestamp-encoded
  msg/ses IDs (mirrors ``packages/opencode/src/id/id.ts``) + the canonical
  12-tool agent array + ``stream: true``
- responses: same headers/IDs + the same tools in FLAT responses format +
  ``stream: true``
- systemone: NO gate — plain requests pass

``complete()`` therefore streams internally (chat/responses) and aggregates.
A Bearer key is optional for free models (``Bearer public`` passes); the
credgoo ``opencode`` key is attached when present (usage tracking + paid).

Jev (System One) models are decision models: the last user message must be
JSON ``{"state": ..., "questions": ...}}``; the response content is the
``answers`` object as JSON.
"""
import json
import os
import re
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

# Official docs tables (endpoints + pricing) — the authoritative, versioned
# source for dialect mapping and free/paid detection. Parsed from the repo,
# so model additions upstream flow through without code changes here.
_DOCS_URL = ("https://raw.githubusercontent.com/anomalyco/opencode/dev/"
             "packages/web/src/content/docs/zen.mdx")
_DOCS_TTL = 3600.0
_DOCS_CACHE: Optional[dict[str, Any]] = None
_DOCS_CACHE_TS = 0.0


def _parse_price_cell(cell: str) -> Optional[float]:
    """'Free' -> 0.0, '$0.042' -> 0.042, '-'/'' -> None."""
    cell = cell.strip()
    if not cell or cell == "-":
        return None
    if cell.lower() == "free":
        return 0.0
    try:
        return float(cell.replace("$", "").replace(",", ""))
    except ValueError:
        return None


def _parse_docs_mdx(text: str) -> dict[str, Any]:
    """Parse the Endpoints + Pricing tables from the zen docs MDX.

    Returns {"endpoints": {id: {name, endpoint, sdk}}, "pricing": {id: {input, output}}}.
    Pricing rows carry display names (sometimes with '(≤ 200K tokens)' tier
    qualifiers); they are mapped to ids via the Endpoints table's names. For
    tiered models the first ('≤') row wins.
    """
    endpoints: dict[str, dict[str, str]] = {}
    pricing_by_name: dict[str, dict[str, Optional[float]]] = {}
    mode: Optional[str] = None
    for raw in text.splitlines():
        if not raw.strip().startswith("|"):
            continue
        cells = [c.strip() for c in raw.strip().strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue  # separator row
        header = " ".join(cells).lower()
        if "model id" in header and "endpoint" in header:
            mode = "endpoints"
            continue
        if "input" in header and "output" in header and "cached" in header:
            mode = "pricing"
            continue
        if mode == "endpoints" and len(cells) >= 3:
            name, mid, endpoint = cells[0], cells[1], cells[2].strip("`")
            endpoints[mid] = {"name": name, "endpoint": endpoint, "sdk": cells[3].strip("`") if len(cells) > 3 else ""}
        elif mode == "pricing" and len(cells) >= 3:
            name = re.sub(r"\s*\([^)]*\)\s*$", "", cells[0]).strip()  # drop tier qualifiers
            if name and name not in pricing_by_name:  # first row = base (≤) tier
                pricing_by_name[name] = {
                    "input": _parse_price_cell(cells[1]),
                    "output": _parse_price_cell(cells[2]),
                }
    pricing: dict[str, dict[str, Optional[float]]] = {}
    for mid, entry in endpoints.items():
        price = pricing_by_name.get(entry["name"])
        if price is not None:
            pricing[mid] = price
    return {"endpoints": endpoints, "pricing": pricing}


def _docs_tables() -> Optional[dict[str, Any]]:
    """Docs tables, cached; stale cache beats none on fetch failure."""
    global _DOCS_CACHE, _DOCS_CACHE_TS
    now = time.time()
    if _DOCS_CACHE is not None and now - _DOCS_CACHE_TS < _DOCS_TTL:
        return _DOCS_CACHE
    try:
        text = requests.get(_DOCS_URL, timeout=15).text
        tables = _parse_docs_mdx(text)
        if tables["endpoints"] or tables["pricing"]:
            _DOCS_CACHE, _DOCS_CACHE_TS = tables, now
        return _DOCS_CACHE
    except Exception:
        return _DOCS_CACHE


class OpenCodeProvider(OpenAICompatibleChatProvider):
    """Provider for the OpenCode/Zen model router (dynamic catalog)."""

    BASE_URL = "https://opencode.ai/zen/v1"
    PROVIDER_ID = "opencode"
    ERROR_PROVIDER_NAME = "OpenCode"
    DEFAULT_MODEL = "mimo-v2.5-free"
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
    # Dialect routing — derived from the docs endpoints table, not prefixes
    # ------------------------------------------------------------------ #
    @classmethod
    def _dialect_for(cls, model_id: Optional[str]) -> str:
        """'chat' | 'responses' | 'systemone' | 'anthropic' | 'google'."""
        tables = _docs_tables() or {}
        entry = (tables.get("endpoints") or {}).get(model_id or "", {})
        endpoint = entry.get("endpoint", "")
        if endpoint.endswith("/systemone"):
            return "systemone"
        if endpoint.endswith("/responses"):
            return "responses"
        if endpoint.endswith("/chat/completions"):
            return "chat"
        if "/messages" in endpoint:
            return "anthropic"
        if "/models/" in endpoint:
            return "google"
        return "chat"  # unknown/docs unavailable → the common case

    def _unsupported_dialect_error(self, dialect: str, model_id: str):
        return map_provider_error(
            self._error_name(),
            ValueError(
                f"{self._error_name()} model '{model_id}' speaks the '{dialect}' API, "
                "which this OpenAI-compatible provider does not serve "
                "(Anthropic-messages / Google-generateContent models)."
            ),
        )

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

    @classmethod
    def _agent_tools_flat(cls) -> list[dict[str, Any]]:
        """Canonical toolset in the flat Responses-API tool format."""
        out = []
        for tool in cls._agent_tools():
            fn = tool.get("function", {})
            flat = {
                "type": "function",
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            }
            if tool.get("strict") is not None:
                flat["strict"] = tool["strict"]
            out.append(flat)
        return out

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

    # ------------------------------------------------------------------ #
    # chat dialect (/chat/completions) — streamed, aggregated
    # ------------------------------------------------------------------ #
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
    # systemone dialect (/systemone — jev decision models). No free-tier
    # gate. Wrapped into chat: last user message = JSON {"state","questions"};
    # response content = the answers object as JSON.
    # ------------------------------------------------------------------ #
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
                'OpenCode systemone (jev) models need the last user message to be a JSON object '
                '"state" and "questions" keys (System One API).'
            )
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                "OpenCode systemone (jev) models need the last user message to be valid JSON "
                f'{{"state": ..., "questions": ...}}}} (System One API): {e}'
            ) from e
        if not isinstance(parsed, dict) or "state" not in parsed or "questions" not in parsed:
            raise ValueError(
                'OpenCode systemone (jev) payload must contain "state" and "questions" keys.'
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

    # ------------------------------------------------------------------ #
    # responses dialect (/responses — muse-spark, gpt family). Free-tier
    # gate like chat, but tools in the FLAT format and Responses SSE events.
    # ------------------------------------------------------------------ #
    def _responses_endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/responses"

    def _build_responses_payload(self, request: ChatCompletionRequest) -> dict[str, Any]:
        input_items: list[dict[str, Any]] = []
        for message in request.messages:
            role = message.role

            # The Responses API represents tool results as a flat
            # function_call_output item — NOT as a role="tool" message.
            # role="tool" is rejected upstream with
            # "input[N] did not match any supported type".
            if role == "tool":
                output = message.content
                if isinstance(output, (dict, list)):
                    try:
                        output = json.dumps(output, ensure_ascii=False)
                    except Exception:
                        output = str(output)
                input_items.append({
                    "type": "function_call_output",
                    "call_id": message.tool_call_id or "",
                    "output": output or "",
                })
                continue

            # System messages are expressed as "developer" role in the responses API.
            if role == "system":
                role = "developer"

            part_type = "output_text" if role == "assistant" else "input_text"
            content = message.content
            parts: list[dict[str, Any]] = []
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type")
                    if ptype in ("text", "input_text", "output_text"):
                        parts.append({"type": part_type, "text": part.get("text", "")})
                    elif ptype == "image_url":
                        url = (part.get("image_url") or {}).get("url", "")
                        if url:
                            parts.append({"type": "input_image", "image_url": url})
            elif content:
                parts.append({"type": part_type, "text": content})

            if parts:
                input_items.append({"role": role, "content": parts})

            # Assistant tool calls become function_call items after the message.
            if role == "assistant" and message.tool_calls:
                for tc in message.tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    input_items.append({
                        "type": "function_call",
                        "call_id": tc.get("id") or "",
                        "name": (fn.get("name") if isinstance(fn, dict) else None) or tc.get("name", ""),
                        "arguments": (fn.get("arguments") if isinstance(fn, dict) else "") or tc.get("arguments", ""),
                    })

        tools_flat = list(self._agent_tools_flat())
        if request.tools:
            by_name = {t.get("name"): t for t in tools_flat}
            for tool in request.tools:
                fn = tool.get("function", {})
                name = fn.get("name")
                if not name:
                    continue
                by_name[name] = {
                    "type": "function",
                    "name": name,
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                }
            tools_flat = list(by_name.values())

        payload: dict[str, Any] = {
            "model": request.model,
            "input": input_items,
            "tools": tools_flat,
            "stream": True,
            "store": False,
        }
        if tools_flat:
            payload["tool_choice"] = request.tool_choice or "auto"
        # GPT-family responses models burn a chunk of the output budget on
        # mandatory opaque reasoning before any text (effort "none" → 400,
        # like Kilo). max_output_tokens is a cap, not a target — floor it so
        # small caller limits don't starve the visible text entirely.
        if request.max_tokens is not None:
            payload["max_output_tokens"] = max(request.max_tokens, 1024)
        else:
            payload["max_output_tokens"] = 32000
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        from ..core import REASONING_OFF
        effort = request.reasoning_effort or "low"
        if effort in REASONING_OFF:
            effort = "none"
        payload["reasoning"] = {"effort": effort}
        return payload

    async def _responses_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[tuple[Optional[str], Optional[dict[str, Any]], Optional[dict[str, Any]], Optional[str]]]:
        """Yield (content_delta, tool_call_item, usage, finish) tuples from SSE."""
        payload = self._build_responses_payload(request)
        headers = self._build_headers()
        client = await self._get_async_client()
        saw_delta = False
        async with client.stream("POST", self._responses_endpoint(), headers=headers, json=payload, timeout=60.0) as response:
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
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                etype = data.get("type", "")
                if etype == "response.output_text.delta":
                    saw_delta = True
                    yield data.get("delta"), None, None, None
                elif etype == "response.output_item.done":
                    item = data.get("item") or {}
                    if item.get("type") == "function_call":
                        yield None, item, None, None
                elif etype in ("response.completed", "response.incomplete"):
                    resp = data.get("response") or {}
                    usage = resp.get("usage") or {}
                    if not saw_delta:
                        # Buffered backend: the full output arrives only on the
                        # completed event — replay it as content/tool events.
                        for out_item in resp.get("output") or []:
                            if out_item.get("type") == "message":
                                for part in out_item.get("content") or []:
                                    if part.get("type") in ("output_text", "text") and part.get("text"):
                                        yield part["text"], None, None, None
                            elif out_item.get("type") == "function_call":
                                yield None, out_item, None, None
                    finish = "stop" if etype == "response.completed" else "length"
                    yield None, None, usage, finish
                elif etype in ("response.failed", "error", "response.error"):
                    err = (data.get("response") or {}).get("error") or data.get("error") or {}
                    raise map_provider_error(
                        self._error_name(),
                        Exception(f"{self._error_name()} responses stream error: {json.dumps(err)[:300]}"),
                    )

    async def _responses_acomplete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        usage: dict[str, Any] = {}
        finish: Optional[str] = None
        try:
            async for content, item, event_usage, event_finish in self._responses_stream(request):
                if content:
                    content_parts.append(content)
                if item:
                    tool_calls.append({
                        "id": item.get("call_id") or item.get("id"),
                        "type": "function",
                        "function": {"name": item.get("name"), "arguments": item.get("arguments", "")},
                    })
                if event_usage:
                    usage = event_usage
                if event_finish:
                    finish = event_finish
        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            raise map_provider_error(self._error_name(), e)
        if tool_calls:
            finish = "tool_calls"
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        return ChatCompletionResponse(
            message=ChatMessage(
                role="assistant",
                content="".join(content_parts) or None,
                tool_calls=tool_calls or None,
                tool_call_id=None,
            ),
            provider=self.PROVIDER_ID,
            model=request.model,
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": usage.get("total_tokens", input_tokens + output_tokens),
            },
            raw_response={},
            finish_reason=finish or "stop",
            thinking=None,
        )

    # ------------------------------------------------------------------ #
    # routing
    # ------------------------------------------------------------------ #
    async def acomplete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs,
    ) -> ChatCompletionResponse:
        dialect = self._dialect_for(request.model or self.DEFAULT_MODEL)
        if dialect == "systemone":
            return await self._systemone_acomplete(request)
        if dialect == "responses":
            return await self._responses_acomplete(request)
        if dialect in ("anthropic", "google"):
            raise self._unsupported_dialect_error(dialect, request.model or "")
        return await self._chat_acomplete(request, **provider_specific_kwargs)

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs,
    ) -> AsyncIterator[ChatCompletionResponse]:
        dialect = self._dialect_for(request.model or self.DEFAULT_MODEL)
        if dialect == "systemone":
            yield await self._systemone_acomplete(request)
            return
        if dialect == "responses":
            try:
                async for content, item, usage, finish in self._responses_stream(request):
                    if content is None and item is None and usage is None and finish is None:
                        continue
                    if item is not None:
                        yield ChatCompletionResponse(
                            message=ChatMessage(
                                role="assistant",
                                content=None,
                                tool_calls=[{
                                    "id": item.get("call_id") or item.get("id"),
                                    "type": "function",
                                    "function": {"name": item.get("name"), "arguments": item.get("arguments", "")},
                                }],
                            ),
                            provider=self.PROVIDER_ID,
                            model=request.model,
                            usage={},
                            raw_response={},
                            finish_reason=None,
                            thinking=None,
                        )
                    elif content is not None:
                        yield ChatCompletionResponse(
                            message=ChatMessage(role="assistant", content=content),
                            provider=self.PROVIDER_ID,
                            model=request.model,
                            usage={},
                            raw_response={},
                            finish_reason=None,
                            thinking=None,
                        )
                    if usage:
                        yield ChatCompletionResponse(
                            message=ChatMessage(role="assistant", content=None),
                            provider=self.PROVIDER_ID,
                            model=request.model,
                            usage={
                                "prompt_tokens": usage.get("input_tokens", 0),
                                "completion_tokens": usage.get("output_tokens", 0),
                                "total_tokens": usage.get("total_tokens", usage.get("input_tokens", 0) + usage.get("output_tokens", 0)),
                            },
                            raw_response={},
                            finish_reason=finish,
                            thinking=None,
                        )
            except Exception as e:
                if isinstance(e, UniInferError):
                    raise
                raise map_provider_error(self._error_name(), e)
            return
        if dialect in ("anthropic", "google"):
            raise self._unsupported_dialect_error(dialect, request.model or "")
        async for chunk in super().astream_complete(request, **provider_specific_kwargs):
            yield chunk

    # ------------------------------------------------------------------ #
    # model listing — fully dynamic
    # ------------------------------------------------------------------ #
    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list["ModelInfo"]:
        """Dynamic catalog, no hardcoded model lists.

        1. model universe: live ``GET {BASE_URL}/models`` (bare ids, always
           current; falls back to the docs endpoints table on failure)
        2. dialect/name/pricing from the parsed official docs tables
           (see :func:`_docs_tables`) — free = pricing Input 'Free' (0.0)
        3. metadata (context window, max tokens, capabilities) enriched from
           the pi.dev catalog where the id is known there
        """
        live_ids: list[str] = []
        try:
            r = requests.get(f"{cls.BASE_URL.rstrip('/')}/models", timeout=30,
                             headers={"Authorization": "Bearer public"})
            r.raise_for_status()
            live_ids = [m.get("id") for m in r.json().get("data", []) if m.get("id")]
        except Exception:
            live_ids = []

        tables = _docs_tables() or {}
        endpoints = tables.get("endpoints") or {}
        pricing = tables.get("pricing") or {}
        if not live_ids:
            live_ids = list(endpoints)

        pi_meta: dict[str, dict[str, Any]] = {}
        try:
            r = requests.get("https://pi.dev/api/models/providers/opencode", timeout=30)
            r.raise_for_status()
            pi_meta = r.json()
        except Exception:
            pi_meta = {}

        out: list[ModelInfo] = []
        for mid in live_ids:
            entry = endpoints.get(mid, {})
            price = pricing.get(mid)
            meta = pi_meta.get(mid, {})
            pi_cost = meta.get("cost") or {}
            if price is not None:
                cost = {"input": price.get("input"), "output": price.get("output")}
                free = price.get("input") == 0.0
            elif pi_cost:
                cost = pi_cost
                free = pi_cost.get("input", 1) == 0
            else:
                cost = None
                free = mid.endswith("-free")  # last-resort heuristic, no id pinning
            inputs = meta.get("input") or []
            caps = {}
            if meta.get("reasoning"):
                caps["reasoning"] = True
            if "image" in inputs:
                caps["vision"] = True
            out.append(
                ModelInfo(
                    id=mid,
                    name=entry.get("name") or meta.get("name") or mid,
                    owned_by="opencode",
                    context_window=meta.get("contextWindow"),
                    max_output=meta.get("maxTokens"),
                    cost=cost,
                    access="free" if free else "paid",
                    capabilities=caps or None,
                    modalities=inputs or None,
                    raw={"endpoint": entry.get("endpoint", ""), "sdk": entry.get("sdk", "")},
                )
            )
        return out
