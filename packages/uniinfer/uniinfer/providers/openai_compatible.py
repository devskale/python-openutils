import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import httpx
import requests

from ..core import REASONING_OFF, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ChatProvider, ModelInfo
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


def normalize_tool_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise the transcript shapes strict OpenAI-compatible gateways refuse.

    A ``role="tool"`` result must be introduced by exactly ONE assistant
    message holding the matching ``tool_calls``; gateways that enforce this
    answer otherwise with a bare ``400 invalid_request_error`` (opencode zen)
    or a silent empty completion (kilo) — neither names the offending message.
    Two real shapes violate it, both produced by agents (pi) and by mid-session
    provider switches:

    * an assistant run split by its own results — one assistant message per
      tool call: ``assistant(c1) tool(c1) assistant(c2) tool(c2)``
    * a result whose introducing assistant turn has no calls at all (the old
      provider's turns were flattened to text), or no turn above at all

    The rewrite is minimal and lossless: split runs are folded into a single
    assistant turn with every call, and results are re-emitted directly behind
    it — the shape every OpenAI-compatible backend accepts. Results with no
    parent call anywhere in the transcript are presented as user turns instead
    of being dropped. Transcripts the gateway already accepts pass through
    byte-identical; enable per provider via ``STRICT_TOOL_HISTORY``.
    """
    # Pass 1 — fold split assistant runs: assistant/tool/assistant(tool_calls)
    # becomes one assistant message holding all of the run's calls.
    merged: list[dict[str, Any]] = []
    pending_tools: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "tool":
            pending_tools.append(msg)
            continue
        is_next_split_leg = (
            msg.get("role") == "assistant"
            and msg.get("tool_calls")
            and bool(merged)
            and merged[-1].get("role") == "assistant"
            and merged[-1].get("tool_calls")
        )
        if not is_next_split_leg:
            # Close the current assistant turn: its results come before
            # whatever follows (a later user turn must not overtake them).
            merged.extend(pending_tools)
            pending_tools = []
            merged.append(msg)
        else:
            # Split leg: fold the calls into the assistant turn above; the
            # parked results stay parked (their run is not closed yet).
            base = merged[-1]
            base["tool_calls"] = list(base["tool_calls"]) + list(msg["tool_calls"])
            if msg.get("content") and not base.get("content"):
                base["content"] = msg["content"]

    # Pass 2 — a tool run survives only directly behind its assistant turn
    # (with calls). Everything else becomes a user turn.
    out: list[dict[str, Any]] = []
    index = 0
    total = len(merged)
    while index < total:
        msg = merged[index]
        if msg.get("role") == "tool":
            run = []
            while index < total and merged[index].get("role") == "tool":
                run.append(merged[index])
                index += 1
            parent = out[-1] if out else None
            if parent is not None and parent.get("role") == "assistant" and parent.get("tool_calls"):
                known = {c.get("id") for c in parent["tool_calls"]}
                claimed = [t for t in run if (t.get("tool_call_id") or t.get("id")) in known]
                orphans = [t for t in run if t not in claimed]
                out.extend(claimed)  # results belong here — the gateway takes this shape
                # Partial match: demote only what the turn above did not call.
                out.extend(
                    {"role": "user",
                     "content": "<tool_result>{}</tool_result>".format(t.get("content", ""))}
                    for t in orphans
                )
                continue
            out.extend(
                {"role": "user",
                 "content": "<tool_result>{}</tool_result>".format(t.get("content", ""))}
                for t in run
            )
            continue
        out.append(msg)
        index += 1
    return out


def normalize_tool_call_ids(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rewrite tool-call ids to a short gateway-neutral format, consistently.

    Ids only need to be internally consistent within one request (assistant
    ``tool_calls[].id`` <-> ``tool.tool_call_id``), but gateways validate what
    they accept: a transcript carrying ids another backend minted (e.g. TU
    vLLM's ``chatcmpl-tool-<hex>`` after a mid-session model switch to kilo)
    gets a silent empty completion instead of an error. Renaming every id to
    ``call_<n>`` — the same mapping applied to both sides of each pair — is
    semantically transparent and sidesteps whatever id grammar a gateway
    enforces. Unmatched ``tool_call_id`` entries (no assistant call above) are left
    alone; renaming those would fabricate a pairing. Deterministic: the same
    transcript always yields the same ids.
    """
    mapping: dict[str, str] = {}
    counter = 0
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if not isinstance(tc, dict):
                    continue
                old = tc.get("id")
                if not old:
                    continue
                if old not in mapping:
                    counter += 1
                    mapping[old] = "call_%08x" % counter
                tc["id"] = mapping[old]
        elif msg.get("role") == "tool":
            old = msg.get("tool_call_id")
            if old and old in mapping:
                msg["tool_call_id"] = mapping[old]
    return messages


logger = logging.getLogger(__name__)


def _is_empty_completion(finish_reason, content, thinking, tool_calls) -> bool:
    """True for a completion that says stop but carries nothing: no content,
    no reasoning, no tool calls. Never a legitimate chat answer — the
    silent-empty failure mode of flaky gateways (kilo LB roulette)."""
    if finish_reason not in (None, "stop"):
        return False
    if tool_calls:
        return False
    if isinstance(content, str):
        if content.strip():
            return False
    elif content:
        return False
    if thinking and str(thinking).strip():
        return False
    return True


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
        # Keyless-instance override (z.B. public vLLM endpoints via the
        # instances overlay): drop the class-level key requirement.
        if kwargs.get("REQUIRES_API_KEY") is not None:
            self.REQUIRES_API_KEY = bool(kwargs["REQUIRES_API_KEY"])

    # ------------------------------------------------------------------ #
    # model listing — a template method. The mechanics (credgoo key,
    # GET {base_url}/models, JSON parse, error-map) live here; each subclass
    # declares its dialect by overriding _model_info(raw). Override
    # _models_url / _resolve_api_key only when a provider deviates.
    # ------------------------------------------------------------------ #
    @classmethod
    def _resolve_api_key(cls, api_key: Optional[str]) -> Optional[str]:
        if api_key:
            return api_key
        service = getattr(cls, "CREDGOO_SERVICE", None)
        if not service:
            return None
        try:
            from credgoo import get_api_key
            return get_api_key(service)
        except Exception:
            return None

    @classmethod
    def _models_url(cls, base_url: str) -> str:
        return f"{base_url.rstrip('/')}/models"

    @classmethod
    def _extra_request_headers(cls) -> dict:
        """Extra headers for the /models GET (e.g. app attribution). Default none."""
        return {}

    @classmethod
    def _fetch_entries(cls, api_key: Optional[str], url: str) -> list[dict]:
        headers = {**cls._extra_request_headers()}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code != 200:
            raise map_provider_error(
                cls.ERROR_PROVIDER_NAME or cls.PROVIDER_ID,
                Exception(f"{cls.ERROR_PROVIDER_NAME} API error: {r.status_code} - {r.text}"),
                status_code=r.status_code, response_body=r.text,
            )
        data = r.json()
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        return data if isinstance(data, list) else []

    @classmethod
    def _model_info(cls, raw: dict) -> ModelInfo:
        """Dialect hook: turn one raw entry into a ModelInfo. Default is bare
        (id + owned_by); subclasses override to add access/cost/capabilities
        from their raw fields."""
        return ModelInfo(id=raw.get("id") or raw.get("name"), owned_by=raw.get("owned_by"), raw=raw)

    @classmethod
    def list_models(cls, api_key: Optional[str] = None, base_url: Optional[str] = None) -> list[ModelInfo]:
        key = cls._resolve_api_key(api_key)
        url = cls._models_url(base_url or cls.BASE_URL)
        entries = cls._fetch_entries(key, url)
        return [cls._model_info(e) for e in entries if isinstance(e, dict)]

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

    # Backends whose tool-call grammar folding cannot express these schema
    # keywords. Strict/grammar providers (e.g. Kilo -> ModelRun) reject the
    # whole request with "unsupported schema keyword" for each one, so the
    # keywords have to be dropped before the request leaves the proxy.
    # Set to True only for those providers; the default (False) keeps the
    # keywords, which are semantically useful where they are accepted.
    STRICT_GRAMMAR_SCHEMAS = False

    # Rewrite agent transcripts that strict gateways refuse (split assistant
    # tool-call runs, results without a calling turn) before the request
    # leaves. Opt-in: permissive backends must see the history untouched.
    # See normalize_tool_history() for the exact shapes and the rewrite rules.
    STRICT_TOOL_HISTORY: bool = False

    # Rewrite tool-call ids (assistant calls + matching results) to a short
    # gateway-neutral format. For gateways that choke on ids another backend
    # minted — kilo answers a transcript carrying TU vLLM's chatcmpl-tool-*
    # ids with a silent empty completion. Off by default: gateways that
    # tolerate any id grammar must keep seeing it untouched.
    NORMALIZE_TOOL_CALL_IDS: bool = False

    # Retry a completion that came back completely empty (finish=stop, no
    # content, no reasoning, no tool calls). Kilo's load balancer sometimes
    # routes to replicas that answer exactly that — the same transcript
    # replayed lands on a healthy replica. 0 = relay the empty answer as-is.
    EMPTY_COMPLETION_RETRIES: int = 0

    # Schema keywords dropped entirely when STRICT_GRAMMAR_SCHEMAS is on.
    # `pattern` is the common offender: pi's tool schemas carry it (e.g.
    # herdr_agent's `name`), and grammar folding has no way to validate a
    # regex while decoding, so it refuses the request outright.
    STRICT_GRAMMAR_DROP_KEYS = ("pattern",)

    def _sanitize_tools_schema(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Neutralize parameter-schema constructs that strict grammar-based tool
        backends (e.g. qwen via ModelRun) reject with
        ``more than one JSON reading of the same emitted value``.

        The classic offender is a union that mixes ``object`` with another type
        (``oneOf``/``anyOf`` or a multi-type array) — the exact shape MCP-style
        ``args`` params use (object-or-string). Grammar folding cannot decide
        which alternative a token belongs to, so the whole request 400s.

        Collapsing such a node to an open ``{"type": "object"}`` is
        deterministic (always foldable) and matches the real semantics of an
        object-bag parameter. Non-object unions are left untouched; plain
        object branches are left as-is. A deep copy is produced so the incoming
        request is never mutated.

        When :attr:`STRICT_GRAMMAR_SCHEMAS` is set, the validation-only keywords
        in :attr:`STRICT_GRAMMAR_DROP_KEYS` (e.g. ``pattern``) are additionally
        removed at every depth, because that backend 400s on them with
        ``unsupported schema keyword``. They carry no type information, so
        dropping them only weakens client-side validation, not the grammar.
        """
        import copy

        out = copy.deepcopy(tools)

        def _clean(node: Any) -> Any:
            if not isinstance(node, dict):
                return node
            desc = node.get("description")
            t = node.get("type")
            # Strict-grammar backends reject these keywords outright; drop them
            # before the union handling below, so the stripped node is what gets
            # collapsed/inspected.
            if self.STRICT_GRAMMAR_SCHEMAS:
                for key in self.STRICT_GRAMMAR_DROP_KEYS:
                    node.pop(key, None)
            # Multi-type array containing object -> open object.
            if isinstance(t, list):
                if "object" in t:
                    node = {"type": "object"}
                else:
                    node["type"] = t[0] if t else "string"
            # oneOf/anyOf union with an object alternative -> open object.
            elif not isinstance(t, list):
                for key in ("oneOf", "anyOf"):
                    alts = node.get(key)
                    if isinstance(alts, list) and any(
                        isinstance(a, dict) and a.get("type") == "object" for a in alts
                    ):
                        node = {"type": "object"}
                        break
            if desc:
                node.setdefault("description", desc)
            # Recurse into nested schema containers.
            for k in ("properties", "additionalProperties", "items", "prefixItems"):
                v = node.get(k)
                if isinstance(v, dict) and k == "properties":
                    node[k] = {kk: _clean(vv) for kk, vv in v.items()}
                elif isinstance(v, dict):
                    node[k] = _clean(v)
            return node

        for spec in out:
            fn = spec.get("function") or {}
            if fn.get("parameters"):
                fn["parameters"] = _clean(fn["parameters"])
        return out

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

        messages = self._flatten_messages(request.messages)
        if self.STRICT_TOOL_HISTORY:
            messages = normalize_tool_history(messages)
        if self.NORMALIZE_TOOL_CALL_IDS:
            messages = normalize_tool_call_ids(messages)
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": defaults.get("temperature", request.temperature),
            "stream": stream,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
            # Weicher, konfigurierbarer Cap (model_defaults.json) — z.B. fuer
            # Free-Tier-Modelle, deren Tier-Limit weit unter dem API-Hardcap
            # liegt (groq qwen3.8-27b: API 16384, Tier ~1000).
            soft_cap = defaults.get("max_tokens_cap")
            if isinstance(soft_cap, int) and soft_cap > 0:
                payload["max_tokens"] = min(payload["max_tokens"], soft_cap)
        if request.tools:
            payload["tools"] = self._sanitize_tools_schema(request.tools)
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice

        if request.reasoning_effort is not None:
            payload.update(self._reasoning_payload(request.reasoning_effort))
        if getattr(request, "chat_template_kwargs", None):
            # qwen-3.x nothink + other chat-template params (e.g. enable_thinking=False)
            payload["chat_template_kwargs"] = request.chat_template_kwargs

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

    def _retry_client(self) -> httpx.AsyncClient:
        """A throwaway client on a fresh connection for an empty-completion
        retry: new TCP/TLS handshake, new load-balancer routing, own lifecycle
        (closed by the caller). Overrides stay testable."""
        return httpx.AsyncClient(timeout=60.0)

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

        retries = max(0, int(getattr(self, "EMPTY_COMPLETION_RETRIES", 0) or 0))
        for attempt in range(retries + 1):
            if attempt == 0:
                client = await self._get_async_client()
                response = await client.post(endpoint, headers=headers, json=payload, timeout=60.0)
            else:
                # Fresh connection: new TCP/TLS, new load-balancer routing —
                # the retry exists because the route, not the request, was bad.
                fresh = self._retry_client()
                try:
                    response = await fresh.post(endpoint, headers=headers, json=payload, timeout=60.0)
                finally:
                    await fresh.aclose()
            try:
                if response.status_code != 200:
                    error_msg = f"{self._error_name()} API error: {response.status_code} - {response.text}"
                    raise map_provider_error(
                        self._error_name(),
                        Exception(error_msg),
                        status_code=response.status_code,
                        response_body=response.text,
                    )

                response_data = response.json()
                choice = (response_data.get("choices") or [{}])[0]
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

                if attempt < retries and _is_empty_completion(
                    choice.get("finish_reason"), message.content, reasoning_content, message.tool_calls
                ):
                    logger.warning(
                        "[%s] empty completion (finish=stop, no content) — retrying on a fresh connection (attempt %d/%d)",
                        self._error_name(), attempt + 1, retries + 1,
                    )
                    continue

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
        raise map_provider_error(self._error_name(), Exception("unreachable: retry loop exhausted"))

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

        retries = max(0, int(getattr(self, "EMPTY_COMPLETION_RETRIES", 0) or 0))
        attempt = 0
        while True:
            # A stream that ends with finish=stop and never produced content,
            # reasoning or tool calls is the silent-empty failure of flaky
            # gateways (kilo LB roulette). Nothing content-wise reached the
            # caller, so replaying the attempt on a fresh connection (new
            # TCP/TLS, new load-balancer routing) is duplication-free.
            # Finish/usage chunks are held until the retry decision — they
            # would otherwise leak the dead attempt's terminal state.
            saw_visible = False
            held: list[ChatCompletionResponse] = []
            if attempt == 0:
                client = await self._get_async_client()
            else:
                client = self._retry_client()
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
                            # Terminal usage-only chunk (choices:[]). vLLM emits
                            # this when stream_options.include_usage is set.
                            # Held with the finish chunk: a retry discards it as
                            # stale usage of a dead attempt.
                            if data.get("usage"):
                                held.append(ChatCompletionResponse(
                                    message=ChatMessage(role="assistant", content=None),
                                    provider=self.PROVIDER_ID,
                                    model=data.get("model", request.model),
                                    usage=data["usage"],
                                    raw_response=data,
                                    finish_reason=None,
                                    thinking=None,
                                ))
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

                        if content or reasoning_content or tool_calls:
                            saw_visible = True

                        chunk = ChatCompletionResponse(
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
                        if finish_reason is not None or data.get("usage"):
                            held.append(chunk)
                            continue
                        yield chunk

                if attempt < retries and not saw_visible and held:
                    logger.warning(
                        "[%s] empty stream (finish=stop, no content) — replaying on a fresh connection (attempt %d/%d)",
                        self._error_name(), attempt + 1, retries + 1,
                    )
                    attempt += 1
                    continue
                for chunk in held:
                    yield chunk
                return
            except Exception as e:
                if isinstance(e, UniInferError):
                    raise
                raise map_provider_error(self._error_name(), e)
            finally:
                if attempt > 0 and not client.is_closed:
                    await client.aclose()
