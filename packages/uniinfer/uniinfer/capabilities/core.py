"""Capability test suite — probe what a model can do and exercise each feature.

A reusable matrix runner: given a target ``provider@model`` (+ resolved api key /
base url), it (1) probes the model's declared capability profile, (2) exercises
chat / tool-calling / image, (3) checks thinking on vs off, and (4) optionally
runs perf probes (max speed across varying context sizes, context ceiling,
rate-limit ceiling).

Every probe returns ``pass | fail | skip | error`` + evidence. ``skip`` means the
feature is known not to apply to this backend (e.g. image probe vs an Ollama
provider that flattens messages to text-only) — a correct finding, not a bug.

This module is backend-agnostic and callable from the CLI, the proxy endpoint,
or pytest. It spends no tokens on the cheap ``probe`` step (metadata only).
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from uniinfer.completion import Target

FIXTURES = Path(__file__).parent / "fixtures"
PROBE_RESULTS_PATH = Path(__file__).parent.parent / "models" / "_probe_results.json"
MODELS_JSON_PATH = Path(__file__).parent.parent / "models" / "models.json"

VALID_STATUS = ("pass", "fail", "skip", "error")


# --------------------------------------------------------------------------- #
# Data shapes
# --------------------------------------------------------------------------- #
@dataclass
class ProbeResult:
    """Outcome of a single capability probe."""

    name: str
    status: str
    evidence: str = ""
    latency_ms: Optional[float] = None
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUS:
            raise ValueError(
                f"status must be one of {VALID_STATUS}, got {self.status!r}"
            )

    @property
    def ok(self) -> bool:
        return self.status == "pass"


@dataclass
class ProbeTarget:
    """A model to test and how to reach it."""

    provider_model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    # Thinking models need room — reasoning can consume many tokens before the
    # visible answer. Keep this ≫ 1–2k. Short-answer probes stop early anyway.
    max_tokens: int = 4096
    timeout: float = 90.0
    heavy_perf: bool = False

    def __post_init__(self) -> None:
        if "@" not in self.provider_model:
            raise ValueError(
                f"provider_model must be 'provider@model', got {self.provider_model!r}"
            )
        self.provider_name, self.model_name = self.provider_model.split("@", 1)
        if not self.provider_name or not self.model_name:
            raise ValueError(
                "provider_model: provider and model must both be non-empty"
            )


def _completion_target(t: ProbeTarget) -> Target:
    """Build a non-recording completion Target from a probe target.

    Probes must not inflate model-access metadata, so record_access=False.
    """
    return Target(t.provider_model, t.api_key, t.base_url, record_access=False)


@dataclass
class CapabilityReport:
    """Full matrix for one target."""

    target: str
    profile: dict[str, Any]
    results: list[ProbeResult]

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "profile": self.profile,
            "results": [r.__dict__ for r in self.results],
            "summary": self.summary,
        }

    @property
    def summary(self) -> str:
        n = len(self.results)
        ok = sum(1 for r in self.results if r.status == "pass")
        return f"{ok}/{n} pass"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def load_tools() -> list[dict]:
    """Canonical single-tool definition used by the basic tool-calling probe."""
    return json.loads((FIXTURES / "tools.json").read_text())


def load_multi_tools() -> list[dict]:
    """Multi-tool set used by the tool-selection probe (3 tools)."""
    return json.loads((FIXTURES / "tools_multi.json").read_text())


def load_person_schema() -> dict:
    """JSON schema used by the structured-output probe."""
    return json.loads((FIXTURES / "person_schema.json").read_text())


def image_b64() -> str:
    """The bundled vision target (PNG) as base64 (no data-url prefix)."""
    return base64.b64encode((FIXTURES / "image.png").read_bytes()).decode()


def image_data_url() -> str:
    return f"data:image/png;base64,{image_b64()}"


# --------------------------------------------------------------------------- #
# Per-provider capability hints (providers whose ChatProvider drops an input
# modality before it reaches the backend). Kept explicit so a probe can `skip`
# instead of producing a false `pass`.
# --------------------------------------------------------------------------- #
_PROVIDERS_WITHOUT_IMAGE_FORWARD: set[str] = set()


# --------------------------------------------------------------------------- #
# Probe: capability profile (cheap — metadata only, no tokens)
# --------------------------------------------------------------------------- #
async def probe_profile(t: ProbeTarget) -> ProbeResult:
    """Discover what the model can do from backend metadata."""
    started = time.monotonic()
    try:
        if t.provider_name == "ollama":
            profile = await _ollama_show_profile(t)
        else:
            profile = await _catalog_profile(t)
        caps = profile.get("capabilities", [])
        evidence = "caps=" + (",".join(caps) if caps else "?")
        if profile.get("context_length"):
            evidence += f" ctx={profile['context_length']}"
        return ProbeResult(
            "probe",
            "pass",
            evidence=evidence,
            latency_ms=_ms(started),
            detail={"profile": profile},
        )
    except Exception as e:  # noqa: BLE001
        return ProbeResult("probe", "error", _short_error(e), latency_ms=_ms(started))


async def _ollama_show_profile(t: ProbeTarget) -> dict[str, Any]:
    """Ollama ``/api/show`` → capabilities, context length, families."""
    base = (t.base_url or "").rstrip("/")
    headers = {"Authorization": f"Bearer {t.api_key}"} if t.api_key else {}
    async with httpx.AsyncClient(timeout=t.timeout) as client:
        r = await client.post(
            f"{base}/api/show", headers=headers, json={"name": t.model_name}
        )
        r.raise_for_status()
        data = r.json()
    info = data.get("model_info", {}) or {}
    ctx = next((v for k, v in info.items() if "context_length" in k.lower()), None)
    return {
        "capabilities": data.get("capabilities") or [],
        "context_length": ctx,
        "families": (data.get("details", {}) or {}).get("families"),
        "parameter_size": (data.get("details", {}) or {}).get("parameter_size"),
    }


async def _catalog_profile(t: ProbeTarget) -> dict[str, Any]:
    """Best-effort declared profile from the cached catalog (non-ollama).

    Normalises the catalog's capability dict + modalities into the same
    vocabulary Ollama's ``/api/show`` uses (``completion|tools|thinking|vision``)
    so downstream skip-logic is uniform across providers.
    """
    try:
        from uniinfer.proxy_services.models_registry import load_catalog

        catalog = load_catalog(t.provider_name)  # NOTE: takes a string, not a list
        for m in (
            catalog.get("providers", {}).get(t.provider_name, {}).get("models", [])
        ):
            if m.get("id") == t.model_name:
                caps = m.get("capabilities") or {}
                mods = m.get("modalities") or {}
                inputs = mods.get("input") or []
                # Only assert capabilities when the catalog actually carries
                # capability/modality metadata. Sparse entries (e.g. tu) have
                # none — return [] so probes run empirically instead of
                # false-skipping real capabilities.
                if not (caps or mods):
                    return {
                        "capabilities": [],
                        "context_length": m.get("context_window"),
                        "note": "no capability metadata; probing empirically",
                    }
                declared = ["completion"]
                if caps.get("tool_call") or caps.get("function_calling"):
                    declared.append("tools")
                if caps.get("reasoning"):
                    declared.append("thinking")
                if caps.get("structured_output"):
                    declared.append("structured_output")
                if "image" in inputs:
                    declared.append("vision")
                return {
                    "capabilities": declared,
                    "context_length": m.get("context_window"),
                    "max_output": m.get("max_output"),
                    "modalities": mods,
                }
    except Exception:  # noqa: BLE001
        pass
    return {"capabilities": [], "note": "catalog lookup unavailable"}


# --------------------------------------------------------------------------- #
# Generation helper — unifies thinking control across backends
# --------------------------------------------------------------------------- #
def _model_quirks() -> dict:
    """Per-model quirks index (fixtures/model_quirks.json), keyed 'provider/model'.

    Scoped, data-driven overrides — e.g. models that only accept temperature=1
    (moonshot kimi-code). Add an entry here rather than generalising code paths.
    """
    try:
        return json.loads((FIXTURES / "model_quirks.json").read_text())
    except Exception:  # noqa: BLE001
        return {}


def _quirk(provider_model: str, key: str, default: Any = None) -> Any:
    """Resolve a special param from the quirks index.

    Lookup order: model-specific ('provider/model') → provider-wide
    ('provider') → ``default``. Supports per-model or per-provider overrides
    (e.g. temperature for models/providers that reject the default).
    """
    provider, model = provider_model.split("@", 1)
    idx = _model_quirks()
    entry = idx.get(f"{provider}/{model}") or idx.get(provider) or {}
    return entry.get(key, default)


async def _generate(
    t: ProbeTarget, prompt: str, thinking_on: bool, *, max_tokens: Optional[int] = None
) -> tuple[str, str]:
    """Return (content, thinking). Thinking is toggled via ``reasoning_effort``
    (none/high), which each provider maps to its native knob (ollama ``think``,
    vLLM ``chat_template_kwargs.enable_thinking``). Dispatched through a
    non-recording completion Target.

    ``max_tokens`` overrides ``t.max_tokens``. The thinking probes pass a
    *moderate* cap — capability detection only needs to observe that reasoning
    is produced, not generate a full ≫1–2k chain (which on slow models blows the
    probe timeout). Real callers should still use a generous ``t.max_tokens``.
    """
    mt = max_tokens or t.max_tokens

    effort = "none" if not thinking_on else "high"

    async def call(temp: float):
        return await _completion_target(t).acomplete(
            [{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=mt,
            reasoning_effort=effort,
        )

    resp = await call(_quirk(t.provider_model, "temperature", 0.7))
    return _as_text(getattr(resp.message, "content", "")), _as_text(
        getattr(resp, "thinking", "")
    )


# --------------------------------------------------------------------------- #
# Quiet completion — thinking OFF so reasoning can't eat the token budget
# --------------------------------------------------------------------------- #
async def _complete_quiet(
    t: ProbeTarget,
    messages: list[dict],
    *,
    tools: Optional[list[dict]] = None,
    tool_choice: Optional[Any] = None,
    max_tokens: Optional[int] = None,
) -> Any:
    """Non-streaming completion with reasoning OFF (``reasoning_effort="none"``)
    so reasoning can't eat the token budget. Dispatched through a non-recording
    completion Target; ``tools``/images reach the backend. ``max_tokens``
    overrides ``t.max_tokens`` (used by the perf probe to cap generation and
    stay token-frugal).
    """
    mt = max_tokens or t.max_tokens

    async def call(temp: float):
        return await _completion_target(t).acomplete(
            messages,
            temperature=temp,
            max_tokens=mt,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort="none",
        )

    return await call(_quirk(t.provider_model, "temperature", 0.7))


# --------------------------------------------------------------------------- #
# Probe: chat completion (thinking OFF so reasoning can't eat the token budget)
# --------------------------------------------------------------------------- #
async def probe_chat(t: ProbeTarget) -> ProbeResult:
    started = time.monotonic()
    try:
        resp = await asyncio.wait_for(
            _complete_quiet(
                t, [{"role": "user", "content": "Reply with exactly one word: ready"}]
            ),
            t.timeout,
        )
        content = _as_text(getattr(resp.message, "content", "")).strip()
        status = "pass" if content else "fail"
        return ProbeResult(
            "chat", status, evidence=_preview(content), latency_ms=_ms(started)
        )
    except Exception as e:  # noqa: BLE001
        return ProbeResult("chat", "error", _short_error(e), latency_ms=_ms(started))


# --------------------------------------------------------------------------- #
# Probe: tool calling — selection (precision) + parameter structure + negative
# --------------------------------------------------------------------------- #
_WEATHER_PROMPT = (
    "What is the weather in Paris right now? You must call the get_weather tool."
)
_NO_TOOL_PROMPT = "What is 2+2? Answer with just the number."


def _tool_call_names(tool_calls: list) -> list[str]:
    """Pull function names out of OpenAI-shaped tool_calls (dicts or objects)."""
    names = []
    for tc in tool_calls or []:
        if isinstance(tc, dict):
            names.append(tc.get("function", {}).get("name", "?"))
        else:
            names.append(getattr(getattr(tc, "function", None), "name", "?"))
    return names


def _tool_call_args(tool_calls: list, index: int = 0) -> dict:
    """Parse the JSON arguments of the Nth tool call to a dict ({} on any failure)."""
    if not tool_calls or index >= len(tool_calls):
        return {}
    tc = tool_calls[index]
    raw = (
        tc.get("function", {}).get("arguments")
        if isinstance(tc, dict)
        else getattr(getattr(tc, "function", None), "arguments", None)
    )
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return {}


async def probe_tool_calling(t: ProbeTarget) -> ProbeResult:
    """Tool-calling capability probe — three facets:

    1. **selection** (precision): offer 3 tools, ask for one by name, check the
       *correct* tool is called (not just any tool). This is the gap the
       ecosystem (LLMToolCallingTester / DeepEval) exposes — boolean "did a tool
       fire" hides a model that calls the wrong tool.
    2. **parameter structure**: the called tool's arguments parse as JSON and
       carry the required key (``location`` for weather, ``ticker`` for stock).
    3. **negative**: with tools available but a prompt that needs none, the
       model should *not* call a tool (catches over-eager callers).

    Status is ``pass`` only if all three facets pass. A partial pass (e.g.
    selection correct but parameter missing) is ``fail`` with evidence naming
    the failing facet, so the detail is not lost.
    """
    caps = getattr(t, "profile", {}).get("capabilities") or []
    if caps and "tools" not in caps:
        return ProbeResult(
            "tool_calling", "skip", evidence="model declares no tools capability"
        )
    started = time.monotonic()
    tools = load_multi_tools()
    facets: dict[str, str] = {}
    try:
        # --- facet 1+2: selection + parameter structure (weather) ---
        resp = await asyncio.wait_for(
            _complete_quiet(
                t,
                [{"role": "user", "content": _WEATHER_PROMPT}],
                tools=tools,
                tool_choice="auto",
            ),
            t.timeout,
        )
        tool_calls = getattr(resp.message, "tool_calls", None) or []
        names = _tool_call_names(tool_calls)
        if "get_weather" in names:
            facets["selection"] = "pass"
            args = _tool_call_args(tool_calls, names.index("get_weather"))
            facets["parameters"] = (
                "pass" if "location" in args else f"fail (no 'location' arg; got {sorted(args)})"
            )
        elif names:
            facets["selection"] = f"fail (called {names[0]}, expected get_weather)"
            facets["parameters"] = "skip (wrong tool)"
        else:
            facets["selection"] = "fail (no tool called)"
            facets["parameters"] = "skip (no tool)"

        # --- facet 3: negative (should NOT call a tool) ---
        resp_neg = await asyncio.wait_for(
            _complete_quiet(
                t,
                [{"role": "user", "content": _NO_TOOL_PROMPT}],
                tools=tools,
                tool_choice="auto",
            ),
            t.timeout,
        )
        neg_calls = getattr(resp_neg.message, "tool_calls", None) or []
        facets["negative"] = "pass" if not neg_calls else f"fail (called {_tool_call_names(neg_calls)})"

        all_pass = all(v == "pass" for v in facets.values())
        status = "pass" if all_pass else "fail"
        evidence = "; ".join(f"{k}={v}" for k, v in facets.items())
        return ProbeResult(
            "tool_calling",
            status,
            evidence=evidence,
            latency_ms=_ms(started),
            detail={
                "facets": facets,
                "tool_calls": tool_calls,
                "negative_tool_calls": neg_calls,
            },
        )
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "does not support tools" in msg.lower():
            return ProbeResult(
                "tool_calling",
                "skip",
                evidence="model does not support tools (400)",
                latency_ms=_ms(started),
            )
        return ProbeResult(
            "tool_calling",
            "error",
            _short_error(e),
            latency_ms=_ms(started),
        )


# --------------------------------------------------------------------------- #
# Probe: image / vision
# --------------------------------------------------------------------------- #
async def probe_image(t: ProbeTarget) -> ProbeResult:
    caps = getattr(t, "profile", {}).get("capabilities") or []
    if caps and "vision" not in caps:
        return ProbeResult(
            "image", "skip", evidence="model declares no vision capability"
        )
    if t.provider_name in _PROVIDERS_WITHOUT_IMAGE_FORWARD:
        return ProbeResult(
            "image",
            "skip",
            evidence=f"{t.provider_name} provider does not forward images",
        )
    started = time.monotonic()
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What colors are in this image? Answer briefly.",
                    },
                    {"type": "image_url", "image_url": {"url": image_data_url()}},
                ],
            }
        ]
        resp = await asyncio.wait_for(_complete_quiet(t, messages), t.timeout)
        content = _as_text(getattr(resp.message, "content", "")).strip()
        status = "pass" if content else "fail"
        return ProbeResult(
            "image", status, evidence=_preview(content), latency_ms=_ms(started)
        )
    except Exception as e:  # noqa: BLE001
        return ProbeResult("image", "error", _short_error(e), latency_ms=_ms(started))


# --------------------------------------------------------------------------- #
# Probe: structured output (response_format: json_schema conformance)
# --------------------------------------------------------------------------- #
_STRUCT_PROMPT = (
    "Extract the person's details as JSON: name is Alice, age is 30, "
    "city is Berlin. Return only the JSON object."
)


def _validate_structured(content: str, schema: dict) -> tuple[str, dict]:
    """Lightweight structural validation of a JSON response against ``schema``.

    Returns ``(status, parsed)`` where status is ``pass``/``fail`` and ``parsed``
    is the decoded dict (``{}`` on decode failure). No jsonschema dependency —
    we check required keys + declared type, which is enough to catch models
    that declare structured_output but don't honor it (the Requesty matrix gap).
    """
    try:
        obj = json.loads(content)
    except Exception:  # noqa: BLE001
        return "fail (not valid JSON)", {}
    if not isinstance(obj, dict):
        return f"fail (not an object: {type(obj).__name__})", {}
    required = schema.get("required", [])
    missing = [k for k in required if k not in obj]
    if missing:
        return f"fail (missing required keys: {missing})", obj
    props = schema.get("properties", {})
    type_ok = True
    for key, spec in props.items():
        if key in obj:
            expected = spec.get("type")
            actual = type(obj[key]).__name__
            if expected and not _type_matches(actual, expected):
                type_ok = False
                return f"fail ({key}: expected {expected}, got {actual})", obj
    return ("pass" if type_ok else "fail"), obj


_PY_TO_JSON_TYPE = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}


def _type_matches(py_type: str, json_type: str) -> bool:
    """True if a Python type name satisfies a JSON-schema type declaration."""
    return _PY_TO_JSON_TYPE.get(py_type) == json_type or (
        json_type == "number" and py_type in ("int", "float")
    )


async def probe_structured_output(t: ProbeTarget) -> ProbeResult:
    """Structured-output probe — sends ``response_format: json_schema`` and
    validates the response parses as JSON and conforms to the schema.

    The catalog records ``structured_output`` as a capability, but Requesty's
    244-model matrix shows it's "a compatibility mess" — declared but not
    honored. This probe catches that empirically. Skips if the model declares
    no structured_output capability.
    """
    caps = getattr(t, "profile", {}).get("capabilities") or []
    if caps and "structured_output" not in caps:
        return ProbeResult(
            "structured_output",
            "skip",
            evidence="model declares no structured_output capability",
        )
    started = time.monotonic()
    schema = load_person_schema()
    try:
        resp = await asyncio.wait_for(
            _completion_target(t).acomplete(
                [{"role": "user", "content": _STRUCT_PROMPT}],
                temperature=0.0,
                max_tokens=t.max_tokens,
                reasoning_effort="none",
                extra={"response_format": {"type": "json_schema", "json_schema": {"name": "person", "schema": schema, "strict": True}}},
            ),
            t.timeout,
        )
        content = _as_text(getattr(resp.message, "content", "")).strip()
        if not content:
            return ProbeResult(
                "structured_output",
                "fail",
                evidence="empty response",
                latency_ms=_ms(started),
            )
        status, parsed = _validate_structured(content, schema)
        return ProbeResult(
            "structured_output",
            "pass" if status == "pass" else "fail",
            evidence=f"{status}; keys={sorted(parsed)}",
            latency_ms=_ms(started),
            detail={"schema_required": schema.get("required", []), "parsed": parsed},
        )
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        if "response_format" in msg or "json_schema" in msg or "not support" in msg:
            return ProbeResult(
                "structured_output",
                "skip",
                evidence=f"model does not support response_format (400): {_short_error(e)}",
                latency_ms=_ms(started),
            )
        return ProbeResult(
            "structured_output",
            "error",
            _short_error(e),
            latency_ms=_ms(started),
        )


# --------------------------------------------------------------------------- #
# Probe: thinking on / off
# --------------------------------------------------------------------------- #
async def probe_thinking_on(t: ProbeTarget) -> ProbeResult:
    return await _thinking(t, on=True)


async def probe_thinking_off(t: ProbeTarget) -> ProbeResult:
    return await _thinking(t, on=False)


async def _thinking(t: ProbeTarget, on: bool) -> ProbeResult:
    """Toggle thinking and report whether reasoning was produced."""
    started = time.monotonic()
    label = "thinking_on" if on else "thinking_off"
    caps = getattr(t, "profile", {}).get("capabilities") or []
    if caps and "thinking" not in caps:
        return ProbeResult(
            label, "skip", evidence="model declares no thinking capability"
        )
    try:
        content, thinking = await asyncio.wait_for(
            _generate(t, "What is 7 times 6? Show your reasoning.", on, max_tokens=768),
            t.timeout,
        )
        content, thinking = _as_text(content), _as_text(thinking)
        produced = bool(thinking) or "<think>" in content
        if on:
            if produced:
                status = "pass"
                evidence = f"reasoning_chars={len(thinking)}"
            elif caps:
                # No separate reasoning channel captured, but the model declares
                # reasoning — many OpenAI-compatible backends (groq/mistral/
                # openrouter) reason in-content with no reasoning_content field.
                status = "pass"
                evidence = "declared reasoning; no separate channel (in-content)"
            else:
                status = "fail"
                evidence = "no reasoning produced"
        else:
            # off: pass if the call succeeded; report whether any leaked through
            status = "pass"
            evidence = f"reasoning_chars={len(thinking)} ({'none' if not thinking else 'LEAKED'})"
        return ProbeResult(
            label,
            status,
            evidence=evidence,
            latency_ms=_ms(started),
            detail={
                "thinking_chars": len(thinking),
                "content_preview": _preview(content, 80),
            },
        )
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "does not support thinking" in msg.lower():
            return ProbeResult(
                label,
                "fail",
                evidence="model does not support thinking (400)",
                latency_ms=_ms(started),
            )
        return ProbeResult(label, "error", _short_error(e), latency_ms=_ms(started))


# --------------------------------------------------------------------------- #
# Registry + runner
# --------------------------------------------------------------------------- #
PROBES: dict[str, Callable[[ProbeTarget], "Any"]] = {
    "probe": probe_profile,
    "chat": probe_chat,
    "tool_calling": probe_tool_calling,
    "structured_output": probe_structured_output,
    "image": probe_image,
    "thinking_on": probe_thinking_on,
    "thinking_off": probe_thinking_off,
}

DEFAULT_PROBES = [
    "probe",
    "chat",
    "tool_calling",
    "structured_output",
    "image",
    "thinking_on",
    "thinking_off",
]


def save_probe_result(report: CapabilityReport) -> None:
    """Persist a probe result to the sidecar and the models.json entry.

    Mirrors the ``_speed_results.json`` pattern so ``generate_models.py`` can
    re-merge on regeneration. Keyed by ``provider/model``.
    """
    entry = {
        "profile": report.profile,
        "results": [
            {
                "name": r.name,
                "status": r.status,
                "evidence": r.evidence,
                "detail": r.detail,
            }
            for r in report.results
        ],
        "summary": report.summary,
        "tested_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    key = _probe_key(report.target)
    data = {}
    if PROBE_RESULTS_PATH.exists():
        try:
            data = json.loads(PROBE_RESULTS_PATH.read_text())
        except Exception:  # noqa: BLE001
            data = {}
    data[key] = entry
    PROBE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROBE_RESULTS_PATH.write_text(json.dumps(data, indent=2, sort_keys=True))
    _merge_probe_into_models_json(key, entry)


def _probe_key(target: str) -> str:
    provider, model = target.split("@", 1)
    return f"{provider}/{model}"


def _merge_probe_into_models_json(key: str, entry: dict) -> None:
    """Update the matching model entry in models.json with a ``probed`` field."""
    if not MODELS_JSON_PATH.exists():
        return
    try:
        catalog = json.loads(MODELS_JSON_PATH.read_text())
    except Exception:  # noqa: BLE001
        return
    provider, model = key.split("/", 1)
    prov = catalog.get("providers", {}).get(provider, {})
    for m in prov.get("models", []):
        if m.get("id") == model:
            m["probed"] = entry
            break
    else:
        return
    MODELS_JSON_PATH.write_text(json.dumps(catalog, indent=2))


async def softprobe_catalog(
    *,
    providers: Optional[str] = None,
    stale_days: Optional[int] = None,
    force: bool = False,
    ollama_key: Optional[str] = None,
    ollama_url: Optional[str] = None,
    on_progress: Optional[Callable[[str, Any, str], None]] = None,
) -> dict[str, int]:
    """Probe (metadata ONLY — zero inference tokens) every catalog model.

    Reads declared capabilities from the catalog (local JSON) or Ollama
    ``/api/show``. Never calls chat/tools/image/thinking/perf. Skips entries
    fresher than ``stale_days`` so reprobes stagger. Persists to
    ``_probe_results.json`` and updates each model's ``probed`` field in
    ``models.json``. Shared by the ``--softprobe`` CLI and the daily refresh.
    """
    from uniinfer.proxy_services.models_registry import load_catalog

    existing: dict[str, Any] = {}
    if PROBE_RESULTS_PATH.exists():
        try:
            existing = json.loads(PROBE_RESULTS_PATH.read_text())
        except Exception:  # noqa: BLE001
            existing = {}
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=stale_days)
        if (stale_days and not force)
        else None
    )
    catalog = load_catalog(providers) if providers else load_catalog()
    provs = catalog.get("providers", {})

    probed = skipped = errors = 0
    for pname, pdata in provs.items():
        for m in pdata.get("models", []):
            mid = m.get("id")
            if not mid:
                continue
            pm = f"{pname}@{mid}"
            pkey = f"{pname}/{mid}"
            if cutoff:
                ent = existing.get(pkey)
                tat = ent.get("tested_at") if ent else None
                if tat:
                    try:
                        if datetime.fromisoformat(tat.replace("Z", "+00:00")) >= cutoff:
                            skipped += 1
                            continue
                    except Exception:  # noqa: BLE001
                        pass
            tgt = ProbeTarget(
                provider_model=pm,
                api_key=ollama_key if pname == "ollama" else None,
                base_url=ollama_url if pname == "ollama" else None,
            )
            try:
                r = await probe_profile(tgt)  # metadata only — 0 tokens
                save_probe_result(
                    CapabilityReport(
                        target=pm,
                        profile=r.detail.get("profile", {}) or {},
                        results=[r],
                    )
                )
                probed += 1
                if on_progress:
                    on_progress(pm, r, "ok")
            except Exception as e:  # noqa: BLE001
                errors += 1
                if on_progress:
                    on_progress(pm, e, "error")
    return {"probed": probed, "skipped": skipped, "errors": errors}


async def run_capabilities(
    target: ProbeTarget,
    probes: Optional[list[str]] = None,
    perf: bool = False,
    save: bool = False,
) -> CapabilityReport:
    """Run the capability matrix against ``target`` (sequential).

    If ``save``, persist the result to ``_probe_results.json`` and the matching
    models.json ``probed`` field.
    """
    selected = DEFAULT_PROBES if probes is None else probes
    results: list[ProbeResult] = []
    profile: dict[str, Any] = {}

    # Probe policy (model_quirks.json _probe_policy): some providers are excluded
    # from empirical probing (e.g. paid/not-installed) — short-circuit to a skip.
    _policy = _model_quirks().get("_probe_policy", {}) or {}
    if target.provider_name in (_policy.get("no_empirical") or []):
        return CapabilityReport(
            target=target.provider_model,
            profile={},
            results=[
                ProbeResult(
                    "probe",
                    "skip",
                    evidence=f"{target.provider_name} excluded from empirical probing (policy)",
                )
            ],
        )

    for name in selected:
        fn = PROBES.get(name)
        if fn is None:
            results.append(ProbeResult(name, "skip", evidence="unknown probe"))
            continue
        r = await _safe(fn, target)
        results.append(r)
        if name == "probe":
            # expose the profile to later probes (thinking probes skip if undeclared)
            profile = target.profile = r.detail.get("profile", {}) or {}  # type: ignore[attr-defined]

    if perf:
        # Lazy import: perf.py imports core's helpers, so avoid an import cycle.
        from .perf import PERF_PROBES

        for name, fn in PERF_PROBES.items():
            results.append(await _safe(fn, target))

    report = CapabilityReport(
        target=target.provider_model, profile=profile, results=results
    )
    if save:
        try:
            save_probe_result(report)
        except Exception:  # noqa: BLE001
            pass  # persistence must never break a probe run
    return report


async def _safe(fn: Callable[[ProbeTarget], Any], t: ProbeTarget) -> ProbeResult:
    try:
        return await fn(t)
    except Exception as e:  # noqa: BLE001
        return ProbeResult(getattr(fn, "__name__", "probe"), "error", _short_error(e))


# --------------------------------------------------------------------------- #
# Pretty print (CLI)
# --------------------------------------------------------------------------- #
def format_report(report: CapabilityReport) -> str:
    lines = [f"# Capability report: {report.target}", ""]
    if report.profile:
        lines.append("## profile")
        for k, v in report.profile.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
    lines.append(f"## matrix  ({report.summary})")
    width = max((len(r.name) for r in report.results), default=8)
    for r in report.results:
        mark = {"pass": "✅", "fail": "❌", "skip": "⏭️ ", "error": "💥"}[r.status]
        lat = f" {r.latency_ms:.0f}ms" if r.latency_ms is not None else ""
        lines.append(f"  {mark} {r.name:<{width}}  {r.evidence}{lat}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# tiny helpers
# --------------------------------------------------------------------------- #
def _as_text(content: Any) -> str:
    """Coerce model content (may be a list of parts, e.g. reasoning models) to a string."""
    if content is None:
        return ""
    if isinstance(content, list):
        return "".join(
            p.get("text", "")
            if isinstance(p, dict)
            else (p if isinstance(p, str) else "")
            for p in content
        )
    return content if isinstance(content, str) else str(content)


def _short_error(e: BaseException) -> str:
    """Concise error string: pull the provider's human message out of an
    embedded JSON blob (rate-limit / billing errors), else type + str."""
    msg = str(e)
    m = re.search(r"['\"]?message['\"]?\s*:\s*['\"]([^'\"]+)['\"]", msg)
    if m:
        return f"{type(e).__name__}: {m.group(1)[:160]}"
    return f"{type(e).__name__}: {msg[:160]}"


def _ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000)


def _preview(s: str, n: int = 80) -> str:
    s = (s or "").replace("\n", " ").strip()
    return (s[:n] + "…") if len(s) > n else s
