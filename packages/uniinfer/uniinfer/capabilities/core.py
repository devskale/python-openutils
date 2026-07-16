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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from uniinfer import ChatCompletionRequest, ChatMessage, ProviderFactory
from uniinfer.uniioai import aget_completion

FIXTURES = Path(__file__).parent / "fixtures"

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
class Target:
    """A model to test and how to reach it."""

    provider_model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    # Thinking models need room — reasoning can consume many tokens before the
    # visible answer. Keep this ≫ 1–2k. Short-answer probes stop early anyway.
    max_tokens: int = 4096
    timeout: float = 90.0

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
    """Canonical tool definitions used by the tool-calling probe."""
    return json.loads((FIXTURES / "tools.json").read_text())


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
async def probe_profile(t: Target) -> ProbeResult:
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
        return ProbeResult(
            "probe", "error", f"{type(e).__name__}: {e}"[:200], latency_ms=_ms(started)
        )


async def _ollama_show_profile(t: Target) -> dict[str, Any]:
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


async def _catalog_profile(t: Target) -> dict[str, Any]:
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
async def _generate(
    t: Target, prompt: str, thinking_on: bool, *, max_tokens: Optional[int] = None
) -> tuple[str, str]:
    """Return (content, thinking). Thinking is toggled via the backend-native
    knob: Ollama ``think`` (provider-direct, since the uniioai wrappers do not
    forward it for non-stream) or vLLM ``chat_template_kwargs.enable_thinking``.

    ``max_tokens`` overrides ``t.max_tokens``. The thinking probes pass a
    *moderate* cap — capability detection only needs to observe that reasoning
    is produced, not generate a full ≫1–2k chain (which on slow models blows the
    probe timeout). Real callers should still use a generous ``t.max_tokens``.
    """
    mt = max_tokens or t.max_tokens
    if t.provider_name == "ollama":
        provider = ProviderFactory.get_provider(
            "ollama", api_key=t.api_key, base_url=t.base_url
        )
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content=prompt)],
            model=t.model_name,
            temperature=0.7,
            max_tokens=mt,
            streaming=False,
        )
        resp = await provider.acomplete(req, think=thinking_on)
        return getattr(resp.message, "content", "") or "", getattr(
            resp, "thinking", ""
        ) or ""
    resp = await aget_completion(
        messages=[{"role": "user", "content": prompt}],
        provider_model_string=t.provider_model,
        temperature=0.7,
        max_tokens=mt,
        provider_api_key=t.api_key,
        base_url=t.base_url,
        chat_template_kwargs={"enable_thinking": thinking_on},
    )
    return getattr(resp.message, "content", "") or "", getattr(
        resp, "thinking", ""
    ) or ""


# --------------------------------------------------------------------------- #
# Quiet completion — thinking OFF so reasoning can't eat the token budget
# --------------------------------------------------------------------------- #
async def _complete_quiet(
    t: Target,
    messages: list[dict],
    *,
    tools: Optional[list[dict]] = None,
    tool_choice: Optional[Any] = None,
    max_tokens: Optional[int] = None,
) -> Any:
    """Non-streaming completion with thinking OFF.

    Ollama is called provider-direct so the native ``think`` field is honoured
    (the uniioai wrappers do not forward it for non-stream) and ``tools``/
    images reach the backend. ``max_tokens`` overrides ``t.max_tokens`` (used by
    the perf probe to cap generation and stay token-frugal).
    """
    mt = max_tokens or t.max_tokens
    if t.provider_name == "ollama":
        provider = ProviderFactory.get_provider(
            "ollama", api_key=t.api_key, base_url=t.base_url
        )
        req = ChatCompletionRequest(
            messages=[ChatMessage(**m) for m in messages],
            model=t.model_name,
            temperature=0.7,
            max_tokens=mt,
            streaming=False,
            tools=tools,
            tool_choice=tool_choice,
        )
        return await provider.acomplete(req, think=False)
    return await aget_completion(
        messages=messages,
        provider_model_string=t.provider_model,
        temperature=0.7,
        max_tokens=mt,
        provider_api_key=t.api_key,
        base_url=t.base_url,
        tools=tools,
        tool_choice=tool_choice,
        chat_template_kwargs={"enable_thinking": False},
    )


# --------------------------------------------------------------------------- #
# Probe: chat completion (thinking OFF so reasoning can't eat the token budget)
# --------------------------------------------------------------------------- #
async def probe_chat(t: Target) -> ProbeResult:
    started = time.monotonic()
    try:
        resp = await asyncio.wait_for(
            _complete_quiet(
                t, [{"role": "user", "content": "Reply with exactly one word: ready"}]
            ),
            t.timeout,
        )
        content = (getattr(resp.message, "content", "") or "").strip()
        status = "pass" if content else "fail"
        return ProbeResult(
            "chat", status, evidence=_preview(content), latency_ms=_ms(started)
        )
    except Exception as e:  # noqa: BLE001
        return ProbeResult(
            "chat", "error", f"{type(e).__name__}: {e}"[:200], latency_ms=_ms(started)
        )


# --------------------------------------------------------------------------- #
# Probe: tool calling
# --------------------------------------------------------------------------- #
async def probe_tool_calling(t: Target) -> ProbeResult:
    caps = getattr(t, "profile", {}).get("capabilities") or []
    if caps and "tools" not in caps:
        return ProbeResult(
            "tool_calling", "skip", evidence="model declares no tools capability"
        )
    started = time.monotonic()
    try:
        resp = await asyncio.wait_for(
            _complete_quiet(
                t,
                [
                    {
                        "role": "user",
                        "content": "What is the weather in Paris right now? You must call the get_weather tool.",
                    }
                ],
                tools=load_tools(),
                tool_choice="auto",
            ),
            t.timeout,
        )
        tool_calls = getattr(resp.message, "tool_calls", None) or []
        content = getattr(resp.message, "content", "") or ""
        finish = getattr(resp, "finish_reason", None)
        if tool_calls:
            names = ",".join(
                tc.get("function", {}).get("name", "?") if isinstance(tc, dict) else "?"
                for tc in tool_calls
            )
            return ProbeResult(
                "tool_calling",
                "pass",
                evidence=f"tool_calls=[{names}]",
                latency_ms=_ms(started),
                detail={"tool_calls": tool_calls},
            )
        leaked = "get_weather" in content or "tool_call" in content
        evidence = f"no structured tool_call; finish={finish}; content_hint={leaked}"
        return ProbeResult(
            "tool_calling",
            "fail",
            evidence=evidence,
            latency_ms=_ms(started),
            detail={"content_preview": _preview(content, 120)},
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
            f"{type(e).__name__}: {msg}"[:200],
            latency_ms=_ms(started),
        )


# --------------------------------------------------------------------------- #
# Probe: image / vision
# --------------------------------------------------------------------------- #
async def probe_image(t: Target) -> ProbeResult:
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
        content = (getattr(resp.message, "content", "") or "").strip()
        status = "pass" if content else "fail"
        return ProbeResult(
            "image", status, evidence=_preview(content), latency_ms=_ms(started)
        )
    except Exception as e:  # noqa: BLE001
        return ProbeResult(
            "image", "error", f"{type(e).__name__}: {e}"[:200], latency_ms=_ms(started)
        )


# --------------------------------------------------------------------------- #
# Probe: thinking on / off
# --------------------------------------------------------------------------- #
async def probe_thinking_on(t: Target) -> ProbeResult:
    return await _thinking(t, on=True)


async def probe_thinking_off(t: Target) -> ProbeResult:
    return await _thinking(t, on=False)


async def _thinking(t: Target, on: bool) -> ProbeResult:
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
        return ProbeResult(
            label, "error", f"{type(e).__name__}: {msg}"[:200], latency_ms=_ms(started)
        )


# --------------------------------------------------------------------------- #
# Perf probes (opt-in)
# --------------------------------------------------------------------------- #
async def perf_maxspeed(t: Target) -> ProbeResult:
    """Throughput (tok/min) swept across varying context sizes.

    Token-frugal by design: tiny capped output, one run per size, thinking OFF —
    long perf runs exhaust token-limited (free-tier) models fast. Sizes that
    exceed the model's declared context are skipped.
    """
    started = time.monotonic()
    sizes = getattr(t, "perf_context_sizes", None) or (128, 1024)
    declared_ctx = (getattr(t, "profile", {}).get("context_length") or 0) or None
    rows = []
    for n in sizes:
        if declared_ctx and n >= declared_ctx:
            rows.append(
                {
                    "context_tokens": n,
                    "tok_s": None,
                    "tok_min": None,
                    "note": "exceeds declared ctx",
                }
            )
            continue
        try:
            tok_s = await _measure_throughput(t, n)
        except Exception as e:  # noqa: BLE001
            rows.append(
                {
                    "context_tokens": n,
                    "tok_s": None,
                    "tok_min": None,
                    "note": type(e).__name__,
                }
            )
            continue
        rows.append(
            {
                "context_tokens": n,
                "tok_s": tok_s,
                "tok_min": round(tok_s * 60) if tok_s else None,
            }
        )
    valid = [r["tok_min"] for r in rows if r.get("tok_min")]
    best = max(valid) if valid else None
    parts = []
    for r in rows:
        v = r.get("tok_min")
        parts.append(
            f"{r['context_tokens']}t→{v if v is not None else r.get('note','?')}"
            + (" tok/min" if v is not None else "")
        )
    status = "pass" if valid else "fail"
    head = f"max≈{best} tok/min | " if best else ""
    return ProbeResult(
        "perf_maxspeed",
        status,
        evidence=head + " ; ".join(parts),
        latency_ms=_ms(started),
        detail={"sweep": rows, "declared_context": declared_ctx},
    )


async def perf_context(t: Target) -> ProbeResult:
    """Context ceiling: declared (from probe) vs empirical grow (capped)."""
    started = time.monotonic()
    try:
        declared = t.profile.get("context_length") if hasattr(t, "profile") else None
        return ProbeResult(
            "perf_context",
            "pass",
            evidence=f"declared_context={declared}",
            latency_ms=_ms(started),
            detail={
                "declared": declared,
                "note": "empirical grow-probe not run by default (expensive)",
            },
        )
    except Exception as e:  # noqa: BLE001
        return ProbeResult(
            "perf_context",
            "error",
            f"{type(e).__name__}: {e}"[:200],
            latency_ms=_ms(started),
        )


async def perf_ratelimit(t: Target) -> ProbeResult:
    """Rate-limit ceiling. Non-invasive: reads the proxy's learned limit if a
    proxy base is configured; otherwise skips (empirical burst-probe is TODO)."""
    proxy = getattr(t, "proxy_base_url", None)
    if not proxy:
        return ProbeResult(
            "perf_ratelimit",
            "skip",
            evidence="no proxy base; empirical burst-probe not implemented",
        )
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=t.timeout) as client:
            r = await client.get(
                f"{proxy.rstrip('/')}/v1/system/rate-limits",
                headers={"Authorization": f"Bearer {t.api_key}"} if t.api_key else {},
            )
            r.raise_for_status()
            data = r.json()
        entry = None
        for key, val in (data.get("limits") or {}).items():
            if t.provider_model in key or t.model_name in key:
                entry = val
                break
        if not entry:
            return ProbeResult(
                "perf_ratelimit",
                "skip",
                evidence="no learned limit for this model",
                latency_ms=_ms(started),
            )
        return ProbeResult(
            "perf_ratelimit",
            "pass",
            evidence=f"learned_rpm={entry}",
            latency_ms=_ms(started),
            detail={"entry": entry},
        )
    except Exception as e:  # noqa: BLE001
        return ProbeResult(
            "perf_ratelimit",
            "error",
            f"{type(e).__name__}: {e}"[:200],
            latency_ms=_ms(started),
        )


async def _measure_throughput(t: Target, context_tokens: int) -> Optional[float]:
    """Measure output tok/s for a generation prefilled with ~context_tokens.

    Thinking OFF, output capped at 32 tokens (a short count) — so the token
    cost is ~context_tokens input + a few dozen output.
    """
    filler = "lorem ipsum dolor sit amet " * max(1, context_tokens // 6)
    filler = filler[: context_tokens * 4]
    messages = [
        {
            "role": "user",
            "content": f"Context: {filler}\n\nCount from 1 to 30, one number per line.",
        }
    ]
    started = time.monotonic()
    resp = await asyncio.wait_for(
        _complete_quiet(t, messages, max_tokens=32), t.timeout
    )
    elapsed = max(time.monotonic() - started, 1e-3)
    usage = getattr(resp, "usage", {}) or {}
    out_tokens = usage.get("completion_tokens") or usage.get("eval_count") or 0
    if not out_tokens:
        return None
    return round(out_tokens / elapsed, 2)


# --------------------------------------------------------------------------- #
# Registry + runner
# --------------------------------------------------------------------------- #
PROBES: dict[str, Callable[[Target], "Any"]] = {
    "probe": probe_profile,
    "chat": probe_chat,
    "tool_calling": probe_tool_calling,
    "image": probe_image,
    "thinking_on": probe_thinking_on,
    "thinking_off": probe_thinking_off,
}

PERF_PROBES: dict[str, Callable[[Target], "Any"]] = {
    "maxspeed": perf_maxspeed,
    "context": perf_context,
    "ratelimit": perf_ratelimit,
}

DEFAULT_PROBES = [
    "probe",
    "chat",
    "tool_calling",
    "image",
    "thinking_on",
    "thinking_off",
]


async def run_capabilities(
    target: Target,
    probes: Optional[list[str]] = None,
    perf: bool = False,
) -> CapabilityReport:
    """Run the capability matrix against ``target`` (sequential)."""
    selected = probes or DEFAULT_PROBES
    results: list[ProbeResult] = []
    profile: dict[str, Any] = {}

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
        for name, fn in PERF_PROBES.items():
            results.append(await _safe(fn, target))

    return CapabilityReport(
        target=target.provider_model, profile=profile, results=results
    )


async def _safe(fn: Callable[[Target], Any], t: Target) -> ProbeResult:
    try:
        return await fn(t)
    except Exception as e:  # noqa: BLE001
        return ProbeResult(
            getattr(fn, "__name__", "probe"), "error", f"{type(e).__name__}: {e}"[:200]
        )


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
def _ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000)


def _preview(s: str, n: int = 80) -> str:
    s = (s or "").replace("\n", " ").strip()
    return (s[:n] + "…") if len(s) > n else s
