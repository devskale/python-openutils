"""UniInfer capability test suite.

Public API for probing what a model can do and exercising each feature
(chat, tool-calling, image, thinking on/off) plus optional perf probes
(maxspeed across varying contexts, context ceiling, rate-limit ceiling).
"""
from .core import (
    DEFAULT_PROBES,
    PERF_PROBES,
    PROBES,
    CapabilityReport,
    ProbeResult,
    ProbeTarget,
    format_report,
    load_tools,
    image_b64,
    image_data_url,
    perf_context,
    perf_maxspeed,
    perf_ratelimit,
    probe_chat,
    probe_image,
    probe_profile,
    probe_thinking_off,
    probe_thinking_on,
    probe_tool_calling,
    run_capabilities,
    save_probe_result,
    softprobe_catalog,
)

# Back-compat alias for the renamed probe-config dataclass; prefer ProbeTarget.
Target = ProbeTarget

__all__ = [
    "DEFAULT_PROBES",
    "PERF_PROBES",
    "PROBES",
    "CapabilityReport",
    "ProbeResult",
    "ProbeTarget",
    "Target",
    "format_report",
    "load_tools",
    "image_b64",
    "image_data_url",
    "perf_context",
    "perf_maxspeed",
    "perf_ratelimit",
    "probe_chat",
    "probe_image",
    "probe_profile",
    "probe_thinking_off",
    "probe_thinking_on",
    "probe_tool_calling",
    "run_capabilities",
    "save_probe_result",
]
