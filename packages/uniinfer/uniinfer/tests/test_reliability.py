"""Reliability monitor: pattern detection + notification contract.

Hermetic — no network, no LLM calls. Pins the detection rules derived from
the 2026-09-29/30 TU wedge forensics: 5xx wedge bursts, slow-open crawl mode
(TTFT near the open-timeout boundary), sustained error rate; plus 429
exclusion, cooldown suppression, recovery notices, the disabled switch, and
the StatsCollector.record ingestion seam.
"""
import logging

import pytest

import uniinfer.proxy_services.stats as stats_mod
from uniinfer.proxy_services.reliability import ReliabilityMonitor
from uniinfer.proxy_services.stats import StatsCollector

_KNOB_ENVS = (
    "UNIINFER_RELIABILITY_DISABLED",
    "UNIINFER_RELIABILITY_WEBHOOK",
    "UNIINFER_RELIABILITY_NTFY",
    "UNIINFER_RELIABILITY_NTFY_TOKEN",
    "UNIINFER_RELIABILITY_COOLDOWN_S",
    "UNIINFER_RELIABILITY_WEDGE_MIN",
)


@pytest.fixture
def monitor(monkeypatch):
    ReliabilityMonitor._instance = None
    m = ReliabilityMonitor()
    for var in _KNOB_ENVS:
        monkeypatch.delenv(var, raising=False)
    yield m
    ReliabilityMonitor._instance = None


def test_wedge_burst_degrades(monitor, caplog):
    """3 hard 5xx failures within the window = degraded + one WARNING line."""
    with caplog.at_level(logging.WARNING, logger="uniioai_proxy"):
        monitor.note("tu@deepseek", status=200, latency_ms=1000.0, ttft_ms=900.0)
        monitor.note("tu@deepseek", status=504, latency_ms=90000.0, ttft_ms=None)
        monitor.note("tu@deepseek", status=504, latency_ms=90000.0, ttft_ms=None)
        assert monitor.snapshot()["degraded_models"] == []
        monitor.note("tu@deepseek", status=504, latency_ms=90000.0, ttft_ms=None)
    snap = monitor.snapshot()
    assert snap["degraded_models"] == ["tu@deepseek"]
    assert snap["models"]["tu@deepseek"]["reason"] == "wedge_burst"
    assert snap["channels"] == ["log"]
    assert any(
        "reliability" in r.getMessage() and "degraded" in r.getMessage()
        for r in caplog.records
    )


def test_rate_limits_are_not_unreliability(monitor):
    """429s are expected backoff signals, not upstream unreliability."""
    for _ in range(10):
        monitor.note("tu@x", status=429, latency_ms=100.0, ttft_ms=None)
    assert monitor.snapshot()["degraded_models"] == []


def test_cooldown_suppresses_repeat_alerts(monitor, monkeypatch):
    monkeypatch.setenv("UNIINFER_RELIABILITY_COOLDOWN_S", "3600")
    for _ in range(3):
        monitor.note("tu@x", status=504, latency_ms=1.0, ttft_ms=None)
    n_events = len(monitor.snapshot()["last_events"])
    for _ in range(3):
        monitor.note("tu@x", status=504, latency_ms=1.0, ttft_ms=None)
    snap = monitor.snapshot()
    assert len(snap["last_events"]) == n_events
    assert snap["models"]["tu@x"]["state"] == "degraded"


def test_recovery_after_clean_streak(monitor):
    for _ in range(3):
        monitor.note("tu@x", status=504, latency_ms=1.0, ttft_ms=None)
    assert monitor.snapshot()["degraded_models"] == ["tu@x"]
    for _ in range(10):
        monitor.note("tu@x", status=200, latency_ms=1000.0, ttft_ms=800.0)
    snap = monitor.snapshot()
    assert snap["degraded_models"] == []
    assert snap["last_events"][-1]["event"] == "recovered"


def test_slow_open_crawl_mode(monitor):
    """Streams that succeed but only just (TTFT at the open-timeout boundary)
    mark crawl mode — the 90.6s-for-0KB observation from 2026-09-30."""
    for _ in range(17):
        monitor.note("tu@x", status=200, latency_ms=90000.0, ttft_ms=900.0)
    assert monitor.snapshot()["degraded_models"] == []
    for _ in range(3):
        monitor.note("tu@x", status=200, latency_ms=95000.0, ttft_ms=85000.0)
    assert monitor.snapshot()["models"]["tu@x"]["reason"] == "slow_open"


def test_sustained_error_rate(monitor, monkeypatch):
    monkeypatch.setenv("UNIINFER_RELIABILITY_WEDGE_MIN", "50")  # isolate the rate rule
    for i in range(20):
        status = 500 if i % 10 == 0 else 200  # 2/20 = 10%
        monitor.note("tu@x", status=status, latency_ms=1000.0, ttft_ms=900.0)
    assert monitor.snapshot()["models"]["tu@x"]["reason"] == "error_rate"


def test_disabled_switch(monitor, monkeypatch):
    monkeypatch.setenv("UNIINFER_RELIABILITY_DISABLED", "1")
    for _ in range(5):
        monitor.note("tu@x", status=504, latency_ms=1.0, ttft_ms=None)
    snap = monitor.snapshot()
    assert snap["enabled"] is False
    assert snap["models"] == {}


def test_stats_record_feeds_monitor(monitor, tmp_path, monkeypatch):
    """The ingestion seam: every StatsCollector.record outcome reaches the
    monitor — no router/middleware changes needed, now or in the future."""
    monkeypatch.setattr(stats_mod, "_SNAPSHOT_PATH", str(tmp_path / "stats.json"))
    StatsCollector._instance = None
    try:
        collector = StatsCollector()
        for _ in range(3):
            collector.record("tu@deepseek", status=504, latency_ms=90000.0, usage=None)
        assert monitor.snapshot()["degraded_models"] == ["tu@deepseek"]
    finally:
        StatsCollector._instance = None


def test_send_test_notification_log_only(monitor):
    out = monitor.send_test_notification()
    assert out["channels"]["log"].startswith("ok")
    assert out["event"]["event"] == "test"


def test_ntfy_token_sends_bearer_auth(monitor, monkeypatch):
    """Reserved (account-protected) topics publish under an access token —
    without the Bearer header ntfy.sh answers 403 and alerts silently die."""
    import httpx

    import uniinfer.proxy_services.reliability as rel

    monkeypatch.setenv("UNIINFER_RELIABILITY_NTFY", "kontext-rel")
    monkeypatch.setenv("UNIINFER_RELIABILITY_NTFY_TOKEN", "tk_secret")
    captured = {}

    class _Resp:
        status_code = 200

    class _Client:
        def __init__(self, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, content=None, headers=None, **kw):
            captured["url"] = url
            captured["headers"] = headers
            return _Resp()

    monkeypatch.setattr(httpx, "Client", _Client)
    results = rel._deliver_sync(
        {"event": "test", "model": "tu@x", "reason": "r", "detail": "d"}, rel._cfg()
    )
    assert results == [("ntfy", True, "HTTP 200")]
    assert captured["url"] == "https://ntfy.sh/kontext-rel"
    assert captured["headers"]["Authorization"] == "Bearer tk_secret"
