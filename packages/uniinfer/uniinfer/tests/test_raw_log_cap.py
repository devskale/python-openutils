"""Raw-log cap: log_raw_response must keep its sink bounded.

Regression for the strukt2meta/logs incident: uniinfer's always-on error-path
logging (non-200, preemption, empty stream) appends to
``logs/{provider}_raw_chat.log`` relative to CWD — under the worker's
``cwd: packages/strukt2meta`` that grew tu_raw_chat.log to 1.7 GB on tu and
293 MB on pi5 with no rotation and no gate (the UNIINFER_DEBUG_RAW gate from
2026-06-24 only covers the happy-path logging, not the error paths).

Contract: after an append that pushes the file over the cap, the file shrinks
to roughly half the cap (oldest half dropped, newest entries kept) — and the
cap never raises into the caller's error path.
"""
import os

from uniinfer.logging_utils import _enforce_raw_log_cap, log_raw_response


def test_cap_noop_when_under_limit(tmp_path):
    log = tmp_path / "tu_raw_chat.log"
    log.write_bytes(b"x" * 1000)
    _enforce_raw_log_cap(str(log), max_bytes=5000)
    assert log.stat().st_size == 1000


def test_cap_truncates_oldest_half(tmp_path):
    log = tmp_path / "tu_raw_chat.log"
    # 10 KB of lines, cap 5 KB -> expect ~2.5 KB kept (the NEWEST bytes)
    lines = b"".join(b"line-%06d\n" % i for i in range(1000))  # 10 bytes/line
    log.write_bytes(lines)
    _enforce_raw_log_cap(str(log), max_bytes=5000)
    size = log.stat().st_size
    assert size <= 2600, f"cap kept {size} bytes"
    assert size >= 2000, f"cap dropped too much: {size} bytes"
    # newest content survives
    tail = log.read_bytes()
    assert b"line-000999" in tail
    assert b"line-000000" not in tail


def test_cap_missing_file_is_silent(tmp_path):
    _enforce_raw_log_cap(str(tmp_path / "nope.log"))  # must not raise


def test_log_raw_response_caps_after_append(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIINFER_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("UNIINFER_RAW_LOG_MAX_BYTES", "5000")
    log = tmp_path / "tu_raw_chat.log"
    body = "x" * 900
    # 8 appends of ~1 KB each -> over the 5 KB default cap -> capped
    for i in range(8):
        log_raw_response(
            provider="tu",
            operation="chat.completions",
            raw_response={"status_code": 500, "body": body},
            log_file=str(log),
        )
        # bounded: never more than cap (5 KB) + one entry (~1 KB) past the append
        assert log.stat().st_size <= 5000 + 1100, f"append #{i} left {log.stat().st_size} bytes"
    # still valid JSON lines after capping (cap cuts on byte boundary of a
    # half; the first partial line may be truncated — tolerate it)
    kept = [l for l in log.read_text().splitlines() if l.startswith("{")]
    assert kept, "cap dropped every parseable entry"
    import json

    json.loads(kept[-1])  # newest entry intact
