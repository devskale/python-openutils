"""Watchdog + memory guard: the proxy must never freeze silently.

- sd_notify: READY on arm, WATCHDOG pings at WatchdogSec/3 (unix dgram).
- memory guard: gc + pool clear, then controlled exit when RSS stays high.
"""

import asyncio
import os
import socket as _socket
import tempfile

import pytest

from uniinfer.proxy_services.sd_notify import memory_guard_task, sd_notify, watchdog_task


def test_sd_notify_no_socket_is_noop():
    os.environ.pop("NOTIFY_SOCKET", None)
    assert sd_notify("READY=1") is False


def test_sd_notify_sends_datagram():
    sock_path = tempfile.mkdtemp(prefix="sdn-", dir="/tmp") + "/notify.sock"
    rx = _socket.socket(_socket.AF_UNIX, _socket.SOCK_DGRAM)
    rx.bind(sock_path)
    rx.settimeout(2)
    os.environ["NOTIFY_SOCKET"] = sock_path
    try:
        assert sd_notify("READY=1") is True
        assert rx.recv(64).decode() == "READY=1"
    finally:
        os.environ.pop("NOTIFY_SOCKET", None)
        rx.close()


@pytest.mark.asyncio
async def test_watchdog_sends_ready_then_pings():
    sock_path = tempfile.mkdtemp(prefix="sdn-", dir="/tmp") + "/notify.sock"
    rx = _socket.socket(_socket.AF_UNIX, _socket.SOCK_DGRAM)
    rx.bind(sock_path)
    rx.settimeout(2)
    os.environ["NOTIFY_SOCKET"] = sock_path
    os.environ["WATCHDOG_USEC"] = str(int(0.3 * 1e6))  # 0.3s -> ping every 0.1s
    rx.setblocking(False)
    try:
        task = asyncio.create_task(watchdog_task())
        msgs = []
        for _ in range(150):  # ~3s budget, polling without blocking the loop
            try:
                msgs.append(rx.recv(64).decode())
            except BlockingIOError:
                pass
            if len(msgs) >= 4:
                break
            await asyncio.sleep(0.02)
        task.cancel()
        assert msgs[0] == "READY=1"
        assert all(m == "WATCHDOG=1" for m in msgs[1:]), msgs
    finally:
        os.environ.pop("NOTIFY_SOCKET", None)
        os.environ.pop("WATCHDOG_USEC", None)
        rx.close()


@pytest.mark.asyncio
async def test_watchdog_unarmed_without_usec():
    os.environ["NOTIFY_SOCKET"] = tempfile.mkdtemp(prefix="sdn-", dir="/tmp") + "/n.sock"
    os.environ.pop("WATCHDOG_USEC", None)
    try:
        task = asyncio.create_task(watchdog_task())
        await asyncio.sleep(0.05)  # returns immediately (no ping loop)
        assert task.done()
    finally:
        os.environ.pop("NOTIFY_SOCKET", None)


@pytest.mark.asyncio
async def test_mem_guard_disabled_by_default(monkeypatch):
    monkeypatch.delenv("UNIINFER_MEM_GUARD_MB", raising=False)
    task = asyncio.create_task(memory_guard_task(get_rss_mb=lambda: 9999.0, exit_fn=lambda c: None))
    await asyncio.sleep(0.05)
    assert task.done()


@pytest.mark.asyncio
async def test_mem_guard_recovers_without_exit(monkeypatch):
    monkeypatch.setenv("UNIINFER_MEM_GUARD_MB", "300")
    monkeypatch.setenv("UNIINFER_MEM_GUARD_INTERVAL_S", "0.01")
    rss_values = iter([350.0, 100.0])  # over the line, then recovered
    exits = []
    task = asyncio.create_task(memory_guard_task(
        get_rss_mb=lambda: next(rss_values, 100.0), exit_fn=lambda c: exits.append(c)))
    await asyncio.sleep(0.2)
    task.cancel()
    assert exits == []  # recovery worked -> no exit


@pytest.mark.asyncio
async def test_mem_guard_exits_when_stays_high(monkeypatch):
    monkeypatch.setenv("UNIINFER_MEM_GUARD_MB", "300")
    monkeypatch.setenv("UNIINFER_MEM_GUARD_INTERVAL_S", "0.01")
    exits = []
    task = asyncio.create_task(memory_guard_task(
        get_rss_mb=lambda: 400.0, exit_fn=lambda c: exits.append(c)))
    await asyncio.sleep(0.2)
    task.cancel()
    assert exits == [75]  # controlled exit for clean restart


def test_health_reports_asyncio_task_count():
    """Task-leak gauge: /health must expose len(asyncio.all_tasks()) so a
    monotonic climb is visible in one curl (async-concurrency.com pattern)."""
    from fastapi.testclient import TestClient
    import uniinfer.proxy_app as pa

    with TestClient(pa.app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        tasks = body.get("asyncio_tasks")
        assert isinstance(tasks, int) and tasks >= 1


def test_mem_trace_jemalloc_fields(monkeypatch):
    """Trace line carries the live/retained split when jemalloc is active
    (tomorrow's verification: live grows = leak, retained grows = allocator)."""
    import uniinfer.proxy_middleware as pm

    monkeypatch.setattr(pm, "jemalloc_stats", lambda: {"allocated": 22.5, "resident": 28.0, "retained": 5.0})
    f = pm._jemalloc_fields()
    assert f == {"j_live": 22.5, "j_res": 28.0, "j_ret": 5.0}

    monkeypatch.setattr(pm, "jemalloc_stats", lambda: {"allocated": None})
    assert pm._jemalloc_fields() == {}

    def boom():
        raise RuntimeError("no jemalloc")
    monkeypatch.setattr(pm, "jemalloc_stats", boom)
    assert pm._jemalloc_fields() == {}
