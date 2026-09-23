"""systemd sd_notify support + memory guard for the uniioai proxy.

The proxy's event loop can freeze (live finding 2026-09-23: RSS grew past
the cgroup memory.high -> kernel reclaim throttled the process until GET /
took 194s and SIGTERM was ignored). Two structural defenses:

- sd_notify watchdog: the app pings ``WATCHDOG=1`` from the event loop. A
  frozen loop stops pinging and systemd restarts the unit (Type=notify +
  WatchdogSec). No dependency on python-systemd — the notify protocol is a
  datagram to $NOTIFY_SOCKET.
- memory guard: periodically checks RSS; over the guard line it tries
  recovery (gc + pooled-client clear) and exits in a controlled way if the
  memory stays high, trading a ~30s clean restart for an indefinite freeze.
"""
from __future__ import annotations

import asyncio
import gc
import logging
import os
import socket as _socket

logger = logging.getLogger("uniioai_proxy")


def sd_notify(message: str) -> bool:
    """Send one sd_notify datagram to $NOTIFY_SOCKET. Returns False when not
    running under systemd (socket unset) or on any send failure."""
    addr = os.getenv("NOTIFY_SOCKET")
    if not addr:
        return False
    if addr.startswith("@"):
        addr = "\0" + addr[1:]
    try:
        with _socket.socket(_socket.AF_UNIX, _socket.SOCK_DGRAM) as s:
            s.connect(addr)
            s.sendall(message.encode())
        return True
    except Exception as e:  # noqa: BLE001
        logger.debug("sd_notify failed: %s", e)
        return False


async def watchdog_task() -> None:
    """Ping READY once, then WATCHDOG at 1/3 of WatchdogSec (systemd's
    recommended schedule). Only arms when WATCHDOG_USEC is set by systemd."""
    if not os.getenv("NOTIFY_SOCKET"):
        logger.debug("sd_notify: no NOTIFY_SOCKET — watchdog disabled")
        return
    sd_notify("READY=1")
    interval = float(os.getenv("WATCHDOG_USEC", "0")) / 1e6 / 3.0
    if interval <= 0:
        logger.info("sd_notify: READY sent, but no WATCHDOG_USEC — pings disabled")
        return
    logger.info("sd_notify: watchdog armed (ping every %.1fs)", interval)
    while True:
        sd_notify("WATCHDOG=1")
        await asyncio.sleep(interval)


def _current_rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:  # noqa: BLE001
        pass
    import resource
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (ru / 1024.0) if os.uname().sys == "Darwin" else float(ru / 1024.0)


async def memory_guard_task(get_rss_mb=_current_rss_mb, exit_fn=os._exit) -> None:
    """Exit in a controlled way when RSS stays above the guard line.

    Recovery attempt first: gc.collect() + closing the pooled TU clients
    (clear_wedge_state). If RSS is still over the line afterwards, exit(75)
    — Restart=always brings the process back clean instead of letting the
    cgroup reclaim throttle the loop into an unkillable freeze.
    UNIINFER_MEM_GUARD_MB (0/unset = disabled; on amd: 300 against max 350).
    """
    limit = float(os.getenv("UNIINFER_MEM_GUARD_MB", "0") or 0)
    if limit <= 0:
        return
    interval = float(os.getenv("UNIINFER_MEM_GUARD_INTERVAL_S", "60") or 60)
    logger.info("memory guard armed at %.0fMB (interval %.0fs)", limit, interval)
    while True:
        await asyncio.sleep(interval)
        rss = get_rss_mb()
        if rss <= limit:
            continue
        logger.critical("memory guard: RSS %.0fMB > %.0fMB — attempting recovery (gc + pool clear)", rss, limit)
        gc.collect()
        try:
            from uniinfer.providers.tu import clear_wedge_state
            await clear_wedge_state()
        except Exception as e:  # noqa: BLE001
            logger.warning("memory guard: pool clear failed: %s", e)
        await asyncio.sleep(min(5.0, interval))
        rss2 = get_rss_mb()
        if rss2 > limit:
            logger.critical("memory guard: RSS still %.0fMB after recovery — controlled exit for clean restart", rss2)
            exit_fn(75)
            return  # prod os._exit never returns; tests stop the loop here
        else:
            logger.info("memory guard: recovery brought RSS down to %.0fMB", rss2)
