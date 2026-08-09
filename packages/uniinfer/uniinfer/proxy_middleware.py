"""ASGI plumbing for the uniioai proxy: request middleware, memory tracer, and
system-introspection helpers (/proc, cgroup, jemalloc).

Split from proxy_app.py for locality: the middleware + tracer + system readers
are ASGI-layer concerns, distinct from the app factory, routes, and business
logic. proxy_app.py imports and wires them.
"""
from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import time
import tracemalloc
import uuid
from collections import Counter
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("uniioai_proxy")

MAX_REQUEST_SIZE = 10 * 1024 * 1024


# ---------------------------------------------------------------------------
# System introspection — used by /health + the tracer.
# ---------------------------------------------------------------------------
def read_proc_status():
    """Return (rss_kb, peak_kb, swap_kb) from /proc/self/status (Linux), else None."""
    try:
        rss = peak = swap = None
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
                elif line.startswith("VmPeak:"):
                    peak = int(line.split()[1])
                elif line.startswith("VmSwap:"):
                    swap = int(line.split()[1])
        return rss, peak, swap
    except FileNotFoundError:
        return None


def cgroup_mem():
    """Best-effort cgroup v2 memory.current/high/max in bytes (None if unavailable)."""
    try:
        with open("/proc/self/cgroup") as f:
            path = f.read().strip().splitlines()[0].split(":", 2)[2]
        base = f"/sys/fs/cgroup{path}"
    except Exception:
        return {}
    out = {}
    for key in ("memory.current", "memory.high", "memory.max"):
        try:
            with open(f"{base}/{key}") as f:
                v = f.read().strip()
            out[key] = None if v == "max" else int(v)
        except Exception:
            out[key] = None
    return out


def jemalloc_stats() -> dict:
    """Best-effort live jemalloc allocator stats via mallctl. Returns {} if
    jemalloc/mallctl is unavailable (e.g. mac, or a non-jemalloc allocator)."""
    try:
        import ctypes
        lib = ctypes.CDLL("libjemalloc.so.2")
        lib.mallctl.argtypes = [ctypes.c_char_p, ctypes.c_void_p,
                                ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p,
                                ctypes.c_size_t]

        def read_u64(key: str):
            buf = ctypes.c_uint64(0)
            sz = ctypes.c_size_t(8)
            if lib.mallctl(key.encode(), ctypes.byref(buf), ctypes.byref(sz), None, 0) == 0:
                return buf.value
            return None

        out = {}
        for k in ("stats.allocated", "stats.active", "stats.resident", "stats.retained"):
            v = read_u64(k)
            if v is not None:
                out[k.rsplit(".", 1)[1]] = round(v / (1024 * 1024), 1)
        return out
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Opt-in memory/object tracer — diagnostic for slow long-run RSS growth.
# OFF by default (zero overhead). Enable with UNIINFER_MEM_TRACE=1.
# ---------------------------------------------------------------------------
_MEM_TRACE_PATH = os.path.join(os.getcwd(), "logs", "mem_trace.log")


def _mem_trace_logger() -> logging.Logger:
    lg = logging.getLogger("uniinfer.memtrace")
    if not lg.handlers:
        try:
            os.makedirs(os.path.dirname(_MEM_TRACE_PATH), exist_ok=True)
            h = RotatingFileHandler(_MEM_TRACE_PATH, maxBytes=2_000_000, backupCount=2)
            h.setFormatter(logging.Formatter("%(message)s"))
            lg.addHandler(h)
            lg.setLevel(logging.INFO)
            lg.propagate = False
        except Exception:
            pass
    return lg


async def mem_trace_loop(_app) -> None:
    """Periodically log the memory growth dimensions. See module comment."""
    lg = _mem_trace_logger()
    interval = float(os.getenv("UNIINFER_MEM_TRACE_INTERVAL", "300"))
    use_malloc = os.getenv("UNIINFER_MEM_TRACE_MALLOC", "") in {"1", "true", "yes"}
    if use_malloc:
        tracemalloc.start(25)
    prev_snap = tracemalloc.take_snapshot() if use_malloc else None
    t0 = time.time()
    lg.info(f"# mem-tracer start malloc={use_malloc} interval={interval}s pid={os.getpid()}")
    while True:
        await asyncio.sleep(interval)
        try:
            data_kb = None
            proc = read_proc_status()
            rss_kb = proc[0] if (proc and proc[0] is not None) else None
            try:
                with open("/proc/self/status") as f:
                    for line in f:
                        if line.startswith("VmData:"):
                            data_kb = int(line.split()[1]); break
            except Exception:
                pass
            gc.collect()
            by = Counter(type(o).__name__ for o in gc.get_objects())
            total = sum(by.values())
            try:
                tasks = len(asyncio.all_tasks())
            except Exception:
                tasks = None
            line = {"t": time.strftime("%H:%M:%S"), "up_s": int(time.time() - t0),
                    "rss_kb": rss_kb, "data_kb": data_kb, "objs": total,
                    "tasks": tasks, "top": dict(by.most_common(10))}
            lg.info(json.dumps(line))
            if use_malloc:
                snap = tracemalloc.take_snapshot()
                stats = snap.compare_to(prev_snap, "lineno")
                for stat in sorted(stats, key=lambda s: abs(s.size_diff), reverse=True)[:6]:
                    if stat.size_diff:
                        fr = stat.traceback[0]
                        lg.info(f"  diff {stat.size_diff//1024:+d}KB {fr.filename}:{fr.lineno}")
                for stat in sorted(stats, key=lambda s: abs(s.count_diff), reverse=True)[:6]:
                    if stat.count_diff:
                        fr = stat.traceback[0]
                        lg.info(f"  cnt {stat.count_diff:+d}x {fr.filename}:{fr.lineno}")
                prev_snap = snap
        except Exception as e:
            lg.info(f"trace error: {e!r}")


# ---------------------------------------------------------------------------
# Pure-ASGI request middleware — replaces the former BaseHTTPMiddleware that
# leaked under streaming (Starlette #1012). Streams responses straight through
# with zero buffering; adds request-id logging + body-size limit.
# ---------------------------------------------------------------------------
async def _asgi_send_json(send, status_code: int, body: dict) -> None:
    payload = json.dumps(body).encode()
    await send({"type": "http.response.start", "status": status_code,
                "headers": [(b"content-type", b"application/json"),
                             (b"content-length", str(len(payload)).encode())]})
    await send({"type": "http.response.body", "body": payload})


class LeanHTTPMiddleware:
    """Pure-ASGI request logging + body-size limit. No buffering, no leak."""

    def __init__(self, app, max_request_size: int = MAX_REQUEST_SIZE):
        self.app = app
        self.max_request_size = max_request_size

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        for k, v in scope.get("headers", []):
            if k == b"content-length":
                try:
                    if int(v) > self.max_request_size:
                        await _asgi_send_json(send, 413, {"detail": "Request too large"})
                        return
                except ValueError:
                    pass
                break
        request_id = str(uuid.uuid4())
        st = scope.setdefault("state", {})
        if isinstance(st, dict):
            st["request_id"] = request_id
        method, path = scope.get("method"), scope.get("path")
        logger.debug("[%s] START %s %s", request_id, method, path)
        started = time.time()
        resp = {"status": None, "ctype": None}

        async def send_wrap(message):
            if message["type"] == "http.response.start":
                resp["status"] = message["status"]
                hs = list(message.get("headers", []))
                hs.append((b"x-request-id", request_id.encode()))
                for k, v in message.get("headers", []):
                    if k == b"content-type":
                        resp["ctype"] = v.decode("latin-1", "replace"); break
                message["headers"] = hs
            await send(message)

        try:
            await self.app(scope, receive, send_wrap)
        except Exception as e:
            logger.error("[%s] ERROR %s %s - %.2fms - %s", request_id, method, path,
                         (time.time() - started) * 1000, e)
            raise
        ct = resp["ctype"] or ""
        if "text/event-stream" not in ct:
            logger.info("[%s] END %s %s - Status: %s - Duration: %.2fms",
                        request_id, method, path, resp["status"], (time.time() - started) * 1000)
