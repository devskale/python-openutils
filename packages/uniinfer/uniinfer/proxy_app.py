from logging.handlers import RotatingFileHandler
import json
import logging
from importlib.metadata import PackageNotFoundError, version
import uuid
import time
import sys
import asyncio
import resource
from fastapi.security import HTTPBearer  # Import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
# _LeanHTTPMiddleware below (also pure ASGI) replaces the former @app.middleware
# ("http") request-logging/size-limit middlewares for the same reason.
import os
from contextlib import asynccontextmanager
import httpx
from dotenv import load_dotenv
import gc
import tracemalloc
from collections import Counter
from sys import getsizeof
from uniinfer.auth import validate_proxy_token

from uniinfer.proxy_routers.models import create_models_router
from uniinfer.proxy_routers.images import create_images_router
from uniinfer.proxy_routers.audio import create_audio_router
from uniinfer.proxy_routers.chat import create_chat_router
from uniinfer.proxy_routers.tools import create_tools_router
from uniinfer.proxy_routers.smoke import create_smoke_router
from uniinfer.proxy_routers.capabilities import create_capabilities_router
from uniinfer.proxy_routers.stats import create_stats_router

# Load environment variables from .env file
load_dotenv()


# --- Setup Logging ---
# Configure root logger to capture logs from all modules (including uniinfer)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Lean logging at throughput: silence per-call / per-stream chatty loggers so a
# high request rate doesn't emit a log record (string format + disk write +
# journald ingest) for every upstream call, every cached-key lookup, and every
# stream start/end. Errors/warnings still surface in full. We keep one concise
# END line per request (see log_requests middleware) plus full ERROR detail.
for _noisy in ("httpx", "credgoo", "uniinfer.completion"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# Create rotating file handler (2MB max, 5 backup files)
# Guard against duplicate handlers on reload/import.
if not root_logger.handlers:
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "uniioai_proxy.log")

    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=2 * 1024 * 1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)

    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(console_handler)

# Proxy specific logger
logger = logging.getLogger("uniioai_proxy")

# Add File, Form, UploadFile
# Add FileResponse and CORSMiddleware imports
# Import run_in_threadpool
# Import HTTPAuthorizationCredentials
try:
    from uniinfer.config.providers import PROVIDER_CONFIGS
except ImportError:
    # Fallback: define minimal configs
    PROVIDER_CONFIGS = {
        'ollama': {'extra_params': {'base_url': 'http://localhost:11434'}},
        'cloudflare': {'extra_params': {}}
    }

# Add the local source directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # /home/ubuntu/code/llmapi
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    UNIINFER_VERSION = version("uniinfer")
except PackageNotFoundError:
    UNIINFER_VERSION = "unknown"

# ---------------------------------------------------------------------------
# Opt-in memory/object tracer — diagnostic for the slow long-run RSS growth.
# OFF by default (zero overhead in production). Enable with UNIINFER_MEM_TRACE=1
# and it appends a JSON line every UNIINFER_MEM_TRACE_INTERVAL s (default 300) to
# logs/mem_trace.log: RSS/VmData, gc object census by type, active asyncio
# tasks. Set UNIINFER_MEM_TRACE_MALLOC=1 to also run tracemalloc and log the top
# growing allocation sites (locates *where* the bytes come from; more overhead).
# Goal: over a long real run, reveal WHICH dimension grows — objects-by-type
# (a real leak to hunt at the source), or flat-objects-rising-RSS (native).
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


async def _mem_trace_loop(_app: FastAPI) -> None:
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
            proc = _read_proc_status()
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
                # top by COUNT diff — surfaces many SMALL allocations (e.g. tuples)
                # that don't show in the size-diff but drive gc.get_objects() growth.
                for stat in sorted(stats, key=lambda s: abs(s.count_diff), reverse=True)[:6]:
                    if stat.count_diff:
                        fr = stat.traceback[0]
                        lg.info(f"  cnt {stat.count_diff:+d}x {fr.filename}:{fr.lineno}")
                prev_snap = snap
        except Exception as e:
            lg.info(f"trace error: {e!r}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App-lifetime shared httpx client.

    One connection pool / TLS context reused across all requests, instead of a
    brand-new AsyncClient per request. The per-request pattern created and
    destroyed a pool on every call — the main source of malloc churn that bloats
    a long-running process's RSS (on amd this accumulated to ~400 MB over 3
    days vs a ~50 MB baseline)."""
    app.state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=10.0),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        http2=True,  # HTTP/2 multiplexing avoids the httpcore #1091 pool leak
    )
    trace_task = None
    if os.getenv("UNIINFER_MEM_TRACE", "") in {"1", "true", "yes"}:
        trace_task = asyncio.create_task(_mem_trace_loop(app))
        logger.info("memory tracer enabled -> logs/mem_trace.log (UNIINFER_MEM_TRACE)")
    try:
        yield
    finally:
        if trace_task is not None:
            trace_task.cancel()
        await app.state.http.aclose()


app = FastAPI(
    title="UniIOAI API",
    description="OpenAI-compatible API wrapper using UniInfer",
    version=UNIINFER_VERSION,
    lifespan=lifespan,
)


# Rate limiting via slowapi was removed: its SlowAPIASGIMiddleware re-sends
# http.response.start on every response-body chunk, which corrupts multi-chunk
# responses (FileResponse / StreamingResponse) — it truncated the webdemo at
# 64KB. It also never fired (storage stayed empty; the real rate limit is
# upstream TU via transparent 429 + bearer auth). Auth + upstream limiting remain.


MAX_REQUEST_SIZE = 10 * 1024 * 1024  # used by _LeanHTTPMiddleware (pure-ASGI size limit)




class _LeanHTTPMiddleware:
    """Pure-ASGI request logging + body-size limit.

    Replaces the two former @app.middleware("http") middlewares. Those (and
    SlowAPIMiddleware) are BaseHTTPMiddleware, which buffers every streaming
    response through an anyio memory stream + background task — under concurrent
    SSE that retains request bodies/parsed JSON and leaks RSS (Starlette #1012).
    This pure-ASGI version streams responses straight through with zero
    buffering, so SSE/concurrent streams can't accumulate.
    """

    def __init__(self, app, max_request_size: int = MAX_REQUEST_SIZE):
        self.app = app
        self.max_request_size = max_request_size

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        # body-size limit via the content-length header (no body read)
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
        # Skip the END line for SSE (stream duration would be misleading), as before.
        ct = resp["ctype"] or ""
        if "text/event-stream" not in ct:
            logger.info("[%s] END %s %s - Status: %s - Duration: %.2fms",
                        request_id, method, path, resp["status"], (time.time() - started) * 1000)


async def _asgi_send_json(send, status_code: int, body: dict) -> None:
    payload = json.dumps(body).encode()
    await send({"type": "http.response.start", "status": status_code,
                "headers": [(b"content-type", b"application/json"),
                             (b"content-length", str(len(payload)).encode())]})
    await send({"type": "http.response.body", "body": payload})


app.add_middleware(_LeanHTTPMiddleware)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", "unknown")
    # Strip the `input` and `ctx` fields from each error and DO NOT echo the
    # full request body. Large chat requests (hundreds of messages) made the
    # old 422 body megabytes-large, which the OpenAI SDK on the client side
    # cannot fold into `error.message` — surfacing as the opaque
    # "422 status code (no body)". Keep the errors actionable but compact.
    compact_errors = [
        {k: v for k, v in err.items() if k not in ("input", "ctx")}
        for err in exc.errors()
    ]
    logger.error(f"[{request_id}] Validation error for {request.method} {request.url}: {compact_errors}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": compact_errors}),
    )

# --- Rate Limit Helpers ---




# --- Add CORS Middleware ---
# Allow requests from any origin for the web demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Middleware for Request Logging and ID ---

# --- Mount Static Files for Web Demo ---
webdemo_dir = os.path.join(script_dir, "examples", "webdemo")
if os.path.isdir(webdemo_dir):
    app.mount("/webdemo", StaticFiles(directory=webdemo_dir),
              name="webdemo_static")

# Define the security scheme
security = HTTPBearer()

# --- Security Dependencies moved to auth.py ---


# --- Model Parsing Helper ---

def parse_provider_model(provider_model: str, allowed_providers: list[str] | None = None, task_name: str | None = None) -> tuple[str, str]:
    """Parse 'provider@model' and optionally validate the provider.

    HTTP-seam adapter over uniinfer.completion.parse_provider_model: translates
    the library ValueError to HTTPException(400), then enforces the HTTP-layer
    allowed-providers constraint.
    """
    from uniinfer.completion import parse_provider_model as _parse

    try:
        provider_name, model_name = _parse(provider_model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if allowed_providers and provider_name not in allowed_providers:
        if len(allowed_providers) == 1:
            prov_list = f"'{allowed_providers[0]}'"
        else:
            prov_list = f"{', '.join([f'\'{p}\'' for p in allowed_providers[:-1]])} and '{
                allowed_providers[-1]}'"

        msg = f"Only {prov_list} provider{'s' if len(allowed_providers) > 1 else ''} supported"
        if task_name:
            msg += f" for {task_name}"
        msg += "."
        raise HTTPException(status_code=400, detail=msg)

    return provider_name, model_name

# --- API Endpoints ---

# --- Add Endpoint to Serve Web Demo HTML ---
@app.get("/webdemo", include_in_schema=False)
async def get_web_demo():
    """Serves the web demo HTML file.

    Cache-busts the bundled CSS/JS by stamping the webdemo dir's newest mtime
    into the `?v=__BUILD__` query params in webdemo.html — so any asset change
    gets a fresh URL automatically (no manual ?v= bumping)."""
    html_file_path = os.path.join(
        script_dir, "examples", "webdemo", "webdemo.html")
    if not os.path.exists(html_file_path):
        raise HTTPException(status_code=404, detail="webdemo.html not found")
    with open(html_file_path, encoding="utf-8") as f:
        html = f.read()
    build = str(int(max(
        os.path.getmtime(os.path.join(webdemo_dir, fn))
        for fn in os.listdir(webdemo_dir)
        if os.path.isfile(os.path.join(webdemo_dir, fn))
    )))
    return HTMLResponse(html.replace("__BUILD__", build))


# --- Performance Dashboard (TTFT / tok/s / caching) ---
# Serves the perf dashboard HTML and reads/writes the same _speed_results.json
# that `uniinfer --speedtest` produces, so CLI and dashboard share one history.
SPEED_RESULTS_PATH = os.path.join(script_dir, "models", "_speed_results.json")
PROBE_RESULTS_PATH = os.path.join(script_dir, "models", "_probe_results.json")


@app.get("/perf", include_in_schema=False)
async def get_perf_dashboard():
    """Serves the LLM performance dashboard (TTFT / tok/s / caching)."""
    html_file_path = os.path.join(script_dir, "examples", "webdemo", "perf.html")
    if not os.path.exists(html_file_path):
        raise HTTPException(status_code=404, detail="perf.html not found")
    return FileResponse(html_file_path)


@app.get("/perf/results", include_in_schema=False)
async def get_perf_results():
    """Returns the saved speed-test history (provider/model -> aggregated metrics)."""
    if not os.path.exists(SPEED_RESULTS_PATH):
        return {}
    try:
        with open(SPEED_RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


@app.post("/perf/results", include_in_schema=False)
async def save_perf_result(request: Request):
    """Saves a live-measured run into the shared history.

    Body: {"key": "tu/qwen-3.6-35b", "result": {...metrics...}}
    Merges into _speed_results.json (same file the CLI writes).
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    key = body.get("key")
    result = body.get("result")
    if not key or not isinstance(result, dict):
        raise HTTPException(status_code=400, detail="Body must contain 'key' and 'result'")

    existing = {}
    if os.path.exists(SPEED_RESULTS_PATH):
        try:
            with open(SPEED_RESULTS_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing[key] = result
    os.makedirs(os.path.dirname(SPEED_RESULTS_PATH), exist_ok=True)
    with open(SPEED_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    return {"ok": True, "saved": key}


# --- Capability-probe dashboard + integration guide ---
@app.get("/capabilities", include_in_schema=False)
async def get_capabilities_dashboard():
    """Serves the capability-probe dashboard HTML."""
    html_file_path = os.path.join(script_dir, "examples", "webdemo", "capabilities.html")
    if not os.path.exists(html_file_path):
        raise HTTPException(status_code=404, detail="capabilities.html not found")
    return FileResponse(html_file_path)


@app.get("/capabilities/results", include_in_schema=False)
async def get_capabilities_results():
    """Returns the saved capability-probe history (provider/model -> matrix)."""
    if not os.path.exists(PROBE_RESULTS_PATH):
        return {}
    try:
        with open(PROBE_RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


@app.get("/guide", include_in_schema=False)
async def get_integration_guide():
    """Serves the integration-guide page (renders docs/integration.md)."""
    html_file_path = os.path.join(script_dir, "examples", "webdemo", "guide.html")
    if not os.path.exists(html_file_path):
        raise HTTPException(status_code=404, detail="guide.html not found")
    return FileResponse(html_file_path)


@app.get("/guide.md", include_in_schema=False)
async def get_integration_guide_md():
    """Serves the canonical integration guide markdown (single source of truth)."""
    md_file_path = os.path.join(script_dir, "..", "docs", "integration.md")
    if not os.path.exists(md_file_path):
        raise HTTPException(status_code=404, detail="integration.md not found")
    return FileResponse(md_file_path, media_type="text/markdown")


app.include_router(create_tools_router())
app.include_router(create_models_router(UNIINFER_VERSION))
app.include_router(create_smoke_router())
app.include_router(create_capabilities_router(parse_provider_model=parse_provider_model, provider_configs=PROVIDER_CONFIGS))
app.include_router(create_stats_router())
app.include_router(create_images_router(parse_provider_model))
app.include_router(create_audio_router(parse_provider_model))


app.include_router(
    create_chat_router(
        parse_provider_model=parse_provider_model,
        provider_configs=PROVIDER_CONFIGS,
    )
)


@app.get("/", include_in_schema=False)
async def root():
    """Serve the unified web app (login-gated Chat / Models / Images / Audio)."""
    html = os.path.join(script_dir, "examples", "webdemo", "webdemo.html")
    if not os.path.exists(html):
        raise HTTPException(status_code=404, detail="webdemo.html not found")
    return FileResponse(html)


_HEALTH_START = time.time()


def _read_proc_status():
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


def _cgroup_mem():
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


def _jemalloc_stats() -> dict:
    """Best-effort live jemalloc allocator stats via mallctl (only meaningful
    when jemalloc is LD_PRELOADed). Distinguishes a live native leak
    (stats.allocated grows) from allocator retention (allocated flat but
    resident/RSS grows — decayed pages not yet returned). Returns {} if
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
                out[k.rsplit(".", 1)[1]] = round(v / (1024 * 1024), 1)  # MB
        return out
    except Exception:
        return {}


@app.get("/health", include_in_schema=False)
async def health(request: Request):
    """Lightweight health probe (no auth, no upstream cost). Surfaces the
    signals that matter for this memory-constrained proxy — memory vs the
    cgroup cap, swap, page-fault rate, allocator, and event-loop lag — with a
    computed ok/warn/crit status.  curl http://host:port/health"""
    # event-loop responsiveness (elevated when the loop is blocked/thrashing)
    _t0 = time.monotonic()
    await asyncio.sleep(0)
    loop_ms = (time.monotonic() - _t0) * 1000

    # memory
    rss_kb = peak_kb = swap_kb = None
    proc = _read_proc_status()
    if proc and proc[0] is not None:
        rss_kb, peak_kb, swap_kb = proc
    else:
        # non-Linux fallback (mac): ru_maxrss is bytes on Darwin, KB on Linux
        _ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_kb = _ru // 1024 if sys.platform == "darwin" else _ru
        rss_kb = peak_kb
    rss_mb = rss_kb // 1024 if rss_kb else None

    cg = _cgroup_mem()
    # Caps: cgroup memory.high/max if present, else env overrides (lets non-cgroup
    # hosts — e.g. macOS, or a parity stand-in — report the same limits / headroom
    # as the constrained deployment). UNIINFER_MEM_HIGH_MB / UNIINFER_MEM_MAX_MB.
    def _env_mb(name: str) -> int | None:
        v = os.getenv(name)
        return int(v) if v else None
    high_mb = ((cg["memory.high"] // (1024 * 1024)) if cg.get("memory.high") else None) or _env_mb("UNIINFER_MEM_HIGH_MB")
    max_mb = ((cg["memory.max"] // (1024 * 1024)) if cg.get("memory.max") else None) or _env_mb("UNIINFER_MEM_MAX_MB")
    cur_mb = ((cg["memory.current"] // (1024 * 1024)) if cg.get("memory.current") else None) or rss_mb
    headroom_mb = (high_mb - rss_mb) if (high_mb and rss_mb) else None

    # page-fault rate since the previous /health call
    majflt_now = None
    try:
        with open("/proc/self/stat") as f:
            majflt_now = int(f.read().split()[11])  # field 12 = majflt
    except Exception:
        pass
    last = getattr(request.app.state, "_health_last", None)
    majflt_per_s = None
    if majflt_now is not None and last and last.get("majflt") is not None:
        dt = time.monotonic() - last["t"]
        if dt > 0:
            majflt_per_s = (majflt_now - last["majflt"]) / dt
    request.app.state._health_last = {"majflt": majflt_now, "t": time.monotonic()}

    ld = os.environ.get("LD_PRELOAD", "")
    allocator = {
        "jemalloc": "jemalloc" in ld,
        "malloc_arena_max": os.environ.get("MALLOC_ARENA_MAX"),
        "malloc_mmap_threshold": os.environ.get("MALLOC_MMAP_THRESHOLD_"),
    }
    # live jemalloc split — the key diagnostic for WHICH kind of native growth:
    #  live_mb  grows   = real in-use native allocations (leak to hunt)
    #  retained_mb grows = decayed pages jemalloc hasn't returned (allocator)
    jstats = _jemalloc_stats()
    if jstats.get("allocated") is not None:
        allocator["live_allocated_mb"] = jstats.get("allocated")
        allocator["active_mb"] = jstats.get("active")
        allocator["resident_mb"] = jstats.get("resident")
        allocator["retained_mb"] = jstats.get("retained")
        allocator["unreturned_mb"] = round(max(0.0, (jstats.get("resident") or 0)
                                               - (jstats.get("allocated") or 0)), 1)

    # status thresholds tuned to the unit's 200M soft / 300M hard caps
    status = "ok"
    if (swap_kb and swap_kb > 0) or (majflt_per_s and majflt_per_s > 50) \
            or loop_ms > 1000 or (max_mb and rss_mb and rss_mb >= 0.95 * max_mb):
        status = "crit"
    elif (rss_mb and high_mb and rss_mb >= 0.85 * high_mb) \
            or (majflt_per_s and majflt_per_s > 10) or loop_ms > 100:
        status = "warn"

    return {
        "status": status,
        "version": UNIINFER_VERSION,
        "uptime_seconds": int(time.time() - _HEALTH_START),
        "event_loop_latency_ms": round(loop_ms, 2),
        "memory": {
            "rss_mb": rss_mb,
            "peak_mb": (peak_kb // 1024) if peak_kb else None,
            "swap_kb": swap_kb,
            "cgroup_mb": {"current": cur_mb, "high": high_mb, "max": max_mb},
            "headroom_mb": headroom_mb,
            "majflt_per_s": round(majflt_per_s, 1) if majflt_per_s is not None else None,
        },
        "allocator": allocator,
    }


@app.get("/debug/mem", include_in_schema=False)
async def debug_mem(api_bearer_token: str = Depends(validate_proxy_token)):
    """Live object census by type (on-demand, low overhead). Diagnostic for
    memory investigations — diff counts between two calls to spot accumulation."""
    gc.collect()
    by_count: Counter = Counter()
    by_size: Counter = Counter()
    total = 0
    for o in gc.get_objects():
        t = type(o).__name__
        by_count[t] += 1
        try:
            by_size[t] += getsizeof(o)
        except Exception:
            pass
        total += 1
    return {
        "total_objects": total,
        "top_by_count": by_count.most_common(20),
        "top_by_size_kb": [(t, round(s / 1024)) for t, s in by_size.most_common(20)],
    }


# --- Run the API (for local development) ---
def main():
    import uvicorn
    import argparse
    from importlib.metadata import version

    parser = argparse.ArgumentParser(description="Run the UniIOAI API server.")
    parser.add_argument('--version', action='version',
                        version='%(prog)s ' + version('uniinfer'),
                        help="Show program's version number and exit")
    parser.add_argument('--reload', action='store_true',
                        help='Enable auto-reloading')
    parser.add_argument('--port', type=int, default=8123,
                        help='Port to run the server on')

    args = parser.parse_args()

    logger.info(
        f"Starting UniIOAI API server (reload={args.reload} at port {args.port})..."
    )
    uvicorn.run(
        "uniinfer.proxy_app:app",
        host="0.0.0.0",
        port=args.port,
        workers=1,
        reload=args.reload,
        access_log=False,
    )


if __name__ == "__main__":
    main()
    # Example curl commands:
    # List models:
    # curl http://localhost:8123/v1/models

    # Non-streaming (replace YOUR_API_TOKEN):
    # curl -X POST http://localhost:8123/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer YOUR_API_TOKEN" -d '{"model": "groq@llama3-8b-8192", "messages": [{"role": "user", "content": "Say hello!"}], "stream": false}'
    # Non-streaming with base_url (e.g., for Ollama):
    # curl -X POST http://localhost:8123/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer YOUR_API_TOKEN_OR_CREDGOO_COMBO" -d '{"model": "ollama@llama3", "messages": [{"role": "user", "content": "Say hello!"}], "stream": false, "base_url": "http://localhost:11434"}'

    # Streaming (replace YOUR_API_TOKEN):
    # curl -N -X POST http://localhost:8123/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer YOUR_API_TOKEN" -d '{"model": "groq@llama3-8b-8192", "messages": [{"role": "user", "content": "Tell me a short story about a robot learning to paint."}], "stream": true}'
    # Streaming with base_url (e.g., for Ollama):
    # curl -N -X POST http://localhost:8123/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer YOUR_API_TOKEN_OR_CREDGOO_COMBO" -d '{"model": "ollama@llama3", "messages": [{"role": "user", "content": "Tell me a short story."}], "stream": true, "base_url": "http://localhost:11434"}'
