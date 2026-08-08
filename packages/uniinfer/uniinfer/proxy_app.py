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
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import os
from contextlib import asynccontextmanager
import httpx
from dotenv import load_dotenv
import gc
from collections import Counter
from sys import getsizeof
from uniinfer.auth import validate_proxy_token

from uniinfer.proxy_routers.models import create_models_router
from uniinfer.proxy_routers.media import create_media_router
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
    )
    try:
        yield
    finally:
        await app.state.http.aclose()


app = FastAPI(
    title="UniIOAI API",
    description="OpenAI-compatible API wrapper using UniInfer",
    version=UNIINFER_VERSION,
    lifespan=lifespan,
)


# --- Rate Limiting Setup ---
# Enable headers to let clients know their limits
limiter = Limiter(key_func=get_remote_address, headers_enabled=True)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

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


def get_chat_rate_limit():
    return os.getenv("UNIINFER_RATE_LIMIT_CHAT", "100/minute")


def get_embeddings_rate_limit():
    return os.getenv("UNIINFER_RATE_LIMIT_EMBEDDINGS", "200/minute")


def get_media_rate_limit():
    return os.getenv("UNIINFER_RATE_LIMIT_MEDIA", "50/minute")


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

MAX_REQUEST_SIZE = 10 * 1024 * 1024

@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    # print(f"DEBUG: Middleware Content-Length: {content_length}")
    if content_length:
        content_length = int(content_length)
        if content_length > MAX_REQUEST_SIZE:
            # print("DEBUG: Request too large")
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"detail": "Request too large"}
            )
    return await call_next(request)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Store request_id in request state for access in endpoints
    request.state.request_id = request_id

    logger.debug(f"[{request_id}] START {request.method} {request.url}")

    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        # For streaming (SSE) responses the body hasn't flowed yet at this point,
        # so process_time would be just the response *setup* time (misleadingly
        # tiny). Skip the END line for those; non-streaming + errors still log.
        # BaseHTTPMiddleware wraps the response, so isinstance/media_type are
        # unreliable here — the content-type header is preserved and reliable.
        if "text/event-stream" not in response.headers.get("content-type", ""):
            process_time = (time.time() - start_time) * 1000
            logger.info(
                f"[{request_id}] END {request.method} {request.url} - Status: {response.status_code} - Duration: {process_time:.2f}ms")
        return response
    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        logger.error(
            f"[{request_id}] ERROR {request.method} {request.url} - Duration: {process_time:.2f}ms - Exception: {e}")
        raise

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
app.include_router(create_media_router(parse_provider_model, limiter, get_media_rate_limit))


app.include_router(
    create_chat_router(
        parse_provider_model=parse_provider_model,
        provider_configs=PROVIDER_CONFIGS,
        limiter=limiter,
        get_chat_rate_limit=get_chat_rate_limit,
        get_embeddings_rate_limit=get_embeddings_rate_limit,
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
    high_mb = (cg["memory.high"] // (1024 * 1024)) if cg.get("memory.high") else None
    max_mb = (cg["memory.max"] // (1024 * 1024)) if cg.get("memory.max") else None
    cur_mb = (cg["memory.current"] // (1024 * 1024)) if cg.get("memory.current") else None
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
