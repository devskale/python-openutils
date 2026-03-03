from logging.handlers import RotatingFileHandler
import logging
from importlib.metadata import PackageNotFoundError, version
import uuid
import time
import sys
from fastapi.security import HTTPBearer  # Import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import os
from dotenv import load_dotenv

from uniinfer.proxy_routers.models import create_models_router
from uniinfer.proxy_routers.media import create_media_router
from uniinfer.proxy_routers.chat import create_chat_router

# Load environment variables from .env file
load_dotenv()


# --- Setup Logging ---
# Configure root logger to capture logs from all modules (including uniinfer)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

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

app = FastAPI(
    title="UniIOAI API",
    description="OpenAI-compatible API wrapper using UniInfer",
    version=UNIINFER_VERSION,
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
    logger.error(f"[{request_id}] Validation error for {request.method} {request.url}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
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

    logger.info(f"[{request_id}] START {request.method} {request.url}")

    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        logger.info(
            f"[{request_id}] END {request.method} {request.url} - Status: {response.status_code} - Duration: {process_time:.2f}ms")
        response.headers["X-Request-ID"] = request_id
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
    """
    Parses 'provider@model' string and optionally validates the provider.
    Raises HTTPException (400) if format is invalid or provider is not allowed.
    """
    if '@' not in provider_model:
        raise HTTPException(
            status_code=400, detail="Invalid model format. Expected 'provider@modelname'.")

    parts = provider_model.split('@', 1)
    provider_name = parts[0]
    model_name = parts[1]

    if not provider_name or not model_name:
        raise HTTPException(
            status_code=400, detail="Invalid model format. Provider or model name is empty.")

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
    """Serves the web demo HTML file."""
    # Serve webdemo.html as the default file for /webdemo/
    html_file_path = os.path.join(
        script_dir, "examples", "webdemo", "webdemo.html")
    if not os.path.exists(html_file_path):
        raise HTTPException(status_code=404, detail="webdemo.html not found")
    return FileResponse(html_file_path)


app.include_router(create_models_router(UNIINFER_VERSION))
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


@app.get("/")
async def root():
    return {"message": "UniIOAI API is running. Visit /webdemo or /webdemo/webdemo.html for the interactive demo, or use POST /v1/chat/completions, POST /v1/embeddings, or GET /v1/models"}


# --- Run the API (for local development) ---
def main():
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="Run the UniIOAI API server.")
    parser.add_argument('--reload', action='store_true',
                        help='Enable auto-reloading')
    parser.add_argument('--port', type=int, default=8123,
                        help='Port to run the server on')

    args = parser.parse_args()

    logger.info(
        f"Starting UniIOAI API server (reload={args.reload} at port {args.port})..."
    )
    uvicorn.run(
        "uniinfer.uniioai_proxy:app",
        host="0.0.0.0",
        port=args.port,
        workers=1,
        reload=args.reload
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
