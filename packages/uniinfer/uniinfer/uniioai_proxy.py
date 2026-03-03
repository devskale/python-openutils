from logging.handlers import RotatingFileHandler
import logging
from typing import Any, AsyncGenerator
from importlib.metadata import PackageNotFoundError, version
import re
import uuid
import json
import time
import sys
import asyncio
from fastapi.security import HTTPBearer  # Import HTTPBearer
from starlette.concurrency import iterate_in_threadpool
from pydantic import BaseModel, Field, field_validator
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
    from uniinfer.examples.providers_config import PROVIDER_CONFIGS
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
    from uniinfer.uniioai import stream_completion, astream_completion
    from uniinfer.errors import UniInferError, ProviderError
except ImportError as e:
    logger.error(f"Error importing from uniinfer.uniioai: {e}")
    logger.error(f"Python path: {sys.path[:3]}")
    sys.exit(1)

try:
    UNIINFER_VERSION = version("uniinfer")
except PackageNotFoundError:
    UNIINFER_VERSION = "unknown"

app = FastAPI(
    title="UniIOAI API",
    description="OpenAI-compatible API wrapper using UniInfer",
    version=UNIINFER_VERSION,
)


def _is_openai_strict_mode() -> bool:
    """Return False to preserve TU/vLLM-style reasoning fields (reasoning_content)."""
    return False


def _normalize_nonstream_content(content: Any, tool_calls: Any) -> str | None:
    """Normalize assistant content for stricter OpenAI compatibility.

    If not a tool call and content is missing, return an empty string instead of null.
    """
    if tool_calls:
        return content
    if content is None and _is_openai_strict_mode():
        return ""
    return content

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

# --- Pydantic Models for OpenAI Compatibility ---


class ChatMessageInput(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequestInput(BaseModel):
    model: str
    messages: list[ChatMessageInput]
    temperature: float | None = 0.7
    max_tokens: int | None = 32768
    max_completion_tokens: int | None = None
    stream: bool | None = False
    base_url: str | None = None
    tools: list[dict] | None = None
    tool_choice: Any | None = None
    reasoning_effort: str | None = None

    def get_effective_max_tokens(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens

    @field_validator('messages')
    @classmethod
    def validate_messages_count(cls, v):
        # print(f"DEBUG: Validating messages count: {len(v)}")
        if len(v) > 500:
            raise ValueError('Too many messages. Maximum allowed is 500.')
        return v

    @field_validator('model')
    @classmethod
    def validate_model_format(cls, v):
        if '@' not in v:
             raise ValueError("Invalid model format. Expected 'provider@modelname'.")
        if not re.match(r'^[^@]+@[^@]+$', v):
             raise ValueError("Incorrect model format. Exactly one '@' expected.")
        return v


class ChatMessageOutput(BaseModel):
    role: str
    content: str | None = None
    reasoning_content: str | None = None  # Add support for thinking/reasoning
    tool_calls: list[dict] | None = None


class ChoiceDelta(BaseModel):
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None  # Add support for thinking/reasoning
    tool_calls: list[dict] | None = None


class StreamingChoice(BaseModel):
    index: int = 0
    delta: ChoiceDelta
    finish_reason: str | None = None


class StreamingChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[StreamingChoice]


class NonStreamingChoice(BaseModel):
    index: int = 0
    message: ChatMessageOutput
    finish_reason: str = "stop"  # Assuming stop, uniinfer might not provide this


class NonStreamingChatCompletion(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[NonStreamingChoice]
    # uniinfer doesn't provide usage yet
    usage: dict[str, int] | None = None


# --- Models for /v1/models endpoint ---

class Model(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "skaledev"  # Or determine dynamically if needed


class ModelList(BaseModel):
    object: str = "list"
    data: list[Model]


# --- Add ProviderList model for OpenAI‐style list response ---
class ProviderList(BaseModel):
    object: str = "list"
    data: list[str]


# --- Models for Embedding Endpoint ---

class EmbeddingRequest(BaseModel):
    model: str  # Expected format: "provider@modelname"
    input: list[str]  # List of texts to embed
    user: str | None = None  # Optional user identifier


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: dict[str, int] | None = None


# --- Helper Functions ---

# Update signature: remove api_bearer_token, add provider_api_key
async def stream_response_generator(messages: list[dict], provider_model: str, temp: float, max_tok: int, provider_api_key: str | None, base_url: str | None, tools: list[dict] | None = None, tool_choice: Any | None = None) -> AsyncGenerator[str, None]:
    """Generates OpenAI-compatible SSE chunks from uniioai.stream_completion using a thread pool."""
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    model_name = provider_model

    # First chunk sends the role
    first_chunk_data = StreamingChatCompletionChunk(
        id=completion_id,
        created=created_time,
        model=model_name,
        choices=[StreamingChoice(delta=ChoiceDelta(role="assistant"))]
    )
    yield f"data: {first_chunk_data.model_dump_json(exclude_none=True)}\n\n"

    seen_tool_calls = False
    sent_finish_reason = False

    try:
        # Create the synchronous generator instance, passing the retrieved key
        sync_generator = stream_completion(
            # Pass provider_api_key
            messages, provider_model, temp, max_tok, provider_api_key=provider_api_key, base_url=base_url, tools=tools, tool_choice=tool_choice)

        # Iterate over the synchronous generator in a thread pool
        iterator_obj = iterate_in_threadpool(sync_generator)
        async for chunk in iterator_obj:
            # Check for finish_reason in the chunk (if supported by provider)
            chunk_finish_reason = getattr(chunk, 'finish_reason', None)

            if chunk.message:
                delta_kwargs = {}
                if chunk.message.content:
                    delta_kwargs["content"] = chunk.message.content
                if chunk.thinking and not _is_openai_strict_mode():
                    delta_kwargs["reasoning_content"] = chunk.thinking
                if chunk.message.tool_calls:
                    delta_kwargs["tool_calls"] = chunk.message.tool_calls
                    seen_tool_calls = True

                # Prepare choice arguments
                choice_kwargs = {}

                # If we have content or tools, add them to delta
                if delta_kwargs:
                    choice_kwargs['delta'] = ChoiceDelta(**delta_kwargs)
                else:
                    # If no content/tools, but we have finish_reason, we still need a delta object (empty)
                    choice_kwargs['delta'] = ChoiceDelta()

                # If this chunk has finish_reason, include it
                if chunk_finish_reason:
                    choice_kwargs['finish_reason'] = chunk_finish_reason
                    sent_finish_reason = True

                # Only yield if there's delta content OR finish_reason
                if delta_kwargs or chunk_finish_reason:
                    chunk_data = StreamingChatCompletionChunk(
                        id=completion_id,
                        created=created_time,
                        model=model_name,
                        choices=[StreamingChoice(**choice_kwargs)]
                    )
                    json_data = chunk_data.model_dump_json(exclude_none=True)
                    logger.debug(f"Yielding chunk: {json_data}")
                    yield f"data: {json_data}\n\n"

        # Last chunk signals completion if not already sent
        if not sent_finish_reason:
            finish_reason = "tool_calls" if seen_tool_calls else "stop"
            logger.info(
                f"Stream finished. Fallback finish_reason: {finish_reason}")

            final_chunk_data = StreamingChatCompletionChunk(
                id=completion_id,
                created=created_time,
                model=model_name,
                choices=[StreamingChoice(
                    delta=ChoiceDelta(), finish_reason=finish_reason)]
            )
            yield f"data: {final_chunk_data.model_dump_json(exclude_none=True)}\n\n"

    except NameError as e:
        # Specific catch for missing 'payload' or similar undefined names
        logger.error(f"NameError during streaming: {e}")
        error_chunk = {
            "error": {
                "message": f"Stream internal error: {e}",
                "type": "NameError",
                "code": None
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
    except (UniInferError, ValueError) as e:
        logger.error(f"Error during streaming: {e}")
        status_code = getattr(e, 'status_code', None)
        message = str(e)
        if isinstance(e, ProviderError) and e.response_body:
            message = f"{message} | Provider Response: {e.response_body}"

        error_chunk = {
            "error": {
                "message": message,
                "type": type(e).__name__,
                "code": status_code
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
    except Exception as e:
        logger.exception(f"Unexpected error during streaming: {e}")
        error_chunk = {
            "error": {
                "message": f"Unexpected server error: {type(e).__name__}",
                "type": "internal_server_error",
                "code": None
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

    yield "data: [DONE]\n\n"


# --- Async Stream Response Generator ---
async def astream_response_generator(messages: list[dict], provider_model: str, temp: float, max_tok: int, provider_api_key: str | None, base_url: str | None, tools: list[dict] | None = None, tool_choice: Any | None = None, request_id: str | None = None, reasoning_effort: str | None = None) -> AsyncGenerator[str, None]:
    """Generates OpenAI-compatible SSE chunks from uniioai.astream_completion using async."""
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    model_name = provider_model

    stream_label = f"[{request_id}] " if request_id else ""
    logger.info(f"{stream_label}Async stream start for {model_name}")
    chunk_count = 0
    heartbeat_count = 0
    last_yield_time = time.monotonic()
    idle_warning_threshold = 10.0
    heartbeat_interval = float(os.getenv("UNIINFER_STREAM_HEARTBEAT", "5"))
    idle_timeout = float(os.getenv("UNIINFER_STREAM_IDLE_TIMEOUT", "30"))

    first_chunk_data = StreamingChatCompletionChunk(
        id=completion_id,
        created=created_time,
        model=model_name,
        choices=[StreamingChoice(delta=ChoiceDelta(role="assistant"))]
    )
    yield f"data: {first_chunk_data.model_dump_json(exclude_none=True)}\n\n"

    seen_tool_calls = False
    sent_finish_reason = False

    try:
        async_iter = astream_completion(
            messages, provider_model, temp, max_tok, provider_api_key=provider_api_key, base_url=base_url, tools=tools, tool_choice=tool_choice, reasoning_effort=reasoning_effort
        ).__aiter__()
        while True:
            try:
                chunk = await asyncio.wait_for(async_iter.__anext__(), timeout=heartbeat_interval)
            except asyncio.TimeoutError:
                now = time.monotonic()
                idle_for = now - last_yield_time
                if idle_for >= idle_timeout:
                    logger.warning(
                        f"{stream_label}Async stream idle timeout after {idle_for:.2f}s for {model_name}")
                    error_chunk = {
                        "error": {
                            "message": f"Stream idle timeout after {idle_for:.2f}s",
                            "type": "stream_timeout",
                            "code": None
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    sent_finish_reason = True
                    break

                heartbeat_count += 1
                logger.debug(
                    f"{stream_label}Async stream heartbeat {heartbeat_count} for {model_name} (idle {idle_for:.2f}s)")
                yield ": keep-alive\n\n"
                continue
            except StopAsyncIteration:
                break

            if isinstance(chunk, dict):
                for choice in chunk.get('choices', []):
                    delta = choice.get('delta', {}) or {}
                    if not isinstance(delta, dict):
                        continue

                    if _is_openai_strict_mode():
                        delta.pop('reasoning_content', None)
                        delta.pop('thinking', None)
                    else:
                        # Unify to TU/vLLM-style field expected by pi clients.
                        if delta.get('thinking') and not delta.get('reasoning_content'):
                            delta['reasoning_content'] = delta['thinking']
                        # Emit only reasoning_content in non-strict mode for consistency.
                        delta.pop('thinking', None)

                json_data = json.dumps(chunk)
                logger.debug(f"Yielding async chunk (dict): {json_data}")
                yield f"data: {json_data}\n\n"
                chunk_count += 1
                now = time.monotonic()
                if now - last_yield_time > idle_warning_threshold:
                    logger.warning(
                        f"{stream_label}Async stream idle gap {now - last_yield_time:.2f}s for {model_name}")
                last_yield_time = now
                
                choice = chunk.get('choices', [{}])[0]
                if choice.get('finish_reason'):
                    sent_finish_reason = True
                delta = choice.get('delta', {})
                if delta.get('tool_calls'):
                    seen_tool_calls = True
            else:
                chunk_finish_reason = getattr(chunk, 'finish_reason', None)

                if chunk.message:
                    choice_kwargs = {}
                    if chunk.message.content:
                        choice_kwargs['content'] = chunk.message.content
                    if chunk.thinking and not _is_openai_strict_mode():
                        choice_kwargs['reasoning_content'] = chunk.thinking
                    if chunk.message.tool_calls:
                        choice_kwargs['tool_calls'] = chunk.message.tool_calls
                        seen_tool_calls = True

                    choice_kwargs_with_delta = {}
                    if choice_kwargs:
                        choice_kwargs_with_delta['delta'] = ChoiceDelta(**choice_kwargs)
                    else:
                        choice_kwargs_with_delta['delta'] = ChoiceDelta()

                    if chunk_finish_reason:
                        choice_kwargs_with_delta['finish_reason'] = chunk_finish_reason
                        sent_finish_reason = True

                    if choice_kwargs or chunk_finish_reason:
                        chunk_data = StreamingChatCompletionChunk(
                            id=completion_id,
                            created=created_time,
                            model=model_name,
                            choices=[StreamingChoice(**choice_kwargs_with_delta)]
                        )
                        json_data = chunk_data.model_dump_json(exclude_none=True)
                        logger.debug(f"Yielding async chunk: {json_data}")
                        yield f"data: {json_data}\n\n"
                        chunk_count += 1
                        now = time.monotonic()
                        if now - last_yield_time > idle_warning_threshold:
                            logger.warning(
                                f"{stream_label}Async stream idle gap {now - last_yield_time:.2f}s for {model_name}")
                        last_yield_time = now

        if not sent_finish_reason:
            finish_reason = "tool_calls" if seen_tool_calls else "stop"
            logger.info(
                f"Async stream finished. Fallback finish_reason: {finish_reason}")

            final_chunk_data = StreamingChatCompletionChunk(
                id=completion_id,
                created=created_time,
                model=model_name,
                choices=[StreamingChoice(
                    delta=ChoiceDelta(), finish_reason=finish_reason)]
            )
            yield f"data: {final_chunk_data.model_dump_json(exclude_none=True)}\n\n"
            chunk_count += 1
            sent_finish_reason = True

    except NameError as e:
        logger.error(f"NameError during async streaming: {e}")
        error_chunk = {
            "error": {
                "message": f"Stream internal error: {e}",
                "type": "NameError",
                "code": None
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
    except (UniInferError, ValueError) as e:
        logger.error(f"Error during async streaming: {e}")
        status_code = getattr(e, 'status_code', None)
        message = str(e)
        if isinstance(e, ProviderError) and e.response_body:
            message = f"{message} | Provider Response: {e.response_body}"

        error_chunk = {
            "error": {
                "message": message,
                "type": type(e).__name__,
                "code": status_code
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
    except Exception as e:
        logger.exception(f"Unexpected error during async streaming: {e}")
        error_chunk = {
            "error": {
                "message": f"Unexpected server error: {type(e).__name__}",
                "type": "internal_server_error",
                "code": None
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

    logger.info(
        f"{stream_label}Async stream end for {model_name} (chunks={chunk_count}, heartbeats={heartbeat_count}, finish_sent={sent_finish_reason})")
    logger.info(f"{stream_label}Async stream sending [DONE] for {model_name}")

    yield "data: [DONE]\n\n"


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
