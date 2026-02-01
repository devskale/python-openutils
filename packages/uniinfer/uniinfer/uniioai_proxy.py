from logging.handlers import RotatingFileHandler
import logging
from typing import List, Optional, Dict, Any, AsyncGenerator, Union, Tuple
import re
import subprocess
import uuid
import json
import time
import sys
import requests
import base64
from fastapi.security.http import HTTPAuthorizationCredentials
from fastapi.security import HTTPBearer  # Import HTTPBearer
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from fastapi import FastAPI, HTTPException, Request, Depends, File, Form, UploadFile
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# --- Setup Logging ---
# Configure root logger to capture logs from all modules (including uniinfer)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Create rotating file handler (2MB max, 5 backup files)
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
    from uniinfer.uniioai import stream_completion, get_completion, get_provider_api_key, list_providers, list_models_for_provider, get_embeddings, list_embedding_providers, list_embedding_models_for_provider
    from uniinfer.errors import UniInferError, AuthenticationError, ProviderError, RateLimitError
    from uniinfer.auth import validate_proxy_token, get_optional_proxy_token, verify_provider_access
except ImportError as e:
    logger.error(f"Error importing from uniinfer.uniioai: {e}")
    logger.error(f"Python path: {sys.path[:3]}")
    sys.exit(1)


app = FastAPI(
    title="UniIOAI API",
    description="OpenAI-compatible API wrapper using UniInfer",
    version="0.1.0",
)

# --- Rate Limiting Setup ---
# Enable headers to let clients know their limits
limiter = Limiter(key_func=get_remote_address, headers_enabled=True)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

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

def parse_provider_model(provider_model: str, allowed_providers: Optional[List[str]] = None, task_name: Optional[str] = None) -> Tuple[str, str]:
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
    content: Union[str, List[Dict[str, Any]], None] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequestInput(BaseModel):
    model: str  # Expected format: "provider@modelname"
    messages: List[ChatMessageInput]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500
    stream: Optional[bool] = False
    base_url: Optional[str] = None  # Add base_url field
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[Any] = None
    # Add other common OpenAI parameters if needed, e.g., top_p, frequency_penalty


class ChatMessageOutput(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None


class ChoiceDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None


class StreamingChoice(BaseModel):
    index: int = 0
    delta: ChoiceDelta
    finish_reason: Optional[str] = None


class StreamingChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamingChoice]


class NonStreamingChoice(BaseModel):
    index: int = 0
    message: ChatMessageOutput
    finish_reason: str = "stop"  # Assuming stop, uniinfer might not provide this


class NonStreamingChatCompletion(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[NonStreamingChoice]
    # uniinfer doesn't provide usage yet
    usage: Optional[Dict[str, int]] = None


# --- Models for /v1/models endpoint ---

class Model(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "skaledev"  # Or determine dynamically if needed


class ModelList(BaseModel):
    object: str = "list"
    data: List[Model]


# --- Add ProviderList model for OpenAI‐style list response ---
class ProviderList(BaseModel):
    object: str = "list"
    data: List[str]


# --- Models for Embedding Endpoint ---

class EmbeddingRequest(BaseModel):
    model: str  # Expected format: "provider@modelname"
    input: List[str]  # List of texts to embed
    user: Optional[str] = None  # Optional user identifier


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Optional[Dict[str, int]] = None


# --- Predefined Models ---
# The models.txt file is expected to be in the package root (one level up from uniioai_proxy.py)
MODELS_FILE_PATH = os.path.join(os.path.dirname(script_dir), "models.txt")


def parse_models_file():
    """Parses the models.txt file to extract available models."""
    models = []
    if not os.path.exists(MODELS_FILE_PATH):
        # Fallback to predefined if file doesn't exist
        return PREDEFINED_MODELS

    current_provider = None
    try:
        with open(MODELS_FILE_PATH, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Check for provider header
                # "Available models for mistral:"
                provider_match = re.match(r"Available models for (\w+):", line)
                if provider_match:
                    current_provider = provider_match.group(1)
                    continue

                # Check for model item
                # "- mistral-tiny"
                if line.startswith("- ") and current_provider:
                    model_name = line[2:].strip()
                    models.append(f"{current_provider}@{model_name}")
    except Exception as e:
        logger.error(f"Error reading models file: {e}")
        return PREDEFINED_MODELS

    return models if models else PREDEFINED_MODELS


PREDEFINED_MODELS = [
    "mistral@mistral-tiny-latest",
    "ollama@qwen2.5:3b",
    "openrouter@google/gemma-3-12b-it:free",
    "arli@Mistral-Nemo-12B-Instruct-2407",
    "internlm@internlm3-latest",
    "stepfun@step-1-flash",
    "upstage@solar-mini-250401",
    "bigmodel@glm-4-flash",
    "ngc@google/gemma-3-27b-it",
    "cohere@command-r",
    "moonshot@kimi-latest",
    "groq@llama3-8b-8192",
    "gemini@models/gemma-3-27b-it",
    "chutes@Qwen/Qwen3-235B-A22B",
    "pollinations@grok"
]


# --- Helper Functions ---

# Update signature: remove api_bearer_token, add provider_api_key
async def stream_response_generator(messages: List[Dict], provider_model: str, temp: float, max_tok: int, provider_api_key: Optional[str], base_url: Optional[str], tools: Optional[List[Dict]] = None, tool_choice: Optional[Any] = None) -> AsyncGenerator[str, None]:
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
    yield f"data: {first_chunk_data.model_dump_json()}\n\n"

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
                    json_data = chunk_data.model_dump_json()
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
            yield f"data: {final_chunk_data.model_dump_json()}\n\n"

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


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """
    OpenAI-compatible endpoint to list available models.
    Returns a list of models supported by UniInfer, read from models.txt.
    """
    models = parse_models_file()
    model_data = [Model(id=model_id) for model_id in models]
    return ModelList(data=model_data)


@app.post("/v1/system/update-models")
async def update_models():
    """
    Triggers 'uniinfer -l --list-models' to update the models.txt file.
    """
    try:
        # Run uniinfer -l --list-models and capture output
        # Assuming 'uniinfer' is in the path. If not, we might need to use sys.executable -m uniinfer.uniinfer_cli
        cmd = ["uniinfer", "-l", "--list-models"]

        # Check if uniinfer is in path, otherwise try python module execution
        if subprocess.call(["which", "uniinfer"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
            cmd = [sys.executable, "-m",
                   "uniinfer.uniinfer_cli", "-l", "--list-models"]

        result = await run_in_threadpool(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Write to models.txt
        with open(MODELS_FILE_PATH, 'w') as f:
            f.write(result.stdout)

        return {"status": "success", "message": "Models updated successfully", "output_length": len(result.stdout)}
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update models: {e.stderr}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}")


# --- Update /v1/providers to return ProviderList ---
@app.get("/v1/providers", response_model=ProviderList)
async def get_providers(api_bearer_token: str = Depends(validate_proxy_token)):
    """
    OpenAI‐style endpoint to list available providers.
    """
    try:
        providers = list_providers()
        return ProviderList(data=providers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
@limiter.limit(get_chat_rate_limit)
async def chat_completions(request: Request, request_input: ChatCompletionRequestInput, api_bearer_token: Optional[str] = Depends(get_optional_proxy_token)):
    """
    OpenAI-compatible chat completions endpoint.
    Uses the 'model' field in the format 'provider@modelname'.
    Requires Bearer token authentication (used for key retrieval).
    Optionally accepts a 'base_url'. If provider is 'ollama' and no base_url is provided,
    it defaults to 'https://amp1.mooo.com:11444'.
    """
    base_url = request_input.base_url  # Get base_url from request first
    provider_model = request_input.model
    messages_dict = [msg.model_dump() for msg in request_input.messages]

    # Debug logging for request
    logger.info(
        f"Received chat completion request for model: {provider_model}")
    logger.debug(
        f"Request params: stream={request_input.stream}, tools_count={len(request_input.tools) if request_input.tools else 0}")
    if request_input.tools:
        logger.debug(
            f"Tools: {[t.get('function', {}).get('name') for t in request_input.tools]}")

    try:
        # --- API Key Retrieval & Base URL Logic ---
        provider_name, _ = parse_provider_model(provider_model)

        # Set default base_url for ollama if not provided
        if provider_name == "ollama" and base_url is None:
            base_url = PROVIDER_CONFIGS.get("ollama", {}).get(
                "extra_params", {}).get("base_url")
        logger.debug(f"DEBUG: Using base_url: {base_url}")

        provider_api_key = verify_provider_access(
            api_bearer_token, provider_name)
        # --- End API Key Retrieval & Base URL Logic ---

        if request_input.stream:
            # Use the async generator with StreamingResponse
            return StreamingResponse(
                stream_response_generator(
                    messages=messages_dict,
                    provider_model=provider_model,
                    temp=request_input.temperature,
                    max_tok=request_input.max_tokens,
                    provider_api_key=provider_api_key,  # Pass retrieved key
                    base_url=base_url,  # Pass potentially modified base_url
                    tools=request_input.tools,
                    tool_choice=request_input.tool_choice
                ),
                media_type="text/event-stream"
            )
        else:
            # Wrap synchronous get_completion in run_in_threadpool
            full_content = await run_in_threadpool(
                get_completion,  # The sync function
                messages=messages_dict,
                provider_model_string=provider_model,
                temperature=request_input.temperature,
                max_tokens=request_input.max_tokens,
                provider_api_key=provider_api_key,  # Pass retrieved key
                base_url=base_url,  # Pass potentially modified base_url
                tools=request_input.tools,
                tool_choice=request_input.tool_choice
            )

            # Handle response which can be content string or ChatMessage object
            content = None
            tool_calls = None

            if hasattr(full_content, 'tool_calls'):
                # It's a ChatMessage object
                content = full_content.content
                tool_calls = full_content.tool_calls
            else:
                # It's a string
                content = full_content

            # Format the response according to OpenAI spec
            finish_reason = "tool_calls" if tool_calls else "stop"
            response_data = NonStreamingChatCompletion(
                model=provider_model,
                choices=[
                    NonStreamingChoice(
                        message=ChatMessageOutput(
                            role="assistant", content=content, tool_calls=tool_calls),
                        finish_reason=finish_reason
                    )
                ]
                # Usage data is not available from uniioai currently
            )
            return response_data

    # Catches ValueErrors from uniioai completion functions (e.g., model format)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Note: AuthenticationError from uniioai.py key retrieval is handled above
    except AuthenticationError as e:
        status_code = getattr(e, 'status_code', 401) or 401
        detail = str(e)
        if e.response_body:
            detail = f"{detail} | Response: {e.response_body}"
        raise HTTPException(status_code=status_code, detail=detail)
    except RateLimitError as e:
        status_code = getattr(e, 'status_code', 429) or 429
        detail = str(e)
        if e.response_body:
            detail = f"{detail} | Response: {e.response_body}"
        raise HTTPException(status_code=status_code, detail=detail)
    except HTTPException:
        raise
    except ProviderError as e:
        status_code = getattr(e, 'status_code', 500) or 500
        detail = f"Provider Error ({provider_name}): {e}"
        if e.response_body:
            detail = f"{detail} | Response: {e.response_body}"
        raise HTTPException(status_code=status_code, detail=detail)
    except UniInferError as e:  # Catches general uniinfer errors
        raise HTTPException(status_code=500, detail=f"UniInfer Error: {e}")
    except HTTPException:
        raise
    except Exception as e:
        # Catch-all for unexpected errors
        logger.exception(
            f"Unexpected error in /v1/chat/completions: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal Server Error: {type(e).__name__}")


# --- Add Embedding Endpoint ---
@app.post("/v1/embeddings", response_model=EmbeddingResponse)
@limiter.limit(get_embeddings_rate_limit)
async def create_embeddings(request: Request, request_input: EmbeddingRequest, api_bearer_token: Optional[str] = Depends(get_optional_proxy_token)):
    """
    OpenAI-compatible embeddings endpoint.
    Uses the 'model' field in the format 'provider@modelname'.
    Authentication optional for Ollama, required for other providers.
    """
    provider_model = request_input.model
    input_texts = request_input.input

    try:
        # Validate model format
        provider_name, _ = parse_provider_model(provider_model)

        # For Ollama, we don't require authentication
        if provider_name == 'ollama':
            provider_api_key = None
            # Get base_url from provider config for Ollama
            base_url = PROVIDER_CONFIGS.get("ollama", {}).get(
                "extra_params", {}).get("base_url")
        else:
            # For other providers, require authentication
            provider_api_key = verify_provider_access(
                api_bearer_token, provider_name)
            base_url = None

        # Get embeddings using the synchronous function in a thread pool
        embeddings_result = await run_in_threadpool(
            get_embeddings,
            input_texts=input_texts,
            provider_model_string=provider_model,
            provider_api_key=provider_api_key,
            base_url=base_url
        )

        # Format the response according to OpenAI spec
        embedding_data = []
        for i, embedding in enumerate(embeddings_result['embeddings']):
            embedding_data.append(EmbeddingData(
                embedding=embedding,
                index=i
            ))

        response_data = EmbeddingResponse(
            data=embedding_data,
            model=provider_model,
            usage=embeddings_result['usage']
        )
        return response_data

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AuthenticationError as e:
        status_code = getattr(e, 'status_code', 401) or 401
        detail = str(e)
        if e.response_body:
            detail = f"{detail} | Response: {e.response_body}"
        raise HTTPException(status_code=status_code, detail=detail)
    except RateLimitError as e:
        status_code = getattr(e, 'status_code', 429) or 429
        detail = str(e)
        if e.response_body:
            detail = f"{detail} | Response: {e.response_body}"
        raise HTTPException(status_code=status_code, detail=detail)
    except ProviderError as e:
        status_code = getattr(e, 'status_code', 500) or 500
        detail = f"Provider Error ({provider_name}): {e}"
        if e.response_body:
            detail = f"{detail} | Response: {e.response_body}"
        raise HTTPException(status_code=status_code, detail=detail)
    except UniInferError as e:
        raise HTTPException(status_code=500, detail=f"UniInfer Error: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            f"Unexpected error in /v1/embeddings: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal Server Error: {type(e).__name__}")


# --- Change dynamic list models to return ModelList ---
@app.get("/v1/models/{provider_name}", response_model=ModelList)
async def dynamic_list_models(provider_name: str, api_bearer_token: str = Depends(validate_proxy_token)):
    """
    List available models for a specific provider, formatted OpenAI‐style.
    """
    try:
        raw_models = list_models_for_provider(provider_name, api_bearer_token)
        model_objs = [Model(id=m) for m in raw_models]
        return ModelList(data=model_objs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Add Embedding Providers Endpoint ---
@app.get("/v1/embedding/providers", response_model=ProviderList)
async def get_embedding_providers(request: Request):
    """
    OpenAI‐style endpoint to list available embedding providers.
    Authentication optional.
    """
    try:
        providers = list_embedding_providers()
        return ProviderList(data=providers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Add Embedding Models Endpoint ---
@app.get("/v1/embedding/models/{provider_name}", response_model=ModelList)
async def dynamic_list_embedding_models(provider_name: str, api_bearer_token: Optional[str] = Depends(get_optional_proxy_token)):
    """
    List available embedding models for a specific provider, formatted OpenAI‐style.
    Authentication optional for Ollama.
    """
    try:
        # For Ollama, we don't require authentication
        if provider_name == 'ollama':
            api_bearer_token = None
        else:
            # For other providers, require authentication
            if not api_bearer_token:
                raise HTTPException(
                    status_code=401, detail="Authentication required for this provider")

        raw_models = list_embedding_models_for_provider(
            provider_name, api_bearer_token or "")
        model_objs = [Model(id=m) for m in raw_models]
        return ModelList(data=model_objs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "UniIOAI API is running. Visit /webdemo for the interactive demo, or use POST /v1/chat/completions, POST /v1/embeddings, or GET /v1/models"}


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
# --- Models for Image Generation Endpoint ---


class ImageGenerationRequest(BaseModel):
    model: str
    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = "512x512"
    seed: Optional[int] = None


class ImageData(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int = Field(default_factory=lambda: int(time.time()))
    data: List[ImageData]
    model: str


@app.post("/v1/images/generations", response_model=ImageGenerationResponse)
@limiter.limit(get_media_rate_limit)
async def generate_images(request: Request, request_input: ImageGenerationRequest, api_bearer_token: Optional[str] = Depends(get_optional_proxy_token)):
    """
    OpenAI-compatible image generations endpoint.
    Accepts provider@model naming and generates images via Pollinations image API.
    """
    provider_model = request_input.model
    prompt = request_input.prompt
    n = request_input.n or 1
    size = request_input.size or "512x512"
    seed = request_input.seed

    try:
        provider_name, model_name = parse_provider_model(
            provider_model, allowed_providers=['pollinations', 'tu'], task_name="images")

        api_key = None
        if api_bearer_token:
            try:
                api_key = get_provider_api_key(api_bearer_token, provider_name)
            except (ValueError, AuthenticationError):
                api_key = None

        width, height = 512, 512
        try:
            if isinstance(size, str) and 'x' in size:
                w_str, h_str = size.split('x', 1)
                width = int(w_str)
                height = int(h_str)
        except Exception:
            width, height = 512, 512

        data_items: List[ImageData] = []

        # TU provider (Aqueduct) - uses OpenAI-compatible API
        if provider_name == 'tu':
            # Fallback to env var if no key provided in request
            if not api_key:
                api_key = os.environ.get("TU_API_KEY")

            if not api_key:
                raise HTTPException(
                    status_code=401, detail="API key required for TU provider")

            tu_base_url = "https://aqueduct.ai.datalab.tuwien.ac.at/v1"
            tu_endpoint = f"{tu_base_url}/images/generations"

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            payload = {
                "model": model_name,
                "prompt": prompt,
                "n": n,
                "size": size
            }

            try:
                resp = requests.post(
                    tu_endpoint, headers=headers, json=payload, timeout=120)
                resp.raise_for_status()
                tu_data = resp.json()

                # Process TU response (OpenAI-compatible format)
                for item in tu_data.get("data", []):
                    # TU may return b64_json or url
                    b64_json = item.get("b64_json")
                    url = item.get("url")

                    if b64_json:
                        data_items.append(
                            ImageData(b64_json=b64_json, url=url))
                    elif url:
                        # If only URL is provided, fetch and encode
                        try:
                            img_resp = requests.get(url, timeout=60)
                            img_resp.raise_for_status()
                            b64 = base64.b64encode(
                                img_resp.content).decode('utf-8')
                            data_items.append(ImageData(b64_json=b64, url=url))
                        except requests.exceptions.RequestException as e:
                            raise HTTPException(
                                status_code=502, detail=f"Failed to fetch image from TU URL: {e}")

            except requests.exceptions.HTTPError as e:
                raise HTTPException(status_code=e.response.status_code if e.response else 500,
                                    detail=f"TU API error: {e.response.text if e.response else str(e)}")
            except requests.exceptions.RequestException as e:
                raise HTTPException(
                    status_code=502, detail=f"TU API request failed: {str(e)}")

        # Pollinations provider - existing logic
        else:
            # Enforce known image models; default to turbo when unknown
            allowed_models = {"turbo", "flux", "gptimage"}
            if model_name not in allowed_models:
                model_name = "turbo"

            encoded_prompt = requests.utils.quote(prompt)
            base_url = "https://image.pollinations.ai/prompt"

            for i in range(n):
                this_seed = seed if seed is not None else int(time.time()) + i
                url_primary = f"{base_url}/{encoded_prompt}?model={model_name}&width={width}&height={height}&seed={this_seed}"
                url_fallback = f"{base_url}/{encoded_prompt}?width={width}&height={height}&seed={this_seed}"
                headers = {"Accept": "image/jpeg", "User-Agent": "UniIOAI/0.1"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                used_url = url_primary
                ok = False
                try:
                    resp = requests.get(
                        url_primary, headers=headers, timeout=60)
                    if resp.status_code == 200:
                        ok = True
                    else:
                        used_url = url_fallback
                        resp_fb = requests.get(
                            url_fallback, headers=headers, timeout=60)
                        resp = resp_fb
                        if resp.status_code == 200:
                            ok = True
                except requests.exceptions.RequestException:
                    used_url = url_fallback
                    try:
                        resp = requests.get(
                            url_fallback, headers=headers, timeout=60)
                        if resp.status_code == 200:
                            ok = True
                    except requests.exceptions.RequestException:
                        ok = False
                if not ok:
                    turbo_primary = f"{base_url}/{encoded_prompt}?model=turbo&width={width}&height={height}&seed={this_seed}"
                    turbo_fallback = f"{base_url}/{encoded_prompt}?width={width}&height={height}&seed={this_seed}"
                    try:
                        rtp = requests.get(
                            turbo_primary, headers=headers, timeout=60)
                        if rtp.status_code == 200:
                            used_url = turbo_primary
                            resp = rtp
                            ok = True
                        else:
                            rtf = requests.get(
                                turbo_fallback, headers=headers, timeout=60)
                            if rtf.status_code == 200:
                                used_url = turbo_fallback
                                resp = rtf
                                ok = True
                    except requests.exceptions.RequestException:
                        ok = False
                    if not ok:
                        primary_status = None
                        fallback_status = None
                        try:
                            r1 = requests.get(
                                url_primary, headers=headers, timeout=10)
                            primary_status = r1.status_code
                            r2 = requests.get(
                                url_fallback, headers=headers, timeout=10)
                            fallback_status = r2.status_code
                        except Exception:
                            pass
                        detail = {
                            "message": "Pollinations image error",
                            "primary_url": url_primary,
                            "primary_status": primary_status,
                            "fallback_url": url_fallback,
                            "fallback_status": fallback_status,
                        }
                        raise HTTPException(status_code=502, detail=detail)
                b64 = base64.b64encode(resp.content).decode('utf-8')
                data_items.append(ImageData(b64_json=b64, url=used_url))

        return ImageGenerationResponse(data=data_items, model=provider_model)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Models for TTS Endpoint ---

class TTSRequestModel(BaseModel):
    model: str  # Expected format: "provider@modelname"
    input: str
    voice: Optional[str] = None
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0
    instructions: Optional[str] = None


@app.post("/v1/audio/speech")
@limiter.limit(get_media_rate_limit)
async def generate_speech(
        request: Request, request_input: TTSRequestModel, api_bearer_token: Optional[str] = Depends(get_optional_proxy_token)):
    """
    OpenAI-compatible text-to-speech endpoint.
    Generates audio from text using TTS models.
    """
    provider_model = request_input.model
    input_text = request_input.input

    try:
        provider_name, model_name = parse_provider_model(
            provider_model, allowed_providers=['tu'], task_name="TTS")

        api_key = verify_provider_access(api_bearer_token, provider_name)

        if not api_key:
            raise HTTPException(
                status_code=401, detail="API key required for TU provider")

        # Import TTS classes
        from uniinfer import TTSRequest
        from uniinfer.providers.tu_tts import TuAITTSProvider

        # Create TTS provider
        tts_provider = TuAITTSProvider(api_key=api_key)

        # Create TTS request
        tts_request = TTSRequest(
            input=input_text,
            model=model_name,
            voice=request_input.voice,
            response_format=request_input.response_format or "mp3",
            speed=request_input.speed or 1.0,
            instructions=request_input.instructions
        )

        # Generate speech (run in thread pool to avoid blocking)
        response = await run_in_threadpool(
            tts_provider.generate_speech,
            tts_request
        )

        # Return audio content
        from fastapi.responses import Response
        return Response(
            content=response.audio_content,
            media_type=response.content_type
        )

    except HTTPException:
        raise
    except ProviderError as e:
        status_code = getattr(e, 'status_code', 500) or 500
        detail = str(e)
        if e.response_body:
            detail = f"{detail} | Response: {e.response_body}"
        raise HTTPException(status_code=status_code, detail=detail)
    except Exception as e:
        logger.exception(
            f"Unexpected error in /v1/audio/speech: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Models for STT Endpoint ---

class STTResponseModel(BaseModel):
    text: str


class STTVerboseResponseModel(BaseModel):
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    segments: Optional[List[Dict]] = None


@app.post("/v1/audio/transcriptions")
@limiter.limit(get_media_rate_limit)
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
    api_bearer_token: Optional[str] = Depends(get_optional_proxy_token)
):
    """
    OpenAI-compatible speech-to-text endpoint.
    Transcribes audio files to text.
    """

    provider_model = model

    try:
        provider_name, model_name = parse_provider_model(
            provider_model, allowed_providers=['tu'], task_name="STT")

        api_key = verify_provider_access(api_bearer_token, provider_name)

        if not api_key:
            raise HTTPException(
                status_code=401, detail="API key required for TU provider")

        # Read audio file content
        audio_content = await file.read()

        # Import STT classes
        from uniinfer import STTRequest
        from uniinfer.providers.tu_stt import TuAISTTProvider

        # Create STT provider
        stt_provider = TuAISTTProvider(api_key=api_key)

        # Create STT request
        stt_request = STTRequest(
            file=audio_content,
            model=model_name,
            language=language,
            prompt=prompt,
            response_format=response_format or "json",
            temperature=temperature or 0.0
        )

        # Transcribe audio (run in thread pool to avoid blocking)
        response = await run_in_threadpool(
            stt_provider.transcribe,
            stt_request
        )

        # Return transcription based on response format
        if response_format == "verbose_json":
            return STTVerboseResponseModel(
                text=response.text,
                language=response.language,
                duration=response.duration,
                segments=response.segments
            )
        else:
            return STTResponseModel(text=response.text)

    except HTTPException:
        raise
    except ProviderError as e:
        status_code = getattr(e, 'status_code', 500) or 500
        detail = str(e)
        if e.response_body:
            detail = f"{detail} | Response: {e.response_body}"
        raise HTTPException(status_code=status_code, detail=detail)
    except Exception as e:
        logger.exception(
            f"Unexpected error in /v1/audio/transcriptions: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
