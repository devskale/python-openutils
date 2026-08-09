import asyncio
import base64
import json
import os
import time
import urllib.parse
import logging
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Callable, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from uniinfer.auth import get_optional_proxy_token, verify_provider_access
from uniinfer.provider_access import get_provider_api_key, list_model_names_for_provider


logger = logging.getLogger("uniioai_proxy")


@asynccontextmanager
async def _http_client(request: Request):
    """Yield the app-lifetime shared httpx client when present (prod), else a
    short-lived fallback (tests / standalone). Sharing one client reuses the
    connection pool + TLS and avoids the per-request allocation churn that
    bloats a long-running process's RSS."""
    shared = getattr(request.app.state, "http", None)
    if shared is not None:
        yield shared
    else:
        async with httpx.AsyncClient(timeout=120.0) as client:
            yield client


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


class TTSRequestModel(BaseModel):
    model: str
    input: str
    voice: Optional[str] = None
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0
    instructions: Optional[str] = None


class STTResponseModel(BaseModel):
    text: str


class STTVerboseResponseModel(BaseModel):
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    segments: Optional[List[Dict]] = None


def create_media_router(
    parse_provider_model: Callable[..., tuple[str, str]],
    get_media_rate_limit: Callable[[], str],
) -> APIRouter:
    router = APIRouter()
    # Bound concurrent image generation. Each in-flight gen buffers whole images
    # (raw bytes -> base64 -> JSON, several MB each, multiple live copies);
    # without a cap, N parallel requests allocate N x that simultaneously and
    # thrash a small box. Excess requests queue here (backpressure) instead of
    # all churning at once.
    image_sem = asyncio.Semaphore(int(os.getenv("UNIINFER_IMAGE_CONCURRENCY", "2")))

    @router.get("/v1/image/models/{provider_name}")
    async def list_image_models(
        request: Request,
        provider_name: str,
        api_bearer_token: Optional[str] = Depends(get_optional_proxy_token),
    ):
        try:
            models = []

            if provider_name == "pollinations":
                try:
                    api_key_for_list = None
                    if api_bearer_token:
                        try:
                            api_key_for_list = get_provider_api_key(api_bearer_token, "pollinations")
                        except Exception:
                            pass
                    async with _http_client(request) as client:
                        headers = {"User-Agent": "UniIOAI/0.1"}
                        if api_key_for_list:
                            headers["Authorization"] = f"Bearer {api_key_for_list}"
                        resp = await client.get("https://gen.pollinations.ai/v1/models", headers=headers, timeout=10)
                        resp.raise_for_status()
                        for model in resp.json().get("data", []):
                            if "image" in model.get("output_modalities", []):
                                models.append(model["id"])
                except Exception as e:
                    logger.error("Failed to fetch Pollinations image models: %s, using fallback list", e)
                    models = ["flux", "kontext", "gptimage", "gptimage-large", "zimage", "klein"]

            elif provider_name == "tu":
                token_for_tu = api_bearer_token or os.getenv("TU_API_KEY")
                if not token_for_tu:
                    raise HTTPException(status_code=401, detail="Authentication required for provider 'tu'")

                raw_models = list_model_names_for_provider("tu", token_for_tu)
                image_markers = ("image", "z-image", "dall-e", "stable-diffusion", "sdxl", "flux")
                models = sorted(set(m for m in raw_models if any(marker in m.lower() for marker in image_markers)))
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image generation not supported for provider '{provider_name}'. Supported providers: pollinations, tu",
                )

            return {
                "object": "list",
                "data": [{"id": m, "object": "model", "owned_by": "skaledev"} for m in models],
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Error listing image models for %s: %s", provider_name, e)
            raise HTTPException(status_code=500, detail=f"Failed to list image models: {str(e)}")

    @router.post("/v1/images/generations")
    async def generate_images(
        request: Request,
        request_input: ImageGenerationRequest,
        api_bearer_token: Optional[str] = Depends(get_optional_proxy_token),
    ):
        async with image_sem:
            try:
                provider_model = request_input.model
                prompt = request_input.prompt
                n = request_input.n or 1
                size = request_input.size or "512x512"
                seed = request_input.seed

                provider_name, model_name = parse_provider_model(
                    provider_model, allowed_providers=["pollinations", "tu"], task_name="images"
                )

                api_key = None
                if api_bearer_token:
                    try:
                        api_key = get_provider_api_key(api_bearer_token, provider_name)
                    except Exception:
                        api_key = None

                width, height = 512, 512
                try:
                    if isinstance(size, str) and "x" in size:
                        w_str, h_str = size.split("x", 1)
                        width = int(w_str)
                        height = int(h_str)
                except Exception:
                    width, height = 512, 512

                data_items: List[dict] = []

                async with _http_client(request) as client:
                    if provider_name == "tu":
                        if not api_key:
                            raise HTTPException(status_code=401, detail="API key required for TU provider")

                        # Stream the upstream response (httpx best practice
                        # for bounded memory) instead of one-shot client.post()
                        # + json(), which buffers the full body in httpcore and
                        # retained native buffer per request (the image RSS creep).
                        # The async-with deterministically aclose()s the response.
                        async with client.stream(
                            "POST",
                            "https://aqueduct.ai.datalab.tuwien.ac.at/v1/images/generations",
                            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                            json={"model": model_name, "prompt": prompt, "n": n, "size": size},
                        ) as resp:
                            if resp.status_code != 200:
                                err = (await resp.aread()).decode("utf-8", "replace")[:500]
                                raise HTTPException(status_code=resp.status_code, detail=err)
                            raw = await resp.aread()
                        tu_data = json.loads(raw)
                        del raw
                        for item in tu_data.get("data", []):
                            b64_json = item.get("b64_json")
                            url = item.get("url")
                            if b64_json:
                                data_items.append({"b64_json": b64_json, **({"url": url} if url else {})})
                            elif url:
                                img_resp = await client.get(url, timeout=60)
                                img_resp.raise_for_status()
                                b64 = base64.b64encode(img_resp.content).decode("utf-8")
                                del img_resp
                                data_items.append({"b64_json": b64, "url": url})
                        del tu_data  # drop the parsed tree early; b64 strings are aliased into data_items
                    else:
                        encoded_prompt = urllib.parse.quote(prompt)
                        base_url = "https://gen.pollinations.ai/image"
                        for i in range(n):
                            this_seed = seed if seed is not None else int(time.time()) + i
                            url = f"{base_url}/{encoded_prompt}?model={model_name}&width={width}&height={height}&seed={this_seed}"

                            headers = {"Accept": "image/jpeg", "User-Agent": "UniIOAI/0.1"}
                            if api_key:
                                headers["Authorization"] = f"Bearer {api_key}"

                            resp = await client.get(url, headers=headers, timeout=60)
                            if resp.status_code != 200:
                                detail = resp.text if resp.text else "Failed to generate image from Pollinations"
                                raise HTTPException(status_code=resp.status_code, detail=detail)

                            b64 = base64.b64encode(resp.content).decode("utf-8")
                            del resp
                            data_items.append({"b64_json": b64, "url": url})

                # Stream the OpenAI-shaped JSON response instead of building one
                # big JSONResponse body. Each base64 image is then serialized
                # chunk-by-chunk straight to the socket, so we never hold the
                # full serialized body in memory on top of data_items (~halves
                # the peak RSS of an image request on a memory-constrained box).
                created = int(time.time())
                model_for_resp = provider_model

                async def _image_json_stream():
                    yield ('{"created":%d,"data":[' % created).encode()
                    first = True
                    for item in data_items:
                        yield (b"" if first else b",") + json.dumps(item).encode()
                        first = False
                    yield b'],"model":' + json.dumps(model_for_resp).encode() + b'}'

                return StreamingResponse(_image_json_stream(), media_type="application/json")

            except HTTPException:
                raise
            except Exception as e:
                logger.exception("Unexpected error in generate_images endpoint: %s", e)
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    @router.post("/v1/audio/speech")
    async def generate_speech(
        request: Request,
        request_input: TTSRequestModel,
        api_bearer_token: Optional[str] = Depends(get_optional_proxy_token),
    ):
        try:
            provider_name, model_name = parse_provider_model(request_input.model, allowed_providers=["tu"], task_name="TTS")
            api_key = verify_provider_access(api_bearer_token, provider_name)
            if not api_key:
                raise HTTPException(status_code=401, detail="API key required for TU provider")

            from uniinfer import TTSRequest
            from uniinfer.providers.tu_tts import TuAITTSProvider

            tts_provider = TuAITTSProvider(api_key=api_key)
            tts_request = TTSRequest(
                input=request_input.input,
                model=model_name,
                voice=request_input.voice,
                response_format=request_input.response_format or "mp3",
                speed=request_input.speed or 1.0,
                instructions=request_input.instructions,
            )
            response = await tts_provider.agenerate_speech(tts_request)
            return Response(content=response.audio_content, media_type=response.content_type)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Unexpected error in generate_speech endpoint: %s", e)
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    @router.post("/v1/audio/transcriptions")
    async def transcribe_audio(
        request: Request,
        file: UploadFile = File(...),
        model: str = Form(...),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: Optional[str] = Form("json"),
        temperature: Optional[float] = Form(0.0),
        api_bearer_token: Optional[str] = Depends(get_optional_proxy_token),
    ):
        try:
            provider_name, model_name = parse_provider_model(model, allowed_providers=["tu"], task_name="STT")
            api_key = verify_provider_access(api_bearer_token, provider_name)
            if not api_key:
                raise HTTPException(status_code=401, detail="API key required for TU provider")

            audio_content = await file.read()

            from uniinfer import STTRequest
            from uniinfer.providers.tu_stt import TuAISTTProvider

            stt_provider = TuAISTTProvider(api_key=api_key)
            stt_request = STTRequest(
                file=audio_content,
                model=model_name,
                language=language,
                prompt=prompt,
                response_format=response_format or "json",
                temperature=temperature if temperature is not None else 0.0,
            )
            response = await stt_provider.atranscribe(stt_request)

            if response_format == "verbose_json":
                return STTVerboseResponseModel(
                    text=response.text,
                    language=response.language,
                    duration=response.duration,
                    segments=response.segments,
                )
            return STTResponseModel(text=response.text)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Unexpected error in transcribe_audio endpoint: %s", e)
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    return router
