"""Image-generation + image-model-listing HTTP routes.

Split from the former media.py for locality: image work lives here, audio
(TTS/STT) lives in audio.py. The dispatch logic itself is in
``uniinfer.images.ImageTarget``; this module is the thin HTTP adapter.
"""
import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional, Callable

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from uniinfer.auth import get_optional_proxy_token
from uniinfer.provider_access import get_provider_api_key
from uniinfer.images import ImageTarget, ImageData, ImageGenerationError

logger = logging.getLogger("uniioai_proxy")

# One unified marker list for image-model name detection (fallback when the
# catalog's structured type is missing/wrong). Used by list_image_models only.
_IMAGE_MARKERS = (
    "image", "gpt-image", "dall-e", "stable-diffusion", "sdxl",
    "flux", "z-image", "imagen", "step-image", "search-image",
)


@asynccontextmanager
async def _http_client(request: Request):
    """Yield the app-lifetime shared httpx client when present, else a fallback."""
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


def create_images_router(
    parse_provider_model: Callable[..., tuple[str, str]],
) -> APIRouter:
    router = APIRouter()
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

            else:
                # All non-pollinations providers (tu, openai, openrouter, kilo,
                # cloudflare, stepfun, ...) route through the catalog: one code path,
                # one marker list, catalog type as primary signal.
                from uniinfer.proxy_services.models_registry import Catalog
                try:
                    all_models = Catalog().list_resolved()
                    models = sorted(set(
                        m["id"].split("@", 1)[-1]
                        for m in all_models
                        if m.get("provider") == provider_name
                        and (m.get("type") == "image"
                             or any(k in m["id"].lower() for k in _IMAGE_MARKERS))
                    ))
                except Exception:
                    models = []

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

                api_key = None
                if api_bearer_token:
                    try:
                        pname, _ = parse_provider_model(provider_model)
                        api_key = get_provider_api_key(api_bearer_token, pname)
                    except Exception:
                        api_key = None

                try:
                    target = ImageTarget(provider_model, api_key=api_key)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e))

                async with _http_client(request) as client:
                    try:
                        items: list[ImageData] = await target.agenerate(
                            prompt, n=n, size=size, client=client,
                        )
                    except PermissionError as e:
                        raise HTTPException(status_code=401, detail=str(e))
                    except ImageGenerationError as e:
                        raise HTTPException(status_code=e.status_code, detail=e.body)

                data_items = [item.to_dict() for item in items]

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

    return router
