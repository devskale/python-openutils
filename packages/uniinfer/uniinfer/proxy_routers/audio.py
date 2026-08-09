"""Audio (TTS + STT) HTTP routes — thin adapter over TTSTarget / STTTarget.

The dispatch logic (provider resolution, request building, dispatch, response
extraction) lives in ``uniinfer.audio``; this module is the HTTP layer.
"""
import logging
from typing import Optional, Callable

from fastapi import APIRouter, Depends, HTTPException, Request, File, Form, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from uniinfer.auth import get_optional_proxy_token, verify_provider_access
from uniinfer.completion import parse_provider_model
from uniinfer.audio import TTSTarget, STTTarget

logger = logging.getLogger("uniioai_proxy")


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
    segments: Optional[list] = None


def create_audio_router(
    parse_model: Callable[..., tuple[str, str]],
) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/audio/speech")
    async def generate_speech(
        request: Request,
        request_input: TTSRequestModel,
        api_bearer_token: Optional[str] = Depends(get_optional_proxy_token),
    ):
        try:
            provider_name, _ = parse_model(request_input.model)
            api_key = verify_provider_access(api_bearer_token, provider_name)
            if not api_key:
                raise HTTPException(status_code=401, detail=f"API key required for {provider_name}")

            try:
                target = TTSTarget(request_input.model, api_key=api_key)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

            result = await target.asynthesize(
                request_input.input,
                voice=request_input.voice,
                response_format=request_input.response_format or "mp3",
                speed=request_input.speed or 1.0,
                instructions=request_input.instructions,
            )
            return Response(content=result.audio_content, media_type=result.content_type)
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
            provider_name, _ = parse_model(model)
            api_key = verify_provider_access(api_bearer_token, provider_name)
            if not api_key:
                raise HTTPException(status_code=401, detail=f"API key required for {provider_name}")

            audio_content = await file.read()

            try:
                target = STTTarget(model, api_key=api_key)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

            result = await target.atranscribe(
                audio_content,
                language=language,
                prompt=prompt,
                response_format=response_format or "json",
                temperature=temperature if temperature is not None else 0.0,
            )

            if response_format == "verbose_json":
                return STTVerboseResponseModel(
                    text=result.text,
                    language=result.language,
                    duration=result.duration,
                    segments=result.segments,
                )
            return STTResponseModel(text=result.text)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Unexpected error in transcribe_audio endpoint: %s", e)
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    return router
