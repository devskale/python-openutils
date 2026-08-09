"""Audio (TTS + STT) HTTP routes.

Split from the former media.py for locality: audio work lives here, images
live in images.py. Currently only the TU provider is supported for audio.
"""
import logging
from typing import Optional, Callable

from fastapi import APIRouter, Depends, HTTPException, Request, File, Form, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from uniinfer.auth import get_optional_proxy_token, verify_provider_access

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
    parse_provider_model: Callable[..., tuple[str, str]],
) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/audio/speech")
    async def generate_speech(
        request: Request,
        request_input: TTSRequestModel,
        api_bearer_token: Optional[str] = Depends(get_optional_proxy_token),
    ):
        try:
            provider_name, model_name = parse_provider_model(
                request_input.model, allowed_providers=["tu"], task_name="TTS"
            )
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
            provider_name, model_name = parse_provider_model(
                model, allowed_providers=["tu"], task_name="STT"
            )
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
