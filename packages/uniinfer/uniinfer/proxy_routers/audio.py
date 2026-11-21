"""Audio (TTS + STT) HTTP routes — thin adapter over TTSTarget / STTTarget.

The dispatch logic (provider resolution, request building, dispatch, response
extraction) lives in ``uniinfer.audio``; this module is the HTTP layer.
"""
import logging
from typing import Optional, Callable

from fastapi import APIRouter, Depends, HTTPException, Request, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from uniinfer.auth import get_optional_proxy_token, verify_provider_access
from uniinfer.completion import parse_provider_model
from uniinfer.audio import TTSTarget, STTTarget
from uniinfer.proxy_services.key_usage import get_key_usage

logger = logging.getLogger("uniioai_proxy")


class TTSRequestModel(BaseModel):
    model: str
    input: str
    voice: Optional[str] = None
    # None → pfad-spezifischer Default: mp3 (buffered) bzw. wav (streaming,
    # roh streambarer Container; ffplay & Co. brauchen das)
    response_format: Optional[str] = None
    speed: Optional[float] = 1.0
    instructions: Optional[str] = None
    language: Optional[str] = None
    stream: bool = False
    stream_format: Optional[str] = None


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
            inst = None
            try:
                from uniinfer.config.instances import resolve_instance
                spec = resolve_instance(parse_model(request_input.model)[0])
                if (getattr(spec, "base_url", None) or "").rstrip("/"):
                    inst = spec
            except Exception:
                pass
            if inst is not None:
                api_key = verify_provider_access(
                    api_bearer_token, inst.alias, own_key=request.headers.get("x-provider-key")
                )
                target = TTSTarget(
                    f"{inst.provider}@{parse_model(request_input.model)[1]}",
                    api_key=api_key, base_url=inst.base_url)
            else:
                provider_name, _ = parse_model(request_input.model)
                api_key = verify_provider_access(
                    api_bearer_token, provider_name, own_key=request.headers.get("x-provider-key")
                )
                if not api_key:
                    raise HTTPException(status_code=401, detail=f"API key required for {provider_name}")
                try:
                    target = TTSTarget(request_input.model, api_key=api_key)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e))

            get_key_usage().note(api_bearer_token, request_input.model, api_key)
            if request_input.stream:
                # Live streaming: chunks flow the moment the upstream sends
                # them (vLLM-Omni stream:true) — no proxy-side buffering. The
                # media_type names the container format; "audio" stream_format
                # delivers raw wav/pcm bytes, "sse" OpenAI speech.audio events.
                fmt = request_input.response_format or "wav"
                media = "audio/wav" if fmt in ("wav", "pcm") else f"audio/{fmt}"
                return StreamingResponse(
                    target.astream_synthesize(
                        request_input.input,
                        voice=request_input.voice,
                        response_format=fmt,
                        speed=request_input.speed or 1.0,
                        instructions=request_input.instructions,
                        language=request_input.language,
                        stream_format=request_input.stream_format,
                    ),
                    media_type=media,
                )
            result = await target.asynthesize(
                request_input.input,
                voice=request_input.voice,
                response_format=request_input.response_format or "mp3",
                speed=request_input.speed or 1.0,
                instructions=request_input.instructions,
                language=request_input.language,
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
            audio_content = await file.read()

            inst = None
            try:
                from uniinfer.config.instances import resolve_instance
                spec = resolve_instance(parse_model(model)[0])
                if (getattr(spec, "base_url", None) or "").rstrip("/"):
                    inst = spec
            except Exception:
                pass
            if inst is not None:
                api_key = verify_provider_access(
                    api_bearer_token, inst.alias, own_key=request.headers.get("x-provider-key")
                )
                try:
                    target = STTTarget(
                        f"{inst.provider}@{parse_model(model)[1]}",
                        api_key=api_key, base_url=inst.base_url)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e))
            else:
                provider_name, _ = parse_model(model)
                api_key = verify_provider_access(
                    api_bearer_token, provider_name, own_key=request.headers.get("x-provider-key")
                )
                if not api_key:
                    raise HTTPException(status_code=401, detail=f"API key required for {provider_name}")
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
            get_key_usage().note(api_bearer_token, model, api_key)

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

    @router.websocket("/v1/audio/transcriptions/stream")
    async def stream_transcriptions(ws: WebSocket):
        """Realtime STT passthrough: pipe raw PCM chunks to an openai-compat instance.

        Query params: ``model`` (``alias@bare`` or bare alias), ``token`` (proxy
        bearer — browsers cannot set WS headers), ``language`` (optional, default
        upstream's). Protocol is the upstream's verbatim: binary frames in
        (int16 PCM 16 kHz mono, empty binary frame = end-of-audio), JSON
        ``partial``/``final`` messages out. Close codes: 4400 bad request,
        4401 auth failed, 4404 unknown instance, 4502 upstream unreachable.
        """
        import asyncio

        from uniinfer.auth import verify_provider_access
        from uniinfer.config.instances import resolve_instance

        async def _close(code: int, reason: str) -> None:
            await ws.close(code=code, reason=reason)

        model = ws.query_params.get("model") or ""
        token = ws.query_params.get("token") or ""
        language = ws.query_params.get("language")
        if not model:
            await ws.accept()
            await _close(4400, "query param 'model' is required")
            return
        try:
            alias, _bare = parse_model(model)
        except Exception:
            await ws.accept()
            await _close(4400, f"invalid model spec: {model}")
            return
        try:
            spec = resolve_instance(alias)
        except Exception:
            await ws.accept()
            await _close(4404, f"unknown instance: {alias}")
            return
        base = (getattr(spec, "base_url", None) or "").rstrip("/")
        if not base:
            await ws.accept()
            await _close(4404, f"instance '{alias}' has no base_url (no WS passthrough)")
            return
        try:
            api_key = verify_provider_access(token if token else None, alias)
        except HTTPException:
            await ws.accept()
            await _close(4401, "authentication failed")
            return

        import websockets

        ws_base = base
        if base.startswith("http://"):
            ws_base = "ws://" + base[len("http://"):]
        elif base.startswith("https://"):
            ws_base = "wss://" + base[len("https://"):]
        upstream_url = f"{ws_base}/audio/transcriptions/stream"
        if language:
            upstream_url += f"?language={language}"
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            up = await websockets.connect(upstream_url, additional_headers=headers, max_size=2**22)
        except Exception as e:
            await ws.accept()
            await _close(4502, f"upstream connect failed: {e}")
            return

        await ws.accept()

        async def _pump_up() -> None:
            async for msg in up:
                if isinstance(msg, str):
                    await ws.send_text(msg)
                else:
                    await ws.send_bytes(msg)

        async def _pump_down() -> None:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    return
                if msg.get("bytes") is not None:
                    await up.send(msg["bytes"])
                elif msg.get("text") is not None:
                    await up.send(msg["text"])

        pumps = [asyncio.create_task(_pump_up()), asyncio.create_task(_pump_down())]
        try:
            done, _pending = await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                t.result()
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("error in STT stream passthrough (%s)", alias)
        finally:
            for t in pumps:
                t.cancel()
            await asyncio.gather(*pumps, return_exceptions=True)
            await up.close()
            try:
                await ws.close()
            except Exception:
                pass

    return router
