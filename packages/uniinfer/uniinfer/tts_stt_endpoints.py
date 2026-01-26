
# --- Models for TTS Endpoint ---

class TTSRequest(BaseModel):
    model: str  # Expected format: "provider@modelname"
    input: str
    voice: Optional[str] = None
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0
    instructions: Optional[str] = None

@app.post("/v1/audio/speech")
async def generate_speech(request_input: TTSRequest, request: Request):
    """
    OpenAI-compatible text-to-speech endpoint.
    Generates audio from text using TTS models.
    """
    auth_header = request.headers.get("authorization")
    api_bearer_token = None
    if auth_header and auth_header.startswith("Bearer "):
        api_bearer_token = auth_header[7:]

    provider_model = request_input.model
    input_text = request_input.input

    try:
        if '@' not in provider_model:
            raise HTTPException(status_code=400, detail="Invalid model format. Expected 'provider@modelname'.")
        provider_name, model_name = provider_model.split('@', 1)

        if provider_name != 'tu':
            raise HTTPException(status_code=400, detail="Only 'tu' provider supported for TTS.")

        api_key = None
        if api_bearer_token:
            try:
                api_key = get_provider_api_key(api_bearer_token, provider_name)
            except (ValueError, AuthenticationError):
                raise HTTPException(status_code=401, detail="API key retrieval failed")
        
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required for TU provider")

        # Import TTS classes
        from uniinfer import TTSRequest as UniTTSRequest
        from uniinfer.providers.tu_tts import TuAITTSProvider

        # Create TTS provider
        tts_provider = TuAITTSProvider(api_key=api_key)

        # Create TTS request
        tts_request = UniTTSRequest(
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
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# --- Models for STT Endpoint ---

class STTResponse(BaseModel):
    text: str

class STTVerboseResponse(BaseModel):
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    segments: Optional[List[Dict]] = None

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
    request: Request = None
):
    """
    OpenAI-compatible speech-to-text endpoint.
    Transcribes audio files to text.
    """
    auth_header = request.headers.get("authorization")
    api_bearer_token = None
    if auth_header and auth_header.startswith("Bearer "):
        api_bearer_token = auth_header[7:]

    provider_model = model

    try:
        if '@' not in provider_model:
            raise HTTPException(status_code=400, detail="Invalid model format. Expected 'provider@modelname'.")
        provider_name, model_name = provider_model.split('@', 1)

        if provider_name != 'tu':
            raise HTTPException(status_code=400, detail="Only 'tu' provider supported for STT.")

        api_key = None
        if api_bearer_token:
            try:
                api_key = get_provider_api_key(api_bearer_token, provider_name)
            except (ValueError, AuthenticationError):
                raise HTTPException(status_code=401, detail="API key retrieval failed")
        
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required for TU provider")

        # Read audio file content
        audio_content = await file.read()

        # Import STT classes
        from uniinfer import STTRequest as UniSTTRequest
        from uniinfer.providers.tu_stt import TuAISTTProvider

        # Create STT provider
        stt_provider = TuAISTTProvider(api_key=api_key)

        # Create STT request
        stt_request = UniSTTRequest(
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
            return STTVerboseResponse(
                text=response.text,
                language=response.language,
                duration=response.duration,
                segments=response.segments
            )
        else:
            return STTResponse(text=response.text)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
